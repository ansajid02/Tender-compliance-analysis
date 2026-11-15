"""Vector store: Qdrant with hybrid search (dense embeddings (Gemini by default) + BM25-style sparse vectors, RRF).

Access control lives HERE: every search requires a project id, applied as a Qdrant filter, so chunks of
another project are never even retrieved (not filtered out afterwards). Role (tender | bid) is a filter too.

- QDRANT_URL set   -> Qdrant server (docker compose)
- QDRANT_URL empty -> embedded local Qdrant in data/qdrant (one process at a time; tests use ":memory:")
- writing goes through LangChain's QdrantVectorStore (Documents in, points out); searching uses the Qdrant
  query API directly, because LangChain's hybrid mode prefetches only k candidates per retriever
"""

from __future__ import annotations

import atexit
import threading
import zlib
from collections import Counter
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient, models

from agentcore import embed

from . import config
from .text import tokens

DENSE, SPARSE = "dense", "sparse"
Mode = Literal["hybrid", "dense", "sparse"]
K1, B, AVG_LEN = 1.2, 0.75, 120.0  # BM25 term-frequency saturation; IDF is applied by Qdrant (Modifier.IDF)


class AgentcoreEmbeddings(Embeddings):
    """LangChain Embeddings over agentcore.embed (EMBED_BACKEND: Gemini by default; traced)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed.embed(texts, kind="document", agent="embed.index").tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed.embed([text], kind="query", agent="embed.query")[0].tolist()


class BM25Sparse(SparseEmbeddings):
    """Keyword vectors: hashed normalised tokens (Arabic + English) with BM25 tf weights."""

    @staticmethod
    def _vector(toks: list[str], query: bool) -> SparseVector:
        counts = Counter(zlib.crc32(t.encode("utf-8")) for t in toks)
        if query:
            return SparseVector(indices=list(counts), values=[1.0] * len(counts))
        n = len(toks)
        vals = [tf * (K1 + 1) / (tf + K1 * (1 - B + B * n / AVG_LEN)) for tf in counts.values()]
        return SparseVector(indices=list(counts), values=vals)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(tokens(t), query=False) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(tokens(text), query=True)


class Store:
    def __init__(self, client: QdrantClient | None = None, collection: str = config.QDRANT_COLLECTION) -> None:
        self.client = client or _default_client()
        self.collection = collection
        self.dense = AgentcoreEmbeddings()
        self.sparse = BM25Sparse()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ write

    def _ensure(self, dim: int) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection):
                return
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            if config.QDRANT_URL:  # payload indexes make filtered search fast (no-op in local mode)
                for key in ("project_id", "role", "doc_id"):
                    self.client.create_payload_index(self.collection, f"metadata.{key}",
                                                     models.PayloadSchemaType.KEYWORD)

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        if not self.client.collection_exists(self.collection):
            self._ensure(len(self.dense.embed_query("dimension probe")))
        vs = QdrantVectorStore(client=self.client, collection_name=self.collection, embedding=self.dense,
                               sparse_embedding=self.sparse, retrieval_mode=RetrievalMode.HYBRID,
                               vector_name=DENSE, sparse_vector_name=SPARSE)
        vs.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs], batch_size=96)
        return len(docs)

    def delete(self, project_id: str, doc_id: str | None = None) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(self.collection, points_selector=models.FilterSelector(
            filter=_filter(project_id, None, [doc_id] if doc_id else None)))

    # ------------------------------------------------------------ read

    def search(self, query: str, project_id: str, role: str | None = None, *, k: int = config.TOP_K,
               mode: Mode = "hybrid", candidates: int = config.CANDIDATES,
               doc_ids: list[str] | None = None) -> list[tuple[Document, float]]:
        flt = _filter(project_id, role, doc_ids)
        if not self.client.collection_exists(self.collection):
            return []
        dense_q = self.dense.embed_query(query) if mode in ("hybrid", "dense") else None
        sv = self.sparse.embed_query(query)
        sparse_q = models.SparseVector(indices=sv.indices, values=sv.values)
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dense_q, using=DENSE, query_filter=flt,
                                           limit=k, with_payload=True)
        elif mode == "sparse":
            if not sv.indices:
                return []
            res = self.client.query_points(self.collection, query=sparse_q, using=SPARSE, query_filter=flt,
                                           limit=k, with_payload=True)
        else:
            prefetch = [models.Prefetch(query=dense_q, using=DENSE, filter=flt, limit=candidates)]
            if sv.indices:
                prefetch.append(models.Prefetch(query=sparse_q, using=SPARSE, filter=flt, limit=candidates))
            res = self.client.query_points(self.collection, prefetch=prefetch,
                                           query=models.FusionQuery(fusion=models.Fusion.RRF),
                                           query_filter=flt, limit=k, with_payload=True)
        return [(_to_doc(p.payload), float(p.score)) for p in res.points]

    def count(self, project_id: str, role: str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, count_filter=_filter(project_id, role, None), exact=True).count


def _filter(project_id: str, role: str | None, doc_ids: list[str] | None) -> models.Filter:
    if not project_id:
        raise ValueError("project_id is required for every vector store operation")
    must: list = [models.FieldCondition(key="metadata.project_id", match=models.MatchValue(value=project_id))]
    if role:
        must.append(models.FieldCondition(key="metadata.role", match=models.MatchValue(value=role)))
    if doc_ids:
        must.append(models.FieldCondition(key="metadata.doc_id", match=models.MatchAny(any=doc_ids)))
    return models.Filter(must=must)


def _to_doc(payload: dict | None) -> Document:
    payload = payload or {}
    return Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}))


_init_lock = threading.Lock()  # parallel graph tasks may ask for the store at the same moment


@lru_cache(maxsize=1)
def _default_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.QDRANT_PATH))
    atexit.register(client.close)  # release the local-mode lock cleanly (no shutdown traceback)
    return client


_store: Store | None = None


def get_store() -> Store:
    global _store
    with _init_lock:  # embedded Qdrant allows ONE client per folder: never create two
        if _store is None:
            _store = Store()
        return _store


def set_store(store: Store | None) -> None:
    """Tests: swap in a Store over QdrantClient(':memory:')."""
    global _store
    _store = store


def rrf_merge(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion of id lists (used to merge searches in the reviewer)."""
    scores: dict[str, float] = {}
    for ids in rankings:
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: -scores[c])


__all__ = ["Store", "get_store", "set_store", "rrf_merge", "BM25Sparse", "AgentcoreEmbeddings"]
