"""Rerank retrieved passages for a query, traced.

    order = rerank.rerank("query", ["passage", ...])   # -> indices, most relevant first

RERANK_BACKEND (in .env):
- "none": keep the retrieval order (default until the retrieval eval shows reranking helps)
- "llm":  ask the flash LLM of the current backend (Gemini by default) to order the candidates
- "bedrock": Cohere Rerank 3.5 on AWS Bedrock (RERANK_MODEL, RERANK_REGION; only if LLM_BACKEND/keys
          for Bedrock are set up)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm, trace
from .config import settings
from .embed import EmbedError, _invoke

MAX_DOC_CHARS = 4000


class RerankError(RuntimeError):
    pass


class _LLMRank(BaseModel):
    ranked: list[int] = Field(description="candidate numbers, most relevant first")


def _complete(order: list[int], n: int) -> list[int]:
    """Dedupe, drop out-of-range numbers, append anything the reranker left out (in original order)."""
    seen = [i for i in dict.fromkeys(order) if 0 <= i < n]
    return seen + [i for i in range(n) if i not in set(seen)]


def _bedrock(query: str, docs: list[str]) -> list[int]:
    s = settings()
    if not s.rerank_model.startswith("cohere.rerank"):
        raise RerankError(f"unsupported RERANK_MODEL '{s.rerank_model}' (use cohere.rerank-v3-5:0)")
    body = {"query": query, "documents": [d[:MAX_DOC_CHARS] for d in docs], "top_n": len(docs), "api_version": 2}
    try:
        data, _ = _invoke(s.rerank_model, body, region=s.rerank_region)
    except EmbedError as e:
        raise RerankError(f"{e}. Rerank models exist only in some regions: set RERANK_REGION "
                          "(e.g. us-west-2), or RERANK_BACKEND=llm / none.") from e
    return [r["index"] for r in sorted(data["results"], key=lambda r: -r["relevance_score"])]


def _llm(query: str, docs: list[str]) -> list[int]:
    listing = "\n".join(f"{i}. {d[:500]}" for i, d in enumerate(docs))
    out = llm.generate(f"Query: {query}\n\nCandidates:\n{listing}\n\nRank the candidates by how useful they are "
                       "for answering the query (most useful first). Return candidate numbers only.",
                       schema=_LLMRank, agent="rerank", max_output_tokens=600)
    return out.ranked


def rerank(query: str, docs: list[str], backend: str | None = None) -> list[int]:
    backend = backend or settings().rerank_backend
    if backend == "none" or len(docs) < 2:
        return list(range(len(docs)))
    fn = {"bedrock": _bedrock, "llm": _llm}.get(backend)
    if fn is None:
        raise RerankError(f"unknown RERANK_BACKEND '{backend}' (none | llm | bedrock)")
    with trace.span("rerank", backend, n=len(docs)):
        return _complete(fn(query, docs), len(docs))
