"""Text embeddings through one traced function (like llm.generate).

    vecs = embed.embed(["text", ...], kind="document")   # -> float32 array (n, dim), unit-normalised

EMBED_BACKEND (in .env):
- "gemini" (default): EMBED_MODEL (gemini-embedding-001: 100+ languages incl. Arabic) with retrieval task
      types (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY) and EMBED_DIM output dimensions (768 / 1536 / 3072).
      Same credentials as the LLM: GEMINI_API_KEY, or Vertex AI via GOOGLE_CLOUD_PROJECT.
- "bedrock": EMBED_MODEL on Bedrock runtime (cohere.embed-multilingual-v3, cohere.embed-v4:0,
      amazon.titan-embed-text-v2:0)
- "hash": deterministic bag-of-words hashing, no network. For tests and offline dev only:
      it matches words, not meaning.
Changing the model or dimension needs a fresh index (delete data/qdrant, re-ingest).
"""

from __future__ import annotations

import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np

from . import trace
from .config import settings
from .llm import thread_safe_cache
from .pricing import cost_usd

Kind = Literal["document", "query"]
COHERE_BATCH = 96
COHERE_MAX_CHARS = 2048  # Cohere on Bedrock: longer texts are truncated (truncate=END)
TITAN_WORKERS = 8


class EmbedError(RuntimeError):
    pass


@thread_safe_cache
def bedrock_runtime(region: str | None = None):
    """boto3 bedrock-runtime client. boto3 reads AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) from the
    environment (.env is loaded into it by agentcore.config); without it, the normal AWS credential chain."""
    import boto3
    from botocore.config import Config

    return boto3.client("bedrock-runtime", region_name=region or settings().bedrock_region,
                        config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=120))


def _invoke(model: str, body: dict, region: str | None = None) -> tuple[dict, int]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        resp = bedrock_runtime(region).invoke_model(
            modelId=model, body=json.dumps(body), contentType="application/json", accept="application/json")
    except ClientError as e:
        raise EmbedError(f"Bedrock InvokeModel '{model}' failed: {e}") from e
    except BotoCoreError as e:
        raise EmbedError(f"Bedrock connection error for '{model}': {e}") from e
    headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0) or 0)
    return json.loads(resp["body"].read()), tokens


def _bedrock(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int]:
    if model.startswith("cohere.embed"):
        body = {"texts": [t[:COHERE_MAX_CHARS] for t in texts],
                "input_type": "search_query" if kind == "query" else "search_document", "truncate": "END"}
        if "v4" in model:
            body["embedding_types"] = ["float"]
        data, tokens = _invoke(model, body)
        vecs = data["embeddings"]
        if isinstance(vecs, dict):  # v4 returns {"float": [...]}
            vecs = vecs["float"]
        return np.array(vecs, dtype=np.float32), tokens
    if model.startswith("amazon.titan-embed"):
        dim = settings().embed_dim

        def one(t: str) -> tuple[list[float], int]:
            data, _ = _invoke(model, {"inputText": t, "dimensions": dim, "normalize": True})
            return data["embedding"], int(data.get("inputTextTokenCount", 0))

        with ThreadPoolExecutor(TITAN_WORKERS) as pool:
            rows = list(pool.map(one, texts))
        return np.array([r[0] for r in rows], dtype=np.float32), sum(r[1] for r in rows)
    raise EmbedError(f"unsupported Bedrock embedding model '{model}' (use cohere.embed-* or amazon.titan-embed-*)")


GEMINI_BATCH = 100  # texts per request on the Gemini API
GEMINI_WORKERS = 8  # Vertex AI: one text per request, in parallel


def _gemini(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int]:
    from google.genai import errors, types

    from .llm import LLMError, _gemini_client

    s = settings()
    cfg = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
                                   output_dimensionality=s.embed_dim)
    try:
        client = _gemini_client()
    except LLMError as e:
        raise EmbedError(str(e)) from e

    def call(batch: list[str]) -> tuple[list[list[float]], int]:
        try:
            # one Content per text: gemini-embedding-2 (multimodal) would merge a plain list into ONE vector
            resp = client.models.embed_content(
                model=model, contents=[types.Content(parts=[types.Part(text=t)]) for t in batch], config=cfg)
        except errors.APIError as e:
            raise EmbedError(f"Gemini embeddings '{model}' failed ({e.code}): {str(e)[:300]}") from e
        if len(resp.embeddings) != len(batch):
            raise EmbedError(f"'{model}' returned {len(resp.embeddings)} embedding(s) for {len(batch)} texts")
        tokens = sum(int(getattr(getattr(e, "statistics", None), "token_count", 0) or 0) for e in resp.embeddings)
        return [e.values for e in resp.embeddings], tokens

    if s.gemini_vertex:
        with ThreadPoolExecutor(GEMINI_WORKERS) as pool:
            rows = list(pool.map(lambda t: call([t]), texts))
        return np.array([r[0][0] for r in rows], dtype=np.float32), sum(r[1] for r in rows)
    vecs, tokens = [], 0
    for i in range(0, len(texts), GEMINI_BATCH):
        v, t = call(texts[i:i + GEMINI_BATCH])
        vecs.extend(v)
        tokens += t
    return np.array(vecs, dtype=np.float32), tokens


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _hash(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int]:
    dim = settings().embed_dim
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _TOKEN.findall(t.lower()):
            h = zlib.crc32(tok.encode("utf-8"))
            out[i, h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    return out, 0


BACKENDS: dict[str, Callable[[str, list[str], Kind], tuple[np.ndarray, int]]] = {
    "gemini": _gemini, "bedrock": _bedrock, "hash": _hash}


def current_model() -> tuple[str, str]:
    s = settings()
    return s.embed_backend, (f"hash-{s.embed_dim}" if s.embed_backend == "hash" else s.embed_model)


def embed(texts: list[str], kind: Kind = "document", *, agent: str = "embed") -> np.ndarray:
    backend, model = current_model()
    if backend not in BACKENDS:
        raise EmbedError(f"unknown EMBED_BACKEND '{backend}' (choose from {', '.join(BACKENDS)})")
    batch_size = COHERE_BATCH if model.startswith("cohere.") else 512  # backends batch further themselves
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with trace.span("embed", agent, backend=backend, model=model, n=len(batch), embed_kind=kind) as sp:
            vecs, tokens = BACKENDS[backend](model, batch, kind)
            sp.update(input_tokens=tokens, output_tokens=0,
                      cost_usd=0.0 if backend == "hash" else cost_usd(backend, model, tokens, 0))
        out.append(vecs)
    if not out:
        return np.zeros((0, settings().embed_dim), dtype=np.float32)
    arr = np.vstack(out)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)
