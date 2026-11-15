"""Settings for this project. Values come from this folder's .env file.

Provider: Google Gemini by default (LLM, embeddings, LLM rerank), through the google-genai SDK:
- GEMINI_API_KEY set           -> Gemini API (Google AI Studio key)
- else GOOGLE_CLOUD_PROJECT set -> Vertex AI (gcloud auth application-default login)
AWS Bedrock stays available as a second backend (LLM_BACKEND=bedrock / EMBED_BACKEND=bedrock).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # this project folder (self-contained)

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EMBED_MODEL = {"gemini": "gemini-embedding-001", "bedrock": "cohere.embed-multilingual-v3", "hash": "hash"}


def _env(name: str, default: str = "") -> str:
    """Unset OR empty (`KEY=` in .env) both mean: use the default."""
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    llm_backend: str  # "gemini" | "bedrock"
    # Gemini (google-genai)
    gemini_api_key: str
    gcp_project: str
    gcp_location: str
    gemini_model_flash: str
    gemini_model_pro: str
    gemini_thinking: str  # minimal | low | medium | high | "" (model default)
    # AWS Bedrock
    bedrock_api_key: str
    bedrock_region: str
    bedrock_model_flash: str
    bedrock_model_pro: str
    bedrock_effort: str
    # embeddings / rerank
    embed_backend: str  # "gemini" | "bedrock" | "hash" (offline, tests only)
    embed_model: str
    embed_dim: int
    rerank_backend: str  # "none" | "llm" | "bedrock"
    rerank_model: str
    rerank_region: str
    trace_dir: Path

    @property
    def gemini_vertex(self) -> bool:
        """True = Vertex AI (Google Cloud project); False = Gemini API key."""
        return not self.gemini_api_key and bool(self.gcp_project)

    def model_for(self, backend: str, tier: str) -> str:
        model = getattr(self, f"{backend}_model_{tier}", "")
        if not model:
            raise RuntimeError(f"No model set for backend '{backend}', tier '{tier}'. Set "
                               f"{backend.upper()}_MODEL_{tier.upper()} in .env (`make models` lists options).")
        return model


@lru_cache(maxsize=1)
def settings() -> Settings:
    embed_backend = _env("EMBED_BACKEND", "gemini")
    return Settings(
        llm_backend=_env("LLM_BACKEND", "gemini"),
        gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        gcp_project=_env("GOOGLE_CLOUD_PROJECT"),
        gcp_location=_env("GOOGLE_CLOUD_LOCATION", "global"),
        gemini_model_flash=_env("GEMINI_MODEL_FLASH", "gemini-3.6-flash"),
        gemini_model_pro=_env("GEMINI_MODEL_PRO", "gemini-3.8-flash"),
        gemini_thinking=_env("GEMINI_THINKING", "low").lower(),
        bedrock_api_key=_env("AWS_BEARER_TOKEN_BEDROCK"),
        bedrock_region=_env("AWS_REGION", "us-east-1"),
        bedrock_model_flash=_env("BEDROCK_MODEL_FLASH", "anthropic.claude-haiku-5-5"),
        bedrock_model_pro=_env("BEDROCK_MODEL_PRO", "anthropic.claude-sonnet-5-5"),
        bedrock_effort=_env("BEDROCK_EFFORT", "low"),
        embed_backend=embed_backend,
        embed_model=_env("EMBED_MODEL", DEFAULT_EMBED_MODEL.get(embed_backend, "")),
        embed_dim=int(_env("EMBED_DIM", "1536")),
        rerank_backend=_env("RERANK_BACKEND", "none"),
        rerank_model=_env("RERANK_MODEL", "cohere.rerank-v3-5:0"),
        rerank_region=_env("RERANK_REGION") or _env("AWS_REGION", "us-east-1"),
        trace_dir=Path(_env("TRACE_DIR") or PROJECT_ROOT / "traces"),
    )
