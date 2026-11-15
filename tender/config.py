"""Tender-checker knobs. Everything can be overridden in .env."""

from __future__ import annotations

import os
from pathlib import Path

from agentcore.config import PROJECT_ROOT


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, "").strip() or default)


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, "").strip() or default).expanduser()


DATA_DIR = _path("DATA_DIR", PROJECT_ROOT / "data")
PROJECTS_DIR = DATA_DIR / "projects"
QDRANT_PATH = _path("QDRANT_PATH", DATA_DIR / "qdrant")  # local (embedded) Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()     # set -> Qdrant server (docker compose)
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "").strip() or "chunks"
SAMPLES_DIR = _path("SAMPLES_DIR", PROJECT_ROOT / "samples")
RESULTS_DIR = _path("RESULTS_DIR", PROJECT_ROOT / "results")

# upload limits (validate.py)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
MAX_PAGES = _int("MAX_PAGES", 600)
MAX_ZIP_UNCOMPRESSED_MB = _int("MAX_ZIP_UNCOMPRESSED_MB", 300)
MAX_ZIP_RATIO = _int("MAX_ZIP_RATIO", 150)

# chunking
CHUNK_CHARS = _int("CHUNK_CHARS", 1400)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

# check workflow
EXTRACT_WINDOW_CHARS = _int("EXTRACT_WINDOW_CHARS", 7000)
TOP_K = _int("TOP_K", 6)              # bid chunks shown to the judge
CANDIDATES = _int("CANDIDATES", 30)   # per retriever (dense, sparse) before fusion
REVIEW_TOP_K = _int("REVIEW_TOP_K", 14)
CONCURRENCY = _int("CONCURRENCY", 8)  # parallel LLM calls in fan-outs
EXTRACT_TIER = os.environ.get("EXTRACT_TIER", "").strip() or "flash"
JUDGE_TIER = os.environ.get("JUDGE_TIER", "").strip() or "pro"
REVIEW_TIER = os.environ.get("REVIEW_TIER", "").strip() or "pro"
