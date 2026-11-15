# Tender compliance checker API + web UI. Non-root, pinned base images, OCR (Tesseract eng + ara) included.
FROM python:3.12.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

WORKDIR /app
# dependencies first (cached layer), then the code
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY agentcore ./agentcore
COPY tender ./tender
COPY prices.example.toml ./prices.toml
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /app/traces && chown -R app:app /app/data /app/traces
USER app

ENV PATH="/opt/venv/bin:$PATH" DATA_DIR=/app/data TRACE_DIR=/app/traces \
    SAMPLES_DIR=/app/data/samples RESULTS_DIR=/app/data/results \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata OCR_LANGUAGES=eng+ara
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "tender.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
