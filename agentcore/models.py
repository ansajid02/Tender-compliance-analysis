"""`make models [FILTER=flash]`: models the configured backend can use.

Gemini: live list from the API (needs GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT), with token limits and
which methods each model supports (generateContent = LLM, embedContent = embeddings).
Bedrock: a static list (the Mantle APIs have no list endpoint); access is per AWS account and region.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from .config import settings
from .pricing import _prices

console = Console()

BEDROCK = [
    ("anthropic.claude-haiku-5-5", "Claude Haiku 5.5"),
    ("anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("anthropic.claude-opus-5-5", "Claude Opus 5.5"),
    ("qwen.qwen3-235b-a22b-2507", "Qwen3 235B A22B"),
    ("cohere.embed-multilingual-v3", "Cohere Embed Multilingual v3 (embeddings)"),
    ("cohere.rerank-v3-5:0", "Cohere Rerank 3.5 (rerank, some regions only)"),
]


def _price(model: str) -> str:
    p = _prices().get(model)
    return f"{p['input']:.2f} / {p['output']:.2f}" if p else "? (add to prices.toml)"


def list_gemini(filter_text: str) -> None:
    from .llm import _gemini_client

    s = settings()
    t = Table(title=f"Gemini models ({'Vertex AI ' + s.gcp_project if s.gemini_vertex else 'Gemini API'})"
                    f" matching '{filter_text}'")
    t.add_column("model id", no_wrap=True)
    for col in ("display name", "in / out tokens", "methods", "$ / 1M in / out"):
        t.add_column(col, overflow="fold")
    for m in _gemini_client().models.list(config={"page_size": 100}):
        name = (m.name or "").split("/")[-1]
        if filter_text.lower() not in name.lower():
            continue
        actions = ", ".join(a.replace("Content", "") for a in (m.supported_actions or [])
                            if a in ("generateContent", "embedContent"))
        t.add_row(name, m.display_name or "", f"{m.input_token_limit or '?'} / {m.output_token_limit or '?'}",
                  actions, _price(name))
    console.print(t)


def list_bedrock() -> None:
    t = Table(title=f"AWS Bedrock ({settings().bedrock_region}) - enable access in the Bedrock console")
    for col in ("model id", "name", "$ / 1M in / out"):
        t.add_column(col)
    for mid, name in BEDROCK:
        t.add_row(mid, name, _price(mid))
    console.print(t)


def main() -> None:
    s = settings()
    filter_text = sys.argv[1] if len(sys.argv) > 1 else ""
    if s.llm_backend == "bedrock":
        list_bedrock()
    else:
        list_gemini(filter_text or "gemini")
    console.print(f"Current .env: LLM_BACKEND={s.llm_backend}  flash={s.model_for(s.llm_backend, 'flash')}  "
                  f"pro={s.model_for(s.llm_backend, 'pro')}")
    console.print(f"              EMBED_BACKEND={s.embed_backend} EMBED_MODEL={s.embed_model} EMBED_DIM={s.embed_dim}  "
                  f"RERANK_BACKEND={s.rerank_backend}")


if __name__ == "__main__":
    from .embed import EmbedError
    from .llm import LLMError

    try:
        main()
    except (LLMError, EmbedError, RuntimeError) as e:
        Console().print(f"[red]error[/] {e}")
        raise SystemExit(1) from None
