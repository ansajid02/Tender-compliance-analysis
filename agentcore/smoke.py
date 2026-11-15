"""`make smoke`: one structured LLM call + one embedding call on the configured backends, traced."""

from __future__ import annotations

import sys

from pydantic import BaseModel
from rich.console import Console

from . import embed, llm, trace
from .config import settings


class Capital(BaseModel):
    country: str
    capital: str
    capital_in_arabic: str


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "flash"
    s = settings()
    console = Console()
    where = "Vertex AI" if s.gemini_vertex else "Gemini API"
    console.print(f"LLM backend: {s.llm_backend}" + (f" ({where})" if s.llm_backend == "gemini" else "")
                  + f" · embeddings: {s.embed_backend} {s.embed_model} ({s.embed_dim} dims)")
    with trace.run("smoke") as run_id, trace.collect() as events:
        answer = llm.generate("What is the capital of Saudi Arabia?", schema=Capital, tier=tier, agent="smoke")
        console.print(answer)
        vecs = embed.embed(["يجب على المورد تقديم شهادة الزكاة", "The bidder shall submit a Zakat certificate",
                            "Backups are kept for 90 days"], kind="document", agent="smoke")
        console.print(f"embeddings {vecs.shape}: cosine(AR, EN same meaning) = {float(vecs[0] @ vecs[1]):.3f}, "
                      f"cosine(EN, unrelated) = {float(vecs[1] @ vecs[2]):.3f}")
    for e in events:
        console.print(f"{e['kind']}: model={e['model']} in={e.get('input_tokens')} out={e.get('output_tokens')} "
                      f"latency={e['latency_ms']}ms cost={e.get('cost_usd')}")
    console.print(f"trace: {trace.trace_path(run_id)}")


if __name__ == "__main__":
    from .embed import EmbedError
    from .llm import LLMError

    try:
        main()
    except (LLMError, EmbedError, RuntimeError) as e:
        Console().print(f"[red]error[/] {e}")
        raise SystemExit(1) from None
