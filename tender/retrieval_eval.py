"""Retrieval eval (FLOW step 4): does the bid passage that answers a requirement reach the judge?

For every gold item with a known answer in the bid (bid_text; MET / PARTIAL), search the project's bid with
the requirement as the query, in each mode, and find the rank of the first chunk that contains bid_text.
Reports recall@1/3/6 (6 = what the judge sees, TOP_K) and MRR per mode: dense, sparse (keywords), hybrid.
Cost: one embedding per query per dense/hybrid mode (fractions of a cent). --rerank llm adds LLM calls.
"""

from __future__ import annotations

from agentcore import rerank as rr

from . import config
from .store import Store, get_store
from .text import quote_in

MODES = ("dense", "sparse", "hybrid")
KS = (1, 3, 6)


def evaluate(project_id: str, gold: list[dict], *, k: int = 10, rerank: str = "none",
             store: Store | None = None) -> dict:
    store = store or get_store()
    items = [g for g in gold if g.get("bid_text") and g.get("status") in ("MET", "PARTIAL")]
    modes = list(MODES) + (["hybrid+rerank"] if rerank != "none" else [])
    ranks: dict[str, list[int | None]] = {m: [] for m in modes}
    misses: dict[str, list[str]] = {m: [] for m in modes}
    for g in items:
        query = g.get("requirement") or g["tender_quote"]
        for mode in modes:
            hits = [d for d, _ in store.search(query, project_id, role="bid", k=k * 3 if "rerank" in mode else k,
                                               mode="hybrid" if "hybrid" in mode else mode,
                                               candidates=config.CANDIDATES)]
            if mode == "hybrid+rerank":
                hits = [hits[i] for i in rr.rerank(query, [d.page_content for d in hits], backend=rerank)[:k]]
            rank = next((i for i, d in enumerate(hits, 1) if quote_in(g["bid_text"], d.page_content)), None)
            ranks[mode].append(rank)
            if rank is None or rank > config.TOP_K:
                misses[mode].append(g["key"])
    n = len(items)
    out = {"queries": n, "k": k, "modes": {}}
    for mode in modes:
        rs = ranks[mode]
        out["modes"][mode] = {**{f"recall@{x}": round(sum(1 for r in rs if r and r <= x) / n, 3) if n else None
                                 for x in KS},
                              "mrr": round(sum(1 / r for r in rs if r) / n, 3) if n else None,
                              f"missed_in_top{config.TOP_K}": misses[mode]}
    return out
