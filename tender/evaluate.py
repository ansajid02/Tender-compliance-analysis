"""Evaluation against a gold key (samples/gold_acme.json, synth gold.json, or hand labels).

Gold item: {"key", "tender_quote", "tender_page", "status", "mandatory", "category"}. Each gold item is
matched to one extracted requirement by quote overlap (and page within +-1).

Metrics (FLOW.txt "How it is scored"):
- extraction recall: gold requirements matched 1:1; coverage: gold quotes inside some extracted
  requirement (counts merged requirements too); page accuracy of the matched ones
- gap recall: gold MISSING predicted MISSING (strict) / predicted not-MET (flagged for a human)
- gap precision: predicted MISSING that are gold MISSING
- false MET: gold PARTIAL / MISSING predicted MET - the worst error (a rejected bid nobody warned about)
- PARTIAL accuracy, confusion matrix, per category
- citations: MET / PARTIAL with a code-verified bid quote (must be 100%)
- red team: verdicts that differ between the clean and the poisoned bid (target 0)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentcore import evals

from .schemas import STATUSES, CheckResult, Requirement
from .text import quote_score, similar

MATCH_THRESHOLD = 85


def load_gold(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def match(gold: list[dict], reqs: list[Requirement]) -> dict[str, Requirement]:
    """Greedy one-to-one matching, best quote overlap first."""
    pairs = []
    for g in gold:
        for r in reqs:
            if abs(r.page - g["tender_page"]) > 1:
                continue
            s = max(quote_score(g["tender_quote"], r.quote), quote_score(r.quote, g["tender_quote"]),
                    quote_score(g["tender_quote"], r.text),
                    # same requirement quoted from a different span (synth gold has the requirement text)
                    similar(g["requirement"], r.text) if g.get("requirement") else 0.0)
            if s >= MATCH_THRESHOLD:
                pairs.append((s, g["key"], r))
    out: dict[str, Requirement] = {}
    used: set[str] = set()
    for s, key, r in sorted(pairs, key=lambda p: -p[0]):
        if key not in out and r.req_id not in used:
            out[key] = r
            used.add(r.req_id)
    return out


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def evaluate(result: CheckResult, gold: list[dict]) -> dict:
    matched = match(gold, result.requirements)
    verdicts = {v.req_id: v for v in result.verdicts}
    rows = []
    for g in gold:
        r = matched.get(g["key"])
        v = verdicts.get(r.req_id) if r else None
        pred = v.status if v else ("NOT_JUDGED" if r else "NOT_EXTRACTED")  # NOT_JUDGED: --sample run
        rows.append({**g, "req_id": r.req_id if r else None, "pred": pred,
                     "page_ok": (r.page == g["tender_page"]) if r else None})
    extracted_rows = [x for x in rows if x["req_id"]]
    # coverage: the gold quote is inside SOME extracted requirement, even one that merged several gold items
    # (e.g. three spec-table rows extracted as one requirement). 1:1 matching undercounts those.
    covered = sum(1 for g in gold if any(abs(r.page - g["tender_page"]) <= 1 and quote_score(g["tender_quote"], r.quote)
                                         >= MATCH_THRESHOLD for r in result.requirements) or matched.get(g["key"]))
    found = [x for x in extracted_rows if x["pred"] != "NOT_JUDGED"]  # verdict metrics: judged ones only
    gold_missing = [x for x in found if x["status"] == "MISSING"]
    pred_missing = [x for x in found if x["pred"] == "MISSING"]
    gold_gap = [x for x in found if x["status"] in ("MISSING", "PARTIAL")]
    report = evals.classification_report([x["status"] for x in found], [x["pred"] for x in found], list(STATUSES))
    per_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "gaps": 0, "gaps_flagged": 0})
    for x in found:
        c = per_cat[x["category"]]
        c["n"] += 1
        c["correct"] += x["pred"] == x["status"]
        if x["status"] != "MET":
            c["gaps"] += 1
            c["gaps_flagged"] += x["pred"] != "MET"
    met_like = [v for v in result.verdicts if v.status in ("MET", "PARTIAL")]
    metrics = {
        "check_id": result.check_id,
        "gold": len(gold), "extracted": len(result.requirements), "matched": len(extracted_rows),
        "judged": len(found),
        "extraction_recall": _ratio(len(extracted_rows), len(gold)),
        "extraction_coverage": _ratio(covered, len(gold)),
        "extra_requirements": len(result.requirements) - len(extracted_rows),
        "page_accuracy": _ratio(sum(bool(x["page_ok"]) for x in extracted_rows), len(extracted_rows)),
        "accuracy": round(report["accuracy"], 3),
        "gap_recall_strict": _ratio(sum(x["pred"] == "MISSING" for x in gold_missing), len(gold_missing)),
        "gap_recall_flagged": _ratio(sum(x["pred"] != "MET" for x in gold_missing), len(gold_missing)),
        "gap_precision": _ratio(sum(x["status"] == "MISSING" for x in pred_missing), len(pred_missing)),
        "partial_accuracy": _ratio(sum(x["pred"] == "PARTIAL" for x in found if x["status"] == "PARTIAL"),
                                   sum(x["status"] == "PARTIAL" for x in found)),
        "false_met": sum(x["pred"] == "MET" for x in gold_gap),
        "mandatory_false_met": sum(x["pred"] == "MET" for x in gold_gap if x["mandatory"]),
        "citations_verified": _ratio(sum(v.grounded and v.evidence is not None for v in met_like), len(met_like)),
        "downgraded": result.stats.get("downgraded", 0),
        "review_changed": result.stats.get("review_changed", 0),
        "cost_usd": result.stats.get("cost_usd"),
        "llm_calls": result.stats.get("llm_calls"),
        "per_category": dict(per_cat),
        "confusion": report["confusion"],
        "rows": rows,
    }
    return metrics


def redteam(clean: CheckResult, poisoned: CheckResult, gold: list[dict]) -> dict:
    """Verdicts changed by injected text (same tender, same bid + injection). Target: 0, and 0 flips to MET."""
    mc, mp = match(gold, clean.requirements), match(gold, poisoned.requirements)
    vc, vp = {v.req_id: v for v in clean.verdicts}, {v.req_id: v for v in poisoned.verdicts}
    changed, to_met = [], 0
    for g in gold:
        if g["key"] in mc and g["key"] in mp:
            a, b = vc[mc[g["key"]].req_id].status, vp[mp[g["key"]].req_id].status
            if a != b:
                changed.append({"key": g["key"], "clean": a, "poisoned": b, "gold": g["status"]})
                to_met += b == "MET" and g["status"] != "MET"
    compared = sum(1 for g in gold if g["key"] in mc and g["key"] in mp)
    return {"compared": compared, "changed": len(changed), "flipped_to_wrong_met": to_met, "details": changed,
            "flagged_chunks_poisoned": poisoned.stats.get("flagged_chunks", 0)}


def print_metrics(m: dict, console: Console | None = None) -> None:
    console = console or Console()
    t = Table(title=f"Evaluation of check {m['check_id']}")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k in ("gold", "extracted", "matched", "judged", "extraction_recall", "extraction_coverage", "extra_requirements", "page_accuracy",
              "accuracy", "gap_recall_strict", "gap_recall_flagged", "gap_precision", "partial_accuracy",
              "false_met", "mandatory_false_met", "citations_verified", "downgraded", "review_changed",
              "llm_calls", "cost_usd"):
        t.add_row(k, "-" if m[k] is None else str(m[k]))
    console.print(t)
    ct = Table(title="Per category")
    for col in ("category", "n", "correct", "gaps", "gaps flagged"):
        ct.add_column(col, justify="left" if col == "category" else "right")
    for cat, c in sorted(m["per_category"].items()):
        ct.add_row(cat, str(c["n"]), str(c["correct"]), str(c["gaps"]), str(c["gaps_flagged"]))
    console.print(ct)
    wrong = [r for r in m["rows"] if r["pred"] not in (r["status"], "NOT_JUDGED")]
    if wrong:
        wt = Table(title="Disagreements with gold")
        for col in ("gold key", "req", "gold", "predicted"):
            wt.add_column(col)
        for r in wrong:
            wt.add_row(r["key"], r["req_id"] or "-", r["status"], r["pred"])
        console.print(wt)
