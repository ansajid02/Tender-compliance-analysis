"""The compliance check as a LangGraph workflow. Every step is a node; fan-outs use Send.

  plan --Send per window--> extract --> merge (quote check + dedupe)
       --Send per requirement--> judge (retrieve + judge + grounding) --> gather
       --Send per mandatory MISSING--> review --> report

max_concurrency (CONCURRENCY in .env) bounds parallel LLM calls. A failed LLM call never kills the check:
that window is skipped (counted) or that requirement becomes UNCLEAR with the error as its reason.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentcore import llm, trace
from agentcore.llm import LLMError

from . import config, projects
from .extract import Window, build_windows, extract_window, merge_and_verify
from .judge import check_requirement, review
from .schemas import CheckResult, Requirement, Verdict


class CheckState(TypedDict, total=False):
    project_id: str
    sample: int  # > 0: judge only this many requirements, evenly spread (cheap trial runs)
    to_judge: list[Requirement]
    windows: list[Window]
    raw: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    requirements: list[Requirement]
    stats: dict
    verdicts: Annotated[list[Verdict], operator.add]
    reviewed: Annotated[list[Verdict], operator.add]
    result: CheckResult


def plan(state: CheckState) -> dict:
    windows = build_windows(state["project_id"])
    if not windows:
        raise ValueError("this project has no tender document: upload one with role=tender first")
    return {"windows": windows}


def to_extract(state: CheckState) -> list[Send]:
    return [Send("extract", {"project_id": state["project_id"], "window": w}) for w in state["windows"]]


def extract(task: dict) -> dict:
    w: Window = task["window"]
    with trace.tags(window=w.window_id, pages=f"{w.pages[0]}-{w.pages[-1]}" if w.pages else ""):
        try:
            return {"raw": extract_window(w)}
        except LLMError as e:
            return {"errors": [f"extraction of pages {w.pages[:1]}..{w.pages[-1:]} failed: {e}"]}


def merge(state: CheckState) -> dict:
    errors = state.get("errors", [])
    if errors and len(errors) == len(state["windows"]):  # nothing worked: fail loudly, not "0 requirements"
        raise LLMError(f"requirement extraction failed for every part of the tender. First error: {errors[0]}")
    reqs, stats = merge_and_verify(state["project_id"], state.get("raw", []))
    stats["windows"] = len(state["windows"])
    return {"requirements": reqs, "stats": stats, "to_judge": sample_evenly(reqs, state.get("sample", 0))}


def sample_evenly(reqs: list[Requirement], n: int) -> list[Requirement]:
    """n evenly spaced requirements (reading order). Never looks at any gold key: an unbiased trial sample."""
    if n <= 0 or n >= len(reqs):
        return reqs
    if n == 1:
        return reqs[:1]
    return [reqs[round(i * (len(reqs) - 1) / (n - 1))] for i in range(n)]


def to_judge(state: CheckState) -> list[Send] | str:
    if not state["to_judge"]:
        return "report"
    return [Send("judge", {"project_id": state["project_id"], "req": r}) for r in state["to_judge"]]


def judge_node(task: dict) -> dict:
    req: Requirement = task["req"]
    with trace.tags(req_id=req.req_id):
        try:
            return {"verdicts": [check_requirement(req, task["project_id"])]}
        except LLMError as e:
            return {"verdicts": [Verdict(req_id=req.req_id, status="UNCLEAR", reason=f"judge error: {e}"[:400])],
                    "errors": [f"{req.req_id}: judge failed"]}


def to_review(state: CheckState) -> list[Send] | str:
    reqs = {r.req_id: r for r in state["requirements"]}
    todo = [v for v in state["verdicts"] if v.status == "MISSING" and reqs[v.req_id].mandatory]
    if not todo:
        return "report"
    return [Send("review", {"project_id": state["project_id"], "req": reqs[v.req_id], "verdict": v}) for v in todo]


def review_node(task: dict) -> dict:
    req: Requirement = task["req"]
    with trace.tags(req_id=req.req_id):
        try:
            return {"reviewed": [review(req, task["verdict"], task["project_id"])]}
        except LLMError as e:
            return {"errors": [f"{req.req_id}: review failed ({e})"[:300]]}


def report(state: CheckState) -> dict:
    failed = [v for v in state.get("verdicts", []) if v.reason.startswith("judge error")]
    if failed and len(failed) * 2 > len(state.get("verdicts", [])):  # a matrix of errors is not a result
        raise LLMError(f"{len(failed)} of {len(state['verdicts'])} judge calls failed. First error: {failed[0].reason}")
    reviewed = {v.req_id: v for v in state.get("reviewed", [])}
    verdicts = sorted((reviewed.get(v.req_id, v) for v in state.get("verdicts", [])), key=lambda v: v.req_id)
    stats = dict(state.get("stats", {}))
    stats.update({
        "judged": len(verdicts),
        "reviewed": len(reviewed),
        "review_changed": sum(v.review_changed for v in reviewed.values()),
        "downgraded": sum(v.downgraded_from is not None for v in verdicts),
        "flagged_chunks": len({c for v in verdicts for c in v.flagged_chunks}),
        "errors": len(state.get("errors", [])),
        "sampled": len(state.get("to_judge", [])) if state.get("sample") else None,
    })
    project = projects.load(state["project_id"])
    warnings = list(state.get("errors", []))
    if stats["sampled"] is not None and stats["sampled"] < len(state.get("requirements", [])):
        warnings.insert(0, f"SAMPLE RUN: only {stats['sampled']} of {len(state['requirements'])} requirements were "
                           "judged (evenly spread). Run without --sample for the full matrix.")
    for d in project.documents:
        warnings += [f"{d.filename}: {w}" for w in d.warnings]
    if stats["flagged_chunks"]:
        warnings.append(f"{stats['flagged_chunks']} bid excerpt(s) contain instruction-like text aimed at the "
                        "evaluator (possible prompt injection); they were treated as data")
    if stats.get("quote_unverified"):
        warnings.append(f"{stats['quote_unverified']} extracted requirement(s) dropped: their quote was not "
                        "found in the tender")
    check_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    result = CheckResult(check_id=check_id, project_id=state["project_id"], requirements=state.get("requirements", []),
                         verdicts=verdicts, stats=stats, warnings=warnings)
    return {"result": result}


def build_graph():
    g = StateGraph(CheckState)
    g.add_node("plan", plan)
    g.add_node("extract", extract)
    g.add_node("merge", merge)
    g.add_node("judge", judge_node)
    g.add_node("gather", lambda state: {})  # barrier: runs once, after every judge task
    g.add_node("review", review_node)
    g.add_node("report", report)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", to_extract, ["extract"])
    g.add_edge("extract", "merge")
    g.add_conditional_edges("merge", to_judge, ["judge", "report"])
    g.add_edge("judge", "gather")
    g.add_conditional_edges("gather", to_review, ["review", "report"])
    g.add_edge("review", "report")
    g.add_edge("report", END)
    return g.compile()


Progress = Callable[[str, int, int], None]  # (stage, done, total)


def run_check(project_id: str, *, actor: str = "cli", concurrency: int = config.CONCURRENCY,
              sample: int = 0, budget_usd: float | None = None, on_progress: Progress | None = None) -> CheckResult:
    """Run the workflow. budget_usd caps this run's LLM spend; on_progress gets (stage, done, total)."""
    project = projects.load(project_id)
    if not project.docs("bid"):
        raise ValueError("this project has no bid document: upload one with role=bid first")
    from .store import get_store

    get_store()  # open the vector store once, before the parallel judge tasks need it
    report_progress = on_progress or (lambda *a: None)
    counts = {"extract": 0, "judge": 0, "review": 0}
    totals = {"extract": 0, "judge": 0, "review": 0}
    state: dict = {}
    with llm.budget(budget_usd), trace.run("check") as run_id, trace.tags(project_id=project_id), \
            trace.collect() as events:
        report_progress("extracting requirements", 0, 0)
        for mode, chunk in build_graph().stream({"project_id": project_id, "sample": sample},
                                                config={"max_concurrency": concurrency, "recursion_limit": 50},
                                                stream_mode=["updates", "values"]):
            if mode == "values":
                state = chunk
                continue
            for node, update in chunk.items():
                if node == "plan":
                    totals["extract"] = len(update.get("windows", []))
                elif node == "merge":
                    totals["judge"] = len(update.get("to_judge", []))
                    report_progress("judging requirements", 0, totals["judge"])
                elif node in counts:
                    counts[node] += 1
                    stage = {"extract": "extracting requirements", "judge": "judging requirements",
                             "review": "reviewing MISSING requirements"}[node]
                    report_progress(stage, counts[node], totals[node] or counts[node])
                elif node == "report":
                    report_progress("writing the report", 1, 1)
    result: CheckResult = state["result"]
    summary = trace.summarize(events)["total"]
    result.stats.update(run_id=run_id, llm_calls=summary["llm_calls"], input_tokens=summary["input_tokens"],
                        output_tokens=summary["output_tokens"], cost_usd=round(summary["cost_usd"], 4),
                        cost_unknown_calls=summary["cost_unknown"], budget_usd=budget_usd)
    projects.save_check(result)
    projects.audit(project_id, "check", actor=actor, check_id=result.check_id, run_id=run_id,
                   requirements=len(result.requirements), cost_usd=result.stats["cost_usd"])
    return result
