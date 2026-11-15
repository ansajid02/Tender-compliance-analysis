"""Compliance matrix + critical-gaps summary + exports (JSON / CSV / XLSX).

The summary is written by code from the verdicts (no LLM), so it cannot add anything that is not in the
matrix. Order: mandatory MISSING first, then mandatory PARTIAL / UNCLEAR, then optional, then MET.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter

from .schemas import STATUSES, CheckResult, MatrixRow, Project

_RANK = {"MISSING": 0, "PARTIAL": 1, "UNCLEAR": 2, "MET": 3}
COLUMNS = ["req_id", "status", "mandatory", "category", "requirement", "tender_page", "tender_section",
           "tender_quote", "bid_quote", "bid_file", "bid_page", "reason", "reviewed", "downgraded"]


def matrix(result: CheckResult) -> list[MatrixRow]:
    verdicts = {v.req_id: v for v in result.verdicts}
    rows = []
    for r in result.requirements:
        v = verdicts.get(r.req_id)
        if v is None:
            continue
        ev = v.evidence
        rows.append(MatrixRow(
            req_id=r.req_id, category=r.category, mandatory=r.mandatory, requirement=r.text, tender_quote=r.quote,
            tender_page=r.page, tender_section=" ".join(x for x in (r.clause, r.section.split(" > ")[-1]) if x),
            status=v.status, reason=v.reason, bid_quote=ev.quote if ev else "", bid_file=ev.filename if ev else "",
            bid_page=ev.page if ev else None, reviewed=v.reviewed, downgraded=v.downgraded_from is not None))
    return sorted(rows, key=lambda m: (not m.mandatory, _RANK[m.status], m.req_id))


def summary(result: CheckResult, project: Project | None = None) -> str:
    rows = matrix(result)
    mand = [r for r in rows if r.mandatory]
    c_all, c_mand = Counter(r.status for r in rows), Counter(r.status for r in mand)
    lines = [f"# Compliance check{f': {project.name}' if project else ''}", ""]
    if project:
        for d in project.documents:
            lines.append(f"- {d.role}: {d.filename} ({d.pages} pages)")
        lines.append("")
    lines.append(f"**{len(rows)} requirements** ({len(mand)} mandatory): "
                 + ", ".join(f"{c_all[s]} {s}" for s in STATUSES))
    lines.append(f"Mandatory: " + ", ".join(f"{c_mand[s]} {s}" for s in STATUSES))
    lines.append("")
    critical = [r for r in mand if r.status == "MISSING"]
    weak = [r for r in mand if r.status in ("PARTIAL", "UNCLEAR")]
    if critical:
        lines += ["## Critical gaps (mandatory, MISSING)", ""]
        lines += [f"- **{r.req_id}** [{r.category}] {r.requirement} (tender p. {r.tender_page}"
                  f"{', ' + r.tender_section if r.tender_section else ''})" for r in critical]
        lines.append("")
    if weak:
        lines += ["## Needs work (mandatory, PARTIAL or UNCLEAR)", ""]
        for r in weak:
            where = f"; bid p. {r.bid_page}" if r.bid_page else ""
            lines.append(f"- **{r.req_id}** {r.status}: {r.requirement} (tender p. {r.tender_page}{where}) "
                         f"- {r.reason}")
        lines.append("")
    if not critical and not weak:
        lines += ["No mandatory gaps found. Check the UNCLEAR and optional items before submitting.", ""]
    if result.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in result.warnings] + [""]
    st = result.stats
    if "llm_calls" in st:
        lines.append(f"_{st.get('llm_calls')} LLM calls, ${st.get('cost_usd', 0):.4f}"
                     f"{' (+ unpriced calls)' if st.get('cost_unknown_calls') else ''}; "
                     f"{st.get('quote_unverified', 0)} unverified requirement(s) dropped, "
                     f"{st.get('downgraded', 0)} verdict(s) downgraded by the grounding check, "
                     f"{st.get('review_changed', 0)} MISSING changed by the reviewer._")
    return "\n".join(lines)


def to_json(result: CheckResult) -> str:
    return json.dumps({"check_id": result.check_id, "project_id": result.project_id,
                       "created_at": result.created_at, "stats": result.stats, "warnings": result.warnings,
                       "matrix": [r.model_dump() for r in matrix(result)]}, indent=2, ensure_ascii=False)


def to_csv(result: CheckResult) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in matrix(result):
        w.writerow(r.model_dump())
    return "﻿" + buf.getvalue()  # BOM: Excel opens UTF-8 (Arabic) correctly


_FILL = {"MET": "C6EFCE", "PARTIAL": "FFEB9C", "MISSING": "FFC7CE", "UNCLEAR": "D9D9D9"}


def to_xlsx(result: CheckResult) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Compliance matrix"
    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in matrix(result):
        d = r.model_dump()
        ws.append([d[c] if d[c] is not None else "" for c in COLUMNS])
        ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor=_FILL[r.status])
    widths = {"requirement": 50, "tender_quote": 50, "bid_quote": 50, "reason": 50, "tender_section": 25}
    for i, col in enumerate(COLUMNS, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 12)
        for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    s = wb.create_sheet("Summary")
    for line in summary(result).splitlines():
        s.append([line])
    s.column_dimensions["A"].width = 140
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
