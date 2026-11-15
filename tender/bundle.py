"""`tender bundle`: build the shareable test-data folder (+ zip) for Google Drive.

    dist/tender-checker-test-data/
      1-sample-fictional/            fictional tender + bid + poisoned bid + gold key
      2-pildat-laptops-printers/     real public RFQ + 3 fictional bids (strong / weak / poisoned) + gold keys
      3-idlo-philippines-laptops/    real public RFQ + 3 fictional bids + gold keys
      4-idlo-ukraine-printer-synth/  real public RFQ + LLM-written synthetic bid (+ poisoned) + gold key
      5-extra-tenders/               more real public RFQs (no bids) to try `tender synth` on
      6-ready-made-results/          already-checked projects + their vectors: see results in the UI for $0
      README.md, SOURCES.md, checksums.sha256

Never included: .env, API keys, data/app.db (users + key hashes), traces.
Maintainer tool: it reads this machine's data/ folder (downloaded RFQs, synth output, checked projects).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from agentcore.config import PROJECT_ROOT

from . import config, demo_sets, samples

NAME = "tender-checker-test-data"
READY_PROJECTS = ["demo", "demo-poisoned", "idlo", "p-8bf75a66ee", "p-90ae3d1e93"]
RAW = config.DATA_DIR / "tenders" / "raw"
EXTRA = [  # (source file, bundle name, publisher URL, note)
    (RAW / "rfq_no._2026-0004.pdf", "CTBTO_RFQ_2026-0004.pdf",
     "https://www.ctbto.org/sites/default/files/2026-02/rfq_no._2026-0004.pdf", "11 pages, clean text"),
    (RAW / "cand1" / "1787325158740-ekxs713a.pdf", "ISA_RFQ_laptops_2026.pdf",
     "https://doc.isa.int/procurement/documents/1787325158740-ekxs713a.pdf", "15 pages, many tables"),
    (RAW / "cand4" / "rfq-bidding-documents-office-equipmnet-eng.docx", "Kyrgyz_MoE_RFQ_office_equipment.docx",
     "https://edu.gov.kg/media/uploads/2026/02/16/rfq-bidding-documents-office-equipmnet-eng.docx",
     "21 pages, a WORD tender (tests the DOCX tender path)"),
    (RAW / "015-RFQ-BPA-Solar-Fans.pdf", "Iraq_RFQ_solar_fans.pdf",
     "https://www.iraq-businessnews.com/wp-content/uploads/2026/03/015-RFQ-BPA-Solar-Fans.pdf", "27 pages, many tables"),
    (RAW / "RFQ-Water-Supply-Scheme-Ghousabad-Quetta.pdf", "UNHabitat_RFQ_water_supply_Quetta.pdf",
     "https://unhabitat.org.pk/wp-content/uploads/2026/08/RFQ-Water-Supply-Scheme-Ghousabad-Quetta.pdf",
     "34 pages, pages 18-20 are SCANNED: needs OCR (Docker image or Tesseract)"),
]
IDLO_PRINTER_URL = "https://www.idlo.int/sites/default/files/rfq_office_printer.pdf"


def _counts(gold: list[dict]) -> str:
    c = {s: sum(g["status"] == s for g in gold) for s in ("MET", "PARTIAL", "MISSING")}
    return f"{c['MET']} MET / {c['PARTIAL']} PARTIAL / {c['MISSING']} MISSING"


def _ready_results(out: Path) -> list[str]:
    """Copy checked projects + a vector store holding only their points (no users, no keys)."""
    from qdrant_client import QdrantClient, models

    dest = out / "6-ready-made-results"
    (dest / "projects").mkdir(parents=True)
    kept = []
    for pid in READY_PROJECTS:
        src = config.PROJECTS_DIR / pid
        if src.exists():
            shutil.copytree(src, dest / "projects" / pid)
            kept.append(pid)
    qsrc, qdst = config.QDRANT_PATH, dest / "qdrant"
    shutil.copytree(qsrc, qdst, ignore=shutil.ignore_patterns(".lock"))
    client = QdrantClient(path=str(qdst))  # the COPY: never touches the running server's store
    try:
        for coll in client.get_collections().collections:
            client.delete(coll.name, points_selector=models.FilterSelector(filter=models.Filter(must_not=[
                models.FieldCondition(key="metadata.project_id", match=models.MatchAny(any=kept))])))
        left = sum(client.count(c.name, exact=True).count for c in client.get_collections().collections)
    finally:
        client.close()
    (qdst / ".lock").unlink(missing_ok=True)
    (dest / "README.md").write_text("\n".join([
        "# Ready-made results",
        "",
        "Already-checked projects, so you can explore the UI without spending any model credit.",
        "Copy into the project's `data/` folder with the server STOPPED:",
        "",
        "```bash",
        "mkdir -p data/projects",
        "cp -R test-data/6-ready-made-results/projects/* data/projects/",
        "rm -rf data/qdrant && cp -R test-data/6-ready-made-results/qdrant data/qdrant",
        "```",
        "",
        "Then `make user NAME=you` (admin users see every project) and `make serve`.",
        "Requires the default embeddings (`EMBED_MODEL=gemini-embedding-001`, `EMBED_DIM=1536`) for Ask to work.",
        "",
        "| project id | what it is |",
        "|---|---|",
        "| demo | fictional cloud-hosting RFP + clean bid (sample set 1) |",
        "| demo-poisoned | same RFP + poisoned bid (red team) |",
        "| idlo | IDLO Ukraine printer RFQ + synthetic bid (set 4), 25-requirement sample check |",
        "| p-8bf75a66ee | PILDAT RFQ + strong bid (set 2) |",
        "| p-90ae3d1e93 | PILDAT RFQ + weak bid (set 2) |",
        "",
        f"Vector points included: {left}. No users or API keys are included.",
    ]), encoding="utf-8")
    return kept


def build(out_root: Path | None = None) -> Path:
    out_root = out_root or PROJECT_ROOT / "dist"
    out = out_root / NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []

    # 1. fictional sample
    s1 = out / "1-sample-fictional"
    samples.write_all(s1)
    gold = json.loads((s1 / samples.GOLD).read_text())
    (s1 / "README.md").write_text("\n".join([
        "# Fictional sample: cloud hosting RFP (6 pages)",
        "",
        "Invented ministry and companies. Planted answers: 5 MISSING (some with decoys: ISO 9001 for ISO 27001,",
        "RPO for RTO), 3 PARTIAL (90 vs 120 days validity, 2 vs 3 references, 99.9% vs 99.95%).",
        "",
        f"- `{samples.TENDER_PDF}`: the tender",
        f"- `{samples.BID_DOCX}`: the bid, expected {_counts(gold)}",
        f"- `{samples.BID_POISONED}`: the same bid + hidden and visible prompt injection (verdicts must not change)",
        f"- `{samples.GOLD}`: the answer key for both bids",
    ]), encoding="utf-8")
    rows.append(("1-sample-fictional", samples.TENDER_PDF, f"{samples.BID_DOCX} / {samples.BID_POISONED}", _counts(gold)))

    # 2 + 3. real RFQs with hand-written fictional bids
    demo_sets.build(config.DATA_DIR / "demo_sets")
    for folder, s in (("2-pildat-laptops-printers", demo_sets.PILDAT), ("3-idlo-philippines-laptops", demo_sets.IDLO_PH)):
        shutil.copytree(config.DATA_DIR / "demo_sets" / s["name"], out / folder)
        for bid in s["bids"]:
            g = json.loads((out / folder / f"gold_{Path(bid['file']).stem}.json").read_text())
            rows.append((folder, "tender.pdf", bid["file"], _counts(g)))

    # 4. real RFQ with the LLM-written synthetic bid
    s4 = out / "4-idlo-ukraine-printer-synth"
    s4.mkdir()
    shutil.copy(RAW / "rfq_office_printer.pdf", s4 / "tender.pdf")
    for f in ("bid_synth.docx", "bid_synth_poisoned.docx", "gold.json"):
        shutil.copy(config.DATA_DIR / "synth" / "idlo" / f, s4 / f)
    g4 = json.loads((s4 / "gold.json").read_text())
    (s4 / "README.md").write_text("\n".join([
        "# IDLO RFQ UA-2025-000140: office printer for Ukraine (13 pages, real public RFQ)",
        "",
        "`bid_synth.docx` was written by `tender synth` (gemini-3.1-flash-lite) with randomly planted gaps;",
        f"`gold.json` is its answer key ({len(g4)} requirements: {_counts(g4)}). `bid_synth_poisoned.docx` adds",
        "prompt injection. The tender has ~78 requirements: a full check costs about $0.15; `--sample 25` about $0.05.",
    ]), encoding="utf-8")
    rows.append(("4-idlo-ukraine-printer-synth", "tender.pdf", "bid_synth.docx / bid_synth_poisoned.docx", _counts(g4)))

    # 5. extra tenders (no bids)
    s5 = out / "5-extra-tenders"
    s5.mkdir()
    lines = ["# Extra real public RFQs (no bids)", "", "Try `tender synth <project>` on them to generate a bid with planted "
             "gaps and its answer key.", "", "| file | note |", "|---|---|"]
    for src, name, url, note in EXTRA:
        if src.exists():
            shutil.copy(src, s5 / name)
            lines.append(f"| {name} | {note} |")
    (s5 / "README.md").write_text("\n".join(lines), encoding="utf-8")

    # 6. ready-made results
    kept = _ready_results(out)

    # sources + top README + checksums
    src_lines = ["# Sources", "", "Real tenders are public RFQs downloaded from their publishers for testing this tool.",
                 "They belong to their publishers: share this folder for testing only, do not republish the RFQs.",
                 "All bids, companies and the fictional sample are invented.", "",
                 f"- PILDAT (Pakistan) RFQ PILDAT/02-2026-01: {demo_sets.PILDAT['url']}",
                 f"- IDLO Philippines RFQ PHIL_2025_001: {demo_sets.IDLO_PH['url']}",
                 f"- IDLO Ukraine RFQ UA-2025-000140: {IDLO_PRINTER_URL}"]
    src_lines += [f"- {name}: {url}" for src, name, url, _ in EXTRA if src.exists()]
    (out / "SOURCES.md").write_text("\n".join(src_lines), encoding="utf-8")
    table = ["| folder | tender | bid(s) | answer key |", "|---|---|---|---|"]
    table += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    (out / "README.md").write_text("\n".join([
        "# Tender Checker: test data",
        "",
        "Put this folder inside the cloned project and name it `test-data` (git-ignored):",
        "`mv ~/Downloads/tender-checker-test-data <path-to>/5a-tender-checker/test-data`,",
        "then follow \"Recreate and run\" in the project README.",
        "",
        "## Test sets (upload in the web UI: one project per bid)",
        "",
        *table,
        "",
        "Each set folder has a README with the expected verdicts. Score a finished check (server stopped):",
        "`uv run tender eval <project-id> --gold test-data/<set>/<gold file>.json`",
        "",
        "## Ready-made results (`6-ready-made-results/`)",
        "",
        f"Projects already checked ({', '.join(kept)}): copy them into `data/` to explore the UI without any",
        "model cost. See that folder's README.",
        "",
        "## Not included",
        "",
        "Secrets (.env, API keys), the user database (data/app.db), traces.",
        "",
        "See SOURCES.md for where each tender comes from.",
    ]), encoding="utf-8")
    sums = []
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}")
    (out / "checksums.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")

    zip_base = out_root / NAME
    shutil.make_archive(str(zip_base), "zip", root_dir=out_root, base_dir=NAME)
    return out
