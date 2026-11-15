"""CLI: python -m tender <command>   (the Makefile wraps these; `make help`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from agentcore import trace
from agentcore.embed import EmbedError
from agentcore.llm import LLMError
from agentcore.rerank import RerankError

from . import config, projects, report

console = Console()
RESULTS_DIR = config.RESULTS_DIR


def cmd_samples(a) -> None:
    from . import samples

    for k, p in samples.write_all().items():
        console.print(f"{k:9} {p}")


def cmd_demo_sets(a) -> None:
    from .demo_sets import build

    for p in build():
        console.print(f"wrote {p}")
    console.print("See data/demo_sets/*/README.md for what each bid should score.")


def cmd_bundle(a) -> None:
    from .bundle import build

    out = build()
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    console.print(f"wrote {out} ({len(files)} files, {size:.1f} MB)")
    console.print(f"zip:  {out.parent / (out.name + '.zip')}")


def cmd_new(a) -> None:
    p = projects.create(a.name, a.id)
    console.print(f"created project [bold]{p.project_id}[/] ({p.name})")


def cmd_list(a) -> None:
    t = Table(title="Projects")
    for col in ("id", "name", "documents", "created"):
        t.add_column(col)
    for p in projects.list_projects():
        docs = ", ".join(f"{d.role}: {d.filename}" for d in p.documents) or "-"
        t.add_row(p.project_id, p.name, docs, p.created_at)
    console.print(t)


def cmd_ingest(a) -> None:
    from .ingest.pipeline import ingest
    from .ingest.validate import ValidationError

    with trace.run("ingest"):
        for f in a.files:
            path = Path(f)
            try:
                rec, created = ingest(a.project, path.name, path.read_bytes(), a.role)
            except ValidationError as e:
                console.print(f"[red]rejected[/] {e}")
                sys.exit(1)
            state = "added" if created else "already in the project (same file), skipped"
            console.print(f"{a.role} {rec.filename}: {state} - {rec.pages} pages, {rec.chunks} chunks")
            for w in rec.warnings:
                console.print(f"  [yellow]warning[/] {w}")


def cmd_check(a) -> None:
    from .graph import run_check

    result = run_check(a.project, concurrency=a.concurrency, sample=a.sample)
    console.print(Markdown(report.summary(result, projects.load(a.project))))
    console.print(f"[dim]check {result.check_id} · trace traces/{result.stats.get('run_id')}.jsonl · "
                  f"`make matrix PROJECT={a.project}`[/]")


def cmd_matrix(a) -> None:
    result = projects.load_check(a.project, a.check)
    if a.format == "md":
        t = Table(title=f"Compliance matrix {result.check_id}", show_lines=True)
        for col in ("req", "status", "mand.", "category", "requirement (tender page)", "bid evidence", "reason"):
            t.add_column(col, overflow="fold")
        colors = {"MET": "green", "PARTIAL": "yellow", "MISSING": "red", "UNCLEAR": "white"}
        for r in report.matrix(result):
            ev = f'"{r.bid_quote}" ({r.bid_file} p. {r.bid_page})' if r.bid_quote else "-"
            t.add_row(r.req_id, Text(r.status, style=colors[r.status]), "yes" if r.mandatory else "no", r.category,
                      Text(f"{r.requirement} (p. {r.tender_page})"), Text(ev), Text(r.reason))
        console.print(t)
        return
    data = {"json": report.to_json, "csv": report.to_csv, "xlsx": report.to_xlsx}[a.format](result)
    out = Path(a.out or RESULTS_DIR / f"matrix_{a.project}_{result.check_id}.{a.format}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    console.print(f"wrote {out}")


def cmd_ask(a) -> None:
    from .ask import ask

    with trace.run("ask"):
        ans = ask(a.project, a.question, a.scope)
    console.print(Text(ans.answer))
    for c in ans.citations:
        console.print(Text(f'  [{c.excerpt_id}] {c.role} {c.filename} p. {c.page}: "{c.quote}"'), style="dim")
    for i in ans.issues + ([ans.refused] if ans.refused else []):
        console.print(f"  [yellow]{i}[/]")


def cmd_delete(a) -> None:
    if not a.yes:
        console.print("This deletes the project's files, chunks, vectors and results. Re-run with --yes.")
        sys.exit(1)
    projects.delete(a.project)
    console.print(f"deleted {a.project}")


def cmd_synth(a) -> None:
    from .synth import synthesize

    paths = synthesize(a.project, missing=a.missing, partial=a.partial, seed=a.seed)
    gold = json.loads(paths["gold"].read_text(encoding="utf-8"))
    counts = {s: sum(g["status"] == s for g in gold) for s in ("MET", "PARTIAL", "MISSING")}
    console.print(f"{len(gold)} requirements, planted: {counts}")
    for k, p in paths.items():
        console.print(f"{k:9} {p}")


def cmd_eval(a) -> None:
    from .evaluate import evaluate, load_gold, print_metrics

    result = projects.load_check(a.project, a.check)
    m = evaluate(result, load_gold(Path(a.gold)))
    print_metrics(m, console)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{a.project}_{result.check_id}.json"
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]wrote {out}[/]")


def cmd_redteam(a) -> None:
    from .evaluate import load_gold, redteam

    r = redteam(projects.load_check(a.clean), projects.load_check(a.poisoned), load_gold(Path(a.gold)))
    console.print(f"compared {r['compared']} requirements: {r['changed']} verdict(s) changed by injected text, "
                  f"{r['flipped_to_wrong_met']} flipped to a wrong MET; "
                  f"{r['flagged_chunks_poisoned']} excerpt(s) flagged as injection")
    for d in r["details"]:
        console.print(f"  {d['key']}: clean={d['clean']} poisoned={d['poisoned']} gold={d['gold']}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"redteam_{a.clean}_{a.poisoned}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")


def cmd_retrieval_eval(a) -> None:
    from .evaluate import load_gold
    from .retrieval_eval import evaluate as reval

    projects.load(a.project)
    with trace.run("retrieval-eval") as run_id, trace.collect() as events:
        r = reval(a.project, load_gold(Path(a.gold)), rerank=a.rerank)
    t = Table(title=f"Retrieval: is the answering bid passage retrieved? ({r['queries']} requirements with a known answer)")
    for col in ("mode", "recall@1", "recall@3", "recall@6 (judge sees 6)", "MRR", "missed"):
        t.add_column(col)
    for mode, m in r["modes"].items():
        missed = next(v for k, v in m.items() if k.startswith("missed"))
        t.add_row(mode, str(m["recall@1"]), str(m["recall@3"]), str(m["recall@6"]), str(m["mrr"]), ", ".join(missed) or "-")
    console.print(t)
    cost = trace.summarize(events)["total"]
    console.print(f"[dim]{cost['calls']} calls · trace {run_id}[/]")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"retrieval_{a.project}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")


def cmd_trace(a) -> None:
    s = trace.summarize(trace.load(a.run))
    t = Table(title=f"Trace {a.run}")
    for col in ("kind:name", "calls", "errors", "in tok", "out tok", "cost $", "latency s"):
        t.add_column(col, justify="left" if col == "kind:name" else "right")
    for k, r in sorted(s["by_name"].items()):
        t.add_row(k, str(r["calls"]), str(r["errors"]), str(r["input_tokens"]), str(r["output_tokens"]),
                  f"{r['cost_usd']:.4f}" + ("+?" if r["cost_unknown"] else ""), f"{r['latency_ms'] / 1000:.1f}")
    console.print(t)
    console.print(s["total"])


def cmd_serve(a) -> None:
    import socket

    import uvicorn

    with socket.socket() as sock:  # already running? say so instead of a lock traceback
        if sock.connect_ex((a.host if a.host != "0.0.0.0" else "127.0.0.1", a.port)) == 0:
            console.print(f"[yellow]Something is already running on port {a.port}[/] - probably the server: open "
                          f"http://127.0.0.1:{a.port}. To restart it, stop the other process first "
                          f"(Ctrl+C in its terminal, or `lsof -ti tcp:{a.port} | xargs kill`).")
            sys.exit(1)
    if not config.QDRANT_URL:
        lock = config.QDRANT_PATH / ".lock"
        try:
            import portalocker

            with lock.open("r+") if lock.exists() else open(lock, "w") as fh:
                portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
                portalocker.unlock(fh)
        except portalocker.exceptions.LockException:
            console.print("[yellow]The local vector store (data/qdrant) is in use by another process[/] - a running "
                          "server or a CLI command (check, ingest, ask). Stop it first: "
                          f"`lsof -t {lock}` shows its process id.")
            sys.exit(1)
        except OSError:
            pass  # folder not created yet: nothing can hold it
    console.print(f"Tender checker API on http://{a.host}:{a.port}  (UI: /  docs: /docs)")
    uvicorn.run("tender.api:app", host=a.host, port=a.port, log_level="info")


def cmd_user(a) -> None:
    from . import db

    user, key = db.create_user(a.name, is_admin=a.admin)
    console.print(f"created user [bold]{user.name}[/]{' (admin)' if a.admin else ''}")
    console.print(f"API key (shown once, store it now): [bold]{key}[/]")


def cmd_key(a) -> None:
    from . import db

    if a.action == "create":
        console.print(f"API key for {a.target} (shown once): [bold]{db.create_key(a.target)}[/]")
    else:
        console.print(f"revoked {db.revoke_key(a.target)} key(s) with prefix {a.target}")


def cmd_grant(a) -> None:
    from . import db

    projects.load(a.project)
    db.grant(a.project, a.user)
    console.print(f"{a.user} can now access {a.project}")


def cmd_users(a) -> None:
    from . import db

    t = Table(title="Users")
    for col in ("name", "admin", "keys"):
        t.add_column(col)
    for u in db.list_users():
        t.add_row(u["name"], "yes" if u["admin"] else "", ", ".join(u["keys"]))
    console.print(t)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tender", description="Tender & bid compliance checker (Google Gemini)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("samples", help="write the fictional sample tender / bid / gold key").set_defaults(fn=cmd_samples)
    sub.add_parser("demo-sets", help="2 real small RFQs x 3 fictional bids + answer keys (data/demo_sets)"
                   ).set_defaults(fn=cmd_demo_sets)
    sub.add_parser("bundle", help="build the shareable test-data folder + zip in dist/ (no secrets)"
                   ).set_defaults(fn=cmd_bundle)
    p = sub.add_parser("new", help="create a project")
    p.add_argument("name")
    p.add_argument("--id", help="project id (default: random)")
    p.set_defaults(fn=cmd_new)
    sub.add_parser("list", help="list projects").set_defaults(fn=cmd_list)
    p = sub.add_parser("ingest", help="upload tender or bid files")
    p.add_argument("project")
    p.add_argument("files", nargs="+")
    p.add_argument("--role", choices=["tender", "bid"], required=True)
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("check", help="run the compliance check")
    p.add_argument("project")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--sample", type=int, default=0, help="judge only N requirements, evenly spread (cheap trial)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("matrix", help="show / export the compliance matrix")
    p.add_argument("project")
    p.add_argument("--check")
    p.add_argument("--format", choices=["md", "json", "csv", "xlsx"], default="md")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_matrix)
    p = sub.add_parser("ask", help="ask a question about the tender / bid")
    p.add_argument("project")
    p.add_argument("question")
    p.add_argument("--scope", choices=["tender", "bid", "both"], default="both")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("delete", help="delete a project and everything in it")
    p.add_argument("project")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_delete)
    p = sub.add_parser("synth", help="synthetic bid with planted gaps for the project's tender")
    p.add_argument("project")
    p.add_argument("--missing", type=float, default=0.2)
    p.add_argument("--partial", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("eval", help="score the latest check against a gold key")
    p.add_argument("project")
    p.add_argument("--gold", required=True)
    p.add_argument("--check")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("redteam", help="compare checks on a clean and a poisoned bid")
    p.add_argument("clean")
    p.add_argument("poisoned")
    p.add_argument("--gold", required=True)
    p.set_defaults(fn=cmd_redteam)
    p = sub.add_parser("serve", help="run the HTTP API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("user", help="create a user and print their API key")
    p.add_argument("action", choices=["create"])
    p.add_argument("name")
    p.add_argument("--admin", action="store_true", help="may see every project")
    p.set_defaults(fn=cmd_user)
    p = sub.add_parser("key", help="create another key for a user, or revoke keys by prefix")
    p.add_argument("action", choices=["create", "revoke"])
    p.add_argument("target", help="user name (create) or key prefix like tk_Ab12Cd (revoke)")
    p.set_defaults(fn=cmd_key)
    p = sub.add_parser("grant", help="give a user access to a project")
    p.add_argument("user")
    p.add_argument("project")
    p.set_defaults(fn=cmd_grant)
    sub.add_parser("users", help="list users and key prefixes").set_defaults(fn=cmd_users)
    p = sub.add_parser("retrieval-eval", help="recall@k of the answering bid passage: dense vs keyword vs hybrid")
    p.add_argument("project")
    p.add_argument("--gold", required=True)
    p.add_argument("--rerank", choices=["none", "llm", "bedrock"], default="none")
    p.set_defaults(fn=cmd_retrieval_eval)
    p = sub.add_parser("trace", help="summarise a trace")
    p.add_argument("run")
    p.set_defaults(fn=cmd_trace)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (projects.NotFound, ValueError, FileNotFoundError, LLMError, EmbedError, RerankError) as e:
        console.print(f"[red]error[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
