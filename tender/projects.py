"""Projects on disk: data/projects/<project_id>/

    project.json            Project (documents registry)
    files/<doc_id>.<ext>    original upload (the UI opens the PDF at a cited page)
    docs/<doc_id>/pages.json    visible text per page (quote verification)
    docs/<doc_id>/chunks.jsonl  the chunks that were indexed (same ids as in Qdrant)
    checks/<check_id>.json  CheckResult of each compliance check
    audit.jsonl             who uploaded / checked / asked what

The API step moves the registry + audit log to Postgres; this module's functions stay the interface.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path

from langchain_core.documents import Document

from . import config
from .schemas import CheckResult, Project, now

_lock = threading.Lock()
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class NotFound(KeyError):
    pass


def _dir(project_id: str) -> Path:
    if not _ID.match(project_id):  # ids become paths: never let "../" in
        raise NotFound(f"invalid project id '{project_id}'")
    return config.PROJECTS_DIR / project_id


def create(name: str, project_id: str | None = None) -> Project:
    pid = project_id or f"p-{uuid.uuid4().hex[:10]}"
    d = _dir(pid)
    if d.exists():
        raise ValueError(f"project '{pid}' already exists")
    p = Project(project_id=pid, name=name)
    save(p)
    audit(pid, "create_project", name=name)
    return p


def save(p: Project) -> None:
    d = _dir(p.project_id)
    d.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = d / "project.json.tmp"
        tmp.write_text(p.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(d / "project.json")


def load(project_id: str) -> Project:
    f = _dir(project_id) / "project.json"
    if not f.exists():
        raise NotFound(f"project '{project_id}' not found")
    return Project.model_validate_json(f.read_text(encoding="utf-8"))


def list_projects() -> list[Project]:
    if not config.PROJECTS_DIR.exists():
        return []
    return [Project.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted(config.PROJECTS_DIR.glob("*/project.json"))]


def delete(project_id: str) -> None:
    """Removes files, pages, chunks, vectors and results. The audit line is the last thing written, then gone."""
    from .store import get_store

    load(project_id)
    get_store().delete(project_id)
    shutil.rmtree(_dir(project_id))


def save_doc_data(project_id: str, doc_id: str, ext: str, data: bytes, pages: list[str],
                  chunks: list[Document]) -> None:
    d = _dir(project_id)
    (d / "files").mkdir(parents=True, exist_ok=True)
    (d / "files" / f"{doc_id}{ext}").write_bytes(data)
    dd = d / "docs" / doc_id
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    with (dd / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")


def doc_pages(project_id: str, doc_id: str) -> list[str]:
    return json.loads((_dir(project_id) / "docs" / doc_id / "pages.json").read_text(encoding="utf-8"))


def doc_chunks(project_id: str, doc_id: str) -> list[Document]:
    path = _dir(project_id) / "docs" / doc_id / "chunks.jsonl"
    return [Document(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_path(project_id: str, doc_id: str) -> Path:
    matches = list((_dir(project_id) / "files").glob(f"{doc_id}.*"))
    if not matches:
        raise NotFound(f"file for document '{doc_id}' not found")
    return matches[0]


def save_check(result: CheckResult) -> Path:
    d = _dir(result.project_id) / "checks"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{result.check_id}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_check(project_id: str, check_id: str | None = None) -> CheckResult:
    d = _dir(project_id) / "checks"
    if check_id:
        path = d / f"{check_id}.json"
    else:
        found = sorted(d.glob("*.json")) if d.exists() else []
        if not found:
            raise NotFound(f"project '{project_id}' has no compliance check yet")
        path = found[-1]
    if not path.exists():
        raise NotFound(f"check '{check_id}' not found")
    return CheckResult.model_validate_json(path.read_text(encoding="utf-8"))


def audit(project_id: str, action: str, actor: str = "cli", **detail) -> None:
    d = _dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": now(), "actor": actor, "action": action, **detail}, ensure_ascii=False, default=str)
    with _lock, (d / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def list_checks(project_id: str) -> list[dict]:
    """Newest first: id, time and verdict counts of every saved check."""
    d = _dir(project_id) / "checks"
    out = []
    for f in sorted(d.glob("*.json"), reverse=True) if d.exists() else []:
        r = CheckResult.model_validate_json(f.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for v in r.verdicts:
            counts[v.status] = counts.get(v.status, 0) + 1
        out.append({"check_id": r.check_id, "created_at": r.created_at, "requirements": len(r.requirements),
                    "judged": len(r.verdicts), "counts": counts, "cost_usd": r.stats.get("cost_usd"),
                    "sampled": r.stats.get("sampled")})
    return out
