"""HTTP API (FastAPI). OpenAPI docs at /docs, the web UI at /.

Auth: an API key per user (`X-API-Key: tk_...` or `Authorization: Bearer tk_...`), created with
`tender user create NAME`. Users see only projects they created or were granted; admins see all.
A project the caller may not access answers 404, never 403 (its existence is not revealed).

Long work runs as background jobs (POST returns a job id; poll GET /jobs/{id}):
    POST /projects/{id}/documents   upload tender / bid files   -> ingest job per file
    POST /projects/{id}/check       compliance check            -> check job (sample + budget options)
Streaming: POST /projects/{id}/ask answers as Server-Sent Events (status -> answer -> done). The answer is
sent only after its citations are verified (no unverified claim ever reaches the client).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentcore import llm, trace
from agentcore.config import settings

from . import config, db, jobs, projects, report, viewer
from .ingest.validate import ValidationError, validate

log = logging.getLogger("tender.api")
WEB_DIR = Path(__file__).parent / "web"
CHECK_BUDGET_USD = float(os.environ.get("CHECK_BUDGET_USD", "").strip() or 1.0)
MAX_CHECK_BUDGET_USD = float(os.environ.get("MAX_CHECK_BUDGET_USD", "").strip() or 5.0)
ASK_BUDGET_USD = float(os.environ.get("ASK_BUDGET_USD", "").strip() or 0.02)


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = db.fail_interrupted_jobs()
    if n:
        log.warning("%d job(s) from a previous run marked failed", n)
    from .store import get_store

    get_store()  # one vector-store client for the whole process
    yield
    n = db.fail_interrupted_jobs("interrupted: the server was stopped while this job was running. Run it again.")
    if n:
        log.warning("server stopping: %d running job(s) marked failed", n)


app = FastAPI(title="Tender & Bid Compliance Checker", version="0.1.0", lifespan=lifespan,
              description="Lists every tender requirement a bid misses, with page citations. "
                          "Authenticate with `X-API-Key: tk_...`.")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.exception_handler(projects.NotFound)
async def _not_found(request, exc):
    return JSONResponse({"detail": "not found"}, status_code=404)


# ---------------------------------------------------------------- auth


def current_user(x_api_key: str | None = Header(None), authorization: str | None = Header(None)) -> db.User:
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    user = db.authenticate(key)
    if user is None:
        raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
    return user


def project_for(project_id: str, user: db.User = Depends(current_user)) -> projects.Project:
    try:
        p = projects.load(project_id)
    except projects.NotFound:
        raise HTTPException(404, "not found") from None
    if not db.can_access(user, project_id):
        raise HTTPException(404, "not found")
    return p


# ---------------------------------------------------------------- basics


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    """No auth: for Docker / the host. HTTP 503 + 'degraded' if a dependency is down."""
    from .store import get_store

    try:
        get_store().client.get_collections()
        vectors = True
    except Exception:  # noqa: BLE001
        vectors = False
    ok_db = db.ping()
    s = settings()
    body = {"status": "ok" if ok_db and vectors else "degraded", "database": ok_db, "vector_store": vectors,
            "llm_backend": s.llm_backend, "models": {"flash": s.model_for(s.llm_backend, "flash"),
                                                     "pro": s.model_for(s.llm_backend, "pro")}}
    return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)


@app.get("/me")
def me(user: db.User = Depends(current_user)) -> dict:
    return {"name": user.name, "admin": user.is_admin,
            "defaults": {"check_budget_usd": CHECK_BUDGET_USD, "max_check_budget_usd": MAX_CHECK_BUDGET_USD}}


# ---------------------------------------------------------------- projects


class NewProject(BaseModel):
    name: str = Field(min_length=1, max_length=120)


def _project_view(p: projects.Project) -> dict:
    checks = projects.list_checks(p.project_id)
    return {"project_id": p.project_id, "name": p.name, "created_at": p.created_at,
            "documents": [d.model_dump(exclude={"project_id", "sha256"}) for d in p.documents],
            "latest_check": checks[0] if checks else None, "checks": len(checks)}


@app.post("/projects", status_code=201)
def create_project(body: NewProject, user: db.User = Depends(current_user)) -> dict:
    p = projects.create(body.name)
    db.grant(p.project_id, user.name)
    projects.audit(p.project_id, "create_project", actor=user.name)
    db.audit(user.name, "create_project", p.project_id, name=body.name)
    return _project_view(p)


@app.get("/projects")
def list_projects(user: db.User = Depends(current_user)) -> list[dict]:
    allowed = db.member_projects(user)
    return [_project_view(p) for p in projects.list_projects() if allowed is None or p.project_id in allowed]


@app.get("/projects/{project_id}")
def get_project(p: projects.Project = Depends(project_for)) -> dict:
    view = _project_view(p)
    view["check_history"] = projects.list_checks(p.project_id)
    view["jobs"] = [j.to_dict() for j in db.project_jobs(p.project_id)]
    return view


@app.delete("/projects/{project_id}")
def delete_project(p: projects.Project = Depends(project_for), user: db.User = Depends(current_user)) -> dict:
    """Deletes files, page texts, chunks, vectors, results and jobs. The audit entry stays (in the DB)."""
    projects.delete(p.project_id)
    db.forget_project(p.project_id)
    db.audit(user.name, "delete_project", p.project_id, name=p.name)
    return {"deleted": p.project_id}


# ---------------------------------------------------------------- documents


async def _read_limited(f: UploadFile) -> bytes:
    limit = config.MAX_UPLOAD_MB * 1_000_000
    data = await f.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{f.filename}: over the {config.MAX_UPLOAD_MB} MB limit")
    return data


@app.post("/projects/{project_id}/documents", status_code=202)
async def upload(role: Literal["tender", "bid"] = Form(...), files: list[UploadFile] = File(...),
                 p: projects.Project = Depends(project_for), user: db.User = Depends(current_user)) -> dict:
    """Validates each file now (fast); parsing, chunking and embedding run as a background job."""
    from .ingest.pipeline import ingest

    accepted, rejected = [], []
    for f in files:
        name = Path(f.filename or "upload").name
        data = await _read_limited(f)
        try:
            validate(name, data)
        except ValidationError as e:
            rejected.append({"filename": name, "error": str(e)})
            continue

        def work(progress, name=name, data=data) -> dict:
            progress("parsing, chunking and embedding", 0, 1)
            with trace.run("ingest"):
                rec, created = ingest(p.project_id, name, data, role, actor=user.name)
            return {"doc_id": rec.doc_id, "filename": rec.filename, "pages": rec.pages, "chunks": rec.chunks,
                    "created": created, "warnings": rec.warnings}

        job = jobs.submit("ingest", p.project_id, user, work, params={"filename": name, "role": role})
        accepted.append({"filename": name, "job_id": job.id})
    db.audit(user.name, "upload", p.project_id, role=role, files=[a["filename"] for a in accepted],
             rejected=len(rejected))
    if not accepted:
        raise HTTPException(400, {"rejected": rejected})
    return {"accepted": accepted, "rejected": rejected}


def _doc(p: projects.Project, doc_id: str):
    d = next((d for d in p.documents if d.doc_id == doc_id), None)
    if d is None:
        raise HTTPException(404, "not found")
    return d


@app.get("/projects/{project_id}/documents/{doc_id}/file")
def download(doc_id: str, p: projects.Project = Depends(project_for)) -> FileResponse:
    d = _doc(p, doc_id)
    return FileResponse(projects.file_path(p.project_id, doc_id), filename=d.filename)


@app.get("/projects/{project_id}/documents/{doc_id}/pages/{page}/image", responses={200: {"content": {"image/png": {}}}})
def page_image(doc_id: str, page: int, q: str = Query("", max_length=2000),
               p: projects.Project = Depends(project_for)) -> Response:
    """PDF page as PNG with the quote highlighted (the citation viewer)."""
    d = _doc(p, doc_id)
    if not d.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "not a PDF: use /text for DOCX pages")
    try:
        png, hits = viewer.pdf_page_png(projects.file_path(p.project_id, doc_id), page, q)
    except IndexError as e:
        raise HTTPException(404, str(e)) from None
    return Response(png, media_type="image/png", headers={"X-Highlights": str(hits), "Cache-Control": "private, max-age=300"})


@app.get("/projects/{project_id}/documents/{doc_id}/pages/{page}/text")
def page_text(doc_id: str, page: int, q: str = Query("", max_length=2000),
              p: projects.Project = Depends(project_for)) -> dict:
    _doc(p, doc_id)
    try:
        return viewer.page_text(p.project_id, doc_id, page, q)
    except IndexError as e:
        raise HTTPException(404, str(e)) from None


# ---------------------------------------------------------------- check + results


class CheckRequest(BaseModel):
    sample: int = Field(0, ge=0, le=2000, description="judge only N requirements, evenly spread (0 = all)")
    budget_usd: float | None = Field(None, gt=0, description="cap on this check's LLM spend (default CHECK_BUDGET_USD)")


@app.post("/projects/{project_id}/check", status_code=202)
def start_check(body: CheckRequest | None = None, p: projects.Project = Depends(project_for),
                user: db.User = Depends(current_user)) -> dict:
    from .graph import run_check

    body = body or CheckRequest()
    if not p.docs("tender") or not p.docs("bid"):
        raise HTTPException(400, "upload a tender and a bid first")
    budget = min(body.budget_usd or CHECK_BUDGET_USD, MAX_CHECK_BUDGET_USD)

    def work(progress) -> dict:
        r = run_check(p.project_id, actor=user.name, sample=body.sample, budget_usd=budget, on_progress=progress)
        return {"check_id": r.check_id, "requirements": len(r.requirements), "judged": len(r.verdicts),
                "cost_usd": r.stats.get("cost_usd")}

    job = jobs.submit("check", p.project_id, user, work, params={"sample": body.sample, "budget_usd": budget})
    db.audit(user.name, "check", p.project_id, job_id=job.id, sample=body.sample, budget_usd=budget)
    return {"job_id": job.id, "budget_usd": budget}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: db.User = Depends(current_user)) -> dict:
    job = db.get_job(job_id)
    if job is None or not db.can_access(user, job.project_id):
        raise HTTPException(404, "not found")
    return job.to_dict()


@app.get("/projects/{project_id}/matrix")
def matrix(format: Literal["json", "csv", "xlsx"] = "json", check_id: str | None = None,
           p: projects.Project = Depends(project_for)):
    result = projects.load_check(p.project_id, check_id)
    stem = f"compliance_{p.project_id}_{result.check_id}"
    if format == "csv":
        return Response(report.to_csv(result), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})
    if format == "xlsx":
        return Response(report.to_xlsx(result),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.xlsx"'})
    data = json.loads(report.to_json(result))
    docs = {d.doc_id: d.filename for d in p.documents}
    by_req = {r.req_id: r for r in result.requirements}
    for row in data["matrix"]:  # ids the UI needs to open citations
        req = by_req[row["req_id"]]
        row["tender_doc_id"] = req.doc_id
        v = next(v for v in result.verdicts if v.req_id == row["req_id"])
        row["bid_doc_id"] = v.evidence.doc_id if v.evidence else None
    data["documents"] = docs
    return data


@app.get("/projects/{project_id}/summary", response_class=PlainTextResponse)
def summary(check_id: str | None = None, p: projects.Project = Depends(project_for)) -> str:
    return report.summary(projects.load_check(p.project_id, check_id), p)


@app.get("/projects/{project_id}/audit")
def audit(p: projects.Project = Depends(project_for)) -> list[dict]:
    return db.audit_log(p.project_id)


# ---------------------------------------------------------------- ask (SSE)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scope: Literal["tender", "bid", "both"] = "both"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/projects/{project_id}/ask")
async def ask_stream(body: AskRequest, p: projects.Project = Depends(project_for),
                     user: db.User = Depends(current_user)) -> StreamingResponse:
    from .ask import ask

    def answer():
        with llm.budget(ASK_BUDGET_USD), trace.run("ask"):
            return ask(p.project_id, body.question, body.scope, actor=user.name)

    async def events():
        yield _sse("status", {"stage": "searching the tender and bid"})
        try:
            ans = await anyio.to_thread.run_sync(answer)
            yield _sse("status", {"stage": "citations verified"})
            yield _sse("answer", ans.model_dump())
        except llm.LLMError as e:
            yield _sse("error", {"message": str(e)[:300]})
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
