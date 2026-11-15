"""Background jobs for the API: ingestion and compliance checks run in a thread pool, tracked in the DB.

- one job at a time per project (an upload never races a check on the same project)
- progress (stage, done/total) is written to the job row; the UI polls GET /jobs/{id}
- errors are stored as a short, safe message; the traceback goes to the server log only
- at startup, jobs left queued/running by a previous process are marked failed (they will never finish)
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import db

log = logging.getLogger("tender.jobs")

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_project_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

Work = Callable[[Callable[[str, int, int], None]], dict]  # work(progress) -> result


def pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(int(os.environ.get("JOB_WORKERS", "").strip() or 2),
                                       thread_name_prefix="job")
        return _pool


def submit(kind: str, project_id: str, user: db.User | None, work: Work, params: dict | None = None) -> db.Job:
    job = db.new_job(kind, project_id, user, params)
    ctx = contextvars.copy_context()
    pool().submit(ctx.run, _run, job.id, project_id, work)
    return job


def _run(job_id: str, project_id: str, work: Work) -> None:
    last = {"t": 0.0, "stage": ""}

    def progress(stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        if stage != last["stage"] or now - last["t"] > 0.5 or (total and done >= total):  # throttle DB writes
            last.update(t=now, stage=stage)
            db.update_job(job_id, stage=stage[:80], done=done, total=total)

    with _project_locks[project_id]:
        db.update_job(job_id, status="running", stage="starting")
        try:
            result = work(progress)
            db.update_job(job_id, status="done", stage="done", result=result)
        except Exception as e:  # noqa: BLE001 - every failure must end the job, never hang it
            msg = f"{type(e).__name__}: {e}"
            if "interpreter shutdown" in msg:  # the server was stopped (Ctrl+C) while this job ran
                msg = "interrupted: the server was stopped while this job was running. Run it again."
            else:
                log.exception("job %s failed", job_id)
            db.update_job(job_id, status="failed", stage="failed", error=msg[:1000])


def wait(job_id: str, timeout: float = 60.0) -> db.Job:
    """Tests / CLI: block until the job is finished."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = db.get_job(job_id)
        if job and job.status in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")
