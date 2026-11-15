"""Service database (SQLAlchemy): users, hashed API keys, project access, background jobs, audit log.

DATABASE_URL: postgresql+psycopg://user:pass@host/db in docker compose; default SQLite in data/app.db.
What lives where:
- here (Postgres):        who may do what (users, keys, project members), jobs, the audit log (it survives
                          a project's deletion, so "who deleted what" is never lost)
- data volume (files):    uploads, page texts, chunks, check results (projects.py); vectors (Qdrant)
API keys are random 256-bit tokens shown once; only their SHA-256 is stored (fast to check, useless if leaked).
Tables are created on first use (no migrations yet: add Alembic before the schema changes in production).
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config

KEY_PREFIX = "tk_"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    prefix: Mapped[str] = mapped_column(String(16))  # shown in listings, e.g. "tk_Ab12Cd"
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class ProjectMember(Base):
    __tablename__ = "project_members"
    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # ingest | check
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.id, "kind": self.kind, "project_id": self.project_id, "status": self.status,
                "stage": self.stage, "done": self.done, "total": self.total, "params": self.params,
                "result": self.result, "error": self.error,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None}


class AuditEvent(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    user: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(32))
    project_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------- engine / sessions


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip() or f"sqlite:///{config.DATA_DIR / 'app.db'}"


@lru_cache(maxsize=4)
def _engine(url: str):
    if url.startswith("sqlite:///"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, connect_args={"check_same_thread": False})
    else:
        eng = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    Base.metadata.create_all(eng)
    return eng


def session() -> Session:
    return sessionmaker(_engine(database_url()), expire_on_commit=False)()


def ping() -> bool:
    try:
        with session() as s:
            s.execute(text("select 1"))
        return True
    except Exception:  # noqa: BLE001 - health check: any failure = down
        return False


# ---------------------------------------------------------------- users + keys


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_user(name: str, is_admin: bool = False) -> tuple[User, str]:
    """-> (user, API key). The key is shown once; only its hash is stored."""
    with session() as s:
        if s.scalar(select(User).where(User.name == name)):
            raise ValueError(f"user '{name}' already exists (use `tender key create {name}` for another key)")
        user = User(name=name, is_admin=is_admin)
        s.add(user)
        s.commit()
    return user, create_key(name)


def create_key(name: str) -> str:
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with session() as s:
        user = s.scalar(select(User).where(User.name == name))
        if user is None:
            raise ValueError(f"no user '{name}'")
        s.add(ApiKey(user_id=user.id, prefix=key[:10], key_hash=hash_key(key)))
        s.commit()
    return key


def revoke_key(prefix: str) -> int:
    with session() as s:
        keys = s.scalars(select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked.is_(False))).all()
        for k in keys:
            k.revoked = True
        s.commit()
        return len(keys)


def authenticate(key: str | None) -> User | None:
    if not key or not key.startswith(KEY_PREFIX):
        return None
    with session() as s:
        row = s.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(key), ApiKey.revoked.is_(False)))
        if row is None:
            return None
        row.last_used_at = _now()
        user = s.get(User, row.user_id)
        s.commit()
        return user


def list_users() -> list[dict]:
    with session() as s:
        out = []
        for u in s.scalars(select(User).order_by(User.id)):
            keys = s.scalars(select(ApiKey).where(ApiKey.user_id == u.id)).all()
            out.append({"name": u.name, "admin": u.is_admin,
                        "keys": [f"{k.prefix}...{' (revoked)' if k.revoked else ''}" for k in keys]})
        return out


# ---------------------------------------------------------------- project access


def grant(project_id: str, user_name: str) -> None:
    with session() as s:
        user = s.scalar(select(User).where(User.name == user_name))
        if user is None:
            raise ValueError(f"no user '{user_name}'")
        if not s.get(ProjectMember, (project_id, user.id)):
            s.add(ProjectMember(project_id=project_id, user_id=user.id))
            s.commit()


def can_access(user: User, project_id: str) -> bool:
    if user.is_admin:
        return True
    with session() as s:
        return s.get(ProjectMember, (project_id, user.id)) is not None


def member_projects(user: User) -> set[str] | None:
    """Project ids the user may see; None = all (admin)."""
    if user.is_admin:
        return None
    with session() as s:
        return set(s.scalars(select(ProjectMember.project_id).where(ProjectMember.user_id == user.id)))


def forget_project(project_id: str) -> None:
    with session() as s:
        for m in s.scalars(select(ProjectMember).where(ProjectMember.project_id == project_id)):
            s.delete(m)
        for j in s.scalars(select(Job).where(Job.project_id == project_id)):
            s.delete(j)
        s.commit()


# ---------------------------------------------------------------- jobs + audit


def new_job(kind: str, project_id: str, user: User | None, params: dict | None = None) -> Job:
    job = Job(id=f"j-{secrets.token_hex(8)}", kind=kind, project_id=project_id,
              user_id=user.id if user else None, params=params or {})
    with session() as s:
        s.add(job)
        s.commit()
    return job


def update_job(job_id: str, **fields) -> None:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        if fields.get("status") == "running" and job.started_at is None:
            job.started_at = _now()
        if fields.get("status") in ("done", "failed"):
            job.finished_at = _now()
        s.commit()


def get_job(job_id: str) -> Job | None:
    with session() as s:
        return s.get(Job, job_id)


def project_jobs(project_id: str, limit: int = 20) -> list[Job]:
    with session() as s:
        return list(s.scalars(select(Job).where(Job.project_id == project_id)
                              .order_by(Job.created_at.desc()).limit(limit)))


def fail_interrupted_jobs(reason: str = "interrupted: the server restarted") -> int:
    """Jobs that were queued/running when the server stopped will never finish (startup and shutdown)."""
    with session() as s:
        jobs = s.scalars(select(Job).where(Job.status.in_(("queued", "running")))).all()
        for j in jobs:
            j.status, j.stage, j.error, j.finished_at = "failed", "failed", reason, _now()
        s.commit()
        return len(jobs)


def audit(user: str, action: str, project_id: str | None = None, **detail) -> None:
    with session() as s:
        s.add(AuditEvent(user=user, action=action, project_id=project_id, detail=detail))
        s.commit()


def audit_log(project_id: str | None = None, limit: int = 100) -> list[dict]:
    with session() as s:
        q = select(AuditEvent).order_by(AuditEvent.ts.desc()).limit(limit)
        if project_id:
            q = q.where(AuditEvent.project_id == project_id)
        return [{"ts": e.ts.isoformat(), "user": e.user, "action": e.action, "project_id": e.project_id,
                 "detail": e.detail} for e in s.scalars(q)]
