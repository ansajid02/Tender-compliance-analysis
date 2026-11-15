"""Records shared across ingestion, the check workflow, reports and the API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["tender", "bid"]
Status = Literal["MET", "PARTIAL", "MISSING", "UNCLEAR"]
STATUSES: tuple[Status, ...] = ("MET", "PARTIAL", "MISSING", "UNCLEAR")
Category = Literal["technical", "commercial", "legal", "documents", "deadline", "local_content",
                   "qualification", "other"]
CATEGORIES: tuple[str, ...] = ("technical", "commercial", "legal", "documents", "deadline", "local_content",
                               "qualification", "other")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- ingestion


class Block(BaseModel):
    """One structural unit of a parsed document, in reading order."""
    kind: Literal["heading", "paragraph", "table"]
    text: str  # tables: markdown
    page: int  # 1-based
    level: int = 0  # headings: 1 = top


class ParsedDoc(BaseModel):
    pages: list[str]  # plain text per page (index 0 = page 1), used to verify quotes
    blocks: list[Block]
    page_count: int
    warnings: list[str] = Field(default_factory=list)
    hidden_spans: int = 0  # invisible text dropped (white / tiny / hidden runs): a poisoning signal
    needs_ocr_pages: list[int] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    doc_id: str  # sha256 of the file bytes (first 16 hex): re-upload of the same file = no duplicates
    project_id: str
    role: Role
    filename: str
    pages: int
    chunks: int
    sha256: str
    uploaded_at: str = Field(default_factory=now)
    warnings: list[str] = Field(default_factory=list)
    hidden_spans: int = 0


class Project(BaseModel):
    project_id: str
    name: str
    created_at: str = Field(default_factory=now)
    documents: list[DocumentRecord] = Field(default_factory=list)

    def docs(self, role: Role | None = None) -> list[DocumentRecord]:
        return [d for d in self.documents if role is None or d.role == role]


# ---------------------------------------------------------------- check


class Requirement(BaseModel):
    req_id: str  # R001, ...
    text: str = Field(description="the requirement as one self-contained statement")
    quote: str = Field(description="verbatim words from the tender that state it")
    doc_id: str
    page: int
    section: str = ""
    clause: str = ""
    category: Category = "other"
    mandatory: bool = True
    deadline: str | None = None
    quote_verified: bool = False
    page_corrected: bool = False  # the extractor cited the wrong page; the quote was found on another one


class Evidence(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    quote: str


class Verdict(BaseModel):
    req_id: str
    status: Status
    reason: str
    evidence: Evidence | None = None  # required for MET / PARTIAL after the grounding check
    grounded: bool = False
    downgraded_from: Status | None = None  # set when the grounding check rejected the judge's quote
    reviewed: bool = False  # mandatory MISSING verdicts get a second, wider search
    review_changed: bool = False
    candidates: list[str] = Field(default_factory=list)  # chunk ids the judge saw
    flagged_chunks: list[str] = Field(default_factory=list)  # chunks with injection-like text


class MatrixRow(BaseModel):
    req_id: str
    category: str
    mandatory: bool
    requirement: str
    tender_quote: str
    tender_page: int
    tender_section: str
    status: Status
    reason: str
    bid_quote: str = ""
    bid_file: str = ""
    bid_page: int | None = None
    reviewed: bool = False
    downgraded: bool = False


class CheckResult(BaseModel):
    check_id: str
    project_id: str
    created_at: str = Field(default_factory=now)
    requirements: list[Requirement]
    verdicts: list[Verdict]
    stats: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
