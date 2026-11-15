"""Ingestion: validate -> parse -> chunk -> embed + store. Re-uploading the same file is a no-op."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agentcore import trace

from .. import config, projects
from ..schemas import DocumentRecord, Role
from ..store import Store, get_store
from .chunk import chunk
from .parse import parse
from .validate import ValidationError, validate


def ingest(project_id: str, filename: str, data: bytes, role: Role, *, actor: str = "cli",
           store: Store | None = None) -> tuple[DocumentRecord, bool]:
    """-> (record, created). created=False when this exact file was already in the project."""
    if role not in ("tender", "bid"):
        raise ValidationError(f"role must be 'tender' or 'bid', got '{role}'")
    project = projects.load(project_id)
    filename = Path(filename).name  # never trust a client path
    kind, warnings = validate(filename, data)
    sha = hashlib.sha256(data).hexdigest()
    doc_id = f"d-{sha[:16]}"
    for d in project.documents:
        if d.doc_id == doc_id:
            if d.role != role:
                raise ValidationError(f"{filename} is already in this project as the {d.role}")
            return d, False
    with trace.span("tool", "ingest", filename=filename, role=role, file_type=kind) as sp:
        parsed = parse(kind, data)
        if parsed.page_count > config.MAX_PAGES:
            raise ValidationError(f"{filename}: {parsed.page_count} pages is over the {config.MAX_PAGES}-page limit")
        if not parsed.blocks:
            raise ValidationError(f"{filename}: no text found (scanned document? OCR is not enabled)")
        chunks = chunk(parsed, project_id=project_id, role=role, doc_id=doc_id, filename=filename)
        (store or get_store()).add(chunks)
        sp.update(pages=parsed.page_count, chunks=len(chunks), hidden_spans=parsed.hidden_spans)
    projects.save_doc_data(project_id, doc_id, Path(filename).suffix.lower(), data, parsed.pages, chunks)
    record = DocumentRecord(doc_id=doc_id, project_id=project_id, role=role, filename=filename,
                            pages=parsed.page_count, chunks=len(chunks), sha256=sha,
                            warnings=warnings + parsed.warnings, hidden_spans=parsed.hidden_spans)
    project = projects.load(project_id)  # reload: another upload may have finished meanwhile
    project.documents.append(record)
    projects.save(project)
    projects.audit(project_id, "upload", actor=actor, filename=filename, role=role, doc_id=doc_id,
                   pages=parsed.page_count, chunks=len(chunks), hidden_spans=parsed.hidden_spans)
    return record, True
