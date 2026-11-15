"""Structure-aware chunking: parsed blocks -> LangChain Documents with citation metadata.

- a chunk never crosses a section heading or a page break (page citations are exact); each chunk starts
  with its section title (better retrieval)
- paragraphs are packed up to CHUNK_CHARS; an oversized paragraph is split with LangChain's
  RecursiveCharacterTextSplitter
- tables are split by rows, the header row repeated in every piece
- metadata: project, role, doc, file, page (+ page_end), section path, clause numbers, content hash
- chunk ids are deterministic (uuid5 of project + doc + position + content), so re-ingesting a file
  overwrites its points instead of duplicating them
"""

from __future__ import annotations

import hashlib
import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..schemas import Block, ParsedDoc
from ..text import is_arabic

_NS = uuid.UUID("6f1c2b0e-5a7d-4c1e-9a43-1d2f3e4a5b6c")
_CLAUSE = re.compile(
    r"^\s*(?:(?:clause|article|section|item)\s+)?\(?(\d{1,2}(?:\.\d{1,3}){0,4})[.)]?\s+\S"
    r"|^\s*(?:المادة|البند|الفقرة)\s+\(?([\d٠-٩.]+|\S+)", re.IGNORECASE)


def clause_of(text: str) -> str:
    m = _CLAUSE.match(text)
    return (m.group(1) or m.group(2) or "") if m else ""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk(parsed: ParsedDoc, *, project_id: str, role: str, doc_id: str, filename: str,
          max_chars: int = config.CHUNK_CHARS) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=config.CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ". ", "، ", "; ", " ", ""])
    out: list[Document] = []
    headings: dict[int, str] = {}
    buf: list[Block] = []

    def section() -> str:
        return " > ".join(headings[k] for k in sorted(headings))

    def title() -> str:
        return headings[max(headings)] if headings else ""

    def add(text: str, first: int, last: int, kind: str, clauses: list[str]) -> None:
        head = title()
        body = f"{head}\n{text}" if head and not text.startswith(head) else text
        h = content_hash(body)
        idx = len(out)
        out.append(Document(page_content=body, metadata={
            "chunk_id": str(uuid.uuid5(_NS, f"{project_id}:{doc_id}:{idx}:{h}")),
            "project_id": project_id, "role": role, "doc_id": doc_id, "filename": filename,
            "page": first, "page_end": last, "section": section(), "kind": kind,
            "clause": clauses[0] if clauses else "", "clauses": ", ".join(dict.fromkeys(c for c in clauses if c)),
            "lang": "ar" if is_arabic(body) else "en", "chunk_index": idx, "content_hash": h,
        }))

    def flush() -> None:
        if buf:
            add("\n".join(b.text for b in buf), buf[0].page, buf[-1].page, "text", [clause_of(b.text) for b in buf])
            buf.clear()

    for b in parsed.blocks:
        if b.kind == "heading":
            flush()
            level = b.level or 1
            for k in [k for k in headings if k >= level]:
                del headings[k]
            headings[level] = b.text
            continue
        if b.kind == "table":
            flush()
            lines = b.text.split("\n")
            header, rows = lines[:2], lines[2:]
            piece: list[str] = []
            for row in rows:
                if piece and len("\n".join(header + piece + [row])) > max_chars:
                    add("\n".join(header + piece), b.page, b.page, "table", [])
                    piece = []
                piece.append(row)
            add("\n".join(header + piece), b.page, b.page, "table", [])
            continue
        if len(b.text) > max_chars:
            flush()
            for part in splitter.split_text(b.text):
                add(part, b.page, b.page, "text", [clause_of(part)])
            continue
        if buf and (b.page != buf[-1].page or sum(len(x.text) + 1 for x in buf) + len(b.text) > max_chars):
            flush()  # a chunk never crosses a page: every citation's page number is exact
        buf.append(b)
    flush()
    return out
