"""Citation viewer: show a cited page with the quote highlighted.

PDF:  the page rendered as PNG by PyMuPDF, with highlight annotations on the quote (no PDF.js in the browser)
DOCX: no fixed pages exist, so the page's text (as indexed) with the quote's position marked
Quotes are verified against normalised text, so the exact string may not be on the page (line breaks,
hyphenation): we fall back to the quote's first and last words.
"""

from __future__ import annotations

from pathlib import Path

from . import projects
from .text import normalize

DPI = 110


def _search(page, quote: str) -> list:
    hits = page.search_for(quote, quads=False)
    if hits:
        return hits
    words = quote.split()
    for part in (" ".join(words[:7]), " ".join(words[-7:]), " ".join(words[:4])):
        if len(part) >= 8:
            hits = page.search_for(part)
            if hits:
                return hits
    return []


def pdf_page_png(path: Path, page_no: int, quote: str = "") -> tuple[bytes, int]:
    """-> (PNG bytes, number of highlighted areas)."""
    import pymupdf

    with pymupdf.open(path) as doc:
        if not 1 <= page_no <= doc.page_count:
            raise IndexError(f"page {page_no} out of range (1-{doc.page_count})")
        page = doc[page_no - 1]
        hits = _search(page, quote) if quote.strip() else []
        for r in hits:
            annot = page.add_highlight_annot(r)
            annot.set_colors(stroke=(1.0, 0.85, 0.1))
            annot.update()
        return page.get_pixmap(dpi=DPI, annots=True).tobytes("png"), len(hits)


def page_text(project_id: str, doc_id: str, page_no: int, quote: str = "") -> dict:
    """Page text + the character span of the quote in it (or null)."""
    pages = projects.doc_pages(project_id, doc_id)
    if not 1 <= page_no <= len(pages):
        raise IndexError(f"page {page_no} out of range (1-{len(pages)})")
    text = pages[page_no - 1]
    span = None
    if quote.strip():
        i = text.find(quote)
        if i < 0:  # tolerate whitespace / case differences: search on a normalised copy, map back roughly
            low = text.lower()
            for part in (quote.lower(), " ".join(quote.lower().split()[:6])):
                i = low.find(part)
                if i >= 0:
                    span = [i, i + len(part)]
                    break
        else:
            span = [i, i + len(quote)]
    return {"page": page_no, "pages": len(pages), "text": text, "highlight": span,
            "found": span is not None or bool(quote and normalize(quote) in normalize(text))}
