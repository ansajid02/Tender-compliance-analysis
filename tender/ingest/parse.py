"""Parse PDF / DOCX into ordered blocks (headings, paragraphs, tables) with page numbers.

Only VISIBLE text is kept: white text, near-zero font sizes, hidden Word runs, comments, metadata and
headers/footers are dropped. Hidden text is how a poisoned bid tries to talk to the judge, so it is
counted (ParsedDoc.hidden_spans) and reported, never indexed.

PDF:  PyMuPDF text dict (fonts, sizes, colours) + find_tables(); repeated top/bottom lines = header/footer.
DOCX: the WordprocessingML body directly (python-docx for styles + numbering). DOCX has no fixed pages:
      page numbers come from the page breaks Word stores when it saves (w:lastRenderedPageBreak), or
      from explicit page breaks if the file was never rendered by Word. They can be off by a page.
"""

from __future__ import annotations

import functools
import os
import re
import statistics
import unicodedata
from collections import Counter
from io import BytesIO

from ..schemas import Block, ParsedDoc
from ..text import normalize

_PAGE_NO = re.compile(r"^\s*(page|صفحة|الصفحة)?\s*\d+\s*((of|/|من)\s*\d+)?\s*$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")


def parse(kind: str, data: bytes) -> ParsedDoc:
    doc = parse_pdf(data) if kind == "pdf" else parse_docx(data)
    # NFKC: PDF ligatures ("ﬁ" -> "fi") and Arabic presentation forms ("ﻳﺠﺐ" -> "يجب") become plain letters
    nfkc = lambda t: unicodedata.normalize("NFKC", t)  # noqa: E731
    return doc.model_copy(update={"pages": [nfkc(p) for p in doc.pages],
                                  "blocks": [b.model_copy(update={"text": nfkc(b.text)}) for b in doc.blocks]})


# ================================================================ PDF

MIN_FONT = 3.0
MARGIN = 0.07  # top / bottom share of the page where headers and footers live


def _near_white(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return min(r, g, b) >= 0xF0


def _dark_areas(page) -> list:
    """Filled shapes dark enough to carry light text (heading banners, table header rows) + images."""
    areas = []
    for d in page.get_drawings():
        fill = d.get("fill")
        if fill and 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2] < 0.6:  # relative luminance
            areas.append(tuple(d["rect"]))
    for img in page.get_image_info():
        areas.append(tuple(img["bbox"]))
    return areas


def _visible(span: dict, page_rect, dark_areas: list | None = None) -> bool:
    if span.get("size", 10) < MIN_FONT:
        return False
    if span.get("alpha", 255) == 0:
        return False
    x0, y0, x1, y1 = span["bbox"]
    if not (x1 > page_rect.x0 and x0 < page_rect.x1 and y1 > page_rect.y0 and y0 < page_rect.y1):
        return False
    # White text is hidden on a white page, but real documents print white headings on dark banners:
    # it counts as visible only when it sits on a dark filled shape or an image.
    if _near_white(span.get("color", 0)):
        return _inside(span["bbox"], dark_areas or [])
    return True


def _inside(bbox, rects) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "").strip() or "eng+ara"


@functools.lru_cache(maxsize=1)
def ocr_available() -> bool:
    """OCR_ENABLED=0 turns it off; otherwise used when Tesseract is installed (it is in the Docker image)."""
    if os.environ.get("OCR_ENABLED", "1").strip() in ("0", "false", "no"):
        return False
    import pymupdf

    try:
        return bool(pymupdf.get_tessdata())
    except Exception:  # noqa: BLE001 - "Tesseract is not installed"
        return False


def parse_pdf(data: bytes) -> ParsedDoc:
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    hidden = 0
    needs_ocr: list[int] = []
    ocr_pages: list[int] = []
    # pass 1: visible lines per page (+ tables), with style info
    pages_items: list[list[tuple[float, str, dict]]] = []  # (y, kind, payload)
    margin_lines: Counter[str] = Counter()
    with doc:
        for pno, page in enumerate(doc, 1):
            rect = page.rect
            items: list[tuple[float, str, dict]] = []
            table_rects = []
            try:
                tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
            except Exception:  # noqa: BLE001 - table detection is best effort
                tables = []
            words = page.get_text("words", sort=True) if tables else []
            for tab in tables:
                # cell text from the page's words, not tab.extract(): extract() drops the spaces in PDFs that
                # position each word separately ("TheQuoteshallbe...")
                rows = [[_cell_text(c, words) for c in row.cells] for row in tab.rows]
                table_rects.append(tuple(tab.bbox))
                items.append((tab.bbox[1], "table", {"rows": [r for r in rows if any(r)]}))
            d = page.get_text("dict", sort=True)
            dark = _dark_areas(page)
            ocr_used = False
            if len(page.get_text("text").strip()) < 20 and page.get_images() and ocr_available():
                # scanned page: Tesseract (eng+ara) builds a text layer; the same pipeline then reads it
                tp = page.get_textpage_ocr(language=OCR_LANGUAGES, dpi=300, full=True)
                d = page.get_text("dict", sort=True, textpage=tp)
                ocr_used = True
                ocr_pages.append(pno)
            n_chars = 0
            for b in d["blocks"]:
                if b.get("type") != 0:
                    continue
                lines = []
                for ln in b["lines"]:
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    vis = [s for s in spans if _visible(s, rect, dark)]
                    hidden += len(spans) - len(vis)
                    if not vis or _inside(ln["bbox"], table_rects):
                        continue
                    text = "".join(s["text"] for s in vis).strip()
                    if not text:
                        continue
                    lines.append({"text": text, "size": max(s["size"] for s in vis),
                                  "bold": all(s["flags"] & 16 or "bold" in s["font"].lower() for s in vis),
                                  "bbox": ln["bbox"], "chars": len(text)})
                    n_chars += len(text)
                if lines:
                    y = lines[0]["bbox"][1]
                    in_margin = y < rect.height * MARGIN or lines[-1]["bbox"][3] > rect.height * (1 - MARGIN)
                    blk = {"lines": lines, "margin": in_margin}
                    if in_margin:
                        margin_lines[_DIGITS.sub("#", " ".join(l["text"] for l in lines).lower())] += 1
                    items.append((y, "block", blk))
            if n_chars < 20 and page.get_images() and not ocr_used:
                needs_ocr.append(pno)
            items.sort(key=lambda it: it[0])
            pages_items.append(items)

    n_pages = len(pages_items)
    repeated = {k for k, c in margin_lines.items() if c >= max(2, 0.4 * n_pages)}
    sizes = [l["size"] for items in pages_items for _, k, p in items if k == "block"
             for l in p["lines"] for _ in range(max(1, l["chars"] // 20))]
    body = statistics.median(sizes) if sizes else 10.0

    blocks: list[Block] = []
    page_texts: list[str] = []
    for pno, items in enumerate(pages_items, 1):
        texts = []
        outside = normalize(" ".join(_join_lines([l["text"] for l in p["lines"]])
                                     for _, k, p in items if k == "block"))
        for _, kind, p in items:
            if kind == "table":
                # the grid can swallow a paragraph line next to the table: drop rows that are page text
                rows = [r for r in p["rows"] if not (len(normalize(" ".join(r))) > 15
                                                     and normalize(" ".join(r)) in outside)]
                if len(rows) >= 2:
                    md = to_markdown(rows)
                    blocks.append(Block(kind="table", text=md, page=pno))
                    texts.append(md)
                continue
            lines = p["lines"]
            joined = _join_lines([l["text"] for l in lines])
            if p["margin"] and (_DIGITS.sub("#", " ".join(l["text"] for l in lines).lower()) in repeated
                                or _PAGE_NO.match(joined)):
                continue  # header / footer / page number
            texts.append(joined)
            size = max(l["size"] for l in lines)
            is_heading = len(joined) < 160 and len(lines) <= 3 and (
                size >= body * 1.15 or (all(l["bold"] for l in lines) and not joined.endswith(".")))
            if is_heading:
                level = 1 if size >= body * 1.5 else 2 if size >= body * 1.2 else 3
                blocks.append(Block(kind="heading", text=joined, page=pno, level=level))
            else:
                blocks.append(Block(kind="paragraph", text=joined, page=pno))
        page_texts.append("\n".join(texts))

    warnings = []
    if hidden:
        warnings.append(f"{hidden} invisible text span(s) dropped (white, tiny or off-page text)")
    if ocr_pages:
        warnings.append(f"pages {ocr_pages} were scanned: text read with OCR ({OCR_LANGUAGES}), check quotes there")
    if needs_ocr:
        warnings.append(f"pages {needs_ocr} look scanned (no text layer) and OCR is not available "
                        "(install Tesseract or use the Docker image): their content is missing")
    return ParsedDoc(pages=page_texts, blocks=blocks, page_count=n_pages, warnings=warnings,
                     hidden_spans=hidden, needs_ocr_pages=needs_ocr)


def _cell_text(bbox, words) -> str:
    if not bbox:
        return ""
    x0, y0, x1, y1 = bbox
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    return _cell(" ".join(w[4] for w in inside))


def _cell(c) -> str:
    return re.sub(r"\s+", " ", (c or "")).strip().replace("|", "/")


def to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]  # merged cells leave empty columns
    rows = [[r[i] for i in keep] for r in rows]
    width = len(keep)
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    return "\n".join(md + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln  # hyphenated word across lines
        else:
            out = f"{out} {ln}" if out else ln
    return out.strip()


# ================================================================ DOCX


def parse_docx(data: bytes) -> ParsedDoc:
    import docx
    from docx.oxml.ns import qn

    document = docx.Document(BytesIO(data))
    body = document.element.body
    numbering = _Numbering(document)
    styles = {s.style_id: s for s in document.styles}
    rendered = bool(body.findall(".//" + qn("w:lastRenderedPageBreak")))

    W = lambda tag: qn(f"w:{tag}")  # noqa: E731
    fallback_tag = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"

    state = {"page": 1, "hidden": 0}
    blocks: list[Block] = []
    page_texts: dict[int, list[str]] = {}

    def run_hidden(r) -> bool:
        rpr = r.find(W("rPr"))
        if rpr is None:
            return False
        if rpr.find(W("vanish")) is not None or rpr.find(W("specVanish")) is not None:
            v = (rpr.find(W("vanish")) if rpr.find(W("vanish")) is not None else rpr.find(W("specVanish")))
            if v.get(W("val")) not in ("0", "false"):
                return True
        color = rpr.find(W("color"))
        if color is not None and (color.get(W("val")) or "").upper() in ("FFFFFF", "FFFFFE") and not on_dark(r):
            return True  # white text, unless it sits on dark shading (a heading bar / header cell)
        sz = rpr.find(W("sz"))
        if sz is not None and (sz.get(W("val")) or "").isdigit() and int(sz.get(W("val"))) < 6:  # < 3 pt
            return True
        return False

    def on_dark(r) -> bool:
        """The run, its paragraph or its table cell has a dark background fill (w:shd)."""
        for el in [r, *r.iterancestors()]:
            for props in ("rPr", "pPr", "tcPr"):
                pr = el.find(W(props))
                shd = pr.find(W("shd")) if pr is not None else None
                fill = (shd.get(W("fill")) or "") if shd is not None else ""
                if len(fill) == 6 and fill.lower() != "auto":
                    rgb = [int(fill[i:i + 2], 16) / 255 for i in (0, 2, 4)]
                    if 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.6:
                        return True
        return False

    def para_text(p) -> tuple[str, int, int]:
        """-> (visible text, page breaks before the first text, page breaks after it)."""
        parts: list[str] = []
        before = after = 0
        for r in p.iter(W("r")):
            if any(a.tag in (W("del"), fallback_tag) for a in r.iterancestors()):
                continue  # deleted tracked change / duplicate fallback content
            hidden_run = run_hidden(r)
            for child in r:
                tag = child.tag
                is_break = (tag == W("lastRenderedPageBreak") and rendered) or (
                    tag == W("br") and child.get(W("type")) == "page" and not rendered)
                if is_break:
                    if "".join(parts).strip():
                        after += 1
                    else:
                        before += 1
                elif tag == W("t"):
                    if hidden_run:
                        if (child.text or "").strip():
                            state["hidden"] += 1
                    else:
                        parts.append(child.text or "")
                elif tag == W("tab") and not hidden_run:
                    parts.append("\t")
                elif tag in (W("br"), W("cr")) and not hidden_run:
                    parts.append("\n")
        return re.sub(r"[ \t]+", " ", "".join(parts)).strip(), before, after

    def emit(kind: str, text: str, level: int = 0) -> None:
        blocks.append(Block(kind=kind, text=text, page=state["page"], level=level))
        page_texts.setdefault(state["page"], []).append(text)

    for el in body.iterchildren():
        if el.tag == W("p"):
            ppr = el.find(W("pPr"))
            style_id = None
            if ppr is not None and ppr.find(W("pStyle")) is not None:
                style_id = ppr.find(W("pStyle")).get(W("val"))
            if not rendered and ppr is not None and ppr.find(W("pageBreakBefore")) is not None:
                state["page"] += 1
            text, before, after = para_text(el)
            state["page"] += before
            if text:
                prefix = numbering.label(ppr)
                if prefix:
                    text = f"{prefix} {text}"
                level = _heading_level(styles.get(style_id), ppr, W)
                if level and len(text) < 200:
                    emit("heading", text, level)
                else:
                    emit("paragraph", text)
            state["page"] += after
            if not rendered and ppr is not None and ppr.find(W("sectPr")) is not None:
                st = ppr.find(W("sectPr")).find(W("type"))
                if st is None or st.get(W("val")) in (None, "nextPage", "oddPage", "evenPage"):
                    state["page"] += 1
        elif el.tag == W("tbl"):
            rows = []
            for tr in el.iter(W("tr")):
                cells = []
                for tc in tr.findall(W("tc")):
                    cell_parts = []
                    for p in tc.iter(W("p")):
                        t, before, after = para_text(p)
                        state["page"] += before + after
                        if t:
                            cell_parts.append(t)
                    cells.append(" ".join(cell_parts).replace("|", "/"))
                if any(c for c in cells):
                    rows.append(cells)
            if rows:
                emit("table", to_markdown(rows))

    n_pages = max(page_texts) if page_texts else 1
    warnings = []
    if state["hidden"]:
        warnings.append(f"{state['hidden']} hidden text run(s) dropped (hidden, white or tiny text)")
    if not rendered:
        warnings.append("page numbers estimated from explicit page breaks (file not saved by Word)")
    return ParsedDoc(pages=["\n".join(page_texts.get(i, [])) for i in range(1, n_pages + 1)],
                     blocks=blocks, page_count=n_pages, warnings=warnings, hidden_spans=state["hidden"])


def _heading_level(style, ppr, W) -> int:
    if ppr is not None and ppr.find(W("outlineLvl")) is not None:
        val = ppr.find(W("outlineLvl")).get(W("val"))
        if val and val.isdigit() and int(val) < 9:
            return int(val) + 1
    while style is not None:
        name = (style.name or "").lower()
        if name == "title":
            return 1
        m = re.match(r"heading\s*(\d)", name)
        if m:
            return int(m.group(1))
        style = style.base_style
    return 0


class _Numbering:
    """Best-effort Word auto-numbering ("3.2.1", "(a)"): clause numbers live in numbering.xml, not the text."""

    def __init__(self, document) -> None:
        from docx.oxml.ns import qn

        self.qn = qn
        self.formats: dict[tuple[str, int], tuple[str, str, int]] = {}  # (numId, lvl) -> (fmt, lvlText, start)
        self.counters: dict[str, list[int]] = {}
        try:
            root = document.part.numbering_part.element
        except (KeyError, NotImplementedError, AttributeError):
            return
        abstract = {}
        for an in root.findall(qn("w:abstractNum")):
            levels = {}
            for lvl in an.findall(qn("w:lvl")):
                ilvl = int(lvl.get(qn("w:ilvl")))
                fmt = lvl.find(qn("w:numFmt"))
                txt = lvl.find(qn("w:lvlText"))
                start = lvl.find(qn("w:start"))
                levels[ilvl] = (fmt.get(qn("w:val")) if fmt is not None else "decimal",
                                txt.get(qn("w:val")) if txt is not None else "",
                                int(start.get(qn("w:val"))) if start is not None else 1)
            abstract[an.get(qn("w:abstractNumId"))] = levels
        for num in root.findall(qn("w:num")):
            aid = num.find(qn("w:abstractNumId"))
            if aid is None:
                continue
            for ilvl, spec in abstract.get(aid.get(qn("w:val")), {}).items():
                self.formats[(num.get(qn("w:numId")), ilvl)] = spec

    def label(self, ppr) -> str:
        qn = self.qn
        if ppr is None or ppr.find(qn("w:numPr")) is None:
            return ""
        npr = ppr.find(qn("w:numPr"))
        num_el, lvl_el = npr.find(qn("w:numId")), npr.find(qn("w:ilvl"))
        if num_el is None:
            return ""
        num_id = num_el.get(qn("w:val"))
        ilvl = int(lvl_el.get(qn("w:val"))) if lvl_el is not None else 0
        spec = self.formats.get((num_id, ilvl))
        if spec is None or num_id == "0":
            return ""
        fmt, text, start = spec
        if fmt == "bullet":
            return "•"
        counters = self.counters.setdefault(num_id, [])
        while len(counters) <= ilvl:
            counters.append(self.formats.get((num_id, len(counters)), ("decimal", "", 1))[2] - 1)
        counters[ilvl] += 1
        del counters[ilvl + 1:]
        out = text
        for i, c in enumerate(counters):
            f = self.formats.get((num_id, i), ("decimal", "", 1))[0]
            out = out.replace(f"%{i + 1}", _fmt(c, f))
        return out.strip()


def _fmt(n: int, fmt: str) -> str:
    if fmt == "lowerLetter":
        return chr(ord("a") + (n - 1) % 26)
    if fmt == "upperLetter":
        return chr(ord("A") + (n - 1) % 26)
    if fmt in ("lowerRoman", "upperRoman"):
        vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
        out, k = "", n
        for v, s in vals:
            while k >= v:
                out, k = out + s, k - v
        return out if fmt == "lowerRoman" else out.upper()
    return str(n)
