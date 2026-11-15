"""Upload validation: a bid is adversarial input, so reject what we should never parse.

Rejected: wrong type / magic bytes, too large, too many pages, encrypted files, PDFs with JavaScript or
launch actions, DOCX with macros or ActiveX, zip bombs (huge or extreme compression ratio).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .. import config

ALLOWED = {".pdf", ".docx"}
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .doc / encrypted Office files
_PDF_DANGER = re.compile(r"/(JavaScript|JS|Launch|SubmitForm|ImportData)\b")


class ValidationError(ValueError):
    """The file is rejected. The message is safe to show to the user."""


def validate(filename: str, data: bytes) -> tuple[str, list[str]]:
    """-> (kind 'pdf' | 'docx', warnings). Raises ValidationError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValidationError(f"{filename}: only PDF and DOCX files are accepted (got '{ext or 'no extension'}')")
    size_mb = len(data) / 1e6
    if size_mb > config.MAX_UPLOAD_MB:
        raise ValidationError(f"{filename}: {size_mb:.1f} MB is over the {config.MAX_UPLOAD_MB} MB limit")
    if not data:
        raise ValidationError(f"{filename}: empty file")
    if ext == ".pdf":
        return "pdf", _validate_pdf(filename, data)
    return "docx", _validate_docx(filename, data)


def _validate_pdf(filename: str, data: bytes) -> list[str]:
    import pymupdf

    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise ValidationError(f"{filename}: not a PDF file (bad header)")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001 - any parser failure = reject
        raise ValidationError(f"{filename}: the PDF cannot be opened ({type(e).__name__})") from e
    warnings: list[str] = []
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise ValidationError(f"{filename}: encrypted / password-protected PDFs are not accepted")
        if doc.page_count == 0:
            raise ValidationError(f"{filename}: the PDF has no pages")
        if doc.page_count > config.MAX_PAGES:
            raise ValidationError(f"{filename}: {doc.page_count} pages is over the {config.MAX_PAGES}-page limit")
        for xref in range(1, doc.xref_length()):
            try:
                obj = doc.xref_object(xref, compressed=True)
            except Exception:  # noqa: BLE001 - broken object: skip, the page parser decides
                continue
            m = _PDF_DANGER.search(obj)
            if m:
                raise ValidationError(f"{filename}: PDFs with active content (/{m.group(1)}) are not accepted")
        if doc.embfile_count():
            warnings.append(f"{doc.embfile_count()} embedded file(s) ignored")
    return warnings


def _validate_docx(filename: str, data: bytes) -> list[str]:
    if data.startswith(_OLE_MAGIC):
        raise ValidationError(f"{filename}: encrypted DOCX or legacy .doc files are not accepted "
                              "(save as a normal .docx)")
    if not data.startswith(b"PK"):
        raise ValidationError(f"{filename}: not a DOCX file (bad header)")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValidationError(f"{filename}: corrupt DOCX (bad zip)") from e
    with zf:
        infos = zf.infolist()
        names = {i.filename for i in infos}
        if "word/document.xml" not in names:
            raise ValidationError(f"{filename}: not a Word document (word/document.xml missing)")
        if len(infos) > 5000:
            raise ValidationError(f"{filename}: too many parts in the DOCX ({len(infos)})")
        total = sum(i.file_size for i in infos)
        packed = max(1, sum(i.compress_size for i in infos))
        if total > config.MAX_ZIP_UNCOMPRESSED_MB * 1e6 or total / packed > config.MAX_ZIP_RATIO:
            raise ValidationError(f"{filename}: rejected as a possible zip bomb "
                                  f"({total / 1e6:.0f} MB uncompressed, ratio {total / packed:.0f}x)")
        if any(n.lower().endswith("vbaproject.bin") for n in names) or "macroEnabled" in _read(zf, "[Content_Types].xml"):
            raise ValidationError(f"{filename}: documents with macros are not accepted")
        if any(n.lower().startswith("word/activex/") for n in names):
            raise ValidationError(f"{filename}: documents with ActiveX controls are not accepted")
        warnings = []
        if any(n.startswith("word/embeddings/") for n in names):
            warnings.append("embedded objects ignored")
        if "word/comments.xml" in names:
            warnings.append("comments ignored (never treated as bid content)")
    return warnings


def _read(zf: zipfile.ZipFile, name: str) -> str:
    try:
        return zf.read(name)[:200_000].decode("utf-8", "ignore")
    except KeyError:
        return ""
