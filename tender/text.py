"""Text utilities: normalisation (English + Arabic) and fuzzy quote matching."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# Arabic diacritics (tashkeel), superscript alef, and tatweel
_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

QUOTE_THRESHOLD = 90  # rapidfuzz partial_ratio (0-100) for "this quote is in that text"
MIN_QUOTE_CHARS = 12


def normalize(text: str) -> str:
    """For matching/search: NFKC, lowercase, Arabic letter normalisation, no punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _AR_DIACRITICS.sub("", text).translate(_AR_MAP)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) / len(letters) > 0.5


def quote_score(quote: str, text: str) -> float:
    """0-100: how well `quote` appears inside `text` (exact substring after normalisation = 100).

    Long quotes are allowed small differences (PDF hyphenation, ligatures, an LLM fixing a typo);
    short quotes must match exactly, or fuzzy matching would accept almost anything.
    """
    q, t = normalize(quote), normalize(text)
    if len(q) < MIN_QUOTE_CHARS or not t:
        return 0.0
    if q in t:
        return 100.0
    if len(q) < 30:
        return 0.0
    if q.replace(" ", "") in t.replace(" ", ""):
        return 99.0  # same letters, broken spacing (PDFs that position every word separately)
    return float(fuzz.partial_ratio(q, t))


def quote_in(quote: str, text: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    return quote_score(quote, text) >= threshold


def similar(a: str, b: str) -> float:
    """0-100 similarity of two statements, word order ignored (for merging near-duplicates).
    token_sort, not token_set: token_set scores "VAT registration certificate" vs "Commercial Registration
    (CR) certificate" as near-identical because one word set is almost a subset of the other."""
    return float(fuzz.token_sort_ratio(normalize(a), normalize(b)))
