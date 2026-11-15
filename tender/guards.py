"""Guardrails (code first, no LLM): prompt-injection patterns, prompt leaks, sensitive numbers.

A bid is adversarial input. Its chunks are wrapped in delimiters and treated as data, but we also:
- flag chunks whose text talks to "the evaluator / AI / system" (reported in the matrix warnings)
- refuse any MET / PARTIAL whose evidence quote is itself injection text
- check questions (ASK) for injection and answers for system-prompt leaks (canary) and secrets
"""

from __future__ import annotations

import re
import secrets

_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions|prompts?|rules)",
    r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|requirements)",
    r"\b(mark|rate|score|treat|classify|output|return)\s+(every|all|each|any)\s+(of\s+the\s+)?requirements?\b.{0,40}"
    r"\b(met|compliant|satisfied|fulfilled|pass)",
    r"\b(output|return|answer|respond with)\s+\"?met\"?\s+(for|to)\s+(every|all|each)",
    r"\b(note|message|instruction)s?\s+(to|for)\s+(the\s+)?(ai|llm|model|assistant|evaluat\w*|review\w*)"
    r"(\s+(system|model|tool))?\b",
    r"\b(you are now|act as|pretend to be|new instructions|system prompt|developer mode)\b",
    r"\bpre-?approved\s+by\b",
    r"^\s*(system|assistant)\s*:",
    r"(تجاهل|تجاهلي)\s+(جميع\s+|كل\s+)?(التعليمات|الأوامر)",
    r"(اعتبر|اعتبار)\s+(جميع|كل)\s+المتطلبات\s+(مستوفاة|مستوفية)",
]
INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.MULTILINE)

_SECRETS = [
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[aws-key]"),
    (re.compile(r"\b(?:ABSK|bedrock-api-key-)[A-Za-z0-9+/=_-]{20,}"), "[api-key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[api-key]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card-or-account-number]"),
    (re.compile(r"\bSA\d{2}[0-9A-Z ]{18,30}\b"), "[iban]"),
    (re.compile(r"\bAE\d{2}[0-9 ]{19,25}\b"), "[iban]"),
]


def injection_hits(text: str) -> list[str]:
    return [m.group(0).strip() for m in INJECTION.finditer(text or "")]


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text or ""))


def new_canary() -> str:
    """A random marker put in system prompts: if it shows up in an answer, the prompt leaked."""
    return f"canary-{secrets.token_hex(6)}"


def check_question(question: str, max_chars: int = 2000) -> str | None:
    """-> reason the question is refused, or None if it is fine."""
    if not question.strip():
        return "empty question"
    if len(question) > max_chars:
        return f"question longer than {max_chars} characters"
    if looks_like_injection(question):
        return "the question looks like a prompt-injection attempt"
    return None


def clean_answer(answer: str, canary: str) -> tuple[str, list[str]]:
    """-> (answer safe to show, issues found). A leaked system prompt replaces the whole answer."""
    issues = []
    if canary and canary in answer:
        return "I can't answer that.", ["system prompt leak blocked"]
    for pattern, label in _SECRETS:
        if pattern.search(answer):
            answer = pattern.sub(label, answer)
            issues.append(f"redacted {label}")
    return answer, issues
