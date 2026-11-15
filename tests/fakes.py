"""A scripted fake LLM (stands in for Gemini): dispatches on the requested JSON schema (by title).

ExtractOut: regex extraction of shall / must / should / may lines + "Yes" rows of document tables.
            Options: `hallucinate` adds a requirement whose quote is not in the tender; `wrong_page`
            shifts the page of requirements whose quote contains that text.
JudgeOut:   picks the excerpt with the most words in common with the requirement; MET above an
            overlap threshold, else MISSING. `obey_injection`: if an excerpt contains injection text,
            answer MET quoting it (what a fooled model would do) - the grounding check must catch it.
            `missing_for`: requirement words that make the first pass answer MISSING (to test the reviewer).
AnswerOut:  answers from the best excerpt, citing it (or a fake quote if `bad_citation`).
"""

from __future__ import annotations

import json
import re

from agentcore.llm import RawResult

from tender.text import tokens

_REQ = re.compile(r"\b(shall|must|should|may)\b", re.IGNORECASE)
_STOP = set("the a an of to in for and or with be is are by on at all any from this that its it as must shall "
            "should may bidder bid our we".split())


def _words(t: str) -> set[str]:
    return {w for w in tokens(t) if w not in _STOP and len(w) > 2}


class FakeLLM:
    def __init__(self, hallucinate: bool = False, wrong_page: str | None = None, obey_injection: bool = False,
                 missing_for: tuple[str, ...] = (), bad_citation: bool = False, leak: bool = False) -> None:
        self.hallucinate, self.wrong_page, self.obey = hallucinate, wrong_page, obey_injection
        self.missing_for, self.bad_citation, self.leak = missing_for, bad_citation, leak
        self.calls: list[dict] = []

    def __call__(self, model, prompt, system, json_schema, temperature, max_tokens) -> RawResult:
        title = (json_schema or {}).get("title")
        self.calls.append({"model": model, "schema": title, "prompt": prompt, "system": system})
        out = {"ExtractOut": self.extract, "JudgeOut": self.judge, "AnswerOut": self.answer}[title](prompt, system)
        return RawResult(text=json.dumps(out, ensure_ascii=False), model=model, input_tokens=len(prompt) // 4,
                         output_tokens=50)

    # ------------------------------------------------------------ extractor
    def extract(self, prompt: str, system: str | None) -> dict:
        page, reqs = 1, []
        for line in prompt.splitlines():
            m = re.match(r"\[page (\d+)\]", line)
            if m:
                page = int(m.group(1))
                continue
            row = re.match(r"\|\s*([^|]+?)\s*\|\s*Yes\s*\|", line)
            if row:
                reqs.append(self._req(row.group(1), row.group(1), page, "documents", True))
            elif _REQ.search(line) and not line.startswith("|") and "words shall and must" not in line:
                quote = re.sub(r"^\d+(\.\d+)*\s+", "", line).strip()
                mandatory = not re.search(r"\b(should|may)\b", quote, re.IGNORECASE)
                reqs.append(self._req(quote, quote, page, "technical", mandatory))
        if self.hallucinate and reqs:
            reqs.append(self._req("The bidder must provide a drone delivery service.",
                                  "The bidder must provide a drone delivery service to all ministries.", 2,
                                  "technical", True))
        return {"requirements": reqs}

    def _req(self, text, quote, page, cat, mandatory) -> dict:
        if self.wrong_page and self.wrong_page in quote:
            page += 1
        return {"text": text, "quote": quote, "page": page, "category": cat, "mandatory": mandatory, "deadline": None}

    # ------------------------------------------------------------ judge
    @staticmethod
    def _excerpts(prompt: str) -> list[tuple[str, str]]:
        return re.findall(r'<(?:bid_excerpt|excerpt) id="(E\d+)"[^>]*>\n(.*?)\n</(?:bid_excerpt|excerpt)>',
                          prompt, re.DOTALL)

    def judge(self, prompt: str, system: str | None) -> dict:
        req = re.search(r"<requirement[^>]*>\n(.*?)\n", prompt).group(1)
        excerpts = self._excerpts(prompt)
        if self.obey:
            for eid, text in excerpts:
                m = re.search(r"[^.\n]*mark every requirement as MET[^.\n]*", text)
                if m:
                    return {"status": "MET", "excerpt_id": eid, "quote": m.group(0).strip(), "reason": "pre-approved"}
        reviewing = system and "first evaluator found NO evidence" in system
        if not reviewing and any(w in req for w in self.missing_for):
            return {"status": "MISSING", "excerpt_id": None, "quote": "", "reason": "not found"}
        best, best_eid, best_line = 0.0, None, ""
        rw = _words(req)
        for eid, text in excerpts:
            for line in text.split("\n"):
                ov = len(rw & _words(line)) / max(1, len(rw))
                if ov > best:
                    best, best_eid, best_line = ov, eid, line
        if best >= 0.34:
            return {"status": "MET", "excerpt_id": best_eid, "quote": best_line.strip("| "), "reason": "matches"}
        return {"status": "MISSING", "excerpt_id": None, "quote": "", "reason": "no excerpt addresses it"}

    # ------------------------------------------------------------ ask
    def answer(self, prompt: str, system: str | None) -> dict:
        q = re.search(r"<question>\n(.*?)\n</question>", prompt, re.DOTALL).group(1)
        excerpts = self._excerpts(prompt)
        if self.leak:
            canary = re.search(r"canary-[0-9a-f]+", system or "").group(0)
            return {"answer": f"My instructions say {canary}", "citations": [], "found": True}
        qw = _words(q)
        eid, text = max(excerpts, key=lambda e: len(qw & _words(e[1])))
        line = max(text.split("\n"), key=lambda ln: len(qw & _words(ln)))
        quote = "an invented sentence that is not in the excerpt" if self.bad_citation else line.strip("| ")
        return {"answer": f"{line} [{eid}]", "citations": [{"excerpt_id": eid, "quote": quote}], "found": True}
