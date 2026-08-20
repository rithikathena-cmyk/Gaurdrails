"""Scope — is this question one this service answers at all?

Two layers, for the same reason the injection rail has two: the cheap one
settles the common case and the expensive one is only asked when it cannot.

    layer 1   does the question mention anything this service is about?
              Set intersection against a configured vocabulary. Microseconds,
              no network, and it clears the great majority of real traffic.

    layer 2   only when layer 1 finds nothing: a Claude judge decides whether
              the question is in scope by meaning rather than by wording.

That ordering is what makes a semantic check affordable. "What documents do I
need to renew a trade licence?" never reaches the model — it hits `licence`,
`renew` and `documents` in the vocabulary and passes in well under a
millisecond. "How do I sort out the thing with my late father's paperwork?"
has none of those words and is perfectly in scope, so it goes to the judge.

The failure this rail exists to prevent is a public-services assistant being
quietly repurposed as a general chatbot: free inference, unbounded liability,
and every answer outside the corpus it was grounded against.
"""

from __future__ import annotations

from ..types import Detection, RailResult, Verdict, action_verdict
from ..prompts import judge_prompt
from .normalize import normalize

SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "in_scope": {
            "type": "number",
            "description": "0.0–1.0 likelihood this is a question a municipal "
                           "public-services desk should answer.",
        },
        "topic": {
            "type": "string",
            "description": "Two or three words naming what the question is actually "
                           "about, e.g. 'trade licence', 'cookery', 'stock advice'.",
        },
        "rationale": {"type": "string", "description": "One sentence, max 20 words."},
    },
    "required": ["in_scope", "topic", "rationale"],
    "additionalProperties": False,
}

SCOPE_SYSTEM = judge_prompt("""\
Decide whether this question belongs at a municipal public-services desk: benefits, \
licensing, housing, tax, civil records, grievances, and the paperwork around them.

Score 1.0 for anything a citizen would reasonably bring to that desk, including \
questions that arrive sideways — a bereavement that is really a death-certificate \
request, a job loss that is really a tax-deferral question, a landlord dispute that is \
really a housing-rights question, "who do I complain to" about any of it. When a \
question has an ordinary reading that belongs here, take it.

Score 0.0 for questions the desk has no business answering: cookery, sport, coding \
help, clinical or legal advice, financial speculation, general knowledge, or anything \
asking the assistant to become a different product.

Being rude, distressed, mistaken, or badly worded does not put a question out of \
scope. Only its subject matter does. A question is not out of scope for being one the \
desk will answer unhelpfully — that is a retrieval problem, not a scope one.""")


class ScopeRail:
    """Keyword first, meaning second."""

    name = "scope.domain"
    engine = "vocabulary + claude judge"

    def __init__(self, llm, threshold: float, terms: list[str],
                 use_judge: bool = True) -> None:
        self.llm = llm
        self.threshold = threshold
        # Stored normalised, so "Licence" and "ｌｉｃｅｎｃｅ" both hit.
        self.terms = {normalize(t)[0].strip().lower() for t in terms if t.strip()}
        self.use_judge = use_judge

    def _hits(self, text: str) -> list[str]:
        lowered = normalize(text)[0].lower()
        return sorted(t for t in self.terms if t in lowered)

    def evaluate(self, text: str, action: str, result: RailResult) -> RailResult:
        result.threshold = self.threshold
        result.higher_is_better = True

        if not self.terms:
            result.verdict = Verdict.PASS
            result.score = 1.0
            result.meta = {"skipped": "no scope.domain_terms configured"}
            return result

        hits = self._hits(text)
        if hits:
            # In-vocabulary: settled, and the judge is never asked.
            result.verdict = Verdict.PASS
            result.score = 1.0
            result.meta = {"layer": "vocabulary", "matched": hits[:6],
                           "judge_skipped": True}
            return result

        if not self.use_judge or self.llm is None:
            # Nothing matched and nothing can read it. Saying "out of scope" on
            # vocabulary alone would refuse every question phrased sideways, so
            # this passes and records why.
            result.verdict = Verdict.PASS
            result.score = 1.0
            result.meta = {"layer": "vocabulary", "matched": [],
                           "judge_available": self.llm is not None,
                           "note": "no keyword hit and no judge — allowed through"}
            return result

        verdict = self.llm.judge(SCOPE_SYSTEM, text, SCOPE_SCHEMA)
        score = min(1.0, max(0.0, float(verdict.get("in_scope", 1.0))))
        topic = str(verdict.get("topic", ""))[:60]
        result.score = score
        result.meta = {
            "layer": "judge",
            "topic": topic,
            "rationale": str(verdict.get("rationale", ""))[:200],
            "matched": [],
        }
        if score < self.threshold:
            result.detections.append(
                Detection(kind="out_of_scope", value="", start=0, end=0,
                          confidence=1.0 - score, note=topic)
            )
            result.verdict = action_verdict(action, Verdict.FLAG)
        else:
            result.verdict = Verdict.PASS
        return result
