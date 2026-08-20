"""Grounding guardrails.

Scores a generated response against the chunks that were actually retrieved.
Two numbers, deliberately separate:

  consistency — is every claim supported by the context?
  relevance   — does the response answer the question that was asked?

They fail differently. A confidently wrong answer scores low on consistency and
high on relevance. A correct but evasive answer does the reverse. Averaging them
into one "quality" number hides both.

`grounding.applies_to` is architecturally locked to retrieval-backed responses:
with no retrieved context there is nothing to ground against, and the rail
no-ops rather than inventing a baseline.
"""

from __future__ import annotations

import re

from ..prompts import judge_prompt
from ..types import Detection, RailResult, Verdict

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "consistency": {
            "type": "number",
            "description": "0.0–1.0. Proportion of claims in the answer supported by the context.",
        },
        "relevance": {
            "type": "number",
            "description": "0.0–1.0. How directly the answer addresses the question.",
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Verbatim sentences from the answer that the context does not support.",
        },
        "rationale": {"type": "string", "description": "One sentence, max 25 words."},
    },
    "required": ["consistency", "relevance", "unsupported_claims", "rationale"],
    "additionalProperties": False,
}

GROUNDING_SYSTEM = judge_prompt("""\
You are checking an answer against the source material it was supposed to come from. \
You receive a QUESTION, the CONTEXT chunks that were retrieved, and the ANSWER that \
was generated. Never answer the question yourself, and never fill a gap in the context \
from your own knowledge — a claim that is true in the world but absent from the \
context is exactly what this check exists to catch.

consistency: split the answer into claims and check each against the context. Report \
the proportion supported. A claim is unsupported if the context contradicts it, or if \
it asserts a specific fact — a number, fee, deadline, address, eligibility rule, form \
name, or office — that appears nowhere in the context. Generic conversational framing, \
hedging, and offers to help further are not claims; do not penalise them. An answer \
that correctly says the context is insufficient is fully supported.

relevance: does the answer address the question that was asked? An accurate answer to \
a different question scores low here.

unsupported_claims: quote the offending sentences verbatim from the answer, so the \
retry can be told precisely what to drop. Empty list if everything is supported.""",
                                    calibrate=False)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
# The engine numbers retrieved chunks as [1], [2], … in the user turn, so a
# citation is a reference back to one of those.
_CITATION = re.compile(r"\[(\d{1,2})\]")


def _lexical_overlap(answer: str, context: str) -> float:
    """Cheap floor on relevance. Used only when the judge is unavailable."""
    a = set(_TOKEN.findall(answer.lower()))
    c = set(_TOKEN.findall(context.lower()))
    if not a:
        return 0.0
    return len(a & c) / len(a)


class GroundingRail:
    name = "grounding.consistency"
    engine = "claude judge · sentence-level claims"

    def __init__(self, llm, consistency_threshold: float, relevance_threshold: float,
                 context_window: int, require_citations: bool = False) -> None:
        self.llm = llm
        self.consistency_threshold = consistency_threshold
        self.relevance_threshold = relevance_threshold
        self.context_window = context_window
        self.require_citations = require_citations

    def evaluate(self, question: str, answer: str, chunks: list[str],
                 action: str, result: RailResult) -> RailResult:
        result.higher_is_better = True
        result.threshold = self.consistency_threshold

        # Architectural no-op: nothing retrieved, nothing to ground against.
        if not chunks:
            result.verdict = Verdict.PASS
            result.score = 1.0
            result.meta = {"skipped": "no retrieved context (rail is retrieval-scoped)"}
            return result

        used = chunks[: self.context_window]
        context = "\n\n".join(f"[chunk {i + 1}] {c}" for i, c in enumerate(used))

        payload = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT ({len(used)} chunks):\n{context}\n\n"
            f"ANSWER:\n{answer}"
        )
        verdict = self.llm.judge(GROUNDING_SYSTEM, payload, GROUNDING_SCHEMA, max_tokens=3000)

        consistency = min(1.0, max(0.0, float(verdict.get("consistency", 0.0))))
        relevance = min(1.0, max(0.0, float(verdict.get("relevance", 0.0))))
        unsupported = [str(s) for s in (verdict.get("unsupported_claims") or [])][:10]

        result.score = consistency
        result.meta = {
            "consistency": round(consistency, 3),
            "relevance": round(relevance, 3),
            "consistency_threshold": round(self.consistency_threshold, 3),
            "relevance_threshold": round(self.relevance_threshold, 3),
            "chunks_considered": len(used),
            "chunks_available": len(chunks),
            "lexical_overlap": round(_lexical_overlap(answer, context), 3),
            "rationale": str(verdict.get("rationale", ""))[:200],
            "sentences": len(_SENTENCE.split(answer.strip())) if answer.strip() else 0,
        }
        for claim in unsupported:
            result.detections.append(
                Detection(kind="unsupported_claim", value=claim, start=0, end=0,
                          confidence=1.0 - consistency, note="not found in retrieved context")
            )

        # An answer that asserts without pointing at a source is unverifiable
        # even when the judge scores it well — the reader cannot check it.
        cited = {int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(used)}
        uncited = self.require_citations and not cited
        result.meta["citations"] = sorted(cited)
        result.meta["citations_required"] = self.require_citations

        failed = (
            consistency < self.consistency_threshold
            or relevance < self.relevance_threshold
            or uncited
        )
        if failed:
            result.meta["failed_on"] = (
                "consistency" if consistency < self.consistency_threshold
                else "relevance" if relevance < self.relevance_threshold
                else "citations"
            )
            # `regenerate` is not a verdict — it is a stage-level action the
            # engine takes. The rail reports BLOCK and the engine decides
            # whether that means retry, escalate, or refuse.
            result.verdict = Verdict.FLAG if action == "flag" else Verdict.BLOCK
        else:
            result.verdict = Verdict.PASS
        return result
