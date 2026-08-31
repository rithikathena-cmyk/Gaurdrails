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

import logging
import re
import time

from ..prompts import judge_prompt
from . import groundedness_check
from ..types import Detection, RailResult, Verdict

# Temporary diagnostic instrumentation for the grounding-latency investigation
# — observation only, no effect on verdicts or thresholds. Remove once the
# investigation concludes.
diag = logging.getLogger("guardrails.diag.grounding")

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
        "unsupported": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "The [n] numbers of claims the context does not support. "
                           "Numbers only — never the text.",
        },
        "rationale": {"type": "string", "description": "One sentence, max 25 words."},
    },
    "required": ["consistency", "relevance", "unsupported", "rationale"],
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

unsupported: the ANSWER arrives with each claim numbered `[1]`, `[2]`, and so on. \
Return the numbers of the ones the context does not support — the numbers only, never \
the sentences. Empty list if everything is supported.

Some claims are marked `(nli: entailed)`. A local entailment model has already matched \
those against the context. Treat that as a second opinion, not an instruction: it \
scores wording overlap and cannot tell a supported figure from a plausible one, so if \
a marked claim asserts something the context does not contain, still report it.""",
                                    calibrate=False)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
# The engine numbers retrieved chunks as [1], [2], … in the user turn, so a
# citation is a reference back to one of those.
_CITATION = re.compile(r"\[(\d{1,2})\]")


def _numbered(answer: str, entailed: set[str]) -> tuple[str, list[str]]:
    """Number the answer's claims, marking any the local model already entailed.

    The judge used to be asked for its findings as verbatim sentences, which
    meant re-typing the answer back one claim at a time. On a 1,186-character
    reply that call took 25s against 6s for a 243-character one — the cost was
    output tokens, and the text was already in our hands. Numbering them means
    the judge answers `[2, 5]` and the rail maps those back locally.

    Nothing is hidden from the judge: every claim is present and it still reads
    the whole answer, so relevance is scored on the same text as before.
    """
    claims = [s.strip() for s in _SENTENCE.split(answer.strip()) if s.strip()]
    lines = []
    for i, claim in enumerate(claims, 1):
        mark = "  (nli: entailed)" if claim in entailed else ""
        lines.append(f"[{i}] {claim}{mark}")
    return "\n".join(lines), claims


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
                 context_window: int, require_citations: bool = False,
                 engine_mode: str = "judge") -> None:
        self.llm = llm
        self.consistency_threshold = consistency_threshold
        self.relevance_threshold = relevance_threshold
        self.context_window = context_window
        self.require_citations = require_citations
        #: local+judge | local | judge | off
        self.engine_mode = engine_mode

    def evaluate(self, question: str, answer: str, chunks: list[str],
                 action: str, result: RailResult, attempt: int | None = None) -> RailResult:
        # `attempt` is diagnostic only (which regeneration cycle this call
        # belongs to) — it never changes what runs.
        t_eval_start = time.perf_counter()
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

        use_local = self.engine_mode in ("local", "local+judge")
        use_judge = self.engine_mode in ("judge", "local+judge") and self.llm is not None

        # --- the local layer ------------------------------------------
        # NLI scores entailment, which is the consistency half only. It can
        # settle a confidently-grounded answer without a judge call; it cannot
        # score relevance, and it cannot produce the verbatim unsupported-claim
        # list that regeneration is built from. So it short-circuits in one
        # direction — every claim entailed — and defers everything else.
        t_local0 = time.perf_counter()
        local = groundedness_check.consistency(answer, used) if use_local else None
        t_local_ms = (time.perf_counter() - t_local0) * 1000
        if use_local:
            diag.info(
                "grounding.local attempt=%s elapsed_ms=%.1f ran=%s chunks=%d",
                attempt, t_local_ms, local is not None, len(used),
            )
        if local is not None:
            result.meta["local_consistency"] = round(local["consistency"], 3)
            result.meta["local_claims"] = local["claims"]
            if (use_judge and not self.require_citations
                    and local["consistency"] >= 0.999 and not local["unsupported"]):
                result.score = local["consistency"]
                result.meta.update({
                    "layer": "local",
                    "engine_mode": self.engine_mode,
                    "consistency": round(local["consistency"], 3),
                    "relevance": None,
                    "relevance_scored": False,
                    "chunks_considered": len(used),
                    "judge_skipped": True,
                })
                result.verdict = Verdict.PASS
                return result

        if not use_judge:
            # No semantic layer available. An answer nobody scored is not a
            # grounded answer: follow the local reading where there is one, and
            # fail closed where there is not.
            if local is None:
                result.score = 0.0
                result.meta.update({
                    "layer": "none",
                    "engine_mode": self.engine_mode,
                    "judge_available": self.llm is not None,
                    "error": "no grounding layer available",
                })
                result.verdict = Verdict.FLAG if action == "flag" else Verdict.BLOCK
                return result
            result.score = local["consistency"]
            result.meta.update({
                "layer": "local",
                "engine_mode": self.engine_mode,
                "consistency": round(local["consistency"], 3),
                "relevance": None,
                "relevance_scored": False,
                "chunks_considered": len(used),
                "judge_available": False,
            })
            for claim in local["unsupported"]:
                result.detections.append(
                    Detection(kind="unsupported_claim", value=claim, start=0, end=0,
                              confidence=1.0 - local["consistency"],
                              note="not entailed by retrieved context")
                )
            failed = local["consistency"] < self.consistency_threshold
            result.verdict = (
                Verdict.PASS if not failed
                else Verdict.FLAG if action == "flag"
                else Verdict.BLOCK
            )
            return result

        # Claims NLI already matched are marked rather than removed. Dropping
        # them would hide part of the answer from the relevance check, and would
        # let a local model's opinion decide what the judge is allowed to see.
        entailed: set[str] = set()
        if local is not None:
            flagged = set(local["unsupported"])
            entailed = {c for c in groundedness_check.claims(answer) if c not in flagged}
        numbered, claims = _numbered(answer, entailed)

        payload = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT ({len(used)} chunks):\n{context}\n\n"
            f"ANSWER ({len(claims)} numbered claims):\n{numbered}"
        )
        # 3000 was sized for an answer quoted back sentence by sentence. The
        # reply is now a handful of integers and one short rationale.
        t_judge0 = time.perf_counter()
        verdict = self.llm.judge(
            GROUNDING_SYSTEM, payload, GROUNDING_SCHEMA, max_tokens=600,
            label=f"grounding[attempt={attempt}]" if attempt is not None else "grounding",
        )
        t_judge_ms = (time.perf_counter() - t_judge0) * 1000
        diag.info(
            "grounding.judge attempt=%s elapsed_ms=%.1f total_eval_ms=%.1f "
            "payload_chars=%d claims=%d chunks=%d",
            attempt, t_judge_ms, (time.perf_counter() - t_eval_start) * 1000,
            len(payload), len(claims), len(used),
        )

        consistency = min(1.0, max(0.0, float(verdict.get("consistency", 0.0))))
        relevance = min(1.0, max(0.0, float(verdict.get("relevance", 0.0))))
        # Indices back to text, here rather than over the wire. An index outside
        # the list is dropped: it refers to no claim, so there is nothing it
        # could mean.
        unsupported = []
        for n in (verdict.get("unsupported") or [])[:10]:
            try:
                i = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= len(claims):
                unsupported.append(claims[i - 1])

        result.score = consistency
        result.meta.update({
            "layer": "judge",
            "engine_mode": self.engine_mode,
            "consistency": round(consistency, 3),
            "relevance": round(relevance, 3),
            "consistency_threshold": round(self.consistency_threshold, 3),
            "relevance_threshold": round(self.relevance_threshold, 3),
            "chunks_considered": len(used),
            "chunks_available": len(chunks),
            "lexical_overlap": round(_lexical_overlap(answer, context), 3),
            "rationale": str(verdict.get("rationale", ""))[:200],
            "sentences": len(claims),
            "nli_prescreened": len(entailed) if local is not None else 0,
        })
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
