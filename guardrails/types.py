"""Core value types for the guardrail engine.

Everything the engine produces is one of these. The tracer, the API layer, and
the frontend all read the same shapes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """A rail's decision.

    Ordering matters: `precedence()` resolves competing verdicts to the most
    restrictive one. This ordering is a safety invariant — see registry.py,
    `policy.verdict_precedence`.
    """

    PASS = "pass"
    FLAG = "flag"
    MASK = "mask"
    BLOCK = "block"

    @property
    def rank(self) -> int:
        return {"pass": 0, "flag": 1, "mask": 2, "block": 3}[self.value]


def precedence(verdicts: list[Verdict]) -> Verdict:
    """Most restrictive verdict wins. No rail can soften another rail's result."""
    if not verdicts:
        return Verdict.PASS
    return max(verdicts, key=lambda v: v.rank)


# Configured action → verdict. Every rail resolves through this, because each
# rail rolling its own mapping is how `pass` came to mean `block` in three of
# them: an operator setting a rail to monitoring-only got the opposite.
_ACTION_VERDICT = {
    "block": Verdict.BLOCK,
    "mask": Verdict.MASK,
    "flag": Verdict.FLAG,
    "pass": Verdict.PASS,
    # A stage-level action, not a rail verdict — the engine decides whether a
    # failed output rail means retry, escalate, or refuse. The rail says BLOCK
    # and lets the engine route it.
    "regenerate": Verdict.BLOCK,
}


def action_verdict(action: str, default: Verdict = Verdict.BLOCK) -> Verdict:
    """Resolve a configured action. Unknown actions take the strict default."""
    return _ACTION_VERDICT.get(str(action).strip().lower(), default)


class Surface(str, Enum):
    """Where in the pipeline a rail is evaluating.

    Each of these is a genuinely different trust boundary, which is why they
    are separate columns in the severity matrix rather than one "input" and one
    "output". A document being ingested is not a user prompt; a tool result is
    not a retrieved chunk. Posture differs, so the surface differs.
    """

    USER_PROMPT = "user.prompt"
    USER_FEEDBACK = "user.feedback"
    INGEST = "ingest.document"
    RETRIEVAL = "retrieval"
    LLM_RESPONSE = "llm.response"
    LLM_ASK_USER = "llm.ask_user"
    AGENT_TOOL = "agent.tool"
    AGENT_DATA = "agent.data"


@dataclass
class Detection:
    """One concrete thing a rail found."""

    kind: str  # "US_SSN", "prompt_injection", "hate", ...
    value: str  # the matched text (raw — audit-only, never returned to a client)
    start: int
    end: int
    confidence: float = 1.0
    note: str = ""

    def redacted(self) -> dict[str, Any]:
        """Client-safe view. Never leaks the matched value."""
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "note": self.note,
        }


@dataclass
class RailResult:
    """What a single rail returns."""

    rail: str
    engine: str
    verdict: Verdict
    score: float = 0.0
    threshold: float = 0.0
    unit: str = "score"  # "score" | "count"
    higher_is_better: bool = False
    detections: list[Detection] = field(default_factory=list)
    duration_ms: float = 0.0
    text_out: str | None = None  # set when the rail rewrote the text (masking)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rail": self.rail,
            "engine": self.engine,
            "verdict": self.verdict.value,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "detections": [d.redacted() for d in self.detections],
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "meta": self.meta,
        }


@dataclass
class StageTrace:
    """One stage of the pipeline — a group of rails that ran together."""

    name: str
    subtitle: str = ""
    kind: str = "rail"  # "rail" | "model" | "retrieval" | "retry"
    start_ms: float = 0.0
    duration_ms: float = 0.0
    verdict: Verdict = Verdict.PASS
    rails: list[RailResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subtitle": self.subtitle,
            "kind": self.kind,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "verdict": self.verdict.value,
            "rails": [r.to_dict() for r in self.rails],
            "notes": self.notes,
        }


@dataclass
class Trace:
    """The full record of one request through the stack.

    Built incrementally as the request runs — tracing is not bolted on
    afterwards, every rail writes into it as it completes.
    """

    request_id: str = field(default_factory=lambda: "req_" + uuid.uuid4().hex[:8])
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    stages: list[StageTrace] = field(default_factory=list)
    verdict: Verdict = Verdict.PASS
    total_ms: float = 0.0
    guardrail_ms: float = 0.0
    regenerations: int = 0
    fail_mode_triggered: bool = False

    # ---- convenience -------------------------------------------------
    @property
    def rails(self) -> list[RailResult]:
        return [r for s in self.stages for r in s.rails]

    def rail_count(self) -> dict[str, int]:
        out = {"pass": 0, "flag": 0, "mask": 0, "block": 0}
        for r in self.rails:
            out[r.verdict.value] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "stages": [s.to_dict() for s in self.stages],
            "verdict": self.verdict.value,
            "total_ms": round(self.total_ms, 2),
            "guardrail_ms": round(self.guardrail_ms, 2),
            "guardrail_pct": (
                round(self.guardrail_ms / self.total_ms * 100, 1) if self.total_ms else 0.0
            ),
            "regenerations": self.regenerations,
            "fail_mode_triggered": self.fail_mode_triggered,
            "rail_count": self.rail_count(),
            "rails_evaluated": len(self.rails),
        }


@dataclass
class EvaluationResult:
    """What `engine.evaluate()` hands back for one surface."""

    verdict: Verdict
    text: str  # possibly masked
    results: list[RailResult]
    duration_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    @property
    def masked(self) -> bool:
        return self.verdict is Verdict.MASK
