"""judge-only vs local+judge, measured on the same cases.

The question this answers is not "does the local layer work". It is whether
adding it changes what the stack *catches*, and at what price. Those are
separate numbers and they move in opposite directions, so a single accuracy
figure would hide the trade:

    recall down          the local layer blocked something the judge would have
                         cleared, or short-circuited past a case the judge would
                         have caught. This is the number that matters most —
                         a false negative on a security control is the failure
                         you do not find out about.

    precision down       the local layer blocked legitimate traffic. Cheaper to
                         discover, expensive to live with: the appeal case
                         (`why was my message blocked?`) scores 0.998 on the
                         injection model, and blocking it makes every refusal
                         un-appealable.

    judge calls down     the point of the exercise. Reported per request rather
                         than in total, so it can be read against the latency
                         column directly.

Both arms run the same `RailCase` list from the same suite file, so the only
variable is the engine configuration. Without an API key neither arm has a
judge, and the comparison degrades to deterministic vs deterministic+local —
still a real measurement, and it names itself as the narrower one in the report.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Policy, load
from ..engine import Engine
from ..tracing import AuditLog, Tracer
from .suite import RANK, SURFACES, Suite

#: The two configurations under test. Everything else is held constant.
ARMS: dict[str, dict[str, Any]] = {
    "judge-only": {
        "content.engine": "judge",
        "prompt_attack.engine": "judge",
        "grounding.engine": "judge",
    },
    "local+judge": {
        "content.engine": "local+judge",
        "prompt_attack.engine": "local+judge",
        "grounding.engine": "local+judge",
    },
}


class CountingLLM:
    """Wraps a judge to count how often it is actually consulted.

    The whole claim of the local layer is "fewer judge calls for the same
    verdicts", and a claim like that has to be counted rather than assumed.
    Delegates everything; the only state it owns is the tally.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def judge(self, system: str, user: str, schema: dict[str, Any],
              *, max_tokens: int = 2048) -> dict[str, Any]:
        self.calls += 1
        return self._inner.judge(system, user, schema, max_tokens=max_tokens)


@dataclass
class CaseOutcome:
    id: str
    expected: str
    actual: str
    ms: float
    judge_calls: int
    #: rail name -> which layer settled it. Kept per rail rather than as one
    #: label because a stage runs several rails concurrently: the content rail
    #: settling locally saves nothing if `scope.domain` next to it still calls
    #: a judge, and a single label per case hides exactly that.
    layers: dict[str, str] = field(default_factory=dict)


@dataclass
class ArmResult:
    """One configuration's numbers. Every rate is reported on its own."""

    name: str
    outcomes: list[CaseOutcome] = field(default_factory=list)
    judge_available: bool = False
    local_available: bool = False

    # -- the confusion matrix -----------------------------------------
    @property
    def tp(self) -> int:
        return sum(1 for o in self.outcomes if o.expected != "pass" and o.actual != "pass")

    @property
    def fp(self) -> int:
        return sum(1 for o in self.outcomes if o.expected == "pass" and o.actual != "pass")

    @property
    def fn(self) -> int:
        return sum(1 for o in self.outcomes if o.expected != "pass" and o.actual == "pass")

    @property
    def tn(self) -> int:
        return sum(1 for o in self.outcomes if o.expected == "pass" and o.actual == "pass")

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return round(self.tp / d, 3) if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return round(self.tp / d, 3) if d else None

    @property
    def exact_match(self) -> float | None:
        if not self.outcomes:
            return None
        hits = sum(1 for o in self.outcomes if o.actual == o.expected)
        return round(hits / len(self.outcomes), 3)

    @property
    def under_caught(self) -> list[str]:
        """Cases this arm caught less severely than labelled.

        Distinct from a false negative — `mask` where `block` was expected is
        not a miss, but it is a downgrade, and a downgrade is worth naming
        before it becomes one.
        """
        return [o.id for o in self.outcomes
                if o.actual != "pass" and RANK[o.actual] < RANK[o.expected]]

    # -- cost -----------------------------------------------------------
    @property
    def judge_calls(self) -> int:
        return sum(o.judge_calls for o in self.outcomes)

    @property
    def calls_per_request(self) -> float | None:
        if not self.outcomes:
            return None
        return round(self.judge_calls / len(self.outcomes), 3)

    @property
    def settled_locally(self) -> int:
        """Rail evaluations settled without a judge, across all cases."""
        return sum(1 for o in self.outcomes for layer in o.layers.values()
                   if layer in ("local", "pattern"))

    @property
    def judge_calls_by_rail(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            for rail, layer in o.layers.items():
                if layer == "judge":
                    out[rail] = out.get(rail, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def local_wins_by_rail(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            for rail, layer in o.layers.items():
                if layer == "local":
                    out[rail] = out.get(rail, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def fallback_rate(self) -> float | None:
        """How often a request reached the judge at all."""
        if not self.outcomes:
            return None
        reached = sum(1 for o in self.outcomes if o.judge_calls > 0)
        return round(reached / len(self.outcomes), 3)

    @property
    def block_rate(self) -> float | None:
        if not self.outcomes:
            return None
        blocked = sum(1 for o in self.outcomes if o.actual == "block")
        return round(blocked / len(self.outcomes), 3)

    @property
    def p50_ms(self) -> float | None:
        return round(statistics.median(o.ms for o in self.outcomes), 1) if self.outcomes else None

    @property
    def p95_ms(self) -> float | None:
        if not self.outcomes:
            return None
        ordered = sorted(o.ms for o in self.outcomes)
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1)

    @property
    def total_ms(self) -> float:
        return round(sum(o.ms for o in self.outcomes), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "judge_available": self.judge_available,
            "local_available": self.local_available,
            "cases": len(self.outcomes),
            "true_positives": self.tp, "false_positives": self.fp,
            "false_negatives": self.fn, "true_negatives": self.tn,
            "precision": self.precision, "recall": self.recall,
            "exact_verdict_match": self.exact_match,
            "under_caught": self.under_caught,
            "judge_calls": self.judge_calls,
            "calls_per_request": self.calls_per_request,
            "settled_without_judge": self.settled_locally,
            "judge_calls_by_rail": self.judge_calls_by_rail,
            "local_wins_by_rail": self.local_wins_by_rail,
            "fallback_rate": self.fallback_rate,
            "block_rate": self.block_rate,
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms, "total_ms": self.total_ms,
        }


@dataclass
class Comparison:
    arms: list[ArmResult] = field(default_factory=list)
    suite: str = ""
    degraded: str = ""

    def by_name(self, name: str) -> ArmResult | None:
        return next((a for a in self.arms if a.name == name), None)

    @property
    def regressions(self) -> list[str]:
        """Ways `local+judge` is worse than `judge-only`, in plain words.

        Empty is the only result that justifies turning the local layer on. A
        recall regression is listed first because it is the one that costs
        something you cannot see in production.
        """
        base, arm = self.by_name("judge-only"), self.by_name("local+judge")
        if base is None or arm is None:
            return []
        out: list[str] = []
        if (arm.recall or 0) < (base.recall or 0):
            out.append(f"recall {base.recall} -> {arm.recall} "
                       f"({arm.fn - base.fn:+d} false negatives)")
        newly_missed = {o.id for o in arm.outcomes if o.expected != "pass" and o.actual == "pass"}
        newly_missed -= {o.id for o in base.outcomes if o.expected != "pass" and o.actual == "pass"}
        if newly_missed:
            out.append(f"newly missed: {', '.join(sorted(newly_missed))}")
        if (arm.precision or 0) < (base.precision or 0):
            out.append(f"precision {base.precision} -> {arm.precision} "
                       f"({arm.fp - base.fp:+d} false positives)")
        newly_blocked = {o.id for o in arm.outcomes if o.expected == "pass" and o.actual != "pass"}
        newly_blocked -= {o.id for o in base.outcomes if o.expected == "pass" and o.actual != "pass"}
        if newly_blocked:
            out.append(f"newly blocked legitimate: {', '.join(sorted(newly_blocked))}")
        for extra in set(arm.under_caught) - set(base.under_caught):
            out.append(f"downgraded: {extra}")
        return out

    def to_dict(self) -> dict[str, Any]:
        base, arm = self.by_name("judge-only"), self.by_name("local+judge")
        delta: dict[str, Any] = {}
        if base and arm:
            delta = {
                "recall": _delta(base.recall, arm.recall),
                "precision": _delta(base.precision, arm.precision),
                "false_negatives": arm.fn - base.fn,
                "false_positives": arm.fp - base.fp,
                "judge_calls": arm.judge_calls - base.judge_calls,
                "judge_calls_saved_pct": (
                    round(100 * (base.judge_calls - arm.judge_calls) / base.judge_calls, 1)
                    if base.judge_calls else None
                ),
                "p50_ms": _delta(base.p50_ms, arm.p50_ms),
                "total_ms": _delta(base.total_ms, arm.total_ms),
            }
        return {
            "suite": self.suite,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "degraded": self.degraded,
            "arms": [a.to_dict() for a in self.arms],
            "delta": delta,
            "regressions": self.regressions,
            "verdict": "no regression" if not self.regressions else "REGRESSION",
        }


def _delta(a: float | None, b: float | None) -> float | None:
    return round(b - a, 3) if a is not None and b is not None else None


def _engine_for(arm: str, config_path: str, llm: Any, audit: AuditLog) -> tuple[Engine, Any]:
    policy = load(config_path)
    for key, value in ARMS[arm].items():
        policy.values[key] = value
    counting = CountingLLM(llm) if llm is not None else None
    return Engine(policy, counting, audit), counting


def run(suite: Suite, config_path: str = "config/policy.yaml", *,
        llm: Any = None, audit: AuditLog | None = None) -> Comparison:
    """Run every rail case under both arms and report the difference."""
    from ..rails import deberta_injection_check, toxicity_check

    audit = audit or AuditLog()
    comparison = Comparison(suite=suite.source)

    # Load before timing anything. A request path deliberately never waits for
    # a model, so an unwarmed run would put a one-off 14-second disk read into
    # whichever case happened to be first — measuring the filesystem and
    # reporting it as rail latency.
    if toxicity_check.available():
        toxicity_check.warm()
        deberta_injection_check.warm()
    local_up = (toxicity_check.classifier() is not None
                and deberta_injection_check.classifier() is not None)
    if llm is None:
        comparison.degraded = (
            "no judge configured — comparing deterministic against "
            "deterministic+local, not judge-only against local+judge"
        )
    if not local_up:
        comparison.degraded = (
            (comparison.degraded + "; " if comparison.degraded else "")
            + "local models unavailable — both arms will behave identically"
        )

    for name in ARMS:
        engine, counting = _engine_for(name, config_path, llm, audit)
        result = ArmResult(name=name, judge_available=llm is not None,
                           local_available=local_up)

        for case in suite.rails:
            surface = SURFACES.get(case.surface)
            if surface is None:
                continue
            before = counting.calls if counting else 0
            began = time.perf_counter()
            outcome = engine.evaluate(case.text, surface, Tracer(session_id="eval"), "eval")
            ms = (time.perf_counter() - began) * 1000
            result.outcomes.append(CaseOutcome(
                id=case.id, expected=case.expect, actual=outcome.verdict.value,
                ms=ms, judge_calls=(counting.calls - before) if counting else 0,
                layers={r.rail: str(r.meta.get("layer", "")) for r in outcome.results
                        if r.meta.get("layer")},
            ))
        comparison.arms.append(result)

    return comparison


def render(comparison: Comparison) -> str:
    """The report, as a table. Rates side by side so the trade is visible."""
    d = comparison.to_dict()
    lines: list[str] = []
    if comparison.degraded:
        lines.append(f"  note: {comparison.degraded}\n")

    rows = [
        ("precision", "precision"), ("recall", "recall"),
        ("false positives", "false_positives"), ("false negatives", "false_negatives"),
        ("exact verdict", "exact_verdict_match"), ("block rate", "block_rate"),
        ("judge calls", "judge_calls"), ("calls/request", "calls_per_request"),
        ("reached judge", "fallback_rate"), ("settled locally", "settled_without_judge"),
        ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"), ("total ms", "total_ms"),
    ]
    arms = d["arms"]
    lines.append("  " + "metric".ljust(18) + "".join(a["arm"].rjust(16) for a in arms))
    lines.append("  " + "-" * (18 + 16 * len(arms)))
    for label, key in rows:
        cells = "".join(str(a.get(key)).rjust(16) for a in arms)
        lines.append("  " + label.ljust(18) + cells)

    if d["delta"].get("judge_calls_saved_pct") is not None:
        lines.append(f"\n  judge calls saved: {d['delta']['judge_calls_saved_pct']}%")

    # Which rails still pay for a judge is the actionable half of the result.
    # A stage costs its slowest rail, so one rail going local saves nothing
    # while a sibling in the same stage is still making the call.
    lines.append("\n  judge calls by rail")
    for arm in arms:
        by_rail = arm.get("judge_calls_by_rail") or {}
        wins = arm.get("local_wins_by_rail") or {}
        lines.append(f"    {arm['arm']}")
        for rail, n in by_rail.items():
            saved = wins.get(rail, 0)
            note = f"   ({saved} settled locally)" if saved else ""
            lines.append(f"      {rail:<26}{n:>4}{note}")
        if not by_rail:
            lines.append("      (none)")
    lines.append("")
    if comparison.regressions:
        lines.append("  REGRESSION — do not enable the local layer on these numbers:")
        lines.extend(f"    - {r}" for r in comparison.regressions)
    else:
        lines.append("  no regression in precision, recall, or severity")
    return "\n".join(lines) + "\n"
