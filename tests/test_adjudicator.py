"""The adjudicator — the one place a model may revisit a verdict.

Two properties matter more than the happy path, and both are asserted here:

    cost      it must not run on ordinary traffic. A rail scoring nowhere near
              its threshold has already decided; spending a model call on that
              would make every request pay for the 2% that are genuinely close.

    floor     it must not be able to erase an incident. Raising a verdict is
              free; lowering one stops at FLAG, needs confidence, and is refused
              outright for deterministic and failed rails.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Corpus, Engine, load
from backend.guardrails.rails.adjudicator import DOWNGRADE_FLOOR, Adjudicator
from backend.guardrails.llm import Generation
from backend.guardrails.tracing import Tracer
from backend.guardrails.types import RailResult, Surface, Verdict
from tests.conftest import REPO


class ScriptedJudge:
    """Returns one verdict, and counts how often it was asked for anything."""

    model = "stub"

    def __init__(self, verdict="pass", confidence=0.9, rationale="scripted"):
        self.calls = 0
        self.payload = {"verdict": verdict, "confidence": confidence,
                        "rationale": rationale}
        self.last_user = ""

    def judge(self, system, user, schema, *, max_tokens=2048):
        self.calls += 1
        self.last_user = user
        props = set(schema.get("properties", {}))
        if "verdict" in props and "confidence" in props:
            return dict(self.payload)
        if "in_scope" in props:
            return {"in_scope": 1.0, "topic": "stub", "rationale": "stub"}
        if "entities" in props:
            return {"entities": []}
        if "consistency" in props:
            return {"consistency": 1.0, "relevance": 1.0,
                    "unsupported": [], "rationale": "stub"}
        if "injection" in props:
            return {"injection": 0.0, "technique": "none", "rationale": "stub"}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}

    def generate(self, system, messages, **kw):
        return Generation(text="Renewal needs the current licence and proof of address.",
                          model=self.model)


def scored(rail: str, score: float, threshold: float, verdict=Verdict.PASS, **kw) -> RailResult:
    return RailResult(rail=rail, engine="test", verdict=verdict,
                      score=score, threshold=threshold, **kw)


# ── the trigger: what counts as marginal ───────────────────────────
def test_a_score_nowhere_near_its_threshold_never_reaches_the_model():
    """The cost property. Ordinary traffic must not pay for the ambiguous band."""
    judge = ScriptedJudge()
    adj = Adjudicator(judge, margin=0.08)
    ruling = adj.review("how do I renew a trade licence",
                        [scored("content.safety", 0.02, 0.49)], Verdict.PASS)
    assert ruling is None
    assert judge.calls == 0


def test_a_score_within_the_margin_is_reviewed():
    judge = ScriptedJudge(verdict="pass", confidence=0.9)
    adj = Adjudicator(judge, margin=0.08)
    ruling = adj.review("a genuinely ambiguous request",
                        [scored("content.safety", 0.47, 0.49)], Verdict.PASS)
    assert ruling is not None
    assert judge.calls == 1
    assert ruling.triggers == ["content.safety"]


def test_the_margin_is_symmetric_around_the_threshold():
    """0.02 under the line is exactly as undecided as 0.02 over it."""
    adj = Adjudicator(ScriptedJudge(), margin=0.08)
    under = adj.marginal([scored("content.safety", 0.47, 0.49)])
    over = adj.marginal([scored("content.safety", 0.51, 0.49)])
    assert len(under) == 1 and len(over) == 1


@pytest.mark.parametrize("rail", ["pii.detect", "words.lexicon", "policy.rules"])
def test_deterministic_rails_are_never_adjudicated(rail):
    """A regex matched or it did not. There is no band to be uncertain in."""
    judge = ScriptedJudge(verdict="pass", confidence=1.0)
    adj = Adjudicator(judge, margin=0.5)
    ruling = adj.review("my api_key = sk-live4f9a2b8ccc12",
                        [scored(rail, 1.0, 1.0, Verdict.BLOCK, unit="count")],
                        Verdict.BLOCK)
    assert ruling is None
    assert judge.calls == 0


def test_a_rail_that_errored_is_never_adjudicated():
    """Its verdict is a fail-closed default, not a score — softening it would
    undo the guarantee exactly when the stack is least healthy."""
    judge = ScriptedJudge(verdict="pass", confidence=1.0)
    adj = Adjudicator(judge, margin=0.5)
    r = scored("content.safety", 0.0, 0.49, Verdict.BLOCK)
    r.error = "latency budget exceeded (20000ms)"
    assert adj.review("anything", [r], Verdict.BLOCK) is None
    assert judge.calls == 0


def test_a_rail_outside_the_configured_list_is_not_adjudicated():
    judge = ScriptedJudge()
    adj = Adjudicator(judge, margin=0.08, rails=["prompt_attack"])
    assert adj.review("x", [scored("content.safety", 0.48, 0.49)], Verdict.PASS) is None
    assert judge.calls == 0


# ── raising ────────────────────────────────────────────────────────
def test_a_marginal_pass_can_be_raised_to_a_block():
    adj = Adjudicator(ScriptedJudge(verdict="block", confidence=0.8), margin=0.08)
    ruling = adj.review("a marginal request",
                        [scored("content.safety", 0.47, 0.49)], Verdict.PASS)
    assert ruling.verdict is Verdict.BLOCK
    assert ruling.direction == "raised"
    assert ruling.changed


def test_raising_needs_no_confidence_at_all():
    """Deciding a marginal request is worse than it scored is the safe direction."""
    adj = Adjudicator(ScriptedJudge(verdict="block", confidence=0.05), margin=0.08)
    ruling = adj.review("x", [scored("content.safety", 0.47, 0.49)], Verdict.PASS)
    assert ruling.verdict is Verdict.BLOCK


# ── lowering, and the floor ────────────────────────────────────────
def test_a_marginal_block_can_be_lowered_to_a_flag():
    adj = Adjudicator(ScriptedJudge(verdict="flag", confidence=0.9), margin=0.08)
    ruling = adj.review("a frustrated but ordinary complaint",
                        [scored("content.safety", 0.50, 0.49, Verdict.BLOCK)],
                        Verdict.BLOCK)
    assert ruling.verdict is Verdict.FLAG
    assert ruling.direction == "lowered"
    assert not ruling.clamped


def test_a_downgrade_to_pass_is_clamped_at_the_floor():
    """The core safety property: no single confident model call may erase an
    incident. `pass` leaves no record; `flag` does."""
    adj = Adjudicator(ScriptedJudge(verdict="pass", confidence=1.0), margin=0.08)
    ruling = adj.review("x", [scored("content.safety", 0.50, 0.49, Verdict.BLOCK)],
                        Verdict.BLOCK)
    assert ruling.verdict is DOWNGRADE_FLOOR is Verdict.FLAG
    assert ruling.clamped
    assert ruling.verdict is not Verdict.PASS


def test_an_unconfident_downgrade_is_refused():
    adj = Adjudicator(ScriptedJudge(verdict="flag", confidence=0.3),
                      margin=0.08, min_confidence=0.6)
    ruling = adj.review("x", [scored("content.safety", 0.50, 0.49, Verdict.BLOCK)],
                        Verdict.BLOCK)
    assert ruling.verdict is Verdict.BLOCK
    assert not ruling.changed
    assert "too unconfident" in ruling.rationale


def test_an_unusable_verdict_leaves_the_rails_decision_standing():
    adj = Adjudicator(ScriptedJudge(verdict="probably fine?", confidence=1.0), margin=0.08)
    ruling = adj.review("x", [scored("content.safety", 0.50, 0.49, Verdict.BLOCK)],
                        Verdict.BLOCK)
    assert ruling.verdict is Verdict.BLOCK
    assert not ruling.changed


# ── what the model is shown ────────────────────────────────────────
def test_the_evidence_names_the_marginal_rail_and_its_distance():
    judge = ScriptedJudge()
    adj = Adjudicator(judge, margin=0.08)
    adj.review("the request text", [scored("content.safety", 0.47, 0.49)], Verdict.PASS)
    assert "the request text" in judge.last_user
    assert "scored 0.470 against a 0.490 threshold" in judge.last_user
    assert "MARGINAL: content.safety (0.020 from its line)" in judge.last_user


# ── disabled paths ─────────────────────────────────────────────────
def test_no_model_means_no_adjudication():
    adj = Adjudicator(None, margin=0.08)
    assert adj.review("x", [scored("content.safety", 0.48, 0.49)], Verdict.PASS) is None


def test_disabled_by_configuration_means_no_adjudication():
    judge = ScriptedJudge()
    adj = Adjudicator(judge, margin=0.08, enabled=False)
    assert adj.review("x", [scored("content.safety", 0.48, 0.49)], Verdict.PASS) is None
    assert judge.calls == 0


# ── wired into the engine ──────────────────────────────────────────
def _engine(judge, tmp_path):
    return Engine(load(REPO / "config" / "policy.yaml"), judge,
                  AuditLog(tmp_path / "a.log"), Corpus(seed=True))


def test_the_ruling_reaches_the_trace(tmp_path):
    engine = _engine(ScriptedJudge(), tmp_path)
    result = engine.converse("what documents renew a trade licence", "s")
    rails = [r.rail for stage in result.trace.stages for r in stage.rails]
    # Ordinary traffic: nothing marginal, so the adjudicator leaves no rail at all.
    assert "adjudicator.review" not in rails


def test_the_adjudicator_is_declared_in_the_registry():
    """Every parameter declared once — the Parameters page reads this."""
    from backend.guardrails.registry import ADJUSTABLE, LOCKED

    assert "adjudicator.margin" in ADJUSTABLE
    assert "adjudicator.min_confidence" in ADJUSTABLE
    assert "adjudicator.downgrade_floor" in LOCKED
    assert LOCKED["adjudicator.downgrade_floor"].value == "flag"
