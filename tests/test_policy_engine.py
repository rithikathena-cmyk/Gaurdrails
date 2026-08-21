"""The deterministic Policy Engine — unit tests, independent of any agent.

Every specialist agent and the Supervisor route through this module's
`decide()`; these tests exercise its actual decision table directly rather
than only indirectly through an agent's scripted end-to-end run.
"""

from __future__ import annotations

import pytest

from backend.guardrails.agents.policy_engine import (
    ACTION_RANK, PolicyEngine, floor_from_agent_results, floor_from_policy,
)
from backend.guardrails.agents.types import ActionOutcome, AgentDecision, AgentResult


@pytest.fixture
def engine():
    return PolicyEngine()


# ── floor_from_policy: raw config strings -> GuardrailAction ────────────
@pytest.mark.parametrize("raw,expected", [
    ("pass", "ALLOW"), ("allow", "ALLOW"),
    ("flag", "FLAG"),
    ("mask", "MASK"),
    ("redact", "REDACT"),
    ("block", "BLOCK"), ("regenerate", "BLOCK"),
    ("escalate", "ESCALATE"), ("human_review", "ESCALATE"),
    ("MASK", "MASK"), ("  block  ", "BLOCK"),  # case and whitespace insensitive
])
def test_floor_from_policy_maps_every_real_config_value(raw, expected):
    assert floor_from_policy(raw) == expected


def test_floor_from_policy_fails_closed_on_an_unknown_string():
    assert floor_from_policy("some-typo-in-the-yaml") == "BLOCK"


# ── the core comparison: most restrictive wins ───────────────────────────
def test_a_permissive_recommendation_is_overridden_by_a_stricter_floor(engine):
    d = engine.decide("ALLOW", has_findings=True, policy_action="mask")
    assert d.final_action == "MASK"
    assert d.overridden is True


def test_a_stricter_recommendation_than_the_floor_is_upheld(engine):
    d = engine.decide("BLOCK", has_findings=True, policy_action="mask")
    assert d.final_action == "BLOCK"
    assert d.overridden is False


def test_a_recommendation_equal_to_the_floor_is_not_flagged_as_overridden(engine):
    d = engine.decide("MASK", has_findings=True, policy_action="mask")
    assert d.final_action == "MASK"
    assert d.overridden is False


def test_no_findings_means_no_floor_regardless_of_policy_action(engine):
    """The floor only ever comes from something the agent's own tools
    actually found — a configured action with nothing to apply it to
    contributes nothing."""
    d = engine.decide("FLAG", has_findings=False, policy_action="block")
    assert d.floor_action == "ALLOW"
    assert d.final_action == "FLAG"
    assert d.overridden is False


def test_no_policy_action_means_no_floor_even_with_findings(engine):
    d = engine.decide("FLAG", has_findings=True, policy_action="")
    assert d.floor_action == "ALLOW"
    assert d.final_action == "FLAG"


# ── ESCALATE, from either side — the case that must never crash ─────────
def test_recommendation_escalate_with_a_confident_floor_yields_the_floor(engine):
    d = engine.decide("ESCALATE", has_findings=True, policy_action="block")
    assert d.final_action == "BLOCK"
    assert d.overridden is True


def test_recommendation_escalate_with_no_floor_stays_escalate(engine):
    d = engine.decide("ESCALATE", has_findings=False, policy_action="")
    assert d.final_action == "ESCALATE"
    assert d.overridden is False


def test_floor_escalate_a_real_config_value_overrides_a_concrete_recommendation(engine):
    """`grounding.action_on_fail: human_review` is a real, shipped policy
    option — `floor_from_policy` maps it to ESCALATE. Before this was fixed,
    this exact combination raised `KeyError` inside `ACTION_RANK[a]`, since
    ESCALATE has no severity rank to compare against the other five."""
    d = engine.decide("MASK", has_findings=True, policy_action="human_review")
    assert d.final_action == "ESCALATE"
    assert d.overridden is True


def test_floor_escalate_and_recommendation_escalate_together(engine):
    d = engine.decide("ESCALATE", has_findings=True, policy_action="escalate")
    assert d.final_action == "ESCALATE"
    assert d.overridden is False


def test_no_combination_of_findings_and_policy_action_ever_raises(engine):
    """Every recommendation against every possible floor source, including
    every raw config string this module maps to ESCALATE — none of the 6x9
    combinations should raise."""
    from backend.guardrails.agents.types import GuardrailAction

    actions = list(ACTION_RANK) + ["ESCALATE"]
    raw_policy_values = ["pass", "flag", "mask", "redact", "block", "regenerate",
                         "escalate", "human_review", ""]
    for rec in actions:
        for raw in raw_policy_values:
            engine.decide(rec, has_findings=True, policy_action=raw)  # must not raise


# ── floor_from_agent_results: the Supervisor's own floor ────────────────
def _result(action: str) -> AgentResult:
    return AgentResult(
        request_id="r", agent="stub", version="1.0.0", status="completed",
        decision=AgentDecision(action=action, confidence=0.9, rationale="x", findings=[]),
        outcome=ActionOutcome(action=action, capability="x", summary="x"),
    )


def test_floor_from_agent_results_takes_the_most_restrictive():
    results = {"pii": _result("MASK"), "content": _result("BLOCK")}
    assert floor_from_agent_results(results) == "BLOCK"


def test_floor_from_agent_results_with_no_agents_is_allow():
    assert floor_from_agent_results({}) == "ALLOW"


def test_floor_from_agent_results_ignores_an_escalated_agent():
    """An escalated agent contributes nothing to the floor — "unknown" is
    not a severity level. The other agent's real outcome still applies."""
    results = {"pii": _result("ESCALATE"), "content": _result("FLAG")}
    assert floor_from_agent_results(results) == "FLAG"


def test_floor_from_agent_results_all_escalated_falls_back_to_allow():
    results = {"pii": _result("ESCALATE"), "content": _result("ESCALATE")}
    assert floor_from_agent_results(results) == "ALLOW"


def test_floor_from_agent_results_feeding_straight_into_decide_never_raises(engine):
    """The exact composition the Supervisor performs: compute a floor from
    agent outcomes, including an all-escalated set, then feed it straight
    into `decide()` — this must not be the `KeyError` `test_the_supervisor_*`
    integration tests would only catch indirectly."""
    all_escalated = {"pii": _result("ESCALATE"), "content": _result("ESCALATE")}
    floor = floor_from_agent_results(all_escalated)
    d = engine.decide("BLOCK", has_findings=True, policy_action=floor.lower())
    assert d.final_action == "BLOCK"
