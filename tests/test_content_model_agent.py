"""`ContentModelAgent` — the nested agent `ContentSafetyAgent` now delegates
`score_content_categories` to, instead of calling the tool flat.

`toxicity_check.classifier` is stubbed to `None` for every test in this
suite by `tests/conftest.py`'s autouse `no_local_models` fixture; tests that
need a specific score monkeypatch `toxicity_check.score` directly, the same
way `test_local_rails.py` does for the production rail's own local layer.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.content_model_agent import (
    CONTENT_MODEL_TOOL_NAMES, ContentModelAgent,
)
from backend.guardrails.agents.content_tools import ToolNotAllowed
from backend.guardrails.rails import toxicity_check
from tests.conftest import REPO


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedContentModelLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if "needs_local_score" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "local_content_verdict" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(more=False, rationale="plan"):
    return {"needs_local_score": True, "tools": ["score_content_categories"],
            "more_evidence_needed": more, "rationale": rationale}


def no_plan(rationale="nothing content-relevant here"):
    return {"needs_local_score": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(verdict, confidence=0.9, rationale="stub decision", findings=None):
    return {"local_content_verdict": verdict, "confidence": confidence, "rationale": rationale,
            "findings": findings or []}


def finding(entity="violence", risk="high", confidence=0.9, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


def test_plain_text_needs_no_scoring(engine):
    llm = ScriptedContentModelLLM(plans=[no_plan()])
    result = ContentModelAgent(llm, engine).run("what are your opening hours", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_high_score_reaches_the_decision_call(engine, monkeypatch):
    monkeypatch.setattr(toxicity_check, "score",
                        lambda text: {"hate": 0.1, "violence": 0.92, "insults": 0.05})
    llm = ScriptedContentModelLLM(
        plans=[full_plan()],
        decisions=[decision("BLOCK", findings=[finding("violence")])])
    result = ContentModelAgent(llm, engine).run("a violent threat", owner="citizen")
    assert result.decision.action == "BLOCK"
    call = next(c for c in result.tool_calls if c.tool == "score_content_categories")
    assert call.status == "ok"
    assert call.result["scores"]["violence"] == 0.92


def test_the_decision_is_genuinely_the_models(engine, monkeypatch):
    """Same score, a different scripted verdict — this agent's DECIDE is a
    real judge call the test controls, not a hardcoded threshold check."""
    monkeypatch.setattr(toxicity_check, "score",
                        lambda text: {"hate": 0.1, "violence": 0.92, "insults": 0.05})
    llm = ScriptedContentModelLLM(
        plans=[full_plan()],
        decisions=[decision("FLAG", rationale="describes distress, not a threat",
                            findings=[finding("violence", risk="low")])])
    result = ContentModelAgent(llm, engine).run("this is killing me", owner="citizen")
    assert result.decision.action == "FLAG"


def test_a_tool_name_outside_the_allowlist_is_rejected(engine):
    assert CONTENT_MODEL_TOOL_NAMES == ("score_content_categories",)
    llm = ScriptedContentModelLLM(
        plans=[{"needs_local_score": True, "tools": ["get_content_policy"],
                "more_evidence_needed": False, "rationale": "plan"}])
    with pytest.raises(ToolNotAllowed):
        ContentModelAgent(llm, engine).run("some text", owner="citizen")


def test_tool_call_budget_exhaustion_escalates(engine):
    llm = ScriptedContentModelLLM(plans=[full_plan()] * 10)
    result = ContentModelAgent(llm, engine, max_tool_calls=0).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert result.decision.action == "ESCALATE"


def test_no_policy_floor_the_agents_own_decision_is_final(engine, monkeypatch):
    """Unlike every other agent in this package, `ContentModelAgent` has no
    `PolicyEngine` floor of its own — `content.action.user_prompt` here must
    NOT override an ALLOW recommendation, because `ContentSafetyAgent`'s own
    POLICY step (not exercised by this test) is what actually enforces the
    floor for the request."""
    engine.policy.values["content.action.user_prompt"] = "block"
    monkeypatch.setattr(toxicity_check, "score",
                        lambda text: {"hate": 0.1, "violence": 0.92, "insults": 0.05})
    llm = ScriptedContentModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("violence")])])
    result = ContentModelAgent(llm, engine).run("a violent threat", owner="citizen")
    assert not result.policy_decision.overridden
    assert result.policy_decision.final_action == "ALLOW"


def test_agent_nested_model_floor_on_restores_the_floor(engine, monkeypatch):
    """`agent.nested_model_floor="on"` is the escape hatch back to the
    floor-governed behaviour every other agent in this package still has."""
    engine.policy.values["agent.nested_model_floor"] = "on"
    engine.policy.values["content.action.user_prompt"] = "block"
    monkeypatch.setattr(toxicity_check, "score",
                        lambda text: {"hate": 0.1, "violence": 0.92, "insults": 0.05})
    llm = ScriptedContentModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("violence")])])
    result = ContentModelAgent(llm, engine).run("a violent threat", owner="citizen")
    assert result.policy_decision.overridden
    assert result.policy_decision.final_action == "BLOCK"
