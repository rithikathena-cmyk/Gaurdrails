"""`InjectionModelAgent` — the nested agent `PromptInjectionAgent` now
delegates `classify_injection` to, instead of calling the tool flat.

`deberta_injection_check.classifier` is stubbed to `None` for every test in
this suite by `tests/conftest.py`'s autouse `no_local_models` fixture; tests
that need a specific score monkeypatch `deberta_injection_check.score`
directly.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.injection_model_agent import (
    INJECTION_MODEL_TOOL_NAMES, InjectionModelAgent,
)
from backend.guardrails.agents.injection_tools import ToolNotAllowed
from backend.guardrails.rails import deberta_injection_check
from tests.conftest import REPO


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedInjectionModelLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if "needs_local_classification" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "local_injection_verdict" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(more=False, rationale="plan"):
    return {"needs_local_classification": True, "tools": ["classify_injection"],
            "more_evidence_needed": more, "rationale": rationale}


def no_plan(rationale="an ordinary question, nothing to classify"):
    return {"needs_local_classification": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(verdict, confidence=0.9, rationale="stub decision", findings=None):
    return {"local_injection_verdict": verdict, "confidence": confidence, "rationale": rationale,
            "findings": findings or []}


def finding(entity="local_classifier", risk="medium", confidence=0.8, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


def test_plain_text_needs_no_classification(engine):
    llm = ScriptedInjectionModelLLM(plans=[no_plan()])
    result = InjectionModelAgent(llm, engine).run("what documents do I need", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_high_score_reaches_the_decision_call(engine, monkeypatch):
    monkeypatch.setattr(deberta_injection_check, "score", lambda text: 0.93)
    monkeypatch.setattr(deberta_injection_check, "looks_like_a_meta_question",
                        lambda text: False)
    llm = ScriptedInjectionModelLLM(
        plans=[full_plan()],
        decisions=[decision("FLAG", findings=[finding()])])
    result = InjectionModelAgent(llm, engine).run(
        "some oddly phrased instruction-like text", owner="citizen")
    assert result.decision.action == "FLAG"
    call = next(c for c in result.tool_calls if c.tool == "classify_injection")
    assert call.result["local_score"] == 0.93


def test_a_meta_question_is_weighed_not_ignored(engine, monkeypatch):
    """Same high score, flagged as a meta-question — the decision call sees
    both facts and the script proves it's the one weighing them, not a
    hardcoded score >= threshold rule."""
    monkeypatch.setattr(deberta_injection_check, "score", lambda text: 0.93)
    monkeypatch.setattr(deberta_injection_check, "looks_like_a_meta_question",
                        lambda text: True)
    llm = ScriptedInjectionModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", rationale="classifier score is high but this is a "
                                              "legitimate meta-question about the service")])
    result = InjectionModelAgent(llm, engine).run(
        "why was my last request refused", owner="citizen")
    assert result.decision.action == "ALLOW"


def test_a_tool_name_outside_the_allowlist_is_rejected(engine):
    assert INJECTION_MODEL_TOOL_NAMES == ("classify_injection",)
    llm = ScriptedInjectionModelLLM(
        plans=[{"needs_local_classification": True, "tools": ["detect_injection_patterns"],
                "more_evidence_needed": False, "rationale": "plan"}])
    with pytest.raises(ToolNotAllowed):
        InjectionModelAgent(llm, engine).run("some text", owner="citizen")


def test_tool_call_budget_exhaustion_escalates(engine):
    llm = ScriptedInjectionModelLLM(plans=[full_plan()] * 10)
    result = InjectionModelAgent(llm, engine, max_tool_calls=0).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert result.decision.action == "ESCALATE"


def test_no_policy_floor_the_agents_own_decision_is_final(engine, monkeypatch):
    """Unlike every other agent in this package, `InjectionModelAgent` has no
    `PolicyEngine` floor of its own — `prompt_attack.action` here must NOT
    override an ALLOW recommendation, because `PromptInjectionAgent`'s own
    POLICY step (not exercised by this test) is what actually enforces the
    floor for the request."""
    engine.policy.values["prompt_attack.action"] = "block"
    monkeypatch.setattr(deberta_injection_check, "score", lambda text: 0.93)
    monkeypatch.setattr(deberta_injection_check, "looks_like_a_meta_question",
                        lambda text: False)
    llm = ScriptedInjectionModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding()])])
    result = InjectionModelAgent(llm, engine).run("some text", owner="citizen")
    assert not result.policy_decision.overridden
    assert result.policy_decision.final_action == "ALLOW"


def test_agent_nested_model_floor_on_restores_the_floor(engine, monkeypatch):
    """`agent.nested_model_floor="on"` is the escape hatch back to the
    floor-governed behaviour every other agent in this package still has."""
    engine.policy.values["agent.nested_model_floor"] = "on"
    engine.policy.values["prompt_attack.action"] = "block"
    monkeypatch.setattr(deberta_injection_check, "score", lambda text: 0.93)
    monkeypatch.setattr(deberta_injection_check, "looks_like_a_meta_question",
                        lambda text: False)
    llm = ScriptedInjectionModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding()])])
    result = InjectionModelAgent(llm, engine).run("some text", owner="citizen")
    assert result.policy_decision.overridden
    assert result.policy_decision.final_action == "BLOCK"
