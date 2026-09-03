"""`NERAgent` — the nested agent `PIIAgent` now delegates `detect_pii_presidio`
to, instead of calling the tool flat.

Same two standing requirements every agent test file in this package checks:

    genuine autonomy   the decision comes from a scripted judge call, keyed
                       by schema shape. Scripting a different answer to the
                       same input must produce a different result.

    hard boundaries    a tool name outside this agent's one-tool allowlist
                       fails in Python before any function runs; the
                       capability layer denies every forbidden name the same
                       way it does for every other agent, because it is the
                       same shared capability layer, not a duplicate.

Presidio's real model is expensive to load — stubbed for every test here the
same way `test_pii_agent.py` stubs it.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.ner_agent import NER_TOOL_NAMES, NERAgent
from backend.guardrails.agents.tools import ToolNotAllowed
from backend.guardrails.rails import presidio_ner
from tests.conftest import REPO


@pytest.fixture(autouse=True)
def no_presidio_model(monkeypatch):
    monkeypatch.setattr(presidio_ner, "find", lambda *a, **k: [])
    monkeypatch.setattr(presidio_ner, "available", lambda: True)
    yield


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedNERLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if "needs_ner_scan" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "ner_verdict" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(more=False, rationale="plan"):
    return {"needs_ner_scan": True, "tools": ["detect_pii_presidio"],
            "more_evidence_needed": more, "rationale": rationale}


def no_plan(rationale="nothing name- or address-shaped here"):
    return {"needs_ner_scan": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(verdict, confidence=0.9, rationale="stub decision", findings=None):
    return {"ner_verdict": verdict, "confidence": confidence, "rationale": rationale,
            "findings": findings or []}


def finding(entity="PERSON", risk="medium", confidence=0.85, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


def test_plain_text_needs_no_scan(engine):
    llm = ScriptedNERLLM(plans=[no_plan()])
    result = NERAgent(llm, engine).run("opening hours are 9 to 5", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_name_reaches_the_decision_call(engine, monkeypatch):
    monkeypatch.setattr(presidio_ner, "find",
                        lambda *a, **k: [{"kind": "PERSON", "start": 0, "end": 11,
                                          "confidence": 0.85}])
    llm = ScriptedNERLLM(plans=[full_plan()],
                         decisions=[decision("MASK", findings=[finding("PERSON")])])
    result = NERAgent(llm, engine).run("Meera Balan called about her claim", owner="citizen")
    assert result.decision.action == "MASK"
    call = next(c for c in result.tool_calls if c.tool == "detect_pii_presidio")
    assert call.status == "ok"
    assert call.result["findings"][0]["kind"] == "PERSON"


def test_the_decision_is_genuinely_the_models(engine, monkeypatch):
    """Same input, a different scripted verdict — proves DECIDE is a real
    judge call read by this test, not a hardcoded `if found: MASK`."""
    monkeypatch.setattr(presidio_ner, "find",
                        lambda *a, **k: [{"kind": "PERSON", "start": 0, "end": 11,
                                          "confidence": 0.3}])
    llm = ScriptedNERLLM(
        plans=[full_plan()],
        decisions=[decision("ESCALATE", rationale="confidence too low to act on",
                            findings=[finding("PERSON", confidence=0.3)])])
    result = NERAgent(llm, engine).run("Meera Balan called about her claim", owner="citizen")
    assert result.decision.action == "ESCALATE"


def test_a_tool_name_outside_the_allowlist_is_rejected(engine):
    assert NER_TOOL_NAMES == ("detect_pii_presidio",)
    llm = ScriptedNERLLM(plans=[{"needs_ner_scan": True, "tools": ["detect_pii_entities"],
                                "more_evidence_needed": False, "rationale": "plan"}])
    with pytest.raises(ToolNotAllowed):
        NERAgent(llm, engine).run("some text", owner="citizen")


def test_tool_call_budget_exhaustion_escalates(engine):
    llm = ScriptedNERLLM(plans=[full_plan()] * 10)
    result = NERAgent(llm, engine, max_tool_calls=0).run("Meera Balan", owner="citizen")
    assert result.status == "escalated"
    assert result.decision.action == "ESCALATE"


def test_no_policy_floor_the_agents_own_decision_is_final(engine):
    """Unlike every other agent in this package, `NERAgent` has no
    `PolicyEngine` floor of its own — `pii.action.user_prompt` here must NOT
    override an ALLOW recommendation even when a finding exists, because
    `PIIAgent`'s own POLICY step (not exercised by this test) is what
    actually enforces the floor for the request."""
    engine.policy.values["pii.action.user_prompt"] = "mask"
    llm = ScriptedNERLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("PERSON")])])
    result = NERAgent(llm, engine).run("Meera Balan", owner="citizen")
    assert not result.policy_decision.overridden
    assert result.policy_decision.final_action == "ALLOW"


def test_agent_nested_model_floor_on_restores_the_floor(engine):
    """`agent.nested_model_floor="on"` is the escape hatch back to the
    floor-governed behaviour every other agent in this package still has."""
    engine.policy.values["agent.nested_model_floor"] = "on"
    engine.policy.values["pii.action.user_prompt"] = "mask"
    llm = ScriptedNERLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("PERSON")])])
    result = NERAgent(llm, engine).run("Meera Balan", owner="citizen")
    assert result.policy_decision.overridden
    assert result.policy_decision.final_action == "MASK"
