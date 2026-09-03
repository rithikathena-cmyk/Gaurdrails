"""`GroundingModelAgent` — the nested agent `GroundingAgent` now delegates
`check_local_entailment` to, instead of calling the tool flat.

`groundedness_check.classifier` is stubbed to `None` for every test in this
suite by `tests/conftest.py`'s autouse `no_local_models` fixture; tests that
need a specific score monkeypatch `groundedness_check.consistency` directly.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.grounding_model_agent import (
    GROUNDING_MODEL_TOOL_NAMES, GroundingModelAgent,
)
from backend.guardrails.agents.grounding_tools import ToolNotAllowed
from backend.guardrails.rails import groundedness_check
from tests.conftest import REPO

CONTEXT = ["A trade licence renewed after its expiry attracts a 25 percent late "
          "surcharge. A licence more than 180 days past expiry is treated as lapsed."]


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedGroundingModelLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if "needs_local_entailment" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "local_entailment_verdict" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(more=False, rationale="plan"):
    return {"needs_local_entailment": True, "tools": ["check_local_entailment"],
            "more_evidence_needed": more, "rationale": rationale}


def no_plan(rationale="no checkable claims"):
    return {"needs_local_entailment": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(verdict, confidence=0.9, rationale="stub decision", findings=None):
    return {"local_entailment_verdict": verdict, "confidence": confidence,
            "rationale": rationale, "findings": findings or []}


def finding(entity="grounded", risk="low", confidence=0.9, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


def test_no_retrieved_context_is_an_architectural_no_op(engine):
    """Checked before PLAN ever runs — mirrors `GroundingRail`'s and
    `GroundingAgent`'s own unconditional shortcut for the same case."""
    llm = ScriptedGroundingModelLLM(plans=[full_plan()])
    result = GroundingModelAgent(llm, engine).run("A generic reply.", chunks=[])
    assert result.decision.action == "ALLOW"
    assert llm.plan_calls == 0


def test_a_purely_conversational_reply_needs_no_check(engine):
    llm = ScriptedGroundingModelLLM(plans=[no_plan()])
    result = GroundingModelAgent(llm, engine).run(
        "Happy to help with anything else.", chunks=CONTEXT)
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []


def test_an_unsupported_claim_reaches_the_decision_call(engine, monkeypatch):
    monkeypatch.setattr(groundedness_check, "consistency",
                        lambda answer, chunks: {"consistency": 0.2, "supported": 0,
                                                 "claims": 1, "unsupported": ["a fabricated fee"]})
    monkeypatch.setattr(groundedness_check, "claims", lambda answer: ["a fabricated fee"])
    llm = ScriptedGroundingModelLLM(
        plans=[full_plan()],
        decisions=[decision("BLOCK", findings=[finding("a fabricated fee", risk="high")])])
    result = GroundingModelAgent(llm, engine).run(
        "The late fee is exactly 500 rupees.", question="What is the late fee?",
        chunks=CONTEXT)
    assert result.decision.action == "BLOCK"
    call = next(c for c in result.tool_calls if c.tool == "check_local_entailment")
    assert call.result["consistency"] == 0.2


def test_the_decision_is_genuinely_the_models(engine, monkeypatch):
    """Same low local score, a different scripted verdict — proves DECIDE is
    a real judge call the test controls, not a hardcoded threshold."""
    monkeypatch.setattr(groundedness_check, "consistency",
                        lambda answer, chunks: {"consistency": 0.4, "supported": 0,
                                                 "claims": 1, "unsupported": ["a figure"]})
    monkeypatch.setattr(groundedness_check, "claims", lambda answer: ["a figure"])
    llm = ScriptedGroundingModelLLM(
        plans=[full_plan()],
        decisions=[decision("FLAG", rationale="mostly grounded, one uncertain figure")])
    result = GroundingModelAgent(llm, engine).run(
        "The fee is roughly 500 rupees.", question="What is the fee?", chunks=CONTEXT)
    assert result.decision.action == "FLAG"


def test_a_tool_name_outside_the_allowlist_is_rejected(engine):
    assert GROUNDING_MODEL_TOOL_NAMES == ("check_local_entailment",)
    llm = ScriptedGroundingModelLLM(
        plans=[{"needs_local_entailment": True, "tools": ["extract_claims"],
                "more_evidence_needed": False, "rationale": "plan"}])
    with pytest.raises(ToolNotAllowed):
        GroundingModelAgent(llm, engine).run("An answer.", chunks=CONTEXT)


def test_tool_call_budget_exhaustion_escalates(engine):
    llm = ScriptedGroundingModelLLM(plans=[full_plan()] * 10)
    result = GroundingModelAgent(llm, engine, max_tool_calls=0).run(
        "An answer.", chunks=CONTEXT)
    assert result.status == "escalated"
    assert result.decision.action == "ESCALATE"


def test_no_policy_floor_the_agents_own_decision_is_final(engine, monkeypatch):
    """Unlike every other agent in this package, `GroundingModelAgent` has no
    `PolicyEngine` floor of its own — `grounding.action_on_fail` here must
    NOT override an ALLOW recommendation, because `GroundingAgent`'s own
    POLICY step (not exercised by this test) is what actually enforces the
    floor for the request."""
    engine.policy.values["grounding.action_on_fail"] = "block"
    monkeypatch.setattr(groundedness_check, "consistency",
                        lambda answer, chunks: {"consistency": 0.2, "supported": 0,
                                                 "claims": 1, "unsupported": ["a fabricated fee"]})
    monkeypatch.setattr(groundedness_check, "claims", lambda answer: ["a fabricated fee"])
    llm = ScriptedGroundingModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("a fabricated fee")])])
    result = GroundingModelAgent(llm, engine).run(
        "The fee is 500 rupees.", question="What is the fee?", chunks=CONTEXT)
    assert not result.policy_decision.overridden
    assert result.policy_decision.final_action == "ALLOW"


def test_agent_nested_model_floor_on_restores_the_floor(engine, monkeypatch):
    """`agent.nested_model_floor="on"` is the escape hatch back to the
    floor-governed behaviour every other agent in this package still has."""
    engine.policy.values["agent.nested_model_floor"] = "on"
    engine.policy.values["grounding.action_on_fail"] = "block"
    monkeypatch.setattr(groundedness_check, "consistency",
                        lambda answer, chunks: {"consistency": 0.2, "supported": 0,
                                                 "claims": 1, "unsupported": ["a fabricated fee"]})
    monkeypatch.setattr(groundedness_check, "claims", lambda answer: ["a fabricated fee"])
    llm = ScriptedGroundingModelLLM(
        plans=[full_plan()],
        decisions=[decision("ALLOW", findings=[finding("a fabricated fee")])])
    result = GroundingModelAgent(llm, engine).run(
        "The fee is 500 rupees.", question="What is the fee?", chunks=CONTEXT)
    assert result.policy_decision.overridden
    assert result.policy_decision.final_action == "BLOCK"
