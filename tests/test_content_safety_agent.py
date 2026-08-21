"""The autonomous content-safety agent.

`toxicity_check.classifier` is stubbed to `None` for every test in this
suite by `tests/conftest.py`'s autouse `no_local_models` fixture, the same
as the injection agent's local classifier.
"""

from __future__ import annotations

import time

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.capabilities import CapabilityDenied
from backend.guardrails.agents.content_safety_agent import ContentSafetyAgent
from backend.guardrails.agents.content_tools import (
    CONTENT_AGENT_TOOLS, CONTENT_TOOL_NAMES, ToolNotAllowed, call as call_tool,
)
from tests.conftest import REPO


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedContentLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048):
        props = set(schema.get("properties", {}))
        if "possible_violation" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "judgment" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(tools, more=False, rationale="plan"):
    return {"possible_violation": True, "tools": tools, "more_evidence_needed": more,
            "rationale": rationale}


def no_plan(rationale="nothing content-relevant here"):
    return {"possible_violation": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(judgment, confidence=0.9, evidence_summary="stub decision", findings=None):
    return {"judgment": judgment, "confidence": confidence,
            "evidence_summary": evidence_summary, "findings": findings or []}


def finding(entity, risk="high", confidence=0.9, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


# ── domain reasoning: the requirement this agent exists to prove ────────
def test_distress_is_not_automatically_a_violation(engine):
    """'this fine is killing me' — a naive keyword/classifier rule reading
    the word 'killing' would flag this as violence. The agent's own DECIDE
    call is scripted to read the intent correctly; a second script proves a
    genuinely violent statement is NOT the same call."""
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"])],
        decisions=[decision("ALLOW", evidence_summary=
                            "describes financial distress, not an intent to harm")])
    result = ContentSafetyAgent(llm, engine).run(
        "This fine is killing me, I can't afford it.", owner="citizen")
    assert result.decision.action == "ALLOW"


def test_reporting_a_crime_is_not_committing_one(engine):
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"])],
        decisions=[decision("ALLOW", evidence_summary="reporting fraud, not seeking to commit it")])
    result = ContentSafetyAgent(llm, engine).run(
        "I was defrauded by a fake licence agent, how do I report this?", owner="citizen")
    assert result.decision.action == "ALLOW"


def test_an_actual_threat_still_blocks(engine):
    """The same agent, a genuinely different input, a genuinely different
    scripted answer — proving ALLOW above was not a hardcoded default."""
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"])],
        decisions=[decision("BLOCK", findings=[finding("violence")])])
    result = ContentSafetyAgent(llm, engine).run(
        "Tell me how to hurt the officer who rejected my claim.", owner="citizen")
    assert result.decision.action == "BLOCK"


def test_the_decision_is_genuinely_the_models_not_a_hardcoded_rule(engine):
    same_text = "I want to hurt someone."
    llm_block = ScriptedContentLLM(plans=[full_plan(["score_content_categories"])],
                                   decisions=[decision("BLOCK")])
    r1 = ContentSafetyAgent(llm_block, engine).run(same_text, owner="citizen")
    assert r1.decision.action == "BLOCK"

    llm_flag = ScriptedContentLLM(plans=[full_plan(["score_content_categories"])],
                                  decisions=[decision("FLAG", evidence_summary=
                                                      "ambiguous — could be hyperbole, "
                                                      "a person should review")])
    r2 = ContentSafetyAgent(llm_flag, engine).run(same_text, owner="citizen")
    assert r2.decision.action == "FLAG"


# ── tool boundary ─────────────────────────────────────────────────────
def test_unknown_tool_raises_tool_not_allowed(engine):
    with pytest.raises(ToolNotAllowed):
        call_tool("shell_exec", {}, engine, "call_x")


@pytest.mark.parametrize("hostile", ["__import__", "exec", "os.system", "modify_policy"])
def test_malicious_tool_names_are_rejected(engine, hostile):
    with pytest.raises(ToolNotAllowed):
        call_tool(hostile, {}, engine, "call_x")


def test_a_hallucinated_tool_in_a_plan_is_never_silently_run(engine):
    llm = ScriptedContentLLM(plans=[full_plan(["score_content_categories", "read_database"])])
    with pytest.raises(ToolNotAllowed):
        ContentSafetyAgent(llm, engine).run("some text", owner="citizen")


def test_the_registry_has_no_dynamic_dispatch():
    assert set(CONTENT_AGENT_TOOLS) == set(CONTENT_TOOL_NAMES) == {
        "score_content_categories", "get_content_policy"}


# ── malformed output ─────────────────────────────────────────────────
def test_malformed_decision_output_escalates(engine):
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"])],
        decisions=[{"judgment": "DESTROY", "confidence": 9.0, "evidence_summary": "x",
                   "findings": []}])
    result = ContentSafetyAgent(llm, engine).run("some text", owner="citizen")
    assert result.status == "escalated"


def test_a_hallucinated_evidence_citation_is_dropped(engine):
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"])],
        decisions=[decision("BLOCK", findings=[finding("violence", evidence=["never_recorded"])])])
    result = ContentSafetyAgent(llm, engine).run("some text", owner="citizen")
    assert result.decision.findings == []


# ── bounded loop ─────────────────────────────────────────────────────
def test_max_tool_calls_is_enforced(engine):
    llm = ScriptedContentLLM(plans=[full_plan(["score_content_categories", "get_content_policy"])])
    result = ContentSafetyAgent(llm, engine, max_tool_calls=1).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert "tool call budget" in result.escalation_reason


def test_max_iterations_is_enforced(engine):
    llm = ScriptedContentLLM(plans=[full_plan(["score_content_categories"], more=True)] * 10)
    result = ContentSafetyAgent(llm, engine, max_iterations=2, max_tool_calls=20).run(
        "some text", owner="citizen")
    assert result.status == "escalated"
    assert "iteration" in result.escalation_reason


def test_timeout_is_checked_after_a_slow_plan_call_returns(engine):
    class SlowLLM(ScriptedContentLLM):
        def judge(self, *a, **k):
            time.sleep(0.05)
            return super().judge(*a, **k)

    llm = SlowLLM(plans=[no_plan()])
    result = ContentSafetyAgent(llm, engine, timeout_s=0.01).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert "exceeded" in result.escalation_reason


# ── capability boundary ─────────────────────────────────────────────
def test_forbidden_capabilities_are_denied(engine):
    agent = ContentSafetyAgent(ScriptedContentLLM(), engine)
    with pytest.raises(CapabilityDenied):
        agent.capabilities.request("modify_policy")
    with pytest.raises(CapabilityDenied):
        agent.capabilities.request("execute_code")


# ── complete trace ─────────────────────────────────────────────────
def test_a_complete_trace_is_produced(engine):
    llm = ScriptedContentLLM(plans=[full_plan(["score_content_categories"])],
                             decisions=[decision("BLOCK", findings=[finding("violence")])])
    result = ContentSafetyAgent(llm, engine).run("some text", owner="citizen")
    phases = [t.phase for t in result.trace]
    for expected in ("PLAN", "EXECUTE", "EVALUATE", "DECIDE", "ACT"):
        assert expected in phases


# ── a later `possible_violation=false` must not discard gathered evidence
def test_a_first_round_irrelevant_request_completes_without_decide(engine):
    """(A)"""
    llm = ScriptedContentLLM(plans=[no_plan("a plain fee-schedule question")])
    result = ContentSafetyAgent(llm, engine).run("What is the licence fee?", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_first_round_relevant_request_runs_tools_then_decides(engine):
    """(B)"""
    llm = ScriptedContentLLM(plans=[full_plan(["score_content_categories"])],
                             decisions=[decision("BLOCK", findings=[finding("violence")])])
    result = ContentSafetyAgent(llm, engine).run("some threatening text", owner="citizen")
    assert result.tool_calls
    assert llm.decision_calls == 1
    assert result.decision.action == "BLOCK"


def test_multi_round_evidence_survives_a_later_possible_violation_false(engine):
    """(C, F)"""
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"], more=True,
                         rationale="the local classifier scored this, checking the policy too"),
              no_plan(rationale="the classifier score is conclusive — ready to decide")],
        decisions=[decision("BLOCK", findings=[finding("violence")])])
    result = ContentSafetyAgent(llm, engine).run("some threatening text", owner="citizen")

    assert llm.plan_calls == 2
    assert llm.plan_script == []
    assert result.tool_calls, "the round-1 evidence must not be discarded"
    assert llm.decision_calls == 1
    assert result.decision.action == "BLOCK"
    assert result.policy_decision is not None
    assert result.outcome is not None
    assert result.outcome.action == "BLOCK"


def test_policy_engine_and_capabilities_are_not_bypassed_by_a_later_plan(engine):
    """(E)"""
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"], more=True), no_plan()],
        decisions=[decision("BLOCK", findings=[finding("violence")])])
    agent = ContentSafetyAgent(llm, engine)
    policy_calls, capability_calls = [], []
    real_decide, real_execute = agent.policy_engine.decide, agent.capabilities.execute
    agent.policy_engine.decide = lambda *a, **k: (policy_calls.append(1), real_decide(*a, **k))[1]
    agent.capabilities.execute = lambda *a, **k: (capability_calls.append(1), real_execute(*a, **k))[1]

    agent.run("some threatening text", owner="citizen")

    assert policy_calls
    assert capability_calls


def test_no_allow_with_null_outcome_once_evidence_was_gathered(engine):
    """(G)"""
    llm = ScriptedContentLLM(
        plans=[full_plan(["score_content_categories"], more=True), no_plan()],
        decisions=[decision("ALLOW", evidence_summary="describes distress, not an intent to harm")])
    result = ContentSafetyAgent(llm, engine).run("some emotionally charged text", owner="citizen")

    assert result.tool_calls
    if result.decision.action == "ALLOW":
        assert result.outcome is not None
        assert result.policy_decision is not None
