"""The autonomous grounding agent.

`groundedness_check.classifier` is already stubbed to `None` for every test
in this suite by `tests/conftest.py`'s autouse `no_local_models` fixture.
"""

from __future__ import annotations

import time

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.capabilities import CapabilityDenied
from backend.guardrails.agents.grounding_agent import GroundingAgent
from backend.guardrails.agents.grounding_tools import (
    GROUNDING_AGENT_TOOLS, GROUNDING_TOOL_NAMES, ToolNotAllowed, call as call_tool,
)
from tests.conftest import REPO

CONTEXT = ["A trade licence renewed after its expiry attracts a 25 percent late "
          "surcharge. A licence more than 180 days past expiry is treated as lapsed."]


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedGroundingLLM:
    def __init__(self, plans=None, decisions=None,
                grounding_model_plans=None, grounding_model_decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        # `GroundingModelAgent`'s own nested PLAN/DECIDE — `check_local_entailment`
        # delegates to it now rather than calling the tool flat. Left
        # unscripted, it falls back to an honest "nothing found" default
        # rather than raising on an unrecognized shape.
        self.grounding_model_plan_script = list(grounding_model_plans or [])
        self.grounding_model_decision_script = list(grounding_model_decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if "needs_grounding_review" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "grounding_verdict" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        if "needs_local_entailment" in props:
            if self.grounding_model_plan_script:
                return self.grounding_model_plan_script.pop(0)
            return {"needs_local_entailment": False, "tools": [], "more_evidence_needed": False,
                    "rationale": "default stub — no script left"}
        if "local_entailment_verdict" in props:
            if self.grounding_model_decision_script:
                return self.grounding_model_decision_script.pop(0)
            return {"local_entailment_verdict": "ALLOW", "confidence": 1.0,
                    "rationale": "default stub — no script left", "findings": []}
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(tools, more=False, rationale="plan"):
    return {"needs_grounding_review": True, "tools": tools, "more_evidence_needed": more,
            "rationale": rationale}


def no_plan(rationale="no checkable claims"):
    return {"needs_grounding_review": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(verdict, confidence=0.9, evidence_summary="stub decision", findings=None):
    return {"grounding_verdict": verdict, "confidence": confidence,
            "evidence_summary": evidence_summary, "findings": findings or []}


def finding(entity, risk="high", confidence=0.9, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


# ── domain cases ───────────────────────────────────────────────────────
def test_fully_grounded_answer(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("ALLOW", evidence_summary="both claims match the context")])
    result = GroundingAgent(llm, engine).run(
        "The late surcharge is 25 percent, and a licence lapses after 180 days.",
        question="what is the late surcharge", chunks=CONTEXT, owner="citizen")
    assert result.decision.action == "ALLOW"


def test_partially_grounded_answer_flags(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("FLAG", evidence_summary=
                            "the surcharge figure is supported; the processing time is not",
                            findings=[finding("processing takes 10 working days", risk="medium")])])
    result = GroundingAgent(llm, engine).run(
        "The late surcharge is 25 percent. Processing takes 10 working days.",
        question="what is the late surcharge", chunks=CONTEXT, owner="citizen")
    assert result.decision.action == "FLAG"


def test_unsupported_answer_blocks(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("BLOCK", findings=[
            finding("the surcharge is 50 percent", risk="high")])])
    result = GroundingAgent(llm, engine).run(
        "The late surcharge is 50 percent.",
        question="what is the late surcharge", chunks=CONTEXT, owner="citizen")
    assert result.decision.action == "BLOCK"


def test_contradictory_source_blocks(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("BLOCK", evidence_summary=
                            "the context says 180 days; the answer says 90 — a direct "
                            "contradiction, not merely an absent fact",
                            findings=[finding("lapses after 90 days", risk="critical")])])
    result = GroundingAgent(llm, engine).run(
        "A licence lapses after 90 days.",
        question="when does a licence lapse", chunks=CONTEXT, owner="citizen")
    assert result.decision.action == "BLOCK"


def test_empty_retrieval_is_an_architectural_no_op(engine):
    """No chunks — allowed without a single judge call, exactly the rail's
    own no-op, not a decision this agent's PLAN step made."""
    llm = ScriptedGroundingLLM()
    result = GroundingAgent(llm, engine).run(
        "I'm not sure — nothing in our records covers that.",
        question="what is the fee for something obscure", chunks=[], owner="citizen")
    assert result.decision.action == "ALLOW"
    assert llm.plan_calls == 0
    assert llm.decision_calls == 0
    assert any("no retrieved context" in t.summary for t in result.trace)


def test_a_long_answer_with_multiple_claims(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("ALLOW", findings=[
            finding("25 percent surcharge", risk="low"),
            finding("180 day lapse window", risk="low"),
        ])])
    answer = ("The late surcharge is 25 percent of the standard fee. It applies from "
             "the day after expiry. A licence more than 180 days past expiry is "
             "treated as lapsed and cannot be renewed.")
    result = GroundingAgent(llm, engine).run(
        answer, question="tell me about late renewal", chunks=CONTEXT, owner="citizen")
    assert len(result.decision.findings) == 2


def test_the_decision_is_genuinely_the_models_not_a_hardcoded_rule(engine):
    same_answer = "The late surcharge is 50 percent."
    llm_block = ScriptedGroundingLLM(plans=[full_plan(["extract_claims"])],
                                     decisions=[decision("BLOCK")])
    r1 = GroundingAgent(llm_block, engine).run(
        same_answer, chunks=CONTEXT, owner="citizen")
    assert r1.decision.action == "BLOCK"

    llm_flag = ScriptedGroundingLLM(plans=[full_plan(["extract_claims"])],
                                    decisions=[decision("FLAG", evidence_summary=
                                                        "possibly a different fee tier, worth review")])
    r2 = GroundingAgent(llm_flag, engine).run(
        same_answer, chunks=CONTEXT, owner="citizen")
    assert r2.decision.action == "FLAG"


# ── measurements the increment asks to surface ────────────────────────
def test_claim_extraction_tool_reports_a_count(engine):
    res = call_tool("extract_claims",
                    {"answer": "The standard renewal fee is 1,200 rupees for this licence "
                               "class. Payment is due within 30 working days of the notice."},
                    engine, "call_x")
    assert res.status == "ok"
    assert res.result["claim_count"] == 2
    assert len(res.result["claims"]) == 2


def test_tool_calls_carry_duration_for_latency_measurement(engine):
    llm = ScriptedGroundingLLM(plans=[full_plan(["extract_claims"])],
                               decisions=[decision("ALLOW")])
    result = GroundingAgent(llm, engine).run(
        "The late surcharge is 25 percent.", chunks=CONTEXT, owner="citizen")
    assert all(c.duration_ms >= 0 for c in result.tool_calls)
    assert result.duration_ms >= 0


# ── tool boundary ─────────────────────────────────────────────────────
def test_unknown_tool_raises_tool_not_allowed(engine):
    with pytest.raises(ToolNotAllowed):
        call_tool("shell_exec", {}, engine, "call_x")


@pytest.mark.parametrize("hostile", ["__import__", "exec", "modify_policy"])
def test_malicious_tool_names_are_rejected(engine, hostile):
    with pytest.raises(ToolNotAllowed):
        call_tool(hostile, {}, engine, "call_x")


def test_the_registry_has_no_dynamic_dispatch():
    assert set(GROUNDING_AGENT_TOOLS) == set(GROUNDING_TOOL_NAMES) == {
        "extract_claims", "check_local_entailment", "get_grounding_policy"}


# ── malformed output, bounded loop, capability boundary ──────────────
def test_malformed_decision_output_escalates(engine):
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims"])],
        decisions=[{"grounding_verdict": "NUKE", "confidence": 6.0,
                   "evidence_summary": "x", "findings": []}])
    result = GroundingAgent(llm, engine).run("some answer", chunks=CONTEXT, owner="citizen")
    assert result.status == "escalated"


def test_max_tool_calls_is_enforced(engine):
    llm = ScriptedGroundingLLM(plans=[full_plan(
        ["extract_claims", "check_local_entailment", "get_grounding_policy"])])
    result = GroundingAgent(llm, engine, max_tool_calls=1).run(
        "some answer", chunks=CONTEXT, owner="citizen")
    assert result.status == "escalated"


def test_max_iterations_is_enforced(engine):
    llm = ScriptedGroundingLLM(plans=[full_plan(["extract_claims"], more=True)] * 10)
    result = GroundingAgent(llm, engine, max_iterations=2, max_tool_calls=20).run(
        "some answer", chunks=CONTEXT, owner="citizen")
    assert result.status == "escalated"
    assert "iteration" in result.escalation_reason


def test_timeout_is_checked_after_a_slow_plan_call_returns(engine):
    class SlowLLM(ScriptedGroundingLLM):
        def judge(self, *a, **k):
            time.sleep(0.05)
            return super().judge(*a, **k)

    llm = SlowLLM(plans=[full_plan(["extract_claims"])], decisions=[decision("ALLOW")])
    result = GroundingAgent(llm, engine, timeout_s=0.01).run(
        "some answer", chunks=CONTEXT, owner="citizen")
    assert result.status == "escalated"


def test_forbidden_capabilities_are_denied(engine):
    agent = GroundingAgent(ScriptedGroundingLLM(), engine)
    with pytest.raises(CapabilityDenied):
        agent.capabilities.request("modify_policy")


def test_a_complete_trace_is_produced(engine):
    llm = ScriptedGroundingLLM(plans=[full_plan(["extract_claims", "check_local_entailment"])],
                               decisions=[decision("BLOCK", findings=[finding("bad claim")])])
    result = GroundingAgent(llm, engine).run(
        "some answer", chunks=CONTEXT, owner="citizen")
    phases = [t.phase for t in result.trace]
    for expected in ("PLAN", "EXECUTE", "EVALUATE", "DECIDE", "ACT"):
        assert expected in phases


# ── a later `needs_grounding_review=false` must not discard evidence ────
def test_a_first_round_irrelevant_request_completes_without_decide(engine):
    """(A)"""
    llm = ScriptedGroundingLLM(plans=[no_plan("a purely conversational reply")])
    result = GroundingAgent(llm, engine).run(
        "Happy to help with that.", chunks=CONTEXT, owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_first_round_relevant_request_runs_tools_then_decides(engine):
    """(B)"""
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"])],
        decisions=[decision("BLOCK", findings=[finding("bad claim")])])
    result = GroundingAgent(llm, engine).run("some answer", chunks=CONTEXT, owner="citizen")
    assert result.tool_calls
    assert llm.decision_calls == 1
    assert result.decision.action == "BLOCK"


def test_multi_round_evidence_survives_a_later_needs_grounding_review_false(engine):
    """(C, F)"""
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"], more=True,
                         rationale="claims extracted, checking entailment next"),
              no_plan(rationale="entailment checked — enough to decide")],
        decisions=[decision("FLAG", findings=[finding("one uncertain claim", risk="medium")])])
    result = GroundingAgent(llm, engine).run("some answer", chunks=CONTEXT, owner="citizen")

    assert llm.plan_calls == 2
    assert llm.plan_script == []
    assert result.tool_calls, "the round-1 evidence must not be discarded"
    assert llm.decision_calls == 1
    assert result.decision.action == "FLAG"
    assert result.policy_decision is not None
    assert result.outcome is not None


def test_policy_engine_and_capabilities_are_not_bypassed_by_a_later_plan(engine):
    """(E)"""
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"], more=True), no_plan()],
        decisions=[decision("BLOCK", findings=[finding("bad claim")])])
    agent = GroundingAgent(llm, engine)
    policy_calls, capability_calls = [], []
    real_decide, real_execute = agent.policy_engine.decide, agent.capabilities.execute
    agent.policy_engine.decide = lambda *a, **k: (policy_calls.append(1), real_decide(*a, **k))[1]
    agent.capabilities.execute = lambda *a, **k: (capability_calls.append(1), real_execute(*a, **k))[1]

    agent.run("some answer", chunks=CONTEXT, owner="citizen")

    assert policy_calls
    assert capability_calls


def test_no_allow_with_null_outcome_once_evidence_was_gathered(engine):
    """(G)"""
    llm = ScriptedGroundingLLM(
        plans=[full_plan(["extract_claims", "check_local_entailment"], more=True), no_plan()],
        decisions=[decision("ALLOW", evidence_summary="every claim is supported after all")])
    result = GroundingAgent(llm, engine).run("some answer", chunks=CONTEXT, owner="citizen")

    assert result.tool_calls
    if result.decision.action == "ALLOW":
        assert result.outcome is not None
        assert result.policy_decision is not None
