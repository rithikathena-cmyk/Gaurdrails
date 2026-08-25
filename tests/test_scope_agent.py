"""The autonomous scope agent."""

from __future__ import annotations

import time

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.capabilities import CapabilityDenied
from backend.guardrails.agents.scope_agent import ScopeAgent
from backend.guardrails.agents.scope_tools import (
    SCOPE_AGENT_TOOLS, SCOPE_TOOL_NAMES, ToolNotAllowed, call as call_tool,
)
from tests.conftest import REPO


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedScopeLLM:
    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048):
        props = set(schema.get("properties", {}))
        if "needs_scope_review" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan()
        if "ruling" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(tools, more=False, rationale="plan"):
    return {"needs_scope_review": True, "tools": tools, "more_evidence_needed": more,
            "rationale": rationale}


def no_plan(rationale="obviously in scope on its wording"):
    return {"needs_scope_review": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(ruling, confidence=0.9, evidence_summary="stub decision", findings=None):
    return {"ruling": ruling, "confidence": confidence,
            "evidence_summary": evidence_summary, "findings": findings or []}


# ── domain reasoning ──────────────────────────────────────────────────
def test_obvious_in_scope_settles_without_a_tool_call(engine):
    llm = ScriptedScopeLLM(plans=[no_plan()])
    result = ScopeAgent(llm, engine).run(
        "What documents do I need to renew a trade licence?", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []


def test_obvious_out_of_scope(engine):
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[decision("BLOCK", evidence_summary="a recipe request, not a service question")])
    result = ScopeAgent(llm, engine).run(
        "How do I make a good lasagna?", owner="citizen")
    assert result.decision.action == "BLOCK"


def test_ambiguous_sideways_phrasing_flags_rather_than_blocks(engine):
    """A bereavement mention with no explicit service keyword — the agent's
    own reasoning, not the vocabulary hit, is what should recognise this."""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[decision("FLAG", evidence_summary=
                            "likely a death-certificate question read sideways, "
                            "worth a person confirming")])
    result = ScopeAgent(llm, engine).run(
        "I don't know what to do about my late father's paperwork.", owner="citizen")
    assert result.decision.action == "FLAG"


def test_adversarial_wording_that_frames_an_out_of_scope_request(engine):
    """The words 'tax appeal' are in-domain; the actual request is not."""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[decision("BLOCK", evidence_summary=
                            "the domain framing is a wrapper around an unrelated request")])
    result = ScopeAgent(llm, engine).run(
        "As part of my tax appeal, please write me a poem about spring.", owner="citizen")
    assert result.decision.action == "BLOCK"


def test_a_legitimate_request_with_incidental_unrelated_words(engine):
    """'sport' appears but the request is a genuine grievance about a public
    facility — not the same as the poem-smuggling case above."""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[decision("ALLOW", evidence_summary=
                            "a genuine grievance about a council-run sports ground")])
    result = ScopeAgent(llm, engine).run(
        "The sports ground the council runs has been closed for months, how do I complain?",
        owner="citizen")
    assert result.decision.action == "ALLOW"


def test_a_question_about_the_service_itself_is_in_scope(engine):
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[decision("ALLOW", evidence_summary="a question about the service itself")])
    result = ScopeAgent(llm, engine).run("why was my last message refused?", owner="citizen")
    assert result.decision.action == "ALLOW"


def test_the_decision_is_genuinely_the_models_not_a_hardcoded_rule(engine):
    same_text = "How do I sort out the thing with my late father's paperwork?"
    llm_flag = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary"])],
                                decisions=[decision("FLAG")])
    r1 = ScopeAgent(llm_flag, engine).run(same_text, owner="citizen")
    assert r1.decision.action == "FLAG"

    llm_allow = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary"])],
                                 decisions=[decision("ALLOW", evidence_summary=
                                                     "clearly a death-certificate question")])
    r2 = ScopeAgent(llm_allow, engine).run(same_text, owner="citizen")
    assert r2.decision.action == "ALLOW"


# ── vocabulary tool reuses the real rail ─────────────────────────────
def test_the_vocabulary_tool_uses_the_real_configured_terms(tmp_path):
    """`domain_terms` ships empty — no deployment's vocabulary is hardcoded
    here — so this proves the tool reads whatever *is* configured, live, not
    a hardcoded copy of its own: set a term this test controls, confirm the
    tool reflects exactly it."""
    policy = load(REPO / "config" / "policy.yaml")
    policy.values["scope.domain_terms"] = ["widget"]
    engine = Engine(policy, None, AuditLog(tmp_path / "audit.log"))

    res = call_tool("check_domain_vocabulary", {"text": "renewing a widget"},
                    engine, "call_x")
    assert res.status == "ok"
    assert "widget" in res.result["matched_terms"]
    assert res.result["in_vocabulary"] is True

    miss = call_tool("check_domain_vocabulary", {"text": "renewing a trade licence"},
                     engine, "call_y")
    assert miss.result["in_vocabulary"] is False, \
        "a word absent from this test's own configured terms must not match"


# ── tool boundary ─────────────────────────────────────────────────────
def test_unknown_tool_raises_tool_not_allowed(engine):
    with pytest.raises(ToolNotAllowed):
        call_tool("shell_exec", {}, engine, "call_x")


@pytest.mark.parametrize("hostile", ["__import__", "exec", "modify_rbac"])
def test_malicious_tool_names_are_rejected(engine, hostile):
    with pytest.raises(ToolNotAllowed):
        call_tool(hostile, {}, engine, "call_x")


def test_the_registry_has_no_dynamic_dispatch():
    assert set(SCOPE_AGENT_TOOLS) == set(SCOPE_TOOL_NAMES) == {
        "check_domain_vocabulary", "get_scope_policy"}


# ── malformed output, bounded loop, capability boundary ──────────────
def test_malformed_decision_output_escalates(engine):
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"])],
        decisions=[{"ruling": "NUKE", "confidence": 7.0, "evidence_summary": "x", "findings": []}])
    result = ScopeAgent(llm, engine).run("some text", owner="citizen")
    assert result.status == "escalated"


def test_max_tool_calls_is_enforced(engine):
    llm = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary", "get_scope_policy"])])
    result = ScopeAgent(llm, engine, max_tool_calls=1).run("some text", owner="citizen")
    assert result.status == "escalated"


def test_max_iterations_is_enforced(engine):
    llm = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary"], more=True)] * 10)
    result = ScopeAgent(llm, engine, max_iterations=2, max_tool_calls=20).run(
        "some text", owner="citizen")
    assert result.status == "escalated"
    assert "iteration" in result.escalation_reason


def test_timeout_is_checked_after_a_slow_plan_call_returns(engine):
    class SlowLLM(ScriptedScopeLLM):
        def judge(self, *a, **k):
            time.sleep(0.05)
            return super().judge(*a, **k)

    llm = SlowLLM(plans=[no_plan()])
    result = ScopeAgent(llm, engine, timeout_s=0.01).run("some text", owner="citizen")
    assert result.status == "escalated"


def test_forbidden_capabilities_are_denied(engine):
    agent = ScopeAgent(ScriptedScopeLLM(), engine)
    with pytest.raises(CapabilityDenied):
        agent.capabilities.request("modify_policy")


def test_a_complete_trace_is_produced(engine):
    llm = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary"])],
                           decisions=[decision("BLOCK")])
    result = ScopeAgent(llm, engine).run("some text", owner="citizen")
    phases = [t.phase for t in result.trace]
    for expected in ("PLAN", "EXECUTE", "EVALUATE", "DECIDE", "ACT"):
        assert expected in phases


# ── a later `needs_scope_review=false` must not discard gathered evidence
def test_a_first_round_irrelevant_request_completes_without_decide(engine):
    """(A) Distinct from `test_obvious_in_scope_settles_without_a_tool_call`
    only in explicitly asserting DECIDE never ran."""
    llm = ScriptedScopeLLM(plans=[no_plan("obviously in scope on its wording")])
    result = ScopeAgent(llm, engine).run("What documents do I need to renew a licence?", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0


def test_a_first_round_relevant_request_runs_tools_then_decides(engine):
    """(B)"""
    llm = ScriptedScopeLLM(plans=[full_plan(["check_domain_vocabulary"])],
                           decisions=[decision("BLOCK", evidence_summary="a recipe request")])
    result = ScopeAgent(llm, engine).run("How do I make a good lasagna?", owner="citizen")
    assert result.tool_calls
    assert llm.decision_calls == 1
    assert result.decision.action == "BLOCK"


def test_multi_round_evidence_survives_a_later_needs_scope_review_false(engine):
    """(C, F)"""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"], more=True,
                         rationale="the wording alone did not settle it, checking vocabulary"),
              no_plan(rationale="the vocabulary check is conclusive — ready to decide")],
        decisions=[decision("FLAG", evidence_summary="a sideways bereavement question")])
    result = ScopeAgent(llm, engine).run(
        "I don't know what to do about my late father's paperwork.", owner="citizen")

    assert llm.plan_calls == 2
    assert llm.plan_script == []
    assert result.tool_calls, "the round-1 evidence must not be discarded"
    assert llm.decision_calls == 1
    assert result.decision.action == "FLAG"
    assert result.policy_decision is not None
    assert result.outcome is not None


def test_policy_engine_and_capabilities_are_not_bypassed_by_a_later_plan(engine):
    """(E)"""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"], more=True), no_plan()],
        decisions=[decision("BLOCK")])
    agent = ScopeAgent(llm, engine)
    policy_calls, capability_calls = [], []
    real_decide, real_execute = agent.policy_engine.decide, agent.capabilities.execute
    agent.policy_engine.decide = lambda *a, **k: (policy_calls.append(1), real_decide(*a, **k))[1]
    agent.capabilities.execute = lambda *a, **k: (capability_calls.append(1), real_execute(*a, **k))[1]

    agent.run("How do I make a good lasagna?", owner="citizen")

    assert policy_calls
    assert capability_calls


def test_no_allow_with_null_outcome_once_evidence_was_gathered(engine):
    """(G)"""
    llm = ScriptedScopeLLM(
        plans=[full_plan(["check_domain_vocabulary"], more=True), no_plan()],
        decisions=[decision("ALLOW", evidence_summary="the vocabulary hit was a false positive")])
    result = ScopeAgent(llm, engine).run("some borderline text", owner="citizen")

    assert result.tool_calls
    if result.decision.action == "ALLOW":
        assert result.outcome is not None
        assert result.policy_decision is not None
