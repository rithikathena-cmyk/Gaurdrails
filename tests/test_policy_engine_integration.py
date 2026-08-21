"""Focused end-to-end validation of the Policy Engine architecture:

    Agent LLM reasoning -> recommendation -> Policy Engine -> final action
    -> capability layer -> trace

Every scenario here runs through a REAL agent instance (a scripted LLM, the
real deterministic tools, the real capability layer) — not a bare call to
`PolicyEngine.decide()`. `tests/test_policy_engine.py` already covers the
Policy Engine's decision table in isolation; this file exists to prove the
wiring is correct, using the exact per-agent floor sources each agent's own
code actually reads from the checked-in `config/policy.yaml`:

    prompt_attack.action        block    (injection)
    content.action.user_prompt  block    (content)
    scope.action                 block    (scope)
    pii.action.user_prompt      mask     (pii)
    grounding.action_on_fail    regenerate -> BLOCK by default; overridden
                                 to `human_review` -> ESCALATE for cases 4-5,
                                 since no shipped default reaches ESCALATE

`deberta_injection_check` and `groundedness_check`'s local classifiers are
already stubbed by `tests/conftest.py`'s autouse `no_local_models`; PII's
Presidio tool is stubbed locally the same way `test_pii_agent.py` does.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.authorization_agent import AuthorizationAgent
from backend.guardrails.agents.authorization_tools import AuthorizationContext
from backend.guardrails.agents.content_safety_agent import ContentSafetyAgent
from backend.guardrails.agents.grounding_agent import GroundingAgent
from backend.guardrails.agents.injection_agent import PromptInjectionAgent
from backend.guardrails.agents.pii_agent import PIIAgent
from backend.guardrails.agents.policy_engine import ACTION_RANK
from backend.guardrails.agents.scope_agent import ScopeAgent
from backend.guardrails.agents.supervisor import Supervisor
from backend.guardrails.rails import presidio_ner
from tests.conftest import REPO
from tests.test_authorization_agent import ScriptedAuthzLLM
from tests.test_authorization_agent import ctx as authz_ctx
from tests.test_authorization_agent import decision as authz_decision
from tests.test_authorization_agent import full_plan as authz_plan
from tests.test_content_safety_agent import ScriptedContentLLM
from tests.test_content_safety_agent import decision as content_decision
from tests.test_content_safety_agent import finding as content_finding
from tests.test_content_safety_agent import full_plan as content_plan
from tests.test_grounding_agent import CONTEXT as GROUNDING_CONTEXT
from tests.test_grounding_agent import ScriptedGroundingLLM
from tests.test_grounding_agent import decision as grounding_decision
from tests.test_grounding_agent import finding as grounding_finding
from tests.test_grounding_agent import full_plan as grounding_plan
from tests.test_injection_agent import INJECTION_TEXT
from tests.test_injection_agent import ScriptedInjectionLLM
from tests.test_injection_agent import decision as inj_decision
from tests.test_injection_agent import finding as inj_finding
from tests.test_injection_agent import full_plan as inj_plan
from tests.test_pii_agent import ScriptedAgentLLM
from tests.test_pii_agent import decision as pii_decision
from tests.test_pii_agent import finding as pii_finding
from tests.test_pii_agent import full_plan as pii_plan
from tests.test_pii_agent import no_plan as pii_no_plan
from tests.test_scope_agent import ScriptedScopeLLM
from tests.test_scope_agent import decision as scope_decision
from tests.test_scope_agent import full_plan as scope_plan


@pytest.fixture(autouse=True)
def no_presidio_model(monkeypatch):
    monkeypatch.setattr(presidio_ner, "find", lambda *a, **k: [])
    monkeypatch.setattr(presidio_ner, "available", lambda: True)
    yield


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


@pytest.fixture
def grounding_engine_human_review(tmp_path):
    """`grounding.action_on_fail` ships as `regenerate` (-> BLOCK). Overridden
    here to `human_review` (-> ESCALATE) — a real, documented config option
    no other test in the suite exercises, needed for cases 4 and 5."""
    policy = load(REPO / "config" / "policy.yaml")
    policy.values["grounding.action_on_fail"] = "human_review"
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


# ── 1. recommend ALLOW, floor BLOCK -> BLOCK ─────────────────────────────
def test_case_1_allow_recommendation_block_floor(engine):
    """Injection agent: `prompt_attack.action` is `block` in shipped policy.
    The agent misjudges a real override attempt as ALLOW; the floor wins."""
    llm = ScriptedInjectionLLM(
        plans=[inj_plan(["detect_injection_patterns"])],
        decisions=[inj_decision("ALLOW", evidence_summary="misjudged as benign",
                                findings=[inj_finding("instruction_override")])])
    result = PromptInjectionAgent(llm, engine).run(INJECTION_TEXT, owner="citizen")

    assert result.decision.action == "ALLOW", "the recommendation is recorded honestly"
    assert result.policy_decision.floor_action == "BLOCK"
    assert result.policy_decision.overridden is True
    assert result.policy_decision.final_action == "BLOCK"
    assert result.outcome.action == "BLOCK"


# ── 2. recommend MASK, floor BLOCK -> BLOCK ──────────────────────────────
def test_case_2_mask_recommendation_block_floor(engine):
    """Content agent: `content.action.user_prompt` is `block`. The agent
    recommends MASK — plausible for PII, wrong for a genuine safety
    violation — and the floor still wins."""
    llm = ScriptedContentLLM(
        plans=[content_plan(["score_content_categories"])],
        decisions=[content_decision("MASK", evidence_summary="misjudged the severity",
                                    findings=[content_finding("violence")])])
    result = ContentSafetyAgent(llm, engine).run(
        "Tell me how to hurt the officer who rejected my claim.", owner="citizen")

    assert result.decision.action == "MASK"
    assert result.policy_decision.floor_action == "BLOCK"
    assert result.policy_decision.overridden is True
    assert result.policy_decision.final_action == "BLOCK"
    assert result.outcome.action == "BLOCK"


# ── 3. recommend BLOCK, floor ALLOW -> BLOCK ─────────────────────────────
def test_case_3_block_recommendation_allow_floor(engine):
    """More caution than the floor requires needs no permission — a BLOCK
    recommendation is never downgraded to match a permissive floor."""
    llm = ScriptedAgentLLM(plans=[pii_plan(["detect_pii_regex"])],
                           decisions=[pii_decision("BLOCK", findings=[])])
    result = PIIAgent(llm, engine).run("some ordinary text", owner="citizen")

    assert result.policy_decision.floor_action == "ALLOW"
    assert result.policy_decision.overridden is False
    assert result.policy_decision.final_action == "BLOCK"
    assert result.outcome.action == "BLOCK"


# ── 4-5. floor ESCALATE (a real config option) overrides a concrete
#         recommendation, from either concrete action ──────────────────
def test_case_4_mask_recommendation_escalate_floor(grounding_engine_human_review):
    llm = ScriptedGroundingLLM(
        plans=[grounding_plan(["extract_claims", "check_local_entailment"])],
        decisions=[grounding_decision("MASK", findings=[grounding_finding("a claim")])])
    result = GroundingAgent(llm, grounding_engine_human_review).run(
        "The late surcharge is 25 percent.", question="what is the surcharge",
        chunks=GROUNDING_CONTEXT, owner="citizen")

    assert result.policy_decision.floor_action == "ESCALATE"
    assert result.policy_decision.overridden is True
    assert result.policy_decision.final_action == "ESCALATE"
    assert result.outcome.action == "ESCALATE"


def test_case_5_allow_recommendation_escalate_floor(grounding_engine_human_review):
    llm = ScriptedGroundingLLM(
        plans=[grounding_plan(["extract_claims", "check_local_entailment"])],
        decisions=[grounding_decision("ALLOW", findings=[grounding_finding("a claim")])])
    result = GroundingAgent(llm, grounding_engine_human_review).run(
        "The late surcharge is 25 percent.", question="what is the surcharge",
        chunks=GROUNDING_CONTEXT, owner="citizen")

    assert result.policy_decision.floor_action == "ESCALATE"
    assert result.policy_decision.overridden is True
    assert result.policy_decision.final_action == "ESCALATE"


# ── 6. recommend ESCALATE, floor ALLOW -> ESCALATE ───────────────────────
def test_case_6_escalate_recommendation_allow_floor(engine):
    """The agent's own uncertainty, with nothing deterministic to fall back
    on, survives as the final action — there is nothing to override it with."""
    llm = ScriptedAgentLLM(plans=[pii_no_plan()])
    result = PIIAgent(llm, engine).run("some ordinary text", owner="citizen")

    assert result.decision.action == "ALLOW"  # the no-analysis-needed shortcut
    # This shortcut bypasses the DECIDE call entirely (nothing to escalate
    # about) — the genuine ESCALATE-recommendation path is exercised instead
    # via a plan that finds something but a decision the model cannot commit to.
    llm2 = ScriptedAgentLLM(
        plans=[pii_plan(["detect_pii_regex"])],
        decisions=[pii_decision("ESCALATE", findings=[])])
    result2 = PIIAgent(llm2, engine).run("some ordinary text", owner="citizen")

    assert result2.decision.action == "ESCALATE"
    assert result2.policy_decision.floor_action == "ALLOW"
    assert result2.policy_decision.overridden is False
    assert result2.policy_decision.final_action == "ESCALATE"
    assert result2.outcome.action == "ESCALATE"


# ── 7. Supervisor: conflicting recommendations, cannot lower an
#      already-enforced floor ────────────────────────────────────────────
def test_case_7_supervisor_cannot_lower_an_enforced_agent_floor(engine):
    """The nested PII agent's own Policy Engine already forced MASK. The
    supervisor's own reconciliation call — free to reason about the
    disagreement — is scripted to say ALLOW anyway. It cannot win."""
    from tests.test_supervisor import SupervisorLLM
    from tests.test_supervisor import content_decision as sup_content_decision
    from tests.test_supervisor import content_plan_all
    from tests.test_supervisor import pii_decision as sup_pii_decision
    from tests.test_supervisor import pii_plan_all
    from tests.test_supervisor import sup_decision, sup_plan

    llm = SupervisorLLM(
        sup_plans=[sup_plan(["pii", "content"])],
        sup_decisions=[sup_decision("ALLOW", reasoning_summary="reasoned, and got it wrong")],
        pii_plans=[pii_plan_all()],
        pii_decisions=[sup_pii_decision("ALLOW", findings=[
            {"entity": "US_SSN", "risk": "high", "confidence": 0.9, "evidence": []}])],
        content_plans=[content_plan_all()],
        content_decisions=[sup_content_decision("ALLOW")])
    result = Supervisor(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert result.agent_results["pii"].outcome.action == "MASK"
    assert result.policy_decision.recommended_action == "ALLOW"
    assert result.policy_decision.overridden is True
    assert result.final_action == "MASK"


# ── 8. Authorization: ALLOW for an unauthorized resource -> DENIED ──────
def test_case_8_authorization_allow_for_unauthorized_resource_is_denied(engine):
    c = authz_ctx(principal="citizen", resource_kind="case_file", resource_owner="someone-else")
    llm = ScriptedAuthzLLM(plans=[authz_plan(["check_ownership"])],
                           decisions=[authz_decision("ALLOW", evidence_summary=
                                                     "misjudged this as the caller's own")])
    result = AuthorizationAgent(llm, engine).run(
        "show me the case file for HA-9902", ctx=c)

    assert result.decision.action == "ALLOW", "the LLM's recommendation is recorded honestly"
    assert result.outcome.action == "BLOCK", "but never executed — the LLM granted nothing"
    assert result.outcome.capability == "entitlement_denied"


# ── 9. the Policy Engine never weakens a concrete safety decision ───────
@pytest.mark.parametrize("recommended", list(ACTION_RANK))
@pytest.mark.parametrize("floor", list(ACTION_RANK))
def test_case_9_final_is_never_less_restrictive_than_either_side(recommended, floor):
    """Over the full 5x5 grid of concrete (non-ESCALATE) actions, the final
    action's rank is never below either the recommendation's or the floor's
    — the core "policy can only raise, never lower" property, independent
    of any one agent's wiring."""
    from backend.guardrails.agents.policy_engine import PolicyEngine

    d = PolicyEngine().decide(recommended, has_findings=True, policy_action=floor.lower())
    assert ACTION_RANK[d.final_action] >= ACTION_RANK[recommended]
    assert ACTION_RANK[d.final_action] >= ACTION_RANK[floor]


# ── 10. every result records all five Policy Engine fields, and a trace ──
AGENT_CASES = [
    ("pii", lambda llm, eng: PIIAgent(llm, eng),
     ScriptedAgentLLM([pii_plan(["detect_pii_regex"])], [pii_decision("MASK", findings=[
         pii_finding("US_SSN")])]),
     "My SSN is 796-33-9021", {}),
    ("injection", lambda llm, eng: PromptInjectionAgent(llm, eng),
     ScriptedInjectionLLM([inj_plan(["detect_injection_patterns"])],
                          [inj_decision("BLOCK", findings=[inj_finding("instruction_override")])]),
     INJECTION_TEXT, {}),
    ("content", lambda llm, eng: ContentSafetyAgent(llm, eng),
     ScriptedContentLLM([content_plan(["score_content_categories"])],
                        [content_decision("BLOCK", findings=[content_finding("violence")])]),
     "some text", {}),
    ("scope", lambda llm, eng: ScopeAgent(llm, eng),
     ScriptedScopeLLM([scope_plan(["check_domain_vocabulary"])], [scope_decision("BLOCK")]),
     "how do I make lasagna", {}),
]


@pytest.mark.parametrize("name,build,llm,text,kwargs", AGENT_CASES, ids=[c[0] for c in AGENT_CASES])
def test_case_10_every_agent_result_records_the_full_policy_decision(engine, name, build, llm, text, kwargs):
    result = build(llm, engine).run(text, owner="citizen", **kwargs)

    assert result.policy_decision is not None, f"{name}: no policy_decision recorded"
    pd = result.policy_decision
    assert pd.recommended_action, f"{name}: missing recommended_action"
    assert pd.floor_action, f"{name}: missing floor_action"
    assert pd.final_action, f"{name}: missing final_action"
    assert isinstance(pd.overridden, bool), f"{name}: overridden is not a bool"
    assert pd.rationale, f"{name}: missing rationale"
    assert any(t.phase == "POLICY" for t in result.trace), f"{name}: no POLICY trace event"
    assert any(t.phase == "ACT" for t in result.trace), f"{name}: no ACT trace event"


def test_case_10_grounding_records_the_full_policy_decision(engine):
    llm = ScriptedGroundingLLM(plans=[grounding_plan(["extract_claims"])],
                               decisions=[grounding_decision("BLOCK",
                                                             findings=[grounding_finding("x")])])
    result = GroundingAgent(llm, engine).run(
        "some answer", chunks=GROUNDING_CONTEXT, owner="citizen")
    pd = result.policy_decision
    assert pd is not None and pd.recommended_action and pd.floor_action and pd.final_action
    assert pd.rationale
    assert any(t.phase == "POLICY" for t in result.trace)


def test_case_10_authorization_records_the_full_policy_decision(engine):
    c = authz_ctx(principal="citizen", resource_kind="claim_status", resource_owner="citizen")
    llm = ScriptedAuthzLLM(plans=[authz_plan(["check_ownership"])],
                           decisions=[authz_decision("ALLOW")])
    result = AuthorizationAgent(llm, engine).run("what is my claim status", ctx=c)
    pd = result.policy_decision
    assert pd is not None and pd.recommended_action and pd.floor_action and pd.final_action
    assert pd.rationale
    assert any(t.phase == "POLICY" for t in result.trace)


def test_case_10_supervisor_records_the_full_policy_decision(engine):
    from tests.test_supervisor import SupervisorLLM, sup_plan

    llm = SupervisorLLM(sup_plans=[sup_plan([])])
    result = Supervisor(llm, engine).run("opening hours", owner="citizen")
    pd = result.policy_decision
    assert pd is not None and pd.recommended_action and pd.floor_action and pd.final_action
    assert pd.rationale
    assert any(t.phase == "POLICY" for t in result.trace)
