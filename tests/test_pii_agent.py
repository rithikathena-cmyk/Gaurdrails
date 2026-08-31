"""The autonomous PII agent, and the boundaries around it.

Two things are asserted throughout, and neither is "does it return MASK for
an SSN" on its own — a hardcoded `if ssn: MASK` would pass that trivially.

    genuine autonomy   the decision comes from a scripted judge call, keyed
                       by schema shape exactly like `ScriptedJudge` in
                       `test_adjudicator.py`. Scripting a *different* answer
                       to the same input must produce a *different* result —
                       see `test_the_decision_is_genuinely_the_models`.

    hard boundaries    a tool name outside the fixed allowlist, or a
                       capability outside the six `GuardrailAction` values,
                       must fail in Python before any model has a say —
                       these are tested directly, not through the agent.

Presidio's real model is expensive to load (real, first-call cost:
seconds). Every test here stubs `presidio_ner.find` except the one that
exists specifically to prove the tool wrapper still calls the real thing.
"""

from __future__ import annotations

import time

import pytest

from backend.guardrails import AuditLog, Engine, load
from backend.guardrails.agents.capabilities import CapabilityDenied, PIICapabilities
from backend.guardrails.agents.pii_agent import PIIAgent
from backend.guardrails.agents.tools import (
    PII_AGENT_TOOLS, PII_TOOL_NAMES, ToolNotAllowed, call as call_tool,
)
from backend.guardrails.rails import presidio_ner
from tests.conftest import REPO

#: Captured at import time, before any test can monkeypatch the module
#: attribute — the one genuine reference the integration test needs.
_REAL_PRESIDIO_FIND = presidio_ner.find


@pytest.fixture(autouse=True)
def no_presidio_model(monkeypatch):
    """`available()` does a real `import presidio_analyzer`, which is the
    actual expensive step on a cold process — not `engine()` construction.
    Both are stubbed so an orchestration test never pays that cost by
    accident; the one test that should pay it restores both explicitly.
    """
    monkeypatch.setattr(presidio_ner, "find", lambda *a, **k: [])
    monkeypatch.setattr(presidio_ner, "available", lambda: True)
    yield


@pytest.fixture
def engine(tmp_path):
    policy = load(REPO / "config" / "policy.yaml")
    return Engine(policy, None, AuditLog(tmp_path / "audit.log"))


class ScriptedAgentLLM:
    """A model that answers exactly what the test script says, keyed by which
    schema it was asked to fill — the same dispatch style `ScriptedJudge` in
    `test_adjudicator.py` uses for the same reason: a schema's property names
    are a more stable key than call order."""

    def __init__(self, plans=None, decisions=None):
        self.plan_script = list(plans or [])
        self.decision_script = list(decisions or [])
        self.plan_calls = 0
        self.decision_calls = 0
        self.seen_users: list[str] = []

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.seen_users.append(user)
        props = set(schema.get("properties", {}))
        if "needs_analysis" in props:
            self.plan_calls += 1
            if self.plan_script:
                return self.plan_script.pop(0)
            return no_plan("default stub — no script left")
        if "action" in props:
            self.decision_calls += 1
            if self.decision_script:
                return self.decision_script.pop(0)
            return decision("ALLOW", rationale="default stub — no script left")
        raise AssertionError(f"unexpected schema shape: {sorted(props)}")


def full_plan(tools, more=False, rationale="plan"):
    return {"needs_analysis": True, "tools": tools, "more_evidence_needed": more,
            "rationale": rationale}


def no_plan(rationale="nothing PII-relevant here"):
    return {"needs_analysis": False, "tools": [], "more_evidence_needed": False,
            "rationale": rationale}


def decision(action, confidence=0.9, rationale="stub decision", findings=None):
    return {"action": action, "confidence": confidence, "rationale": rationale,
            "findings": findings or []}


def finding(entity, risk="high", confidence=0.9, evidence=None):
    return {"entity": entity, "risk": risk, "confidence": confidence,
            "evidence": evidence or []}


# ── 1-3: detection reaches the model's decision ──────────────────────
def test_agent_detects_ssn(engine):
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("MASK", findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    assert result.decision.action == "MASK"
    regex_call = next(c for c in result.tool_calls if c.tool == "detect_pii_regex")
    assert regex_call.result["findings"][0]["kind"] == "US_SSN"


def test_agent_detects_email(engine):
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("MASK", findings=[finding("EMAIL_ADDRESS")])])
    result = PIIAgent(llm, engine).run("write to meera@example.com", owner="citizen")
    regex_call = next(c for c in result.tool_calls if c.tool == "detect_pii_regex")
    assert regex_call.result["findings"][0]["kind"] == "EMAIL_ADDRESS"
    assert result.decision.action == "MASK"


def test_agent_detects_phone(engine):
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("MASK", findings=[finding("PHONE_NUMBER")])])
    result = PIIAgent(llm, engine).run("call me at 415-555-0143", owner="citizen")
    regex_call = next(c for c in result.tool_calls if c.tool == "detect_pii_regex")
    assert regex_call.result["findings"][0]["kind"] == "PHONE_NUMBER"
    assert result.decision.action == "MASK"


# ── 4-5: tool selection is real, not always-run-everything ───────────
def test_agent_selects_only_the_tools_its_plan_named(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision("ALLOW")])
    result = PIIAgent(llm, engine).run("opening hours are 9 to 5", owner="citizen")
    assert {c.tool for c in result.tool_calls} == {"detect_pii_regex"}


def test_agent_can_call_multiple_tools(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex", "detect_pii_presidio"])],
                           decisions=[decision("ALLOW")])
    result = PIIAgent(llm, engine).run("some ordinary text", owner="citizen")
    assert {c.tool for c in result.tool_calls} == {"detect_pii_regex", "detect_pii_presidio"}


# ── 6: conflicting evidence reaches the decision call intact ─────────
def test_agent_evaluates_conflicting_tool_results(engine):
    """Regex finds a checksum-verified SSN; Presidio — which was never built to
    look for one — reports nothing. Both real, divergent results reach the
    decision call; the scripted answer proves the model saw both rather than
    the loop resolving the conflict itself with a hardcoded rule.
    """
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex", "detect_pii_presidio"])],
        decisions=[decision("MASK", rationale="checksum-verified SSN outweighs "
                                              "presidio's silence on a kind it "
                                              "was never built to find",
                            findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    regex_call = next(c for c in result.tool_calls if c.tool == "detect_pii_regex")
    presidio_call = next(c for c in result.tool_calls if c.tool == "detect_pii_presidio")
    assert regex_call.result["findings"], "regex should have found the SSN"
    assert presidio_call.result["findings"] == [], "presidio should report nothing"
    assert "detect_pii_regex" in llm.seen_users[-1] and "detect_pii_presidio" in llm.seen_users[-1]
    assert result.decision.action == "MASK"


# ── 7-10: each action is reachable, chosen by the model ───────────────
@pytest.mark.parametrize("action", ["MASK", "BLOCK", "ALLOW", "ESCALATE"])
def test_agent_chooses_each_action(engine, action):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision(action)])
    result = PIIAgent(llm, engine).run("some text", owner="citizen")
    assert result.decision.action == action


def test_the_decision_is_genuinely_the_models_not_a_hardcoded_rule(engine):
    """Same SSN-bearing input, a scripted answer that disagrees with what a
    naive `if ssn_found: MASK` would return. If DECIDE were hardcoded this
    could not fail; scripted, it must follow the script."""
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision("FLAG", rationale="operator judgement call")])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    assert result.decision.action == "FLAG"


# ── 11-12: the tool boundary, enforced in Python ──────────────────────
def test_unknown_tool_raises_tool_not_allowed(engine):
    with pytest.raises(ToolNotAllowed):
        call_tool("shell_exec", {}, engine, "call_x")


def test_a_hallucinated_tool_in_a_plan_is_never_silently_run(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex", "read_database"])],
                           decisions=[decision("ALLOW")])
    with pytest.raises(ToolNotAllowed):
        PIIAgent(llm, engine).run("some text", owner="citizen")


def test_the_tool_registry_has_no_dynamic_dispatch(engine):
    """There is no getattr/eval path from a name to a function — only the
    four names below resolve to anything, and nothing widens that set."""
    assert set(PII_AGENT_TOOLS) == set(PII_TOOL_NAMES) == {
        "detect_pii_regex", "detect_pii_presidio", "classify_pii_type", "get_pii_policy",
    }
    for hostile in ("__import__", "exec", "eval", "os.system", "subprocess.run"):
        with pytest.raises(ToolNotAllowed):
            call_tool(hostile, {}, engine, "call_x")


# ── 13-15: capability-layer hard boundaries ────────────────────────────
def test_capability_layer_denies_policy_modification(engine):
    caps = PIICapabilities(engine.pii_rail, engine.vault)
    with pytest.raises(CapabilityDenied):
        caps.request("modify_policy")
    with pytest.raises(CapabilityDenied):
        caps.request("modify_overrides")


def test_capability_layer_denies_rbac_modification(engine):
    caps = PIICapabilities(engine.pii_rail, engine.vault)
    with pytest.raises(CapabilityDenied):
        caps.request("modify_rbac")
    with pytest.raises(CapabilityDenied):
        caps.request("change_role")


def test_capability_layer_denies_self_permission_escalation(engine):
    caps = PIICapabilities(engine.pii_rail, engine.vault)
    with pytest.raises(CapabilityDenied):
        caps.request("grant_permission")
    with pytest.raises(CapabilityDenied):
        caps.request("modify_tool_allowlist")


@pytest.mark.parametrize("capability", sorted(PIICapabilities.FORBIDDEN))
def test_every_named_forbidden_capability_is_denied(engine, capability):
    """Every entry in the explicit forbidden list, not just the three the
    spec named — reveal_vault, execute_code, filesystem/database access,
    disabling a guardrail, and the rest all deny the same way."""
    caps = PIICapabilities(engine.pii_rail, engine.vault)
    with pytest.raises(CapabilityDenied):
        caps.request(capability)


def test_a_decision_action_is_the_only_thing_that_reaches_the_capability_layer(engine):
    """`AgentDecision.action` is a `Literal` of six values — Pydantic itself
    refuses to construct one outside that set, which is the floor beneath
    the capability layer's own six-way dispatch."""
    from pydantic import ValidationError

    from backend.guardrails.agents.types import AgentDecision

    with pytest.raises(ValidationError):
        AgentDecision(action="REVEAL_VAULT", confidence=1.0, rationale="x", findings=[])


# ── 16-18: bounded loop limits ──────────────────────────────────────────
def test_max_tool_calls_is_enforced(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(
        ["detect_pii_regex", "detect_pii_presidio", "classify_pii_type", "get_pii_policy"])])
    result = PIIAgent(llm, engine, max_tool_calls=1).run(
        "My SSN is 796-33-9021", owner="citizen")
    assert result.status == "escalated"
    assert "tool call budget" in result.escalation_reason
    assert len(result.tool_calls) <= 1


def test_max_iterations_is_enforced(engine):
    """A model that always asks for another round never gets to decide — the
    loop stops it rather than looping forever."""
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"], more=True)] * 10)
    result = PIIAgent(llm, engine, max_iterations=2, max_tool_calls=20).run(
        "My SSN is 796-33-9021", owner="citizen")
    assert result.status == "escalated"
    assert "iteration" in result.escalation_reason
    assert llm.plan_calls == 2


def test_timeout_is_enforced(engine):
    class SlowLLM(ScriptedAgentLLM):
        def judge(self, *a, **k):
            time.sleep(0.05)
            return super().judge(*a, **k)

    llm = SlowLLM(plans=[full_plan(["detect_pii_regex"])], decisions=[decision("ALLOW")])
    result = PIIAgent(llm, engine, timeout_s=0.01).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert "exceeded" in result.escalation_reason


def test_timeout_is_enforced_even_when_the_plan_short_circuits(engine):
    """A slow PLAN call that happens to conclude 'nothing needed' still spent
    the budget — the no-analysis-needed short circuit must not skip the
    timeout check the way the main path already did.
    """
    class SlowLLM(ScriptedAgentLLM):
        def judge(self, *a, **k):
            time.sleep(0.05)
            return super().judge(*a, **k)

    llm = SlowLLM(plans=[no_plan()])
    result = PIIAgent(llm, engine, timeout_s=0.01).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert "exceeded" in result.escalation_reason


# ── 19: malformed output escalates ──────────────────────────────────────
def test_malformed_plan_tool_name_escalates_rather_than_running_it(engine):
    """A plan naming a tool outside the enum is malformed input to EXECUTE —
    `ToolNotAllowed` propagates as a security error, not a soft escalation,
    because a model asking for a forbidden tool is not the same failure mode
    as a model returning unparsable JSON."""
    llm = ScriptedAgentLLM(plans=[{"needs_analysis": True, "tools": ["not_a_real_tool"],
                                   "more_evidence_needed": False, "rationale": "x"}])
    with pytest.raises(ToolNotAllowed):
        PIIAgent(llm, engine).run("some text", owner="citizen")


def test_malformed_decision_output_escalates(engine):
    """An action outside the six, or a confidence outside 0..1, fails Pydantic
    validation — caught by the loop and turned into ESCALATE, not a crash."""
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[{"action": "DESTROY_EVERYTHING", "confidence": 2.0,
                   "rationale": "x", "findings": []}])
    result = PIIAgent(llm, engine).run("some text", owner="citizen")
    assert result.status == "escalated"
    assert result.decision.action == "ESCALATE"


def test_a_hallucinated_evidence_citation_is_dropped(engine):
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("MASK", findings=[
            finding("US_SSN", evidence=["a_call_id_nobody_recorded"])])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    assert result.decision.findings == []


# ── 20-21: trace and action execution ───────────────────────────────────
def test_a_complete_trace_is_produced(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision("MASK", findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    phases = [t.phase for t in result.trace]
    for expected in ("PLAN", "EXECUTE", "EVALUATE", "DECIDE", "ACT"):
        assert expected in phases, f"{expected} missing from {phases}"


def test_action_execution_is_traced_and_the_outcome_recorded(engine):
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision("MASK", findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    assert result.outcome is not None
    assert result.outcome.action == "MASK"
    assert "796-33-9021" not in result.outcome.text_out
    assert "<US_SSN:" in result.outcome.text_out
    assert any(t.phase == "ACT" for t in result.trace)


def test_no_analysis_needed_still_produces_a_trace_and_an_allow(engine):
    llm = ScriptedAgentLLM(plans=[no_plan()])
    result = PIIAgent(llm, engine).run("what time do you open", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert result.status == "completed"


# ── the one test that pays the real model-load cost ─────────────────────
def test_detect_pii_presidio_really_calls_the_existing_recognizer(engine, monkeypatch):
    """Every other test in this file stubs `presidio_ner.find` for speed. This
    one restores the true function — captured at import time, so it is
    unaffected by the autouse stub — and proves the tool wrapper still calls
    it, rather than a second, parallel implementation of NER detection."""
    calls = []

    def spy(text, kinds, min_conf, taken=None):
        calls.append((text, sorted(kinds), min_conf))
        return _REAL_PRESIDIO_FIND(text, kinds, min_conf, taken)

    monkeypatch.setattr(presidio_ner, "find", spy)
    res = call_tool("detect_pii_presidio", {"text": "Ravi Kumar lives in Chennai"},
                    engine, "call_x")
    assert calls, "the wrapper never called presidio_ner.find"
    assert res.status == "ok"


# ── the Policy Engine: the agent recommends, it decides ──────────────────
def test_the_policy_engine_overrides_a_permissive_recommendation(engine):
    """The agent finds a checksum-verified SSN but — misjudging it, or being
    scripted to for this test — recommends ALLOW. `pii.action.user_prompt`
    is `mask` in the checked-in policy. The Policy Engine's deterministic
    floor must win: `AgentDecision.action` stays ALLOW as the honest record
    of what the model recommended, but `outcome.action` — what actually gets
    executed — is MASK.
    """
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("ALLOW", rationale="misjudged as not sensitive",
                            findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert result.decision.action == "ALLOW", "the recommendation is recorded honestly"
    assert result.policy_decision is not None
    assert result.policy_decision.recommended_action == "ALLOW"
    assert result.policy_decision.floor_action == "MASK"
    assert result.policy_decision.overridden is True
    assert result.policy_decision.final_action == "MASK"
    assert result.outcome.action == "MASK", "the floor, not the recommendation, was executed"
    assert "796-33-9021" not in result.outcome.text_out


def test_the_policy_engine_upholds_a_recommendation_at_or_above_the_floor(engine):
    """The agent recommends BLOCK for an SSN where policy only requires MASK
    — more caution than the floor needs no permission, and is not overridden
    downward to match it."""
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("BLOCK", rationale="context makes this look adversarial",
                            findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert result.policy_decision.overridden is False
    assert result.policy_decision.final_action == "BLOCK"
    assert result.outcome.action == "BLOCK"


def test_the_policy_engine_leaves_a_recommendation_alone_with_no_findings(engine):
    """Nothing was found — the floor is ALLOW, and the agent's own
    recommendation (whatever it was) is never second-guessed against a floor
    that has nothing behind it."""
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                           decisions=[decision("FLAG", findings=[])])
    result = PIIAgent(llm, engine).run("some ordinary text", owner="citizen")

    assert result.policy_decision.floor_action == "ALLOW"
    assert result.policy_decision.overridden is False
    assert result.policy_decision.final_action == "FLAG"


def test_a_confident_floor_survives_the_agents_own_escalation(engine):
    """The agent could not reach a confident view of its own evidence and
    recommends ESCALATE — but a checksum-verified SSN with a `mask` policy
    is not something the agent's uncertainty un-finds. The Policy Engine
    still enforces the floor.
    """
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"])],
        decisions=[decision("ESCALATE", rationale="conflicting signals, unsure",
                            findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert result.decision.action == "ESCALATE"
    assert result.policy_decision.final_action == "MASK"
    assert result.outcome.action == "MASK"


def test_every_result_carries_a_policy_decision_including_when_escalated(engine):
    """Bounded-loop escalation (budget/timeout/iteration limits, not the
    agent's own uncertainty) also produces a `policy_decision` — the trace
    shape stays uniform whether the agent decided anything or not."""
    llm = ScriptedAgentLLM(plans=[full_plan(
        ["detect_pii_regex", "detect_pii_presidio", "classify_pii_type", "get_pii_policy"])])
    result = PIIAgent(llm, engine, max_tool_calls=1).run(
        "My SSN is 796-33-9021", owner="citizen")
    assert result.status == "escalated"
    assert result.policy_decision is not None
    assert result.policy_decision.recommended_action == "ESCALATE"


# ── a later `needs_analysis=false` must not discard gathered evidence ───
# The live bug this section regression-tests: `needs_analysis` was checked
# on *every* PLAN round, so a model that said "true" in round 1 (gathering
# real evidence) and then "false" in round 2 (meaning "I have enough, ready
# to decide") had that second answer misread as "this request was never
# PII-relevant," discarding the evidence and returning a hardcoded ALLOW —
# skipping DECIDE, POLICY, and ACT entirely.
def test_a_first_round_irrelevant_request_completes_without_decide(engine):
    """(A) `needs_analysis=false` with no evidence gathered yet is the
    genuine, unchanged shortcut — no tool ran, and the DECIDE schema
    (`action`/`findings`) was never asked for."""
    llm = ScriptedAgentLLM(plans=[no_plan("nothing PII-relevant in a fee-schedule question")])
    result = PIIAgent(llm, engine).run("What is the fee for a trade licence?", owner="citizen")
    assert result.decision.action == "ALLOW"
    assert result.tool_calls == []
    assert llm.decision_calls == 0, "DECIDE must not run when nothing was ever relevant"


def test_a_first_round_relevant_request_runs_tools_then_decides(engine):
    """(B) `needs_analysis=true` on round 1 runs tools and reaches DECIDE."""
    llm = ScriptedAgentLLM(plans=[full_plan(["detect_pii_regex"])],
                          decisions=[decision("MASK", findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")
    assert result.tool_calls
    assert llm.decision_calls == 1
    assert result.decision.action == "MASK"


def test_multi_round_evidence_survives_a_later_needs_analysis_false(engine):
    """(C, F) Round 1 gathers real evidence; round 2 is a genuinely
    different scripted PLAN response (`needs_analysis=false`, its own
    distinct rationale) meaning "no more evidence needed," not "irrelevant."
    The evidence from round 1 must still reach DECIDE, POLICY, and ACT."""
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"], more=True,
                         rationale="a structured identifier was found, checking policy next"),
              no_plan(rationale="checksum verified and policy read — enough to decide")],
        decisions=[decision("MASK", findings=[finding("US_SSN")])])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert llm.plan_calls == 2, "two genuinely distinct PLAN calls must have happened"
    assert llm.plan_script == [], "both scripted plan responses were consumed, not reused"
    assert result.tool_calls, "the round-1 tool evidence must not be discarded"
    assert llm.decision_calls == 1, "DECIDE must still run"
    assert result.decision.action == "MASK"
    assert result.policy_decision is not None
    assert result.outcome is not None
    assert result.outcome.action == "MASK"
    assert result.status == "completed"


def test_policy_engine_and_capabilities_are_not_bypassed_by_a_later_plan(engine):
    """(E) A spy around the deterministic layers themselves — not just the
    result shape — proves a later `needs_analysis=false` cannot route
    around POLICY or ACT."""
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"], more=True), no_plan()],
        decisions=[decision("MASK", findings=[finding("US_SSN")])])
    agent = PIIAgent(llm, engine)
    policy_calls, capability_calls = [], []
    real_decide, real_execute = agent.policy_engine.decide, agent.capabilities.execute
    agent.policy_engine.decide = lambda *a, **k: (policy_calls.append(1), real_decide(*a, **k))[1]
    agent.capabilities.execute = lambda *a, **k: (capability_calls.append(1), real_execute(*a, **k))[1]

    agent.run("My SSN is 796-33-9021", owner="citizen")

    assert policy_calls, "PolicyEngine.decide must be called"
    assert capability_calls, "capabilities.execute must be called"


def test_no_allow_with_null_outcome_once_evidence_was_gathered(engine):
    """(G) The exact shape the live bug produced — ALLOW with `outcome=None`
    — must be unreachable once any tool has actually run."""
    llm = ScriptedAgentLLM(
        plans=[full_plan(["detect_pii_regex"], more=True), no_plan()],
        decisions=[decision("ALLOW", rationale="checksum failed, not a real SSN")])
    result = PIIAgent(llm, engine).run("My SSN is 796-33-9021", owner="citizen")

    assert result.tool_calls, "evidence must have been gathered for this assertion to be meaningful"
    if result.decision.action == "ALLOW":
        assert result.outcome is not None, "ALLOW reached with evidence gathered must still go through ACT"
        assert result.policy_decision is not None
