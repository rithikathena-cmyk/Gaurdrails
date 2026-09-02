"""`agent.prefilter_mode` must not make AgentRunner's own pii.entities rail
run twice for one message.

`_guardrail_prefilter.py` hands the *original*, unmodified text straight to
`AgentRunner.run()` when it doesn't stop the request — it decides, it doesn't
rewrite. So when the prefilter chain is on, the same message is seen by two
independent code paths: GuardrailSupervisor/Supervisor's own specialists
first, then AgentRunner's own fixed rail pipeline (the one `entities.py`'s
`EntityRail` — the rail this session's timeout/detach and retrieval-gating
fix both touch — belongs to). That AgentRunner's *own* copy of that rail
still only runs once, prefilter on or off, is the property these tests pin:
enabling the chain should add its own judge calls on top, never cause the
downstream rail to double-fire.

`entities.py`'s `ENTITY_SCHEMA` has exactly one top-level property,
`entities` — no other schema in this codebase does — which is what makes
that call countable independently of everything else the two supervisors
and AgentRunner ask their model.

Note on `routes/agent.py`'s response shape: the `"prefilter"` key only
appears when the chain *stopped* the request (`pre.stopped`) — a clean
message that GuardrailSupervisor/Supervisor let through falls straight into
`runner.run()` and comes back as an ordinary `AgentRunner` payload, no
`"prefilter"` key at all, even though the chain genuinely ran and genuinely
cost real judge calls. These tests use the call-count proxy for that,
not the response shape.
"""

from __future__ import annotations

from backend.guardrails.agent.runner import AgentRunner
from tests.test_agent import ScriptedClaude as AgentScriptedClaude

#: A message with a real capitalised name, so `entities.py`'s cheap gate
#: does not skip the rail before it ever reaches the judge.
CLEAN_MESSAGE = "My name is Meera Balan, what are your opening hours?"


class PrefilterAwareClaude(AgentScriptedClaude):
    """`ScriptedClaude` already answers everything AgentRunner's own rail
    pipeline asks. This adds GuardrailSupervisor's and Supervisor's PLAN/
    DECIDE schemas — distinct field names from every AgentRunner-side one —
    so a clean message can pass all the way through `agent.prefilter_mode=
    agentic` without either supervisor's loop breaking on an untyped stub
    answer, and counts every judge call plus, separately, calls shaped
    exactly like `entities.py`'s `ENTITY_SCHEMA`.
    """

    def __init__(self, script, **kw):
        super().__init__(script, **kw)
        self.entity_schema_calls = 0
        self.total_judge_calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.total_judge_calls += 1
        props = set(schema.get("properties", {}))
        if props == {"entities"}:
            self.entity_schema_calls += 1
        if "checks" in props:                                    # GuardrailSupervisor PLAN
            return {"risk_categories": [], "checks": [], "policy_keys": [],
                    "more_evidence_needed": False, "rationale": "stub"}
        if "risk_score" in props:                                  # GuardrailSupervisor DECIDE
            return {"action": "ALLOW", "risk_score": 0.0, "confidence": 1.0,
                    "triggered_rails": [], "evidence": [], "reason": "stub"}
        if "agents" in props:                                      # Supervisor PLAN
            return {"agents": [], "more_evidence_needed": False, "reason": "stub"}
        if "final_action" in props and "reasoning_summary" in props:  # Supervisor DECIDE
            return {"final_action": "ALLOW", "confidence": 1.0, "reasoning_summary": "stub"}
        return super().judge(system, user, schema, max_tokens=max_tokens, label=label)


def _install(client, script=()):
    """`app_state.engine.llm = llm` (the pattern `test_agent_prefilter_
    parameters.py` uses) is not enough here: `EntityRail`, `ScopeRail`, and
    `PromptAttackRail` all capture `llm` once at `Engine.__init__` and never
    re-read `engine.llm` afterward — only `ContentRail` is rebuilt fresh
    inside `evaluate()` and so is the only one a bare reassignment reaches.
    The `client` fixture builds its engine with no key at all, so without a
    full rebuild `entity_rail.llm` stays `None` and every judge-shaped
    assertion in this file would silently see zero calls no matter what ran.
    Rebuilding the engine is what `state.py`'s own `reload()` does on a real
    key change; this mirrors that instead of reaching into a stale rail."""
    from backend.guardrails.engine import Engine
    from backend.server.state import state as app_state

    llm = PrefilterAwareClaude(list(script))
    app_state.engine = Engine(app_state.policy, llm, app_state.audit, app_state.corpus)
    app_state.agent = AgentRunner(app_state.engine, llm)
    app_state.model_rails = True
    return llm


def _patch(client, values):
    resp = client.patch("/api/parameters", json={"values": values})
    assert resp.status_code == 200, resp.text


# ── Test 6 ───────────────────────────────────────────────────────────
def test_prefilter_off_baseline_calls_entities_exactly_once(client):
    """Baseline: no prefilter chain, AgentRunner's own pii.entities rail
    still only asks its judge once for one message."""
    llm = _install(client, [("answer", "Weekdays, 10 to 5.")])
    resp = client.post("/api/agent/chat", json={"message": CLEAN_MESSAGE})
    assert resp.status_code == 200, resp.text
    assert "prefilter" not in resp.json()
    assert llm.entity_schema_calls == 1, (
        f"expected exactly one ENTITY_SCHEMA call with the prefilter off, "
        f"got {llm.entity_schema_calls}"
    )


# ── Test 7 ───────────────────────────────────────────────────────────
def test_prefilter_on_costs_more_but_still_calls_entities_once(client):
    """`agentic`: GuardrailSupervisor + Supervisor each cost their own PLAN/
    DECIDE judge calls on top — real, additional cost, exactly as
    `registry.py`'s own description of `agent.prefilter_mode` says — but
    AgentRunner's own copy of pii.entities, once it falls through to
    AgentRunner.run(), must still fire exactly once, not twice, for the
    same message."""
    _patch(client, {"agent.prefilter_mode": "agentic"})
    try:
        llm = _install(client, [("answer", "Weekdays, 10 to 5.")])
        resp = client.post("/api/agent/chat", json={"message": CLEAN_MESSAGE})
        assert resp.status_code == 200, resp.text

        # The chain ran and cost real calls (GuardrailSupervisor PLAN+DECIDE,
        # Supervisor PLAN+DECIDE — at least 4) even though a clean message
        # that isn't stopped never puts a "prefilter" key in the response.
        assert llm.total_judge_calls >= 4, (
            f"expected GuardrailSupervisor's and Supervisor's own PLAN/DECIDE "
            f"calls, got only {llm.total_judge_calls} judge calls total — "
            f"the chain may not have run at all"
        )
        assert llm.entity_schema_calls == 1, (
            f"the prefilter chain must not make AgentRunner's own pii.entities "
            f"rail run more than once — got {llm.entity_schema_calls}"
        )
    finally:
        _patch(client, {"agent.prefilter_mode": "off"})


def test_prefilter_on_makes_strictly_more_judge_calls_than_off(client):
    """The `agentic` cost is real, not free — the two modes must not be
    indistinguishable in what they ask the model. Same message, two
    separately-scripted turns, compared by total call count — but
    AgentRunner's own pii.entities call count must not move at all."""
    off_llm = _install(client, [("answer", "Weekdays, 10 to 5.")])
    off_resp = client.post("/api/agent/chat", json={"message": CLEAN_MESSAGE})
    assert off_resp.status_code == 200, off_resp.text

    _patch(client, {"agent.prefilter_mode": "agentic"})
    try:
        on_llm = _install(client, [("answer", "Weekdays, 10 to 5.")])
        on_resp = client.post("/api/agent/chat", json={"message": CLEAN_MESSAGE})
        assert on_resp.status_code == 200, on_resp.text

        assert on_llm.total_judge_calls > off_llm.total_judge_calls, (
            f"prefilter=agentic ({on_llm.total_judge_calls} judge calls) should "
            f"cost strictly more than prefilter=off ({off_llm.total_judge_calls})"
        )
        assert on_llm.entity_schema_calls == off_llm.entity_schema_calls == 1
    finally:
        _patch(client, {"agent.prefilter_mode": "off"})
