"""The autonomous guardrail agents, reachable directly over HTTP.

Deliberately additive. `POST /api/chat` still runs the deterministic
pipeline it always has — `Engine.converse()`, unmodified, no agent anywhere
in that path. This router is a second, separate way to exercise the
Supervisor and its registered specialists against the same live engine,
so the architecture can be driven from the real application rather than
only from unit tests, without the live console's request latency or
behaviour changing for anyone who has not opened this page.

Why additive rather than a replacement: the Supervisor's own PLAN and
DECIDE calls, plus each specialist agent's own PLAN and DECIDE calls, are
each a real judge call — multiple seconds apiece, per the measurements this
project's own README already documents for a single judge call. Wiring the
Supervisor in as *the* decision path for every chat turn would multiply
that by however many agents a request selects, and there is no measurement
yet of what that costs in practice. This router is where that measurement
gets taken, deliberately opt-in.

Two things this router — and only this router — is responsible for wiring,
neither of which the Supervisor or any agent can synthesise for itself:

    AuthorizationContext   built from the signed-in `user` `current_user`
                           already resolved — real role, real permissions —
                           plus `resource_kind`/`resource_owner` when the
                           caller names a specific resource. Empty resource
                           fields are honest: this codebase has no system
                           that derives "which case file" from free text,
                           so a caller who knows one (a case-file view, a
                           claims lookup) supplies it; one who does not
                           leaves it unset, under which entitlement is True
                           by construction — nothing to be entitled *to* yet.
    audit entry            every run this endpoint produces — completed,
                           escalated, or failed before a result existed —
                           is written to the same hash-chained audit log
                           `Engine.converse()` writes to, so an agentic run
                           leaves the trail an ordinary chat turn already does.
    egress resolution      the one deterministic step after ACT: every
                           `outcome.text_out` in the result — the
                           Supervisor's own and every selected agent's —
                           is run through `PIICapabilities.resolve_for_reader`
                           for the signed-in caller, the same entitlement
                           check `Engine.converse()`'s own `vault.unmask`
                           stage already runs for ordinary chat. A vault
                           token an agent's ACT step minted resolves to the
                           real value only if this caller is who it was
                           minted for; every other reader gets the token
                           back exactly as written, never the value.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.guardrails import LLMError
from backend.guardrails.agents.authorization_tools import AuthorizationContext
from backend.guardrails.agents.capabilities import PIICapabilities
from backend.guardrails.agents.guardrail_supervisor import GuardrailSupervisor
from backend.guardrails.agents.guardrail_tools import ALLOWED_GUARDRAIL_TOOLS
from backend.guardrails.agents.supervisor import (
    SUPERVISOR_AGENTS, AgentNotRegistered, Supervisor, SupervisorResult,
)
from backend.guardrails.agents.tools import ToolNotAllowed
from backend.guardrails.agents.types import GuardrailSupervisorResult
from backend.guardrails.types import Surface

from ..auth import User, current_user
from ..state import state

router = APIRouter()


class AgentRunRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    surface: str = Field(default="user.prompt")
    #: The resource this request concerns, when the caller already knows
    #: one — never inferred from `text`. See the module docstring.
    resource_kind: str = Field(default="", max_length=200)
    resource_owner: str = Field(default="", max_length=200)


def _engine():
    if state.error:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    if state.engine is None:
        raise HTTPException(503, detail={"kind": "startup", "message": "engine not ready"})
    return state.engine


@router.get("/agents")
def list_agents() -> dict[str, Any]:
    """The registry the Supervisor selects from — the same dict the code
    itself dispatches through, not a hand-maintained description of it."""
    return {"agents": sorted(SUPERVISOR_AGENTS), "model_rails": state.model_rails}


@router.post("/agents/supervisor/run")
def run_supervisor(req: AgentRunRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    engine = _engine()
    if engine.llm is None:
        raise HTTPException(503, detail={"kind": "no_model",
                                         "message": "no API key configured — the "
                                                    "agents need a live judge call"})
    try:
        surface = Surface(req.surface)
    except ValueError:
        raise HTTPException(422, detail={"kind": "bad_surface", "message": req.surface})

    # Real facts about the real caller, not synthesised — the same role and
    # permission set `require()` already enforces to reach this endpoint at
    # all. `entitled` will not be True merely because this account is an
    # operator: `server.auth`'s permission vocabulary (chat, traces, agents,
    # ...) has no "admin" entry for `AuthorizationContext.entitled`'s own
    # override to match, so this reflects ownership honestly rather than
    # granting every caller of this endpoint blanket access to everyone
    # else's resources.
    ctx = AuthorizationContext(
        principal=user.name, role=user.role, permissions=frozenset(user.permissions),
        resource_kind=req.resource_kind, resource_owner=req.resource_owner,
    )
    request_id = f"supervisor_{uuid.uuid4().hex[:10]}"

    began = time.perf_counter()
    try:
        result = Supervisor(engine.llm, engine).run(
            req.text, surface=surface, owner=user.name, request_id=request_id, ctx=ctx)
    except AgentNotRegistered as exc:
        # Reachable only if a judge somehow returned a name outside the
        # schema's own enum — recorded as a security-relevant event rather
        # than surfaced as an ordinary 4xx, same severity `ToolNotAllowed`
        # gets when it escapes an agent's own run().
        _audit_failed_run(request_id, user.name, req.surface, str(exc),
                          (time.perf_counter() - began) * 1000)
        raise HTTPException(500, detail={"kind": "agent_not_registered",
                                         "message": str(exc)}) from exc
    except LLMError as exc:
        _audit_failed_run(request_id, user.name, req.surface, str(exc),
                          (time.perf_counter() - began) * 1000)
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    _resolve_for_reader(result, engine, user.name)
    _audit_completed_run(result, user.name, req.surface)

    return {
        "result": result.to_dict(),
        "wall_clock_ms": round((time.perf_counter() - began) * 1000, 1),
    }


@router.post("/agents/guardrail-supervisor/run")
def run_guardrail_supervisor(req: AgentRunRequest,
                             user: User = Depends(current_user)) -> dict[str, Any]:
    """The flat, single-hop MVP — `PLAN -> SELECT -> EXECUTE -> OBSERVE ->
    DECIDE -> ENFORCE -> TRACE` over the six tools in `ALLOWED_GUARDRAIL_TOOLS`
    directly, rather than `run_supervisor`'s six specialist agents. Additive,
    same as that endpoint: `POST /api/chat` is untouched either way.

    Deliberately no upfront `engine.llm is None` check, unlike
    `run_supervisor` above: the hard-block pre-check and the deterministic
    risk-band gate both work with no live model at all, exactly like the
    deterministic pipeline's own pattern layer. A request that genuinely
    needs the judge (PLAN, or DECIDE inside the marginal band) and finds no
    key still comes back as a normal 200 with `status: "escalated"` —
    `GuardrailSupervisor.run()` catches that `LLMError` internally, the same
    way it handles any other judge failure.
    """
    engine = _engine()
    try:
        surface = Surface(req.surface)
    except ValueError:
        raise HTTPException(422, detail={"kind": "bad_surface", "message": req.surface})

    ctx = AuthorizationContext(
        principal=user.name, role=user.role, permissions=frozenset(user.permissions),
        resource_kind=req.resource_kind, resource_owner=req.resource_owner,
    )
    request_id = f"guardrail_supervisor_{uuid.uuid4().hex[:10]}"

    began = time.perf_counter()
    try:
        result = GuardrailSupervisor(engine.llm, engine).run(
            req.text, surface=surface, owner=user.name, request_id=request_id, ctx=ctx)
    except ToolNotAllowed as exc:
        _audit_failed_guardrail_run(request_id, user.name, req.surface, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(500, detail={"kind": "tool_not_allowed",
                                         "message": str(exc)}) from exc
    except LLMError as exc:
        _audit_failed_guardrail_run(request_id, user.name, req.surface, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    _resolve_guardrail_result_for_reader(result, engine, user.name)
    _audit_completed_guardrail_run(result, user.name, req.surface)

    return {
        "result": result.to_dict(),
        "wall_clock_ms": round((time.perf_counter() - began) * 1000, 1),
    }


# ---------------------------------------------------------------------------
def _resolve_guardrail_result_for_reader(result: GuardrailSupervisorResult, engine: Any,
                                         reader: str) -> None:
    """Same deterministic egress step `_resolve_for_reader` runs for
    `run_supervisor`, applied to the flat result's own single `outcome`."""
    caps = PIICapabilities(engine.entity_rail, engine.vault, engine.policy)
    if result.outcome is not None and result.outcome.text_out:
        result.outcome.text_out, _ = caps.resolve_for_reader(result.outcome.text_out, reader)


def _audit_completed_guardrail_run(result: GuardrailSupervisorResult, who: str,
                                   surface: str) -> None:
    policy_decision = None
    if result.policy_decision is not None:
        policy_decision = {
            "final_action": result.policy_decision.final_action,
            "recommended_action": result.policy_decision.recommended_action,
            "floor_action": result.policy_decision.floor_action,
            "overridden": result.policy_decision.overridden,
        }
    state.audit.write_guardrail_supervisor_run(
        request_id=result.request_id, who=who, status=result.status,
        hard_blocked=result.hard_blocked,
        tools_run=sorted({c.tool for c in result.tool_calls}),
        risk_score=result.decision.risk_score if result.decision else None,
        judge_calls=result.judge_calls, policy_decision=policy_decision,
        final_action=result.policy_decision.final_action if result.policy_decision else "ESCALATE",
        escalation_reason=result.escalation_reason, duration_ms=result.duration_ms,
        trace=[{"phase": t.phase, "at_ms": t.at_ms} for t in result.trace],
        surface=surface,
    )


def _audit_failed_guardrail_run(request_id: str, who: str, surface: str, reason: str,
                                duration_ms: float) -> None:
    state.audit.write_guardrail_supervisor_run(
        request_id=request_id, who=who, status="failed", hard_blocked=False,
        tools_run=[], risk_score=None, judge_calls=0, policy_decision=None,
        final_action="ESCALATE", escalation_reason=reason,
        duration_ms=round(duration_ms, 1), trace=[], surface=surface,
    )


# ---------------------------------------------------------------------------
def _resolve_for_reader(result: SupervisorResult, engine: Any, reader: str) -> None:
    """The deterministic egress step, applied once per response: the
    Supervisor's own `outcome` and every selected agent's own `outcome` may
    carry a vault token an ACT step minted — resolved here for `reader`, the
    real signed-in caller, exactly the way `Engine.converse()` resolves one
    for `principal` at ordinary chat egress. A fresh `PIICapabilities` is
    used for this alone; it owns no agent's decision, only the one
    deterministic check this function exists to run.
    """
    caps = PIICapabilities(engine.entity_rail, engine.vault, engine.policy)
    if result.outcome is not None and result.outcome.text_out:
        result.outcome.text_out, _ = caps.resolve_for_reader(result.outcome.text_out, reader)
    for agent_result in result.agent_results.values():
        if agent_result.outcome is not None and agent_result.outcome.text_out:
            agent_result.outcome.text_out, _ = caps.resolve_for_reader(
                agent_result.outcome.text_out, reader)


def _audit_completed_run(result: SupervisorResult, who: str, surface: str) -> None:
    """Every status `Supervisor.run` can return — completed or escalated —
    reaches here. No request text, no agent `rationale`/`evidence_summary`:
    see `AuditLog.write_agent_run`'s own docstring for why."""
    agent_decisions = {
        name: {
            "action": ar.decision.action,
            "confidence": round(ar.decision.confidence, 4),
            "status": ar.status,
            "findings": [{"entity": f.entity, "risk": f.risk} for f in ar.decision.findings],
            # The agent's own Policy Engine step (before ACT) and what
            # actually executed can diverge — authorization is the case
            # where they do: the agent's own policy step has no config-driven
            # floor to apply (`has_findings=False`), so it passes the model's
            # recommendation through unchanged, and the deterministic
            # entitlement check only happens one step later, in ACT.
            "policy_final_action": ar.policy_decision.final_action if ar.policy_decision else None,
            "outcome_action": ar.outcome.action if ar.outcome else None,
        }
        for name, ar in result.agent_results.items()
    }
    policy_decision = None
    if result.policy_decision is not None:
        policy_decision = {
            "final_action": result.policy_decision.final_action,
            "recommended_action": result.policy_decision.recommended_action,
            "floor_action": result.policy_decision.floor_action,
            "overridden": result.policy_decision.overridden,
        }
    state.audit.write_agent_run(
        request_id=result.request_id, who=who, status=result.status,
        agents_selected=sorted(result.agent_results), agent_decisions=agent_decisions,
        policy_decision=policy_decision, final_action=result.final_action,
        confidence=result.confidence, escalation_reason=result.escalation_reason,
        duration_ms=result.duration_ms,
        trace=[{"phase": t.phase, "at_ms": t.at_ms} for t in result.trace],
        surface=surface,
    )


def _audit_failed_run(request_id: str, who: str, surface: str, reason: str,
                      duration_ms: float) -> None:
    """A run that never produced a `SupervisorResult` at all — the Supervisor
    itself raised before returning one. Still worth a trail: `who` asked,
    `when`, and `why` it failed."""
    state.audit.write_agent_run(
        request_id=request_id, who=who, status="failed", agents_selected=[],
        agent_decisions={}, policy_decision=None, final_action="ESCALATE",
        confidence=0.0, escalation_reason=reason, duration_ms=round(duration_ms, 1),
        trace=[], surface=surface,
    )
