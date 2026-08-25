"""The real end-to-end pipeline, chained from what already exists.

Additive, same as `agents.py`: `POST /api/chat` is untouched. This route
composes four already-shipping pieces, unmodified — the flat
`GuardrailSupervisor`, the six-specialist `Supervisor`, the one
`PolicyEngine`, and `Engine.converse()` — into a single real, driveable
request so `/summary` can show the actual path one message takes rather than
a static diagram. Nothing here re-implements any guardrail, supervisor,
policy floor, or tracer; it only decides which of the existing ones to call
next and stitches their results together for the UI.

Sequencing, and why:

    GuardrailSupervisor   first — cheap, fast, mostly-deterministic (its own
                          hard-block precheck needs zero judge calls). A
                          request it already blocks never reaches Supervisor
                          or the model at all.
    Supervisor            second, only if GuardrailSupervisor did not stop
                          the request and a model is configured — the deeper,
                          six-specialist-agent reasoning pass. `Supervisor`
                          has no internal guard against `engine.llm is None`
                          (unlike `GuardrailSupervisor`), so this route
                          replicates the same upfront check `run_supervisor`
                          already makes, rather than let it raise.
    PolicyEngine.decide() combines the two layers' own already-enforced
                          final actions the same way `has_findings=True`
                          combines an agent's recommendation with a config
                          floor everywhere else in this codebase — reusing
                          the one existing `PolicyEngine`, not a new rule.
                          Its auto-generated `.rationale` is not surfaced —
                          it reads as "agent vs. floor" when both sides here
                          are already-enforced verdicts, not a fresh one.
    Engine.converse()     only if the combined action is not one of
                          BLOCK/REDACT/ESCALATE — the real retrieval, LLM
                          call, output guardrails and grounding, unmodified.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.guardrails import LLMError
from backend.guardrails.agents.authorization_tools import AuthorizationContext
from backend.guardrails.agents.guardrail_supervisor import GuardrailSupervisor
from backend.guardrails.agents.policy_engine import PolicyEngine
from backend.guardrails.agents.supervisor import AgentNotRegistered, Supervisor
from backend.guardrails.agents.tools import ToolNotAllowed
from backend.guardrails.types import Surface

from ..auth import User, current_user
from ..state import state
from .agents import (
    AgentRunRequest, _audit_completed_guardrail_run, _audit_completed_run,
    _audit_failed_guardrail_run, _audit_failed_run, _resolve_for_reader,
    _resolve_guardrail_result_for_reader,
)

router = APIRouter()

#: Final actions that stop the request before the next stage runs — the same
#: boundary every `PolicyEngine` caller already uses to decide whether ACT
#: proceeds; here it decides whether the *next layer* runs at all.
_STOPPING_ACTIONS = {"BLOCK", "REDACT", "ESCALATE"}


def _engine():
    if state.error:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    if state.engine is None:
        raise HTTPException(503, detail={"kind": "startup", "message": "engine not ready"})
    return state.engine


@router.post("/pipeline/run")
def run_pipeline(req: AgentRunRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    engine = _engine()
    try:
        surface = Surface(req.surface)
    except ValueError:
        raise HTTPException(422, detail={"kind": "bad_surface", "message": req.surface})

    ctx = AuthorizationContext(
        principal=user.name, role=user.role, permissions=frozenset(user.permissions),
        resource_kind=req.resource_kind, resource_owner=req.resource_owner,
    )
    request_id = f"pipeline_{uuid.uuid4().hex[:10]}"
    began = time.perf_counter()

    # ── stage 1: the fast, cheap, mostly-deterministic layer ────────────
    try:
        gs_result = GuardrailSupervisor(engine.llm, engine).run(
            req.text, surface=surface, owner=user.name,
            request_id=f"{request_id}_gs", ctx=ctx)
    except ToolNotAllowed as exc:
        _audit_failed_guardrail_run(f"{request_id}_gs", user.name, req.surface, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(500, detail={"kind": "tool_not_allowed", "message": str(exc)}) from exc
    except LLMError as exc:
        _audit_failed_guardrail_run(f"{request_id}_gs", user.name, req.surface, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    _resolve_guardrail_result_for_reader(gs_result, engine, user.name)
    _audit_completed_guardrail_run(gs_result, user.name, req.surface)

    gs_final = gs_result.policy_decision.final_action if gs_result.policy_decision else "ESCALATE"
    if gs_result.hard_blocked or gs_final in _STOPPING_ACTIONS:
        return _response(request_id, gs_result, None, None, None,
                         "guardrail_supervisor", began)

    # ── stage 2: the deeper, six-specialist-agent reasoning pass ────────
    sup_result = None
    if engine.llm is not None:  # Supervisor has no internal None-guard of its own
        try:
            sup_result = Supervisor(engine.llm, engine).run(
                req.text, surface=surface, owner=user.name,
                request_id=f"{request_id}_sup", ctx=ctx)
        except AgentNotRegistered as exc:
            _audit_failed_run(f"{request_id}_sup", user.name, req.surface, str(exc),
                              (time.perf_counter() - began) * 1000)
            raise HTTPException(500, detail={"kind": "agent_not_registered",
                                             "message": str(exc)}) from exc
        except LLMError as exc:
            _audit_failed_run(f"{request_id}_sup", user.name, req.surface, str(exc),
                              (time.perf_counter() - began) * 1000)
            raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc
        _resolve_for_reader(sup_result, engine, user.name)
        _audit_completed_run(sup_result, user.name, req.surface)

    # ── stage 3: the one deterministic floor, called once more ──────────
    combined = PolicyEngine().decide(
        recommended_action=(sup_result.final_action if sup_result is not None else gs_final),
        has_findings=True,
        policy_action=gs_final.lower(),
    )

    if combined.final_action in _STOPPING_ACTIONS:
        return _response(request_id, gs_result, sup_result, combined, None,
                         "policy_engine", began)

    # ── stage 4: the real retrieval -> LLM -> output-guardrails pipeline ─
    try:
        conv_result = engine.converse(
            req.text, session_id=f"pipeline-{request_id}",
            model=user.model or None, principal=user.name,
        )
    except LLMError as exc:
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    state.record(conv_result.trace.to_dict())

    return _response(request_id, gs_result, sup_result, combined, conv_result,
                     None, began)


def _response(request_id: str, gs_result, sup_result, combined, conv_result,
             stopped_at: str | None, began: float) -> dict[str, Any]:
    if combined is not None:
        final_action = combined.final_action
    elif gs_result.policy_decision is not None:
        final_action = gs_result.policy_decision.final_action
    else:
        final_action = "ESCALATE"

    return {
        "request_id": request_id,
        "guardrail_supervisor": gs_result.to_dict(),
        "supervisor": sup_result.to_dict() if sup_result is not None else None,
        "policy_engine": combined.model_dump() if combined is not None else None,
        "conversation": (
            {
                "reply": conv_result.reply,
                "blocked": conv_result.blocked,
                "refusal_reason": conv_result.refusal_reason,
                "human_review": conv_result.human_review,
                "chunks": conv_result.chunks,
                "trace": conv_result.trace.to_dict(),
            } if conv_result is not None else None
        ),
        "final_action": final_action,
        "stopped_at": stopped_at,
        "wall_clock_ms": round((time.perf_counter() - began) * 1000, 1),
    }
