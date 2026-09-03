"""The GuardrailSupervisor -> Supervisor -> PolicyEngine chain, shared.

Extracted out of `routes/pipeline.py` so `routes/agent.py`'s opt-in
`agent.prefilter_mode="agentic"` hook (see `guardrails/registry.py`) runs the
identical chain rather than a second, driftable copy of it — including one
fix this extraction carries for free: `text` is normalized here, once, which
neither call site did before. `normalize()` (NFKC + homoglyph fold) is a
locked safety invariant everywhere else in this codebase (`engine.py`,
`agent/runner.py`) but was never called anywhere under `guardrails/agents/` —
a homoglyph-obfuscated injection could slip past `GuardrailSupervisor`'s and
`Supervisor`'s own pattern/vocabulary tools purely because they never saw it
normalized first.

Stage 4 (`Engine.converse()`) is deliberately not part of this module —
`routes/pipeline.py` wants it, `routes/agent.py`'s hook wants to fall through
into `AgentRunner.run()` instead. Both callers decide what happens after
`PrefilterOutcome.stopped` is `False`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from backend.guardrails import LLMError
from backend.guardrails.agents.authorization_tools import AuthorizationContext
from backend.guardrails.agents.guardrail_supervisor import GuardrailSupervisor
from backend.guardrails.agents.policy_engine import PolicyEngine
from backend.guardrails.agents.supervisor import AgentNotRegistered, Supervisor
from backend.guardrails.agents.tools import ToolNotAllowed
from backend.guardrails.agents.types import (
    GuardrailSupervisorResult, PolicyDecision, SupervisorResult,
)
from backend.guardrails.engine import Engine
from backend.guardrails.rails.normalize import normalize
from backend.guardrails.types import Surface

from ..auth import User
from .agents import (
    _audit_completed_guardrail_run, _audit_completed_run,
    _audit_failed_guardrail_run, _audit_failed_run, _resolve_for_reader,
    _resolve_guardrail_result_for_reader,
)

#: Final actions that stop the request before the next stage runs — the same
#: boundary `routes/pipeline.py` already used for this chain.
STOPPING_ACTIONS = {"BLOCK", "REDACT", "ESCALATE"}

#: `PrefilterOutcome.final_action` is a `GuardrailAction` (ALLOW/MASK/REDACT/
#: BLOCK/FLAG/ESCALATE) — the frontend's chip/trace CSS and JS only know the
#: four `Verdict` values (pass/flag/mask/block), the same mismatch
#: `agent/runner.py`'s `_AGENTIC_TO_VERDICT` maps for `ToolCall.verdict`.
#: Same resolution here: REDACT -> mask, ESCALATE -> block. Shared by every
#: route that renders a stopped `PrefilterOutcome` (`agent.py`, `chat.py`)
#: rather than each keeping its own copy to drift out of sync.
ACTION_TO_VERDICT = {
    "ALLOW": "pass", "FLAG": "flag", "MASK": "mask",
    "REDACT": "mask", "BLOCK": "block", "ESCALATE": "block",
}


@dataclass
class PrefilterOutcome:
    request_id: str
    guardrail_supervisor: GuardrailSupervisorResult
    supervisor: SupervisorResult | None
    policy_decision: PolicyDecision | None
    stopped: bool
    stopped_at: str | None
    final_action: str
    refusal_text: str
    wall_clock_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "guardrail_supervisor": self.guardrail_supervisor.to_dict(),
            "supervisor": self.supervisor.to_dict() if self.supervisor is not None else None,
            "policy_engine": self.policy_decision.model_dump() if self.policy_decision is not None else None,
            "final_action": self.final_action,
            "stopped_at": self.stopped_at,
            "wall_clock_ms": self.wall_clock_ms,
        }


def _refusal_text(gs_result: GuardrailSupervisorResult, sup_result: SupervisorResult | None) -> str:
    if sup_result is not None and sup_result.reasoning_summary:
        return sup_result.reasoning_summary
    if gs_result.decision is not None and gs_result.decision.reason:
        return gs_result.decision.reason
    return gs_result.escalation_reason or (
        "This request was stopped by the guardrail layer before an answer was generated.")


def run_prefilter_stages(text: str, surface: Surface, *, resource_kind: str, resource_owner: str,
                         user: User, engine: Engine, request_id: str) -> PrefilterOutcome:
    """Stages 1-3 of `routes/pipeline.py`'s chain: GuardrailSupervisor, then
    Supervisor (only if not already stopped and a model is configured), then
    the one PolicyEngine reconciliation. Raises the same HTTPExceptions
    `routes/pipeline.py` already raised for this chain, so every existing
    caller and every new one see identical failure modes."""
    surface_value = surface.value
    ctx = AuthorizationContext(principal=user.name, role=user.role,
                               permissions=frozenset(user.permissions),
                               resource_kind=resource_kind, resource_owner=resource_owner)
    began = time.perf_counter()
    text, _chars_changed = normalize(text)

    try:
        gs_result = GuardrailSupervisor(engine.llm, engine).run(
            text, surface=surface, owner=user.name, request_id=f"{request_id}_gs", ctx=ctx)
    except ToolNotAllowed as exc:
        _audit_failed_guardrail_run(f"{request_id}_gs", user.name, surface_value, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(500, detail={"kind": "tool_not_allowed", "message": str(exc)}) from exc
    except LLMError as exc:
        _audit_failed_guardrail_run(f"{request_id}_gs", user.name, surface_value, str(exc),
                                    (time.perf_counter() - began) * 1000)
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    _resolve_guardrail_result_for_reader(gs_result, engine, user.name)
    _audit_completed_guardrail_run(gs_result, user.name, surface_value)

    gs_final = gs_result.policy_decision.final_action if gs_result.policy_decision else "ESCALATE"
    if gs_result.hard_blocked or gs_final in STOPPING_ACTIONS:
        return PrefilterOutcome(
            request_id=request_id, guardrail_supervisor=gs_result, supervisor=None,
            policy_decision=None, stopped=True, stopped_at="guardrail_supervisor",
            final_action=gs_final, refusal_text=_refusal_text(gs_result, None),
            wall_clock_ms=round((time.perf_counter() - began) * 1000, 1))

    sup_result: SupervisorResult | None = None
    if engine.llm is not None:  # Supervisor has no internal None-guard of its own
        try:
            sup_result = Supervisor(engine.llm, engine).run(
                text, surface=surface, owner=user.name, request_id=f"{request_id}_sup", ctx=ctx)
        except AgentNotRegistered as exc:
            _audit_failed_run(f"{request_id}_sup", user.name, surface_value, str(exc),
                              (time.perf_counter() - began) * 1000)
            raise HTTPException(500, detail={"kind": "agent_not_registered",
                                             "message": str(exc)}) from exc
        except LLMError as exc:
            _audit_failed_run(f"{request_id}_sup", user.name, surface_value, str(exc),
                              (time.perf_counter() - began) * 1000)
            raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc
        _resolve_for_reader(sup_result, engine, user.name)
        _audit_completed_run(sup_result, user.name, surface_value)

    combined = PolicyEngine().decide(
        recommended_action=(sup_result.final_action if sup_result is not None else gs_final),
        has_findings=True, policy_action=gs_final.lower(),
    )

    stopped = combined.final_action in STOPPING_ACTIONS
    return PrefilterOutcome(
        request_id=request_id, guardrail_supervisor=gs_result, supervisor=sup_result,
        policy_decision=combined, stopped=stopped,
        stopped_at="policy_engine" if stopped else None,
        final_action=combined.final_action,
        refusal_text=_refusal_text(gs_result, sup_result) if stopped else "",
        wall_clock_ms=round((time.perf_counter() - began) * 1000, 1))
