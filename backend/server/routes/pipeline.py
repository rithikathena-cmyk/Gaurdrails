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
from backend.guardrails.types import Surface

from ..auth import User, current_user
from ..state import state
from ._guardrail_prefilter import STOPPING_ACTIONS, run_prefilter_stages
from .agents import AgentRunRequest

router = APIRouter()


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

    request_id = f"pipeline_{uuid.uuid4().hex[:10]}"

    # ── stages 1-3: GuardrailSupervisor -> Supervisor -> PolicyEngine ────
    pre = run_prefilter_stages(
        req.text, surface, resource_kind=req.resource_kind, resource_owner=req.resource_owner,
        user=user, engine=engine, request_id=request_id)

    if pre.stopped:
        return _response(pre, None)

    # ── stage 4: the real retrieval -> LLM -> output-guardrails pipeline ─
    try:
        conv_result = engine.converse(
            req.text, session_id=f"pipeline-{request_id}",
            model=user.model or None, principal=user.name,
        )
    except LLMError as exc:
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    state.record(conv_result.trace.to_dict())

    return _response(pre, conv_result)


def _response(pre, conv_result) -> dict[str, Any]:
    body = pre.to_dict()
    body["conversation"] = (
        {
            "reply": conv_result.reply,
            "blocked": conv_result.blocked,
            "refusal_reason": conv_result.refusal_reason,
            "human_review": conv_result.human_review,
            "chunks": conv_result.chunks,
            "trace": conv_result.trace.to_dict(),
        } if conv_result is not None else None
    )
    return body
