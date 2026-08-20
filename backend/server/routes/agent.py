"""Agent endpoints.

Two calls, because an agent turn can stop half-way. `POST /api/agent/chat` runs
the loop until the agent either answers or asks to do something that changes
state; `POST /api/agent/approve` answers that question and resumes.

The paused state lives on the server, keyed by a one-use token. The client gets
the token and a human-readable summary of what is about to happen — never the
transcript, and never anything it could edit and hand back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.guardrails import AgentRunner, LLMError
from backend.guardrails.agent import TOOLS

from ..auth import User, cost_micros, current_user, directory
from ..history import history
from .chat import check_budget, usage_in
from ..state import state

router = APIRouter()

SAMPLES = [
    {
        "title": "Multi-step lookup",
        "icon": "tool",
        "blurb": "Search, then fee lookup — two tools, both railed",
        "text": "What do I need to renew a trade licence, and what will it cost for a "
                "400 square foot shop?",
    },
    {
        "title": "Vaulted identifier",
        "icon": "key",
        "blurb": "The claim reference is masked before the model sees it",
        "text": "Can you check the status of my claim CLM-40028871?",
    },
    {
        "title": "Write action",
        "icon": "pen",
        "blurb": "Filing a grievance stops for approval",
        "text": "My claim CLM-40028871 has been open far too long. Check it, and if it is "
                "overdue please file a grievance about the delay.",
    },
]


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=64)


class ApprovalRequest(BaseModel):
    token: str = Field(min_length=4, max_length=64)
    approved: bool
    session_id: str = Field(default="default", max_length=64)


def _runner() -> AgentRunner:
    if state.error:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    if state.agent is None:
        raise HTTPException(503, detail={"kind": "startup", "message": "agent not ready"})
    if not state.model_rails:
        raise HTTPException(503, detail={
            "kind": "llm",
            "message": "The agent needs a model. Set ANTHROPIC_API_KEY and restart.",
        })
    return state.agent


def _payload(result: Any) -> dict[str, Any]:
    trace = result.trace.to_dict()
    state.record(trace)
    body: dict[str, Any] = {
        "reply": result.reply,
        "blocked": result.blocked,
        "refusal_reason": result.refusal_reason,
        "verdict": result.trace.verdict.value,
        "violations": result.violations,
        "human_review": result.human_review,
        "chunks": result.chunks,
        "detections": result.detections,
        "calls": [c.to_dict() for c in result.calls],
        "steps": result.steps,
        "filed": result.filed,
        "trace": trace,
        "approval": None,
    }
    if result.approval is not None:
        state.park(result.approval)
        body["approval"] = result.approval.to_dict()
    return body


@router.get("/agent/tools")
def tools() -> dict[str, Any]:
    enabled = set(state.policy.get("agent.tools_enabled") or []) if state.policy else set()
    return {
        "tools": [
            {
                "name": t.name,
                "kind": t.kind,
                "description": t.description,
                "enabled": t.name in enabled,
                "unmask_args": list(t.unmask_args),
                "approval": t.kind == "write",
                "why_approval": t.why_approval,
            }
            for t in TOOLS.values()
        ],
        "samples": SAMPLES,
        "max_steps": state.policy.get("agent.max_steps") if state.policy else None,
        "max_tool_calls": state.policy.get("agent.max_tool_calls") if state.policy else None,
    }


def _remember(user: User, session_id: str, question: str,
              body: dict[str, Any], calls: list) -> None:
    """Write the agent turn to the transcript, priced like a chat turn."""
    trace = body.get("trace") or {}
    history.append(
        user.name, session_id=session_id, question=question,
        reply=body.get("reply") or "", verdict=trace.get("verdict", ""),
        request_id=trace.get("request_id", ""), mode="agent",
        blocked=bool(body.get("blocked")),
        refusal_reason=body.get("refusal_reason") or "",
        masked=len(body.get("detections") or []),
        tokens=sum(i + o for _, i, o in calls),
        cost_usd=sum(cost_micros(m, i, o) for m, i, o in calls) / 1e6,
        model=next((m for m, _, _ in calls if m), ""),
    )


@router.post("/agent/chat")
def agent_chat(req: AgentRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    runner = _runner()
    check_budget(user)
    try:
        result = runner.run(
            req.message, history=state.history(req.session_id, user.name),
            session_id=req.session_id, principal=user.name,
        )
    except LLMError as exc:
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    # A paused turn is not a finished turn: nothing goes into history until the
    # agent has actually answered.
    if not result.blocked and result.approval is None:
        state.remember(req.session_id, req.message, result.reply, user.name)
    body = _payload(result)
    calls = usage_in(body.get("trace") or {})
    directory.spend(user.name, calls)
    _remember(user, req.session_id, req.message, body, calls)
    return body


@router.post("/agent/approve")
def approve(req: ApprovalRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    runner = _runner()
    pending = state.claim(req.token, user.name)
    if pending is None:
        raise HTTPException(404, detail={
            "kind": "approval",
            "message": "That approval has expired or was already answered. Ask again and "
                       "you will get a fresh one.",
        })
    try:
        result = runner.resume(pending, req.approved, session_id=req.session_id,
                               principal=user.name)
    except LLMError as exc:
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    if not result.blocked and result.approval is None:
        state.remember(req.session_id, pending.question, result.reply, user.name)
    body = _payload(result)
    body["approved"] = req.approved
    body["resumed_from"] = pending.origin_request_id
    calls = usage_in(body.get("trace") or {})
    directory.spend(user.name, calls)
    _remember(user, req.session_id, pending.question, body, calls)
    return body
