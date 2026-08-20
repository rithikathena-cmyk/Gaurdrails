"""Health, policy inspection, audit verification."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.guardrails import AuditLog

from ..auth import require
from ..state import state

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": state.error is None,
        "error": state.error,
        "model_rails": state.model_rails,
        "config": state.policy.source if state.policy else None,
        "overrides": sorted(state.policy.overridden) if state.policy else [],
        "corpus": state.corpus.stats(),
        "agent": {
            "ready": state.agent is not None and state.model_rails,
            "tools": sorted(state.policy.get("agent.tools_enabled") or []) if state.policy else [],
            "max_steps": state.policy.get("agent.max_steps") if state.policy else None,
        },
        "note": None if state.model_rails else
                "ANTHROPIC_API_KEY not set — deterministic rails run, model rails are skipped.",
    }


@router.get("/policy", dependencies=[Depends(require("audit"))])
def policy_view() -> dict[str, Any]:
    if not state.policy:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    return state.policy.to_dict()


@router.get("/audit/verify", dependencies=[Depends(require("audit"))])
def audit_verify() -> dict[str, Any]:
    ok, message = AuditLog("audit.log").verify()
    return {"ok": ok, "message": message}
