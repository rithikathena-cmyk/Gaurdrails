"""Parameter registry — read and edit.

The editing path here is not a runtime override. `policy.runtime_override` is
locked, and it stays locked: no request parameter changes a rail. What this does
is write a validated config file, record the change, and reload — the sanctioned
path, with an author, a diff, and an audit entry.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from guardrails import as_payload
from guardrails.config import ConfigError, reset_overrides, save_overrides
from guardrails.registry import ADJUSTABLE

from ..state import state

log = logging.getLogger("guardrails.server")
router = APIRouter()

CHANGELOG = Path("config-changes.log")


class ParamPatch(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    matrix: dict[str, dict[str, str]] = Field(default_factory=dict)
    author: str = Field(default="console", max_length=64)


def _snapshot() -> dict[str, Any]:
    payload = as_payload()
    p = state.policy
    if p:
        payload.update(
            current={k: p.get(k) for k in ADJUSTABLE},
            baseline=p.baseline_values,
            matrix=p.matrix,
            baseline_matrix=p.baseline_matrix,
            overridden=sorted(p.overridden),
            matrix_overridden=sorted(p.matrix_overridden),
            source=p.source,
            overrides_path=p.overrides_path,
        )
    return payload


def _log_changes(author: str, changes: list[dict[str, Any]]) -> None:
    if not changes:
        return
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "author": author,
        "changes": changes,
    }
    with CHANGELOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    for c in changes:
        log.info("config change by %s: %s %r -> %r", author, c["key"], c["from"], c["to"])


@router.get("/parameters")
def parameters() -> dict[str, Any]:
    """The registry plus everything currently set. The UI renders from this alone."""
    return _snapshot()


@router.patch("/parameters")
def patch_parameters(patch: ParamPatch) -> dict[str, Any]:
    """Apply a change set: validate → write overrides → reload → audit.

    Validation runs against the registry before anything is written, so a
    rejected change leaves the running config untouched.
    """
    if not state.policy:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    if not patch.values and not patch.matrix:
        raise HTTPException(400, detail={"kind": "empty", "message": "no changes supplied"})

    try:
        summary = save_overrides(state.policy, patch.values, patch.matrix)
    except ConfigError as exc:
        raise HTTPException(422, detail={"kind": "invalid", "message": str(exc)}) from exc

    try:
        state.reload()
    except ConfigError as exc:  # pragma: no cover — save_overrides validated first
        raise HTTPException(500, detail={"kind": "reload", "message": str(exc)}) from exc

    _log_changes(patch.author, summary["changes"])
    return {"ok": True, **summary, "snapshot": _snapshot()}


@router.post("/parameters/reset")
def reset_parameters(author: str = "console") -> dict[str, Any]:
    """Drop every override and return to the checked-in baseline."""
    if not state.policy:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})

    before = {k: state.policy.get(k) for k in sorted(state.policy.overridden)}
    result = reset_overrides(state.policy)
    state.reload()

    changes = [
        {"key": k, "from": v, "to": state.policy.get(k)} for k, v in before.items()
    ] if state.policy else []
    _log_changes(author, changes)
    return {"ok": True, **result, "reverted": len(changes), "snapshot": _snapshot()}


@router.get("/parameters/changes")
def change_history(limit: int = 50) -> dict[str, Any]:
    if not CHANGELOG.exists():
        return {"entries": []}
    lines = [ln for ln in CHANGELOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    entries = [json.loads(ln) for ln in lines[-limit:]]
    entries.reverse()
    return {"entries": entries}
