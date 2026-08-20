"""People, and what they may spend.

An operator can see who exists, add someone, and set a ceiling on the model
tokens each person may consume. The ceiling is enforced where the spending
happens — in the chat and agent routes — rather than here, because a limit
checked only on the screen that displays it is decoration.

Two decisions worth stating:

    a limit of 0 means no ceiling, not "may spend nothing". It is the default
    for every account, and it reads the same in the API as it does in the UI.

    usage is counted from what the model actually reported — the `input_tokens`
    and `output_tokens` on every `llm.call` in the trace — not from an estimate
    of the prompt. An estimate would drift from the bill.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import (ASSIGNABLE_MODELS, PRICING, ROLES, User, current_user,
                    directory, require)
from ..history import history

router = APIRouter()


class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=4, max_length=128)
    role: str = "user"
    display: str = Field(default="", max_length=48)
    token_limit: int = Field(default=0, ge=0, le=1_000_000_000)
    daily_limit: int = Field(default=0, ge=0, le=1_000_000_000)
    monthly_limit: int = Field(default=0, ge=0, le=1_000_000_000)
    model: str = ""


class LimitPatch(BaseModel):
    token_limit: int | None = Field(default=None, ge=0, le=1_000_000_000)
    daily_limit: int | None = Field(default=None, ge=0, le=1_000_000_000)
    monthly_limit: int | None = Field(default=None, ge=0, le=1_000_000_000)
    model: str | None = None


def _row(u: User) -> dict[str, Any]:
    d = u.to_dict()
    d["active_sessions"] = directory.sessions_for(u.name)
    return d


def _snapshot() -> dict[str, Any]:
    users = [_row(u) for u in directory.users.values()]
    users.sort(key=lambda r: (r["role"] != "admin", r["name"]))
    return {
        "users": users,
        "total": len(users),
        "by_role": {
            key: sum(1 for r in users if r["role"] == key) for key in ROLES
        },
        "models": ASSIGNABLE_MODELS,
        "roles": [
            {"key": k, "label": v.get("label", k), "blurb": v.get("blurb", ""),
             "permissions": v.get("permissions", [])}
            for k, v in ROLES.items()
        ],
        "tokens_spent": sum(r["tokens_used"] for r in users),
        "cost_usd": round(sum(r["cost_usd"] for r in users), 4),
        "day_cost_usd": round(sum(r["day_cost_usd"] for r in users), 4),
        "month_cost_usd": round(sum(r["month_cost_usd"] for r in users), 4),
        "pricing": [{"model": k, "input_per_mtok": v["in"], "output_per_mtok": v["out"]}
                    for k, v in PRICING.items()],
        "capped": sum(1 for r in users
                      if r["token_limit"] or r["daily_limit"] or r["monthly_limit"]),
        "over_budget": sum(1 for r in users if r["over_budget"]),
    }


@router.get("/users", dependencies=[Depends(require("users"))])
def list_users() -> dict[str, Any]:
    return _snapshot()


@router.post("/users", dependencies=[Depends(require("users"))])
def create_user(body: NewUser) -> dict[str, Any]:
    try:
        user = directory.add_user(
            name=body.name, password=body.password, role=body.role,
            display=body.display, token_limit=body.token_limit, model=body.model,
            daily_limit=body.daily_limit, monthly_limit=body.monthly_limit,
        )
    except ValueError as exc:
        raise HTTPException(422, detail={"kind": "invalid", "message": str(exc)}) from exc
    return {"ok": True, "user": _row(user), **_snapshot()}


@router.patch("/users/{name}", dependencies=[Depends(require("users"))])
def update_user(name: str, body: LimitPatch) -> dict[str, Any]:
    """Set the budget, the model, or both. Absent fields are left alone."""
    user = None
    if any(v is not None for v in (body.token_limit, body.daily_limit, body.monthly_limit)):
        user = directory.set_limits(name, total=body.token_limit,
                                    daily=body.daily_limit, monthly=body.monthly_limit)
    if body.model is not None:
        try:
            user = directory.set_model(name, body.model)
        except ValueError as exc:
            raise HTTPException(422, detail={"kind": "invalid", "message": str(exc)}) from exc
    if user is None:
        raise HTTPException(404, detail={"kind": "missing", "message": f"no user {name!r}"})
    return {"ok": True, "user": _row(user), **_snapshot()}


@router.post("/users/{name}/reset-usage", dependencies=[Depends(require("users"))])
def reset_usage(name: str, window: str = "all") -> dict[str, Any]:
    """`window` is all | total | daily | monthly."""
    if window not in ("all", "total", "daily", "monthly"):
        raise HTTPException(422, detail={"kind": "invalid",
                                         "message": f"unknown window {window!r}"})
    user = directory.reset_usage(name, window)
    if user is None:
        raise HTTPException(404, detail={"kind": "missing", "message": f"no user {name!r}"})
    return {"ok": True, "user": _row(user), **_snapshot()}


@router.delete("/users/{name}", dependencies=[Depends(require("users"))])
def delete_user(name: str, me: User = Depends(current_user)) -> dict[str, Any]:
    """Remove an account, and sign out whatever sessions it had open.

    An operator cannot delete themselves. Not paternalism — the last
    administrator deleting their own account leaves a console nobody can
    administer, and the only way back is a redeploy.
    """
    if (name or "").strip().lower() == me.name:
        raise HTTPException(
            422, detail={"kind": "invalid",
                         "message": "you cannot delete the account you are signed in as"})
    if not directory.remove_user(name):
        raise HTTPException(404, detail={"kind": "missing", "message": f"no user {name!r}"})
    history.forget_user(name)
    return {"ok": True, **_snapshot()}
