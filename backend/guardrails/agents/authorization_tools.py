"""The authorization agent's tool allowlist.

`guardrails/` never imports `server/` — that boundary predates this agent and
is not something this increment gets to cross. `server/auth.py` owns roles
and permissions; `Directory`, `ROLES`, and `PERMISSIONS` stay exactly where
they are. What crosses the boundary is data, not code: the caller (in
practice, a route that has already resolved the signed-in user via
`server/auth.py`) hands the agent an `AuthorizationContext` — the same role
and permission set `require()` already enforces — and these tools are
read-only lookups over that supplied context. Nothing here re-derives a
role from a session cookie, because nothing here can see one.

`get_resource_classification` is the one genuinely new thing in this file —
this codebase has no resource-classification system to reuse, so this is a
small, explicitly illustrative lookup table, not a production entitlement
registry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .tools import ToolNotAllowed
from .types import ToolResult


@dataclass(frozen=True)
class AuthorizationContext:
    """Deterministic facts, resolved by the caller before the agent runs.

    The agent never mutates this and never asks for more than it was given —
    there is no tool here that looks anything up beyond what is already on
    this object. That is the whole point: the agent reasons about what the
    request is asking for, not about who the caller claims to be.
    """

    principal: str
    role: str
    permissions: frozenset[str] = field(default_factory=frozenset)
    resource_kind: str = ""
    resource_owner: str = ""

    @property
    def is_owner(self) -> bool:
        return bool(self.resource_owner) and self.resource_owner == self.principal

    @property
    def entitled(self) -> bool:
        """Owns the resource, or holds a permission that overrides ownership."""
        return self.is_owner or "admin" in self.permissions or not self.resource_owner


#: Illustrative only — see the module docstring. Real classification is
#: future work (the "Phase B: document classification" this project's own
#: notes already named and deferred), not something this file invents in
#: production earnest.
_RESOURCE_CLASSIFICATION: dict[str, str] = {
    "own_record": "owner_only",
    "case_file": "restricted",
    "published_contact": "public",
    "claim_status": "owner_only",
    "audit_log": "admin_only",
    "policy_config": "admin_only",
}


@dataclass(frozen=True)
class GuardrailTool:
    name: str
    fn: Callable[[dict, AuthorizationContext], dict]


def _get_user_role(args: dict, ctx: AuthorizationContext) -> dict:
    return {"role": ctx.role}


def _get_user_permissions(args: dict, ctx: AuthorizationContext) -> dict:
    return {"permissions": sorted(ctx.permissions)}


def _get_resource_classification(args: dict, ctx: AuthorizationContext) -> dict:
    kind = str(args.get("resource_kind") or ctx.resource_kind or "").strip()
    return {"resource_kind": kind,
           "classification": _RESOURCE_CLASSIFICATION.get(kind, "unclassified")}


def _check_permission(args: dict, ctx: AuthorizationContext) -> dict:
    permission = str(args.get("permission", "")).strip()
    return {"permission": permission, "held": permission in ctx.permissions}


def _check_ownership(args: dict, ctx: AuthorizationContext) -> dict:
    return {"principal": ctx.principal, "resource_owner": ctx.resource_owner,
           "is_owner": ctx.is_owner, "entitled": ctx.entitled}


AUTHORIZATION_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "get_user_role": GuardrailTool("get_user_role", _get_user_role),
    "get_user_permissions": GuardrailTool("get_user_permissions", _get_user_permissions),
    "get_resource_classification": GuardrailTool("get_resource_classification", _get_resource_classification),
    "check_permission": GuardrailTool("check_permission", _check_permission),
    "check_ownership": GuardrailTool("check_ownership", _check_ownership),
}

AUTHORIZATION_TOOL_NAMES: tuple[str, ...] = tuple(AUTHORIZATION_AGENT_TOOLS)


def call(name: str, args: dict, ctx: AuthorizationContext, call_id: str) -> ToolResult:
    tool = AUTHORIZATION_AGENT_TOOLS.get(name)
    if tool is None:
        raise ToolNotAllowed(name)
    began = time.perf_counter()
    try:
        result = tool.fn(args, ctx)
        return ToolResult(call_id=call_id, tool=name, status="ok",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          result=result)
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        return ToolResult(call_id=call_id, tool=name, status="error",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          error=f"{type(exc).__name__}: {exc}")
