"""The `GuardrailSupervisor` MVP's forbidden-capability boundary.

A parallel, spec-exact vocabulary to `PIICapabilities.FORBIDDEN`
(`capabilities.py`) — not a replacement for it. That set has thirteen names,
some overlapping this one, some not (`modify_overrides`, `change_role`,
`modify_tool_allowlist`), and 40+ existing tests already assert against it by
name; renaming it to match a different spec would break passing tests for no
functional gain. This module exists so `guardrail_supervisor.py` can be
tested against the literal eleven names an autonomous-guardrail-supervisor
spec enumerates, without touching that file.

Both modules raise the same `CapabilityDenied` — imported here, not
redefined — so a caller catching one boundary's denial catches the other's
too.
"""

from __future__ import annotations

from .capabilities import CapabilityDenied
from .types import ActionOutcome, GuardrailAction

#: Named things the guardrail_supervisor must never be able to reach.
#: Listed explicitly, by name, so a test can assert each one individually
#: rather than by absence — the same reasoning `PIICapabilities.FORBIDDEN`
#: documents for its own set.
FORBIDDEN_CAPABILITIES: frozenset[str] = frozenset({
    "modify_policy", "modify_rbac", "grant_permission", "reveal_secret",
    "reveal_vault", "execute_code", "filesystem_access", "database_access",
    "modify_audit_log", "bypass_approval", "disable_guardrails",
})

#: The six values a decision may execute — lower-cased, mapped back to the
#: `GuardrailAction` literal. Nothing outside this map and outside
#: `FORBIDDEN_CAPABILITIES` resolves to anything: an unrecognized name is
#: denied by the same default `request()` below applies to a listed one.
_ACTION_MAP: dict[str, GuardrailAction] = {
    a.lower(): a for a in ("ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE")
}


def deny_if_forbidden(capability: str) -> None:
    """Raise `CapabilityDenied` for anything in `FORBIDDEN_CAPABILITIES`.

    Called before a model's plan, decision, or any tool argument is acted on
    — regardless of whether the request for it came from the user, from
    text a tool returned, or from prompt injection asking for it. There is
    no code path in `guardrail_supervisor.py` that checks *who* is asking
    before this runs; the denial is unconditional.
    """
    name = str(capability).strip().lower()
    if name in FORBIDDEN_CAPABILITIES:
        raise CapabilityDenied(name)


def request(capability: str) -> ActionOutcome:
    """The generic form, for boundary tests: ask for anything by name.

    Six names succeed, because they are `GuardrailAction` values
    lower-cased. Every other name — every entry in `FORBIDDEN_CAPABILITIES`
    and anything not on either list — denies. There is no default-allow
    path: a name this module has never heard of is exactly as denied as one
    explicitly listed.
    """
    deny_if_forbidden(capability)
    name = str(capability).strip().lower()
    action = _ACTION_MAP.get(name)
    if action is None:
        raise CapabilityDenied(capability)
    return ActionOutcome(action=action, capability=f"request:{name}",
                         summary=f"capability {name!r} resolves to {action}")
