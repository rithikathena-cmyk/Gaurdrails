"""The authorization agent's capability layer.

Wraps `PIICapabilities` for the ordinary five actions — nothing about
BLOCK, FLAG, MASK, REDACT, or ESCALATE is authorization-specific. `ALLOW` is
the one exception, and it is exactly the pattern the vault example in this
project's own design already established:

    Agent wants:          reveal vault data
    Capability boundary:  not permitted
    Result:                DENY capability

Here: the agent recommends ALLOW — let this request through — and the
capability layer checks the one deterministic fact that matters before
letting anything through: is this caller actually entitled to the resource.
`AuthorizationContext.entitled` is supplied by the caller, computed from
`server/auth.py`'s own role and ownership data, never from anything the
agent said. This is not a second decision on the same axis the agent
already decided (mask vs. block vs. flag stays entirely the agent's call);
it is a hard floor under the one action — ALLOW — that would actually
expose something.
"""

from __future__ import annotations

from .authorization_tools import AuthorizationContext
from .capabilities import CapabilityDenied, PIICapabilities
from .types import ActionOutcome, GuardrailAction


class AuthorizationCapabilities:
    def __init__(self, pii_rail, vault, policy: object = None) -> None:
        self._base = PIICapabilities(pii_rail, vault, policy)

    def execute(self, action: GuardrailAction, text: str, *,
               ctx: AuthorizationContext) -> ActionOutcome:
        if action == "ALLOW" and not ctx.entitled:
            return ActionOutcome(
                action="BLOCK", capability="entitlement_denied", text_out="",
                summary=f"agent recommended ALLOW, but {ctx.principal!r} is not "
                        f"entitled to {ctx.resource_kind or 'this resource'} "
                        f"owned by {ctx.resource_owner!r}",
            )
        return self._base.execute(action, text, owner=ctx.principal)

    def resolve_for_reader(self, text: str, reader: str) -> tuple[str, int]:
        return self._base.resolve_for_reader(text, reader)

    def request(self, capability: str, **kwargs: object) -> ActionOutcome:
        return self._base.request(capability, **kwargs)


__all__ = ["AuthorizationCapabilities", "CapabilityDenied"]
