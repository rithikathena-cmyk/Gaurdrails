"""The capability layer: the agent decides WHAT, this decides WHETHER, and
only then does execution happen — HOW.

    PIIAgent:  decision = MASK
                   |
                   v
    PIICapabilities.execute("MASK", ...)
        deterministic membership check  <- WHETHER
        vault/token handling            <- HOW, reusing the production rail
        returns ActionOutcome

The six values `GuardrailAction` can hold are the *entire* vocabulary this
layer understands. There is no path from an agent's decision to anything not
in that vocabulary — not by naming a different action, not by putting
something unexpected in `findings`, not by any field on `AgentDecision`.
`request()` below is the explicit, deliberately narrow escape hatch for
testing that boundary: it accepts an arbitrary capability name and denies
everything that is not one of the six safe actions, by construction, so the
question "can an agent reach X" has one place to answer it rather than one
per forbidden thing.

`policy` connects this layer to the same adjustable-parameter registry every
deterministic rail already reads — three keys, all under `pii.*`, all
enforced here rather than trusted from the agent's own reasoning:

    pii.agent.allow_masked_pii_response   MASK may be refused outright,
                                           fail-closed to ESCALATE
    pii.agent.preserve_masked_tokens      a masked response's token is real
                                           and reversible, or a dead marker
    pii.vault.resolution                  whether `resolve_for_reader` below
                                           may ever try to reveal one

None of the three can make an agent's own decision *more* permissive than it
already was: `allow_masked_pii_response=False` only ever escalates a MASK
that was already going to happen; `preserve_masked_tokens=False` only ever
destroys reversibility a MASK already established; `pii.vault.resolution` only
ever gates a reveal attempt shut, never open — resolution still runs through
`Vault.reveal`'s own owner check regardless of what this policy says.
"""

from __future__ import annotations

import re

from ..rails.pii import PIIRail, Vault
from ..types import RailResult, Verdict
from .types import ActionOutcome, GuardrailAction

#: The exact shape `Engine.converse()`'s own egress stage matches at
#: `engine.py`'s `vault.unmask` rail — kept identical on purpose, so a token
#: this layer produces and one the deterministic pipeline produces are
#: interchangeable to whatever reads them back later.
_TOKEN_RE = re.compile(r"<([A-Z_0-9]+):([0-9a-f]{12})(?:\s…[^>]*)?>")


class CapabilityDenied(PermissionError):
    """Raised for anything outside the six safe actions this layer executes.

    Deliberately a `PermissionError`, not a generic exception — a caller that
    forgets to catch it fails loudly rather than treating a denial as an
    ordinary control-flow branch.
    """

    def __init__(self, capability: str) -> None:
        super().__init__(f"'{capability}' is not a capability this layer grants")
        self.capability = capability


class PIICapabilities:
    """What a PII agent's ACT phase is actually permitted to do.

    Bound to one engine's real `pii_rail` and `vault` — the same objects
    every ordinary request masks through — so `MASK` here produces the exact
    token format, vault entry, and owner semantics a rail-driven request
    would, not a parallel implementation an operator has to reason about
    twice.
    """

    def __init__(self, pii_rail: PIIRail, vault: Vault, policy: object = None) -> None:
        self.pii_rail = pii_rail
        self.vault = vault
        #: Optional on purpose — every test that only exercises the six-action
        #: boundary (`request()`, `FORBIDDEN`) has no policy to thread through
        #: and does not need one; `_policy_bool`/`_policy_str` below treat a
        #: missing policy as "nothing configured, behave as already shipped."
        self.policy = policy

    def execute(self, action: GuardrailAction, text: str, *, owner: str) -> ActionOutcome:
        if action in ("MASK", "REDACT"):
            return self._mask(action, text, owner)
        if action == "BLOCK":
            return ActionOutcome(action=action, capability="block_request",
                                 summary="request refused, nothing generated")
        if action in ("ALLOW", "FLAG"):
            return ActionOutcome(action=action, capability="pass_through", text_out=text,
                                 summary="request continues" + (" — flagged for review"
                                                                if action == "FLAG" else ""))
        if action == "ESCALATE":
            return ActionOutcome(action=action, capability="human_review",
                                 summary="handed to a person, not decided automatically")
        # Unreachable while `GuardrailAction` is the six-member Literal it is —
        # this is the floor beneath that type check, not a substitute for it.
        raise CapabilityDenied(action)

    def _policy_bool(self, key: str, default: bool) -> bool:
        return default if self.policy is None else bool(self.policy.get(key))

    def _policy_str(self, key: str, default: str) -> str:
        return default if self.policy is None else str(self.policy.get(key))

    def _mask(self, action: GuardrailAction, text: str, owner: str) -> ActionOutcome:
        if action == "MASK" and not self._policy_bool("pii.agent.allow_masked_pii_response", True):
            # Fail closed, not open: a deployment that will not let this path
            # hand back a token gets a person instead of a silently upgraded
            # or downgraded action — the same shape `AuthorizationCapabilities`
            # already uses to deny an ALLOW it cannot execute.
            return ActionOutcome(
                action="ESCALATE", capability="human_review",
                summary="a masked PII response is disabled by policy "
                        "(pii.agent.allow_masked_pii_response) — handed to a person")

        result = RailResult(rail="agents.pii.act", engine="pii-agent · vault-token",
                            verdict=Verdict.PASS)
        # The exact call `Engine.evaluate` makes for every ordinary request —
        # no separate masking logic exists for the agentic path.
        out = self.pii_rail.evaluate(text, "mask", result, owner)
        masked = out.text_out if out.text_out is not None else text

        if action == "MASK" and not self._policy_bool("pii.agent.preserve_masked_tokens", True):
            # The rail already minted real vault entries above — this only
            # changes what the *response* carries forward, not the vault
            # itself, so the detections already recorded stand either way.
            masked = _TOKEN_RE.sub("[REDACTED]", masked)

        return ActionOutcome(action=action, capability="mask_pii", text_out=masked,
                             tokens_masked=len(out.detections),
                             summary=f"{len(out.detections)} value(s) masked")

    def resolve_for_reader(self, text: str, reader: str) -> tuple[str, int]:
        """The agentic path's own egress entitlement check — the same shape
        `Engine.converse()`'s own `vault.unmask` stage already runs for every
        ordinary chat reply, applied here to an agent's `outcome.text_out`.

        Gated by `pii.vault.resolution`, but the gate only ever closes the
        door further: even with resolution allowed, `Vault.reveal` is the one
        place ownership is actually checked, and a token this `reader` did
        not mint comes back unchanged, never denied with an error — the
        response stays deliverable, it just does not carry someone else's
        value. Never call this with the text's *original* speaker assumed to
        be the reader; pass whoever is actually about to see the response.
        """
        if self._policy_str("pii.vault.resolution", "owner_only") == "never":
            return text, 0

        revealed = 0

        def _reveal(m: re.Match) -> str:
            nonlocal revealed
            val = self.vault.reveal(m.group(2), reader)
            if val is None:
                return m.group(0)
            revealed += 1
            return val

        return _TOKEN_RE.sub(_reveal, text), revealed

    #: Named things an agent must never be able to reach through this layer.
    #: Listed explicitly rather than left implicit, so a test can assert each
    #: one by name instead of by absence.
    FORBIDDEN = frozenset({
        "reveal_vault", "modify_policy", "modify_overrides", "modify_rbac",
        "grant_permission", "change_role", "disable_guardrail",
        "modify_tool_allowlist", "execute_code", "filesystem_access",
        "database_access", "modify_audit_log", "bypass_approval",
    })

    def request(self, capability: str, **_: object) -> ActionOutcome:
        """The generic form, for boundary tests: ask for anything by name.

        Six names succeed, because they are `GuardrailAction` values lower-
        cased; every other name — including every entry in `FORBIDDEN` and
        anything not on either list — denies. There is no default-allow path.
        """
        mapping = {a.lower(): a for a in ("ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE")}
        if capability in mapping:
            return self.execute(mapping[capability], "", owner="")
        raise CapabilityDenied(capability)
