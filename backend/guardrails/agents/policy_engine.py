"""The deterministic Policy Engine — the one place a final enforcement
action gets decided, for every specialist agent and the Supervisor alike.

Reintroduced deliberately, reversing the "agent is the final decision-maker"
premise Increments 2-12 were built on. Every agent's own DECIDE step still
runs a real judge call and still reasons genuinely over its own evidence —
nothing about that changes. What changes is what happens to its answer:

    Before   AgentDecision.action -> straight to the capability layer
    Now      AgentDecision.action -> PolicyEngine.decide() -> final_action
                                         -> the capability layer

The floor this combines against the recommendation is not a second opinion
on the same evidence the agent already reasoned over — it is the configured
policy action for the surface the agent found something on, read directly
from `config/policy.yaml` through the same `policy.get(...)` a deterministic
rail already reads. An agent recommending ALLOW cannot override a policy
that says a checksum-verified SSN must be masked; an agent recommending
BLOCK is always free to be more restrictive than the floor requires, because
more caution never needs permission from anything.

`GuardrailAction` has no analogue in `guardrails.types.Verdict` — REDACT and
ESCALATE exist here and not there, and `Verdict`'s own precedence ordering
is a locked safety invariant this module does not touch. `ACTION_RANK` below
is a parallel, agent-scoped ordering, not a reuse of the rail-level one.
"""

from __future__ import annotations

from .types import GuardrailAction, PolicyDecision

#: Most restrictive to least. ESCALATE is deliberately absent — it is not a
#: severity level alongside the other five, it is "no confident recommendation
#: was reached," and is handled as a special case in `decide()` rather than
#: ranked against them.
ACTION_RANK: dict[str, int] = {"ALLOW": 0, "FLAG": 1, "MASK": 2, "REDACT": 3, "BLOCK": 4}

#: Raw `config/policy.yaml` action strings, mapped to the six-action vocabulary
#: every agent already returns. "pass" and "regenerate"/"human_review" are not
#: literal `GuardrailAction` values — `regenerate` means "do not deliver this
#: as-is," which is what BLOCK already means at this layer, matching how
#: `types.action_verdict` treats it; `human_review` means exactly what
#: ESCALATE means.
_POLICY_ACTION_MAP: dict[str, GuardrailAction] = {
    "pass": "ALLOW", "allow": "ALLOW",
    "flag": "FLAG",
    "mask": "MASK",
    "redact": "REDACT",
    "block": "BLOCK", "regenerate": "BLOCK",
    "escalate": "ESCALATE", "human_review": "ESCALATE",
}


def floor_from_policy(policy_action: str) -> GuardrailAction:
    return _POLICY_ACTION_MAP.get(str(policy_action).strip().lower(), "BLOCK")


class PolicyEngine:
    """Stateless and deterministic — no judge call, no reasoning, a lookup
    and a comparison. `has_findings=False` means the agent's own tools found
    nothing to ground a floor in, so the floor is `ALLOW` and the agent's
    recommendation (capped at the six-action ceiling by `AgentDecision`
    itself) stands, whatever it was."""

    def decide(self, recommended_action: GuardrailAction, *, has_findings: bool,
              policy_action: str = "") -> PolicyDecision:
        floor: GuardrailAction = floor_from_policy(policy_action) if (
            has_findings and policy_action) else "ALLOW"

        # ESCALATE cannot be ranked against the other five by `ACTION_RANK` —
        # it is not a severity level, it is "defer to a human," which can
        # come from either side: the agent's own uncertainty, or a policy
        # floor configured with `human_review`/`escalate` (grounding's
        # `action_on_fail` genuinely has this option). Either source wins
        # outright, except that a *confident* floor from the other side
        # still stands against an uncertain agent — "the model was unsure"
        # does not un-find a checksum-verified SSN.
        if recommended_action == "ESCALATE" or floor == "ESCALATE":
            if recommended_action == "ESCALATE" and floor == "ESCALATE":
                final: GuardrailAction = "ESCALATE"
            elif recommended_action == "ESCALATE":
                final = "ESCALATE" if floor == "ALLOW" else floor
            else:  # floor == "ESCALATE", agent reached a real recommendation
                final = "ESCALATE"
        else:
            final = max([recommended_action, floor], key=lambda a: ACTION_RANK[a])

        overridden = final != recommended_action
        rationale = (
            f"deterministic floor ({floor}) overrode the agent's "
            f"recommendation ({recommended_action})" if overridden else
            f"agent's recommendation ({recommended_action}) upheld — "
            f"no stricter deterministic floor applied"
        )
        return PolicyDecision(final_action=final, recommended_action=recommended_action,
                              floor_action=floor, overridden=overridden, rationale=rationale)


def floor_from_agent_results(results) -> GuardrailAction:
    """The Supervisor's own floor: the most restrictive of what its selected
    agents' *own* Policy Engine already enforced. A reconciliation call can
    be more restrictive than every agent it is reconciling, but never less —
    the same "more caution needs no permission" rule, one level up.

    An agent that itself escalated contributes nothing to this floor —
    "unknown" is not a severity level `ACTION_RANK` can compare against the
    other five, and treating it as one would raise on the very next
    `PolicyEngine.decide()` call, which ranks a two-value list unconditionally.
    Only if *every* selected agent escalated, leaving nothing concrete to
    build a floor from, does `ALLOW` — the floor's own "nothing to enforce"
    default — stand.
    """
    actions = [r.outcome.action for r in results.values()
              if r.outcome is not None and r.outcome.action != "ESCALATE"]
    if not actions:
        return "ALLOW"
    return max(actions, key=lambda a: ACTION_RANK.get(a, ACTION_RANK["BLOCK"]))
