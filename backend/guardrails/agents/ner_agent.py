"""The autonomous local-NER agent — Presidio, with its own PLAN + DECIDE.

Nested under `PIIAgent`: where `pii_agent.py` used to call the
`detect_pii_presidio` tool as a flat, un-reasoned function — run the model,
return a dict, no reasoning of its own — it now delegates to this agent
instead. Same shape every other agent in this package already has:

    ANALYZE + PLAN   one judge call: is it worth running Presidio here at all
    SELECT + EXECUTE this agent's one tool — `detect_pii_presidio`,
                     `agents/tools.py`'s existing wrapper around
                     `presidio_ner.find`, called through the same dispatcher
                     (`tools.call`), not reimplemented
    OBSERVE + DECIDE a second judge call, given only what Presidio actually
                     found, recommending one of the six actions — final,
                     not a recommendation a floor can override (see below)
    ACT              `PIICapabilities.execute()` — the same capability layer
                     every other agent uses, not a second one

**No `PolicyEngine` floor here by default** — `agent.nested_model_floor`
(registry.py, default `"off"`), unlike every other agent in this package,
which has no toggle for this at all. `PIIAgent`'s own POLICY/ACT step
(unchanged, still floor-governed) is what actually decides what happens to
the text; with the floor off here, this agent's own decision is only ever
read as evidence for that outer call, never executed as the final word on
its own. Turning it off does not remove the floor from the request — it
removes a second, redundant enforcement point one level below the one that
actually matters, on the one agent in this package whose output is never
itself the last word. `AgentResult.policy_decision` is still populated
either way, for shape uniformity with every other agent's result.

Presidio only ever proposes PERSON/ADDRESS (`presidio_ner.KIND_MAP`) — its
silence on anything else is not evidence of anything, the same caveat
`PIIAgent`'s own DECIDE prompt already carries about this tool.

`PIIAgent` folds this agent's `AgentResult` into its own evidence trail as a
`ToolResult` — this agent's own POLICY/ACT still run, for its own complete,
audit-ready record, but it is `PIIAgent`'s own POLICY/ACT (unchanged) that
governs what actually happens to the text. That is the same relationship
`Supervisor` already has with each specialist it delegates to: a nested run
is real evidence for the parent's own DECIDE call, not a bypass of it.

Bounded tighter than the specialists above it on iterations and tool calls
(`max_iterations=2`, `max_tool_calls=2` vs. their 3/8) — there is exactly one
tool to plan around, not up to eight, so there is less to iterate over
before a decision is reachable. `timeout_s` is left at the same 30s the
specialists use, not tightened to match: a live run surfaced that Presidio's
own first-call cold start (downloading/loading its model) can by itself eat
well past a 15s budget, before either of this agent's two judge calls even
happen — a tighter timeout here bought no real safety margin, only a false
choice between "wait for the cold start" and "escalate needlessly."

Its own PLAN/DECIDE schemas deliberately use different field names than
`PIIAgent`'s (`needs_ner_scan` instead of `needs_analysis`; `ner_verdict`
instead of `action`) — the same reason `injection_agent.py`'s docstring
already states for why sibling agents in this package don't share field
names field-for-field: a test harness driving more than one agent through
one scripted model needs to be able to tell their schemas apart by shape.
Reusing `PIIAgent`'s exact field names here would make this agent's own
PLAN/DECIDE calls indistinguishable from its parent's to any such harness.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import ValidationError

from ..engine import PII_ACTION_KEY, Engine
from ..llm import LLMError
from ..prompts import judge_prompt
from ..types import Surface
from .capabilities import PIICapabilities
from .policy_engine import PolicyEngine
from .tools import ToolNotAllowed, call as call_tool
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

#: This agent's entire allowlist — one entry. Validated independently of
#: `tools.PII_AGENT_TOOLS` (which also holds `detect_pii_entities` and
#: `get_pii_policy`): the dispatcher below would technically permit those
#: too, but this tuple is what this agent's own PLAN schema and its own
#: `_execute` boundary check actually allow — the same double-boundary shape
#: (schema enum + Python allowlist) every agent in this package already uses.
NER_TOOL_NAMES: tuple[str, ...] = ("detect_pii_presidio",)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_ner_scan": {
            "type": "boolean",
            "description": "False only if the text plainly holds no capitalised "
                           "name-like span and no street address — nothing a local "
                           "NER pass could find.",
        },
        "tools": {
            "type": "array",
            "description": "Empty, or ['detect_pii_presidio'] — this agent has "
                           "exactly one tool.",
            "items": {"type": "string", "enum": list(NER_TOOL_NAMES)},
        },
        "more_evidence_needed": {
            "type": "boolean",
            "description": "Always false in practice — one tool has nothing left "
                           "to gather after it has run once.",
        },
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["needs_ner_scan", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "ner_verdict": {
            "type": "string",
            "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"],
        },
        "confidence": {"type": "number", "description": "0.0 to 1.0."},
        "rationale": {
            "type": "string",
            "description": "One sentence naming the evidence that decided it.",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "PERSON or ADDRESS."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array", "items": {"type": "string"},
                        "description": "call_id values from the tool result you were "
                                       "shown. Never invent one.",
                    },
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["ner_verdict", "confidence", "rationale", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a local-NER guardrail agent. You have exactly \
one tool:

- detect_pii_presidio   a local, trained NER model — finds PERSON and \
ADDRESS spans only, for nothing and in about a second, no API call. It is \
not built to find an SSN, an email, a phone number, or anything else with a \
fixed shape — its silence says nothing about those, only about names and \
addresses.

needs_ner_scan is true whenever a plausible person's name or a plausible \
address appears anywhere in the text — not only when the text is *about* a \
person, and not only when the name or address is the main subject of the \
sentence. A name or address mentioned once, in passing, alongside other \
content is still enough:

- "My name is Ravi Kumar and my address is 14 Anna Salai, Chennai." -> true \
(a clear name and a clear address, even though the sentence also states \
they belong to "my")
- "Please route this to Priya Nair in the billing team." -> true (a name, \
even though the sentence's main point is routing a request)
- "I renewed my licence at the Anna Nagar office last week." -> true \
("Anna Nagar" could be part of an address-like span; when unsure whether a \
capitalised span is a place name or a personal address, plan the tool \
rather than deciding for it — the tool costs nothing to run)
- "What documents do I need to renew a trade licence?" -> false (no \
capitalised name-like span, no address, nothing this model could find)
- "The fee schedule was last updated in March." -> false (a month name is \
not a person or an address)

Say needs_ner_scan is false only when you would need to invent a name or an \
address to justify running the tool — when the text genuinely has neither, \
not merely when a name or address isn't the point of the sentence. When in \
doubt, plan the tool: it is free, and a missed scan is worse than a wasted \
one.""")

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a local-NER guardrail agent. You are given the \
original text and exactly what Presidio returned — nothing more. Decide the \
one action.

- ALLOW     nothing found, or already masked
- MASK      a name or address should become a reversible vault token
- REDACT    should be removed, not recoverable
- BLOCK     not typically applicable to a bare name or address on its own
- FLAG      it may proceed, but a person should see it
- ESCALATE  the tool's own confidence is too low to act on either way

Presidio scores each span with a trained model's own confidence — weigh a \
low-confidence hit differently from a high-confidence one rather than \
treating every finding as equally certain. Cite only call_id values you were \
actually shown as evidence; never invent one.""")


class NERAgent:
    name = "ner_agent"
    version = "1.0.0"

    def __init__(self, llm: Any, engine: Engine, *,
                 max_iterations: int = 2, max_tool_calls: int = 2,
                 timeout_s: float = 30.0) -> None:
        self.llm = llm
        self.engine = engine
        self.capabilities = PIICapabilities(engine.entity_rail, engine.vault, engine.policy)
        self.policy_engine = PolicyEngine()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s

    # -----------------------------------------------------------------
    def run(self, text: str, *, surface: Surface = Surface.USER_PROMPT,
            owner: str = "", request_id: str = "") -> AgentResult:
        request_id = request_id or f"ner_agent_{uuid.uuid4().hex[:10]}"
        began = time.perf_counter()
        trace: list[TraceEvent] = []
        calls: list[ToolResult] = []
        plan: AgentPlan | None = None

        def elapsed_ms() -> float:
            return (time.perf_counter() - began) * 1000

        def note(phase: str, summary: str) -> None:
            trace.append(TraceEvent(phase=phase, summary=summary, at_ms=round(elapsed_ms(), 1)))

        def timed_out() -> bool:
            return elapsed_ms() > self.timeout_s * 1000

        try:
            for iteration in range(1, self.max_iterations + 1):
                if timed_out():
                    return self._escalate(
                        "ANALYZE", f"exceeded {self.timeout_s}s before a plan completed",
                        trace, calls, began, request_id)

                plan = self._plan(text, calls, note)

                if timed_out():
                    return self._escalate(
                        "PLAN", f"exceeded {self.timeout_s}s during the plan call",
                        trace, calls, began, request_id, plan)

                if not plan.needs_analysis:
                    if not calls:
                        note("DECIDE", "no name- or address-shaped content — nothing to analyze")
                        return self._finish(
                            "completed",
                            AgentDecision(action="ALLOW", confidence=1.0,
                                         rationale="no NER-relevant content found",
                                         findings=[]),
                            plan, trace, calls, began, request_id)
                    break

                budget_left = self.max_tool_calls - len(calls)
                if budget_left <= 0:
                    return self._escalate(
                        "EXECUTE", f"tool call budget ({self.max_tool_calls}) exhausted",
                        trace, calls, began, request_id, plan)

                round_calls, truncated = self._execute(plan.tools, text, budget_left, note)
                calls.extend(round_calls)

                if truncated:
                    return self._escalate(
                        "EXECUTE", f"tool call budget ({self.max_tool_calls}) exhausted "
                                   "before the plan finished running",
                        trace, calls, began, request_id, plan)

                if not plan.more_evidence_needed:
                    break
            else:
                return self._escalate(
                    "PLAN", f"exceeded {self.max_iterations} iterations without "
                            "the plan declaring itself done",
                    trace, calls, began, request_id, plan)

            if timed_out():
                return self._escalate(
                    "EVALUATE", f"exceeded {self.timeout_s}s before a decision completed",
                    trace, calls, began, request_id, plan)

            decision = self._decide(text, calls, note)

        except ToolNotAllowed:
            raise  # a programming error, not a runtime outcome — never masked as ESCALATE
        except LLMError as exc:
            return self._escalate("EVALUATE", f"agent could not reach a decision: {exc}",
                                  trace, calls, began, request_id, plan)
        except ValidationError:
            return self._escalate("EVALUATE", "malformed model output failed validation",
                                  trace, calls, began, request_id, plan)

        # `agent.nested_model_floor` (default "off"): this agent's own
        # decision is final for its own record, but never for the request
        # either way — `PIIAgent`'s own POLICY/ACT (unchanged, still floor-
        # governed) is what actually decides what happens to the text. "on"
        # restores the floor at this layer too. See the module docstring.
        if str(self.engine.policy.get("agent.nested_model_floor")) == "on":
            policy_action = str(self.engine.policy.get(PII_ACTION_KEY[surface]))
            policy_decision = self.policy_engine.decide(
                decision.action, has_findings=bool(decision.findings),
                policy_action=policy_action)
        else:
            policy_decision = PolicyDecision(
                final_action=decision.action, recommended_action=decision.action,
                floor_action=decision.action, overridden=False,
                rationale="no deterministic floor at this layer "
                          "(agent.nested_model_floor=off) — PIIAgent's own "
                          "POLICY step is what enforces one")
        note("POLICY", policy_decision.rationale)

        note("ACT", f"executing {policy_decision.final_action}")
        outcome = self.capabilities.execute(policy_decision.final_action, text, owner=owner)
        note("ACT", outcome.summary)

        return self._finish("completed", decision, plan, trace, calls, began, request_id,
                            outcome=outcome, policy_decision=policy_decision)

    # -----------------------------------------------------------------
    def _plan(self, text: str, prior_calls: list[ToolResult], note: Any) -> AgentPlan:
        evidence = "\n".join(f"{c.tool} ({c.status}): {c.result or c.error}"
                             for c in prior_calls) or "(none yet)"
        user = f"TEXT:\n{text}\n\nEVIDENCE SO FAR:\n{evidence}"
        raw = self.llm.judge(PLAN_SYSTEM, user, PLAN_SCHEMA)
        plan = AgentPlan.model_validate({
            "needs_analysis": raw.get("needs_ner_scan"),
            "tools": raw.get("tools", []),
            "more_evidence_needed": raw.get("more_evidence_needed", False),
            "rationale": raw.get("rationale", ""),
        })
        note("PLAN", plan.rationale or f"tools={list(plan.tools)}")
        return plan

    def _execute(self, tool_names: list[str], text: str, budget_left: int,
                note: Any) -> tuple[list[ToolResult], bool]:
        results: list[ToolResult] = []
        truncated = False

        for name in tool_names:
            if len(results) >= budget_left:
                note("EXECUTE", f"stopped at the tool budget mid-plan ({budget_left})")
                truncated = True
                break
            if name not in NER_TOOL_NAMES:
                raise ToolNotAllowed(name)

            res = call_tool(name, {"text": text}, self.engine, _call_id())
            results.append(res)
            note("EXECUTE", f"{name} -> {len(res.result.get('findings', []))} finding(s)"
                            if res.status == "ok" else f"{name} -> error: {res.error}")

        return results, truncated

    def _decide(self, text: str, calls: list[ToolResult], note: Any) -> AgentDecision:
        evidence = "\n".join(
            f"[{c.call_id}] {c.tool} ({c.status}): {c.result or c.error}" for c in calls
        ) or "(no tool returned anything)"
        user = f"TEXT:\n{text}\n\nTOOL RESULTS:\n{evidence}"
        raw = self.llm.judge(DECISION_SYSTEM, user, DECISION_SCHEMA)
        decision = AgentDecision.model_validate({
            "action": raw.get("ner_verdict"),
            "confidence": raw.get("confidence"),
            "rationale": raw.get("rationale", ""),
            "findings": raw.get("findings", []),
        })

        recorded = {c.call_id for c in calls}
        decision.findings = [f for f in decision.findings if set(f.evidence) <= recorded]
        note("EVALUATE", f"{len(decision.findings)} finding(s), "
                         f"confidence={decision.confidence:.2f}")
        note("DECIDE", f"{decision.action} — {decision.rationale}")
        return decision

    # -----------------------------------------------------------------
    def _escalate(self, phase: str, reason: str, trace: list[TraceEvent],
                  calls: list[ToolResult], began: float, request_id: str,
                  plan: AgentPlan | None = None) -> AgentResult:
        trace.append(TraceEvent(phase="ESCALATE", summary=reason,
                                at_ms=round((time.perf_counter() - began) * 1000, 1)))
        decision = AgentDecision(action="ESCALATE", confidence=0.0, rationale=reason,
                                 findings=[])
        policy_decision = self.policy_engine.decide("ESCALATE", has_findings=False)
        outcome = self.capabilities.execute(policy_decision.final_action, "", owner="")
        return self._finish("escalated", decision, plan, trace, calls, began, request_id,
                            outcome=outcome, policy_decision=policy_decision,
                            escalation_reason=reason)

    def _finish(self, status: str, decision: AgentDecision, plan: AgentPlan | None,
               trace: list[TraceEvent], calls: list[ToolResult], began: float,
               request_id: str, *, outcome: ActionOutcome | None = None,
               policy_decision: PolicyDecision | None = None,
               escalation_reason: str = "") -> AgentResult:
        return AgentResult(
            request_id=request_id, agent=self.name, version=self.version, status=status,
            decision=decision, plan=plan, tool_calls=calls, trace=trace,
            policy_decision=policy_decision, outcome=outcome,
            duration_ms=round((time.perf_counter() - began) * 1000, 1),
            escalation_reason=escalation_reason,
        )


_call_counter = 0


def _call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"ner_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
