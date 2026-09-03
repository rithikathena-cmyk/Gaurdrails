"""The autonomous local-classifier agent — Toxic-BERT, with its own PLAN + DECIDE.

Nested under `ContentSafetyAgent`: where `content_safety_agent.py` used to
call the `score_content_categories` tool as a flat, un-reasoned function, it
now delegates to this agent instead. Same shape every other agent in this
package already has:

    ANALYZE + PLAN   one judge call: is it worth scoring this text locally
    SELECT + EXECUTE this agent's one tool — `score_content_categories`,
                     `agents/content_tools.py`'s existing wrapper around
                     `toxicity_check.score`, called through the same
                     dispatcher, not reimplemented
    OBSERVE + DECIDE a second judge call, given only what the classifier
                     actually scored, recommending one of the six actions —
                     final, not a recommendation a floor can override
    ACT              `PIICapabilities.execute()` — the same capability layer
                     every other agent uses, not a second one

**No `PolicyEngine` floor here by default** — `agent.nested_model_floor`
(registry.py, default `"off"`); see `ner_agent.py`'s module docstring for
the full rationale, identical here: `ContentSafetyAgent`'s own POLICY/ACT
(unchanged, still floor-governed) is what actually decides what happens to
the text; with the floor off, this agent's own decision is only ever
evidence for that outer call.

Toxic-BERT only covers three of the six configured categories (hate,
violence, insults — see `toxicity_check.COVERED`); a category outside that
set was never evaluated by this tool, not cleared, and the DECIDE prompt
below is written to keep that distinction visible rather than letting a
"nothing scored" result read as "nothing wrong."

`ContentSafetyAgent` folds this agent's `AgentResult` into its own evidence
trail as a `ToolResult` — this agent's own POLICY/ACT still run, for its own
complete, audit-ready record, but it is `ContentSafetyAgent`'s own POLICY/ACT
(unchanged) that governs what actually happens to the text, the same
relationship `Supervisor` already has with each specialist it delegates to.

Its own PLAN/DECIDE schemas use field names distinct from every sibling
agent's (`needs_local_score`/`local_content_verdict`, not `needs_analysis`/
`action` or `possible_violation`/`judgment`) — the same reason
`injection_agent.py`'s docstring already states: a test harness driving more
than one agent through one scripted model needs to tell their schemas apart
by shape, not just by which agent instance happened to call it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import ValidationError

from ..engine import Engine
from ..llm import LLMError
from ..prompts import judge_prompt
from ..types import Surface
from .capabilities import PIICapabilities
from .content_tools import ToolNotAllowed, call as call_tool
from .policy_engine import PolicyEngine
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

#: This agent's entire allowlist — one entry, out of `content_tools`'s two
#: (`get_content_policy` stays the parent's own tool, not duplicated here).
CONTENT_MODEL_TOOL_NAMES: tuple[str, ...] = ("score_content_categories",)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_local_score": {
            "type": "boolean",
            "description": "False only if the text plainly holds nothing that could "
                           "touch hate, violence, or insults — including text that is "
                           "merely distressing.",
        },
        "tools": {
            "type": "array",
            "description": "Empty, or ['score_content_categories'] — this agent has "
                           "exactly one tool.",
            "items": {"type": "string", "enum": list(CONTENT_MODEL_TOOL_NAMES)},
        },
        "more_evidence_needed": {"type": "boolean"},
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["needs_local_score", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "local_content_verdict": {
            "type": "string",
            "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"],
        },
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "hate, violence, or insults."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["local_content_verdict", "confidence", "rationale", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a local content-classifier agent. You have \
exactly one tool:

- score_content_categories   a local classifier's own scores for hate, \
violence, and insults only — it cannot speak to misconduct or self-harm at \
all, and reports which categories it even covers.

A user describing something distressing that happened to them, reporting a \
crime, or expressing frustration at the service is not what this tool exists \
to catch — say needs_analysis is false and name no tool rather than running \
one to be sure.""")

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a local content-classifier agent. You are given \
the original text and exactly what the classifier scored — nothing more. \
Decide the one action.

- ALLOW     nothing here violates policy, however distressing the subject
- MASK      not typically applicable here
- REDACT    not typically applicable here
- BLOCK     the text itself expresses or seeks hate, violence, or insults \
aimed at a person
- FLAG      it may proceed, but a person should see it
- ESCALATE  the score alone does not clearly support any of the above

A classifier score is evidence of wording, not of intent: it can flag "this \
is killing me" as violence when it describes distress, not a threat. Do not \
let a high score alone stand in for a reading of what the text is actually \
doing. A category this tool does not cover was never evaluated by it — say \
so rather than treating its silence on misconduct or self-harm as a clean \
bill of health. Cite only call_id values you were actually shown as \
evidence; never invent one.""")


class ContentModelAgent:
    name = "content_model_agent"
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
        request_id = request_id or f"content_model_agent_{uuid.uuid4().hex[:10]}"
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
                        note("DECIDE", "no content-relevant concern — nothing to analyze")
                        return self._finish(
                            "completed",
                            AgentDecision(action="ALLOW", confidence=1.0,
                                         rationale="no content-safety concern found",
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
            raise
        except LLMError as exc:
            return self._escalate("EVALUATE", f"agent could not reach a decision: {exc}",
                                  trace, calls, began, request_id, plan)
        except ValidationError:
            return self._escalate("EVALUATE", "malformed model output failed validation",
                                  trace, calls, began, request_id, plan)

        # `agent.nested_model_floor` (default "off"): this agent's own
        # decision is final for its own record, but never for the request
        # either way — `ContentSafetyAgent`'s own POLICY/ACT (unchanged,
        # still floor-governed) is what actually decides what happens to the
        # text. "on" restores the floor at this layer too.
        if str(self.engine.policy.get("agent.nested_model_floor")) == "on":
            content_action_key = ("content.action.llm_response" if surface == Surface.LLM_RESPONSE
                                  else "content.action.user_prompt")
            policy_action = str(self.engine.policy.get(content_action_key))
            policy_decision = self.policy_engine.decide(
                decision.action, has_findings=bool(decision.findings),
                policy_action=policy_action)
        else:
            policy_decision = PolicyDecision(
                final_action=decision.action, recommended_action=decision.action,
                floor_action=decision.action, overridden=False,
                rationale="no deterministic floor at this layer "
                          "(agent.nested_model_floor=off) — ContentSafetyAgent's "
                          "own POLICY step is what enforces one")
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
            "needs_analysis": raw.get("needs_local_score"),
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
            if name not in CONTENT_MODEL_TOOL_NAMES:
                raise ToolNotAllowed(name)

            res = call_tool(name, {"text": text}, self.engine, _call_id())
            results.append(res)
            note("EXECUTE", f"{name} -> {'ok' if res.status == 'ok' else res.error}")

        return results, truncated

    def _decide(self, text: str, calls: list[ToolResult], note: Any) -> AgentDecision:
        evidence = "\n".join(
            f"[{c.call_id}] {c.tool} ({c.status}): {c.result or c.error}" for c in calls
        ) or "(no tool returned anything)"
        user = f"TEXT:\n{text}\n\nTOOL RESULTS:\n{evidence}"
        raw = self.llm.judge(DECISION_SYSTEM, user, DECISION_SCHEMA)
        decision = AgentDecision.model_validate({
            "action": raw.get("local_content_verdict"),
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
    return f"content_model_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
