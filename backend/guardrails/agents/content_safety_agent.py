"""The autonomous content-safety agent.

Same shape as `PIIAgent` and `PromptInjectionAgent`. Its schema field names
(`possible_violation`, `judgment`, `evidence_summary`) are close to the
injection agent's but not identical — `judgment` rather than `verdict` —
because both agents share the same six-action shape and a supervisor that
runs several specialists together needs one property name per agent to tell
their DECIDE calls apart, not just per pair.

The one thing worth naming: `judge_prompt(calibrate=True)` — the default —
already includes `USER_CALIBRATION`, the block every judge in this
codebase scoring what a person wrote already carries: distress, frustration,
anger at the service, and poor phrasing are ordinary traffic, not
signals. This agent's DECIDE prompt does not re-derive that calibration; it
inherits it, the same way `ContentRail`'s own judge call already does.
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
from .content_tools import CONTENT_TOOL_NAMES, ToolNotAllowed, call as call_tool
from .policy_engine import PolicyEngine
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "possible_violation": {
            "type": "boolean",
            "description": "False if the text plainly holds nothing that could touch "
                           "hate, violence, insults, misconduct, self-harm, or sexual "
                           "content — including text that is merely distressing.",
        },
        "tools": {
            "type": "array",
            "description": "Which of the allowed tools to run next. Empty if "
                           "possible_violation is false.",
            "items": {"type": "string", "enum": list(CONTENT_TOOL_NAMES)},
        },
        "more_evidence_needed": {"type": "boolean"},
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["possible_violation", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "judgment": {
            "type": "string",
            "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"],
        },
        "confidence": {"type": "number"},
        "evidence_summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The category, e.g. "
                              "hate, violence, insults, misconduct, self_harm, sexual."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["judgment", "confidence", "evidence_summary", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a content-safety guardrail agent. You are given \
one piece of text and a fixed list of tools you may ask to run.

- score_content_categories   a local classifier's own scores for hate, \
violence, insults — it does not cover misconduct or self-harm, and returns \
which categories it can even speak to
- get_content_policy         the configured threshold and enabled categories

Most requests worth a second look need score_content_categories. A user \
describing something distressing that happened to them, reporting a crime, \
or expressing frustration at the service is not what this agent exists \
to catch — say so and name no tools rather than running one to be sure.""")

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a content-safety guardrail agent. You are \
given the original text and exactly what the tools you asked for actually \
returned. Decide the one action that should happen.

- ALLOW     nothing here violates policy, however distressing the subject
- MASK      not typically applicable here
- REDACT    not typically applicable here
- BLOCK     the text itself expresses or seeks hate, violence, insults \
aimed at a person, help committing a crime, an intent toward self-harm, or \
sexual content out of place here
- FLAG      it may proceed, but a person should see it
- ESCALATE  the evidence does not clearly support any of the above

The distinction that matters most: discussion, reporting, and help-seeking \
are not the thing they describe. A user who says a decision is "killing me" \
is describing distress, not planning violence. Someone reporting they were \
defrauded is not seeking to defraud anyone. Someone quoting abuse they \
received is not producing it. Someone naming a disability, an eviction, a \
bereavement, or a debt is describing their situation, not violating policy \
by having one. Score the intent of the text, not how upsetting the subject \
is, and do not let a local classifier's score override that reading — a \
classifier that flags "killing me" as violence is scoring words, not intent.

Cite only call_id values you were actually shown as evidence; never invent \
one.""")


class ContentSafetyAgent:
    name = "content_safety_agent"
    version = "1.0.0"

    def __init__(self, llm: Any, engine: Engine, *,
                 max_iterations: int = 3, max_tool_calls: int = 8,
                 timeout_s: float = 30.0) -> None:
        self.llm = llm
        self.engine = engine
        self.capabilities = PIICapabilities(engine.pii_rail, engine.vault, engine.policy)
        self.policy_engine = PolicyEngine()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s

    # -----------------------------------------------------------------
    def run(self, text: str, *, surface: Surface = Surface.USER_PROMPT,
            owner: str = "", request_id: str = "") -> AgentResult:
        request_id = request_id or f"content_agent_{uuid.uuid4().hex[:10]}"
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
                    # Evidence already gathered in an earlier round — `needs_analysis`
                    # going false here means "no more evidence needed," not "this was
                    # never relevant." Fall through to DECIDE with what was already
                    # gathered; never discard it and never skip POLICY/ACT.
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

        content_action_key = ("content.action.llm_response" if surface == Surface.LLM_RESPONSE
                              else "content.action.user_prompt")
        policy_action = str(self.engine.policy.get(content_action_key))
        policy_decision = self.policy_engine.decide(
            decision.action, has_findings=bool(decision.findings), policy_action=policy_action)
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
            "needs_analysis": raw.get("possible_violation"),
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
            if name not in CONTENT_TOOL_NAMES:
                raise ToolNotAllowed(name)

            args = {"text": text}
            res = call_tool(name, args, self.engine, _call_id())
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
            "action": raw.get("judgment"),
            "confidence": raw.get("confidence"),
            "rationale": raw.get("evidence_summary", ""),
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
    return f"content_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
