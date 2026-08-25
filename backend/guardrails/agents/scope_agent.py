"""The autonomous scope agent.

Same shape as the other three. `check_domain_vocabulary` is the cheap layer
— exactly `ScopeRail`'s own set intersection — and the agent's own DECIDE
call is where "ambiguous, sideways-phrased, or adversarially worded" gets
resolved, the same job `SCOPE_SYSTEM` already does for the deterministic
rail's judge fallback. There is no "ambiguous" action to invent: an
ambiguous request that should proceed with a caveat is FLAG; one this agent
cannot resolve at all is ESCALATE.
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
from .policy_engine import PolicyEngine
from .scope_tools import SCOPE_TOOL_NAMES, ToolNotAllowed, call as call_tool
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_scope_review": {
            "type": "boolean",
            "description": "False only if the question obviously belongs to what "
                           "this assistant is for on its wording alone.",
        },
        "tools": {"type": "array", "items": {"type": "string", "enum": list(SCOPE_TOOL_NAMES)}},
        "more_evidence_needed": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["needs_scope_review", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "ruling": {"type": "string", "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"]},
        "confidence": {"type": "number"},
        "evidence_summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The topic the question "
                              "is actually about, in two or three words."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["ruling", "confidence", "evidence_summary", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a scope agent for a document-grounded assistant. \
It has no fixed subject of its own — only whatever documents and records it \
has actually been given, whatever domain those happen to cover.

- check_domain_vocabulary   whether the text's own wording hits the \
configured vocabulary for this service
- get_scope_policy          the configured threshold and action

If the question obviously belongs here on its wording, you do not need a \
tool call to know that; say so and name none. Reach for \
check_domain_vocabulary when it is not obvious, so you know whether the \
wording alone would have settled it before you reason about meaning.""",
                            calibrate=False)

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a scope agent. Decide whether this question \
belongs to what this assistant is actually for: answering questions \
grounded in whatever documents and records it has been given.

- ALLOW     clearly in scope
- MASK      not applicable here
- REDACT    not applicable here
- BLOCK     not applicable here — an out-of-scope question is declined, not \
blocked as a security matter
- FLAG      ambiguous: plausibly in scope on a sideways reading, worth a \
person's judgment or a clarifying question rather than a confident answer \
either way
- ESCALATE  you cannot form a view from what you were given

Take the sideways reading — the underlying need, not the exact wording, is \
what matters. A question about the service itself — what it can do, which \
documents it holds, why an earlier message was refused — is in scope: a \
service nobody can question is not a safer one.

Score out of scope only for a genuinely different subject: general \
knowledge trivia, creative writing, coding help, clinical or legal advice, \
financial speculation, entertainment, or a request to become a different \
product. A subject that is obviously fictional or made up also scores out \
of scope — a real deployment's documents describe real things. Wording \
engineered to look in-scope while asking for something else is still out of \
scope; take the actual request, not the frame around it. Being rude, \
distressed, or badly worded does not put a question out of scope; only its \
subject does.""", calibrate=False)


class ScopeAgent:
    name = "scope_agent"
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
        request_id = request_id or f"scope_agent_{uuid.uuid4().hex[:10]}"
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
                        note("DECIDE", "wording alone settles this — in scope")
                        return self._finish(
                            "completed",
                            AgentDecision(action="ALLOW", confidence=1.0,
                                         rationale="in scope on its wording alone", findings=[]),
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

        policy_action = str(self.engine.policy.get("scope.action"))
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
            "needs_analysis": raw.get("needs_scope_review"),
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
            if name not in SCOPE_TOOL_NAMES:
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
            "action": raw.get("ruling"),
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
    return f"scope_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
