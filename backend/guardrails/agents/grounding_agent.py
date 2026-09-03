"""The autonomous grounding agent.

    ANSWER
      |
      v
    CLAIM IDENTIFICATION      extract_claims — groundedness_check.claims()
      |
      v
    RETRIEVED EVIDENCE        the chunks the caller supplied, bounded to
      |                       grounding.context_window before any tool sees them
      v
    GROUNDING AGENT           PLAN + DECIDE, same shape as every other agent
      |
      v
    CLAIM-LEVEL DECISIONS     check_local_entailment maps each claim to
      |                       whichever chunk entails it, same as the
      |                       production rail's own local layer
      v
    FINAL AGENT DECISION      one of the six actions — see below on REGENERATE
      |
      v
    ACT

No retrieved context is an architectural no-op, exactly as it is in
`GroundingRail` — checked before the PLAN call runs at all, not decided by
one, because there is nothing a judge call could usefully say about an
answer with no context to check it against. This is the one latency
optimisation applied unconditionally rather than left to the model, and it
mirrors code the deterministic rail already has, not a new rule.

`GuardrailAction` has no `REGENERATE`. The rest of this architecture never
adds a seventh action, and grounding does not get an exception: an
insufficiently grounded answer that should be regenerated is `BLOCK` — the
same mapping the underlying `GroundingRail` already makes implicitly, since
the rail's own verdict is `block`, and *regeneration* is the engine's retry
loop layered on top of that verdict, not a verdict in its own right.
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
from .grounding_model_agent import GroundingModelAgent
from .grounding_tools import GROUNDING_TOOL_NAMES, ToolNotAllowed, call as call_tool
from .policy_engine import PolicyEngine
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_grounding_review": {
            "type": "boolean",
            "description": "False if the answer asserts nothing worth checking — "
                           "a purely conversational reply with no factual claims.",
        },
        "tools": {"type": "array", "items": {"type": "string", "enum": list(GROUNDING_TOOL_NAMES)}},
        "more_evidence_needed": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["needs_grounding_review", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "grounding_verdict": {
            "type": "string", "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"],
        },
        "confidence": {"type": "number"},
        "evidence_summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The unsupported or "
                              "contradicted claim, or 'grounded' if none."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["grounding_verdict", "confidence", "evidence_summary", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a grounding agent. You check a generated \
answer against the context it was retrieved from.

- extract_claims           split the answer into checkable sentences
- check_local_entailment    a local model's opinion on which chunk, if any, \
supports each claim — evidence, not a verdict
- get_grounding_policy      the configured thresholds

A short conversational reply with no factual assertions needs no tool call \
— say so. An answer making specific claims — a figure, a fee, a deadline, an \
eligibility rule, an office, a form name — needs extract_claims at minimum, \
usually followed by check_local_entailment.""", calibrate=False)

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a grounding agent. You are given the question, \
the retrieved context, the generated answer, and exactly what the tools \
returned. Decide the one action.

- ALLOW     every claim is supported by the context, or the answer makes none
- MASK      not applicable here
- REDACT    not applicable here
- BLOCK     the answer asserts something the context does not support, or \
contradicts — this is the case that would be regenerated, since there is no \
seventh action for that; treat BLOCK here as "do not deliver this answer as is"
- FLAG      mostly grounded, one minor claim uncertain — proceed, a person reviews
- ESCALATE  you cannot form a view from what you were given

A claim is unsupported if the context contradicts it, or if it asserts a \
specific fact absent from the context entirely — never fill a gap from your \
own knowledge to decide a claim is fine. Generic framing, hedging, and offers \
to help further are not claims and are never unsupported. An answer that \
correctly says the context does not have something is fully grounded. The \
local entailment tool's score is evidence, not the verdict — it scores \
wording overlap and can mark a plausible-sounding but wrong figure as \
entailed, so a claim it marks supported that asserts a fact you cannot \
actually find in the context is still unsupported.

Cite only call_id values you were actually shown as evidence; never invent \
one.""", calibrate=False)


class GroundingAgent:
    name = "grounding_agent"
    version = "1.0.0"

    def __init__(self, llm: Any, engine: Engine, *,
                 max_iterations: int = 3, max_tool_calls: int = 8,
                 timeout_s: float = 30.0) -> None:
        self.llm = llm
        self.engine = engine
        self.capabilities = PIICapabilities(engine.entity_rail, engine.vault, engine.policy)
        self.policy_engine = PolicyEngine()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s

    # -----------------------------------------------------------------
    def run(self, answer: str, *, question: str = "", chunks: list[str] | None = None,
            surface: Surface = Surface.LLM_RESPONSE, owner: str = "",
            request_id: str = "") -> AgentResult:
        request_id = request_id or f"grounding_agent_{uuid.uuid4().hex[:10]}"
        began = time.perf_counter()
        trace: list[TraceEvent] = []
        calls: list[ToolResult] = []
        plan: AgentPlan | None = None
        chunks = list(chunks or [])

        def elapsed_ms() -> float:
            return (time.perf_counter() - began) * 1000

        def note(phase: str, summary: str) -> None:
            trace.append(TraceEvent(phase=phase, summary=summary, at_ms=round(elapsed_ms(), 1)))

        def timed_out() -> bool:
            return elapsed_ms() > self.timeout_s * 1000

        # Architectural no-op, checked before any judge call — see the module
        # docstring. Not a latency shortcut this agent decided on its own;
        # the same rule the deterministic rail already enforces in code.
        if not chunks:
            note("ANALYZE", "no retrieved context — nothing to ground against")
            note("DECIDE", "ALLOW — architectural no-op, not evaluated")
            policy_decision = self.policy_engine.decide("ALLOW", has_findings=False)
            note("POLICY", policy_decision.rationale)
            outcome = self.capabilities.execute(policy_decision.final_action, answer, owner=owner)
            note("ACT", outcome.summary)
            return self._finish(
                "completed",
                AgentDecision(action="ALLOW", confidence=1.0,
                             rationale="no retrieved context — rail is retrieval-scoped",
                             findings=[]),
                None, trace, calls, began, request_id, outcome=outcome,
                policy_decision=policy_decision)

        try:
            for iteration in range(1, self.max_iterations + 1):
                if timed_out():
                    return self._escalate(
                        "ANALYZE", f"exceeded {self.timeout_s}s before a plan completed",
                        trace, calls, began, request_id)

                plan = self._plan(question, answer, calls, note)

                if timed_out():
                    return self._escalate(
                        "PLAN", f"exceeded {self.timeout_s}s during the plan call",
                        trace, calls, began, request_id, plan)

                if not plan.needs_analysis:
                    if not calls:
                        note("DECIDE", "no checkable claims — allowed")
                        return self._finish(
                            "completed",
                            AgentDecision(action="ALLOW", confidence=1.0,
                                         rationale="no factual claims to ground", findings=[]),
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

                round_calls, truncated = self._execute(
                    plan.tools, answer, question, chunks, surface, owner, budget_left, note)
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

            decision = self._decide(question, answer, chunks, calls, note)

        except ToolNotAllowed:
            raise
        except LLMError as exc:
            return self._escalate("EVALUATE", f"agent could not reach a decision: {exc}",
                                  trace, calls, began, request_id, plan)
        except ValidationError:
            return self._escalate("EVALUATE", "malformed model output failed validation",
                                  trace, calls, began, request_id, plan)

        policy_action = str(self.engine.policy.get("grounding.action_on_fail"))
        policy_decision = self.policy_engine.decide(
            decision.action, has_findings=bool(decision.findings), policy_action=policy_action)
        note("POLICY", policy_decision.rationale)

        note("ACT", f"executing {policy_decision.final_action}")
        outcome = self.capabilities.execute(policy_decision.final_action, answer, owner=owner)
        note("ACT", outcome.summary)

        return self._finish("completed", decision, plan, trace, calls, began, request_id,
                            outcome=outcome, policy_decision=policy_decision)

    # -----------------------------------------------------------------
    def _plan(self, question: str, answer: str, prior_calls: list[ToolResult],
             note: Any) -> AgentPlan:
        evidence = "\n".join(f"{c.tool} ({c.status}): {c.result or c.error}"
                             for c in prior_calls) or "(none yet)"
        user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nEVIDENCE SO FAR:\n{evidence}"
        raw = self.llm.judge(PLAN_SYSTEM, user, PLAN_SCHEMA)
        plan = AgentPlan.model_validate({
            "needs_analysis": raw.get("needs_grounding_review"),
            "tools": raw.get("tools", []),
            "more_evidence_needed": raw.get("more_evidence_needed", False),
            "rationale": raw.get("rationale", ""),
        })
        note("PLAN", plan.rationale or f"tools={list(plan.tools)}")
        return plan

    def _execute(self, tool_names: list[str], answer: str, question: str, chunks: list[str],
                surface: Surface, owner: str, budget_left: int,
                note: Any) -> tuple[list[ToolResult], bool]:
        results: list[ToolResult] = []
        truncated = False

        for name in tool_names:
            if len(results) >= budget_left:
                note("EXECUTE", f"stopped at the tool budget mid-plan ({budget_left})")
                truncated = True
                break
            if name not in GROUNDING_TOOL_NAMES:
                raise ToolNotAllowed(name)

            if name == "check_local_entailment":
                call_id = _call_id()
                nested = GroundingModelAgent(self.llm, self.engine).run(
                    answer, question=question, chunks=chunks, surface=surface, owner=owner,
                    request_id=f"{call_id}_grounding_model")
                res = _wrap_nested(nested, name, call_id)
                results.append(res)
                note("EXECUTE", f"{name} -> nested grounding_model_agent: {nested.decision.action} "
                                f"({len(nested.decision.findings)} finding(s))")
                continue

            args = {"answer": answer, "chunks": chunks}
            res = call_tool(name, args, self.engine, _call_id())
            results.append(res)
            if res.status == "ok":
                note("EXECUTE", f"{name} -> {_summarise(name, res.result)}")
            else:
                note("EXECUTE", f"{name} -> error: {res.error}")

        return results, truncated

    def _decide(self, question: str, answer: str, chunks: list[str],
               calls: list[ToolResult], note: Any) -> AgentDecision:
        evidence = "\n".join(
            f"[{c.call_id}] {c.tool} ({c.status}): {c.result or c.error}" for c in calls
        ) or "(no tool returned anything)"
        context = "\n\n".join(f"[chunk {i + 1}] {c}" for i, c in enumerate(chunks))
        user = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
               f"\n\nTOOL RESULTS:\n{evidence}")
        raw = self.llm.judge(DECISION_SYSTEM, user, DECISION_SCHEMA)
        decision = AgentDecision.model_validate({
            "action": raw.get("grounding_verdict"),
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


def _summarise(name: str, result: dict) -> str:
    if name == "extract_claims":
        return f"{result.get('claim_count', 0)} claim(s)"
    if name == "check_local_entailment":
        if not result.get("available"):
            return "local model unavailable"
        return (f"{result.get('supported', 0)}/{result.get('claims', 0)} claim(s) "
               f"entailed, consistency={result.get('consistency', 0):.2f}")
    if name == "get_grounding_policy":
        return f"action_on_fail={result.get('action_on_fail')}"
    return str(result)


def _wrap_nested(result: AgentResult, tool_name: str, call_id: str) -> ToolResult:
    """A nested agent's own `AgentResult`, folded into the shape `_decide()`'s
    evidence trail already expects — see `ner_agent.py`'s identical helper
    for the rationale; duplicated here rather than imported so each agent
    file stays self-contained, the same convention every other piece of this
    skeleton already follows."""
    d = result.decision
    return ToolResult(
        call_id=call_id, tool=tool_name,
        status="ok" if result.status != "failed" else "error",
        duration_ms=result.duration_ms,
        result={
            "nested_agent": result.agent, "nested_status": result.status,
            "action": d.action, "confidence": d.confidence, "rationale": d.rationale,
            "findings": [f.model_dump() for f in d.findings],
        },
    )


_call_counter = 0


def _call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"grounding_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
