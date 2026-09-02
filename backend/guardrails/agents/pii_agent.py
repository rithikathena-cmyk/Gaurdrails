"""The autonomous PII agent.

Reasons over evidence from the deterministic tools in `tools.py` and reaches
its own genuine recommendation — a real judge call, not a hardcoded rule.
That recommendation is not the final word: `PolicyEngine.decide()` combines
it with a deterministic floor read from `config/policy.yaml` before anything
executes, and can raise it — never lower it. What the agent *cannot* do,
regardless of that: act outside `GuardrailAction`'s six values, call a tool
outside `PII_AGENT_TOOLS`, or reach a capability outside `PIICapabilities` —
those are hard boundaries, enforced in Python, not asked of the model.

    ANALYZE + PLAN   one judge call: is there anything here worth looking at,
                     and if so, which of the five tools should run
    SELECT + EXECUTE the plan's tool names, fanned out over whatever kinds
                     the detectors actually found — deterministic Python,
                     no model involved
    OBSERVE + DECIDE a second judge call, given only what the tools actually
                     returned, recommending one of the six actions
    POLICY           `PolicyEngine.decide()` — the recommendation against
                     the deterministic floor; see `policy_engine.py`
    ACT              `PIICapabilities.execute()` — the model never touches
                     the vault, the rail, or the text directly

Bounded by `max_iterations` (PLAN may run more than once if the model asks
for more evidence), `max_tool_calls`, and `timeout_s`. Exceeding any of them,
or a judge call returning something that fails Pydantic validation, ends the
run in ESCALATE — never in a silent guess.
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
from .tools import PII_TOOL_NAMES, ToolNotAllowed, call as call_tool
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, PolicyDecision, ToolResult,
    TraceEvent,
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_analysis": {
            "type": "boolean",
            "description": "False only if the text plainly holds nothing a PII "
                           "tool could find — no identifiers, no names, no "
                           "addresses, nothing masked already.",
        },
        "tools": {
            "type": "array",
            "description": "Which of the allowed tools to run next. Empty if "
                           "needs_analysis is false.",
            "items": {"type": "string", "enum": list(PII_TOOL_NAMES)},
        },
        "more_evidence_needed": {
            "type": "boolean",
            "description": "True if, after seeing this round's results, you "
                           "expect to need another round before deciding.",
        },
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["needs_analysis", "tools", "more_evidence_needed", "rationale"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
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
                    "entity": {"type": "string"},
                    "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array", "items": {"type": "string"},
                        "description": "call_id values from the tool results you were shown. "
                                       "Never invent one.",
                    },
                },
                "required": ["entity", "risk", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "confidence", "rationale", "findings"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the planning step of a PII guardrail agent. You are given one piece of \
text and a fixed list of tools you may ask to run — you do not run them yourself, \
you only choose which ones are worth running.

- detect_pii_regex     structured identifiers with a checksum: SSN, card, IBAN, ...
- detect_pii_presidio  named entities a trained model recognises: people, addresses
- detect_pii_entities  free-form: reads the whole text and names anything that \
looks like a personal identifier, whether or not it matches a known shape or a \
trained label. Reach for this one when the other two found nothing but the text \
still reads like it could be identifying someone — an internal ID with no \
familiar format, a name a trained NER model was never built to catch, anything \
you cannot rule out just because no detector recognised it. Costs a real judge \
call every time; do not plan it as a first move on ordinary text.
- classify_pii_type    what a kind of identifier actually is, once one is found
- get_pii_policy       what the configured action is for a kind, once one is found

Most requests need detect_pii_regex and detect_pii_presidio together — you do \
not know in advance whether the text holds a structured identifier, a name, \
both, or neither. classify_pii_type and get_pii_policy are only useful once \
something has been found, so plan them when you expect a detector to return \
something, not as a first move. detect_pii_entities is worth a second round \
when the first round found nothing but you are not confident that means \
nothing is there.

Say plainly when nothing here needs a tool at all — a request about opening \
hours or a fee schedule does not.""")

DECISION_SYSTEM = judge_prompt("""\
You are the decision step of a PII guardrail agent. You are given the original \
text and exactly what the tools you asked for actually returned — nothing more. \
Decide the one action that should happen to this text.

- ALLOW     nothing sensitive, or already safely masked
- MASK      a reversible identifier should be replaced with a vault token
- REDACT    it should be removed without being recoverable
- BLOCK     the content itself should not proceed at all
- FLAG      it may proceed, but a person should see it
- ESCALATE  the evidence does not clearly support any of the above

Weigh the tools against each other rather than trusting one by default. A \
checksum-verified regex hit is strong evidence on its own; a NER tool finding \
nothing about an identifier it was never built to find is not evidence against \
it — Presidio finds names and addresses, not SSNs, so its silence on an SSN \
means nothing. Cite only call_id values you were actually shown as evidence; \
never invent one and never claim a tool said something it did not return.

If nothing was found, or the tools disagree in a way you cannot resolve with \
what you were given, say so and choose ESCALATE rather than guessing.""")


class PIIAgent:
    name = "pii_agent"
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
        request_id = request_id or f"pii_agent_{uuid.uuid4().hex[:10]}"
        began = time.perf_counter()
        trace: list[TraceEvent] = []
        calls: list[ToolResult] = []
        known_kinds: set[str] = set()
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
                    # The plan call itself is what can take real wall-clock
                    # time — a slow judge call that happens to conclude
                    # "nothing needed" still spent the budget and must not
                    # short-circuit past the check that exists for that.
                    return self._escalate(
                        "PLAN", f"exceeded {self.timeout_s}s during the plan call",
                        trace, calls, began, request_id, plan)

                if not plan.needs_analysis:
                    if not calls:
                        note("DECIDE", "no PII-relevant content — nothing to analyze")
                        return self._finish(
                            "completed",
                            AgentDecision(action="ALLOW", confidence=1.0,
                                         rationale="no PII-relevant content found",
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

                round_calls, round_kinds, truncated = self._execute(
                    plan.tools, text, surface, known_kinds, note, budget_left)
                calls.extend(round_calls)
                known_kinds |= round_kinds

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

        policy_action = str(self.engine.policy.get(PII_ACTION_KEY[surface]))
        policy_decision = self.policy_engine.decide(
            decision.action, has_findings=bool(decision.findings), policy_action=policy_action)
        note("POLICY", policy_decision.rationale)

        note("ACT", f"executing {policy_decision.final_action}")
        outcome = self.capabilities.execute(policy_decision.final_action, text, owner=owner)
        note("ACT", outcome.summary)

        return self._finish("completed", decision, plan, trace, calls, began, request_id,
                            outcome=outcome, policy_decision=policy_decision)

    # -----------------------------------------------------------------
    def _plan(self, text: str, prior_calls: list[ToolResult],
              note: Any) -> AgentPlan:
        evidence = "\n".join(f"{c.tool} ({c.status}): {c.result or c.error}"
                             for c in prior_calls) or "(none yet)"
        user = f"TEXT:\n{text}\n\nEVIDENCE SO FAR:\n{evidence}"
        raw = self.llm.judge(PLAN_SYSTEM, user, PLAN_SCHEMA)
        plan = AgentPlan.model_validate(raw)
        note("PLAN", plan.rationale or f"tools={list(plan.tools)}")
        return plan

    def _execute(self, tool_names: list[str], text: str, surface: Surface,
                known_kinds: set[str], note: Any,
                budget_left: int) -> tuple[list[ToolResult], set[str], bool]:
        results: list[ToolResult] = []
        found_kinds: set[str] = set()
        calls_made = 0
        truncated = False

        for name in tool_names:
            if calls_made >= budget_left:
                note("EXECUTE", f"stopped at the tool budget mid-plan ({budget_left})")
                truncated = True
                break
            if name not in PII_TOOL_NAMES:
                raise ToolNotAllowed(name)

            if name in ("detect_pii_regex", "detect_pii_presidio", "detect_pii_entities"):
                res = call_tool(name, {"text": text}, self.engine, _call_id())
                results.append(res)
                calls_made += 1
                if res.status == "ok":
                    found_kinds |= {f["kind"] for f in res.result.get("findings", [])}
                note("EXECUTE", f"{name} -> {len(res.result.get('findings', []))} finding(s)"
                                if res.status == "ok" else f"{name} -> error: {res.error}")
                continue

            # classify_pii_type / get_pii_policy fan out over kinds discovered
            # this run — the model asks for the capability, the arguments
            # (which kind) come from what was actually found, never guessed.
            kinds_here = sorted(known_kinds | found_kinds)
            ran = 0
            for kind in kinds_here:
                if calls_made >= budget_left:
                    truncated = True
                    break
                args = {"kind": kind} if name == "classify_pii_type" \
                    else {"kind": kind, "surface": surface.value}
                res = call_tool(name, args, self.engine, _call_id())
                results.append(res)
                calls_made += 1
                ran += 1
            if not kinds_here:
                note("EXECUTE", f"{name} skipped — nothing detected yet to classify")
            else:
                note("EXECUTE", f"{name} -> checked {ran} of {len(kinds_here)} kind(s)")
            if truncated:
                break

        return results, found_kinds, truncated

    def _decide(self, text: str, calls: list[ToolResult], note: Any) -> AgentDecision:
        evidence = "\n".join(
            f"[{c.call_id}] {c.tool} ({c.status}): {c.result or c.error}" for c in calls
        ) or "(no tool returned anything)"
        user = f"TEXT:\n{text}\n\nTOOL RESULTS:\n{evidence}"
        raw = self.llm.judge(DECISION_SYSTEM, user, DECISION_SCHEMA)
        decision = AgentDecision.model_validate(raw)

        # A finding whose evidence cites a call_id nobody recorded is not
        # evidence of anything — drop it rather than let a hallucinated
        # citation stand in the trace.
        recorded = {c.call_id for c in calls}
        decision.findings = [
            f for f in decision.findings if set(f.evidence) <= recorded
        ]
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
        # No deterministic floor is computed here — whatever evidence a prior
        # round gathered is not re-examined before escalating. Every path
        # still produces a real `PolicyDecision` so the trace shape is
        # uniform, but a genuine floor found before the escalation triggered
        # is not currently surfaced as the final action; only DECIDE's own
        # completion path (below) does that.
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
    return f"pii_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
