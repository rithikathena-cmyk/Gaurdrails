"""The autonomous, policy-controlled Guardrail Supervisor — MVP.

    PLAN -> SELECT -> EXECUTE -> OBSERVE -> DECIDE -> ENFORCE -> TRACE

A flat, single-hop sibling to `supervisor.py`'s `Supervisor`, not a
replacement for it. `Supervisor` selects among six *specialist agents*, each
of which runs its own nested PLAN/DECIDE judge calls before reaching a
detector. This class calls the six flat tools in `guardrail_tools.py`
directly — one supervisor, one hop, the exact shape an autonomous-guardrail-
supervisor spec asks for. Both classes share the same underlying rails, the
same `PolicyEngine`, and the same `CapabilityDenied` boundary; neither
duplicates the other's detection logic.

Two things this class does that neither `Supervisor` nor any of the six
specialist agents currently do:

    hard-block pre-check   `detect_prompt_injection` and
                           `detect_destructive_intent` run *before* PLAN,
                           deterministically. A high-confidence pattern hit
                           short-circuits straight to BLOCK — the judge is
                           never asked about an obvious case.
    marginal-band gate     DECIDE computes a deterministic risk *proxy*
                           (`_risk_proxy` — max tool confidence, not a
                           calibrated score) from what EXECUTE's tools
                           actually found first. Below
                           `supervisor.risk_low_threshold`: ALLOW, no judge
                           call. Above `supervisor.risk_high_threshold`:
                           BLOCK, no judge call. Only the configured band
                           between them calls the judge — whose own
                           `risk_score` is a different, calibrated number;
                           `GuardrailDecision.risk_source` says which kind
                           any given decision carries.

The LLM only ever recommends. ENFORCE always has the last, deterministic
word: `PolicyEngine.decide()` (reused from `policy_engine.py`, unmodified)
combines the recommendation with a floor read from `config/policy.yaml` and
can only raise it, never lower it. A forbidden-capability name — the eleven
in `guardrail_capabilities.FORBIDDEN_CAPABILITIES` — is denied unconditionally,
regardless of who asked or how: the user, the model's own plan, a tool
result, or text engineered to look like an instruction.

Policy precedence, as enforced by `_enforce()` below, in order:

    1. system security constraints   `GuardrailAction`'s six-value Literal
                                     and `deny_if_forbidden` — enforced by
                                     construction, checked at every ACT
    2. hard deterministic controls   the pre-check above, and the
                                     above-threshold branch of the risk gate
    3. RBAC                          `AuthorizationContext.entitled`, when a
                                     caller supplies one
    4. guardrail policy              the floor read from `config/policy.yaml`
    5. agent recommendation          the judge's `GuardrailDecision.action`,
                                     when the judge ran at all
    6. user request                 carries no independent weight; it is
                                     what steps 2-5 are evaluating
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
from .authorization_tools import AuthorizationContext
from .capabilities import PIICapabilities
from .guardrail_capabilities import deny_if_forbidden
from .guardrail_tools import ALLOWED_GUARDRAIL_TOOLS
from .guardrail_tools import call as call_tool
from .policy_engine import PolicyEngine
from .tools import ToolNotAllowed
from .types import (
    ActionOutcome, GuardrailDecision, GuardrailPlan, GuardrailSupervisorResult,
    PolicyDecision, ToolResult, TraceEvent,
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_categories": {
            "type": "array", "items": {"type": "string"},
            "description": "Which kinds of risk this request plausibly presents — "
                           "PII, injection, destructive intent, scope, or content "
                           "risk. Empty if none apply.",
        },
        "checks": {
            "type": "array",
            "description": "Which of the approved guardrail tools to run next.",
            "items": {"type": "string", "enum": sorted(ALLOWED_GUARDRAIL_TOOLS)},
        },
        "policy_keys": {
            "type": "array", "items": {"type": "string"},
            "description": "Only read when 'get_policy' is in checks: which "
                           "configured policies to look up, by name — pii, "
                           "pii.<entity> (e.g. pii.US_SSN), injection, "
                           "destructive_intent, scope, semantic_risk, or "
                           "semantic_risk.<category>. One structured lookup per "
                           "key; the actual configured value is read for you, "
                           "never inferred.",
        },
        "more_evidence_needed": {
            "type": "boolean",
            "description": "True if, after seeing this round's results, you expect "
                           "to need another round before deciding.",
        },
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["risk_categories", "checks", "policy_keys", "more_evidence_needed",
                "rationale"],
    "additionalProperties": False,
}

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"],
        },
        "risk_score": {"type": "number", "description": "0.0 to 1.0."},
        "confidence": {"type": "number", "description": "0.0 to 1.0."},
        "triggered_rails": {
            "type": "array", "items": {"type": "string"},
            "description": "Which tools you were shown results from actually found "
                           "something — by tool name.",
        },
        "evidence": {
            "type": "array", "items": {"type": "string"},
            "description": "What a tool actually returned that supports this action. "
                           "Never invent a finding no tool reported.",
        },
        "reason": {"type": "string", "description": "One sentence."},
    },
    "required": ["action", "risk_score", "confidence", "triggered_rails",
                "evidence", "reason"],
    "additionalProperties": False,
}

PLAN_SYSTEM = judge_prompt("""\
You are the routing step of the Guardrail Supervisor. You are given one \
request and a fixed list of approved guardrail tools you may call — you do \
not analyze the request yourself, you only decide which tools, if any, are \
worth running.

- detect_pii                  personal identifiers: SSNs, cards, emails, \
phone numbers, names, addresses.
- detect_prompt_injection     attempts to override, extract, or escape the \
assistant's own instructions.
- detect_destructive_intent   requests to delete, modify, or disable \
something with real consequences — data, records, RBAC, guardrails, \
approvals, secrets.
- check_scope                 whether the request belongs at this service \
at all.
- check_semantic_risk         hate, violence, insults, misconduct, \
self-harm, or sexual content that a pattern cannot catch.
- get_policy                  what a configured guardrail policy actually \
says. Structured, not free text: name the policy in `policy_keys` — one or \
more of pii, pii.<entity> (e.g. pii.US_SSN), injection, destructive_intent, \
scope, semantic_risk, or semantic_risk.<category>. Call it once you already \
know what to ask about — after another tool found something, or when a \
policy's own configured action genuinely decides what should happen next —\
 not as a first move.

Call every angle that is genuinely present, and no more. A plain question — \
opening hours, a fee schedule, how a process works — needs no tool at all: \
say so plainly rather than calling one out of caution.""")

DECIDE_SYSTEM = judge_prompt("""\
You are the decision step of the Guardrail Supervisor. You are given the \
original request and exactly what the tools you asked for actually \
returned — nothing more. Choose the one action that should happen to this \
request, and a risk score that reflects how confident that evidence makes \
you.

- ALLOW     nothing found that needs to change
- MASK      a reversible identifier should be replaced with a vault token
- REDACT    something should be removed without being recoverable
- BLOCK     the request should not proceed at all
- FLAG      it may proceed, but a person should see it
- ESCALATE  the evidence does not clearly support any of the above

Cite only what a tool actually returned as evidence. Never invent a finding \
no tool reported, and never assume a check you were not shown the result of.""")


class GuardrailSupervisor:
    """`class GuardrailSupervisor` — see the module docstring for the loop
    and the precedence chain `_enforce()` implements."""

    name = "guardrail_supervisor"
    version = "0.1.0"

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
            owner: str = "", request_id: str = "",
            ctx: AuthorizationContext | None = None) -> GuardrailSupervisorResult:
        """`ctx`, when supplied, is a real `AuthorizationContext` a caller
        resolved from a signed-in session — the same object
        `agents/supervisor.py` already threads to the `authorization` agent.
        It is what rung 3 (RBAC) of `_enforce()` reads. Omitted, RBAC
        contributes nothing to the precedence chain — the same conservative
        default the standalone `authorization` agent documents for itself.
        """
        request_id = request_id or f"guardrail_supervisor_{uuid.uuid4().hex[:10]}"
        began = time.perf_counter()
        trace: list[TraceEvent] = []
        tool_calls: list[ToolResult] = []
        judge_calls = 0
        plan: GuardrailPlan | None = None

        def elapsed_ms() -> float:
            return (time.perf_counter() - began) * 1000

        def note(phase: str, summary: str) -> None:
            trace.append(TraceEvent(phase=phase, summary=summary, at_ms=round(elapsed_ms(), 1)))

        def timed_out() -> bool:
            return elapsed_ms() > self.timeout_s * 1000

        try:
            # ---- hard-block pre-check: zero judge calls -----------------
            hard = self._hard_block_check(text, tool_calls, note)
            if hard is not None:
                policy_decision = self._enforce(hard, ctx, note)
                outcome = self._act(policy_decision.final_action, text, owner, note)
                note("TRACE", f"{request_id} — hard-blocked, judge never called")
                return self._finish(
                    "completed", hard, None, trace, tool_calls, began, request_id,
                    outcome=outcome, policy_decision=policy_decision,
                    hard_blocked=True, judge_calls=0)

            for iteration in range(1, self.max_iterations + 1):
                if timed_out():
                    return self._escalate(
                        "PLAN", f"exceeded {self.timeout_s}s before a plan completed",
                        trace, tool_calls, began, request_id, judge_calls=judge_calls)

                plan = self._plan(text, tool_calls, note)
                judge_calls += 1

                if timed_out():
                    return self._escalate(
                        "PLAN", f"exceeded {self.timeout_s}s during the plan call",
                        trace, tool_calls, began, request_id, plan, judge_calls)

                if not plan.checks:
                    note("SELECT", "no check selected")
                    break

                budget_left = self.max_tool_calls - len(tool_calls)
                if budget_left <= 0:
                    return self._escalate(
                        "EXECUTE", f"tool call budget ({self.max_tool_calls}) exhausted",
                        trace, tool_calls, began, request_id, plan, judge_calls)

                new_calls, truncated = self._execute(
                    plan.checks, plan.policy_keys, text, surface, note, budget_left)
                tool_calls += new_calls

                if truncated:
                    return self._escalate(
                        "EXECUTE", f"tool call budget ({self.max_tool_calls}) exhausted "
                                   "before the plan finished running",
                        trace, tool_calls, began, request_id, plan, judge_calls)

                if not plan.more_evidence_needed:
                    break
            else:
                return self._escalate(
                    "PLAN", f"exceeded {self.max_iterations} iterations without "
                            "the plan declaring itself done",
                    trace, tool_calls, began, request_id, plan, judge_calls)

            if timed_out():
                return self._escalate(
                    "DECIDE", f"exceeded {self.timeout_s}s before a decision completed",
                    trace, tool_calls, began, request_id, plan, judge_calls)

            # ---- DECIDE: deterministic gate first, judge only in the band -
            decision, band_judge_calls = self._decide_or_gate(text, tool_calls, note)
            judge_calls += band_judge_calls

        except ToolNotAllowed:
            raise  # a programming/security error, never masked as ESCALATE
        except LLMError as exc:
            return self._escalate("DECIDE", f"supervisor could not reach a decision: {exc}",
                                  trace, tool_calls, began, request_id, plan, judge_calls)
        except ValidationError:
            return self._escalate("DECIDE", "malformed model output failed validation",
                                  trace, tool_calls, began, request_id, plan, judge_calls)

        # ---- ENFORCE ---------------------------------------------------
        policy_decision = self._enforce(decision, ctx, note)
        outcome = self._act(policy_decision.final_action, text, owner, note)

        # ---- TRACE -------------------------------------------------------
        note("TRACE", f"{request_id} complete — final={policy_decision.final_action} "
                      f"judge_calls={judge_calls}")

        return self._finish(
            "completed", decision, plan, trace, tool_calls, began, request_id,
            outcome=outcome, policy_decision=policy_decision, judge_calls=judge_calls)

    # -----------------------------------------------------------------
    # Hard-block pre-check
    # -----------------------------------------------------------------
    def _hard_block_check(self, text: str, tool_calls: list[ToolResult],
                          note: Any) -> GuardrailDecision | None:
        """Runs the two deterministic detectors that matter most before
        anything else — no judge call either way. §13: obviously dangerous
        cases are answered without asking a model whether to block them.

        Always emits a `PRECHECK` phase, whatever the outcome — the trace
        should never make a hard block look like an unexplained skipped
        PLAN. When a detector fires, a distinct `HARD_BLOCK` phase follows
        it, so the sequence reads `PRECHECK -> HARD_BLOCK -> ENFORCE ->
        TRACE` rather than folding the finding into PRECHECK's own line.
        """
        note("PRECHECK", "running detect_prompt_injection and detect_destructive_intent "
                         "deterministically, before PLAN — no judge call either way")

        injection = call_tool("detect_prompt_injection", {"text": text}, self.engine,
                              _call_id("hardblock"))
        tool_calls.append(injection)
        if injection.status == "ok" and injection.result.get("detected"):
            note("HARD_BLOCK",
                f"detect_prompt_injection matched {injection.result.get('types')} "
                f"(confidence {injection.result.get('confidence')}) — no judge call made")
            return GuardrailDecision(
                action="BLOCK", risk_score=1.0, risk_source="deterministic_proxy",
                confidence=1.0, triggered_rails=["detect_prompt_injection"],
                evidence=[f"detect_prompt_injection: {injection.result}"],
                reason="deterministic hard block — prompt injection pattern matched")

        destructive = call_tool("detect_destructive_intent", {"text": text}, self.engine,
                                _call_id("hardblock"))
        tool_calls.append(destructive)
        if destructive.status == "ok" and destructive.result.get("detected"):
            note("HARD_BLOCK",
                f"detect_destructive_intent matched {destructive.result.get('types')} "
                "— no judge call made")
            return GuardrailDecision(
                action="BLOCK", risk_score=1.0, risk_source="deterministic_proxy",
                confidence=1.0, triggered_rails=["detect_destructive_intent"],
                evidence=[f"detect_destructive_intent: {destructive.result}"],
                reason="deterministic hard block — destructive or capability-escalation "
                      "pattern matched")

        note("PRECHECK", "clear — no deterministic hard-block matched, proceeding to PLAN")
        return None

    # -----------------------------------------------------------------
    # PLAN / SELECT / EXECUTE / OBSERVE
    # -----------------------------------------------------------------
    def _plan(self, text: str, prior_calls: list[ToolResult], note: Any) -> GuardrailPlan:
        if self.llm is None:
            raise LLMError("no API key configured — PLAN needs a live judge call; "
                           "a hard-blocked request never reaches this")
        evidence = "\n".join(f"{c.tool} ({c.status}): {c.result or c.error}"
                             for c in prior_calls) or "(none yet)"
        user = f"REQUEST:\n{text}\n\nEVIDENCE SO FAR:\n{evidence}"
        raw = self.llm.judge(PLAN_SYSTEM, user, PLAN_SCHEMA)
        plan = GuardrailPlan.model_validate(raw)
        note("PLAN", plan.rationale or f"checks={list(plan.checks)}")
        return plan

    def _execute(self, tool_names: list[str], policy_keys: list[str], text: str,
                surface: Surface, note: Any, budget_left: int) -> tuple[list[ToolResult], bool]:
        """`get_policy` is special-cased: it takes a structured `policy`
        (and `surface`) argument, not the free-text `{"text": text}` every
        other flat tool takes, so it fans out one call per entry in
        `policy_keys` rather than one call per name in `tool_names`. A plan
        naming `get_policy` with no `policy_keys` still runs once — the tool
        itself reports "no policy key given" rather than being silently
        skipped, the same honesty every other malformed-input path in this
        loop already has."""
        results: list[ToolResult] = []
        truncated = False
        note("SELECT", f"checks={list(tool_names)} policy_keys={list(policy_keys)}")

        for name in tool_names:
            if len(results) >= budget_left:
                note("EXECUTE", f"stopped at the tool budget mid-plan ({budget_left})")
                truncated = True
                break
            if name not in ALLOWED_GUARDRAIL_TOOLS:
                raise ToolNotAllowed(name)

            if name == "get_policy":
                for key in (policy_keys or [""]):
                    if len(results) >= budget_left:
                        note("EXECUTE", f"stopped at the tool budget mid-plan ({budget_left})")
                        truncated = True
                        break
                    res = call_tool("get_policy", {"policy": key, "surface": surface.value},
                                    self.engine, _call_id("execute"))
                    results.append(res)
                    note("EXECUTE", f"get_policy({key!r}) -> {res.result}"
                                    if res.status == "ok" else f"get_policy({key!r}) -> error: {res.error}")
                if truncated:
                    break
                continue

            res = call_tool(name, {"text": text}, self.engine, _call_id("execute"))
            results.append(res)
            if res.status == "ok":
                note("EXECUTE", f"{name} -> {res.result}")
            else:
                note("EXECUTE", f"{name} -> error: {res.error}")

        note("OBSERVE", f"{len(results)} tool result(s) collected")
        return results, truncated

    # -----------------------------------------------------------------
    # DECIDE — deterministic marginal-band gate, judge only inside it
    # -----------------------------------------------------------------
    def _decide_or_gate(self, text: str, tool_calls: list[ToolResult],
                        note: Any) -> tuple[GuardrailDecision, int]:
        proxy = self._risk_proxy(tool_calls)
        triggered, evidence = self._triggered_and_evidence(tool_calls)
        low = float(self.engine.policy.get("supervisor.risk_low_threshold"))
        high = float(self.engine.policy.get("supervisor.risk_high_threshold"))

        if proxy < low:
            note("DECIDE", f"deterministic ALLOW — risk_proxy {proxy:.2f} below "
                           f"supervisor.risk_low_threshold ({low:.2f}), no judge call")
            return GuardrailDecision(
                action="ALLOW", risk_score=proxy, risk_source="deterministic_proxy",
                confidence=1.0, triggered_rails=triggered, evidence=evidence,
                reason=f"deterministic: risk_proxy {proxy:.2f} below the low threshold"), 0

        if proxy > high:
            action = self._deterministic_high_risk_action(tool_calls)
            note("DECIDE", f"deterministic {action} — risk_proxy {proxy:.2f} above "
                           f"supervisor.risk_high_threshold ({high:.2f}), no judge call")
            return GuardrailDecision(
                action=action, risk_score=proxy, risk_source="deterministic_proxy",
                confidence=1.0, triggered_rails=triggered, evidence=evidence,
                reason=f"deterministic: risk_proxy {proxy:.2f} above the high threshold"), 0

        note("DECIDE", f"risk_proxy {proxy:.2f} is inside the [{low:.2f}, {high:.2f}] band "
                       "— adjudicating with the judge")
        decision = self._decide(text, tool_calls, note)
        return decision, 1

    def _decide(self, text: str, tool_calls: list[ToolResult], note: Any) -> GuardrailDecision:
        if self.llm is None:
            raise LLMError("no API key configured — DECIDE needs a live judge call "
                           "inside the marginal risk band")
        evidence = "\n".join(
            f"[{c.call_id}] {c.tool} ({c.status}): {c.result or c.error}" for c in tool_calls
        ) or "(no tool returned anything)"
        user = f"REQUEST:\n{text}\n\nTOOL RESULTS:\n{evidence}"
        raw = self.llm.judge(DECIDE_SYSTEM, user, DECIDE_SCHEMA)
        decision = GuardrailDecision.model_validate(raw)
        note("DECIDE", f"{decision.action} — {decision.reason}")
        return decision

    @staticmethod
    def _risk_proxy(tool_calls: list[ToolResult]) -> float:
        """`max(tool confidence)` — a coarse, deterministic stand-in for a
        risk score, computed in Python from what the tools actually
        returned, before any judge call is possible. Deliberately named
        `_risk_proxy`, not `_risk_score`: it is a maximum over whatever a
        handful of independent, uncalibrated detectors happened to report —
        never a probability, never model output, and not the same kind of
        number `GuardrailDecision.risk_score` carries when a judge actually
        ran (`risk_source="judge"` on that path; `"deterministic_proxy"`
        here). Good enough to gate "obviously fine" / "obviously not" —
        no more."""
        scores: list[float] = []
        for c in tool_calls:
            if c.status != "ok":
                continue
            r = c.result
            if "confidence" in r:
                scores.append(float(r["confidence"]))
            elif c.tool == "check_scope" and not r.get("in_scope", True):
                scores.append(0.3)
            elif c.tool == "check_semantic_risk":
                scores.append(float(r.get("max_score", 0.0)))
        return max(scores, default=0.0)

    @staticmethod
    def _triggered_and_evidence(tool_calls: list[ToolResult]) -> tuple[list[str], list[str]]:
        triggered: list[str] = []
        evidence: list[str] = []
        for c in tool_calls:
            if c.status != "ok":
                continue
            r = c.result
            if r.get("detected"):
                triggered.append(c.tool)
                types = r.get("types") or []
                evidence.append(
                    f"{c.tool} detected {', '.join(types) or 'a match'} "
                    f"(confidence {float(r.get('confidence', 0)):.2f})")
            elif c.tool == "check_scope" and not r.get("in_scope", True):
                triggered.append(c.tool)
                evidence.append("check_scope: no domain vocabulary matched")
            elif c.tool == "check_semantic_risk" and r.get("worst_category"):
                evidence.append(
                    f"check_semantic_risk: {r['worst_category']} scored "
                    f"{float(r.get('max_score', 0)):.2f}")
        return triggered, evidence

    @staticmethod
    def _deterministic_high_risk_action(tool_calls: list[ToolResult]) -> str:
        """Which action a risk_proxy above the high threshold resolves to,
        without a judge call. Not a blanket BLOCK: PII above the threshold
        still recommends MASK, so ENFORCE's policy floor — not this
        function — is what actually decides whether that is strict enough."""
        by_tool = {c.tool: c.result for c in tool_calls if c.status == "ok"}
        if by_tool.get("detect_destructive_intent", {}).get("detected"):
            return "BLOCK"
        if by_tool.get("detect_prompt_injection", {}).get("detected"):
            return "BLOCK"
        if by_tool.get("detect_pii", {}).get("detected"):
            return "MASK"
        return "BLOCK"

    # -----------------------------------------------------------------
    # ENFORCE — the precedence chain. See the module docstring for the
    # six rungs; this is where they are actually evaluated, in order.
    # -----------------------------------------------------------------
    def _enforce(self, decision: GuardrailDecision, ctx: AuthorizationContext | None,
                note: Any) -> PolicyDecision:
        # Rung 3 — RBAC. Only ever tightens an ALLOW; never loosens anything
        # a hard block or the deterministic gate already decided (those
        # already returned something other than a bare model recommendation
        # by the time this runs). Reuses `AuthorizationContext.entitled` —
        # the exact deterministic check `AuthorizationCapabilities` already
        # applies for the standalone `authorization` agent.
        if ctx is not None and decision.action == "ALLOW" and not ctx.entitled:
            note("ENFORCE", f"RBAC floor: {ctx.principal!r} is not entitled to "
                            f"{ctx.resource_kind or 'this resource'} — overriding ALLOW")
            return PolicyDecision(
                final_action="BLOCK", recommended_action=decision.action,
                floor_action="BLOCK", overridden=True,
                rationale=f"RBAC: {ctx.principal!r} is not entitled to "
                         f"{ctx.resource_kind or 'this resource'}")

        # Rungs 4/5 — the guardrail policy floor vs. the recommendation.
        # `PolicyEngine.decide()` is reused unmodified: it can only raise
        # the recommendation, never lower it.
        policy_action = self._floor_policy_action(decision)
        result = self.policy_engine.decide(
            decision.action, has_findings=bool(decision.triggered_rails),
            policy_action=policy_action)
        note("ENFORCE", result.rationale)
        return result

    def _floor_policy_action(self, decision: GuardrailDecision) -> str:
        """The one configured policy action to use as this decision's floor
        — read from the same keys the deterministic pipeline already reads,
        chosen by which tool actually triggered. Destructive intent has no
        adjustable action: `config/policy.yaml`'s `security_rules`/
        `use_case_rules` already resolve block/mask/flag per rule, and this
        is the aggregate — it is never softer than block."""
        p = self.engine.policy
        if "detect_destructive_intent" in decision.triggered_rails:
            return "block"
        if "detect_prompt_injection" in decision.triggered_rails:
            return str(p.get("prompt_attack.action"))
        if "detect_pii" in decision.triggered_rails:
            return str(p.get("pii.action.user_prompt"))
        if "check_semantic_risk" in decision.triggered_rails:
            return str(p.get("content.action.user_prompt"))
        return ""

    def _act(self, final_action: str, text: str, owner: str, note: Any) -> ActionOutcome:
        # Rung 1, made explicit and testable: `final_action` is always one
        # of the six `GuardrailAction` values by construction, and none of
        # them is in `FORBIDDEN_CAPABILITIES` — this call can never actually
        # raise in the normal flow. It stays here anyway, unconditionally,
        # because the boundary is supposed to hold regardless of who or what
        # is asking, not because this call site happens to be safe today.
        deny_if_forbidden(final_action.lower())
        outcome = self.capabilities.execute(final_action, text, owner=owner)
        note("ENFORCE", outcome.summary)
        return outcome

    # -----------------------------------------------------------------
    def _escalate(self, phase: str, reason: str, trace: list[TraceEvent],
                 tool_calls: list[ToolResult], began: float, request_id: str,
                 plan: GuardrailPlan | None = None,
                 judge_calls: int = 0) -> GuardrailSupervisorResult:
        trace.append(TraceEvent(phase="ESCALATE", summary=reason,
                                at_ms=round((time.perf_counter() - began) * 1000, 1)))
        decision = GuardrailDecision(action="ESCALATE", risk_score=0.0,
                                     risk_source="deterministic_proxy", confidence=0.0,
                                     triggered_rails=[], evidence=[], reason=reason)
        policy_decision = self.policy_engine.decide("ESCALATE", has_findings=False)
        outcome = self.capabilities.execute(policy_decision.final_action, "", owner="")
        return self._finish(
            "escalated", decision, plan, trace, tool_calls, began, request_id,
            outcome=outcome, policy_decision=policy_decision, escalation_reason=reason,
            judge_calls=judge_calls)

    def _finish(self, status: str, decision: GuardrailDecision | None,
               plan: GuardrailPlan | None, trace: list[TraceEvent],
               tool_calls: list[ToolResult], began: float, request_id: str, *,
               outcome: ActionOutcome | None = None,
               policy_decision: PolicyDecision | None = None,
               escalation_reason: str = "", hard_blocked: bool = False,
               judge_calls: int = 0) -> GuardrailSupervisorResult:
        return GuardrailSupervisorResult(
            request_id=request_id, status=status, plan=plan, tool_calls=tool_calls,
            decision=decision, trace=trace, policy_decision=policy_decision,
            outcome=outcome, duration_ms=round((time.perf_counter() - began) * 1000, 1),
            escalation_reason=escalation_reason, hard_blocked=hard_blocked,
            judge_calls=judge_calls,
        )


_call_counter = 0


def _call_id(prefix: str) -> str:
    global _call_counter
    _call_counter += 1
    return f"{prefix}_{_call_counter:04d}_{uuid.uuid4().hex[:6]}"
