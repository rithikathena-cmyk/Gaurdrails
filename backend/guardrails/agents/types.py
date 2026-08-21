"""Typed contracts for the autonomous guardrail agents.

Distinct from `guardrails.agent` (singular) — the conversational tool-use loop
that drives chat and RAG. This package is the reasoning layer around the
existing deterministic rails: an agent here never talks to a user and never
calls a tool that was not already in `guardrails.rails` before this package
existed. `AgentResult` in this package is not the same type as
`guardrails.AgentResult`, and neither module imports the other.

Every model an agent returns is schema-validated. `Claude.judge()` enforces
the JSON *shape*; these enforce the *values* — a `confidence` outside 0..1 or
an `action` outside the six the Policy Engine understands raises here, in
Python, before the value can reach anything that acts on it.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: The only six actions an agent may ever return. Not extensible from a prompt
#: or from config — adding a seventh means editing this literal, in a review.
GuardrailAction = Literal["ALLOW", "MASK", "REDACT", "BLOCK", "FLAG", "ESCALATE"]

RiskLevel = Literal["low", "medium", "high", "critical"]

#: The bounded lifecycle every agent run passes through, in order except that
#: PLAN and EXECUTE may repeat within `max_iterations`.
Phase = Literal[
    "ANALYZE", "PLAN", "SELECT", "EXECUTE", "OBSERVE",
    "EVALUATE", "DECIDE", "POLICY", "ACT", "ESCALATE", "FINAL",
]


class ToolCall(BaseModel):
    """One tool the agent selected, before it runs."""

    call_id: str
    tool: str
    args: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    """What a tool actually returned. The agent's evidence is built only from
    these — never from what the agent expected or assumed a tool would say."""

    call_id: str
    tool: str
    status: Literal["ok", "error"]
    duration_ms: float = 0.0
    #: Safe, structured, redacted. A tool wrapper that put a raw matched value
    #: here would defeat the reason `Detection.redacted()` exists.
    result: dict = Field(default_factory=dict)
    error: str = ""


class PIIFinding(BaseModel):
    """One thing the agent concluded from the evidence — not a raw detection.

    `evidence` names the `call_id`s that support it. A finding whose evidence
    list contains a call_id nobody recorded is not evidence of anything; the
    agent loop drops such findings before they reach a decision.
    """

    entity: str
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """What the agent decided to do next, before doing it."""

    needs_analysis: bool
    tools: list[str] = Field(default_factory=list)
    more_evidence_needed: bool = False
    rationale: str = ""

    @field_validator("tools")
    @classmethod
    def _bounded(cls, v: list[str]) -> list[str]:
        # A model that returns the same tool eleven times is not planning —
        # this is a sanity ceiling, not the enforcement boundary. The real
        # boundary is the allowlist dict in tools.py, which this model never
        # sees and cannot widen.
        if len(v) > 8:
            raise ValueError("a plan naming more than 8 tool calls is not a plan")
        return v


class AgentDecision(BaseModel):
    """The agent's final call. `action` is the only field the capability
    layer reads to decide what to execute — everything else is explanation."""

    action: GuardrailAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    findings: list[PIIFinding] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """One line of the audit-facing record. A concise summary of what
    happened in one phase — never the model's raw chain-of-thought."""

    phase: Phase
    summary: str
    at_ms: float = 0.0


class AgentState(BaseModel):
    """Where one run currently is in its bounded lifecycle."""

    request_id: str
    phase: Phase = "ANALYZE"
    iteration: int = 0
    tool_calls_used: int = 0
    started_at: float = Field(default_factory=time.perf_counter)


class ActionOutcome(BaseModel):
    """What the capability layer actually did, executing the agent's decision."""

    action: GuardrailAction
    capability: str
    text_out: str = ""
    tokens_masked: int = 0
    summary: str = ""


class SupervisorPlan(BaseModel):
    """Which registered agents the supervisor believes are relevant, and why.

    `agents` is validated against a fixed dict in `supervisor.py` at the call
    site, not here — this model only bounds *shape* (a list of strings), the
    same division of labour `AgentPlan.tools` has with `PII_AGENT_TOOLS`.
    """

    agents: list[str] = Field(default_factory=list)
    more_evidence_needed: bool = False
    reason: str = ""

    @field_validator("agents")
    @classmethod
    def _bounded(cls, v: list[str]) -> list[str]:
        if len(v) > 8:
            raise ValueError("a plan naming more than 8 agents is not a plan")
        return v


class SupervisorResult(BaseModel):
    """The complete, audit-ready record of one supervisor run.

    `agent_results` keeps every specialized agent's own full, independently
    valid `AgentResult` — including that agent's own trace — keyed by its
    registry name, so a request that ran the PII agent shows exactly what the
    PII agent itself decided, not only what the supervisor made of it.
    """

    request_id: str
    status: Literal["completed", "escalated"]
    plan: SupervisorPlan | None = None
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    final_action: GuardrailAction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)
    policy_decision: "PolicyDecision | None" = None
    outcome: ActionOutcome | None = None
    duration_ms: float = 0.0
    escalation_reason: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()


class PolicyDecision(BaseModel):
    """What the deterministic Policy Engine actually decided, and why.

    `AgentDecision.action` — every agent's own DECIDE step — is a
    recommendation, not the final word; this is the final word. Kept
    alongside the recommendation in `AgentResult` rather than replacing it,
    so a reviewer sees both what the model thought and what was enforced.
    See `policy_engine.py` for the module that produces one of these.
    """

    final_action: GuardrailAction
    recommended_action: GuardrailAction
    floor_action: GuardrailAction
    overridden: bool = False
    rationale: str = ""


class AgentResult(BaseModel):
    """The complete, audit-ready record of one autonomous agent run."""

    request_id: str
    agent: str
    version: str
    status: Literal["completed", "failed", "escalated"]
    decision: AgentDecision
    plan: AgentPlan | None = None
    tool_calls: list[ToolResult] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    policy_decision: PolicyDecision | None = None
    outcome: ActionOutcome | None = None
    duration_ms: float = 0.0
    escalation_reason: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()
