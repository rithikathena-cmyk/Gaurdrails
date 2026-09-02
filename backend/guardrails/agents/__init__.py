"""Autonomous guardrail agents — reasoning around the existing deterministic
rails, never a replacement for them.

Named `agents` (plural) to stay distinct from `guardrails.agent` (singular),
the conversational tool-use loop that drives chat and RAG. One deliberate
exception: `agent.runner.AgentRunner._agentic_data_scan` imports PIIAgent,
PromptInjectionAgent and ContentSafetyAgent directly, when
`agent.data_check_mode` is `"agentic"`, to re-run the same specialists
`Supervisor` uses against a tool's result. Nothing in this package imports
back from `guardrails.agent` — the dependency runs one way.
"""

from __future__ import annotations

from .authorization_agent import AuthorizationAgent
from .authorization_capabilities import AuthorizationCapabilities
from .authorization_tools import (
    AUTHORIZATION_AGENT_TOOLS, AUTHORIZATION_TOOL_NAMES, AuthorizationContext,
)
from .capabilities import CapabilityDenied, PIICapabilities
from .content_safety_agent import ContentSafetyAgent
from .content_tools import CONTENT_AGENT_TOOLS, CONTENT_TOOL_NAMES
from .grounding_agent import GroundingAgent
from .grounding_tools import GROUNDING_AGENT_TOOLS, GROUNDING_TOOL_NAMES
from .guardrail_capabilities import FORBIDDEN_CAPABILITIES, deny_if_forbidden
from .guardrail_capabilities import request as request_guardrail_capability
from .guardrail_supervisor import GuardrailSupervisor
from .guardrail_tools import ALLOWED_GUARDRAIL_TOOLS, GUARDRAIL_TOOLS
from .injection_agent import PromptInjectionAgent
from .injection_tools import INJECTION_AGENT_TOOLS, INJECTION_TOOL_NAMES
from .pii_agent import PIIAgent
from .scope_agent import ScopeAgent
from .scope_tools import SCOPE_AGENT_TOOLS, SCOPE_TOOL_NAMES
from .supervisor import SUPERVISOR_AGENTS, AgentNotRegistered, Supervisor
from .tools import PII_AGENT_TOOLS, PII_TOOL_NAMES, ToolNotAllowed
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, AgentState,
    GuardrailAction, GuardrailDecision, GuardrailPlan, GuardrailSupervisorResult,
    PIIFinding, SupervisorPlan, SupervisorResult, ToolCall, ToolResult, TraceEvent,
)

__all__ = [
    "ActionOutcome", "AgentDecision", "AgentNotRegistered", "AgentPlan",
    "AgentResult", "AgentState", "ALLOWED_GUARDRAIL_TOOLS", "AuthorizationAgent",
    "AuthorizationCapabilities", "AuthorizationContext", "AUTHORIZATION_AGENT_TOOLS",
    "AUTHORIZATION_TOOL_NAMES", "CapabilityDenied", "CONTENT_AGENT_TOOLS",
    "CONTENT_TOOL_NAMES", "ContentSafetyAgent", "deny_if_forbidden",
    "FORBIDDEN_CAPABILITIES", "GroundingAgent", "GROUNDING_AGENT_TOOLS",
    "GROUNDING_TOOL_NAMES", "GuardrailAction", "GuardrailDecision", "GuardrailPlan",
    "GUARDRAIL_TOOLS", "GuardrailSupervisor", "GuardrailSupervisorResult",
    "INJECTION_AGENT_TOOLS", "INJECTION_TOOL_NAMES", "PIIAgent", "PIICapabilities",
    "PIIFinding", "PII_AGENT_TOOLS", "PII_TOOL_NAMES", "PromptInjectionAgent",
    "request_guardrail_capability", "ScopeAgent", "SCOPE_AGENT_TOOLS",
    "SCOPE_TOOL_NAMES", "SUPERVISOR_AGENTS", "Supervisor", "SupervisorPlan",
    "SupervisorResult", "ToolCall", "ToolNotAllowed", "ToolResult", "TraceEvent",
]
