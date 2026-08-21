"""Autonomous guardrail agents — reasoning around the existing deterministic
rails, never a replacement for them.

Named `agents` (plural) to stay distinct from `guardrails.agent` (singular),
the conversational tool-use loop that drives chat and RAG. Nothing in one
package imports from the other.
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
from .injection_agent import PromptInjectionAgent
from .injection_tools import INJECTION_AGENT_TOOLS, INJECTION_TOOL_NAMES
from .pii_agent import PIIAgent
from .scope_agent import ScopeAgent
from .scope_tools import SCOPE_AGENT_TOOLS, SCOPE_TOOL_NAMES
from .supervisor import SUPERVISOR_AGENTS, AgentNotRegistered, Supervisor
from .tools import PII_AGENT_TOOLS, PII_TOOL_NAMES, ToolNotAllowed
from .types import (
    ActionOutcome, AgentDecision, AgentPlan, AgentResult, AgentState,
    GuardrailAction, PIIFinding, SupervisorPlan, SupervisorResult, ToolCall,
    ToolResult, TraceEvent,
)

__all__ = [
    "ActionOutcome", "AgentDecision", "AgentNotRegistered", "AgentPlan",
    "AgentResult", "AgentState", "AuthorizationAgent", "AuthorizationCapabilities",
    "AuthorizationContext", "AUTHORIZATION_AGENT_TOOLS", "AUTHORIZATION_TOOL_NAMES",
    "CapabilityDenied", "CONTENT_AGENT_TOOLS", "CONTENT_TOOL_NAMES",
    "ContentSafetyAgent", "GroundingAgent", "GROUNDING_AGENT_TOOLS",
    "GROUNDING_TOOL_NAMES", "GuardrailAction", "INJECTION_AGENT_TOOLS",
    "INJECTION_TOOL_NAMES", "PIIAgent", "PIICapabilities", "PIIFinding",
    "PII_AGENT_TOOLS", "PII_TOOL_NAMES", "PromptInjectionAgent", "ScopeAgent",
    "SCOPE_AGENT_TOOLS", "SCOPE_TOOL_NAMES", "SUPERVISOR_AGENTS", "Supervisor",
    "SupervisorPlan", "SupervisorResult", "ToolCall", "ToolNotAllowed",
    "ToolResult", "TraceEvent",
]
