"""An out-of-the-box guardrail stack for LLM applications.

    from backend.guardrails import Engine, load, Claude

    policy = load("config/policy.yaml")
    engine = Engine(policy, Claude())

    engine.ingest("Fee circular", open("circular.txt").read())   # rails, then index
    result = engine.converse("What documents do I need to renew a trade licence?")
    agent = AgentRunner(engine).run("Check claim CLM-40028871 and file a grievance")

    print(result.reply, result.trace.to_dict())
"""

from .config import (
    ConfigError,
    Policy,
    coerce,
    load,
    overrides_path_for,
    reset_overrides,
    save_overrides,
)
from .agent import AgentResult, AgentRunner, PendingApproval, Tool, ToolCall, TOOLS
from .engine import ConversationResult, Engine, IngestResult
from .knowledge import Corpus, Document, IngestError, chunk_text, extract
from .explain import Violation, explain, summarise
from .llm import Claude, LLMError, Refusal
from .evaluation import scenarios
from .registry import (
    FAMILIES,
    LOCK_META,
    PARAMS,
    SEVERITY_LEVELS,
    SURFACES,
    as_payload,
    control_for,
)
from .tracing import AuditLog, Tracer
from .types import EvaluationResult, RailResult, Surface, Trace, Verdict, precedence

__version__ = "1.2.0"

__all__ = [
    "TOOLS",
    "AgentResult",
    "AgentRunner",
    "AuditLog",
    "Corpus",
    "Document",
    "Claude",
    "ConfigError",
    "ConversationResult",
    "Engine",
    "IngestError",
    "IngestResult",
    "PendingApproval",
    "Tool",
    "ToolCall",
    "Violation",
    "EvaluationResult",
    "FAMILIES",
    "LLMError",
    "LOCK_META",
    "PARAMS",
    "Policy",
    "RailResult",
    "Refusal",
    "SEVERITY_LEVELS",
    "SURFACES",
    "Surface",
    "Trace",
    "Tracer",
    "Verdict",
    "as_payload",
    "chunk_text",
    "explain",
    "extract",
    "coerce",
    "control_for",
    "load",
    "overrides_path_for",
    "precedence",
    "reset_overrides",
    "save_overrides",
    "summarise",
]
