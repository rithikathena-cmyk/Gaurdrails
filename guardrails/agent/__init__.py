"""The agentic layer.

    tools.py        what the agent may call, and what each tool is entitled to see
    runner.py       the loop, and the rails on every edge of it
    adjudicator.py  the second opinion on decisions a threshold made by a hair

Split because they answer different questions. `tools.py` is "what can this
thing do"; `runner.py` is "what stops it".
"""

from .adjudicator import (
    ADJUDICABLE,
    DOWNGRADE_FLOOR,
    Adjudication,
    Adjudicator,
)
from .runner import (
    SYSTEM_PROMPT,
    AgentResult,
    AgentRunner,
    PendingApproval,
    ToolCall,
)
from .tools import MASK_TOKEN, TOOLS, CLAIMS, FEES, Tool, ToolContext

__all__ = [
    "ADJUDICABLE",
    "CLAIMS",
    "DOWNGRADE_FLOOR",
    "Adjudication",
    "Adjudicator",
    "FEES",
    "MASK_TOKEN",
    "SYSTEM_PROMPT",
    "TOOLS",
    "AgentResult",
    "AgentRunner",
    "PendingApproval",
    "Tool",
    "ToolCall",
    "ToolContext",
]
