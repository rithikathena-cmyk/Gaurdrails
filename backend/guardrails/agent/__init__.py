"""The agent. The only place a model chooses what happens next.

Everywhere else in this package a model scores something and code decides.
Here it picks a tool, reads what came back, and picks again — which is why
this is one directory and the rails are another.

    tools.py    what the agent may call, and what each tool is entitled to see
    runner.py   the loop, and the rails on every edge of it

The adjudicator used to live here because it is model-driven. It moved to
`rails/`: what it does is rule on a verdict, and this package is the loop that
chooses actions.

Split because they answer different questions. `tools.py` is "what can this
thing do"; `runner.py` is "what stops it".
"""

from .runner import (
    SYSTEM_PROMPT,
    AgentResult,
    AgentRunner,
    PendingApproval,
    ToolCall,
)
from .tools import MASK_TOKEN, TOOLS, CLAIMS, FEES, Tool, ToolContext

__all__ = [
    "CLAIMS",
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
