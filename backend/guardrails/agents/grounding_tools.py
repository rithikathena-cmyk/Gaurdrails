"""The grounding agent's tool allowlist.

`extract_claims` and `check_local_entailment` wrap `groundedness_check.py`'s
`claims()` and `consistency()` directly — the same sentence splitter and the
same local NLI model the production `GroundingRail` uses for its own local
layer. Chunks are bounded to `grounding.context_window` before either tool
sees them, the same bound the rail itself applies, so this agent's calls
never duplicate more of a retrieved document than the rail already would.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..engine import Engine
from .tools import ToolNotAllowed
from .types import ToolResult


@dataclass(frozen=True)
class GuardrailTool:
    name: str
    fn: Callable[[dict, Engine], dict]


def _bounded_chunks(args: dict, engine: Engine) -> list[str]:
    chunks = list(args.get("chunks") or [])
    window = int(engine.policy.get("grounding.context_window"))
    return chunks[:window]


def _extract_claims(args: dict, engine: Engine) -> dict:
    from ..rails import groundedness_check

    answer = str(args.get("answer", ""))
    found = groundedness_check.claims(answer)
    return {"claims": [{"n": i + 1, "text": c} for i, c in enumerate(found)],
           "claim_count": len(found)}


def _check_local_entailment(args: dict, engine: Engine) -> dict:
    from ..rails import groundedness_check

    answer = str(args.get("answer", ""))
    chunks = _bounded_chunks(args, engine)
    result = groundedness_check.consistency(answer, chunks)
    if result is None:
        return {"available": False, "reason": "no local model, or nothing retrieved"}
    return {"available": True, **result}


def _get_grounding_policy(args: dict, engine: Engine) -> dict:
    return {
        "consistency_threshold": float(engine.policy.get("grounding.consistency.threshold")),
        "relevance_threshold": float(engine.policy.get("grounding.relevance.threshold")),
        "context_window": int(engine.policy.get("grounding.context_window")),
        "require_citations": bool(engine.policy.get("grounding.require_citations")),
        "action_on_fail": str(engine.policy.get("grounding.action_on_fail")),
    }


GROUNDING_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "extract_claims": GuardrailTool("extract_claims", _extract_claims),
    "check_local_entailment": GuardrailTool("check_local_entailment", _check_local_entailment),
    "get_grounding_policy": GuardrailTool("get_grounding_policy", _get_grounding_policy),
}

GROUNDING_TOOL_NAMES: tuple[str, ...] = tuple(GROUNDING_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    tool = GROUNDING_AGENT_TOOLS.get(name)
    if tool is None:
        raise ToolNotAllowed(name)
    began = time.perf_counter()
    try:
        result = tool.fn(args, engine)
        return ToolResult(call_id=call_id, tool=name, status="ok",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          result=result)
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        return ToolResult(call_id=call_id, tool=name, status="error",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          error=f"{type(exc).__name__}: {exc}")
