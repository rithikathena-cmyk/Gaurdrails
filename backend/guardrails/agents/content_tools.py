"""The content-safety agent's tool allowlist.

Wraps `rails/toxicity_check.py` and `rails/content.py`'s `CATEGORIES` —
nothing here re-implements scoring. Only two tools exist because that is
what actually exists to wrap: one local classifier, one policy lookup. The
"is this actual harm or discussion/reporting/help-seeking" judgment has no
deterministic tool behind it in this codebase — it is the semantic call the
agent's own DECIDE step makes, the same division `content.py`'s own
`ContentRail` already draws between its local layer and its judge.
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


def _score_content_categories(args: dict, engine: Engine) -> dict:
    """The local layer — `toxicity_check.score`, unchanged. Reports which
    categories this classifier can even speak to, same as the production
    rail: a category outside `COVERED` was never evaluated, not cleared."""
    from ..rails import toxicity_check

    text = str(args.get("text", ""))
    scores = toxicity_check.score(text)
    if scores is None:
        return {"available": False, "scores": {}, "covered": sorted(toxicity_check.COVERED),
                "uncovered": sorted(toxicity_check.UNCOVERED)}
    return {
        "available": True,
        "scores": {c: round(v, 3) for c, v in sorted(scores.items())},
        "covered": sorted(toxicity_check.COVERED),
        "uncovered": sorted(toxicity_check.UNCOVERED),
    }


def _get_content_policy(args: dict, engine: Engine) -> dict:
    """The configured threshold and action per category — read, not decided."""
    from ..rails.content import CATEGORIES

    category = str(args.get("category", "")).strip()
    enabled = [c for c in engine.policy.get("content.enabled_categories") or [] if c in CATEGORIES]
    result = {"enabled_categories": enabled, "action": str(engine.policy.get("content.action.user_prompt"))}
    if category:
        result["category"] = category
        result["threshold"] = float(engine.policy.get(f"content.{category}.threshold"))
        result["enabled"] = category in enabled
    return result


CONTENT_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "score_content_categories": GuardrailTool("score_content_categories", _score_content_categories),
    "get_content_policy": GuardrailTool("get_content_policy", _get_content_policy),
}

CONTENT_TOOL_NAMES: tuple[str, ...] = tuple(CONTENT_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    tool = CONTENT_AGENT_TOOLS.get(name)
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
