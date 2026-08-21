"""The scope agent's tool allowlist.

`check_domain_vocabulary` calls `engine.scope_rail._hits(text)` directly —
the exact set-intersection check the production `ScopeRail` runs, against
the exact same configured vocabulary, not a re-derived copy of it.
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


def _check_domain_vocabulary(args: dict, engine: Engine) -> dict:
    text = str(args.get("text", ""))
    hits = engine.scope_rail._hits(text)  # noqa: SLF001 — the production check, reused whole
    return {"matched_terms": hits, "in_vocabulary": bool(hits)}


def _get_scope_policy(args: dict, engine: Engine) -> dict:
    return {
        "threshold": float(engine.policy.get("scope.threshold")),
        "action": str(engine.policy.get("scope.action")),
        "domain_terms_configured": len(engine.policy.get("scope.domain_terms") or []),
    }


SCOPE_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "check_domain_vocabulary": GuardrailTool("check_domain_vocabulary", _check_domain_vocabulary),
    "get_scope_policy": GuardrailTool("get_scope_policy", _get_scope_policy),
}

SCOPE_TOOL_NAMES: tuple[str, ...] = tuple(SCOPE_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    tool = SCOPE_AGENT_TOOLS.get(name)
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
