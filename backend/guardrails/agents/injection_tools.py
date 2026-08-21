"""The prompt-injection agent's tool allowlist.

Every function here wraps something that already exists in `rails/content.py`
or `rails/deberta_injection_check.py` — none of it re-implements pattern
matching or classification. `INJECTION_PATTERNS` is imported, not copied;
`deberta_injection_check.score()` is called, not re-scored.

Enforcement is the same shape as `tools.py`'s: `call()` looks a name up in a
plain dict. There is no `getattr`, no `eval`, no dynamic import keyed on
anything an agent supplies. `ToolNotAllowed` is `agents.tools`'s own
exception, reused rather than duplicated — the failure mode ("a name outside
this agent's allowlist") is identical regardless of which agent hit it.
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


def _detect_injection_patterns(args: dict, engine: Engine) -> dict:
    """The deterministic layer — `content.INJECTION_PATTERNS`, unchanged.
    Free, and what short-circuits the judge in the production rail."""
    from ..rails.content import INJECTION_PATTERNS

    text = str(args.get("text", ""))
    matches = []
    for pattern, kind, score in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            matches.append({"technique": kind, "confidence": round(score, 3),
                            "start": m.start(), "end": m.end()})
    return {"matches": matches}


def _classify_injection(args: dict, engine: Engine) -> dict:
    """The local classifier layer — `deberta_injection_check.score`, unchanged.

    Off by default (`prompt_attack.engine` ships as `judge`), and `score()`
    already returns `None` rather than a false confidence when the model
    is not loaded — reported here as `available: False` instead of a number,
    exactly as the production rail treats it.

    Known to score a legitimate meta-question about the service as highly as
    a real attack — `looks_like_a_meta_question` is the production rail's own
    deterministic guard against exactly that, reused rather than re-derived.
    """
    from ..rails import deberta_injection_check as local

    text = str(args.get("text", ""))
    score = local.score(text)
    return {
        "local_score": round(score, 3) if score is not None else None,
        "available": score is not None,
        "looks_like_meta_question": local.looks_like_a_meta_question(text),
    }


#: The subset of `INJECTION_PATTERNS` concerned with the model being told it
#: is something else, or with text imitating a system-level message — the
#: instruction-hierarchy question specifically, as distinct from a plain
#: instruction-override phrase. Same table, a different lens over it.
_HIERARCHY_TECHNIQUES = {"role_play", "delimiter_confusion"}


def _inspect_instruction_hierarchy(args: dict, engine: Engine) -> dict:
    from ..rails.content import INJECTION_PATTERNS

    text = str(args.get("text", ""))
    hits = []
    for pattern, kind, score in INJECTION_PATTERNS:
        if kind not in _HIERARCHY_TECHNIQUES:
            continue
        m = pattern.search(text)
        if m:
            hits.append({"technique": kind, "confidence": round(score, 3),
                        "start": m.start(), "end": m.end()})
    return {"hierarchy_concerns": hits}


def _get_injection_policy(args: dict, engine: Engine) -> dict:
    """The configured threshold and action — read, not decided. Reuses the
    same `policy.get` calls the production rail itself reads at request time."""
    return {
        "threshold": float(engine.policy.get("prompt_attack.threshold")),
        "action": str(engine.policy.get("prompt_attack.action")),
        "engine_mode": str(engine.policy.get("prompt_attack.engine")),
    }


INJECTION_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "detect_injection_patterns": GuardrailTool("detect_injection_patterns", _detect_injection_patterns),
    "classify_injection": GuardrailTool("classify_injection", _classify_injection),
    "inspect_instruction_hierarchy": GuardrailTool("inspect_instruction_hierarchy", _inspect_instruction_hierarchy),
    "get_injection_policy": GuardrailTool("get_injection_policy", _get_injection_policy),
}

INJECTION_TOOL_NAMES: tuple[str, ...] = tuple(INJECTION_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    tool = INJECTION_AGENT_TOOLS.get(name)
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
