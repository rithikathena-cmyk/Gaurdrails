"""The PII agent's tool allowlist.

Every function here wraps something that already exists in `rails/entities.py`
or `rails/presidio_ner.py` — none of it re-implements detection or policy
lookup. This module's only job is to expose that existing behaviour to an
agent through a fixed set of names, and to make calling anything else
impossible in Python rather than merely discouraged in a prompt.

`ToolNotAllowed` is the boundary. `select()` looks a name up in a plain dict;
there is no `getattr`, no `eval`, no dynamic import anywhere in this file. An
agent cannot expand its own toolset by asking nicely, because nothing here
reads what the agent asks for except as a key into a dict it does not own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..engine import PII_ACTION_KEY, Engine
from ..types import Surface
from .types import ToolResult


class ToolNotAllowed(Exception):
    """Raised for any tool name outside the fixed allowlist below."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name!r} is not in the PII agent's tool allowlist")
        self.name = name


@dataclass(frozen=True)
class GuardrailTool:
    name: str
    fn: Callable[[dict, Engine], dict]


# ---------------------------------------------------------------------------
# The tools. Each takes (args, engine) and returns a plain, redacted dict —
# never the raw matched value. A caller who needs the raw value for a real
# action (masking) reaches the production rail directly, not through this
# trace-visible layer.
# ---------------------------------------------------------------------------
def _detect_pii_presidio(args: dict, engine: Engine) -> dict:
    """The local NER layer — `presidio_ner.find`, unchanged.

    Named entities only: people and addresses, the two kinds
    `presidio_ner.KIND_MAP` actually maps. Everything else this rail can
    find — an SSN, an email, a phone number, any other kind with no fixed
    shape for Presidio's own recognizers to key off — is
    `detect_pii_entities`'s job, so "presidio found nothing" here is an
    honest report of what this tool covers, not a disagreement to be
    explained away.
    """
    from ..rails import presidio_ner

    text = str(args.get("text", ""))
    kinds = set(engine.policy.get("pii.entity_kinds") or [])
    min_conf = float(engine.policy.get("pii.entity_confidence"))
    hits = presidio_ner.find(text, kinds, min_conf, taken=[])
    return {
        "findings": [
            {"kind": h["kind"], "start": h["start"], "end": h["end"],
             "confidence": round(float(h["confidence"]), 3)}
            for h in hits
        ],
        "available": presidio_ner.available(),
    }


def _detect_pii_entities(args: dict, engine: Engine) -> dict:
    """The free-form judge layer — `EntityRail._judge_entities`, unchanged.

    Unlike `detect_pii_regex`/`detect_pii_presidio`, this one is not gated on
    a shape or a trained label matching — it is asked to name every personal
    identifier in the text, whether or not any pattern here would recognise
    it. This is the tool worth reaching for when the other two found nothing
    but the text still reads like it could be identifying someone: a name,
    an address, an internal ID with no known format.

    `EntityRail.evaluate()`'s own cheap capitalized-word gate does not run
    here — reaching this tool at all is the PII agent's own PLAN-time
    judgement that a free-form pass is worth a real judge call, not a free
    pre-filter's. Costs one judge call (more for long text, windowed).

    Offsets, not the matched substring, go in the response — this module's
    contract everywhere else: never the raw value, only where it is.
    """
    text = str(args.get("text", ""))
    found = engine.entity_rail._judge_entities(text)  # noqa: SLF001 — the existing extractor, reused whole
    findings = []
    search_from = 0
    for e in found[:80]:
        raw = str(e.get("text", ""))
        start = text.find(raw, search_from) if raw else -1
        if start == -1:
            start = text.find(raw) if raw else -1
        end = start + len(raw) if start != -1 else -1
        if start != -1:
            search_from = end
        findings.append({
            "kind": str(e.get("kind", "")), "start": start, "end": end,
            "confidence": round(float(e.get("confidence", 0.0)), 3),
        })
    return {"findings": findings}


def _get_pii_policy(args: dict, engine: Engine) -> dict:
    """The configured action for this kind on this surface — read, not decided.

    Reuses `PII_ACTION_KEY`, the same literal surface→config-key map the
    engine's own rail dispatch uses, so this can never answer a question about
    a surface the engine itself would resolve differently.
    """
    kind = str(args.get("kind", "")).strip()
    surface_name = str(args.get("surface", "user.prompt"))
    try:
        surface = Surface(surface_name)
    except ValueError:
        surface = Surface.USER_PROMPT
    action = str(engine.policy.get(PII_ACTION_KEY[surface]))
    return {
        "kind": kind, "surface": surface.value, "action": action,
        "mask_strategy": str(engine.policy.get("pii.mask_strategy")),
        "entities_enabled": kind in set(engine.policy.get("pii.entity_kinds") or []),
        "reversible": bool(engine.policy.get("pii.reversible")),
    }


PII_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "detect_pii_presidio": GuardrailTool("detect_pii_presidio", _detect_pii_presidio),
    "detect_pii_entities": GuardrailTool("detect_pii_entities", _detect_pii_entities),
    "get_pii_policy": GuardrailTool("get_pii_policy", _get_pii_policy),
}

PII_TOOL_NAMES: tuple[str, ...] = tuple(PII_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    """The only entry point. A name not in `PII_AGENT_TOOLS` never reaches a
    function call — it raises here, before any code the name might have named
    gets a chance to run."""
    tool = PII_AGENT_TOOLS.get(name)
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
