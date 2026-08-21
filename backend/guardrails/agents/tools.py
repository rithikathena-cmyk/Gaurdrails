"""The PII agent's tool allowlist.

Every function here wraps something that already exists in `rails/pii.py` or
`rails/presidio_ner.py` — none of it re-implements detection, checksums, or
policy lookup. This module's only job is to expose that existing behaviour to
an agent through a fixed set of names, and to make calling anything else
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
# The four tools. Each takes (args, engine) and returns a plain, redacted dict
# — never the raw matched value. A caller who needs the raw value for a real
# action (masking, a checksum re-check) reaches the production rail directly,
# not through this trace-visible layer.
# ---------------------------------------------------------------------------
def _detect_pii_regex(args: dict, engine: Engine) -> dict:
    """The deterministic regex + checksum layer — `PIIRail._detect`, unchanged."""
    text = str(args.get("text", ""))
    pairs = engine.pii_rail._detect(text)  # noqa: SLF001 — the existing detector, reused whole
    return {
        "findings": [
            {"kind": det.kind, "start": det.start, "end": det.end,
             "confidence": round(det.confidence, 3),
             "checksum_verified": rec.check is not None}
            for det, rec in pairs
        ],
    }


def _detect_pii_presidio(args: dict, engine: Engine) -> dict:
    """The local NER layer — `presidio_ner.find`, unchanged.

    Named entities only: people, addresses, organisations. It was never going
    to find an SSN — that is `detect_pii_regex`'s job — so "presidio found
    nothing" here is an honest report of what this tool covers, not a
    disagreement to be explained away.
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


def _classify_pii_type(args: dict, engine: Engine) -> dict:
    """What kind of thing a prior detection was — not a new detection pass.

    Looks the kind up against the same `RECOGNIZERS` table the production
    rail masks with, so "is this checksum-backed" answers the same way here
    as it does in the rail that actually acts on it.
    """
    from ..rails.pii import RECOGNIZERS

    kind = str(args.get("kind", "")).strip()
    rec = next((r for r in RECOGNIZERS if r.entity == kind), None)
    if rec is None:
        return {"kind": kind, "known": False}
    return {
        "kind": kind, "known": True,
        "checksum_backed": rec.check is not None,
        "base_confidence": round(rec.confidence, 3),
        "reveal_chars": rec.reveal,
    }


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
        "entities_enabled": kind in set(engine.policy.get("pii.entities") or []),
        "reversible": bool(engine.policy.get("pii.reversible")),
    }


PII_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "detect_pii_regex": GuardrailTool("detect_pii_regex", _detect_pii_regex),
    "detect_pii_presidio": GuardrailTool("detect_pii_presidio", _detect_pii_presidio),
    "classify_pii_type": GuardrailTool("classify_pii_type", _classify_pii_type),
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
