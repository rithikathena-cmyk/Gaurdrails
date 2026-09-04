"""The document-security agent's tool allowlist, and the cheap gate in front
of the agent itself.

Every detector function here wraps something that already exists in
`rails/content.py` or `rails/deberta_injection_check.py` — none of it
re-implements pattern matching or classification. `detect_extraction_artifacts`
is the one new detector this package adds: a heuristic for the class of byte
that PDF/OCR extraction produces (broken icon glyphs, control characters) so a
resume's own contact-info formatting does not read as obfuscation. It reports
a ratio, never a verdict — see its own docstring.

`cheap_risk_score` is not a tool an agent calls — it is what `Engine.ingest()`
calls directly, before ever constructing `DocumentSecurityAgent`, to decide
whether a judge call is worth paying for at all. See its own docstring for why
it has no auto-quarantine ceiling, only an escalation floor.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..engine import Engine
from .tools import ToolNotAllowed
from .types import ToolResult


@dataclass(frozen=True)
class GuardrailTool:
    name: str
    fn: Callable[[dict, Engine], dict]


def _detect_injection_patterns(args: dict, engine: Engine) -> dict:
    """The deterministic layer — `content.INJECTION_PATTERNS`, unchanged.
    Free, and the same table that short-circuits the judge in the production
    prompt-attack rail."""
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

    Off unless `ingest.security_agent.engine` includes `local`, and `score()`
    already returns `None` rather than a false confidence when the model is
    not loaded — reported here as `available: False`, exactly as the
    production rail treats it.
    """
    from ..rails import deberta_injection_check as local

    text = str(args.get("text", ""))
    score = local.score(text)
    return {
        "local_score": round(score, 3) if score is not None else None,
        "available": score is not None,
        "looks_like_meta_question": local.looks_like_a_meta_question(text),
    }


#: The class of byte a broken icon-font mapping or a PDF extraction artifact
#: produces — C0/C1 control characters outside ordinary whitespace. Not a
#: security signature; a known, benign extraction artifact (see this
#: module's docstring and `document_security_agent.py`'s DECISION_SYSTEM).
_CONTROL_CHAR = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

#: A digit, `@`, or `+` within a short window of a control character —
#: contact-info-shaped, the same shape a phone number or email address takes.
_CONTACT_SHAPE = re.compile(r"[\d@+]")


def _detect_extraction_artifacts(args: dict, engine: Engine) -> dict:
    """Ratio of extraction-noise bytes, and whether they cluster near
    contact-info-shaped text — evidence only, never a verdict on its own.

    Built directly from the false positive that motivated this agent: a
    resume's own icon-font glyphs (phone/email/LinkedIn/GitHub icons) came
    back from PDF text extraction as raw control bytes sitting immediately
    before the contact line, and got misread elsewhere in this stack as
    obfuscated/encoded injection content. A high ratio here does not mean
    anything by itself — see `ingest.security_agent.classifier_authority`:
    this tool reports a signal, the agent decides what it means in context.
    """
    text = str(args.get("text", ""))
    if not text:
        return {"control_char_ratio": 0.0, "control_chars": 0, "near_contact_shape": False}
    hits = list(_CONTROL_CHAR.finditer(text))
    ratio = len(hits) / len(text)
    near_contact = any(
        _CONTACT_SHAPE.search(text[max(0, m.start() - 20):m.start() + 20])
        for m in hits
    )
    return {
        "control_char_ratio": round(ratio, 4),
        "control_chars": len(hits),
        "near_contact_shape": near_contact,
    }


def _get_document_security_policy(args: dict, engine: Engine) -> dict:
    """The configured threshold and action — read, not decided. Reuses the
    same `policy.get` calls `Engine.ingest()` itself reads at request time."""
    return {
        "action": str(engine.policy.get("ingest.security_agent.action")),
        "risk_threshold": float(engine.policy.get("ingest.security_agent.risk_threshold")),
        "engine_mode": str(engine.policy.get("ingest.security_agent.engine")),
    }


DOCUMENT_SECURITY_AGENT_TOOLS: dict[str, GuardrailTool] = {
    "detect_injection_patterns": GuardrailTool("detect_injection_patterns", _detect_injection_patterns),
    "classify_injection": GuardrailTool("classify_injection", _classify_injection),
    "detect_extraction_artifacts": GuardrailTool("detect_extraction_artifacts", _detect_extraction_artifacts),
    "get_document_security_policy": GuardrailTool("get_document_security_policy", _get_document_security_policy),
}

DOCUMENT_SECURITY_TOOL_NAMES: tuple[str, ...] = tuple(DOCUMENT_SECURITY_AGENT_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    tool = DOCUMENT_SECURITY_AGENT_TOOLS.get(name)
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


# ---------------------------------------------------------------------------
# The cheap gate. Not a tool — called directly by `Engine.ingest()`, before
# `DocumentSecurityAgent` is ever constructed, to decide whether a judge call
# is worth paying for at all.
# ---------------------------------------------------------------------------
def cheap_risk_score(text: str, engine: Engine) -> tuple[float, dict[str, Any]]:
    """Pure deterministic/local-model signal, 0.0-1.0, no judge call.

    This is a floor for *escalation*, never a ceiling for blocking —
    `ingest.security_agent.risk_threshold` only decides whether
    `DocumentSecurityAgent` gets a look; it cannot itself set MALICIOUS or
    QUARANTINE (`ingest.security_agent.classifier_authority`, a locked
    registry row, documents this as an invariant). The extraction-artifact
    ratio never feeds the score directly for the same reason: a resume's own
    icon-font bytes must not, by themselves, look risky.
    """
    from ..rails.content import INJECTION_PATTERNS
    from ..rails import deberta_injection_check as local

    mode = str(engine.policy.get("ingest.security_agent.engine"))
    if mode == "off" or not text.strip():
        return 0.0, {"engine": mode}

    pattern_score = 0.0
    for pattern, kind, score in INJECTION_PATTERNS:
        if pattern.search(text) and score > pattern_score:
            pattern_score = score

    local_score = local.score(text) if mode in ("local", "local+judge") else None
    artifacts = _detect_extraction_artifacts({"text": text}, engine)

    signals = {
        "pattern_score": round(pattern_score, 3),
        "local_score": round(local_score, 3) if local_score is not None else None,
        **artifacts,
    }
    combined = max(pattern_score, local_score or 0.0)
    return combined, signals
