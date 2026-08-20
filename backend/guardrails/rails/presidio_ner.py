"""Presidio as the cheap layer under the entity judge.

`pii.entities` exists because a name has no shape a regex can match. Until now
the only thing that could find one was a model call — 3.6 seconds and a request
to an API for every prompt with a capital letter in it. Presidio's NER does the
same job locally in about a second and for nothing, which changes what the rail
costs enough to change where it can run.

It is a *layer*, not a replacement, and the difference matters. Run against
one ordinary sentence, Presidio returned:

    PERSON            0.85  'Anitha Selvam'          <- what we want
    PERSON            0.85  'mobile 9962214477'      <- would mask the word "mobile"
    ORGANIZATION      0.85  'SSN'
    URL               0.50  'anitha.se'              <- half an email address
    US_BANK_NUMBER    0.05  '9962214477'
    US_DRIVER_LICENSE 0.01  '9962214477'

...and missed the street address completely. So three things happen to every
span before it is allowed near the vault:

    confidence   below `pii.entity_confidence` is dropped outright.

    kind         only the kinds an operator enabled survive. Presidio's
                 US_SSN, EMAIL_ADDRESS and PHONE_NUMBER are discarded here
                 because the deterministic rail already found them, with a
                 checksum, in a tenth of a millisecond.

    overlap      a span touching one the deterministic rail already claimed is
                 dropped. That is what kills `PERSON 'mobile 9962214477'`: the
                 number is already a masked phone number, so the NER span is a
                 worse duplicate of a better answer.

The engine loads on first use, not at import. It costs eleven seconds to build
and most deployments answer a lot of requests that never need it.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

log = logging.getLogger("guardrails.rails.presidio")

#: Presidio's vocabulary → ours. Anything not here is dropped: either we detect
#: it deterministically already, or it is not an identifier we mask.
KIND_MAP = {
    "PERSON": "PERSON",
    "LOCATION": "ADDRESS",
    "GPE": "ADDRESS",
    "NRP": "PERSON",
}

#: ORGANIZATION is deliberately absent. The distinction that matters — a private
#: employer that identifies a person should be masked, the department the citizen
#: is writing to should not — is a judgement about what an organisation *is*, and
#: NER only knows that it is one. Left to Presidio it masked "Chennai
#: Corporation", the desk's own name, in a sentence about where to go. The judge
#: is told the difference and keeps that kind.
JUDGE_ONLY_KINDS = {"ORGANISATION"}

#: A house number immediately before a name-like span. spaCy's small English
#: model reliably tags "Anna Salai" and drops the "14" in front of it, so the
#: number — the part that narrows a street to a household — was the one piece
#: left in the clear. Matched against the text to the left of a span.
_HOUSE_NUMBER = re.compile(r"(?:(?:no|door|plot|flat|house)\.?\s*)?\d+[A-Za-z]?[,/\s-]*$",
                           re.I)


def _extend_left(text: str, start: int) -> int:
    """Pull a leading house number into the span, if one is sitting there."""
    head = text[max(0, start - 24):start]
    m = _HOUSE_NUMBER.search(head)
    return start - len(m.group(0)) if m else start


#: Loaded once, shared. Building the spaCy pipeline takes about eleven seconds
#: and is pure CPU, so a second one buys nothing.
_engine: Any = None
_engine_failed = False
_lock = threading.Lock()


def available() -> bool:
    """Is presidio importable at all? Cheap — no engine is built."""
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        return False
    return True


def engine() -> Any:
    """The analyzer, built on first use.

    Returns None if presidio or its language model is missing, so the caller
    falls through to the judge rather than failing the request. A missing
    optional dependency is a capability question, not an error.
    """
    global _engine, _engine_failed
    if _engine is not None or _engine_failed:
        return _engine
    with _lock:
        if _engine is not None or _engine_failed:
            return _engine
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            _engine = AnalyzerEngine(nlp_engine=provider.create_engine(),
                                     supported_languages=["en"])
            log.info("presidio analyzer ready")
        except Exception as exc:  # noqa: BLE001 — any failure means "not available"
            _engine_failed = True
            log.warning("presidio unavailable, falling back to the judge: %s", exc)
    return _engine


def find(text: str, kinds: set[str], min_confidence: float,
         taken: list[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    """Named entities Presidio is confident about, filtered to what we mask.

    `taken` is the spans the deterministic rail already claimed. Anything
    touching one is dropped — the regex found it with a checksum, and a
    lower-confidence NER guess over the same characters is a worse answer.
    """
    an = engine()
    if an is None or not text.strip():
        return []
    try:
        results = an.analyze(text=text, language="en")
    except Exception as exc:  # noqa: BLE001 — never fail a request on this
        log.warning("presidio analyze failed: %s", exc)
        return []

    claimed = list(taken or [])
    out: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda x: (-x.score, x.start)):
        kind = KIND_MAP.get(r.entity_type)
        if kind is None or kind not in kinds:
            continue
        if r.score < min_confidence:
            continue
        if any(r.start < end and start < r.end for start, end in claimed):
            continue
        start = r.start
        if kind in ("ADDRESS", "PERSON"):
            widened = _extend_left(text, start)
            # Only take the number if nothing else has claimed it.
            if widened != start and not any(widened < e and s2 < start
                                            for s2, e in claimed):
                start = widened
                # A street mislabelled PERSON is still an address once the
                # house number is attached to it.
                kind = "ADDRESS"
        raw = text[start:r.end].strip()
        if len(raw) < 2:
            continue
        out.append({"text": raw, "kind": kind, "confidence": float(r.score),
                    "start": start, "end": start + len(raw)})
        claimed.append((start, r.end))
    return out
