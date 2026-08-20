"""Named entities — the personal data a regular expression cannot find.

`pii.detect` is precise and blind. It finds an SSN because an SSN has a shape
and a checksum. It cannot find **Meera Balan**, or **14 Anna Salai, Chennai**,
because a name has no shape — which is why a contact sheet ingested earlier
masked the email and the phone number and left the person's name sitting in the
index in the clear.

This rail closes that gap with a model, and is built so it costs as little as
possible:

    gate      a cheap structural pre-check runs first. If the text holds no
              capitalised word that is not a sentence opener, there is nothing
              a name detector could find, and the model is never called.

    judge     otherwise one structured-output call returns the spans, which are
              handed to the same vault the regex recognizers use — same
              masking strategies, same tokens, same authorised unmask at egress.

Findings are span-checked against the text before use. A model asked for
verbatim substrings will occasionally return a near-miss, and masking a span
that does not exist would corrupt the very text it is meant to protect.
"""

from __future__ import annotations

import re

from ..prompts import judge_prompt
from . import presidio_ner
from ..types import Detection, RailResult, Verdict, action_verdict

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "description": "Every personal identifier in the text that is not a "
                           "number pattern. Empty if there are none.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact substring, copied verbatim from the input.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "PERSON | ADDRESS | ORGANISATION | LOCATION",
                    },
                    "confidence": {"type": "number", "description": "0.0–1.0"},
                },
                "required": ["text", "kind", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

ENTITY_SYSTEM = judge_prompt("""\
List every personal identifier in the text. You are extracting, not judging: an \
identifier is reported whether or not its presence is a problem.

Return every occurrence of:
- PERSON: a person's name, including a partial name used as an identifier
- ADDRESS: a street address, house number and street, or postal address
- ORGANISATION: a named private employer or company that identifies a person
- LOCATION: a specific place that would identify a household, such as a village \
together with a house number

Do not return:
- names of public offices, departments, ministries, courts, or municipal bodies
- job titles with no name attached
- cities, districts, or states on their own
- scheme, form, statute, or programme names
- anything already written as a masked token

Copy each `text` verbatim from the input, exactly as it appears, including its \
capitalisation and any surrounding punctuation that belongs to it — the span is looked \
up in the original text and discarded if it cannot be found. Do not correct spelling, \
expand an initial, or normalise a form. Return an empty list rather than guessing.""",
                                calibrate=False)

# Cheap structural gate: a capital letter mid-sentence is the minimum evidence
# that a name could be present. No capitals, no call.
_CANDIDATE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", re.M)
_ALREADY_MASKED = re.compile(r"<[A-Z_0-9]+:[0-9a-f]{12}(?:\s…[^>]*)?>")

KINDS = {"PERSON", "ADDRESS", "ORGANISATION", "LOCATION"}


class EntityRail:
    """Model-backed PII, feeding the same vault as the regex recognizers."""

    name = "pii.entities"
    engine = "claude judge · named entities"

    def __init__(self, llm, vault, confidence_threshold: float, mask_strategy: str,
                 kinds: list[str] | None = None, engine_mode: str = "presidio+judge") -> None:
        self.llm = llm
        self.vault = vault
        self.min_conf = confidence_threshold
        self.strategy = mask_strategy
        self.kinds = {k.upper() for k in (kinds or KINDS)} & KINDS
        #: presidio | judge | presidio+judge. Local NER is a second the request
        #: does not spend on an API call, so it goes first where it is enabled.
        self.engine_mode = engine_mode

    def _replacement(self, kind: str, raw: str) -> str:
        if self.strategy == "redact":
            return "[REDACTED]"
        if self.strategy == "replace":
            return f"<{kind}>"
        if self.strategy == "hash":
            import hashlib

            return f"<{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}>"
        if self.strategy == "partial":
            return raw[0] + "*" * max(0, len(raw) - 1)
        return f"<{kind}:{self.vault.store(kind, raw)}>"

    def evaluate(self, text: str, action: str, result: RailResult,
                 prior: list[Detection] | None = None) -> RailResult:
        """`prior` is what the deterministic rail already claimed, so NER cannot
        return a worse guess over the same characters."""
        prior = prior or []
        result.unit = "count"
        result.threshold = 1.0
        result.meta = {"kinds_enabled": sorted(self.kinds), "strategy": self.strategy}

        use_presidio = self.engine_mode in ("presidio", "presidio+judge")
        use_judge = self.engine_mode in ("judge", "presidio+judge") and self.llm is not None
        if not self.kinds or not (use_presidio or use_judge):
            result.verdict = Verdict.PASS
            result.meta["skipped"] = ("no kinds enabled" if not self.kinds
                                      else "no entity engine configured")
            return result

        # The gate. Masked tokens are stripped first so their entity names do
        # not look like candidates.
        probe = _ALREADY_MASKED.sub(" ", text)
        if not _CANDIDATE.search(probe):
            result.verdict = Verdict.PASS
            result.meta.update(layer="gate", judge_skipped=True,
                               reason="no capitalised candidate in the text")
            return result

        # Local NER first. It costs about a second of CPU rather than several
        # of network, and on the ordinary case it finds the name and the judge
        # is never asked.
        items: list[dict] = []
        layer = ""
        if use_presidio:
            items = presidio_ner.find(text, self.kinds, self.min_conf,
                                      taken=[(d.start, d.end) for d in prior])
            if items:
                layer = "presidio"

        # The judge is the fallback, not the default: asked only when the cheap
        # layer found nothing and there is still a candidate in the text.
        # Presidio never returns ORGANISATION, so a text whose only identifier is
        # one still has to reach the judge.
        if not items and use_judge:
            found = self.llm.judge(ENTITY_SYSTEM, text, ENTITY_SCHEMA)
            items = list((found.get("entities") or [])[:40])
            layer = "judge"

        spans: list[tuple[int, int, str, str, float]] = []
        dropped = 0
        for item in items:
            raw = str(item.get("text", "")).strip()
            kind = str(item.get("kind", "")).strip().upper()
            conf = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            if not raw or kind not in self.kinds or conf < self.min_conf:
                continue
            start = item.get("start")
            if start is None or text[start:start + len(raw)] != raw:
                start = text.find(raw)
            if start < 0:
                # Returned a span that is not in the text. Masking it would
                # rewrite something else, so it is counted and discarded.
                dropped += 1
                continue
            spans.append((start, start + len(raw), kind, raw, conf))

        # Longest match wins on overlap, same rule the regex recognizers use.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[tuple[int, int, str, str, float]] = []
        last_end = -1
        for span in spans:
            if span[0] >= last_end:
                kept.append(span)
                last_end = span[1]

        result.score = float(len(kept))
        result.detections = [
            Detection(kind=kind, value=raw, start=start, end=end, confidence=conf,
                      note="named entity")
            for start, end, kind, raw, conf in kept
        ]
        result.meta.update(layer=layer or "none",
                           by_type=sorted({k for _, _, k, _, _ in kept}),
                           unverifiable_spans=dropped)

        if not kept:
            result.verdict = Verdict.PASS
            return result

        result.verdict = action_verdict(action, Verdict.MASK)
        if result.verdict is Verdict.MASK:
            out = text
            for start, end, kind, raw, _ in sorted(kept, reverse=True):
                out = out[:start] + self._replacement(kind, raw) + out[end:]
            result.text_out = out
        return result
