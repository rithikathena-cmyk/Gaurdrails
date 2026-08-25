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

import concurrent.futures as futures
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
- a cooperative society, union, federation, institute, academy, board, or corporation \
operating under a Registrar of Cooperative Societies, a government department, or \
similar statutory oversight — these are public-sector bodies, not private employers, \
whatever legal form their name takes ("... Union", "... Federation", "Institute of \
...", "... Cooperative Management", and so on)
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

#: (trailing, leading) characters safe to reveal under partial masking, per
#: kind — the same ceiling concept as `Recognizer.reveal`/`reveal_prefix` in
#: `pii.py`. Zero for every kind today: unlike a phone number's last four
#: digits, no prefix or suffix of a name, address, or organisation is
#: established here as safe to leave visible.
_REVEAL_CAP: dict[str, tuple[int, int]] = {}

# A prompt or a reply is a few hundred characters; an ingested document can be
# up to `ingest.max_document_chars` (200,000 by default) — and this rail runs
# on the *whole* document, before it is ever chunked for the index. Sending
# that whole in one judge call was two failures waiting to happen at once: the
# call routinely blew `ingest.latency_budget_ms` by itself, and a document
# with genuinely many named entities returns a JSON array long enough to
# exceed `max_tokens` and come back truncated — an `LLMError` on a document
# that had done nothing wrong except be long. Below this size nothing changes
# — one window is the whole text, exactly as before.
_JUDGE_WINDOW_CHARS = 6000
#: So an entity sitting across a window boundary still appears whole in at
#: least one window rather than being split in both and found in neither.
_JUDGE_WINDOW_OVERLAP = 200
#: Generous for one window's worth of entities without paying for a reply
#: sized to a whole 200,000-character document that mostly is not this rail's.
_JUDGE_MAX_TOKENS = 4096
_JUDGE_MAX_WORKERS = 8


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """`text`, split into overlapping slices small enough to judge reliably.

    A single slice — the common case, every prompt and reply — is returned
    as one window, byte-identical to the whole text.
    """
    if len(text) <= size:
        return [text]
    out: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        out.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += step
    return out


def _spans_of(items: list[dict], text: str) -> list[tuple[int, int]]:
    """Resolve verbatim findings to offsets, skipping any that are not present.

    A model asked for verbatim substrings occasionally returns a near-miss. Such
    a finding cannot corroborate anything, because there is no place in the text
    it agrees about.
    """
    out: list[tuple[int, int]] = []
    for item in items:
        raw = str(item.get("text", "")).strip()
        if not raw:
            continue
        start = text.find(raw)
        if start >= 0:
            out.append((start, start + len(raw)))
    return out


class EntityRail:
    """Model-backed PII, feeding the same vault as the regex recognizers."""

    name = "pii.entities"
    engine = "claude judge · named entities"

    def __init__(self, llm, vault, confidence_threshold: float, mask_strategy: str,
                 kinds: list[str] | None = None, engine_mode: str = "presidio+judge",
                 allowlist: list[str] | None = None,
                 partial_reveal: int = 0, partial_reveal_prefix: int = 0) -> None:
        self.llm = llm
        self.vault = vault
        self.min_conf = confidence_threshold
        self.strategy = mask_strategy
        self.kinds = {k.upper() for k in (kinds or KINDS)} & KINDS
        # Same knobs `pii.py` reads, same reason: a caller may configure a
        # generous reveal count meant for a phone number's last four digits,
        # but a name or an address has no per-kind ceiling raising it above
        # zero here yet — see `_REVEAL_CAP`.
        self.partial_reveal = partial_reveal
        self.partial_reveal_prefix = partial_reveal_prefix
        #: presidio | judge | presidio+judge. Local NER is a second the request
        #: does not spend on an API call, so it goes first where it is enabled.
        self.engine_mode = engine_mode

        # The same published contacts `pii.detect` exempts. This rail used not
        # to receive them at all, so `pii.allowlist` — documented as "published
        # contacts that are exempt from masking" — held against the regex
        # recognizers and not against NER. A department address the operator
        # deliberately published could still be masked here, by a different
        # rail, for the same text.
        self.allow: list[re.Pattern[str]] = []
        for i, pat in enumerate(allowlist or []):
            try:
                self.allow.append(re.compile(pat, re.I))
            except re.error as exc:
                raise ValueError(f"pii.allowlist[{i}] is not a valid regex: {exc}") from exc

    def _allowed_spans(self, text: str) -> list[tuple[int, int]]:
        """Where the published contacts sit. Matched against the whole text for
        the same reason `PIIRail` does it: a detector slices a span to its own
        boundaries, so asking "is this detected value allowlisted" misses a
        fragment of one."""
        spans: list[tuple[int, int]] = []
        for a in self.allow:
            spans.extend((m.start(), m.end()) for m in a.finditer(text))
        return spans

    def _judge_entities(self, text: str) -> list[dict]:
        """Every window's own findings, run concurrently — a stage costs as
        much as its slowest window, not the sum, the same rule the engine
        already applies across rails.

        Each entity's `text` is still verbatim from the window, which is
        verbatim from the original — the existing span resolution below
        already re-locates a finding in the *whole* text regardless of which
        window it came from, so nothing downstream needs to know windowing
        happened at all.

        A window's own failure is not swallowed: `.result()` re-raises it,
        which fails this call the same way a single oversized call always
        did — a document partly checked is not a document checked, and the
        engine's own fail-closed handling is what should decide what happens
        next, not a silent partial scan.
        """
        windows = _windows(text, _JUDGE_WINDOW_CHARS, _JUDGE_WINDOW_OVERLAP)
        if len(windows) == 1:
            found = self.llm.judge(ENTITY_SYSTEM, windows[0], ENTITY_SCHEMA,
                                   max_tokens=_JUDGE_MAX_TOKENS)
            return list((found.get("entities") or [])[:80])

        entities: list[dict] = []
        with futures.ThreadPoolExecutor(max_workers=min(_JUDGE_MAX_WORKERS, len(windows))) as pool:
            jobs = [pool.submit(self.llm.judge, ENTITY_SYSTEM, w, ENTITY_SCHEMA,
                                max_tokens=_JUDGE_MAX_TOKENS)
                   for w in windows]
            for job in jobs:
                found = job.result()
                entities.extend((found.get("entities") or [])[:80])
        return entities

    def _replacement(self, kind: str, raw: str, owner: str) -> str:
        if self.strategy == "redact":
            return "[REDACTED]"
        if self.strategy == "replace":
            return f"<{kind}>"
        if self.strategy == "hash":
            import hashlib

            return f"<{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}>"
        if self.strategy == "partial":
            # No kind here has a non-zero ceiling today (`_REVEAL_CAP`), so this
            # currently always fully masks — same "other entity kinds ignore
            # it" rule `pii.partial_reveal_prefix` documents for pii.py's own
            # non-email/phone recognizers. Driven by config rather than a
            # hardcoded single leading character, so a caller cannot get a
            # name or an organisation partly revealed just by dialing
            # `pii.partial_reveal`/`pii.partial_reveal_prefix` up for contacts.
            tail_cap, head_cap = _REVEAL_CAP.get(kind, (0, 0))
            tail_n = min(self.partial_reveal, tail_cap, len(raw))
            head_n = min(self.partial_reveal_prefix, head_cap, len(raw) - tail_n)
            head = raw[:head_n]
            tail = raw[-tail_n:] if tail_n else ""
            middle = max(0, len(raw) - head_n - tail_n)
            return head + "*" * middle + tail
        return f"<{kind}:{self.vault.store(kind, raw, owner)}>"

    def evaluate(self, text: str, action: str, result: RailResult,
                 prior: list[Detection] | None = None, owner: str = "") -> RailResult:
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
        # of network, and it is the layer that finds the ordinary name.
        proposed: list[dict] = []
        if use_presidio:
            proposed = presidio_ner.find(text, self.kinds, self.min_conf,
                                         taken=[(d.start, d.end) for d in prior])

        items: list[dict] = []
        layer = ""
        corroborated = rejected = 0

        if use_judge and self.engine_mode == "presidio+judge":
            # Presidio proposes; the judge decides. It used to be asked *only*
            # when Presidio found nothing, which meant a Presidio hit was never
            # reviewed — and Presidio hits things that are not people. On this
            # service's own reply it read "Birth", in "Birth and death records",
            # as a person's name at 0.85 and masked it, leaving the line as
            # "<PERSON:…> and death records". Nobody saw it only because egress
            # unmasks for the token's owner; a different reader gets the mangled
            # sentence.
            #
            # `ENTITY_SYSTEM` already tells the judge not to return scheme, form
            # or programme names, or public offices — exactly the class Presidio
            # gets wrong — so it is the right arbiter for this.
            judged = self._judge_entities(text)
            judged_spans = _spans_of(judged, text)
            for item in proposed:
                start, end = int(item.get("start", -1)), int(item.get("end", -1))
                if any(start < b and a < end for a, b in judged_spans):
                    corroborated += 1
                    items.append(item)
                else:
                    rejected += 1
            # The judge's own findings are kept whatever Presidio thought: it
            # returns ORGANISATION, which Presidio never does, so dropping them
            # would lose a kind entirely.
            items.extend(judged)
            # Label what actually ran. With nothing proposed there was no
            # Presidio finding to corroborate, and calling that "presidio+judge"
            # would overstate what the cheap layer contributed.
            layer = "presidio+judge" if proposed else "judge"
        elif proposed:
            items, layer = proposed, "presidio"
        elif use_judge:
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

        # Published contacts are exempt, exactly as they are for `pii.detect`.
        # Dropped after detection rather than before, so the count still reflects
        # what was found — an exemption that made the match invisible would look
        # the same as the detector failing.
        allowed = self._allowed_spans(text)
        exempt = [s for s in spans if any(a <= s[0] and s[1] <= b for a, b in allowed)]
        spans = [s for s in spans if s not in exempt]

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
                           unverifiable_spans=dropped,
                           allowlisted=len(exempt))
        if layer == "presidio+judge":
            result.meta.update(presidio_proposed=len(proposed),
                               presidio_corroborated=corroborated,
                               presidio_rejected=rejected)

        if not kept:
            result.verdict = Verdict.PASS
            return result

        result.verdict = action_verdict(action, Verdict.MASK)
        if result.verdict is Verdict.MASK:
            out = text
            for start, end, kind, raw, _ in sorted(kept, reverse=True):
                out = out[:start] + self._replacement(kind, raw, owner) + out[end:]
            result.text_out = out
        return result
