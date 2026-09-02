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

from ..llm import LLMError
from ..prompts import judge_prompt
from . import presidio_ner
from ..types import Detection, RailResult, Verdict, action_verdict, precedence
from .kind_actions import resolve as resolve_kind_action

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
                        "description": "PERSON | ADDRESS | ORGANISATION | GOVERNMENT | LOCATION",
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
- GOVERNMENT: a public office, department, ministry, court, municipal body, or a \
cooperative society, union, federation, institute, academy, board, or corporation \
operating under a Registrar of Cooperative Societies, a government department, or \
similar statutory oversight — these are public-sector bodies, not private employers, \
whatever legal form their name takes ("... Union", "... Federation", "Institute of \
...", "... Cooperative Management", "... Apex Bank", "District ... Cooperative Bank", \
and so on). Classify it as GOVERNMENT, do not omit it and do not call it ORGANISATION \
or PERSON.
- LOCATION: a specific place that would identify a household, such as a village \
together with a house number

Do not return:
- job titles with no name attached
- cities, districts, states, or nations on their own — "Tamil Nadu", "Kerala", \
"Chennai", "India" name a place, never a person, even standing alone at the start of \
a sentence or immediately before an organisation's name ("Tamil Nadu State Apex \
Cooperative Bank" is one GOVERNMENT body; "Tamil Nadu" inside it is not a separate \
PERSON). If a phrase names a state, union territory, or nation, it is excluded by \
this rule regardless of capitalisation or position — it is never PERSON, and it is \
LOCATION only when it also identifies a specific household address (see LOCATION \
above), which a bare state or country name never does on its own.
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

KINDS = {"PERSON", "ADDRESS", "ORGANISATION", "GOVERNMENT", "LOCATION"}

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
#
# Kept at 6,000, deliberately, even though GOVERNMENT joining the returned
# kinds means more entities per window than before this session: a *smaller*
# window was tried first and made the actual, measured failure worse, not
# better — see `_JUDGE_MAX_TOKENS`'s note. More windows means more sequential
# batches through `_JUDGE_MAX_WORKERS`' 8 concurrent slots, and it was the
# *whole-document* wall time against `ingest.latency_budget_ms` that was
# failing, live, not entity density against `_JUDGE_MAX_TOKENS` — fewer,
# larger windows is the right direction for that, not the wrong one.
_JUDGE_WINDOW_CHARS = 6000
#: So an entity sitting across a window boundary still appears whole in at
#: least one window rather than being split in both and found in neither.
_JUDGE_WINDOW_OVERLAP = 200
#: `max_tokens` is a shared budget for adaptive thinking *and* the JSON reply
#: (`_tuning` turns on `thinking: adaptive` for this call) — `_text_of` drops
#: the thinking block before parsing, so a window dense enough to need real
#: reasoning about the exclusion rules in `ENTITY_SYSTEM` (which office names
#: are public-sector, which "...Union"/"...Federation" is a cooperative body,
#: and so on) can spend enough of a tight budget on that reasoning to leave
#: the entities array truncated — invalid JSON, not oversized JSON. Observed
#: in production on a single 6,000-char window with a dense name list, at
#: 4096; raised to 8192 with headroom for both. Raised again here, to give
#: GOVERNMENT-kind entities — now returned instead of silently dropped —
#: headroom too, without needing a smaller (and, measured, slower overall)
#: window to compensate.
_JUDGE_MAX_TOKENS = 12288
_JUDGE_MAX_WORKERS = 8
#: Same-size attempts `_judge_one` makes before it gives up and splits.
#: Deliberately small: a real ingest run against the RCS Citizen Charter
#: showed two *independent* failure shapes on the same document —
#: occasional truncated JSON from one window (which a retry or two clears),
#: and the *whole scan* running past `ingest.latency_budget_ms` when the
#: rare window needs more than one attempt. Retrying aggressively fixes the
#: first at the direct expense of the second: five attempts per failing
#: window, measured, pushed total wall time past the 60s ingest budget more
#: often than it saved documents from a bad response. Two attempts is the
#: cheaper, still-real improvement over one; `ingest.latency_budget_ms`
#: itself (see registry.py) carries the rest of the headroom.
_JUDGE_ONE_RETRIES = 2
#: `_judge_one`'s split-on-persistent-failure floor: three halvings of a
#: 6,000-char window bottoms out around 750 — small enough that no realistic
#: entity density overflows `_JUDGE_MAX_TOKENS` for it, without splitting a
#: pathological window down to single sentences for no further benefit. This
#: path is the rare last resort, not the common case — see `_JUDGE_ONE_RETRIES`.
_JUDGE_MAX_SPLIT_DEPTH = 3
_JUDGE_MIN_SPLIT_CHARS = 750


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
                 partial_reveal: int = 0, partial_reveal_prefix: int = 0,
                 kind_actions: dict[str, str] | None = None,
                 kind_mask_strategy: dict[str, str] | None = None) -> None:
        self.llm = llm
        self.vault = vault
        self.min_conf = confidence_threshold
        self.strategy = mask_strategy
        self.kinds = {k.upper() for k in (kinds or KINDS)} & KINDS
        #: `pii.kind_actions` — see `kind_actions.py`. A kind with no entry
        #: here gets whatever action the surface was already going to apply.
        self.kind_actions = dict(kind_actions or {})
        #: `pii.kind_mask_strategy` — a kind with no entry here renders with
        #: the global `pii.mask_strategy`.
        self.kind_strategy = dict(kind_mask_strategy or {})
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
            found = self._judge_one(windows[0])
            return list((found.get("entities") or [])[:80])

        entities: list[dict] = []
        with futures.ThreadPoolExecutor(max_workers=min(_JUDGE_MAX_WORKERS, len(windows))) as pool:
            jobs = [pool.submit(self._judge_one, w) for w in windows]
            for job in jobs:
                found = job.result()
                entities.extend((found.get("entities") or [])[:80])
        return entities

    def _judge_one(self, window: str, *, depth: int = 0) -> dict:
        """One window's judge call, retried same-size up to
        `_JUDGE_ONE_RETRIES` times, and — only once every same-size attempt
        has failed — split in half and judged as two smaller calls instead.

        Both failure shapes below come back identically from `self.llm.judge`
        — `LLMError("judge returned non-JSON: ...")`, truncated mid-entity —
        which is why both get the same two-stage answer rather than being
        told apart up front:

        A single window judged alone, even at this same size, succeeds
        reliably in isolation and under this rail's own internal 8-way
        concurrency (`_JUDGE_MAX_WORKERS`) — measured directly, repeatedly,
        against the real RCS Citizen Charter's own opening section. It only
        started failing — close to half the time, on a real ingest run —
        once it ran concurrently with `content.safety` and `prompt_attack`'s
        *own* judge calls, which `evaluate()` submits to the same pool for
        `Surface.INGEST`. Several same-size retries is the honest fix for
        that: whatever the contention actually is, most individual attempts
        still succeed, so a handful of independent ones essentially never all
        fail together.

        The split is the fallback for the other, rarer shape: a window
        genuinely too dense for any single reply to fit in budget — observed
        once, live, on a doubled `_JUDGE_MAX_TOKENS`. Splitting at a
        whitespace boundary near the midpoint keeps every entity intact in
        at least one half, the same reason `_JUDGE_WINDOW_OVERLAP` exists for
        the outer windowing pass. Bounded to `_JUDGE_MAX_SPLIT_DEPTH`
        halvings and a `_JUDGE_MIN_SPLIT_CHARS` floor so a window that is
        dense at *every* scale still fails outright eventually — loudly, as
        an `LLMError` the caller's own fail-closed handling already knows
        what to do with, not a silent partial scan.
        """
        for _ in range(_JUDGE_ONE_RETRIES):
            try:
                return self.llm.judge(ENTITY_SYSTEM, window, ENTITY_SCHEMA,
                                      max_tokens=_JUDGE_MAX_TOKENS)
            except LLMError:
                continue
        if depth >= _JUDGE_MAX_SPLIT_DEPTH or len(window) < _JUDGE_MIN_SPLIT_CHARS:
            return self.llm.judge(ENTITY_SYSTEM, window, ENTITY_SCHEMA,
                                  max_tokens=_JUDGE_MAX_TOKENS)  # let the final LLMError raise
        mid = len(window) // 2
        space = window.find(" ", mid)
        if space < 0:
            space = mid
        left, right = self._judge_one(window[:space], depth=depth + 1), \
            self._judge_one(window[space:], depth=depth + 1)
        return {"entities": [*(left.get("entities") or []), *(right.get("entities") or [])]}

    def _replacement(self, kind: str, raw: str, owner: str) -> str:
        # `pii.kind_mask_strategy` — a kind not listed renders with the
        # global `pii.mask_strategy`, unchanged from before this existed.
        strategy = resolve_kind_action(kind, self.kind_strategy, self.strategy)
        if strategy == "redact":
            return "[REDACTED]"
        if strategy == "replace":
            return f"<{kind}>"
        if strategy == "hash":
            import hashlib

            return f"<{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}>"
        if strategy == "partial":
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
                 prior: list[Detection] | None = None, owner: str = "",
                 surface: str = "") -> RailResult:
        """`prior` is what the deterministic rail already claimed, so NER cannot
        return a worse guess over the same characters.

        `surface` only changes one thing — see the `retrieval` branch below,
        where a `presidio+judge` scan with nothing for Presidio to propose is
        skipped rather than run anyway."""
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

        # Retrieved text is not user input: it is the corpus's own document,
        # already screened once at ingest, run back through the same rail on
        # every question that retrieves it. Asking the judge to re-adjudicate
        # a passage Presidio didn't even flag is where the cost piles up — a
        # citizen charter is thick with real capitalised names, so the gate
        # above almost never skips it, and every one of those calls used to
        # pay for a full multi-window judge scan of all six retrieved chunks
        # regardless (measured live: 36s, well past the 20s rail budget).
        # `user.prompt` keeps the aggressive, always-ask-the-judge behaviour
        # below — a citizen's own message is exactly where a missed name
        # matters most, and it is never six chunks long.
        skip_retrieval_judge = (surface == "retrieval" and not proposed
                                and self.engine_mode == "presidio+judge")
        if use_judge and skip_retrieval_judge:
            layer = "presidio"
            result.meta["retrieval_judge_skipped"] = "no presidio candidate"
        elif use_judge and self.engine_mode == "presidio+judge":
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
            # Pre-existing bug, found while chasing an unrelated ingest
            # failure: this used to be one raw `self.llm.judge(...)` call
            # over the *whole* `text` with no window and the default
            # `max_tokens` (2048) — fine for a prompt or a reply, silently
            # guaranteed to truncate on a real document, since nothing here
            # ever sized the call to the input. `judge`-only mode is not a
            # rare corner: `reseed_builtin_rails()` forces it for exactly a
            # real ~150,000-character document, specifically to avoid a slow
            # Presidio cold-load blocking startup — the single largest text
            # this rail is ever asked to classify was the one case taking
            # the path with no size protection at all. `_judge_entities`
            # already windows, retries, and splits for `presidio+judge`;
            # reusing it here costs nothing when `text` is short (one window
            # is the whole text, unchanged) and fixes it when it is not.
            items = self._judge_entities(text)
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

        # Each kind resolves its own action first — GOVERNMENT can stay ALLOW
        # while PERSON in the same sentence still gets redacted. Precedence
        # still applies across the mix: any kind resolving to `block` blocks
        # the whole result, same rule `pii.py` and the engine both already
        # enforce across a set of findings.
        resolved = [
            (start, end, kind, raw, conf,
             action_verdict(resolve_kind_action(kind, self.kind_actions, action), Verdict.MASK))
            for start, end, kind, raw, conf in kept
        ]
        result.verdict = precedence([v for *_, v in resolved])
        if result.verdict is Verdict.BLOCK:
            return result

        out = text
        for start, end, kind, raw, _, verdict in sorted(resolved, reverse=True):
            if verdict is Verdict.MASK:
                out = out[:start] + self._replacement(kind, raw, owner) + out[end:]
        if out != text:
            result.text_out = out
        return result
