"""Named entities — all PII detection, judge- and Presidio-driven.

Used to be paired with `pii.py`'s regex/checksum rail, which found an SSN
because an SSN has a shape and a checksum but could never find **Meera
Balan**, or **14 Anna Salai, Chennai**, because a name has no shape. `pii.py`
is gone — removed by deliberate choice, trading the checksum layer's speed,
cost, and provable correctness for judge-only detection everywhere, on every
kind. This rail is now the *only* PII detector: every kind that used to have
a fixed pattern (email, phone, national IDs, and so on) is now something the
judge is asked to recognise by description instead, alongside the kinds that
never had a pattern to begin with.

This rail is built so it costs as little as possible:

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
                        "description": "PERSON | ADDRESS | ORGANISATION | GOVERNMENT | "
                                       "LOCATION | EMAIL_ADDRESS | PHONE_NUMBER | US_SSN | "
                                       "CREDIT_CARD | AADHAAR | PAN | IBAN | VEHICLE_PLATE | "
                                       "PASSPORT_NUMBER | NATIONAL_ID | IP_ADDRESS | "
                                       "DATE_OF_BIRTH | OTHER_PII",
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

#: The base instructions, shared by every deployment. `EntityRail.__init__`
#: appends a deployment's own `pii.custom_patterns` (if any) as one more
#: bullet before wrapping the whole thing with `judge_prompt()` — this stays
#: a plain string, not a pre-wrapped `judge_prompt()` call, specifically so
#: that append can happen per instance instead of once at import time.
ENTITY_INSTRUCTIONS = """\
List every personal identifier in the text. You are extracting, not judging: an \
identifier is reported whether or not its presence is a problem. This is the only \
detector in the system — there is no separate regex or checksum layer behind you, so \
a real identifier missed here is a real identifier missed, not merely a name a \
pattern-matcher could not have found anyway. Recognise a kind by its description below, \
not by requiring a rigid character-for-character shape: "796-33-9021", "796 33 9021", \
and "SSN 796339021" are the same US_SSN either way.

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
- EMAIL_ADDRESS: any address of the form local-part@domain, however unusual the \
local part or domain look
- PHONE_NUMBER: a telephone number in any national format — an international number \
with a country code, a US-style 3-3-4 grouping, or an Indian mobile (10 digits, \
starting 6-9) or landline (an STD code starting 0, 2-4 digits, then a 3-4 then 4 digit \
subscriber number, possibly followed by one or more comma- or slash-separated \
extensions sharing the same STD code) — with or without spaces, dashes, or parentheses
- US_SSN: a US Social Security Number, three digits, two digits, four digits, however \
separated — but not a number in a range the SSA never issues: an area of 000, 666, or \
900-999, a group of 00, or a serial of 0000 is not a real SSN and should not be \
returned
- CREDIT_CARD: a 13-19 digit payment card number, however grouped or spaced, whose \
digits satisfy the Luhn check (double every second digit from the right, subtract 9 \
from anything over 9, the total is a multiple of 10) — skip a digit run that fails \
this, it is not a real card number
- AADHAAR: a 12-digit Indian Aadhaar number, however grouped (commonly 4-4-4), that \
does not begin with 0 or 1
- PAN: an Indian Permanent Account Number — 5 letters, 4 digits, 1 letter, where the \
fourth character is one of A, B, C, F, G, H, L, J, P, T, or K (that letter encodes the \
holder type; anything else is not a real PAN)
- IBAN: an International Bank Account Number — 2 letters (country code), 2 digits (a \
real IBAN check digit, not arbitrary), then up to 30 further letters or digits
- VEHICLE_PLATE: a vehicle registration/number plate, any country's format — a short \
code mixing letters and digits, typically 4-10 characters once spaces or dashes are \
removed, such as "TN-07-AB-1234", "AB12 CDE", or "1ABC234"
- PASSPORT_NUMBER: a passport number — typically a letter followed by 6-8 digits, the \
shape India, the UK, and the US all share
- NATIONAL_ID: a national identity number belonging to a country other than the ones \
above — for example a UK National Insurance Number (2 letters, 6 digits, 1 letter from \
A-D, excluding D/F/I/Q/U/V as either of the first two letters and excluding the \
prefixes BG/GB/KN/NK/NT/TN/ZZ)
- IP_ADDRESS: an IPv4 address, four dot-separated numbers 0-255
- DATE_OF_BIRTH: a date explicitly given as someone's date of birth, in any common \
date format
- OTHER_PII: a personal identifier that ties to one specific individual but fits \
none of the kinds above and has no fixed shape — an employee or membership number, a \
case or account reference, a medical or disability detail, a relative's name given \
only to identify someone else, or similar. Only when it identifies a specific person, \
not a category of people.

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
- anything matching one of the named kinds above — classify it as that kind, not \
OTHER_PII, even if the shape looks unusual
- a digit run that fails the check named for its kind above (Luhn for CREDIT_CARD, \
the SSA range rule for US_SSN, the PAN holder-type letter, and so on) — it is not \
that kind of identifier, and not OTHER_PII either unless something else about it \
independently identifies a specific person
- anything already written as a masked token

Copy each `text` verbatim from the input, exactly as it appears, including its \
capitalisation and any surrounding punctuation that belongs to it — the span is looked \
up in the original text and discarded if it cannot be found. Do not correct spelling, \
expand an initial, or normalise a form. Return an empty list rather than guessing."""

#: Every deployment's default prompt — no `pii.custom_patterns` configured.
#: `EntityRail.__init__` builds its own `self.system_prompt` instead of this
#: constant whenever custom patterns are configured; this is the fallback.
ENTITY_SYSTEM = judge_prompt(ENTITY_INSTRUCTIONS, calibrate=False)

# Cheap structural gate — now three independent kinds of evidence, not just
# one: a capital letter mid-sentence (a name could be present), a run of 3+
# digits (a phone/SSN/card/ID/date/plate could be present), or an "@" (an
# email could be present). Before this rail absorbed pii.py's kinds, a
# capital letter was the only thing worth gating on; a message that is
# nothing but a phone number or an email address has neither, and the old,
# name-only gate would have skipped the judge call entirely and missed it.
_CANDIDATE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b|\d{3,}|@", re.M)
_ALREADY_MASKED = re.compile(r"<[A-Z_0-9]+:[0-9a-f]{12}(?:\s…[^>]*)?>")

#: The non-name half of `_CANDIDATE` — a digit run or an "@". Presidio only
#: ever proposes PERSON/ADDRESS (`presidio_ner.KIND_MAP`), so "Presidio found
#: nothing" cannot stand in for "there is nothing to find" any more: it says
#: nothing about whether an SSN, a phone number, or an email is sitting in
#: this exact text. Used below to keep the retrieval judge-skip optimisation
#: scoped to what it was always actually testing — no capitalised name-like
#: candidate — instead of silently also skipping every checksummed kind that
#: used to be caught deterministically, regardless of what Presidio saw.
_SHAPED_CANDIDATE = re.compile(r"\d{3,}|@")

#: Every kind this rail recognises. Most of these used to be pii.py's regex
#: recognizers — EMAIL_ADDRESS through DATE_OF_BIRTH — with no fixed shape
#: to match against any more; they are judge-only now, the same as
#: OTHER_PII always was. `presidio_ner.KIND_MAP` only ever proposes PERSON
#: and ADDRESS (see that module), so `find()` naturally never returns any of
#: these regardless of what's enabled here — no separate judge-only set is
#: needed to keep Presidio from guessing at a checksummed kind it was never
#: taught to recognise. In "presidio+judge" mode (the default) the judge call
#: still runs every time and is what actually finds them; a deployment that
#: sets `pii.entity_engine=presidio` (no judge at all) will not detect any
#: kind in this set below — an expected consequence of choosing that mode,
#: not a bug.
KINDS = {
    "PERSON", "ADDRESS", "ORGANISATION", "GOVERNMENT", "LOCATION", "OTHER_PII",
    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "AADHAAR", "PAN",
    "IBAN", "VEHICLE_PLATE", "PASSPORT_NUMBER", "NATIONAL_ID", "IP_ADDRESS",
    "DATE_OF_BIRTH",
}

#: (trailing, leading) characters safe to reveal under partial masking, per
#: kind — the same ceilings the old `pii.py` `Recognizer.reveal`/
#: `reveal_prefix` values used, carried over so removing the regex layer
#: does not also silently change what "partial" reveals for a phone number
#: or a credit card. EMAIL_ADDRESS is not listed here — it keeps its own
#: shape-preserving branch in `_replacement()` below, unchanged from before.
#: PERSON/ORGANISATION/GOVERNMENT/LOCATION/OTHER_PII stay at zero: no prefix
#: or suffix of a name or an organisation is established here as safe to
#: leave visible.
_REVEAL_CAP: dict[str, tuple[int, int]] = {
    "ADDRESS": (4, 0),
    "PHONE_NUMBER": (4, 2),
    "US_SSN": (4, 0),
    "CREDIT_CARD": (4, 0),
    "AADHAAR": (4, 0),
    "PAN": (4, 0),
    "IBAN": (4, 0),
    "VEHICLE_PLATE": (4, 0),
    "PASSPORT_NUMBER": (4, 0),
    "NATIONAL_ID": (4, 0),
}

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
    """The only PII detector — model-backed, feeding the same vault the old
    regex/checksum rail used to."""

    name = "pii.entities"
    engine = "claude judge · named entities"

    def __init__(self, llm, vault, confidence_threshold: float, mask_strategy: str,
                 kinds: list[str] | None = None, engine_mode: str = "presidio+judge",
                 allowlist: list[str] | None = None,
                 partial_reveal: int = 0, partial_reveal_prefix: int = 0,
                 kind_actions: dict[str, str] | None = None,
                 kind_mask_strategy: dict[str, str] | None = None,
                 custom_patterns: list[str] | None = None) -> None:
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
        # Same knobs `pii.py` used to read, same reason: a caller may
        # configure a generous reveal count meant for a phone number's last
        # four digits, but a name or an organisation has no per-kind ceiling
        # raising it above zero — see `_REVEAL_CAP`.
        self.partial_reveal = partial_reveal
        self.partial_reveal_prefix = partial_reveal_prefix
        #: presidio | judge | presidio+judge. Local NER is a second the request
        #: does not spend on an API call, so it goes first where it is enabled
        #: — but it only ever proposes PERSON/ADDRESS (`presidio_ner.KIND_MAP`);
        #: every other kind in `KINDS` is judge-only regardless of this setting.
        self.engine_mode = engine_mode

        # The same published contacts the old regex rail exempted.
        self.allow: list[re.Pattern[str]] = []
        for i, pat in enumerate(allowlist or []):
            try:
                self.allow.append(re.compile(pat, re.I))
            except re.error as exc:
                raise ValueError(f"pii.allowlist[{i}] is not a valid regex: {exc}") from exc

        # `pii.custom_patterns` — the judge-prompt successor to the old
        # `pii.custom_regex`. A regex could be compiled and matched exactly;
        # a judge cannot offer that guarantee, so this is deliberately
        # best-effort: each configured pattern is shown to the judge as one
        # more thing to recognise by description, the same way every other
        # kind above is, not executed against the text. `self.system_prompt`
        # is built once per instance rather than read from the module-level
        # `ENTITY_SYSTEM` constant precisely so a deployment's own patterns
        # can be folded in without touching the shared default.
        if custom_patterns:
            custom_block = (
                "This deployment also defines its own patterns to look for, on top of "
                "every kind above — tag a match OTHER_PII unless one of the named kinds "
                "above clearly fits better:\n"
                + "\n".join(f"- {p}" for p in custom_patterns)
            )
            self.system_prompt = judge_prompt(ENTITY_INSTRUCTIONS, custom_block, calibrate=False)
        else:
            self.system_prompt = ENTITY_SYSTEM

    def _allowed_spans(self, text: str) -> list[tuple[int, int]]:
        """Where the published contacts sit. Matched against the whole text
        because a detector slices a span to its own boundaries, so asking
        "is this detected value allowlisted" misses a fragment of one."""
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
                return self.llm.judge(self.system_prompt, window, ENTITY_SCHEMA,
                                      max_tokens=_JUDGE_MAX_TOKENS)
            except LLMError:
                continue
        if depth >= _JUDGE_MAX_SPLIT_DEPTH or len(window) < _JUDGE_MIN_SPLIT_CHARS:
            return self.llm.judge(self.system_prompt, window, ENTITY_SCHEMA,
                                  max_tokens=_JUDGE_MAX_TOKENS)  # let the final LLMError raise
        mid = len(window) // 2
        space = window.find(" ", mid)
        if space < 0:
            space = mid
        left, right = self._judge_one(window[:space], depth=depth + 1), \
            self._judge_one(window[space:], depth=depth + 1)
        return {"entities": [*(left.get("entities") or []), *(right.get("entities") or [])]}

    def _replacement(self, kind: str, raw: str, owner: str, *,
                     force_strategy: str | None = None) -> str:
        # `pii.kind_mask_strategy` — a kind not listed renders with the
        # global `pii.mask_strategy`, unchanged from before this existed.
        # `force_strategy` skips both: the caller (today, only a REDACT
        # action's capability-layer call) is asking for one specific
        # strategy regardless of what this kind or this deployment is
        # otherwise configured to render as — see `evaluate()`'s docstring.
        strategy = force_strategy or resolve_kind_action(kind, self.kind_strategy, self.strategy)
        if strategy == "redact":
            return "[REDACTED]"
        if strategy == "replace":
            return f"<{kind}>"
        if strategy == "hash":
            import hashlib

            return f"<{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}>"
        if strategy == "partial":
            # EMAIL_ADDRESS keeps its shape — `jo***@***.com`, never
            # `**********om` — because the local part and the domain are
            # different things to an operator deciding what is safe to leave
            # visible; a flat head/tail slice across the whole string would
            # ignore that and either leak past the `@` or hide the domain
            # along with everything else. Carried over unchanged from the
            # old regex rail's own `_partial_mask`.
            if kind == "EMAIL_ADDRESS" and "@" in raw:
                head_n = min(self.partial_reveal_prefix, 2, len(raw))
                local, _, domain = raw.partition("@")
                head = local[:head_n]
                local_masked = head + "*" * max(0, len(local) - head_n)
                if "." in domain:
                    name, _, tld = domain.rpartition(".")
                    domain_masked = ("*" * max(1, len(name))) + "." + tld
                else:
                    domain_masked = "*" * len(domain)
                return f"{local_masked}@{domain_masked}"

            # Every other kind: only what `_REVEAL_CAP` names gets anything
            # back — driven by config rather than a hardcoded single leading
            # character, so a caller cannot get a name or an organisation
            # partly revealed just by dialing `pii.partial_reveal`/
            # `pii.partial_reveal_prefix` up for contacts.
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
                 surface: str = "", force_strategy: str | None = None) -> RailResult:
        """`prior` is what the deterministic rail already claimed, so NER cannot
        return a worse guess over the same characters.

        `surface` only changes one thing — see the `retrieval` branch below,
        where a `presidio+judge` scan with nothing for Presidio to propose is
        skipped rather than run anyway.

        `force_strategy`, when given, overrides `pii.mask_strategy` and
        `pii.kind_mask_strategy` for every kind masked by this one call —
        the deterministic rail pipeline (`engine.py`) never passes it, so its
        every-request masking is unaffected. It exists for
        `agents/capabilities.py`'s `REDACT` action: `REDACT` and `MASK` used
        to reach this rail identically (`action="mask"` either way, per
        `agent_verdict` this parameter doesn't touch), which meant `REDACT`
        rendered through whatever `pii.mask_strategy` happened to be
        configured — `partial`, say — leaving part of the value directly
        readable in text a caller had just been told was "removed, not
        recoverable." `force_strategy="redact"` is what makes that claim
        true again for the one action that makes it.
        """
        prior = prior or []
        result.unit = "count"
        result.threshold = 1.0
        result.meta = {"kinds_enabled": sorted(self.kinds),
                       "strategy": force_strategy or self.strategy}

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
        #
        # `not _SHAPED_CANDIDATE.search(probe)` is the part that changed when
        # the regex/checksum rail was removed: Presidio proposing nothing
        # used to be a safe stand-in for "nothing to find" because a
        # checksummed kind (SSN, email, ...) would already have been caught
        # deterministically regardless of this skip. That backstop is gone —
        # skipping purely on an empty Presidio proposal would silently never
        # scan a chunk that is *only* an email or a phone number, no name in
        # sight. A digit run or an "@" forces the judge to run even with
        # nothing proposed; the skip is still safe for the common case this
        # was written for, prose with no name-like or digit/@-shaped content.
        skip_retrieval_judge = (surface == "retrieval" and not proposed
                                and self.engine_mode == "presidio+judge"
                                and not _SHAPED_CANDIDATE.search(probe))
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

        # Longest match wins on overlap first — the same rule the regex
        # recognizers use, and resolved over *every* candidate, exempt ones
        # included, so an allowlisted span still knocks out a shorter
        # overlapping guess the same way it would if it were going to be
        # masked. Partitioning before resolving would let an about-to-be-
        # discarded exempt span silently change which overlapping candidate
        # survives.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        all_spans: list[tuple[int, int, str, str, float]] = []
        last_end = -1
        for span in spans:
            if span[0] >= last_end:
                all_spans.append(span)
                last_end = span[1]

        # Published contacts are exempt, exactly as the old regex rail made
        # them. Partitioned after resolution, not instead of detection —
        # every exempt span still appears in `result.detections` below, so
        # an operator can see what the allowlist let through rather than
        # inferring it from silence; only the *masking* below skips them.
        allowed = self._allowed_spans(text)
        exempt = [s for s in all_spans if any(a <= s[0] and s[1] <= b for a, b in allowed)]
        kept = [s for s in all_spans if s not in exempt]

        result.score = float(len(kept))
        result.detections = [
            Detection(kind=kind, value=raw, start=start, end=end, confidence=conf,
                      note="named entity")
            for start, end, kind, raw, conf in all_spans
        ]
        result.meta.update(layer=layer or "none",
                           by_type=sorted({k for _, _, k, _, _ in kept}),
                           unverifiable_spans=dropped,
                           allowlisted=len(exempt),
                           # Carried over from the old regex rail's own meta —
                           # an operator auditing what the allowlist let
                           # through, capped the same way, [:8].
                           allowlisted_values=sorted({s[3] for s in exempt})[:8])
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
                out = out[:start] + self._replacement(
                    kind, raw, owner, force_strategy=force_strategy) + out[end:]
        if out != text:
            result.text_out = out
        return result
