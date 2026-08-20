"""Sensitive-information guardrails.

Recognizers are regex candidates followed by a **checksum gate** wherever the
identifier has one. That gate is locked in the registry, and the reason is
practical rather than theoretical: a 16-digit regex with no Luhn check fires on
order numbers, tracking codes, and timestamps. The queue fills with noise, and
somebody turns the rail off to stop it. Validation is what keeps the rail
switched on.

Reversible masking uses AES-256-GCM with a random token per occurrence — the
registry locks token determinism off, because a stable token is a stable
identifier that lets an observer correlate users without seeing the value.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from typing import Callable

from ..types import Detection, RailResult, Verdict, action_verdict


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, alt = 0, False
    for d in reversed(digits):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff(value: str) -> bool:
    """Aadhaar checksum."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 12 or digits[0] in (0, 1):
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


def iban_mod97(value: str) -> bool:
    v = re.sub(r"[\s-]", "", value).upper()
    if not 15 <= len(v) <= 34:
        return False
    rotated = v[4:] + v[:4]
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rotated)
    if not numeric.isdigit():
        return False
    return int(numeric) % 97 == 1


def ssn_plausible(value: str) -> bool:
    """Reject the ranges the SSA never issues — the classic false-positive source."""
    d = re.sub(r"\D", "", value)
    if len(d) != 9:
        return False
    area, group, serial = d[:3], d[3:5], d[5:]
    if area in ("000", "666") or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def pan_format(value: str) -> bool:
    """Indian PAN: 5 letters, 4 digits, 1 letter, with a valid holder-type char."""
    v = value.upper()
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", v)) and v[3] in "ABCFGHLJPTK"


# ---------------------------------------------------------------------------
# Recognizers
# ---------------------------------------------------------------------------
@dataclass
class Recognizer:
    entity: str
    pattern: re.Pattern[str]
    confidence: float
    check: Callable[[str], bool] | None = None
    reveal: int = 0  # trailing chars safe to show under partial masking


RECOGNIZERS: list[Recognizer] = [
    Recognizer("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), 0.98),
    Recognizer("US_SSN", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"), 0.95, ssn_plausible, 4),
    Recognizer("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), 0.99, luhn, 4),
    Recognizer("AADHAAR", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), 0.95, verhoeff, 4),
    Recognizer("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), 0.92, pan_format, 0),
    Recognizer("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 0.95, iban_mod97, 4),
    Recognizer("PHONE_NUMBER",
               re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}\b"), 0.80, None, 4),
    Recognizer("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.75),
    Recognizer("DATE_OF_BIRTH",
               re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b"), 0.70),
]


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------
class Vault:
    """Reversible mask tokens, AES-256-GCM.

    Registry: `pii.vault_encryption` (compliance-locked) and
    `pii.token_determinism` (safety-locked — random per occurrence).
    """

    def __init__(self, key: bytes | None = None) -> None:
        self._store: dict[str, bytes] = {}
        self._key = key or os.urandom(32)
        self._aead = None
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._aead = AESGCM(self._key)
        except ImportError:  # pragma: no cover - optional dependency
            self._aead = None

    @property
    def encrypted(self) -> bool:
        return self._aead is not None

    def store(self, entity: str, value: str) -> str:
        token_id = secrets.token_hex(6)
        if self._aead is not None:
            nonce = os.urandom(12)
            self._store[token_id] = nonce + self._aead.encrypt(nonce, value.encode(), None)
        else:
            self._store[token_id] = value.encode()
        return token_id

    def reveal(self, token_id: str) -> str | None:
        blob = self._store.get(token_id)
        if blob is None:
            return None
        if self._aead is not None:
            return self._aead.decrypt(blob[:12], blob[12:], None).decode()
        return blob.decode()


# ---------------------------------------------------------------------------
# Rail
# ---------------------------------------------------------------------------
class PIIRail:
    name = "pii.detect"
    engine = "regex recognizers + checksums"

    def __init__(self, entities: list[str], confidence_threshold: float,
                 mask_strategy: str, partial_reveal: int, custom_regex: list[str],
                 vault: Vault, allowlist: list[str] | None = None) -> None:
        self.entities = set(entities)
        self.min_conf = confidence_threshold
        self.strategy = mask_strategy
        self.partial_reveal = partial_reveal
        self.vault = vault
        self.custom: list[Recognizer] = []
        for i, pat in enumerate(custom_regex or []):
            try:
                self.custom.append(
                    Recognizer(f"CUSTOM_{i + 1}", re.compile(pat), 0.90)
                )
            except re.error as exc:
                raise ValueError(f"pii.custom_regex[{i}] is not a valid regex: {exc}") from exc

        # Published contacts. A department's grievance address is printed on the
        # notice board — masking it means the assistant cannot answer "who do I
        # write to", which is most of what this desk is for. A citizen's own
        # address is a different thing entirely and is not covered by this.
        self.allow: list[re.Pattern[str]] = []
        for i, pat in enumerate(allowlist or []):
            try:
                self.allow.append(re.compile(pat, re.I))
            except re.error as exc:
                raise ValueError(f"pii.allowlist[{i}] is not a valid regex: {exc}") from exc

    def _allowed_spans(self, text: str) -> list[tuple[int, int]]:
        """Where the published contacts sit in this text.

        Matched against the whole text rather than the detected value, because a
        recognizer slices a span to its own boundaries: the built-in phone
        pattern takes `800 425 1969` out of `1800 425 1969`, and an allowlist
        entry for the published helpline would never match that fragment. The
        question is not "does the detected value look allowlisted" but "does
        this detection fall inside something the operator published".
        """
        spans: list[tuple[int, int]] = []
        for a in self.allow:
            spans.extend((m.start(), m.end()) for m in a.finditer(text))
        return spans

    def _detect(self, text: str) -> list[tuple[Detection, Recognizer]]:
        found: list[tuple[Detection, Recognizer]] = []
        for rec in list(RECOGNIZERS) + self.custom:
            if rec.entity in self.entities or rec.entity.startswith("CUSTOM_"):
                if rec.confidence < self.min_conf:
                    continue
                for m in rec.pattern.finditer(text):
                    raw = m.group(0)
                    # Checksum gate — locked on. This is what keeps precision
                    # high enough that the rail stays enabled.
                    if rec.check and not rec.check(raw):
                        continue
                    found.append((
                        Detection(kind=rec.entity, value=raw, start=m.start(),
                                  end=m.end(), confidence=rec.confidence,
                                  note="checksum ok" if rec.check else ""),
                        rec,
                    ))
        # Longest match wins on overlap — a card number should not also be
        # reported as a phone number.
        found.sort(key=lambda p: (p[0].start, -(p[0].end - p[0].start)))
        kept: list[tuple[Detection, Recognizer]] = []
        last_end = -1
        for det, rec in found:
            if det.start >= last_end:
                kept.append((det, rec))
                last_end = det.end
        return kept

    def _partition(self, pairs: list[tuple[Detection, Recognizer]], text: str):
        """Split detections into what gets masked and what is exempt.

        Deliberately *after* detection, not instead of it. An allowlisted contact
        is still found, still counted, and still reported in the rail's meta — it
        is simply not rewritten. An exemption that made the match invisible would
        be indistinguishable from the recognizer failing.
        """
        allowed = self._allowed_spans(text)
        masked, exempt = [], []
        for det, rec in pairs:
            covered = any(s <= det.start and det.end <= e for s, e in allowed)
            (exempt if covered else masked).append((det, rec))
        return masked, exempt

    def _replacement(self, det: Detection, rec: Recognizer) -> str:
        raw = det.value
        if self.strategy == "redact":
            return "[REDACTED]"
        if self.strategy == "replace":
            return f"<{det.kind}>"
        if self.strategy == "hash":
            import hashlib

            return f"<{det.kind}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}>"
        if self.strategy == "partial":
            n = min(self.partial_reveal, rec.reveal, len(raw))
            return ("*" * max(0, len(raw) - n)) + (raw[-n:] if n else "")
        # vault-token (default)
        token = self.vault.store(det.kind, raw)
        n = min(self.partial_reveal, rec.reveal, len(raw))
        tail = f" …{raw[-n:]}" if n else ""
        return f"<{det.kind}:{token}{tail}>"

    def evaluate(self, text: str, action: str, result: RailResult) -> RailResult:
        found = self._detect(text)
        pairs, exempt = self._partition(found, text)
        result.unit = "count"
        result.score = float(len(pairs))
        result.threshold = 1.0
        # Every detection is reported, exempt ones included, so an operator can
        # see what the allowlist let through rather than inferring it from silence.
        result.detections = [d for d, _ in found]
        result.meta = {
            "entities_enabled": len(self.entities),
            "strategy": self.strategy,
            "vault_encrypted": self.vault.encrypted,
            "by_type": sorted({d.kind for d, _ in pairs}),
            "allowlisted": len(exempt),
            "allowlisted_values": sorted({d.value for d, _ in exempt})[:8],
        }

        if not pairs:
            # Nothing to mask. If the only matches were published contacts, that
            # is a pass with a record, not a silent nothing.
            result.verdict = Verdict.PASS
            return result

        result.verdict = action_verdict(action, Verdict.MASK)
        if result.verdict is not Verdict.MASK:
            # block / flag / pass: report the detections, rewrite nothing.
            return result

        out = text
        for det, rec in sorted(pairs, key=lambda p: p[0].start, reverse=True):
            out = out[: det.start] + self._replacement(det, rec) + out[det.end:]
        result.text_out = out
        return result
