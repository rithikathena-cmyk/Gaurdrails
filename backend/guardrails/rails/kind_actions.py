"""Per-kind PII overrides — the Policy Engine half of classify/decide.

`pii.kind_actions` (what happens) and `pii.kind_mask_strategy` (what a masked
value looks like when it does) are two admin-editable lists sharing one
syntax and one resolution rule, both shared by `pii.py` (regex/checksum
kinds — EMAIL_ADDRESS, PHONE_NUMBER, AADHAAR, ...) and `entities.py` (NER
kinds — PERSON, ORGANISATION, GOVERNMENT, ...), so one place decides "redact
a person's name, but vault-token an email so it can still be unmasked for
its owner" instead of each rail inventing its own version of the idea, and
instead of one global `pii.mask_strategy` forcing every kind to render the
same way.

Same `pattern => value` string convention `policy.py`'s rule sets already
use, deliberately: an admin editing this list is doing the same kind of edit
they already know how to make. The one difference is what the left side
matches against — a kind name, exactly, not a regex against the text — so
this module does not reuse `policy.parse()`.

A kind with no matching rule is not "do nothing" — `resolve()` falls back to
whatever the surface's own default already says (`pii.action.<surface>` for
actions, `pii.mask_strategy` for rendering), so an empty list reproduces
today's exact behaviour: every kind gets the one global default.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

ACTIONS = {"block", "mask", "flag", "pass"}
STRATEGIES = {"redact", "replace", "hash", "partial", "vault-token"}
_SPLIT = re.compile(r"\s*=>\s*")

#: Everything that changes what a chunk's *masked text* should look like
#: today — not just what `EntityRail`, the only detector left, would *find*.
#: `pii.kind_actions` belongs here even though it never changes a
#: classification: a chunk masked while `GOVERNMENT => pass` was in effect is
#: no longer a correct rendering of the current policy the moment an admin
#: changes that to `GOVERNMENT => mask`, and there is no cheap way to fix the
#: already-baked text without either the judge (GOVERNMENT/ORGANISATION are
#: judge-only — Presidio never proposes them, see `presidio_ner.KIND_MAP`) or
#: information this rail deliberately never stores (raw spans for masked
#: values). Treating a `pii.kind_actions` edit as "everything needs a rescan"
#: is the conservative, always-correct answer: every already-ingested
#: document falls back to a full rescan once, and reads as fresh again the
#: next time it is actually re-ingested. `pii.allowlist` is the one exception
#: genuinely left out — an allowlisted value is still cleartext in the chunk
#: either way, so a change there only ever needs the same cheap deterministic
#: re-check `EntityRail`'s own `_allowed_spans` already does on every call,
#: classification or not.
_CLASSIFICATION_KEYS = (
    "pii.entity_kinds", "pii.entity_engine", "pii.entity_confidence",
    "pii.custom_patterns", "pii.kind_actions",
    # A masked span's *rendering* is baked into the chunk too — a document
    # ingested under vault-token no longer matches today's policy the moment
    # an admin flips `pii.mask_strategy` (or a per-kind override) to redact,
    # for the identical "already-baked text can't reflect a decide-only
    # change without a rescan" reason `pii.kind_actions` is here.
    "pii.mask_strategy", "pii.kind_mask_strategy",
)


def classification_fingerprint(policy: Any) -> str:
    """A short hash of every config value that decides what an already-
    ingested chunk's masked text *should* look like under today's policy.

    Stored on a `Document` at ingest (`pii_policy_version`) and recomputed at
    retrieval time: a match means this document's chunks already reflect
    exactly what today's config would produce, so retrieval can trust what is
    already in the index instead of paying for another judge scan. A
    mismatch — a document that predates this feature, or an admin who
    changed `pii.entity_kinds` or `pii.kind_actions` since — means it cannot
    be trusted, and retrieval falls back to a full rescan of that chunk, the
    same rescan every retrieval used to pay for regardless.
    """
    values = [policy.get(k) for k in _CLASSIFICATION_KEYS]
    blob = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class KindActionError(ValueError):
    pass


def _parse(rules: list[str], valid: set[str], param: str) -> dict[str, str]:
    """`["PERSON => mask", "GOVERNMENT => pass"]` -> `{"PERSON": "mask", ...}`.

    Shared by `parse()` (actions) and `parse_strategy()` (rendering) — same
    `KIND => value` syntax, only the valid right-hand vocabulary differs.
    A later duplicate wins — the same "last one written governs" rule an
    admin already gets from editing a YAML mapping by hand.
    """
    out: dict[str, str] = {}
    for i, raw in enumerate(rules or []):
        text = str(raw).strip()
        if not text or text.startswith("#"):
            continue
        parts = _SPLIT.split(text, maxsplit=1)
        if len(parts) != 2:
            raise KindActionError(
                f"{param}[{i}]: {text!r} is not `KIND => value` — "
                "a kind alone doesn't say what to do with it"
            )
        kind, value = parts[0].strip().upper(), parts[1].strip().lower()
        if value not in valid:
            raise KindActionError(f"{param}[{i}]: {value!r} is not one of {sorted(valid)}")
        out[kind] = value
    return out


def parse(rules: list[str]) -> dict[str, str]:
    """`pii.kind_actions` — see `_parse`."""
    return _parse(rules, ACTIONS, "pii.kind_actions")


def parse_strategy(rules: list[str]) -> dict[str, str]:
    """`pii.kind_mask_strategy` — see `_parse`."""
    return _parse(rules, STRATEGIES, "pii.kind_mask_strategy")


def resolve(kind: str, overrides: dict[str, str], default: str) -> str:
    """This kind's value — its own override if one is configured, else
    `default` (the surface's action, or the global mask strategy) — the
    same default every kind used to get. Generic over `parse()`'s and
    `parse_strategy()`'s dicts alike."""
    return overrides.get(kind.upper(), default)
