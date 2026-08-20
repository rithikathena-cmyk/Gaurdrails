"""Normalization — the rail that is never optional.

`words.normalization` is locked as a safety invariant. Homoglyph substitution
(Cyrillic а for Latin a, fullwidth forms, zero-width joiners) is the single most
common way a lexical filter gets walked past, so the fold runs on every input
with no config path to turn it off.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that render like ASCII but aren't. Not exhaustive — this covers the
# Cyrillic/Greek lookalikes and the invisible characters that show up in real
# evasion attempts.
_HOMOGLYPHS = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ј": "j", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    # Greek
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "Α": "A",
    "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Misc lookalikes
    "ı": "i", "ł": "l", "ø": "o", "ǀ": "l", "⁄": "/",
}

# Zero-width and format characters used to split words past a matcher.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")
_WHITESPACE = re.compile(r"[ \t  - 　]+")


def normalize(text: str) -> tuple[str, int]:
    """Return (normalized_text, characters_changed).

    NFKC first (folds fullwidth, ligatures, and compatibility forms), then
    invisible-character removal, then the homoglyph table, then whitespace
    collapse. Order matters: NFKC resolves the easy cases so the homoglyph
    table only has to cover what NFKC leaves behind.

    Offsets are *not* preserved — this output is used for matching decisions.
    Masking is applied to the original text so the user sees their own input.
    """
    original = text
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE.sub("", out)
    out = "".join(_HOMOGLYPHS.get(ch, ch) for ch in out)
    out = _WHITESPACE.sub(" ", out)

    changed = sum(1 for a, b in zip(original, out) if a != b) + abs(len(original) - len(out))
    return out, changed
