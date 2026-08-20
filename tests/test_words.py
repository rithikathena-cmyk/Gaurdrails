"""Normalization and the word automaton."""

from __future__ import annotations

from guardrails.rails.normalize import normalize
from guardrails.rails.words import Automaton, WordRail
from guardrails.types import RailResult, Verdict


def _result() -> RailResult:
    return RailResult(rail="w", engine="e", verdict=Verdict.PASS)


# ── normalization (locked on) ──────────────────────────────────────
def test_homoglyph_fold():
    """Cyrillic lookalikes must fold, or the whole word filter is walkable."""
    out, changed = normalize("іdiоt")   # Cyrillic і and о
    assert out == "idiot"
    assert changed > 0


def test_zero_width_removal():
    assert normalize("id​iot")[0] == "idiot"


def test_nfkc_folds_fullwidth():
    assert normalize("ｉｄｉｏｔ")[0] == "idiot"


def test_whitespace_collapse():
    assert normalize("a  b")[0] == "a b"


# ── Aho–Corasick ───────────────────────────────────────────────────
def test_automaton_finds_overlapping_patterns():
    a = Automaton()
    for p in ("he", "she", "his", "hers"):
        a.add(p)
    a.build()
    assert {"he", "she", "hers"} <= {p for _, _, p in a.search("ushers")}


def test_pattern_length_cap_is_enforced():
    a = Automaton()
    a.add("x" * 300)          # over words.max_pattern_length (locked at 256)
    a.build()
    assert a.search("x" * 300) == []


# ── the rail ───────────────────────────────────────────────────────
def test_masks_after_normalizing():
    rail = WordRail(["idiot"], [])
    res = rail.evaluate("you are an іdiоt", "mask", _result())
    assert res.verdict is Verdict.MASK
    assert res.score == 1


def test_allowlist_only_exempts_never_adds():
    """Locked precedence: blocklist runs first, allowlist can only subtract."""
    blocked = WordRail(["vermin"], []).evaluate("vermin control", "mask", _result())
    assert blocked.verdict is Verdict.MASK

    exempt = WordRail(["vermin"], ["vermin control"]).evaluate("vermin control", "mask", _result())
    assert exempt.verdict is Verdict.PASS
    assert exempt.meta["exempted_by_allowlist"] == 1


def test_allowlist_cannot_introduce_a_term():
    """An allowlist entry for a term the blocklist doesn't know is a no-op."""
    rail = WordRail(["idiot"], ["something-else"])
    assert rail.evaluate("perfectly fine text", "mask", _result()).verdict is Verdict.PASS


def test_word_boundary_mode_avoids_substring_hits():
    rail = WordRail(["ass"], [], match_mode="word")
    assert rail.evaluate("please assess my application", "mask", _result()).verdict is Verdict.PASS


def test_substring_mode_does_match_inside_words():
    rail = WordRail(["ass"], [], match_mode="substring")
    assert rail.evaluate("please assess my application", "mask", _result()).verdict is Verdict.MASK


def test_block_action_is_honoured():
    rail = WordRail(["idiot"], [])
    assert rail.evaluate("idiot", "block", _result()).verdict is Verdict.BLOCK
