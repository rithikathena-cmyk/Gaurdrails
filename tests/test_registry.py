"""Registry invariants.

The registry is the source of truth for the API, the config validator, and the
Parameters page. These tests keep it honest.
"""

from __future__ import annotations

from backend.guardrails.registry import (
    ADJUSTABLE,
    LOCK_META,
    LOCKED,
    PARAMS,
    SEVERITY_LEVELS,
    SURFACES,
    as_payload,
    control_for,
)


def test_keys_are_unique():
    keys = [p.key for p in PARAMS]
    assert len(keys) == len(set(keys))


def test_every_locked_parameter_explains_itself():
    for p in LOCKED.values():
        assert p.why.strip(), f"{p.key} is locked with no reason given"
        assert p.value.strip(), f"{p.key} is locked with no stated value"
        assert p.lock.value in LOCK_META, f"{p.key} has an unknown lock category"


def test_every_adjustable_parameter_has_a_default():
    for p in ADJUSTABLE.values():
        assert p.default is not None, f"{p.key} has no default"


def test_numeric_parameters_declare_bounds():
    for p in ADJUSTABLE.values():
        if p.type in ("float", "int"):
            assert p.minimum is not None and p.maximum is not None, f"{p.key} is unbounded"
            assert p.minimum <= p.default <= p.maximum, f"{p.key} default is outside its range"


def test_enum_defaults_are_in_their_options():
    for p in ADJUSTABLE.values():
        if p.type == "enum":
            assert p.default in p.options, f"{p.key} default is not an option"


def test_every_parameter_belongs_to_a_declared_family():
    from backend.guardrails.registry import FAMILIES

    for p in PARAMS:
        assert p.family in FAMILIES, f"{p.key} has an unknown family"


def test_every_parameter_gets_a_control():
    for p in PARAMS:
        assert control_for(p) in (
            "locked", "range", "number", "toggle", "select", "tags", "matrix", "text"
        )


def test_locked_metadata_carries_a_theme_token():
    """The frontend colours locks from these tokens rather than hardcoding."""
    for meta in LOCK_META.values():
        assert meta["token"].startswith("--")
        assert meta["glyph"] and meta["label"] and meta["blurb"]


# ── API payload ────────────────────────────────────────────────────
def test_payload_is_self_describing():
    """Everything the Parameters page needs, so it hardcodes none of it."""
    p = as_payload()
    assert p["total"] == len(PARAMS)
    assert p["total_adjustable"] + p["total_locked"] == p["total"]
    assert [s["key"] for s in p["surfaces"]] == [s["key"] for s in SURFACES]
    assert [s["key"] for s in p["severity_levels"]] == [s["key"] for s in SEVERITY_LEVELS]
    assert set(p["locks"]) == set(LOCK_META)


def test_payload_families_cover_every_parameter():
    p = as_payload()
    counted = sum(len(f["params"]) for f in p["families"])
    assert counted == len(PARAMS)


def test_severity_multipliers_are_ordered():
    """high must be stricter than medium, which must be stricter than low."""
    m = {s["key"]: s["multiplier"] for s in SEVERITY_LEVELS}
    assert m["high"] < m["medium"] < m["low"]
    assert m["off"] < 0
