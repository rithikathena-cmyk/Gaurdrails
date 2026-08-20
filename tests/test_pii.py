"""PII detection, masking strategies, and the vault."""

from __future__ import annotations

import pytest

from guardrails.rails.pii import PIIRail, Vault
from guardrails.types import RailResult, Verdict

ENTITIES = ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "AADHAAR"]


def make(strategy="vault-token", vault=None, custom=None, reveal=4):
    return PIIRail(
        entities=ENTITIES, confidence_threshold=0.5, mask_strategy=strategy,
        partial_reveal=reveal, custom_regex=custom or [], vault=vault or Vault(),
    )


def _result() -> RailResult:
    return RailResult(rail="pii", engine="e", verdict=Verdict.PASS)


# ── detection ──────────────────────────────────────────────────────
def test_detects_and_masks():
    res = make().evaluate(
        "SSN 796-33-9021 and card 4539578763621486", "mask", _result())
    assert res.verdict is Verdict.MASK
    assert {"US_SSN", "CREDIT_CARD"} <= {d.kind for d in res.detections}
    assert "796-33-9021" not in res.text_out
    assert "4539578763621486" not in res.text_out


def test_checksum_gate_rejects_lookalikes():
    """A 16-digit order number must not be reported as a card."""
    res = make().evaluate("order 1234567890123456 shipped", "mask", _result())
    assert not [d for d in res.detections if d.kind == "CREDIT_CARD"]


def test_clean_text_passes():
    assert make().evaluate("what documents do I need?", "mask", _result()).verdict is Verdict.PASS


def test_custom_regex_is_applied():
    res = make(custom=[r"CLM-\d{8}"]).evaluate("claim CLM-40028811", "mask", _result())
    assert [d.kind for d in res.detections] == ["CUSTOM_1"]
    assert "CLM-40028811" not in res.text_out


def test_invalid_custom_regex_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not a valid regex"):
        make(custom=["([unclosed"])


def test_overlapping_matches_resolve_to_longest():
    """A card must not also be reported as a phone number."""
    res = make().evaluate("4539 5787 6362 1486", "mask", _result())
    assert [d.kind for d in res.detections] == ["CREDIT_CARD"]


# ── strategies ─────────────────────────────────────────────────────
def test_partial_reveals_only_the_tail():
    res = make("partial").evaluate("card 4539578763621486", "mask", _result())
    assert "1486" in res.text_out
    assert "4539578763621486" not in res.text_out


def test_redact_leaves_nothing():
    res = make("redact").evaluate("card 4539578763621486", "mask", _result())
    assert "1486" not in res.text_out
    assert "[REDACTED]" in res.text_out


def test_hash_is_stable_for_the_same_value():
    a = make("hash").evaluate("card 4539578763621486", "mask", _result()).text_out
    b = make("hash").evaluate("card 4539578763621486", "mask", _result()).text_out
    assert a == b


# ── actions ────────────────────────────────────────────────────────
def test_block_action_blocks():
    res = make().evaluate("SSN 796-33-9021", "block", _result())
    assert res.verdict is Verdict.BLOCK
    assert res.text_out is None


def test_flag_action_does_not_rewrite():
    res = make().evaluate("SSN 796-33-9021", "flag", _result())
    assert res.verdict is Verdict.FLAG
    assert res.text_out is None


# ── vault ──────────────────────────────────────────────────────────
def test_vault_round_trip():
    v = Vault()
    token = v.store("US_SSN", "796-33-9021", "alice")
    assert v.reveal(token, "alice") == "796-33-9021"


def test_vault_rejects_unknown_token():
    assert Vault().reveal("deadbeefcafe", "alice") is None


def test_tokens_are_never_deterministic():
    """Locked: a stable token is a stable identifier across requests."""
    v = Vault()
    assert (v.store("US_SSN", "796-33-9021", "alice")
            != v.store("US_SSN", "796-33-9021", "alice"))


def test_vault_is_encrypted_when_cryptography_is_available():
    assert Vault().encrypted is True
