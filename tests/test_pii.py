"""PII detection, masking strategies, and the vault."""

from __future__ import annotations

import pytest

from backend.guardrails.rails.pii import PIIRail, Vault
from backend.guardrails.types import RailResult, Verdict

ENTITIES = ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "AADHAAR"]


def make(strategy="vault-token", vault=None, custom=None, reveal=4, reveal_prefix=0):
    return PIIRail(
        entities=ENTITIES, confidence_threshold=0.5, mask_strategy=strategy,
        partial_reveal=reveal, partial_reveal_prefix=reveal_prefix,
        custom_regex=custom or [], vault=vault or Vault(),
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


def test_partial_prefix_is_off_by_default():
    """A phone's ceiling allows a prefix, but the global dial defaults to 0."""
    res = make("partial").evaluate("call 555-123-4567", "mask", _result())
    assert not res.text_out.strip().startswith("call 55")


def test_partial_reveals_head_and_tail_for_phone():
    res = make("partial", reveal=2, reveal_prefix=2).evaluate(
        "call 555-123-4567", "mask", _result())
    assert "555-123-4567" not in res.text_out
    assert "55" in res.text_out.split()[1][:2]
    assert res.text_out.rstrip().endswith("67")


def test_partial_prefix_ceiling_caps_ssn_at_zero():
    """SSN has no prefix ceiling — dialing the global knob up must not help."""
    res = make("partial", reveal=0, reveal_prefix=4).evaluate(
        "SSN 796-33-9021", "mask", _result())
    assert not res.text_out.split()[-1].startswith("79")


def test_partial_email_keeps_domain_suffix_and_masks_the_rest():
    res = make("partial", reveal_prefix=2).evaluate(
        "reach me at jordan.baker@example.com", "mask", _result())
    assert "jordan.baker@example.com" not in res.text_out
    assert "@" in res.text_out
    assert res.text_out.rstrip().endswith(".com")
    masked = res.text_out.split()[-1]
    local = masked.split("@")[0]
    assert local.startswith("jo")
    assert "example" not in masked


def test_partial_email_prefix_defaults_to_fully_masked_local_part():
    """Without dialing pii.partial_reveal_prefix up, the local part stays hidden."""
    res = make("partial").evaluate(
        "reach me at jordan.baker@example.com", "mask", _result())
    masked = res.text_out.split()[-1]
    local = masked.split("@")[0]
    assert local == "*" * len("jordan.baker")


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
