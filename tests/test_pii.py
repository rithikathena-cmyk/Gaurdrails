"""PII detection, masking strategies, and the vault.

No deterministic regex/checksum rail exists any more — `EntityRail` is the
only detector, and every kind is judge-only. These tests use a scripted
judge (`_StubJudge`) that reports exactly what the test tells it to, the
same way a real judge call would, rather than exercising a compiled pattern.
Detection itself (does a real judge correctly recognise a given kind) is a
live-model concern this file cannot cover deterministically; what stays
testable here is everything downstream of a finding — masking strategies,
overlap resolution, actions, and the vault.
"""

from __future__ import annotations

from backend.guardrails.rails.entities import EntityRail
from backend.guardrails.rails.vault import Vault
from backend.guardrails.types import RailResult, Verdict

ENTITIES = ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "AADHAAR"]


class _StubJudge:
    """Reports exactly the findings the test configured — the fixed-pattern
    layer's replacement for "the regex matched", now that recognising a kind
    is the judge's job, not a compiled pattern's."""

    def __init__(self, entities):
        self.entities = list(entities)

    def judge(self, system, user, schema, **kwargs):
        return {"entities": list(self.entities)}


def make(strategy="vault-token", vault=None, reveal=4, reveal_prefix=0,
        entities_found=(), custom_patterns=None):
    return EntityRail(
        _StubJudge(entities_found), vault or Vault(),
        confidence_threshold=0.5, mask_strategy=strategy,
        kinds=ENTITIES, engine_mode="judge",
        partial_reveal=reveal, partial_reveal_prefix=reveal_prefix,
        custom_patterns=custom_patterns,
    )


def _result() -> RailResult:
    return RailResult(rail="pii", engine="e", verdict=Verdict.PASS)


# ── detection ──────────────────────────────────────────────────────
def test_detects_and_masks():
    res = make(entities_found=[
        {"text": "796-33-9021", "kind": "US_SSN", "confidence": 0.9},
        {"text": "4539578763621486", "kind": "CREDIT_CARD", "confidence": 0.9},
    ]).evaluate("SSN 796-33-9021 and card 4539578763621486", "mask", _result())
    assert res.verdict is Verdict.MASK
    assert {"US_SSN", "CREDIT_CARD"} <= {d.kind for d in res.detections}
    assert "796-33-9021" not in res.text_out
    assert "4539578763621486" not in res.text_out


def test_clean_text_passes():
    assert make(entities_found=[]).evaluate(
        "what documents do I need?", "mask", _result()).verdict is Verdict.PASS


def test_custom_patterns_are_folded_into_the_prompt():
    """No compiled regex exists any more for pii.custom_patterns — a
    deployment's own patterns are shown to the judge as one more thing to
    recognise by description. What stays testable deterministically is the
    wiring: the configured pattern actually reaches the prompt."""
    rail = make(custom_patterns=["a claim reference, CLM- followed by 8 digits"])
    assert "a claim reference, CLM- followed by 8 digits" in rail.system_prompt


def test_overlapping_matches_resolve_to_longest():
    """A longer match wins over a shorter one covering the same characters —
    the judge reporting two overlapping spans, not a card also read as a
    phone number the way the old regex recognizers could collide."""
    text = "4539 5787 6362 1486"
    res = make(entities_found=[
        {"text": text, "kind": "CREDIT_CARD", "confidence": 0.9},
        {"text": "5787 6362", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ]).evaluate(text, "mask", _result())
    assert [d.kind for d in res.detections] == ["CREDIT_CARD"]


# ── strategies ─────────────────────────────────────────────────────
def test_partial_reveals_only_the_tail():
    res = make("partial", entities_found=[
        {"text": "4539578763621486", "kind": "CREDIT_CARD", "confidence": 0.9},
    ]).evaluate("card 4539578763621486", "mask", _result())
    assert "1486" in res.text_out
    assert "4539578763621486" not in res.text_out


def test_partial_prefix_is_off_by_default():
    """A phone's ceiling allows a prefix, but the global dial defaults to 0."""
    res = make("partial", entities_found=[
        {"text": "555-123-4567", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ]).evaluate("call 555-123-4567", "mask", _result())
    assert not res.text_out.strip().startswith("call 55")


def test_partial_reveals_head_and_tail_for_phone():
    res = make("partial", reveal=2, reveal_prefix=2, entities_found=[
        {"text": "555-123-4567", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ]).evaluate("call 555-123-4567", "mask", _result())
    assert "555-123-4567" not in res.text_out
    assert "55" in res.text_out.split()[1][:2]
    assert res.text_out.rstrip().endswith("67")


def test_partial_prefix_ceiling_caps_ssn_at_zero():
    """SSN has no prefix ceiling — dialing the global knob up must not help."""
    res = make("partial", reveal=0, reveal_prefix=4, entities_found=[
        {"text": "796-33-9021", "kind": "US_SSN", "confidence": 0.9},
    ]).evaluate("SSN 796-33-9021", "mask", _result())
    assert not res.text_out.split()[-1].startswith("79")


def test_partial_email_keeps_domain_suffix_and_masks_the_rest():
    res = make("partial", reveal_prefix=2, entities_found=[
        {"text": "jordan.baker@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.9},
    ]).evaluate("reach me at jordan.baker@example.com", "mask", _result())
    assert "jordan.baker@example.com" not in res.text_out
    assert "@" in res.text_out
    assert res.text_out.rstrip().endswith(".com")
    masked = res.text_out.split()[-1]
    local = masked.split("@")[0]
    assert local.startswith("jo")
    assert "example" not in masked


def test_partial_email_prefix_defaults_to_fully_masked_local_part():
    """Without dialing pii.partial_reveal_prefix up, the local part stays hidden."""
    res = make("partial", entities_found=[
        {"text": "jordan.baker@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.9},
    ]).evaluate("reach me at jordan.baker@example.com", "mask", _result())
    masked = res.text_out.split()[-1]
    local = masked.split("@")[0]
    assert local == "*" * len("jordan.baker")


def test_redact_leaves_nothing():
    res = make("redact", entities_found=[
        {"text": "4539578763621486", "kind": "CREDIT_CARD", "confidence": 0.9},
    ]).evaluate("card 4539578763621486", "mask", _result())
    assert "1486" not in res.text_out
    assert "[REDACTED]" in res.text_out


def test_hash_is_stable_for_the_same_value():
    entities = [{"text": "4539578763621486", "kind": "CREDIT_CARD", "confidence": 0.9}]
    a = make("hash", entities_found=entities).evaluate(
        "card 4539578763621486", "mask", _result()).text_out
    b = make("hash", entities_found=entities).evaluate(
        "card 4539578763621486", "mask", _result()).text_out
    assert a == b


# ── actions ────────────────────────────────────────────────────────
def test_block_action_blocks():
    res = make(entities_found=[
        {"text": "796-33-9021", "kind": "US_SSN", "confidence": 0.9},
    ]).evaluate("SSN 796-33-9021", "block", _result())
    assert res.verdict is Verdict.BLOCK
    assert res.text_out is None


def test_flag_action_does_not_rewrite():
    res = make(entities_found=[
        {"text": "796-33-9021", "kind": "US_SSN", "confidence": 0.9},
    ]).evaluate("SSN 796-33-9021", "flag", _result())
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
