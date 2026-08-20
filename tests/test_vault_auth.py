"""Vault authorization.

A mask token is not a capability. Holding one proves you saw a masked reply; it
says nothing about whether the value behind it is yours. These tests exist
because the vault previously answered `reveal(token)` for anyone who asked, and
one engine-wide vault serves every session — so a token that reached the wrong
transcript reached the wrong person's SSN.

The three surfaces that can leak across users are all covered here: the vault
itself, the conversation history the reply is stored in, and the approval token
that authorises a write.
"""

from __future__ import annotations

import time

import pytest

from guardrails.rails.pii import Vault


SSN = "796-33-9021"


# ── ownership ──────────────────────────────────────────────────────
def test_the_owner_can_read_their_own_value():
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    assert v.reveal(token, "alice") == SSN


def test_another_user_cannot_read_it():
    """The hole this module exists for."""
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    assert v.reveal(token, "bob") is None


def test_an_unauthenticated_caller_cannot_read_an_owned_value():
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    assert v.reveal(token, "") is None


def test_an_owned_caller_cannot_read_an_unowned_value():
    """`""` is a real owner, not a wildcard that matches everything."""
    v = Vault()
    token = v.store("US_SSN", SSN, "")
    assert v.reveal(token, "alice") is None


def test_the_single_tenant_bucket_still_round_trips():
    """The CLI and library callers have no principal and must keep working."""
    v = Vault()
    token = v.store("US_SSN", SSN, "")
    assert v.reveal(token, "") == SSN


def test_owners_are_matched_whole_not_by_prefix():
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    for impostor in ("alic", "alice2", "Alice", " alice", "alice "):
        assert v.reveal(token, impostor) is None


def test_two_users_masking_the_same_value_get_separate_tokens():
    v = Vault()
    a = v.store("US_SSN", SSN, "alice")
    b = v.store("US_SSN", SSN, "bob")
    assert a != b
    assert v.reveal(a, "bob") is None
    assert v.reveal(b, "alice") is None
    assert v.reveal(a, "alice") == SSN


# ── the other refusals ─────────────────────────────────────────────
def test_an_unknown_token_is_refused():
    assert Vault().reveal("deadbeefcafe", "alice") is None


def test_an_expired_token_is_refused():
    v = Vault(ttl_s=0.05)
    token = v.store("US_SSN", SSN, "alice")
    assert v.reveal(token, "alice") == SSN
    time.sleep(0.08)
    assert v.reveal(token, "alice") is None


def test_an_expired_token_is_dropped_not_merely_hidden():
    """An entry with no legitimate use left should not sit in memory."""
    v = Vault(ttl_s=0.05)
    token = v.store("US_SSN", SSN, "alice")
    time.sleep(0.08)
    v.reveal(token, "alice")
    assert token not in v._store  # noqa: SLF001


def test_a_tampered_blob_is_refused_rather_than_raising():
    """AES-GCM detects it; the rail must not turn that into a 500."""
    v = Vault()
    if not v.encrypted:
        pytest.skip("cryptography not installed")
    token = v.store("US_SSN", SSN, "alice")
    entry = v._store[token]                                   # noqa: SLF001
    v._store[token] = type(entry)(                            # noqa: SLF001
        owner=entry.owner, entity=entry.entity,
        blob=entry.blob[:-1] + bytes([entry.blob[-1] ^ 0xFF]),
        created_at=entry.created_at,
    )
    assert v.reveal(token, "alice") is None


def test_rewriting_the_owner_does_not_unlock_the_value():
    """Owner is bound as AEAD associated data, so the gate has a backstop.

    If the entry map itself is tampered with — the owner field swapped to the
    attacker's name so the equality check passes — the ciphertext still will not
    decrypt, because it was sealed against the original owner.
    """
    v = Vault()
    if not v.encrypted:
        pytest.skip("cryptography not installed")
    token = v.store("US_SSN", SSN, "alice")
    entry = v._store[token]                                   # noqa: SLF001
    v._store[token] = type(entry)(                            # noqa: SLF001
        owner="bob", entity=entry.entity, blob=entry.blob,
        created_at=entry.created_at,
    )
    assert v.reveal(token, "bob") is None


# ── audit ──────────────────────────────────────────────────────────
def test_a_refused_reveal_is_recorded():
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    v.reveal(token, "bob")
    denials = v.take_denials()
    assert len(denials) == 1
    assert denials[0]["reason"] == "owner_mismatch"
    assert denials[0]["owner"] == "bob"


def test_each_refusal_reason_is_distinguished_in_the_record():
    v = Vault(ttl_s=0.05)
    live = v.store("US_SSN", SSN, "alice")
    stale = v.store("US_SSN", SSN, "alice")
    v.reveal("deadbeefcafe", "alice")     # unknown
    v.reveal(live, "bob")                 # owner_mismatch
    time.sleep(0.08)
    v.reveal(stale, "alice")              # expired
    assert {d["reason"] for d in v.take_denials()} == {
        "unknown", "owner_mismatch", "expired",
    }


def test_a_denial_never_quotes_the_value_or_the_whole_token():
    """A denial record that echoed the value would defeat the denial."""
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    v.reveal(token, "bob")
    blob = repr(v.take_denials())
    assert SSN not in blob
    assert token not in blob


def test_draining_denials_clears_them():
    v = Vault()
    v.reveal("deadbeefcafe", "alice")
    assert len(v.take_denials()) == 1
    assert v.take_denials() == []


def test_a_successful_reveal_records_nothing():
    v = Vault()
    token = v.store("US_SSN", SSN, "alice")
    assert v.reveal(token, "alice") == SSN
    assert v.take_denials() == []


def test_the_denial_log_is_bounded():
    """Enumerating tokens must not be a way to grow memory without limit."""
    v = Vault()
    for _ in range(Vault.MAX_DENIALS * 2):
        v.reveal("deadbeefcafe", "mallory")
    assert len(v.take_denials()) == Vault.MAX_DENIALS
