"""Checksum gates.

`pii.checksum_validation` is locked on. These tests are the reason it can be:
without them, a 16-digit regex fires on order numbers and tracking codes, the
queue fills with noise, and somebody disables the rail to stop it.
"""

from __future__ import annotations

import pytest

from backend.guardrails.rails.pii import (
    _VERHOEFF_D,
    _VERHOEFF_P,
    iban_mod97,
    luhn,
    pan_format,
    ssn_plausible,
    verhoeff,
)

_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def aadhaar(body: str) -> str:
    """Append a correct Verhoeff check digit to an 11-digit body."""
    c = 0
    for i, d in enumerate(reversed([int(x) for x in body])):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][d]]
    return body + str(_VERHOEFF_INV[c])


# ── Luhn ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("value,ok", [
    ("4539578763621486", True),
    ("4539578763621487", False),   # one digit off
    ("79927398713", False),        # valid Luhn, too short to be a card
    ("1234567890123456", False),   # the classic false positive
])
def test_luhn(value, ok):
    assert luhn(value) is ok


# ── Verhoeff (Aadhaar) ─────────────────────────────────────────────
def test_verhoeff_accepts_valid():
    assert verhoeff(aadhaar("23456789012")) is True


def test_verhoeff_rejects_wrong_check_digit():
    valid = aadhaar("23456789012")
    assert verhoeff(valid[:-1] + str((int(valid[-1]) + 1) % 10)) is False


@pytest.mark.parametrize("value", [
    "123456789012",   # leading 1 is never issued
    "012345678901",   # leading 0 is never issued
    "23456789012",    # wrong length
])
def test_verhoeff_rejects_bad(value):
    assert verhoeff(value) is False


# ── IBAN ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("value,ok", [
    ("GB82WEST12345698765432", True),
    ("GB82WEST12345698765433", False),
    ("DE89370400440532013000", True),
])
def test_iban(value, ok):
    assert iban_mod97(value) is ok


# ── SSN ranges ─────────────────────────────────────────────────────
@pytest.mark.parametrize("value,ok", [
    ("796-33-9021", True),
    ("000-12-3456", False),   # area 000 never issued
    ("666-12-3456", False),   # area 666 never issued
    ("900-12-3456", False),   # 9xx never issued
    ("123-00-4567", False),   # group 00 never issued
    ("123-45-0000", False),   # serial 0000 never issued
])
def test_ssn_ranges(value, ok):
    assert ssn_plausible(value) is ok


# ── PAN ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value,ok", [
    ("ABCPE1234F", True),
    ("ABCZE1234F", False),    # Z is not a valid holder-type character
    ("ABC1E1234F", False),
])
def test_pan(value, ok):
    assert pan_format(value) is ok
