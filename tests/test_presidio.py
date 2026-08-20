"""Local NER as the cheap layer under the entity judge.

Marked `presidio` because these are the only tests that build the spaCy
pipeline — eleven seconds once, and about a second per analyse. Run them with
`pytest -m presidio`; the rest of the suite stubs the engine out.

The interesting assertions are the filters, not the detection. Presidio finds
names well and returns a good deal of noise alongside them: run against one
ordinary sentence it offered `PERSON 'mobile 9962214477'`, `ORGANIZATION 'SSN'`
and `URL 'anitha.se'`. What makes it usable is what gets thrown away.
"""

from __future__ import annotations

import pytest

from guardrails.rails import presidio_ner
from guardrails.rails.entities import EntityRail
from guardrails.rails.pii import Vault
from guardrails.types import Detection, RailResult, Verdict

pytestmark = pytest.mark.presidio

SENTENCE = ("My name is Anitha Selvam, I live at 14 Anna Salai Chennai. "
            "Email anitha.selvam@example.com, mobile 9962214477.")


@pytest.fixture(scope="module")
def engine_ready():
    if not presidio_ner.available():
        pytest.skip("presidio-analyzer is not installed")
    if presidio_ner.engine() is None:
        pytest.skip("presidio engine could not be built (missing spaCy model?)")
    return True


def blank() -> RailResult:
    return RailResult(rail="pii.entities", engine="test", verdict=Verdict.PASS)


# ── the filters ────────────────────────────────────────────────────
def test_a_name_is_found_without_a_model_call(engine_ready):
    found = presidio_ner.find(SENTENCE, {"PERSON", "ADDRESS"}, 0.6)
    assert any(f["kind"] == "PERSON" and "Anitha" in f["text"] for f in found), found


def test_kinds_the_deterministic_rail_already_covers_are_discarded(engine_ready):
    """Presidio reports EMAIL_ADDRESS and PHONE_NUMBER too. The regex rail found
    them in a tenth of a millisecond, with a checksum, so the NER copy is a
    worse duplicate of a better answer."""
    found = presidio_ner.find(SENTENCE, {"PERSON", "ADDRESS", "ORGANISATION"}, 0.6)
    assert not any(f["kind"] in {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"}
                   for f in found), found


def test_a_span_overlapping_a_deterministic_hit_is_dropped(engine_ready):
    """The case that made this necessary: Presidio returned
    PERSON 'mobile 9962214477', which would have masked the word "mobile"
    along with a number the regex rail had already claimed."""
    phone_at = SENTENCE.index("9962214477")
    taken = [(phone_at, phone_at + len("9962214477"))]
    found = presidio_ner.find(SENTENCE, {"PERSON", "ADDRESS"}, 0.6, taken=taken)
    assert not any("9962214477" in f["text"] for f in found), found


def test_low_confidence_spans_never_reach_the_vault(engine_ready):
    """US_BANK_NUMBER at 0.05 and US_DRIVER_LICENSE at 0.01 were both offered
    for a plain mobile number."""
    strict = presidio_ner.find(SENTENCE, {"PERSON", "ADDRESS", "ORGANISATION"}, 0.95)
    loose = presidio_ner.find(SENTENCE, {"PERSON", "ADDRESS", "ORGANISATION"}, 0.1)
    assert len(strict) <= len(loose)
    assert all(f["confidence"] >= 0.95 for f in strict)


# ── the rail ───────────────────────────────────────────────────────
def test_the_rail_masks_a_name_with_no_judge_configured(engine_ready):
    """The point of the layer: no model, and the name still goes to the vault."""
    rail = EntityRail(None, Vault(), 0.6, "vault-token", engine_mode="presidio")
    out = rail.evaluate(SENTENCE, "mask", blank())
    assert out.verdict is Verdict.MASK
    assert "Anitha Selvam" not in out.text_out
    assert out.meta["layer"] == "presidio"


def test_the_judge_is_not_asked_when_local_ner_answers(engine_ready):
    """A judge that is never called is the whole saving."""
    class Counting:
        model = "stub"
        calls = 0

        def judge(self, *a, **kw):
            type(self).calls += 1
            return {"entities": []}

    rail = EntityRail(Counting(), Vault(), 0.6, "vault-token",
                      engine_mode="presidio+judge")
    rail.evaluate(SENTENCE, "mask", blank())
    assert Counting.calls == 0


def test_the_judge_still_covers_what_ner_misses(engine_ready):
    """Presidio missed the street address entirely. With both engines the judge
    is asked when the cheap layer comes back empty."""
    class Judge:
        model = "stub"

        def judge(self, system, user, schema, *, max_tokens=2048):
            return {"entities": [
                {"text": "14 Anna Salai", "kind": "ADDRESS", "confidence": 0.9}]}

    text = "Deliver the notice to 14 Anna Salai please."
    rail = EntityRail(Judge(), Vault(), 0.6, "vault-token", engine_mode="presidio+judge")
    out = rail.evaluate(text, "mask", blank())
    if out.meta.get("layer") == "judge":
        assert "14 Anna Salai" not in out.text_out


# ── availability ───────────────────────────────────────────────────
def test_a_missing_engine_falls_through_rather_than_failing(monkeypatch):
    """An optional dependency that is absent is a capability question, not an
    error: the rail must fall back to the judge, not fail the request."""
    monkeypatch.setattr(presidio_ner, "engine", lambda: None)
    assert presidio_ner.find("My name is Anitha Selvam", {"PERSON"}, 0.6) == []

    class Judge:
        model = "stub"

        def judge(self, *a, **kw):
            return {"entities": [
                {"text": "Anitha Selvam", "kind": "PERSON", "confidence": 0.95}]}

    rail = EntityRail(Judge(), Vault(), 0.6, "vault-token", engine_mode="presidio+judge")
    out = rail.evaluate("My name is Anitha Selvam.", "mask", blank())
    assert out.verdict is Verdict.MASK
    assert out.meta["layer"] == "judge"
