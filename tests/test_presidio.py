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

from backend.guardrails.rails import presidio_ner
from backend.guardrails.rails.entities import EntityRail
from backend.guardrails.rails.pii import Vault
from backend.guardrails.types import Detection, RailResult, Verdict

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


def test_the_judge_reviews_what_local_ner_proposes(engine_ready):
    """`presidio+judge` means Presidio proposes and the judge decides.

    This test used to assert the opposite — that a Presidio hit skipped the
    judge entirely, on the grounds that a judge never called is the whole
    saving. That saving was real and the cost was invisible: Presidio's hit was
    never reviewed, and Presidio hits things that are not people. On this
    service's own reply it read "Birth", in "Birth and death records", as a
    person at 0.85 and masked it. Nobody noticed because egress unmasks for the
    token's owner; another reader would have got "<PERSON:…> and death records".

    A judge call is skipped only where nothing was proposed to review — the
    structural gate still ends most requests before either layer runs.
    """
    class Rejecting:
        model = "stub"
        calls = 0

        def judge(self, *a, **kw):
            type(self).calls += 1
            return {"entities": []}          # corroborates nothing

    rail = EntityRail(Rejecting(), Vault(), 0.6, "vault-token",
                      engine_mode="presidio+judge")
    out = rail.evaluate(SENTENCE, "mask", blank())

    assert Rejecting.calls == 1
    assert out.meta["presidio_proposed"] >= 1
    assert out.meta["presidio_corroborated"] == 0
    assert out.meta["presidio_rejected"] == out.meta["presidio_proposed"]
    assert out.verdict is Verdict.PASS       # an unreviewed guess masks nothing


def test_a_corroborated_proposal_is_still_masked(engine_ready):
    """Rejecting the false positives must not cost us the true ones."""
    class Agreeing:
        model = "stub"

        def judge(self, *a, **kw):
            return {"entities": [
                {"text": "Anitha Selvam", "kind": "PERSON", "confidence": 0.95}]}

    rail = EntityRail(Agreeing(), Vault(), 0.6, "vault-token",
                      engine_mode="presidio+judge")
    out = rail.evaluate(SENTENCE, "mask", blank())
    assert out.verdict is Verdict.MASK
    assert "Anitha Selvam" not in out.text_out


def test_presidio_alone_still_skips_the_judge(engine_ready):
    """`presidio` mode is the setting for a deployment that wants no API call.

    It keeps the old behaviour deliberately: the local layer decides on its own,
    false positives included. The trade is named in the enum rather than hidden.
    """
    class Counting:
        model = "stub"
        calls = 0

        def judge(self, *a, **kw):
            type(self).calls += 1
            return {"entities": []}

    rail = EntityRail(Counting(), Vault(), 0.6, "vault-token", engine_mode="presidio")
    out = rail.evaluate(SENTENCE, "mask", blank())
    assert Counting.calls == 0
    assert out.verdict is Verdict.MASK


def test_a_published_contact_is_exempt_from_ner_too(engine_ready):
    """`pii.allowlist` used to hold against the regex rail only.

    A department address the operator deliberately published could still be
    masked here, by a different rail, on the same text — which is how "who do I
    write to" stops being answerable.
    """
    text = "Birth and death records: records@municipal.gov.in (Registrar)"
    allow = [r"[a-z0-9._%+-]+@[a-z0-9.-]*municipal\.gov\.in"]

    unguarded = EntityRail(None, Vault(), 0.6, "vault-token",
                           engine_mode="presidio").evaluate(text, "mask", blank())
    guarded = EntityRail(None, Vault(), 0.6, "vault-token", engine_mode="presidio",
                         allowlist=allow).evaluate(text, "mask", blank())

    assert "records@municipal.gov.in" in (guarded.text_out or text)
    assert guarded.meta["allowlisted"] >= 1
    # The exemption is applied after detection, so the finding is still recorded.
    assert unguarded.meta["allowlisted"] == 0


def test_the_judge_still_covers_what_ner_misses(engine_ready):
    """Presidio missed the street address entirely. With both engines the judge
    is asked when the cheap layer comes back empty."""
    class Judge:
        model = "stub"

        def judge(self, system, user, schema, *, max_tokens=2048, label=""):
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
