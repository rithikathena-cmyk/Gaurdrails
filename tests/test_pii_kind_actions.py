"""`pii.kind_actions` — the decide/enforce half of classify-then-decide.

`EntityRail` does the classifying — the judge, and Presidio for PERSON/
ADDRESS — this file is about what happens once a kind is known: whether
PERSON gets redacted while GOVERNMENT and ORGANISATION are left alone on the
same text, and whether kinds that used to have their own regex/checksum
recognizer (email, phone, Aadhaar, PAN) still resolve their action correctly
now that they are judge-only, the same as any other kind. The agent that
classifies never decides an action itself — every verdict below comes from
`action_verdict()` + `kind_actions.resolve()`, the same two functions the
engine already used for a single global action per surface.
"""

from __future__ import annotations

from backend.guardrails.rails.entities import EntityRail
from backend.guardrails.rails.vault import Vault
from backend.guardrails.types import RailResult, Verdict


def blank(rail: str) -> RailResult:
    return RailResult(rail=rail, engine="test", verdict=Verdict.PASS)


class CountingJudge:
    """Answers exactly what it is told to, and remembers what it was asked."""

    model = "stub"

    def __init__(self, entities=()):
        self.entities = list(entities)
        self.calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.calls += 1
        props = set(schema.get("properties", {}))
        if "entities" in props:
            return {"entities": self.entities}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}


def _entity_rail(entities_found, allowlist=(), **kind_actions) -> EntityRail:
    """The structured-kind sibling of the NER tests below — same rail class,
    same judge-only detection, just the kinds pii.py's recognizers used to
    own (email, phone, Aadhaar, PAN)."""
    judge = CountingJudge(entities=entities_found)
    return EntityRail(
        judge, Vault(), 0.5, "vault-token", engine_mode="judge",
        kinds=["EMAIL_ADDRESS", "PHONE_NUMBER", "AADHAAR", "PAN"],
        allowlist=list(allowlist), kind_actions=kind_actions,
    )


# ── Test 1: ingestion-shaped classification, per-kind decide ───────────
def test_person_is_redacted_organisation_and_location_are_allowed():
    """The exact example from the spec: a sentence naming a person and a
    government-run bank, each governed by its own kind action."""
    judge = CountingJudge(entities=[
        {"text": "Ramesh Kumar", "kind": "PERSON", "confidence": 0.98},
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "GOVERNMENT",
         "confidence": 0.99},
    ])
    rail = EntityRail(
        judge, Vault(), 0.6, "vault-token", engine_mode="judge",
        kinds=["PERSON", "GOVERNMENT", "ORGANISATION"],
        kind_actions={"PERSON": "mask", "GOVERNMENT": "pass", "ORGANISATION": "pass"},
    )
    text = ("Mr. Ramesh Kumar contacted the Tamil Nadu State Apex Cooperative Bank "
           "regarding his account.")
    result = rail.evaluate(text, "mask", blank("pii.entities"))

    assert result.verdict is Verdict.MASK          # PERSON alone drives the aggregate
    assert "Ramesh Kumar" not in result.text_out
    assert "Tamil Nadu State Apex Cooperative Bank" in result.text_out, (
        "GOVERNMENT => pass must leave the bank's name exactly as written"
    )


def test_a_government_body_and_a_private_employer_get_different_actions():
    """The split the spec asks for: ORGANISATION (private) and GOVERNMENT
    (public) are different kinds so they can carry different policies, even
    though the same text could plausibly be either without this rail."""
    judge = CountingJudge(entities=[
        {"text": "Acme Textiles Pvt Ltd", "kind": "ORGANISATION", "confidence": 0.95},
        {"text": "District Central Cooperative Bank", "kind": "GOVERNMENT",
         "confidence": 0.95},
    ])
    rail = EntityRail(
        judge, Vault(), 0.6, "vault-token", engine_mode="judge",
        kinds=["ORGANISATION", "GOVERNMENT"],
        kind_actions={"ORGANISATION": "mask", "GOVERNMENT": "pass"},
    )
    text = "He works at Acme Textiles Pvt Ltd, which banks with the District Central Cooperative Bank."
    result = rail.evaluate(text, "mask", blank("pii.entities"))

    assert "Acme Textiles Pvt Ltd" not in result.text_out
    assert "District Central Cooperative Bank" in result.text_out


def test_a_kind_with_no_override_falls_back_to_the_surface_action():
    """`pii.kind_actions` only carves out exceptions — a kind nobody
    mentioned still gets exactly what the surface already said."""
    judge = CountingJudge(entities=[
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
    ])
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge",
                      kind_actions={"GOVERNMENT": "pass"})   # PERSON not mentioned
    result = rail.evaluate("A letter from Meera Balan.", "mask", blank("pii.entities"))
    assert result.verdict is Verdict.MASK
    assert "Meera Balan" not in result.text_out


def test_a_kind_resolved_to_block_blocks_the_whole_result_not_just_that_span():
    """Precedence still applies across a mix of per-kind actions — the same
    "most restrictive wins" rule the engine already enforces across rails,
    just now within one rail's own set of findings."""
    judge = CountingJudge(entities=[
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
        {"text": "Some Ministry", "kind": "GOVERNMENT", "confidence": 0.95},
    ])
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge",
                      kinds=["PERSON", "GOVERNMENT"],
                      kind_actions={"PERSON": "block", "GOVERNMENT": "pass"})
    result = rail.evaluate("Meera Balan works at Some Ministry.", "mask", blank("pii.entities"))
    assert result.verdict is Verdict.BLOCK
    assert result.text_out is None, "a blocked result is refused outright, not partly redacted"


# ── Test 2: structured PII kinds resolve their action like any other ───
def test_structured_pii_kinds_are_still_found_and_masked_by_default():
    text = ("Email meera@example.com, call 415-555-0143, Aadhaar 234123412346, "
           "PAN AAAPL1234C.")
    rail = _entity_rail([
        {"text": "meera@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.9},
        {"text": "415-555-0143", "kind": "PHONE_NUMBER", "confidence": 0.9},
        {"text": "234123412346", "kind": "AADHAAR", "confidence": 0.9},
        {"text": "AAAPL1234C", "kind": "PAN", "confidence": 0.9},
    ])
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert result.verdict is Verdict.MASK
    for value in ("meera@example.com", "415-555-0143", "234123412346", "AAAPL1234C"):
        assert value not in result.text_out
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "AADHAAR", "PAN"} <= set(result.meta["by_type"])


def test_one_structured_kind_can_be_relaxed_without_affecting_the_others():
    """PHONE_NUMBER => flag while everything else still masks — proving the
    per-kind override reaches every judge-detected kind alike, not just the
    ones that were always NER-only."""
    text = "Email meera@example.com, call 415-555-0143."
    rail = _entity_rail([
        {"text": "meera@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.9},
        {"text": "415-555-0143", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ], PHONE_NUMBER="flag")
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert result.verdict is Verdict.MASK          # EMAIL_ADDRESS still masks -> aggregate MASK
    assert "meera@example.com" not in result.text_out
    assert "415-555-0143" in result.text_out, "PHONE_NUMBER => flag must leave the number visible"


# ── Test 3: the allowlist still wins over any kind action ───────────────
def test_an_allowlisted_organisation_stays_visible_even_if_its_kind_is_masked():
    """The allowlist is checked before any per-kind action is resolved — an
    exempt value never reaches the decide step at all, the same "detect ->
    exempt -> mask" locked ordering `pii.allowlist_ordering` already
    documents."""
    judge = CountingJudge(entities=[
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "ORGANISATION",
         "confidence": 0.95},
    ])
    rail = EntityRail(
        judge, Vault(), 0.6, "vault-token", engine_mode="judge",
        kinds=["ORGANISATION"],
        allowlist=[r"Tamil Nadu State Apex Cooperative Bank"],
        kind_actions={"ORGANISATION": "mask"},   # would otherwise redact it
    )
    result = rail.evaluate(
        "Write to the Tamil Nadu State Apex Cooperative Bank for details.",
        "mask", blank("pii.entities"),
    )
    assert result.verdict is Verdict.PASS
    assert result.text_out is None
    assert result.meta["allowlisted"] == 1


def test_an_allowlisted_phone_number_stays_visible_even_if_its_kind_is_masked():
    text = "Call the helpline at 1800 425 1969."
    rail = _entity_rail([
        {"text": "1800 425 1969", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ], allowlist=[r"1800[\s-]?425[\s-]?1969"], PHONE_NUMBER="mask")
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert "1800 425 1969" in (result.text_out or text)
    assert result.meta["allowlisted"] == 1


# ── pii.kind_mask_strategy: per-kind rendering, decide/enforce again ────
# `pii.kind_actions` decides *whether* a kind gets masked; this decides what
# the masked text *looks like* once it does — two independent axes, proven
# independent by giving two kinds in the same call two different strategies.
def test_two_structured_kinds_render_with_different_strategies_in_one_call():
    judge = CountingJudge(entities=[
        {"text": "meera@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.9},
        {"text": "415-555-0143", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ])
    rail = EntityRail(
        judge, Vault(), 0.5, "vault-token", engine_mode="judge",
        kinds=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        kind_mask_strategy={"PHONE_NUMBER": "redact"},   # the global default, overridden for one kind
    )
    text = "Email meera@example.com, call 415-555-0143."
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert "<EMAIL_ADDRESS:" in result.text_out, "no override -> the global vault-token"
    assert "[REDACTED]" in result.text_out, "PHONE_NUMBER's own override -> redact"
    assert "415-555-0143" not in result.text_out


def test_two_ner_kinds_render_with_different_strategies_in_one_call():
    judge = CountingJudge(entities=[
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
        {"text": "14 Anna Salai", "kind": "ADDRESS", "confidence": 0.9},
    ])
    rail = EntityRail(
        judge, Vault(), 0.6, "vault-token", engine_mode="judge",
        kinds=["PERSON", "ADDRESS"],
        kind_mask_strategy={"ADDRESS": "replace"},
    )
    result = rail.evaluate("My name is Meera Balan, I live at 14 Anna Salai.",
                           "mask", blank("pii.entities"))
    assert "<PERSON:" in result.text_out, "no override -> the global vault-token"
    assert "<ADDRESS>" in result.text_out, "ADDRESS's own override -> replace, no token"
    assert "14 Anna Salai" not in result.text_out


def test_a_kind_mask_strategy_override_does_not_change_whether_it_is_masked():
    """Strategy is downstream of the decide step, not a second decision —
    overriding PHONE_NUMBER's *rendering* has no say over PHONE_NUMBER's
    *action*, which stays whatever pii.kind_actions (or the surface
    default) already resolved it to."""
    judge = CountingJudge(entities=[
        {"text": "415-555-0143", "kind": "PHONE_NUMBER", "confidence": 0.9},
    ])
    rail = EntityRail(
        judge, Vault(), 0.5, "vault-token", engine_mode="judge",
        kinds=["PHONE_NUMBER"],
        kind_actions={"PHONE_NUMBER": "flag"},           # would not mask at all
        kind_mask_strategy={"PHONE_NUMBER": "redact"},   # irrelevant if never masked
    )
    text = "Call 415-555-0143."
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert result.verdict is Verdict.FLAG
    assert "415-555-0143" in result.text_out if result.text_out else text
    assert "[REDACTED]" not in (result.text_out or "")


def test_an_invalid_kind_mask_strategy_value_is_rejected():
    from backend.guardrails.rails.kind_actions import KindActionError, parse_strategy

    try:
        parse_strategy(["EMAIL_ADDRESS => shred"])
        assert False, "an unknown strategy must not silently pass through"
    except KindActionError as exc:
        assert "shred" in str(exc)
