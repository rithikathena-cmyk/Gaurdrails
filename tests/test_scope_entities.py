"""The two rails that closed the last input-surface gaps.

Both are two-layer, and the layering is the point: the cheap pass has to settle
the ordinary case, or a semantic check is unaffordable. These tests assert that
the model is *not* called when it should not be, which is as much a correctness
property here as catching the thing.
"""

from __future__ import annotations

import pytest

from guardrails import AuditLog, Corpus, Engine, load
from guardrails.rails.entities import EntityRail
from guardrails.rails.pii import Vault
from guardrails.rails.scope import ScopeRail
from guardrails.tracing import Tracer
from guardrails.types import RailResult, Surface, Verdict
from tests.conftest import REPO


class CountingJudge:
    """A model that records how often it was asked anything."""

    model = "stub"

    def __init__(self, payload=None):
        self.calls = 0
        self.payload = payload or {}

    def judge(self, system, user, schema, *, max_tokens=2048):
        self.calls += 1
        props = set(schema.get("properties", {}))
        if "in_scope" in props:
            return {"in_scope": self.payload.get("in_scope", 0.0),
                    "topic": self.payload.get("topic", "cookery"), "rationale": "stub"}
        if "entities" in props:
            return {"entities": self.payload.get("entities", [])}
        if "consistency" in props:
            return {"consistency": 1.0, "relevance": 1.0,
                    "unsupported_claims": [], "rationale": "stub"}
        if "injection" in props:
            return {"injection": 0.0, "technique": "none", "rationale": "stub"}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}


def blank(rail: str) -> RailResult:
    return RailResult(rail=rail, engine="test", verdict=Verdict.PASS)


# ── scope ──────────────────────────────────────────────────────────
def test_a_question_in_the_vocabulary_never_reaches_the_model():
    judge = CountingJudge()
    rail = ScopeRail(judge, 0.4, ["licence", "renew", "grant"])
    result = rail.evaluate("What documents do I need to renew a trade licence?",
                           "flag", blank("scope.domain"))
    assert result.verdict is Verdict.PASS
    assert result.meta["layer"] == "vocabulary"
    assert judge.calls == 0, "the keyword layer must settle this one for free"


def test_a_question_outside_the_vocabulary_asks_the_judge():
    judge = CountingJudge({"in_scope": 0.05, "topic": "cookery"})
    rail = ScopeRail(judge, 0.4, ["licence", "renew", "grant"])
    result = rail.evaluate("What is the best pizza recipe?", "flag", blank("scope.domain"))
    assert judge.calls == 1
    assert result.verdict is Verdict.FLAG
    assert result.meta["topic"] == "cookery"


def test_the_judge_can_rescue_a_question_phrased_sideways():
    """A bereavement is a civil-records question even with none of the words."""
    judge = CountingJudge({"in_scope": 0.9, "topic": "death certificate"})
    rail = ScopeRail(judge, 0.4, ["licence", "renew", "grant"])
    result = rail.evaluate("My father passed away and I must sort out his papers.",
                           "flag", blank("scope.domain"))
    assert result.verdict is Verdict.PASS
    assert result.score == 0.9


def test_without_a_judge_an_unmatched_question_is_allowed_through():
    """Refusing on vocabulary alone would turn every unusual phrasing away."""
    rail = ScopeRail(None, 0.4, ["licence"], use_judge=False)
    result = rail.evaluate("something phrased unusually", "block", blank("scope.domain"))
    assert result.verdict is Verdict.PASS
    assert "allowed through" in result.meta["note"]


def test_no_configured_vocabulary_disables_the_rail():
    judge = CountingJudge()
    rail = ScopeRail(judge, 0.4, [])
    result = rail.evaluate("anything at all", "block", blank("scope.domain"))
    assert result.verdict is Verdict.PASS
    assert judge.calls == 0


def test_scope_only_runs_on_what_the_user_asked(tmp_path):
    """A retrieved chunk being 'off topic' is a different question entirely."""
    engine = Engine(load(REPO / "config" / "policy.yaml"), CountingJudge(),
                    AuditLog(tmp_path / "a.log"), Corpus(seed=True))
    prompt = engine.evaluate("hello", Surface.USER_PROMPT, Tracer(), "s")
    retrieval = engine.evaluate("hello", Surface.RETRIEVAL, Tracer(), "s")
    assert any(r.rail == "scope.domain" for r in prompt.results)
    assert not any(r.rail == "scope.domain" for r in retrieval.results)


# ── named entities ─────────────────────────────────────────────────
def test_text_with_no_capitalised_candidate_never_reaches_the_model():
    judge = CountingJudge()
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge")
    result = rail.evaluate("what is the fee for a licence renewal", "mask",
                           blank("pii.entities"))
    assert result.verdict is Verdict.PASS
    assert result.meta["layer"] == "gate"
    assert judge.calls == 0


def test_a_name_and_an_address_are_masked_into_the_vault():
    judge = CountingJudge({"entities": [
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
        {"text": "14 Anna Salai", "kind": "ADDRESS", "confidence": 0.9},
    ]})
    vault = Vault()
    rail = EntityRail(judge, vault, 0.6, "vault-token", engine_mode="judge")
    text = "My name is Meera Balan, I live at 14 Anna Salai."
    result = rail.evaluate(text, "mask", blank("pii.entities"))
    assert result.verdict is Verdict.MASK
    assert "Meera Balan" not in result.text_out
    assert "<PERSON:" in result.text_out and "<ADDRESS:" in result.text_out


def test_a_span_the_model_invented_is_dropped_rather_than_masked():
    """Masking a span that is not in the text would rewrite something else."""
    judge = CountingJudge({"entities": [
        {"text": "Someone Not Present", "kind": "PERSON", "confidence": 0.99},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge")
    result = rail.evaluate("A letter from Meera about her claim.", "mask",
                           blank("pii.entities"))
    assert result.verdict is Verdict.PASS
    assert result.meta["unverifiable_spans"] == 1


def test_low_confidence_entities_are_ignored():
    judge = CountingJudge({"entities": [
        {"text": "Meera", "kind": "PERSON", "confidence": 0.2},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge")
    result = rail.evaluate("A letter from Meera.", "mask", blank("pii.entities"))
    assert result.verdict is Verdict.PASS


def test_public_bodies_are_not_treated_as_entities_by_configuration():
    """LOCATION is off by default: a city on its own does not identify anyone."""
    policy = load(REPO / "config" / "policy.yaml")
    assert "LOCATION" not in policy.get("pii.entity_kinds")


# ── the composition fix these rails depend on ──────────────────────
def test_two_masking_rails_both_survive(tmp_path):
    """Regression: each rail computed its rewrite from the original text, so the
    last one to finish silently discarded the others. A blocked word next to an
    SSN came out unmasked."""
    engine = Engine(load(REPO / "config" / "policy.yaml"), None,
                    AuditLog(tmp_path / "a.log"), Corpus(seed=True))
    result = engine.evaluate("you are an idiot, my ssn is 796-33-9021",
                             Surface.USER_PROMPT, Tracer(), "s")
    assert "*****" in result.text, "the word rail's masking was lost"
    assert "<US_SSN:" in result.text, "the pii rail's masking was lost"


# ── secrets and destructive intent, deterministic ──────────────────
@pytest.mark.parametrize("text", [
    "my api_key = sk-live4f9a2b8ccc12",
    "the password: hunter2 for the portal",
    "use AKIAIOSFODNN7EXAMPLE for the bucket",
    "here is my -----BEGIN RSA PRIVATE KEY----- blob",
])
def test_a_credential_in_a_prompt_is_blocked(engine, text):
    """Blocked, not masked: masking hides the incident from whoever must rotate it."""
    result = engine.evaluate(text, Surface.USER_PROMPT, Tracer(), "s")
    assert result.verdict is Verdict.BLOCK


@pytest.mark.parametrize("text", [
    "please drop table applicants",
    "delete all the licence records now",
    "just run rm -rf / on the server",
])
def test_destructive_intent_is_blocked_without_a_model(engine, text):
    result = engine.evaluate(text, Surface.USER_PROMPT, Tracer(), "s")
    assert result.verdict is Verdict.BLOCK


@pytest.mark.parametrize("text", [
    "what documents do I need to renew a trade licence",
    "my income is 250,000 and I want the housing grant",
    "how do I object to a property tax assessment",
])
def test_ordinary_questions_are_untouched_by_the_new_rules(engine, text):
    result = engine.evaluate(text, Surface.USER_PROMPT, Tracer(), "s")
    assert result.verdict is Verdict.PASS
