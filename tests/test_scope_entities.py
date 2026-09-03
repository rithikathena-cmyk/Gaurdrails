"""The two rails that closed the last input-surface gaps.

Both are two-layer, and the layering is the point: the cheap pass has to settle
the ordinary case, or a semantic check is unaffordable. These tests assert that
the model is *not* called when it should not be, which is as much a correctness
property here as catching the thing.
"""

from __future__ import annotations

import pytest

from backend.guardrails import AuditLog, Corpus, Engine, load
from backend.guardrails.rails.entities import EntityRail
from backend.guardrails.rails.vault import Vault
from backend.guardrails.rails.scope import ScopeRail
from backend.guardrails.tracing import Tracer
from backend.guardrails.types import RailResult, Surface, Verdict
from tests.conftest import REPO


class CountingJudge:
    """A model that records how often it was asked anything."""

    model = "stub"

    def __init__(self, payload=None):
        self.calls = 0
        self.payload = payload or {}

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.calls += 1
        props = set(schema.get("properties", {}))
        if "in_scope" in props:
            return {"in_scope": self.payload.get("in_scope", 0.0),
                    "topic": self.payload.get("topic", "cookery"), "rationale": "stub"}
        if "entities" in props:
            return {"entities": self.payload.get("entities", [])}
        if "consistency" in props:
            return {"consistency": 1.0, "relevance": 1.0,
                    "unsupported": [], "rationale": "stub"}
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


def test_no_configured_vocabulary_still_asks_the_judge():
    """`domain_terms` ships empty — no deployment's vocabulary is hardcoded —
    so an empty list must not silently disable the rail. The keyword layer
    has nothing to match against and falls straight through, exactly like a
    genuine no-hit case with a populated list: the judge is still asked, and
    still enforced."""
    judge = CountingJudge({"in_scope": 0.9, "topic": "on topic"})
    rail = ScopeRail(judge, 0.4, [])
    result = rail.evaluate("anything at all", "block", blank("scope.domain"))
    assert judge.calls == 1
    assert result.verdict is Verdict.PASS
    assert result.meta["layer"] == "judge"


def test_no_configured_vocabulary_can_still_block():
    """Below `hard_block_threshold` the judge is not merely unsure — this is
    the confidently-unrelated case scope still refuses on its own."""
    judge = CountingJudge({"in_scope": 0.05, "topic": "cookery"})
    rail = ScopeRail(judge, 0.4, [], hard_block_threshold=0.15)
    result = rail.evaluate("what's a good pizza dough recipe", "block", blank("scope.domain"))
    assert judge.calls == 1
    assert result.verdict is Verdict.BLOCK


def test_an_uncertain_score_flags_rather_than_blocking():
    """Below `threshold` but not below `hard_block_threshold`: the judge is
    unsure, not confident, so this no longer refuses on its own — retrieval
    gets to decide instead. See `requires_retrieval` in this module and its
    use in `engine.py` / `agent/runner.py`."""
    judge = CountingJudge({"in_scope": 0.25, "topic": "cooperative federation"})
    rail = ScopeRail(judge, 0.4, [], hard_block_threshold=0.15)
    result = rail.evaluate("what is the address for the consumer cooperative federation",
                           "block", blank("scope.domain"))
    assert judge.calls == 1
    assert result.verdict is Verdict.FLAG


def test_hard_block_threshold_only_applies_under_the_block_action():
    """`flag` never refuses on its own, however confidently unrelated the
    judge is — only `scope.action: block` gets the hard-block tier."""
    judge = CountingJudge({"in_scope": 0.02, "topic": "cookery"})
    rail = ScopeRail(judge, 0.4, [], hard_block_threshold=0.15)
    result = rail.evaluate("what's a good pizza dough recipe", "flag", blank("scope.domain"))
    assert result.verdict is Verdict.FLAG


def test_scope_only_runs_on_what_the_user_asked(tmp_path):
    """A retrieved chunk being 'off topic' is a different question entirely."""
    engine = Engine(load(REPO / "config" / "policy.yaml"), CountingJudge(),
                    AuditLog(tmp_path / "a.log"), Corpus(seed=True))
    prompt = engine.evaluate("hello", Surface.USER_PROMPT, Tracer(), "s")
    retrieval = engine.evaluate("hello", Surface.RETRIEVAL, Tracer(), "s")
    assert any(r.rail == "scope.domain" for r in prompt.results)
    assert not any(r.rail == "scope.domain" for r in retrieval.results)


# ── judge-only mode windows a large document too ────────────────────
# Regression: `engine_mode="judge"` (no Presidio) used to take a completely
# different branch from `presidio+judge` — one raw `self.llm.judge(...)`
# call over the *whole* text, no windowing, and no explicit `max_tokens`
# (silently the SDK default, 2048). Fine for a prompt or a reply; a real
# ~150,000-character document sent that way is guaranteed to come back
# truncated. This is not a hypothetical corner: `Engine.reseed_builtin_
# rails()` forces exactly this mode, specifically for the one real document
# this whole codebase always has — a live ingest run quarantined it on
# `judge returned non-JSON`, repeatedly, every time it took this path,
# while the identical content windowed through `presidio+judge` never
# failed once in the same testing. `TrackingJudge` below records each
# call's own `user` length and `max_tokens`, which a bare call-count
# assertion cannot tell apart from one giant call that happened to be
# asked twice for an unrelated reason.
class TrackingJudge:
    model = "stub"

    def __init__(self, entities_by_window):
        """`entities_by_window(user_text) -> list[dict]` — let each call
        answer for its own slice, so entities from every window are
        distinguishable in the merged result."""
        self._entities_by_window = entities_by_window
        self.calls: list[tuple[int, int]] = []   # (len(user), max_tokens)

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.calls.append((len(user), max_tokens))
        props = set(schema.get("properties", {}))
        if "entities" in props:
            return {"entities": self._entities_by_window(user)}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}


def test_judge_only_mode_windows_a_document_larger_than_one_window():
    from backend.guardrails.rails.entities import _JUDGE_MAX_TOKENS, _JUDGE_WINDOW_CHARS

    # Three windows' worth, each holding its own distinctive capitalised
    # name so a dropped window is visible as a missing entity, not just a
    # missing call.
    names = ["Meera Balan", "Ramesh Kumar", "Anitha Selvam"]
    paragraph = "This paragraph exists only to take up space. " * 200  # > one window
    text = f"\n\n{paragraph}\n\n".join(f"A note about {n}." for n in names)
    assert len(text) > _JUDGE_WINDOW_CHARS * 2, "the fixture must span multiple windows"

    def entities_by_window(user_text: str) -> list[dict]:
        return [{"text": n, "kind": "PERSON", "confidence": 0.9}
                for n in names if n in user_text]

    judge = TrackingJudge(entities_by_window)
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="judge")
    result = rail.evaluate(text, "mask", blank("pii.entities"))

    assert len(judge.calls) > 1, "a document this large must be windowed, not sent whole"
    for length, max_tokens in judge.calls:
        assert length <= _JUDGE_WINDOW_CHARS + 1, (
            f"a {length}-char call was not bounded to one window"
        )
        assert max_tokens == _JUDGE_MAX_TOKENS, (
            f"got max_tokens={max_tokens}, not the rail's own {_JUDGE_MAX_TOKENS} budget "
            "— the SDK default (2048) is what let a real document truncate silently"
        )
    for name in names:
        assert name not in result.text_out, f"{name} from a non-first window was dropped"


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


# ── retrieval-surface candidate gating ─────────────────────────────
# Retrieved text is the corpus's own document, not user input, and a real
# document is thick with capitalised words the cheap `_CANDIDATE` gate above
# cannot tell apart from an actual name — so on retrieval specifically,
# `presidio+judge` only pays for the judge when Presidio itself proposed
# something. `user.prompt` is deliberately unaffected: see
# `test_a_name_and_an_address_are_masked_into_the_vault` and friends above,
# none of which pass `surface=`, so they exercise the always-ask-the-judge
# default this gate must not touch.
RCS_RETRIEVAL_TEXT = "The Tamil Nadu State Apex Cooperative Bank runs this scheme."


def test_retrieval_text_with_no_presidio_candidate_skips_the_judge():
    """The regression this closes: every retrieval-surface question against a
    real document paid for a full judge scan even when Presidio — the cheap
    layer specifically meant to gate it — found nothing at all. `no_local_ner`
    (autouse) already makes Presidio unavailable, i.e. `proposed == []`,
    which is the exact condition this gate is keyed on."""
    judge = CountingJudge({"entities": [
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "ORGANISATION",
         "confidence": 0.9},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="presidio+judge")
    result = rail.evaluate(RCS_RETRIEVAL_TEXT, "mask", blank("pii.entities"),
                           surface="retrieval")
    assert judge.calls == 0, "nothing for Presidio to propose — the judge must not run"
    assert result.verdict is Verdict.PASS
    assert result.meta["layer"] == "presidio"
    assert result.meta["retrieval_judge_skipped"] == "no presidio candidate"


def test_user_prompt_text_with_no_presidio_candidate_still_asks_the_judge():
    """The asymmetry, proven directly: the identical text and identical
    "Presidio found nothing" condition, but on `user.prompt` — where a missed
    name matters most and the text is never six retrieved chunks long — the
    judge still runs, unaffected by the retrieval-only gate above."""
    judge = CountingJudge({"entities": [
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "ORGANISATION",
         "confidence": 0.9},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="presidio+judge")
    result = rail.evaluate(RCS_RETRIEVAL_TEXT, "mask", blank("pii.entities"),
                           surface="user.prompt")
    assert judge.calls == 1
    assert result.meta["layer"] == "judge"
    assert "retrieval_judge_skipped" not in result.meta


# ── Test 4 ──────────────────────────────────────────────────────────
def test_retrieval_text_with_a_presidio_candidate_still_asks_the_judge(monkeypatch):
    """The other half of the gate: Presidio proposing *something* — a real
    PII candidate, not an institution name — still hands the question to the
    judge for corroboration, exactly as `user.prompt` always does. The cost
    saving is specific to "nothing to even ask about", not retrieval as a
    whole."""
    from backend.guardrails.rails import presidio_ner

    text = "A note from Meera Balan about the claim."
    start = text.index("Meera Balan")

    def fake_find(text, kinds, min_confidence, taken=None):
        return [{"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95,
                 "start": start, "end": start + len("Meera Balan")}]

    monkeypatch.setattr(presidio_ner, "find", fake_find)
    judge = CountingJudge({"entities": [
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "vault-token", engine_mode="presidio+judge")
    result = rail.evaluate(text, "mask", blank("pii.entities"), surface="retrieval")
    assert judge.calls == 1, "a real Presidio candidate must still reach the judge"
    assert result.verdict is Verdict.MASK
    assert "Meera Balan" not in result.text_out
    assert result.meta["layer"] == "presidio+judge"


def test_partial_masking_of_a_name_reveals_nothing_regardless_of_config():
    """Regression: partial masking used to hardcode exactly one leading
    character no matter what `pii.partial_reveal_prefix` said. No NER kind
    has a non-zero reveal ceiling yet, so even a generous config still fully
    masks — same "other entity kinds ignore it" rule pii.py's own non-email/
    phone recognizers already follow."""
    judge = CountingJudge({"entities": [
        {"text": "Meera Balan", "kind": "PERSON", "confidence": 0.95},
    ]})
    rail = EntityRail(judge, Vault(), 0.6, "partial", engine_mode="judge",
                      partial_reveal=4, partial_reveal_prefix=2)
    result = rail.evaluate("My name is Meera Balan.", "mask", blank("pii.entities"))
    assert result.verdict is Verdict.MASK
    assert "Meera Balan" not in result.text_out
    assert "*" * len("Meera Balan") in result.text_out


# ── the composition fix these rails depend on ──────────────────────
@pytest.mark.presidio
def test_two_masking_rails_both_survive(tmp_path):
    """Regression: each rail computed its rewrite from the original text, so the
    last one to finish silently discarded the others. A blocked word next to a
    name came out unmasked.

    A name, not an SSN: PII has no deterministic layer any more, and an SSN is
    judge-only now — this test wants no model (`llm=None`), so it needs a kind
    Presidio's real local NER can still find without one. `@pytest.mark.presidio`
    turns off `conftest.py`'s `no_local_ner` stub, which otherwise blocks every
    kind in this suite, checksummed or not, the same way it always blocked
    PERSON/ADDRESS."""
    engine = Engine(load(REPO / "config" / "policy.yaml"), None,
                    AuditLog(tmp_path / "a.log"), Corpus(seed=True))
    result = engine.evaluate("you are an idiot, my name is Meera Balan",
                             Surface.USER_PROMPT, Tracer(), "s")
    assert "*****" in result.text, "the word rail's masking was lost"
    # vault-token, the configured default — not literal asterisks.
    assert "<PERSON:" in result.text, "the entity rail's masking was lost"
    assert "Meera Balan" not in result.text


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
