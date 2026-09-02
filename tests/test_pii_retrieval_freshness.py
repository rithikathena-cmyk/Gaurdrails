"""Ingestion-time PII classification, trusted at retrieval — the change that
actually fixes the latency/grounding failure.

Before this file's own fix, every retrieval-surface question re-ran the full
`pii.entities` judge scan over the joined retrieved chunks, no matter how
many times that exact text had already been scanned — the seed RCS charter's
own institution names cost a multi-window judge call on *every single
question that retrieved it*, routinely exceeding `policy.latency_budget_ms`
and dropping the retrieved context, which is what actually broke grounding
(see `test_scope_retrieval.py::
test_a_normal_in_corpus_question_survives_retrieval_and_grounds` for the
symptom this was chasing).

The fix: `Engine.ingest()` classifies once, on the whole document, before it
is ever chunked (`ingest.mask_before_index`, already locked and unchanged);
it now also stamps the resulting `Document` with a fingerprint of the exact
PII config that classification used. `Engine.converse()`'s retrieval step
checks that fingerprint against today's config before scanning again — a
match skips the expensive rail entirely (`pii.detect` still runs, cheap and
deterministic); a document ingested before this existed, or ingested under a
policy that has since changed in a way that matters, reads as unknown and
falls back to exactly today's full rescan, unchanged.
"""

from __future__ import annotations

import re

from backend.guardrails import AuditLog, Corpus, Document, Engine, load
from backend.guardrails.llm import Generation
from backend.guardrails.rails.kind_actions import classification_fingerprint
from backend.guardrails.types import Verdict
from tests.conftest import REPO


class CountingClaude:
    """Every judge call is answered generically; `entity_schema_calls` counts
    only the ones shaped like `entities.py`'s `ENTITY_SCHEMA` — its single
    top-level property `entities` is not shared by any other schema in this
    codebase, so it isolates exactly the call this fix is about."""

    model = "stub"

    def __init__(self, entities=(), reply="A grounded answer [1]."):
        self.entities = list(entities)
        self.entity_schema_calls = 0
        self.reply = reply

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        props = set(schema.get("properties", {}))
        if props == {"entities"}:
            self.entity_schema_calls += 1
            return {"entities": self.entities}
        if "consistency" in props:
            return {"consistency": 0.95, "relevance": 0.95, "unsupported": [],
                    "rationale": "stub"}
        if "in_scope" in props:
            return {"in_scope": 0.95, "topic": "stub", "rationale": "stub"}
        if "injection" in props:
            return {"injection": 0.0, "technique": "none", "rationale": "stub"}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}

    def generate(self, system, messages, *, max_tokens=4096, model=None):
        return Generation(text=self.reply, model=self.model)


DOC_TEXT = (
    "Registrar of Cooperative Societies — Citizen Charter\n\n"
    "The Tamil Nadu State Apex Cooperative Bank administers short-term credit "
    "across the state through District Central Cooperative Banks."
)


def _policy(**overrides):
    p = load(REPO / "config" / "policy.yaml")
    p.values.update(overrides)
    return p


# ── Test 4: an already-processed chunk skips the expensive judge ───────
def test_an_already_ingested_chunk_does_not_invoke_the_judge_on_retrieval(tmp_path):
    corpus = Corpus(seed=False)
    ingest_llm = CountingClaude()
    ingest_engine = Engine(_policy(), ingest_llm, AuditLog(tmp_path / "a.log"), corpus)
    result = ingest_engine.ingest("RCS Citizen Charter", DOC_TEXT)
    assert result.document.status == "indexed"
    assert result.document.pii_policy_version, "ingest must stamp a fingerprint"

    # A fresh Engine, same corpus, same policy — the point is that *this*
    # engine's own judge is never asked to classify text it did not ingest.
    # Deliberately no capitalised entity-looking phrase in the *question*
    # itself — `pii.entities` also runs on `user.prompt` (correctly, and
    # unaffected by this fix), and a capitalised question would call the
    # judge there regardless of what retrieval does, which is a different
    # call this test is not about.
    query_llm = CountingClaude()
    query_engine = Engine(_policy(), query_llm, AuditLog(tmp_path / "b.log"), corpus)
    res = query_engine.converse("what does the state cooperative bank do?")

    assert res.blocked is False
    assert res.chunks, "retrieved context must have survived"
    assert query_llm.entity_schema_calls == 0, (
        "a chunk from an already-classified document must not re-invoke pii.entities' judge"
    )


# ── Test 5: an unprocessed / uncertain chunk falls back ─────────────────
def test_an_unprocessed_chunk_still_invokes_the_deterministic_fallback(tmp_path):
    """A document seeded directly into the corpus — the shape every document
    ingested before this fix, or ingested with no `Engine` at all, actually
    has: no `pii_policy_version` at all. Retrieval must fall back to exactly
    today's full scan rather than silently trusting an unknown chunk."""
    corpus = Corpus(seed=False)
    corpus.add(Document(
        id="legacy:doc", title="Legacy Document", source="test", kind="txt",
        chars=len(DOC_TEXT), chunks=[DOC_TEXT], status="indexed", verdict="pass",
        rails_applied=False,           # never classified by any Engine
    ))
    llm = CountingClaude(entities=[
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "GOVERNMENT",
         "confidence": 0.9},
    ])
    engine = Engine(_policy(), llm, AuditLog(tmp_path / "a.log"), corpus)
    res = engine.converse("Tell me about the Tamil Nadu State Apex Cooperative Bank.")

    assert res.blocked is False
    assert llm.entity_schema_calls >= 1, (
        "an unprocessed document's chunk must still be scanned on retrieval"
    )


def test_freshness_is_false_for_a_document_with_no_fingerprint():
    """Direct unit check on the predicate itself, isolated from a full
    `converse()` turn — a corpus edited concurrently with a search, or a doc
    missing entirely, must read as unknown, never as fresh."""
    corpus = Corpus(seed=False)
    corpus.add(Document(id="d1", title="d1", chunks=["text"], rails_applied=False))
    engine = Engine(_policy(), None, AuditLog("audit.log"), corpus)
    assert engine._doc_pii_is_fresh("d1") is False          # noqa: SLF001
    assert engine._doc_pii_is_fresh("does-not-exist") is False  # noqa: SLF001


# ── Test 6: a kind_actions change is picked up without re-ingesting ────
def test_a_kind_actions_change_invalidates_freshness_without_reingesting(tmp_path):
    """ORGANISATION => pass at ingest, ORGANISATION => mask afterward — the
    stored classification never gets thrown away or recomputed by touching
    the document, only the *freshness check* changes its mind, which is what
    lets the very next retrieval re-evaluate under the new policy."""
    corpus = Corpus(seed=False)
    ingest_engine = Engine(
        _policy(**{"pii.kind_actions": ["ORGANISATION => pass"]}),
        CountingClaude(), AuditLog(tmp_path / "a.log"), corpus,
    )
    result = ingest_engine.ingest("Charter", DOC_TEXT)
    doc_id = result.document.id

    still_same_policy = Engine(
        _policy(**{"pii.kind_actions": ["ORGANISATION => pass"]}),
        None, AuditLog(tmp_path / "b.log"), corpus,
    )
    assert still_same_policy._doc_pii_is_fresh(doc_id) is True  # noqa: SLF001

    changed_policy = Engine(
        _policy(**{"pii.kind_actions": ["ORGANISATION => mask"]}),
        None, AuditLog(tmp_path / "c.log"), corpus,
    )
    assert changed_policy._doc_pii_is_fresh(doc_id) is False, (  # noqa: SLF001
        "changing what ORGANISATION resolves to must invalidate freshness — "
        "the already-baked chunk no longer reflects the current policy"
    )

    # The document itself, and its chunks, were never touched.
    assert corpus.get(doc_id).chunks == result.document.chunks

    # And retrieval genuinely re-evaluates rather than silently keeping the
    # old decision: with the changed policy, the judge is asked again.
    query_llm = CountingClaude(entities=[
        {"text": "Tamil Nadu State Apex Cooperative Bank", "kind": "GOVERNMENT",
         "confidence": 0.9},
    ])
    query_engine = Engine(
        _policy(**{"pii.kind_actions": ["ORGANISATION => mask"]}),
        query_llm, AuditLog(tmp_path / "d.log"), corpus,
    )
    query_engine.converse("Tell me about the Tamil Nadu State Apex Cooperative Bank.")
    assert query_llm.entity_schema_calls >= 1, (
        "a stale document's chunk must be rescanned, not served from an outdated bake"
    )


def test_the_fingerprint_is_stable_for_an_unchanged_policy():
    a = classification_fingerprint(_policy())
    b = classification_fingerprint(_policy())
    assert a == b


def test_the_fingerprint_changes_when_entity_kinds_changes():
    a = classification_fingerprint(_policy())
    b = classification_fingerprint(_policy(**{"pii.entity_kinds": ["PERSON"]}))
    assert a != b


def test_the_fingerprint_changes_when_the_mask_strategy_changes():
    """A masked span's *rendering* is baked into the chunk too — a document
    ingested under vault-token no longer matches today's policy the moment
    an admin flips to redact, the same reason a kind_actions edit already
    invalidates freshness."""
    a = classification_fingerprint(_policy())
    b = classification_fingerprint(_policy(**{"pii.mask_strategy": "redact"}))
    c = classification_fingerprint(_policy(**{"pii.kind_mask_strategy":
                                              ["EMAIL_ADDRESS => redact"]}))
    assert a != b
    assert a != c
