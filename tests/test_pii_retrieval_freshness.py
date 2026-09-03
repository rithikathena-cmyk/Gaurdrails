"""Retrieval trusting an already-known-fresh PII classification, when one
exists — the mechanism that used to be fed by ingestion, before the ingest
guardrail (and the classification it did) was removed.

Before this file's own original fix, every retrieval-surface question re-ran
the full `pii.entities` judge scan over the joined retrieved chunks, no
matter how many times that exact text had already been scanned — the seed
RCS charter's own institution names cost a multi-window judge call on
*every single question that retrieved it*, routinely exceeding
`policy.latency_budget_ms` and dropping the retrieved context, which is what
actually broke grounding (see `test_scope_retrieval.py::
test_a_normal_in_corpus_question_survives_retrieval_and_grounds` for the
symptom this was chasing).

The fix: a `Document` can carry a fingerprint of the exact PII config that
last classified it (`pii_policy_version`). `Engine.converse()`'s retrieval
step checks that fingerprint against today's config before scanning again —
a match skips the expensive rail entirely (`pii.detect` still runs, cheap
and deterministic). `Engine.ingest()` no longer stamps this fingerprint
itself — nothing is classified at ingest time any more — so every document
ingested today reads as unknown and falls back to exactly today's full
rescan on its first retrieval, same as any document that predates this
feature always has. The mechanism itself, and its fail-closed default, are
what these tests still cover.
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
    corpus.add(Document(id="d1", title="d1", chunks=["text"]))
    engine = Engine(_policy(), None, AuditLog("audit.log"), corpus)
    assert engine._doc_pii_is_fresh("d1") is False          # noqa: SLF001
    assert engine._doc_pii_is_fresh("does-not-exist") is False  # noqa: SLF001


# ── Test 6: a kind_actions change is picked up without re-ingesting ────
def test_a_kind_actions_change_invalidates_freshness_without_reingesting(tmp_path):
    """ORGANISATION => pass under one policy, ORGANISATION => mask under
    another — the stored classification never gets thrown away or
    recomputed by touching the document, only the *freshness check* changes
    its mind, which is what lets the very next retrieval re-evaluate under
    the new policy. `ingest()` no longer stamps a fingerprint of its own, so
    this seeds the corpus directly with one — exactly the shape a document
    ingested before the ingest guardrail was removed still carries, and the
    only way a fresh-classified document reaches the corpus today."""
    corpus = Corpus(seed=False)
    original_policy = _policy(**{"pii.kind_actions": ["ORGANISATION => pass"]})
    doc_id = "legacy:charter"
    corpus.add(Document(
        id=doc_id, title="Charter", chunks=[DOC_TEXT], status="indexed", verdict="pass",
        pii_policy_version=classification_fingerprint(original_policy),
    ))

    still_same_policy = Engine(original_policy, None, AuditLog(tmp_path / "b.log"), corpus)
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
    assert corpus.get(doc_id).chunks == [DOC_TEXT]

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
