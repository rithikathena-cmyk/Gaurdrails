"""The embedding reranker in isolation.

No real model is loaded here — `sentence-transformers` may not even be
installed in the hermetic test environment (`requirements-local.txt` is
optional). `_MODEL.get()` is monkeypatched directly, the same way
`test_pii_agent.py` stubs `presidio_ner.find` rather than loading spaCy for
every test that merely needs to know the wiring around a local model is
correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.guardrails.knowledge.ingest import Hit
from backend.guardrails.rails import embedding_rerank


@pytest.fixture(autouse=True)
def clear_cache():
    embedding_rerank.reset()
    yield
    embedding_rerank.reset()


def _hit(title: str, text: str) -> Hit:
    return Hit(doc_id=title.lower(), title=title, chunk_index=0, text=text,
              score=1.0, coverage=1.0)


class _StubModel:
    """`.encode()` returns a hand-picked vector per text, keyed by a
    substring — deterministic and legible in a failing assertion, unlike a
    real embedding."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        # Real sentence-transformers returns a numpy array per text — `rerank()`
        # uses `@` (matrix-multiply), which plain Python lists don't support.
        return [np.array(self.vectors[t]) for t in texts]


def test_rerank_returns_none_while_the_model_is_still_loading(monkeypatch):
    monkeypatch.setattr(embedding_rerank._MODEL, "get", lambda: None)
    assert embedding_rerank.rerank("query", [_hit("A", "a")], top_k=4) is None


def test_rerank_reorders_hits_by_similarity_to_the_query(monkeypatch):
    hits = [_hit("Far", "far text"), _hit("Near", "near text")]
    vectors = {
        "query": [1.0, 0.0],
        "far text": [0.0, 1.0],   # orthogonal — similarity 0
        "near text": [1.0, 0.0],  # identical — similarity 1
    }
    monkeypatch.setattr(embedding_rerank._MODEL, "get", lambda: _StubModel(vectors))

    ranked = embedding_rerank.rerank("query", hits, top_k=4)
    assert [h.title for h in ranked] == ["Near", "Far"]


def test_rerank_respects_top_k(monkeypatch):
    hits = [_hit("A", "a"), _hit("B", "b"), _hit("C", "c")]
    vectors = {"query": [1.0], "a": [1.0], "b": [0.9], "c": [0.1]}
    monkeypatch.setattr(embedding_rerank._MODEL, "get", lambda: _StubModel(vectors))

    ranked = embedding_rerank.rerank("query", hits, top_k=2)
    assert len(ranked) == 2
    assert [h.title for h in ranked] == ["A", "B"]


def test_rerank_caches_so_a_repeated_chunk_is_not_re_encoded(monkeypatch):
    hits = [_hit("A", "shared text")]
    calls: list[list[str]] = []

    class _CountingModel(_StubModel):
        def encode(self, texts, normalize_embeddings=True):
            calls.append(list(texts))
            return super().encode(texts, normalize_embeddings)

    model = _CountingModel({"query": [1.0], "shared text": [1.0]})
    monkeypatch.setattr(embedding_rerank._MODEL, "get", lambda: model)

    embedding_rerank.rerank("query", hits, top_k=1)
    embedding_rerank.rerank("query", hits, top_k=1)

    encoded_total = sum(len(c) for c in calls)
    assert encoded_total == 2, "the second call should hit the cache, not re-encode"


def test_rerank_on_empty_hits_returns_empty_without_touching_the_model(monkeypatch):
    monkeypatch.setattr(embedding_rerank._MODEL, "get",
                        lambda: (_ for _ in ()).throw(AssertionError("model should not load")))
    assert embedding_rerank.rerank("query", [], top_k=4) == []
