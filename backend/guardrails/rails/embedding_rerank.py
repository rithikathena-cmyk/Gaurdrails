"""A local embedding model, reranking BM25's own candidates.

Retrieval has no semantic layer at all — `Corpus.search()` is lexical, term-
matching BM25. That is precise and blind the same way `pii.detect` is: it
finds "trade licence renewal" because the query shares those words, and it
cannot find the same document from "renew my licence", because a paraphrase
has no term overlap for BM25 to score.

This module does not replace BM25 or touch `Corpus`/its index. It reranks
whatever BM25 already returned — the model never sees the whole corpus, only
`retrieval.embedding_candidates` candidates BM25 already thought were
plausible. That keeps this additive rather than a second retrieval system:
a corpus BM25 finds nothing relevant in still has nothing relevant to
reorder.

The model ID is a fixed constant, not a registry parameter, deliberately —
see the module docstring on `_CACHE` below for why a configurable model here
would be a real correctness bug, not just an inconsistency with the other
three local models (`toxicity_check.py`, `deberta_injection_check.py`,
`groundedness_check.py`) that already fix theirs.

Loads on first use, not at import — same as every other local model here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ._local import LazyModel

if TYPE_CHECKING:
    from ..knowledge.ingest import Hit

log = logging.getLogger("guardrails.rails.embedding_rerank")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: Text -> embedding vector. Keyed by text, not by (model, text), because the
#: model is a fixed constant (see module docstring) — if it ever became
#: configurable, a mode switch without clearing this would silently mix
#: vectors from two different embedding spaces into one similarity ranking,
#: and every score after that would be comparing incompatible numbers while
#: looking completely normal.
_CACHE: dict[str, Any] = {}


def _build() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    # Same reasoning as every other local model here: rails inside a stage
    # already run concurrently on a thread pool, so one thread per model call
    # avoids oversubscription rather than fighting the pool for cores.
    torch.set_num_threads(1)
    return SentenceTransformer(MODEL_ID, device="cpu")


_MODEL = LazyModel(f"embedding reranker ({MODEL_ID})", _build, log)


def available() -> bool:
    """Is the runtime importable at all? Cheap — no model weights are loaded."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:  # noqa: BLE001 — unavailable is unavailable, however it failed
        return False
    return True


def warm(timeout: float | None = None) -> Any:
    """Block until loaded. Evaluation and startup warming only."""
    return _MODEL.warm(timeout)


def rerank(query: str, hits: list["Hit"], top_k: int) -> list["Hit"] | None:
    """BM25's own candidates, reordered by embedding similarity to `query`.

    None if the model has not finished loading — the caller keeps BM25's own
    order, the same "no local answer yet" contract every other local model in
    this codebase already has (`_local.py`'s own docstring: "the local model
    did not answer" and "the local model is not installed" want the same
    behaviour). Retrieval is never worse than plain BM25 this way, only
    sometimes better once the model is warm.
    """
    if not hits:
        return hits
    model = _MODEL.get()
    if model is None:
        return None

    texts = [h.text for h in hits]
    missing = [t for t in {*texts, query} if t not in _CACHE]
    if missing:
        vectors = model.encode(missing, normalize_embeddings=True)
        for text, vector in zip(missing, vectors):
            _CACHE[text] = vector

    q_vec = _CACHE[query]
    ranked = sorted(hits, key=lambda h: -float(_CACHE[h.text] @ q_vec))
    return ranked[:top_k]


def reset() -> None:
    """Drop the loaded model and the vector cache. Tests only."""
    _MODEL.reset()
    _CACHE.clear()
