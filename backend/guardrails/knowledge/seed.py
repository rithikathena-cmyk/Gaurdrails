"""The knowledge base.

`CORPUS` below is the built-in seed. It is empty — the seeded demo
documents have been removed — so a fresh `Corpus(seed=True)` starts with
nothing, and `retrieve()`'s own fallback (used only when no `Corpus` is
bound at all) has nothing to search either. The grounding rail only means
something if the model can plausibly reach past what it was given — a
knowledge base that covers everything never produces an ungrounded answer,
so it never exercises the rail you built. An empty one is the far end of
that same idea: every factual question is now ungrounded until something
real is ingested.

Everything ingested afterwards lives in a `Corpus` (see `ingest.py`), which is
bound here with `use()` at startup. `retrieve()` asks that store when one is
bound and falls back to the seed documents when none is — so the engine has one
retrieval call, whether or not anything has been uploaded.
"""

from __future__ import annotations

import re
from typing import Any

CORPUS: list[dict[str, str]] = []

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in",
    "on", "for", "with", "my", "i", "do", "does", "how", "what", "can", "need",
    "you", "your", "me", "it", "this", "that", "be", "have", "has", "at", "from",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


# The ingested store, bound at startup. Module state rather than a parameter
# because retrieval is a property of the deployment, not of a request.
_ACTIVE: Any = None


def use(corpus: Any) -> None:
    """Bind an ingested corpus. Pass None to fall back to the seed documents."""
    global _ACTIVE
    _ACTIVE = corpus


def active() -> Any:
    return _ACTIVE


def retrieve(query: str, k: int = 4, min_score: float = 0.15) -> list[str]:
    """Return at most `k` context chunks, best first.

    A weak match is worse than no match — it gives the grounding rail
    irrelevant context to score against — so `min_score` is a floor on term
    coverage, not a ranking tweak.
    """
    if _ACTIVE is not None:
        return [hit.as_context() for hit in _ACTIVE.search(query, k, min_score)]

    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[float, str]] = []
    for doc in CORPUS:
        d = _tokens(doc["title"] + " " + doc["text"])
        if not d:
            continue
        overlap = len(q & d)
        if overlap == 0:
            continue
        scored.append((overlap / len(q), f"{doc['title']}: {doc['text']}"))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [text for score, text in scored[:k] if score >= min_score]
