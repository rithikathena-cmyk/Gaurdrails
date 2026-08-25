"""The knowledge base.

`CORPUS` below is the built-in seed — one real document, the RCS Citizen
Charter, read from `seed_documents/` at import time rather than a synthetic
demo set. It exists so a deployment with no persistent disk (this app's
Render free-tier config, notably: `data/corpus.json` resets on every deploy)
still has something real and grounded to answer from the moment it boots —
`Engine.reseed_builtin_rails()` runs it through the same ingest rails a real
upload takes, every startup, so nobody has to re-upload it after a redeploy
just to ask the assistant a question. An empty `CORPUS` (the prior state of
this file) is still the right choice for exercising the grounding rail on
purpose — a knowledge base that covers everything never produces an
ungrounded answer — so keep that in mind before adding a second document
here rather than ingesting it as a real upload instead.

Everything ingested afterwards lives in a `Corpus` (see `ingest.py`), which is
bound here with `use()` at startup. `retrieve()` asks that store when one is
bound and falls back to the seed documents when none is — so the engine has one
retrieval call, whether or not anything has been uploaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SEED_DIR = Path(__file__).resolve().parent / "seed_documents"


def _read(name: str) -> str:
    return (_SEED_DIR / name).read_text(encoding="utf-8")


CORPUS: list[dict[str, str]] = [
    {
        "id": "rcs-citizen-charter-2024-2025",
        "title": "RCS – Citizen Charter 2024-2025 (English Version)",
        "text": _read("rcs-citizen-charter-2024-2025.txt"),
    },
]

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
