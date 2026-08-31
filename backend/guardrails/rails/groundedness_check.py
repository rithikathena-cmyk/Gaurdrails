"""NLI as a local consistency check under the grounding judge.

`GroundingRail` scores two things that fail differently and are deliberately
kept apart:

    consistency — is every claim supported by the retrieved context?
    relevance   — does the answer address the question that was asked?

**A natural-language-inference model can only speak to the first.** Entailment
is a relation between a premise and a hypothesis; "does this answer the user's
question" is not in that relation at all. So this module returns a consistency
score and nothing else, and the rail keeps the judge for relevance and for
deciding which claims are actually unsupported. What this model contributes to
that decision is a pre-screen: claims it entails are marked for the judge, which
still reads every one of them and can overrule the mark.

The method is per-claim, matching the rail's existing sentence segmentation:
each sentence of the answer is checked against each retrieved chunk, a claim
counts as supported if any chunk entails it, and consistency is the supported
proportion. That mirrors what `GROUNDING_SYSTEM` asks the judge for, which is
what makes the two scores comparable at all.

Loaded on first use, like every other local model here.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from ._local import LazyModel

log = logging.getLogger("guardrails.rails.groundedness")
diag = logging.getLogger("guardrails.diag.groundedness")

MODEL_ID = "cross-encoder/nli-deberta-v3-base"

#: `_build` pins the process-wide torch intraop pool to 1 thread so this
#: model doesn't oversubscribe alongside the other rails sharing the
#: concurrent prompt-rails thread pool (see `toxicity_check`). But this
#: rail's own inference call runs alone in its own sequential stage, after
#: that pool has already drained — paying the 1-thread tax there is pure
#: waste. Measured on this model's actual shape (6 chunks x ~4 claims,
#: cross-encoder/nli-deberta-v3-base, CPU): 1 thread ~57s, 4 threads ~4.4s,
#: 8+ threads no further gain — so 4 is raised only for the call itself,
#: immediately restored after, rather than left changed process-wide (which
#: would silently undo the oversubscription guard for every other rail for
#: the rest of the process's life — `torch.set_num_threads` is a single
#: global knob, not one scoped per model or per thread).
_INFERENCE_THREADS = min(4, os.cpu_count() or 1)

#: The label that means "the premise supports the hypothesis". The rest —
#: neutral and contradiction — are both "not supported", and the rail does not
#: need to tell them apart: a fee that contradicts the context and a fee that
#: simply is not in it are equally unusable in an answer.
ENTAILMENT_LABELS = {"entailment", "label_1"}

#: Sentences too short to carry a checkable claim. "Thanks for asking." is not
#: a factual assertion, and `GROUNDING_SYSTEM` already tells the judge not to
#: penalise conversational framing — scoring it here would disagree with the
#: judge on the same text.
MIN_CLAIM_CHARS = 25

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

def _build() -> Any:
    import torch
    from transformers import pipeline

    torch.set_num_threads(1)   # see toxicity_check for why
    return pipeline(
        "text-classification", model=MODEL_ID,
        top_k=None, device=-1, truncation=True, max_length=512,
    )


_MODEL = LazyModel(f"groundedness model ({MODEL_ID})", _build, log)


def available() -> bool:
    """Is the runtime importable at all? Cheap — no model weights are loaded.

    `pipeline` is resolved here, on the calling thread, and not only imported.
    transformers exposes it through a lazy module, so the attribute lookup is
    what triggers the submodule import — and doing that first from the loader
    thread races the main thread still importing the package. It surfaces as
    `Could not import module 'pipeline'`, the model is marked failed for the
    life of the process, and every request quietly falls through to the judge.
    Resolving it from whoever builds the engine costs an import at startup and
    removes the race.
    """
    try:
        from transformers import pipeline  # noqa: F401
    except Exception:  # noqa: BLE001 — unavailable is unavailable, however it failed
        return False
    return True


def classifier() -> Any:
    """The NLI pipeline if it is loaded, else None. Never blocks.

    None while loading as well as when missing. The caller falls back to the
    judge; an absent model never becomes a passing grounding score.
    """
    return _MODEL.get()


def warm(timeout: float | None = None) -> Any:
    """Block until loaded. Evaluation and startup warming only."""
    return _MODEL.warm(timeout)


def claims(answer: str) -> list[str]:
    """Sentences from the answer worth checking."""
    return [s.strip() for s in _SENTENCE.split(answer.strip())
            if len(s.strip()) >= MIN_CLAIM_CHARS]


def consistency(answer: str, chunks: list[str]) -> dict[str, Any] | None:
    """Proportion of the answer's claims that some chunk entails.

    Returns None when the model did not run, which the caller must treat as
    "unknown" rather than "grounded". With no claims long enough to check,
    returns a score of 1.0 and an empty claim list — an answer that asserts
    nothing cannot assert anything unsupported.
    """
    pipe = classifier()
    if pipe is None or not chunks:
        return None

    to_check = claims(answer)
    if not to_check:
        return {"consistency": 1.0, "claims": 0, "supported": 0, "unsupported": []}

    pairs = [{"text": chunk, "text_pair": claim}
             for claim in to_check for chunk in chunks]

    import torch
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(_INFERENCE_THREADS)
    t0 = time.perf_counter()
    try:
        raw = pipe(pairs)
    except Exception as exc:  # noqa: BLE001
        log.warning("groundedness scoring failed: %s", exc)
        return None
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        torch.set_num_threads(prev_threads)
        diag.info("groundedness.infer elapsed_ms=%.1f pairs=%d threads=%d",
                  elapsed_ms, len(pairs), _INFERENCE_THREADS)

    per_pair: list[float] = []
    for row in raw or []:
        rows = row if isinstance(row, list) else [row]
        best = 0.0
        for r in rows:
            if str(r.get("label", "")).lower() in ENTAILMENT_LABELS:
                best = max(best, float(r.get("score", 0.0)))
        per_pair.append(best)

    if len(per_pair) != len(to_check) * len(chunks):
        log.warning("groundedness returned %d scores for %d pairs",
                    len(per_pair), len(to_check) * len(chunks))
        return None

    # A claim is supported if *any* chunk entails it — the rail retrieves
    # several and an answer may legitimately draw on one of them.
    n = len(chunks)
    supported, unsupported = 0, []
    for i, claim in enumerate(to_check):
        best = max(per_pair[i * n:(i + 1) * n], default=0.0)
        if best >= 0.5:
            supported += 1
        else:
            unsupported.append(claim)

    return {
        "consistency": supported / len(to_check),
        "claims": len(to_check),
        "supported": supported,
        # The rail marks everything *not* in here as entailed when it numbers
        # the claims for the judge. A wrong entry costs a claim being marked the
        # judge then has to overrule, which is why the prompt tells it to.
        "unsupported": unsupported[:10],
    }


def reset() -> None:
    """Drop the loaded model. Tests only."""
    _MODEL.reset()
