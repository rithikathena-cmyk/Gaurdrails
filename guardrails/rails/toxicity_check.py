"""Toxic-BERT as the cheap layer under the content judge.

The content rail had no deterministic layer at all. Unlike the injection rail,
which settles most traffic on `INJECTION_PATTERNS` for a tenth of a
millisecond, `ContentRail.evaluate()` went straight to the judge — measured at
3.7s inbound and 3.6s outbound on the same request. This module is the layer
that was missing.

**It covers three of the six configured categories.** toxic-bert is trained on
the Jigsaw taxonomy — `toxic, severe_toxic, obscene, threat, insult,
identity_hate` — which maps onto ours like this:

    hate        <- identity_hate
    violence    <- threat
    insults     <- insult
    sexual      <- nothing          (obscene is a profanity head, not this —
                                     see IGNORED_LABELS for the measurement)
    misconduct  <- nothing
    self_harm   <- nothing

The gaps are not an oversight to be papered over with a nearby label.
`misconduct` is "seeks help committing fraud or forgery", which is a question
about intent that no toxicity head is trained to answer — the text is usually
perfectly polite. `self_harm` has no Jigsaw counterpart either, and it is the
category `policy.yaml` sets lowest of the six on purpose:

    self_harm:  {threshold: 0.40}   # low on purpose — a miss costs far more
                                    # than a false positive

So `content.local_short_circuit_scope` is locked: this model may end a request
early only by **blocking** it, never by clearing it, and never for an uncovered
category in either direction. A local score of 0.0 means "this model found
nothing it was trained to find", which is a different claim from "there is
nothing here", and only one of them is safe to act on.

The engine loads on first use, not at import — most requests never need it, and
a deployment that skipped `requirements-local.txt` must still start.
"""

from __future__ import annotations

import logging
from typing import Any

from ._local import LazyModel

log = logging.getLogger("guardrails.rails.toxicity")

MODEL_ID = "unitary/toxic-bert"

#: Jigsaw's vocabulary → ours. A category absent from the values here has no
#: local coverage at all and is reported as unscored rather than as 0.0.
LABEL_MAP: dict[str, str] = {
    "identity_hate": "hate",
    "threat": "violence",
    "insult": "insults",
}

#: `toxic` and `severe_toxic` are deliberately unmapped. They are the dataset's
#: overall-badness heads, not categories in our taxonomy, and folding them into
#: one of ours would let a generically rude sentence breach a specific
#: threshold that was calibrated for something else.
#:
#: `obscene` is unmapped for a measured reason rather than a theoretical one. It
#: looked like the natural source for `sexual`, but on "you are a worthless
#: idiot" — which contains nothing sexual — it returns **0.75**, against a
#: configured `sexual` threshold of 0.60. It is a profanity head, and profanity
#: is a different question from sexual content. The word rail already covers
#: profanity from the lexicon, deterministically and in a tenth of a
#: millisecond, so mapping this would have added a false positive on top of a
#: check that already exists.
IGNORED_LABELS = {"toxic", "severe_toxic", "obscene"}

#: Categories this model cannot speak to. `misconduct` and `self_harm` are
#: locked as a safety invariant in the registry
#: (`content.local_short_circuit_scope`); `sexual` is here because the only
#: candidate label measured badly. All three keep judge coverage.
UNCOVERED: frozenset[str] = frozenset({"misconduct", "self_harm", "sexual"})

#: What the local layer is allowed to decide on its own.
COVERED: frozenset[str] = frozenset(LABEL_MAP.values())

def _build() -> Any:
    import torch
    from transformers import pipeline

    # Rails inside a stage already run concurrently on a thread pool. Left
    # alone, torch grabs every core per call, so N rails x M threads
    # oversubscribe and each one gets slower. One thread here, parallelism from
    # the pool that already exists.
    torch.set_num_threads(1)
    return pipeline(
        "text-classification", model=MODEL_ID,
        top_k=None, device=-1, truncation=True, max_length=512,
    )


_MODEL = LazyModel(f"toxicity classifier ({MODEL_ID})", _build, log)


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
    """The pipeline if it is loaded, else None. Never blocks.

    Returns None while the model is still loading as well as when it is
    missing, so the caller falls through to the judge rather than holding a
    request open behind a disk read. `ContentRail` decides what that gap
    *means* — "no local model" must not become "no check".
    """
    return _MODEL.get()


def warm(timeout: float | None = None) -> Any:
    """Block until loaded. Evaluation and startup warming only."""
    return _MODEL.warm(timeout)


def score(text: str) -> dict[str, float] | None:
    """Per-category scores in our taxonomy, or None if the model did not run.

    Only the categories this model covers appear in the result. A caller
    must not read a missing key as zero — `UNCOVERED` says which are absent by
    construction, and they are absent whether the text is benign or not.
    """
    pipe = classifier()
    if pipe is None or not text.strip():
        return None
    try:
        raw = pipe(text)
    except Exception as exc:  # noqa: BLE001
        # A model that throws is a model that did not answer. Report that
        # honestly; the rail escalates rather than inventing a clean verdict.
        log.warning("toxicity scoring failed: %s", exc)
        return None

    # transformers returns [[{label, score}, ...]] for a single string with
    # top_k=None, but has returned a bare list in older versions.
    rows = raw[0] if raw and isinstance(raw[0], list) else raw
    out: dict[str, float] = {}
    for row in rows or []:
        label = str(row.get("label", "")).lower()
        if label in IGNORED_LABELS:
            continue
        ours = LABEL_MAP.get(label)
        if ours is None:
            continue
        value = float(row.get("score", 0.0))
        # Two Jigsaw labels can map to one of ours; the strongest wins.
        out[ours] = max(out.get(ours, 0.0), min(1.0, max(0.0, value)))
    return out or None


def reset() -> None:
    """Drop the loaded model. Tests only."""
    _MODEL.reset()
