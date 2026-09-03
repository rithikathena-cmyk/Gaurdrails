"""The guardrails. Every check that can refuse, mask, or flag something.

The dividing line against `agent/` is what a thing decides:

    rails/   decide about *text*. Each one reads a string and returns a
             verdict. They never call a tool, never take an action, and never
             see each other — the engine runs them concurrently and resolves
             what they said with `precedence()`.

    agent/   decides about *what to do next*. It is the only place a model
             chooses an action rather than scoring one.

    normalize.py     NFKC, invisibles, homoglyphs — locked on, runs first
    words.py         Aho–Corasick over the lexicons
    vault.py         reversible masking — AES-256-GCM, scoped to an owner
    entities.py      the only PII detector: gate → presidio → judge, every
                     kind, judge-only wherever presidio_ner.KIND_MAP has no
                     entry (which is most of them)
    presidio_ner.py  the local NER layer entities.py calls
    policy.py        named regex rule sets
    content.py       content safety, and prompt injection
    scope.py         vocabulary first, judge second
    grounding.py     claim-level consistency and relevance
    adjudicator.py   rules on a verdict the threshold decided by a hair

Deterministic rails come first in that list on purpose: where a family can be
settled without a model, it is, and the cheap layer short-circuits the
expensive one. PII no longer has a deterministic layer at all — see
`entities.py`'s own docstring for why.
"""

from .content import ContentRail, PromptAttackRail
from .entities import EntityRail
from .grounding import GroundingRail
from .normalize import normalize
from .vault import Vault
from .words import WordRail

__all__ = [
    "ContentRail",
    "PromptAttackRail",
    "EntityRail",
    "GroundingRail",
    "normalize",
    "Vault",
    "WordRail",
]
