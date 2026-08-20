"""Turning rail results into something a person can act on.

Telling a user what tripped is genuinely useful — a masked SSN they didn't
realise they'd sent, a term they can rephrase. But it is also a feedback
channel: "blocked because it matched an instruction-override pattern" tells an
attacker exactly which phrasing to vary next. So disclosure is graduated, and
one level is capped regardless of configuration.

  detailed  names the category, the entity types, the matched terms
  category  names the family and category, never the specific match  (default)
  minimal   says something was stopped and what to do, nothing about what
  none      no explanation at all

`prompt_attack` never exceeds `category` no matter what the level is set to —
see `policy.disclosure.injection_cap` in the registry. That is a security
boundary, not a preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .types import RailResult, Verdict

LEVELS = ["none", "minimal", "category", "detailed"]


def _rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else LEVELS.index("category")


# Human labels. The rail speaks in entity codes; a citizen should not have to.
ENTITY_LABELS = {
    "US_SSN": "Social Security number",
    "CREDIT_CARD": "payment card number",
    "EMAIL_ADDRESS": "email address",
    "PHONE_NUMBER": "phone number",
    "AADHAAR": "Aadhaar number",
    "PAN": "PAN",
    "IBAN": "bank account number",
    "IP_ADDRESS": "IP address",
    "DATE_OF_BIRTH": "date of birth",
}

CATEGORY_LABELS = {
    "hate": "hateful content",
    "violence": "violence or threats",
    "insults": "abusive language",
    "misconduct": "help with unlawful activity",
    "self_harm": "self-harm",
    "sexual": "sexual content",
}

# A refusal is the wrong place to leave someone in crisis with nothing.
SELF_HARM_SUPPORT = (
    "If you are struggling, please talk to someone — a local crisis line, your "
    "doctor, or someone you trust. This service can't help with this, but people can."
)


@dataclass
class Violation:
    """One thing the user should know about, already disclosure-filtered."""

    family: str          # pii | words | content | injection | grounding | policy
    rail: str
    verdict: str
    title: str
    detail: str
    items: list[str] = field(default_factory=list)
    action_required: bool = False   # can the user fix this by rephrasing?

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "rail": self.rail,
            "verdict": self.verdict,
            "title": self.title,
            "detail": self.detail,
            "items": self.items,
            "action_required": self.action_required,
        }


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
class _Origin(NamedTuple):
    """How to describe a surface to the person reading the reply.

    Three phrasings per surface because the disclosure ladder gives three
    lengths, and a shorter one must not become a vaguer *and* wronger one.
    """

    title: str
    detailed: str
    category: str
    minimal: str


_ORIGIN: dict[str, _Origin] = {
    "prompt": _Origin(
        "Personal details removed",
        "before your message reached the assistant. You don't need to resend them.",
        "from your message before it reached the assistant.",
        "from your message automatically.",
    ),
    "reply": _Origin(
        "Personal details removed from the reply",
        "from the reply before it was shown to you. Nothing you sent caused this.",
        "from the reply before it was shown to you.",
        "from the reply automatically.",
    ),
    "retrieved": _Origin(
        "Personal details removed from the sources",
        "from the documents this answer drew on. They belong to other people, "
        "and the assistant never saw them.",
        "from the documents this answer drew on, before the assistant read them.",
        "from the retrieved documents automatically.",
    ),
    "document": _Origin(
        "Personal details removed from the document",
        "from the document before it was stored. It is indexed without them.",
        "from the document before it was stored.",
        "from the document automatically.",
    ),
}


def _pii(r: RailResult, level: int, origin: str = "prompt") -> Violation | None:
    """Say what was masked, and — crucially — *where*.

    This used to word every PII violation as "in your message ... before it
    reached the assistant", whatever surface it came from. On the agent path,
    where violations are built from the response, that told a citizen she had
    sent a claim reference and a phone number when she had sent neither: both
    were masked in the assistant's own reply. Telling someone they disclosed
    something they did not is the wrong mistake for a tool whose whole job is
    reporting accurately what happened to their data.
    """
    if r.verdict is Verdict.PASS or not r.detections:
        return None
    kinds = [d.kind for d in r.detections]
    labels = sorted({ENTITY_LABELS.get(k, "a restricted identifier") for k in kinds})
    masked = r.verdict is Verdict.MASK
    n = len(r.detections)
    plural = n != 1
    where = _ORIGIN[origin] if origin in _ORIGIN else _ORIGIN["prompt"]

    if level >= _rank("detailed"):
        detail = (
            f"We found and removed {_join(labels)} {where.detailed}"
            if masked else
            f"Your message contains {_join(labels)}, which this channel can't accept."
        )
        items = labels
    elif level >= _rank("category"):
        detail = (
            f"{n} sensitive value{'s' if plural else ''} "
            f"{'were' if plural else 'was'} removed {where.category}"
            if masked else
            "Your message contains sensitive personal details this channel can't accept."
        )
        items = labels if masked else []
    else:
        detail = (f"Sensitive details were removed {where.minimal}"
                  if masked else "Your message contains details this channel can't accept.")
        items = []

    return Violation(
        family="pii", rail=r.rail, verdict=r.verdict.value,
        title=where.title if masked else "Personal details not accepted",
        detail=detail, items=items,
        # Only something the user actually sent is something they can rephrase.
        # Nothing they do changes what a retrieved document or a reply contained.
        action_required=not masked and origin == "prompt",
    )


def _words(r: RailResult, level: int) -> Violation | None:
    if r.verdict is Verdict.PASS or not r.detections:
        return None
    terms = sorted({d.value for d in r.detections})
    if level >= _rank("detailed"):
        detail = f"Your message contains language this service can't process: {_join(terms)}."
        items = terms
    elif level >= _rank("category"):
        n = len(terms)
        detail = (f"Your message contains {n} term{'s' if n != 1 else ''} this service "
                  "can't process. Please rephrase.")
        items = []
    else:
        detail = "Your message contains language this service can't process."
        items = []

    return Violation(
        family="words", rail=r.rail, verdict=r.verdict.value,
        title="Restricted language", detail=detail, items=items, action_required=True,
    )


def _content(r: RailResult, level: int) -> Violation | None:
    if r.verdict is Verdict.PASS:
        return None
    breached = list(r.meta.get("breached") or [])
    labels = [CATEGORY_LABELS.get(c, c) for c in breached]

    if "self_harm" in breached:
        return Violation(
            family="content", rail=r.rail, verdict=r.verdict.value,
            title="We can't help with this",
            detail=SELF_HARM_SUPPORT,
            items=[], action_required=False,
        )

    if level >= _rank("category") and labels:
        detail = (f"This request appears to involve {_join(labels)}, which this "
                  "service can't help with.")
        items = labels if level >= _rank("detailed") else []
    else:
        detail = "This request falls outside what this service can help with."
        items = []

    return Violation(
        family="content", rail=r.rail, verdict=r.verdict.value,
        title="Outside this service's scope", detail=detail, items=items,
        action_required=True,
    )


def _injection(r: RailResult, level: int) -> Violation | None:
    """Capped at `category`. Naming the matched technique teaches probing."""
    if r.verdict is Verdict.PASS:
        return None
    capped = min(level, _rank("category"))
    detail = (
        "This request was stopped by a security check. If you were asking a genuine "
        "question about this service, please rephrase it plainly and try again."
        if capped >= _rank("category") else
        "This request couldn't be processed."
    )
    return Violation(
        family="injection", rail=r.rail, verdict=r.verdict.value,
        title="Blocked by a security check", detail=detail,
        items=[],  # never itemised, at any level
        action_required=True,
    )


def _policy(r: RailResult, level: int) -> Violation | None:
    if r.verdict is Verdict.PASS or not r.detections:
        return None
    sets = sorted({d.kind.replace("policy.", "").replace("_rules", "")
                   for d in r.detections})
    if level >= _rank("detailed"):
        detail = f"Your message matched a {_join(sets)} policy rule."
        items = sets
    elif level >= _rank("category"):
        detail = "Your message matched an organisation policy rule."
        items = []
    else:
        detail = "Your message couldn't be processed."
        items = []
    return Violation(
        family="policy", rail=r.rail, verdict=r.verdict.value,
        title="Policy rule", detail=detail, items=items, action_required=True,
    )


def _grounding(r: RailResult, level: int) -> Violation | None:
    """Not the user's fault. Say so plainly rather than implying they did wrong."""
    if r.verdict is Verdict.PASS:
        return None
    failed_on = r.meta.get("failed_on", "consistency")
    detail = {
        "consistency": "I couldn't confirm parts of that answer against the source "
                       "material, so I haven't given it to you. Nothing you did caused this.",
        "relevance": "The answer I produced didn't actually address your question, so "
                     "I haven't given it to you.",
        "citations": "I couldn't point to a source for that answer, so I haven't "
                     "given it to you.",
    }.get(failed_on, "I couldn't verify that answer against the source material.")
    if level < _rank("category"):
        detail = "I couldn't produce an answer I can stand behind."
    return Violation(
        family="grounding", rail=r.rail, verdict=r.verdict.value,
        title="Answer not verified", detail=detail, items=[], action_required=False,
    )


def _adjudicator(r: RailResult, level: int) -> Violation | None:
    """The reviewed verdict, in the user's words.

    Only speaks when the adjudicator *raised* the verdict — an upheld or lowered
    one is already explained by whichever rail actually fired. Its rationale is
    written for an operator reading the audit log, so it is never shown verbatim
    below `detailed`.
    """
    if r.verdict is not Verdict.BLOCK or r.meta.get("direction") != "raised":
        return None
    detail = ("This request was reviewed and can't be helped with here.")
    items: list[str] = []
    if level >= _rank("detailed"):
        detail = ("Automated checks were inconclusive, so this request was reviewed "
                  "in full and declined.")
        items = [str(r.meta.get("rationale") or "")] if r.meta.get("rationale") else []
    return Violation(
        family="content", rail=r.rail, verdict=r.verdict.value,
        title="We can't help with this",
        detail=detail, items=items, action_required=False,
    )


def _scope(r: RailResult, level: int) -> Violation | None:
    """An off-topic question is not an incident, and must not read like one.

    Without this the scope rail fell through to the generic refusal — "that
    request was stopped before it reached the model" — which tells somebody
    asking about prime numbers that they have done something wrong. They have
    not; they are in the wrong place, and the useful thing is to say which
    place is the right one.
    """
    if r.verdict is not Verdict.BLOCK:
        return None
    detail = ("This desk handles council services — benefits, licensing, housing, "
              "tax, civil records, and the paperwork around them. Ask about one of "
              "those and it can look the answer up.")
    items: list[str] = []
    if level >= _rank("detailed"):
        topic = str(r.meta.get("topic") or "").strip()
        if topic:
            items = [f"read as: {topic}"]
    return Violation(
        family="scope", rail=r.rail, verdict=r.verdict.value,
        title="That is outside what this service covers",
        detail=detail, items=items, action_required=False,
    )


_BUILDERS = {
    "pii.detect": _pii,
    "words.lexicon": _words,
    "content.safety": _content,
    "prompt_attack": _injection,
    "policy.rules": _policy,
    "grounding.consistency": _grounding,
    "adjudicator.review": _adjudicator,
    "scope.domain": _scope,
}


def explain(rails: list[RailResult], level: str = "category",
            origin: str = "prompt") -> list[Violation]:
    """Build user-facing violations from rail results, filtered by disclosure.

    `origin` is the surface these results came from. It decides how the copy
    describes what happened — a value masked in the reply, or in a retrieved
    document, is not something the reader typed.
    """
    if level == "none":
        return []
    rank = _rank(level)
    out: list[Violation] = []
    for r in rails:
        builder = _BUILDERS.get(r.rail)
        if builder is None:
            continue
        try:
            v = builder(r, rank, origin)
        except TypeError:
            # Builders that do not vary by surface keep their two-argument form.
            v = builder(r, rank)
        if v is not None:
            out.append(v)
    # Most restrictive first — that is the one the user most needs to read.
    order = {"block": 0, "mask": 1, "flag": 2, "pass": 3}
    out.sort(key=lambda v: order.get(v.verdict, 9))
    return out


def summarise(violations: list[Violation], request_id: str, blocked: bool) -> str:
    """A short message to show in place of, or above, the reply."""
    if not violations:
        return ""
    lines = [v.detail for v in violations]
    body = " ".join(lines)
    if blocked:
        return f"{body}\n\nReference {request_id}."
    return body
