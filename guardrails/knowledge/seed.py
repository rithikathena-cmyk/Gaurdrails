"""The knowledge base.

`CORPUS` below is the built-in seed: fifteen public-services documents,
deliberately tiny and deliberately incomplete. The grounding rail only means
something if the model can plausibly reach past what it was given — a knowledge
base that covers everything never produces an ungrounded answer, so it never
exercises the rail you built.

Everything ingested afterwards lives in a `Corpus` (see `ingest.py`), which is
bound here with `use()` at startup. `retrieve()` asks that store when one is
bound and falls back to the seed documents when none is — so the engine has one
retrieval call, whether or not anything has been uploaded.
"""

from __future__ import annotations

import re
from typing import Any

CORPUS: list[dict[str, str]] = [
    {
        "id": "trade-licence-renewal",
        "title": "Trade licence renewal",
        "text": "To renew a trade licence you must submit: the existing licence certificate, "
                "proof of premises (lease agreement or ownership deed), a tax clearance "
                "certificate for the previous financial year, and a completed Form 4B. "
                "Renewal applications open 60 days before expiry.",
    },
    {
        "id": "trade-licence-fees",
        "title": "Trade licence fees",
        "text": "The standard renewal fee for a trade licence is 1,200 rupees for premises "
                "under 500 square feet and 2,400 rupees above that. Payment is accepted "
                "online or at any municipal counter.",
    },
    {
        "id": "housing-grant-eligibility",
        "title": "Housing grant eligibility",
        "text": "The housing assistance grant is available to households with a combined "
                "annual income below 300,000 rupees who have resided in the district for "
                "at least two years. Applicants must not own residential property elsewhere.",
    },
    {
        "id": "housing-grant-documents",
        "title": "Housing grant documents",
        "text": "A housing grant application requires an income certificate issued within "
                "the last six months, proof of residence, and a declaration of property "
                "holdings. Applications are processed within 45 working days.",
    },
    {
        "id": "claim-status",
        "title": "Checking a claim",
        "text": "Claim status can be checked using the claim reference number, which begins "
                "with CLM- followed by eight digits and appears on the acknowledgement "
                "letter sent after submission. Reference numbers are not issued by phone.",
    },
    {
        "id": "birth-certificate",
        "title": "Birth certificate copies",
        "text": "Certified copies of a birth certificate can be requested by the registrant, "
                "a parent, or a legal guardian. Requests require photo identification and "
                "a fee of 100 rupees per copy. Processing takes 7 working days.",
    },
    {
        "id": "property-tax",
        "title": "Property tax assessment",
        "text": "Property tax is assessed annually based on the built-up area and the "
                "locality rate published each April. Objections to an assessment must be "
                "filed within 30 days of the date printed on the notice.",
    },
    {
        "id": "grievance",
        "title": "Filing a grievance",
        "text": "Grievances about service delivery can be filed at any municipal office or "
                "through the public grievance portal. Every grievance receives a tracking "
                "number and a first response within 15 working days.",
    },
    {
        "id": "grievance-escalation",
        "title": "Grievance escalation ladder",
        "text": "If a grievance receives no first response within 15 working days it may be "
                "escalated. Level one is the ward office; level two is the Deputy "
                "Commissioner, reachable at grievances@municipal.gov.in or on the toll-free "
                "helpline 1800 425 1969 between 09:30 and 17:30 on working days. Level three "
                "is the Ombudsman, who accepts a case only after level two has been tried "
                "and the tracking number is quoted.",
    },
    {
        "id": "office-directory",
        "title": "Departmental contacts",
        "text": "Trade licensing is handled by the Revenue wing at licensing@municipal.gov.in. "
                "Housing assistance and grant appeals go to housing@municipal.gov.in. Property "
                "tax objections go to assessment@municipal.gov.in. Birth and death records are "
                "held by the Registrar at records@municipal.gov.in. The general enquiry line is "
                "1800 425 1900. None of these accept payment instructions by email.",
    },
    {
        "id": "appeal-deadlines",
        "title": "Appeal deadlines",
        "text": "A rejected housing grant may be appealed within 30 days of the decision "
                "letter. A property tax assessment may be objected to within 30 days of the "
                "notice. A refused trade licence renewal may be appealed within 21 days. Each "
                "deadline runs from the date printed on the letter, not the date it was "
                "received. A late appeal is accepted only with a written reason for delay.",
    },
{
        "id": "case-file-ha9902",
        "title": "Case file HA-9902 — housing appeal",
        "text": "Housing grant appeal HA-9902. Appellant: Anitha Selvam, contactable on "
                "anitha.selvam@example.com or 9962214477, residing at 14 Anna Salai, "
                "Chennai 600002. Original claim CLM-77310945 was rejected on income "
                "grounds; the decision letter is dated 2026-06-02. The appellant disputes "
                "the income assessment and has supplied a revised income certificate. "
                "Assigned to the Housing wing for review.",
    },
    {
        "id": "grievance-log-q2",
        "title": "Grievance log, second quarter",
        "text": "Open grievances awaiting first response. GRV-3341, raised by Rajesh "
                "Kandasamy, rajesh.k@example.com, 9840055120, about a delayed trade "
                "licence renewal, 22 working days open. GRV-3358, raised by Fatima Sheikh, "
                "f.sheikh@example.com, 9791188342, about a property tax assessment she "
                "believes double-counts a mezzanine floor, 9 working days open. GRV-3362, "
                "raised by Joseph Antony, 9445573310, about a birth certificate copy not "
                "received after payment, 31 working days open and past the escalation "
                "threshold.",
    },
    {
        "id": "officer-caseload",
        "title": "Caseload note — assessment wing",
        "text": "Assessment wing caseload, week 24. Three objections carried forward. "
                "AS-4417, owner Arun Mehta, arun.mehta@example.com, 9884433221, built-up "
                "area 1,240 sq ft, objection lodged on the locality rate. AS-4420, owner "
                "Sundari Ganesan, 9600427718, objection lodged on the built-up area "
                "measurement, site visit requested. AS-4431, owner Vikram Pillai, "
                "vikram.p@example.com, no objection lodged, assessment stands. Officers "
                "must not share an owner's contact details with another owner.",
    },

    {
        "id": "identity-documents",
        "title": "Accepted identity documents",
        "text": "Photo identification means one of: Aadhaar card, PAN card, passport, driving "
                "licence, or voter identity card. A self-attested photocopy is accepted at the "
                "counter; the original must be produced for inspection. Staff will never ask "
                "for an Aadhaar number by email or telephone, and no officer will request a "
                "one-time password.",
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
