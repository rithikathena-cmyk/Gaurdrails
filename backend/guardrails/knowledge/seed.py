"""The knowledge base.

`CORPUS` below is the built-in seed: thirty-six public-services documents,
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
    # Three more individual case records, the same shape as HA-9902 above and
    # for the same reason: a record naming one resident is exactly what the
    # retrieval-surface PII rail exists to mask — reversibly, for the wing
    # that owns it, and not for whoever happens to ask. `grievance-log-q2`
    # and `officer-caseload` below are the negative case: PII belongs on a
    # record like this, never aggregated into a log a single search reaches.
    {
        "id": "case-file-tl2214",
        "title": "Case file TL-2214 — trade licence objection",
        "text": "Trade licence objection TL-2214. Applicant: Karthik Subramaniam, "
                "contactable on karthik.subramaniam@example.com or 9884032156, trading "
                "as Subramaniam Hardware at 7 Kamarajar Street, Chennai 600008. Renewal "
                "was refused on 2026-05-20 for an unpaid tax-clearance certificate; the "
                "applicant has since submitted one and disputes the refusal date. "
                "Assigned to the Revenue wing for review.",
    },
    {
        "id": "case-file-pt5583",
        "title": "Case file PT-5583 — property tax objection",
        "text": "Property tax objection PT-5583, against assessment AS-4417. Objector: "
                "Priya Natarajan, contactable on priya.natarajan@example.com or "
                "9791145520, residing at 22 Lloyds Road, Chennai 600086. The objector "
                "disputes the built-up area used in the assessment and has requested a "
                "site visit. Filed 2026-06-11. Assigned to the assessment wing.",
    },
    {
        "id": "case-file-bc7731",
        "title": "Case file BC-7731 — birth certificate correction",
        "text": "Birth certificate correction request BC-7731. Applicant: Suresh Iyer, "
                "contactable on suresh.iyer@example.com or 9962287743, residing at 41 "
                "Bazaar Road, Chennai 600041. Requests correction of a misspelled given "
                "name on the certificate of his daughter, born 2024-03-02. Supporting "
                "hospital discharge summary attached. Assigned to the Registrar.",
    },
    {
        "id": "grievance-log-q2",
        "title": "Grievance log, second quarter",
        # Cases by reference, not by person. A log is a *bulk* document: one
        # ordinary search reaches all of it, so holding three unrelated
        # residents' names, emails and mobile numbers here put them one question
        # away from each other. The retrieval rail masked them every time, which
        # is the right safety net and the wrong thing to depend on — the fix is
        # for the aggregate not to exist. Contact details belong on the
        # individual case record, which is what `case-file-ha9902` models.
        "text": "Open grievances awaiting first response. GRV-3341, a delayed trade "
                "licence renewal, 22 working days open. GRV-3358, a property tax "
                "assessment the objector believes double-counts a mezzanine floor, "
                "9 working days open. GRV-3362, a birth certificate copy not received "
                "after payment, 31 working days open and past the escalation threshold. "
                "Contact details for each grievance are held on its case record and are "
                "not reproduced in this log.",
    },
    {
        "id": "officer-caseload",
        "title": "Caseload note — assessment wing",
        # This one stated the rule in its last sentence and broke it in the
        # three before: a single searchable note carrying three owners' names,
        # emails and mobile numbers is the sharing it forbids, just done in
        # advance. Assessments by reference now, so the closing rule is one the
        # document itself keeps.
        "text": "Assessment wing caseload, week 24. Three objections carried forward. "
                "AS-4417, built-up area 1,240 sq ft, objection lodged on the locality "
                "rate. AS-4420, objection lodged on the built-up area measurement, site "
                "visit requested. AS-4431, no objection lodged, assessment stands. Owner "
                "details are held on the individual assessment record. Officers must not "
                "share an owner's contact details with another owner.",
    },

{
        "id": "death-certificate",
        "title": "Death certificate",
        "text": "A death must be registered within 21 days at the office for the place "
                "it occurred, not the place of residence. The informant is the head of "
                "the household, or the hospital where the death occurred. Certified "
                "copies cost 100 rupees each and take 7 working days. Copies may be "
                "requested by a spouse, child, parent, or the executor of the estate, on "
                "production of photo identification and proof of relationship. A late "
                "registration beyond 21 days needs an affidavit and the approval of the "
                "Registrar.",
    },
    {
        "id": "survivor-benefit",
        "title": "Survivor benefit after a death",
        "text": "When a pensioner dies, the pension stops at the end of that month and "
                "the survivor benefit must be claimed separately — it is not automatic. "
                "A surviving spouse may claim within 12 months of the death. The claim "
                "needs the death certificate, proof of marriage, the pensioner's "
                "reference number, and bank details in the survivor's own name. "
                "Applications are processed within 45 working days. Where the household "
                "has no income in the meantime, an interim payment may be requested at "
                "any municipal office and is decided within 10 working days.",
    },
    {
        "id": "trade-licence-new",
        "title": "A new trade licence",
        "text": "Applying for a trade licence for the first time is a separate process "
                "from extending one you already hold. It requires proof of premises, an "
                "identity document for each partner or director, a site sketch, and "
                "clearance from the health wing where food is prepared or sold. The fee "
                "follows the standard schedule — 1,200 rupees under 500 square feet, "
                "2,400 rupees above — plus a one-time registration charge of 500 rupees. "
                "A first application takes 30 working days; trading before the licence is "
                "issued is an offence.",
    },
    {
        "id": "water-connection",
        "title": "Water and sewerage connection",
        "text": "A new domestic water connection requires proof of ownership or a "
                "no-objection letter from the owner, the property tax assessment number, "
                "and a refundable deposit of 3,000 rupees. Sewerage is connected at the "
                "same time where a main is available within 30 metres. Work is scheduled "
                "within 21 working days of payment. A connection cannot be issued while "
                "property tax on the premises is in arrears.",
    },
    {
        "id": "building-permit",
        "title": "Building plan approval",
        "text": "Construction, extension, or a change of use needs plan approval before "
                "work begins. Submit the site plan, the building plan signed by a "
                "registered architect, proof of ownership, and the current property tax "
                "receipt. Plans up to 300 square metres are decided within 30 working "
                "days; anything larger goes to the technical committee and takes 60. "
                "Approval lapses if work has not started within two years.",
    },
    {
        "id": "marriage-registration",
        "title": "Marriage registration",
        "text": "A marriage is registered at the office for the place it was solemnised "
                "or where either party has lived for at least six months. Both parties "
                "attend with photo identification, proof of date of birth, and three "
                "witnesses. The fee is 200 rupees, and the certificate is issued in 15 "
                "working days. Registration after 90 days of the ceremony needs the "
                "Registrar's written permission.",
    },
    {
        "id": "payments-and-refunds",
        "title": "Paying, and getting money back",
        "text": "Fees are payable online, at any municipal counter, or by demand draft. "
                "Cash is accepted only at a counter and only against a printed receipt — "
                "no officer collects cash anywhere else. A payment made twice, or for a "
                "service later refused by the department, is refundable on written "
                "application within 90 days, and is repaid to the account it came from "
                "within 30 working days. Application fees for a service that was assessed "
                "and rejected on merit are not refundable.",
    },
    {
        "id": "office-hours",
        "title": "When offices are open",
        "text": "Counters are open 09:30 to 17:30 Monday to Friday and 09:30 to 13:00 on "
                "the second and fourth Saturday. They are closed on other Saturdays, "
                "Sundays, and declared public holidays. Token issue for same-day service "
                "stops one hour before closing. The online portal accepts applications at "
                "any time; working-day counts begin on the next working day.",
    },
    {
        "id": "how-we-use-your-details",
        "title": "How your details are used",
        "text": "Contact details are used to reach you about the application they were "
                "given for. They are visible to the wing handling that application and to "
                "an operator reviewing it, and are not shared with another applicant or "
                "with any private party. You may ask what is held about you, and ask for "
                "a correction, in writing to the wing that holds the file. A grievance "
                "about how your details were handled follows the ordinary escalation "
                "route and is answered within 15 working days.",
    },
    {
        "id": "fraud-warning",
        "title": "What staff will never ask you for",
        "text": "No officer will ask for a one-time password, an account password, or a "
                "card number, by telephone, email, or message. No officer will ask for "
                "payment to a personal account or a payment application. No officer will "
                "offer to speed up an application for a fee. A caller who asks for any of "
                "these is not from this department, whatever number they appear to call "
                "from. Report an approach of this kind through the grievance route, "
                "quoting the date and the number that called.",
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
    {
        "id": "income-certificate",
        "title": "Income certificate",
        "text": "An income certificate is issued by the Revenue wing and is valid for six "
                "months from the date of issue. Apply with proof of residence, the latest "
                "salary slip or a self-employment declaration, and photo identification. It "
                "is issued within 10 working days and is required for the housing assistance "
                "grant, fee concessions, and scholarship applications.",
    },
    {
        "id": "domicile-certificate",
        "title": "Domicile certificate",
        "text": "A domicile certificate confirms continuous residence in the district and is "
                "issued by the Revenue wing to a person who has lived there for at least "
                "three years. Apply with proof of residence for the full period — ration "
                "card, voter identity card, or utility bills — and photo identification. "
                "Issued within 15 working days.",
    },
    {
        "id": "senior-citizen-certificate",
        "title": "Senior citizen certificate",
        "text": "A senior citizen certificate is issued to a resident aged 60 or above and is "
                "used to claim fee concessions and priority service at counters. Apply with "
                "proof of age and proof of residence. It is issued free of charge within 7 "
                "working days and does not expire.",
    },
    {
        "id": "rti-request",
        "title": "Filing a Right to Information request",
        "text": "A Right to Information request may be filed with any wing at a fee of 10 "
                "rupees, waived for applicants below the poverty line. A reply is due within "
                "30 days, or 48 hours where the request concerns life or liberty. A first "
                "appeal against a refusal goes to the wing's own appellate officer within 30 "
                "days; a second appeal goes to the State Information Commission.",
    },
    {
        "id": "trade-licence-transfer",
        "title": "Transferring a trade licence",
        "text": "A trade licence may be transferred to a new owner on sale of the business, "
                "but not to a new location — a change of premises needs a fresh licence. "
                "Apply within 30 days of the transfer with the existing licence, the sale "
                "deed or partnership deed, and identity proof for the new owner. The "
                "transfer fee is 500 rupees and is decided within 15 working days.",
    },
    {
        "id": "noise-complaint",
        "title": "Reporting a noise nuisance",
        "text": "A noise nuisance from a commercial premises, construction work, or a "
                "loudspeaker after hours is reported to the Public Works wing through the "
                "grievance portal, quoting the address and the time it occurred. "
                "Construction noise is restricted to between 06:00 and 22:00 on working "
                "days. A loudspeaker running past 22:00 without a written exemption may "
                "be seized on the spot.",
    },
    {
        "id": "pothole-streetlight-complaint",
        "title": "Reporting a pothole or a faulty streetlight",
        "text": "A pothole, a faulty streetlight, or a blocked storm drain is reported to "
                "the Public Works wing through the grievance portal or by calling "
                "1800 425 1900, quoting the nearest landmark rather than a survey number. A "
                "streetlight fault is attended to within 5 working days; a pothole on a main "
                "road within 10, and on a residential street within 20.",
    },
    {
        "id": "pet-licence",
        "title": "Registering a pet",
        "text": "A dog kept within municipal limits must be registered annually with the "
                "Revenue wing for a fee of 100 rupees, and proof of a current rabies "
                "vaccination is required at registration. An unregistered dog found by an "
                "enforcement officer may be impounded. Registration does not apply to "
                "livestock or to cats.",
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
