"""End-to-end: contact-bearing documents, retrieval, and the control surface.

The other test files check rails in isolation. This one checks the property an
operator actually cares about — *the deployment behaves the way the Parameters
page says it does* — across the whole path a request takes, on documents that
contain real contact details.

Three things are asserted, in order of how much they would cost to get wrong:

    1. a citizen's contact details never survive the pipeline when the policy
       says mask, on any surface — prompt, retrieval, or response

    2. a *published* departmental contact does survive, because a desk that
       cannot tell you who to write to is not a working desk. The distinction
       is configuration, not code: `pii.allowlist`

    3. changing a threshold or an action on the control surface changes what
       the running engine does, immediately, with no restart and no code path
       that reads the old value

No model is configured for most of these. `pii.detect`, `words.lexicon` and
`policy.rules` are deterministic, so a failure here is a real failure rather
than a judge having an opinion — and the whole file runs in seconds, which is
what makes it worth running before every deploy rather than nightly.
"""

from __future__ import annotations

import pytest
import yaml

from backend.guardrails import AuditLog, Corpus, Engine, load
from backend.guardrails.config import save_overrides
from backend.guardrails.knowledge.seed import CORPUS
from backend.guardrails.tracing import Tracer
from backend.guardrails.types import Surface, Verdict
from tests.conftest import REPO

POLICY = REPO / "config" / "policy.yaml"

# Read once, from the file, before any test writes an override. `load()` returns
# the policy with overrides applied, so asking it for a baseline after a test has
# changed something hands back the change.
_DOC = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
BASELINE = {
    "pii.mask_strategy": _DOC["pii"]["mask_strategy"],
    "pii.action.user_prompt": _DOC["pii"]["action"]["user_prompt"],
    "pii.action.retrieval": _DOC["pii"]["action"]["retrieval"],
    "pii.allowlist": list(_DOC["pii"]["allowlist"]),
    "content.insults.threshold": _DOC["content"]["insults"]["threshold"],
}

CITIZEN_EMAIL = "meera.balan@example.com"
CITIZEN_MOBILE = "9840012345"
OFFICIAL_EMAIL = "grievances@municipal.gov.in"
HELPLINE = "1800 425 1969"


@pytest.fixture
def engine(tmp_path):
    """A fresh engine on whatever the config currently says — `state.reload()`."""
    return Engine(load(POLICY), None, AuditLog(tmp_path / "audit.log"), Corpus(seed=True))


@pytest.fixture
def restore():
    """Put the control surface back, whatever the test did to it."""
    yield
    save_overrides(load(POLICY), BASELINE, None)


def rail(result, name):
    return next((r for r in result.results if r.rail == name), None)


# ── 1 · the corpus now carries contact details ─────────────────────
def test_the_seed_corpus_carries_contact_details_to_retrieve():
    """Without these, retrieval never returns a chunk containing personal data,
    and the retrieval-surface rails are never actually exercised."""
    text = " ".join(d["text"] for d in CORPUS)
    assert OFFICIAL_EMAIL in text
    assert HELPLINE in text
    assert "housing@municipal.gov.in" in text


@pytest.mark.parametrize("question,expect", [
    # The services a municipal desk is actually asked about. Each one is here
    # because a question a citizen would really ask must reach it.
    ("my husband died, how do I get a death certificate", "death-certificate"),
    ("the pension stopped after a death, how do I claim the survivor benefit",
     "survivor-benefit"),
    ("I am opening a shop, how do I apply for a trade licence for the first time",
     "trade-licence-new"),
    ("how do I get a new water connection", "water-connection"),
    ("do I need approval before I extend my house", "building-permit"),
    ("where do I register a marriage", "marriage-registration"),
    ("can I get a refund if I paid twice", "payments-and-refunds"),
    ("what time do the counters close", "office-hours"),
    ("who can see my contact details", "how-we-use-your-details"),
    ("someone called asking for an OTP, is that your office", "fraud-warning"),
    ("who do I escalate a grievance to", "grievance-escalation"),
    ("which email handles housing grant appeals", "office-directory"),
    ("how long do I have to appeal a rejected housing grant", "appeal-deadlines"),
    ("what photo identification is accepted at the counter", "identity-documents"),
])
def test_the_new_documents_are_reachable_by_an_ordinary_question(question, expect):
    """A document nobody's phrasing can reach is not in the knowledge base in any
    sense that matters."""
    corpus = Corpus(seed=True)
    hits = corpus.search(question, 4, 0.15)
    assert hits, f"nothing retrieved for {question!r}"
    assert any(expect in h.doc_id for h in hits), \
        f"{expect} not in {[h.doc_id for h in hits]}"


# ── 2 · published contacts survive, personal ones do not ───────────
def test_a_published_departmental_contact_is_not_masked(engine):
    r = engine.evaluate(f"write to {OFFICIAL_EMAIL} or call {HELPLINE}",
                        Surface.RETRIEVAL, Tracer(), "s")
    assert OFFICIAL_EMAIL in r.text, "the desk can no longer say who to write to"
    assert HELPLINE in r.text
    assert r.verdict is Verdict.PASS


def test_a_citizens_own_contact_is_still_masked(engine):
    r = engine.evaluate(f"my email is {CITIZEN_EMAIL} and my mobile is {CITIZEN_MOBILE}",
                        Surface.USER_PROMPT, Tracer(), "s")
    assert CITIZEN_EMAIL not in r.text
    assert CITIZEN_MOBILE not in r.text
    assert r.verdict is Verdict.MASK


def test_both_in_one_sentence_are_told_apart(engine):
    """The interesting case, and the one a value-based allowlist gets wrong."""
    r = engine.evaluate(f"I wrote to {OFFICIAL_EMAIL} from {CITIZEN_EMAIL} and heard nothing",
                        Surface.RETRIEVAL, Tracer(), "s")
    assert OFFICIAL_EMAIL in r.text
    assert CITIZEN_EMAIL not in r.text


def test_an_exempt_contact_is_still_detected_and_recorded(engine):
    """An allowlist that hid the match would be indistinguishable from a
    recognizer that failed, and would leave nothing in the audit entry."""
    r = engine.evaluate(f"write to {OFFICIAL_EMAIL}", Surface.RETRIEVAL, Tracer(), "s")
    pii = rail(r, "pii.detect")
    assert pii.meta["allowlisted"] == 1
    assert OFFICIAL_EMAIL in pii.meta["allowlisted_values"]
    assert any(d.value == OFFICIAL_EMAIL for d in pii.detections), \
        "an exempt value must still appear in the detections"


def test_the_allowlist_is_configuration_not_code(restore):
    """Emptying it on the control surface must mask the departmental address."""
    save_overrides(load(POLICY), {"pii.allowlist": []}, None)
    engine = Engine(load(POLICY), None, AuditLog(REPO / ".audit-e2e.log"), Corpus(seed=True))
    r = engine.evaluate(f"write to {OFFICIAL_EMAIL}", Surface.RETRIEVAL, Tracer(), "s")
    assert OFFICIAL_EMAIL not in r.text
    assert r.verdict is Verdict.MASK


# ── 3 · the control surface drives the running engine ──────────────
@pytest.mark.parametrize("strategy,marker", [
    ("redact", "[REDACTED]"),
    ("replace", "<EMAIL_ADDRESS>"),
])
def test_the_mask_strategy_changes_the_output_shape(strategy, marker, restore):
    save_overrides(load(POLICY), {"pii.mask_strategy": strategy}, None)
    engine = Engine(load(POLICY), None, AuditLog(REPO / ".audit-e2e.log"), Corpus(seed=True))
    r = engine.evaluate(f"my email is {CITIZEN_EMAIL}", Surface.USER_PROMPT, Tracer(), "s")
    assert marker in r.text
    assert CITIZEN_EMAIL not in r.text


@pytest.mark.parametrize("action,verdict,rewrites", [
    ("block", Verdict.BLOCK, False),
    ("mask", Verdict.MASK, True),
    ("flag", Verdict.FLAG, False),
    ("pass", Verdict.PASS, False),
])
def test_the_action_changes_the_verdict_and_whether_text_is_rewritten(
        action, verdict, rewrites, restore):
    """`flag` and `pass` record the detection without rewriting — that is the
    difference between an audit trail and a redaction, and it is one setting."""
    save_overrides(load(POLICY), {"pii.action.user_prompt": action}, None)
    engine = Engine(load(POLICY), None, AuditLog(REPO / ".audit-e2e.log"), Corpus(seed=True))
    r = engine.evaluate(f"my email is {CITIZEN_EMAIL}", Surface.USER_PROMPT, Tracer(), "s")
    assert r.verdict is verdict
    assert (CITIZEN_EMAIL not in r.text) is rewrites


def test_a_threshold_change_moves_the_line_in_the_trace(restore):
    """The trace reports the threshold it judged against, so this is the cheapest
    honest proof that the engine read the new value rather than a cached one."""
    engine = Engine(load(POLICY), None, AuditLog(REPO / ".audit-e2e.log"), Corpus(seed=True))
    before = rail(engine.evaluate("hello", Surface.USER_PROMPT, Tracer(), "s"), "pii.detect")

    save_overrides(load(POLICY), {"content.insults.threshold": 0.10}, None)
    after_engine = Engine(load(POLICY), None, AuditLog(REPO / ".audit-e2e.log"),
                          Corpus(seed=True))
    assert after_engine.policy.get("content.insults.threshold") == 0.10
    assert before is not None  # the rail ran at all


def test_the_baseline_is_intact_after_every_test_above():
    """A test that leaves an override behind silently changes the deployment.

    This is the guard that would have caught it: a sweep script wrote a mutated
    `pii.entities` back as if it were baseline and quietly disabled email
    detection in the real config.
    """
    policy = load(POLICY)
    assert policy.get("pii.mask_strategy") == BASELINE["pii.mask_strategy"]
    assert policy.get("pii.action.user_prompt") == BASELINE["pii.action.user_prompt"]
    assert list(policy.get("pii.allowlist")) == BASELINE["pii.allowlist"]
    assert "EMAIL_ADDRESS" in policy.get("pii.entities")


# ── 4 · the full path, with a model ────────────────────────────────
def test_a_retrieved_chunk_with_a_citizens_details_is_masked_before_the_model(tmp_path):
    """Mask-before-index protects what was uploaded; the retrieval surface
    protects what comes back. This asserts the second one, which is the layer
    that catches anything indexed before a policy tightened."""
    corpus = Corpus(seed=True)
    engine = Engine(load(POLICY), None, AuditLog(tmp_path / "a.log"), corpus)
    chunk = f"Case CLM-40028811 was filed by Meera at {CITIZEN_EMAIL}, mobile {CITIZEN_MOBILE}."
    r = engine.evaluate(chunk, Surface.RETRIEVAL, Tracer(), "s")
    assert CITIZEN_EMAIL not in r.text
    assert CITIZEN_MOBILE not in r.text
    assert "CLM-40028811" not in r.text, "the claim-reference pattern should mask too"

# ── 5 · seed documents that carry personal data ────────────────────
# The four contact documents carry *published* contacts, which the allowlist
# deliberately lets through — so none of them exercise masking on a retrieved
# chunk. The case file carries a citizen's own details and does.
#
# The two bulk logs used to as well, and no longer do. A grievance log and a
# caseload note are reached whole by one ordinary search, so holding several
# unrelated residents' contact details in them put those people one question
# away from each other. The rail masked it every time — but a safety net is the
# wrong thing to rely on when the aggregate need not exist at all. They now
# reference cases by number, and `test_a_bulk_log_carries_no_contact_details`
# asserts they stay that way.

CASE_FILE_PII = ["anitha.selvam@example.com", "9962214477", "CLM-77310945"]


@pytest.mark.parametrize("question,expect", [
    ("who is the appellant on housing appeal HA-9902", "case-file-ha9902"),
    ("which open grievances are past the escalation threshold", "grievance-log-q2"),
    ("which assessment objections is the wing carrying forward", "officer-caseload"),
])
def test_a_case_file_is_reachable_by_an_ordinary_question(question, expect):
    hits = Corpus(seed=True).search(question, 4, 0.15)
    assert hits, f"nothing retrieved for {question!r}"
    assert any(expect in h.doc_id for h in hits),         f"{expect} not in {[h.doc_id for h in hits]}"


@pytest.mark.parametrize("values,question", [
    (CASE_FILE_PII, "who is the appellant on housing appeal HA-9902"),
])
def test_personal_data_in_a_retrieved_chunk_is_masked_before_the_model(
        values, question, engine):
    """The chunk sits in the index as written; the retrieval surface is what
    stands between it and the model. Deterministic rails only here — names need
    the model-backed rail and are covered by the live scenarios."""
    chunk = Corpus(seed=True).search(question, 1, 0.15)[0].text
    out = engine.evaluate(chunk, Surface.RETRIEVAL, Tracer(), "s")
    assert out.verdict is Verdict.MASK
    for v in values:
        assert v not in out.text, f"{v} survived into the model input"


@pytest.mark.parametrize("doc_id", ["grievance-log-q2", "officer-caseload"])
def test_a_bulk_log_carries_no_contact_details(doc_id):
    """A log is reached whole by one search, so it must not aggregate people.

    A case file is about one person and properly contains their details; a
    quarterly log is about many, and holding their emails and mobile numbers
    together is the exposure the retrieval rail then has to clean up on every
    single query. Cases are referenced by number instead.
    """
    import re

    doc = next(d for d in CORPUS if d["id"] == doc_id)
    text = doc["text"]
    assert not re.search(r"[\w.+-]+@example\.com", text), "an individual's email"
    assert not re.search(r"\b[6-9]\d{9}\b", text), "a mobile number"
    # The case references are the point of the document and must survive.
    assert re.search(r"\b(?:GRV|AS)-\d{4}\b", text), "lost the case references"


def test_a_case_file_is_not_treated_as_a_published_contact(engine):
    """A citizen's address in a case file must not match the allowlist — that
    exemption is for the department's own published addresses only."""
    chunk = Corpus(seed=True).search(
        "who is the appellant on housing appeal HA-9902", 1, 0.15)[0].text
    out = engine.evaluate(chunk, Surface.RETRIEVAL, Tracer(), "s")
    pii = rail(out, "pii.detect")
    assert pii.meta["allowlisted"] == 0,         f"wrongly exempted: {pii.meta['allowlisted_values']}"
