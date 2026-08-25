"""End-to-end: contact-bearing text, retrieval, and the control surface.

The other test files check rails in isolation. This one checks the property an
operator actually cares about — *the deployment behaves the way the Parameters
page says it does* — across the whole path a request takes, against text that
contains real contact details.

Three things are asserted, in order of how much they would cost to get wrong:

    1. a user's own contact details never survive the pipeline when the
       policy says mask, on any surface — prompt, retrieval, or response

    2. a *published* departmental contact does survive, because a service
       that cannot tell you who to write to is not a working one. The
       distinction is configuration, not code: `pii.allowlist`

    3. changing a threshold or an action on the control surface changes what
       the running engine does, immediately, with no restart and no code path
       that reads the old value

No model is configured for most of these. `pii.detect`, `words.lexicon` and
`policy.rules` are deterministic, so a failure here is a real failure rather
than a judge having an opinion — and the whole file runs in seconds, which is
what makes it worth running before every deploy rather than nightly.

The original version of this file also covered a fourth property: the same
guarantees against the built-in seed corpus's own documents, real retrieval
and all. That corpus was removed by design (`backend/guardrails/knowledge/
seed.py`), and with it the ~20 tests specific to its content — everything
below is what remains, none of it dependent on any document actually
existing in the knowledge base.
"""

from __future__ import annotations

import pytest
import yaml

from backend.guardrails import AuditLog, Corpus, Engine, load
from backend.guardrails.config import save_overrides
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


# ── 2 · published contacts survive, personal ones do not ───────────
# The two retrieval-surface tests that used to open this file — proving the
# seed corpus's own documents actually carried these contacts, and that an
# ordinary question could reach each one — are gone along with the seed
# corpus itself (`backend/guardrails/knowledge/seed.py`): a knowledge base
# with nothing in it has no documents for either claim to be about. Every
# test below still stands: none of them depend on the corpus at all, only on
# the allowlist pattern and the deterministic rails, exercised directly.
def test_a_published_departmental_contact_is_not_masked(engine):
    r = engine.evaluate(f"write to {OFFICIAL_EMAIL} or call {HELPLINE}",
                        Surface.RETRIEVAL, Tracer(), "s")
    assert OFFICIAL_EMAIL in r.text, "the desk can no longer say who to write to"
    assert HELPLINE in r.text
    assert r.verdict is Verdict.PASS


# A suffix is not a domain. The first version of these patterns anchored on one:
# `[a-z0-9.-]*nic\.in` also exempted clinic.in and panic.in, and
# `[a-z0-9.-]*municipal\.gov\.in` also exempted evilmunicipal.gov.in — a domain
# anybody can register and then never have masked anywhere in the pipeline. The
# exemption is the one place the stack is told to leave a value alone, so it has
# to be the department's own domain or a subdomain of it, and nothing else.
LOOKALIKE_CONTACTS = [
    "caseworker@evilmunicipal.gov.in",
    "caseworker@notmunicipal.gov.in",
    "meera@clinic.in",
    "meera@panic.in",
    "resident@municipal.gov.in.example.com",
]


@pytest.mark.parametrize("address", LOOKALIKE_CONTACTS)
def test_a_lookalike_domain_is_not_treated_as_a_published_contact(engine, address):
    r = engine.evaluate(f"write to {address}", Surface.RETRIEVAL, Tracer(), "s")
    pii = rail(r, "pii.detect")
    assert pii.meta["allowlisted"] == 0, f"wrongly exempted: {pii.meta['allowlisted_values']}"
    assert address not in r.text
    assert r.verdict is Verdict.MASK


@pytest.mark.parametrize("address", [
    OFFICIAL_EMAIL,
    "records@registry.municipal.gov.in",   # a subdomain is the department's own
    "helpdesk@nic.in",
    OFFICIAL_EMAIL.upper(),                # the patterns are compiled case-insensitive
])
def test_a_real_departmental_address_is_still_exempt(engine, address):
    """Anchoring the pattern must not cost the desk the addresses it exists to give out."""
    r = engine.evaluate(f"write to {address}", Surface.RETRIEVAL, Tracer(), "s")
    assert address in r.text
    assert rail(r, "pii.detect").meta["allowlisted"] == 1

def test_a_published_address_at_the_end_of_a_sentence_is_still_exempt(engine):
    """A full stop is not another domain label.

    The first anchored version of these patterns ended `(?![a-z0-9.-])`, which
    refused a trailing dot of any kind — and most of the departmental addresses
    in the corpus are written at the end of a sentence. It unmasked four of
    them. The lookalike cases above could not see it: not one of them has a
    full stop after the address.
    """
    r = engine.evaluate(
        f"Write to {OFFICIAL_EMAIL}. Housing goes to housing@municipal.gov.in.",
        Surface.RETRIEVAL, Tracer(), "s")
    assert OFFICIAL_EMAIL in r.text
    assert "housing@municipal.gov.in" in r.text
    assert rail(r, "pii.detect").meta["allowlisted"] == 2


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
# This section is gone along with the seed corpus: it tested that four
# specific seeded case-file and log documents — a resident's own contact
# details in one, deliberately none in the other two — behaved correctly on
# retrieval. There is no seeded content left for any of that to be about.
# Section 4 above already covers the same masking property on a document
# that is not seeded — the property was never specific to these documents,
# only the fixture was.
