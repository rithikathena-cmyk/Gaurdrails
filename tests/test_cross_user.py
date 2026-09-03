"""Isolation between two signed-in people.

The vault is engine-wide and the session store is process-wide, so anything
keyed only by the client-supplied `session_id` is reachable by naming it. Three
things were: the value behind a mask token, the conversation history holding the
already-unmasked reply, and the approval token that authorises a write.

These are the cross-user cases the unit tests in `test_vault_auth.py` cannot
reach, because they only appear once two principals share one process.
"""

from __future__ import annotations

import re
import secrets

import pytest
from fastapi.testclient import TestClient

from tests.test_parameters import Q, StubClaude, _corpus, engine_with
from backend.guardrails import Corpus, Document, Surface, Tracer
from backend.guardrails.rails.vault import CORPUS_OWNER


SSN = "796-33-9021"


@pytest.fixture
def two_users(sandbox, monkeypatch, tmp_path):
    """Two people signed in against one app, as in a real deployment."""
    monkeypatch.setenv("GUARDRAIL_CONFIG", str(sandbox / "policy.yaml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    from backend.server.app import create_app
    from backend.server.state import state as app_state

    app_state.corpus.path = tmp_path / "corpus.json"
    app_state.corpus.reset()
    app_state.sessions.clear()
    app_state.pending.clear()

    app = create_app()
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        bob.post("/api/auth/login", json={"username": "citizen", "password": "citizen"})
        yield alice, bob, app_state


# ── conversation history ───────────────────────────────────────────
def test_naming_someone_elses_session_does_not_hand_back_their_history(two_users):
    """`session_id` is a client-chosen field, so on its own it is a claim.

    The stored reply is post-egress text, which for its owner has already been
    unmasked — so an unnamespaced key leaks the raw value however well the vault
    is scoped.
    """
    alice, bob, state = two_users
    alice.post("/api/chat", json={"message": f"my ssn is {SSN}", "session_id": "shared"})

    for turns in _history_for(state, "citizen"):
        for turn in turns:
            assert SSN not in turn["content"]


def test_each_person_gets_their_own_bucket_for_the_same_session_id(two_users):
    alice, bob, state = two_users
    alice.post("/api/chat", json={"message": "alice was here", "session_id": "shared"})
    bob.post("/api/chat", json={"message": "bob was here", "session_id": "shared"})

    admin_text = _flatten(_history_for(state, "admin"))
    citizen_text = _flatten(_history_for(state, "citizen"))
    assert "alice was here" in admin_text and "alice was here" not in citizen_text
    assert "bob was here" in citizen_text and "bob was here" not in admin_text


def test_resetting_a_session_only_clears_your_own(two_users):
    alice, bob, state = two_users
    alice.post("/api/chat", json={"message": "alice was here", "session_id": "shared"})
    bob.post("/api/chat", json={"message": "bob was here", "session_id": "shared"})

    bob.post("/api/session/reset", params={"session_id": "shared"})
    assert "alice was here" in _flatten(_history_for(state, "admin"))
    assert _flatten(_history_for(state, "citizen")) == ""


def _history_for(state, owner: str) -> list[list[dict]]:
    return [v for k, v in state.sessions.items() if k.split("\x00", 1)[0] == owner]


def _flatten(buckets: list[list[dict]]) -> str:
    return " ".join(t["content"] for turns in buckets for t in turns)


# ── the vault, across two real requests ────────────────────────────
def test_a_token_minted_for_one_person_will_not_unmask_for_another(two_users):
    """The end-to-end version of the unit test: one engine, one vault, two users."""
    alice, bob, state = two_users
    engine = state.engine
    token = engine.vault.store("US_SSN", SSN, "admin")

    assert engine.vault.reveal(token, "admin") == SSN
    assert engine.vault.reveal(token, "citizen") is None
    assert engine.vault.reveal(token, "") is None


# Egress only exists once generation runs, so these drive the engine with the
# suite's scripted model rather than the API, which has no key under test.
def test_the_egress_rail_records_who_it_unmasked_for():
    # A bare SSN retrieves nothing on its own, and with no relevant chunk the
    # turn never reaches egress any more (see the retrieval-relevance gate in
    # `engine.py`) — this test is about the unmask stage, so it needs a real
    # corpus hit to get there at all.
    e = engine_with(StubClaude(reply="ok"), corpus=_corpus())
    res = e.converse(f"{SSN}. {Q}", principal="alice")
    unmask = _rail(res, "vault.unmask")
    assert unmask.meta["principal"] == "alice"


def test_the_owner_gets_their_own_value_back_at_egress():
    """The token the rails minted resolves for the principal that owns it.

    SSN is judge-only now — no deterministic layer exists — so the stub is
    scripted to report it the same way a real judge call would."""
    e = engine_with(StubClaude(reply="ok",
                               entities=[{"text": SSN, "kind": "US_SSN", "confidence": 0.95}]))
    masked = evaluate_prompt(e, SSN, "alice")
    tok = re.search(r"<US_SSN:([0-9a-f]{12})", masked)
    assert tok, masked
    assert e.vault.reveal(tok.group(1), "alice") == SSN
    assert e.vault.reveal(tok.group(1), "mallory") is None


def test_a_foreign_token_in_the_reply_is_refused_and_recorded(monkeypatch):
    """A token that reaches the wrong caller is a security event, not a no-op."""
    # `corpus=_corpus()` and `Q` appended to the prompt: retrieval must find
    # something real or the turn never reaches egress any more (see the
    # retrieval-relevance gate in `engine.py`) — the stubbed reply below is
    # fixed regardless of what was retrieved, so this does not change what
    # the test is actually checking.
    e = engine_with(StubClaude(reply="ok"), corpus=_corpus())
    # Pin the token. `secrets.token_hex(6)` can come out all digits, and a
    # twelve-digit run inside the reply is something the PII recognisers will
    # happily claim as a phone number or an Aadhaar — re-masking the foreign
    # token into a fresh one owned by this caller before egress ever sees it.
    # That made this test fail about one run in a hundred. Letters throughout
    # cannot be mistaken for an identifier.
    monkeypatch.setattr(secrets, "token_hex", lambda n: "abcdefabcdef")
    foreign = e.vault.store("US_SSN", SSN, "somebody-else")
    e.llm.reply = f"your record shows <US_SSN:{foreign}>"

    res = e.converse(f"what does my record show. {Q}", principal="alice")

    assert SSN not in res.reply
    unmask = _rail(res, "vault.unmask")
    assert unmask.meta["tokens_revealed"] == 0
    assert unmask.meta["unmask_denied"] == 1
    assert unmask.meta["denial_reasons"] == ["owner_mismatch"]


# ── retrieval: another resident's details ───────────────────────────
# The live bug this closes. A citizen asking "who is the appellant on housing
# appeal HA-9902" got Anitha Selvam's name, email, mobile and address back —
# the model was truthfully reporting they were masked, and egress then
# substituted the real values into that very sentence, because the retrieval
# scan minted the tokens under `principal` and the asking citizen *was* the
# principal. The fix mints retrieval-surface tokens under `CORPUS_OWNER`
# instead — see engine.py's note on the retrieval `evaluate()` call.
def test_a_retrieved_residents_details_do_not_unmask_for_the_asker():
    """`Engine.evaluate()` is generic — the fix lives at the call site in
    `engine.py`, which is why this goes through `converse()` rather than
    calling `evaluate()` directly with a hand-picked owner. Calling `evaluate()`
    directly would only re-test the vault, which `test_vault_auth.py` already
    covers, and would pass even without the fix.
    """
    corpus = Corpus(seed=False)
    corpus.add(Document(
        id="test:case-file-ha9902", title="Case file HA-9902 — housing appeal",
        source="test", kind="txt",
        chars=80, chunks=["Housing appeal HA-9902. Appellant: Anitha Selvam, "
                          "contactable on anitha.selvam@example.com."],
        status="indexed", verdict="pass"))
    # EMAIL_ADDRESS is judge-only now — no deterministic layer exists — so
    # the stub is scripted to report it the same way a real judge call would.
    engine = engine_with(StubClaude(entities=[
        {"text": "anitha.selvam@example.com", "kind": "EMAIL_ADDRESS", "confidence": 0.95},
    ]), corpus=corpus)
    result = engine.converse(
        "who is the appellant on housing appeal HA-9902 and how do I contact them",
        session_id="s", principal="citizen")

    entries = list(engine.vault._store.values())  # noqa: SLF001
    minted = [e for e in entries if e.entity == "EMAIL_ADDRESS"]
    assert minted, "nothing was minted — the retrieval scan did not run"
    # Not the citizen who asked, not anybody who could sign in and claim it.
    assert all(e.owner == CORPUS_OWNER for e in minted),         f"a retrieved resident's detail was minted under {[e.owner for e in minted]}"

    # And genuinely reversible — for the one owner nobody who signs in can be.
    token = next(t for t, e in engine.vault._store.items() if e.entity == "EMAIL_ADDRESS")  # noqa: SLF001
    assert engine.vault.reveal(token, "citizen") is None
    assert engine.vault.reveal(token, CORPUS_OWNER) == "anitha.selvam@example.com"


def test_a_tool_results_details_do_not_unmask_for_the_caller():
    """`agent.data` gets the same fix, for the same reason: a claim record's
    note field was filled in by whoever filed it, not by the caller asking.
    `CLM-88817766`'s note carries `collections@attacker.example` — already used
    elsewhere to prove the injection in it is withheld; here it proves the
    email address in the same note is not handed back to whoever asked.

    `agent.data_check_mode` defaults to `agentic` now — pinned to `rail`
    here since this test is specifically about the fixed pipeline's own
    vault-minting behaviour, not the specialist agents' scripted verdict.
    """
    from backend.guardrails import AgentRunner, AuditLog, Corpus, Engine, load
    from backend.guardrails.rails.vault import SYSTEM_OWNER
    from tests.test_agent import ScriptedClaude
    from tests.conftest import REPO

    policy = load(REPO / "config" / "policy.yaml")
    policy.values["agent.data_check_mode"] = "rail"
    # EMAIL_ADDRESS is judge-only now — no deterministic layer exists — so
    # the stub is scripted to report it the same way a real judge call would.
    llm = ScriptedClaude([("tool", "check_claim_status", {"reference": "CLM-88817766"}),
                          ("answer", "That claim is in assessment.")],
                         entities=[{"text": "collections@attacker.example",
                                   "kind": "EMAIL_ADDRESS", "confidence": 0.95}])
    engine = Engine(policy, llm, AuditLog("audit.log"), Corpus(seed=True))
    runner = AgentRunner(engine, llm)

    runner.run("what is the status of claim CLM-88817766", session_id="s",
              principal="citizen")

    entries = list(engine.vault._store.values())  # noqa: SLF001
    minted = [e for e in entries if e.entity == "EMAIL_ADDRESS"]
    assert minted, "nothing was minted — the agent.data scan did not run"
    assert all(e.owner == SYSTEM_OWNER for e in minted),         f"a tool result's detail was minted under {[e.owner for e in minted]}"


def evaluate_prompt(engine, text: str, principal: str) -> str:
    from backend.guardrails import Surface, Tracer

    return engine.evaluate(text, Surface.USER_PROMPT, Tracer(), "Prompt rails",
                           owner=principal).text


def _rail(result, name: str):
    found = [r for r in result.trace.rails if r.rail == name]
    assert found, f"{name} did not run"
    return found[0]


# ── the approval gate ──────────────────────────────────────────────
def test_one_person_cannot_approve_another_persons_write(two_users):
    """Approval is the one place a human overrides a deterministic hold.

    Holding the token cannot be sufficient, or any signed-in user who learned
    one could authorise somebody else's destructive call.
    """
    from backend.guardrails.agent.runner import PendingApproval

    _, _, state = two_users
    pending = PendingApproval(
        token="apr_deadbeef", tool="file_grievance", why="write tool",
        summary="File a grievance", args={}, question="file it", owner="admin",
    )
    state.park(pending)

    assert state.claim("apr_deadbeef", "citizen") is None
    # ...and refusing must not consume it — the owner can still answer.
    assert state.claim("apr_deadbeef", "admin") is pending


def test_an_unowned_claim_does_not_match_an_owned_approval(two_users):
    from backend.guardrails.agent.runner import PendingApproval

    _, _, state = two_users
    state.park(PendingApproval(
        token="apr_cafe", tool="file_grievance", why="write tool",
        summary="File a grievance", args={}, question="file it", owner="admin",
    ))
    assert state.claim("apr_cafe", "") is None
