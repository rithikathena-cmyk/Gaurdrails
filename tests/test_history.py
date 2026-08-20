"""Transcripts, and who is allowed to read them.

The store itself is simple enough that the interesting tests are all about the
boundary: a citizen must not be able to reach another person's conversations by
any route, and an operator must be able to reach every one. That boundary had no
automated coverage when it was written, which is exactly the kind of thing that
survives a refactor by accident.

The store tests use a temporary path so a run never touches `data/history.json`.
"""

from __future__ import annotations

import pytest

from server.history import HistoryStore


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history.json")


def add(store, user, session="s1", question="q", reply="a", verdict="pass", **kw):
    store.append(user, session_id=session, question=question, reply=reply,
                 verdict=verdict, request_id=kw.pop("request_id", "req_1"), **kw)


# ── the store ──────────────────────────────────────────────────────
def test_a_turn_is_kept_and_read_back(store):
    add(store, "meera", question="how do I renew a licence", reply="Form 4B.")
    turns = store.turns("meera")
    assert len(turns) == 1
    assert turns[0]["question"] == "how do I renew a licence"
    assert turns[0]["reply"] == "Form 4B."


def test_a_blocked_turn_is_kept_too(store):
    """The refusal is the part that answers 'why could I not get an answer'."""
    add(store, "meera", verdict="block", blocked=True,
        refusal_reason="prompt_attack (instruction_override)")
    assert store.stats("meera")["blocked"] == 1
    assert store.turns("meera")[0]["refusal_reason"].startswith("prompt_attack")


def test_transcripts_survive_a_restart(tmp_path):
    path = tmp_path / "history.json"
    first = HistoryStore(path)
    add(first, "meera", question="remembered?")
    assert HistoryStore(path).turns("meera")[0]["question"] == "remembered?"


def test_turns_group_into_conversations_newest_first(store):
    add(store, "meera", session="old", question="first")
    add(store, "meera", session="new", question="second")
    add(store, "meera", session="new", question="third")
    sessions = store.sessions("meera")
    assert [s["session_id"] for s in sessions] == ["new", "old"]
    assert sessions[0]["turns"] == 2
    assert sessions[0]["opened_with"] == "second"


def test_one_persons_turns_never_appear_in_anothers(store):
    add(store, "meera", question="mine")
    add(store, "rajesh", question="theirs")
    assert [t["question"] for t in store.turns("meera")] == ["mine"]
    assert [t["question"] for t in store.turns("rajesh")] == ["theirs"]


def test_the_oldest_turns_fall_off_rather_than_growing_forever(store, monkeypatch):
    monkeypatch.setattr("server.history.MAX_TURNS_PER_USER", 5)
    for i in range(9):
        add(store, "meera", question=f"q{i}")
    kept = [t["question"] for t in store.turns("meera")]
    assert len(kept) == 5
    assert kept[0] == "q4", "the oldest should be the ones dropped"


def test_removing_an_account_removes_its_transcripts(store):
    add(store, "meera")
    store.forget_user("meera")
    assert store.turns("meera") == []


def test_stats_add_up(store):
    add(store, "meera", masked=2, tokens=100, cost_usd=0.01)
    add(store, "meera", masked=3, tokens=50, cost_usd=0.02, verdict="block", blocked=True)
    s = store.stats("meera")
    assert (s["turns"], s["blocked"], s["masked"], s["tokens"]) == (2, 1, 5, 150)
    assert s["cost_usd"] == pytest.approx(0.03)


# ── the boundary ───────────────────────────────────────────────────
def test_a_citizen_sees_their_own_and_is_offered_nobody_elses(citizen):
    body = citizen.get("/api/history").json()
    assert body["mine"] is True
    assert body["whose"]["name"] == "citizen"
    assert body["people"] == [], "a citizen has nobody to switch to"


def test_a_citizen_cannot_ask_for_another_persons_list(citizen):
    r = citizen.get("/api/history", params={"user": "admin"})
    assert r.status_code == 403
    assert "traces permission" in r.json()["error"]["message"]


def test_a_citizen_cannot_reach_another_persons_transcript_directly(citizen):
    """The detail route is a separate door and has to be locked separately."""
    r = citizen.get("/api/history/admin/anything")
    assert r.status_code == 403


def test_an_operator_may_read_anyone(client):
    body = client.get("/api/history", params={"user": "citizen"}).json()
    assert body["mine"] is False
    assert body["whose"]["name"] == "citizen"
    assert {p["name"] for p in body["people"]} >= {"admin", "citizen"}


def test_an_operator_asking_for_nobody_gets_themselves(client):
    body = client.get("/api/history").json()
    assert body["mine"] is True
    assert body["whose"]["name"] == "admin"


def test_an_unknown_user_is_a_404_not_an_empty_list(client):
    r = client.get("/api/history", params={"user": "nobody-here"})
    assert r.status_code == 404


def test_signing_in_is_required(anonymous):
    assert anonymous.get("/api/history").status_code == 401
