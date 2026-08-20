"""Shared fixtures.

Every test here runs without an API key. The model-backed rails have their own
behaviour; these cover the parts where "it works" is a matter of fact rather
than calibration — and the parts a regression would silently break.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from guardrails import AuditLog, Engine, load

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def policy():
    """The real checked-in policy, with no overrides applied."""
    return load(REPO / "config" / "policy.yaml")


@pytest.fixture
def sandbox(tmp_path) -> Path:
    """A writable copy of config/ so override tests never touch the repo."""
    dst = tmp_path / "config"
    shutil.copytree(REPO / "config", dst)
    (dst / "overrides.yaml").unlink(missing_ok=True)
    return dst


@pytest.fixture
def sandbox_policy(sandbox):
    return load(sandbox / "policy.yaml")


@pytest.fixture
def engine(tmp_path):
    """Engine with no LLM — deterministic rails only."""
    return Engine(
        load(REPO / "config" / "policy.yaml"),
        llm=None,
        audit=AuditLog(tmp_path / "audit.log"),
    )


@pytest.fixture
def client(sandbox, monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDRAIL_CONFIG", str(sandbox / "policy.yaml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)          # keep audit + changelog out of the repo

    from server.app import create_app
    from server.routes import params as params_routes
    from server.state import state as app_state

    params_routes.CHANGELOG = tmp_path / "config-changes.log"
    app_state.corpus.path = tmp_path / "corpus.json"   # never touch data/
    app_state.corpus.reset()
    with TestClient(create_app()) as c:
        # Most of this file tests what the console can do, which means an
        # operator. The citizen's view has its own fixture below.
        c.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        yield c


@pytest.fixture
def citizen(sandbox, monkeypatch, tmp_path):
    """A signed-in account holding `chat` and nothing else."""
    monkeypatch.setenv("GUARDRAIL_CONFIG", str(sandbox / "policy.yaml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    from server.app import create_app
    from server.state import state as app_state

    app_state.corpus.path = tmp_path / "corpus.json"
    app_state.corpus.reset()
    with TestClient(create_app()) as c:
        c.post("/api/auth/login", json={"username": "citizen", "password": "citizen"})
        yield c


@pytest.fixture
def anonymous(sandbox, monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDRAIL_CONFIG", str(sandbox / "policy.yaml"))
    monkeypatch.chdir(tmp_path)
    from server.app import create_app

    with TestClient(create_app()) as c:
        yield c

@pytest.fixture(autouse=True)
def isolate_directory(monkeypatch, tmp_path):
    """Keep the real user directory and transcripts out of the tests.

    `directory` and `history` are module-level singletons loaded at import from
    data/. Without this a limit an operator set on a live account — or tokens
    they spent — decides whether a test passes. That is exactly how a suite
    starts failing for reasons nobody can reproduce.
    """
    from server import auth, history as history_module
    from server.auth import _default_users

    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(auth.directory, "users", _default_users())
    monkeypatch.setattr(auth.directory, "_sessions", {})

    monkeypatch.setattr(history_module.history, "path", tmp_path / "history.json")
    monkeypatch.setattr(history_module.history, "_turns", {})
    yield

@pytest.fixture(autouse=True)
def no_local_ner(monkeypatch, request):
    """Keep the spaCy pipeline out of the suite unless a test asks for it.

    Building it costs about eleven seconds and every analyse costs another,
    which turns a seventeen-second run into three and a half minutes. It also
    makes results depend on a language model version rather than on this code.

    Stubbing the engine rather than editing the policy is deliberate: it is
    exactly the state of a deployment where presidio is not installed, so the
    fallback path gets exercised by every test that touches the rail.

    A test marked `presidio` gets the real thing.
    """
    if request.node.get_closest_marker("presidio"):
        return
    from guardrails.rails import presidio_ner

    monkeypatch.setattr(presidio_ner, "engine", lambda: None)
