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
from guardrails.rails import (
    deberta_injection_check,
    groundedness_check,
    toxicity_check,
)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_local_models(monkeypatch):
    """Run the suite as a base install: `requirements.txt` and nothing else.

    The engine defaults to `local+judge`, so without this every test that
    touches a content, injection, or grounding rail would download and load
    real weights — turning a 90-second hermetic suite into a slow one whose
    results depend on a model cache. Patching the loader to None puts each rail
    on its documented fallback path (escalate to the judge), which is what a
    deployment that skipped `requirements-local.txt` actually does.

    Tests that exercise the local layer stub `score()` directly — see
    `test_local_rails.py`. Nothing in the suite loads a model.
    """
    for module in (toxicity_check, deberta_injection_check, groundedness_check):
        monkeypatch.setattr(module, "classifier", lambda: None)
    yield


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
    # Sessions persist now, so the path needs redirecting too or a test run
    # writes live tokens into the developer's own data/ directory.
    monkeypatch.setattr(auth, "SESSIONS_PATH", tmp_path / "sessions.json")
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

@pytest.fixture(autouse=True)
def ignore_live_overrides(monkeypatch, tmp_path):
    """Stop the suite reading config/overrides.yaml.

    `load(REPO / "config" / "policy.yaml")` picks up whatever an operator last
    saved beside it, so a live setting decides whether a test passes: setting
    scope.action to `block` made two agent tests fail, because the prompt they
    drive the loop with is off-topic and never reached the loop.

    Only the repo's own policy is redirected. A sandbox copy keeps its real
    overrides path, so the tests that are *about* the overrides layer still
    exercise it.
    """
    from guardrails import config as config_module

    real = config_module.overrides_path_for
    repo_policy = (REPO / "config" / "policy.yaml").resolve()
    missing = tmp_path / "no-overrides.yaml"

    def scoped(policy_path):
        try:
            same = Path(policy_path).resolve() == repo_policy
        except OSError:
            same = False
        return missing if same else real(policy_path)

    monkeypatch.setattr(config_module, "overrides_path_for", scoped)
    yield
