"""Shared fixtures.

Every test here runs without an API key. The model-backed rails have their own
behaviour; these cover the parts where "it works" is a matter of fact rather
than calibration — and the parts a regression would silently break.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

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
