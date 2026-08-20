"""The regeneration loop, with a stubbed judge.

This path can't be forced from a prompt — Claude declines to fabricate, so the
grounding rail scores 1.00 and never fails. That is the correct outcome and a
poor test. Stubbing the judge lets us assert the loop itself: a failed output
rail returns to the model, never to the user, and only a second failure
surfaces a human.
"""

from __future__ import annotations

import re

import pytest

from guardrails import AuditLog, Engine, load
from guardrails.types import Verdict
from tests.conftest import REPO


class StubClaude:
    """Scripted judge + generator. Records what it was asked."""

    model = "stub-model"

    def __init__(self, grounding_scores: list[float]) -> None:
        self.grounding_scores = list(grounding_scores)
        self.generations = 0
        self.retry_prompts: list[str] = []

    def judge(self, system, user, schema, *, max_tokens=2048):
        props = set(schema.get("properties", {}))
        if "verdict" in props and "confidence" in props:
            # The adjudicator. Upholding is the honest default for a stub: it
            # exercises the real path without silently changing any outcome.
            m = re.search(r"RESOLVED VERDICT[^:]*: (\w+)", user)
            return {"verdict": m.group(1) if m else "pass", "confidence": 1.0,
                    "rationale": "stub upheld the rails"}
        if "consistency" in props:
            score = self.grounding_scores.pop(0) if self.grounding_scores else 1.0
            return {"consistency": score, "relevance": 1.0,
                    "unsupported_claims": [] if score >= 0.5 else ["an invented fee"],
                    "rationale": "stub"}
        if "injection" in props:
            return {"injection": 0.0, "technique": "none", "rationale": "stub"}
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}

    def generate(self, system, messages, *, max_tokens=4096, model=None):
        from guardrails.llm import Generation

        self.generations += 1
        self.retry_prompts.append(messages[-1]["content"])
        return Generation(text=f"answer attempt {self.generations}", model=self.model)


def build(scores, **overrides):
    """Engine with a scripted judge. `overrides` patch policy values in place."""
    policy = load(REPO / "config" / "policy.yaml")
    policy.values.update(overrides)
    llm = StubClaude(scores)
    return Engine(policy, llm, AuditLog("audit.log")), llm


@pytest.fixture(autouse=True)
def _tmp_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ── the loop ───────────────────────────────────────────────────────
def test_ungrounded_answer_is_regenerated_not_delivered():
    engine, llm = build([0.20, 0.95])          # fail once, then pass
    res = engine.converse("What documents do I need to renew a trade licence?")

    assert res.blocked is False
    assert llm.generations == 2
    assert res.reply == "answer attempt 2"     # the user never sees attempt 1
    assert res.trace.regenerations == 1


def test_retry_prompt_differs_from_the_original():
    """A retry identical to the original just pays twice for the same answer."""
    engine, llm = build([0.20, 0.95])
    engine.converse("What documents do I need to renew a trade licence?")

    first, second = llm.retry_prompts
    assert first != second
    assert "only the context" in second


def test_repeated_failure_escalates_to_human_review():
    engine, llm = build([0.10, 0.10, 0.10], **{"grounding.max_regenerations": 1})
    res = engine.converse("What documents do I need to renew a trade licence?")

    assert res.blocked is True
    assert "review" in res.reply.lower()
    assert res.refusal_reason == "ungrounded after maximum regenerations"
    assert llm.generations == 2                # original + one retry, then stop


def test_max_regenerations_zero_escalates_immediately():
    engine, llm = build([0.10], **{"grounding.max_regenerations": 0})
    res = engine.converse("What documents do I need to renew a trade licence?")

    assert res.blocked is True
    assert llm.generations == 1


def test_flag_action_delivers_the_answer_anyway():
    engine, llm = build([0.10], **{"grounding.action_on_fail": "flag"})
    res = engine.converse("What documents do I need to renew a trade licence?")

    assert res.blocked is False
    assert llm.generations == 1                # no retry under `flag`
    assert res.reply == "answer attempt 1"


def test_grounded_answer_is_delivered_first_time():
    engine, llm = build([0.95])
    res = engine.converse("What documents do I need to renew a trade licence?")

    assert llm.generations == 1
    assert res.trace.regenerations == 0
    assert res.trace.verdict is Verdict.PASS


def test_each_attempt_is_traced_separately():
    engine, _ = build([0.20, 0.95])
    names = [s.name for s in engine.converse(
        "What documents do I need to renew a trade licence?").trace.stages]

    assert "Generation" in names
    assert any(n.startswith("Regeneration") for n in names)
    assert sum(1 for n in names if n.startswith("Grounding")) == 2
