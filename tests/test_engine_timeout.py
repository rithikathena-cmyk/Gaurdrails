"""The latency budget bounds wall-clock time, not just the verdict.

`Engine.evaluate()` runs every rail for a surface concurrently and gives them
`policy.latency_budget_ms` to finish. `test_parameters.py::
test_latency_budget_fails_closed_even_when_fail_mode_is_open` already proves
the *verdict* is right when a rail is too slow. It does not prove the *timing*
is right — and it wasn't: `with ThreadPoolExecutor(...) as pool:` calls
`shutdown(wait=True)` on exit regardless of whether `as_completed` already
timed out, so the call sat there until the slow rail's own worker thread
actually finished — measured live, 36s against a 20s budget, and 114s on an
agent turn layering more rails on top.

These tests pin the fix: a timed-out request returns close to the budget, not
close to however long the slowest rail actually took, and the worker that
kept running in the background afterward cannot mutate a trace this call has
already handed back.
"""

from __future__ import annotations

import time

import pytest

from backend.guardrails import AuditLog, Engine, Surface, Tracer, load
from backend.guardrails.types import Verdict
from tests.conftest import REPO


@pytest.fixture(autouse=True)
def _no_cold_transformers_import(monkeypatch):
    """`_content_rail()` calls `toxicity_check.available()` on every
    `evaluate()` call regardless of whether content is enabled for the
    surface being evaluated — a real `from transformers import pipeline`
    import, several real seconds the first time it happens in a process.
    Pre-existing, unrelated to this file's fix; conftest.py's own
    `no_local_models` stubs `classifier()` but not `available()`, so a wall-
    clock assertion here needs its own stub or the first test in the file
    eats that import cost and looks like a regression that isn't one."""
    from backend.guardrails.rails import toxicity_check

    monkeypatch.setattr(toxicity_check, "available", lambda: False)


class Slow:
    """Every judge call sleeps, then answers however `test_parameters.py`'s
    `StubClaude` would. Used instead of importing `StubClaude` directly so
    this file has no dependency on another test module's internals."""

    model = "stub"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    def judge(self, system, user, schema, *, max_tokens=2048, label=""):
        self.calls += 1
        time.sleep(self.delay)
        props = set(schema.get("properties", {}))
        return {c: 0.0 for c in props if c != "rationale"} | {"rationale": "stub"}

    def generate(self, system, messages, *, max_tokens=4096, model=None):
        from backend.guardrails.llm import Generation

        return Generation(text="a plain answer", model=self.model)


def engine_with(llm, **values) -> Engine:
    policy = load(REPO / "config" / "policy.yaml")
    # Force the judge branch unconditionally, bypassing the retrieval
    # candidate-gate this same session added (`entities.py`) — these tests
    # are pinning the timeout/detach fix, not the gating one, so pii.entities
    # must actually reach the (slow) judge every time.
    policy.values.update({"pii.entity_engine": "judge"} | values)
    return Engine(policy, llm, AuditLog("audit.log"), None)


# `retrieval` disables scope and content by the shipped severity matrix, and
# prompt_attack never runs on it at all — so on this surface pii.entities is
# the only judge-backed rail, and a capitalised phrase clears its gate.
RETRIEVAL_TEXT = "The Tamil Nadu State Apex Cooperative Bank runs this scheme."


# ── Test 1 ───────────────────────────────────────────────────────────
def test_a_slow_pii_entities_does_not_make_the_request_wait_for_it():
    """The correctness bug itself: a rail slower than the budget must not
    hold the request open for its own full duration."""
    llm = Slow(delay=2.0)
    engine = engine_with(llm, **{"policy.latency_budget_ms": 200})

    t0 = time.perf_counter()
    result = engine.evaluate(RETRIEVAL_TEXT, Surface.RETRIEVAL, Tracer(), "test")
    elapsed = time.perf_counter() - t0

    assert llm.calls >= 1, "pii.entities never reached the judge — test isn't exercising it"
    assert result.verdict is Verdict.BLOCK, "still fails closed on a timeout"
    assert elapsed < 1.0, (
        f"took {elapsed:.2f}s against a 200ms budget with a 2s rail — "
        "the request waited for the slow rail's own worker thread"
    )


def test_the_returned_result_never_waited_on_the_timed_out_rail():
    """Same property, asserted directly against `EvaluationResult` rather
    than a wall-clock margin: its own reported `duration_ms` must reflect
    the budget, not the slow rail's actual runtime."""
    llm = Slow(delay=2.0)
    engine = engine_with(llm, **{"policy.latency_budget_ms": 200})
    result = engine.evaluate(RETRIEVAL_TEXT, Surface.RETRIEVAL, Tracer(), "test")
    assert result.duration_ms < 1000, result.duration_ms


# ── Test 2 ───────────────────────────────────────────────────────────
def test_a_late_finishing_worker_cannot_mutate_a_completed_trace():
    """Once `evaluate()` has returned, a rail that is still running in the
    background must not be able to append itself into the trace this call
    already handed back — that trace may already be serialized, logged, or
    on its way to the caller."""
    llm = Slow(delay=1.0)
    engine = engine_with(llm, **{"policy.latency_budget_ms": 100})
    tracer = Tracer()

    engine.evaluate(RETRIEVAL_TEXT, Surface.RETRIEVAL, tracer, "test")
    stage = tracer.trace.stages[-1]
    rails_immediately_after_return = [
        (r.rail, r.verdict, r.error) for r in stage.rails
    ]

    time.sleep(1.5)  # let every detached worker actually finish

    rails_once_the_worker_finished = [
        (r.rail, r.verdict, r.error) for r in stage.rails
    ]
    assert rails_once_the_worker_finished == rails_immediately_after_return, (
        "a rail that finished after the budget expired still mutated a "
        "trace that had already been returned to the caller"
    )


def test_a_late_finishing_worker_does_not_land_in_a_later_stage():
    """`_run_gated` is handed the specific `StageTrace` its job belongs to,
    not `tracer._stage` — which by the time a detached worker finishes may
    already point at a completely different, later stage of the same
    request. Two stages run back to back on the same tracer; the first
    carries the slow rail."""
    llm = Slow(delay=1.0)
    engine = engine_with(llm, **{"policy.latency_budget_ms": 100})
    tracer = Tracer()

    engine.evaluate("hello", Surface.USER_PROMPT, tracer, "first")
    engine.evaluate("world", Surface.LLM_RESPONSE, tracer, "second")

    second_stage_rails_right_after = [r.rail for r in tracer.trace.stages[-1].rails]
    time.sleep(1.5)
    second_stage_rails_once_settled = [r.rail for r in tracer.trace.stages[-1].rails]

    assert second_stage_rails_once_settled == second_stage_rails_right_after, (
        "a rail abandoned in the first stage published itself into the second"
    )
