"""Scenario endpoints.

Each scenario drives the real engine and the real agent, then asserts on what
came back. They are exposed over HTTP so the demo page can run them against a
live deployment rather than replaying a recording — including the two that take
several model calls and a human decision.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from guardrails import LLMError, scenarios as sc

from ..state import state

router = APIRouter()


@router.get("/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [s.to_dict() for s in sc.SCENARIOS],
        "model_rails": state.model_rails,
    }


@router.post("/scenarios/{scenario_id}/run")
def run_scenario(scenario_id: str) -> dict[str, Any]:
    scenario = sc.BY_ID.get(scenario_id)
    if scenario is None:
        raise HTTPException(404, detail={"kind": "not_found", "message": scenario_id})
    if state.error:
        raise HTTPException(500, detail={"kind": "config", "message": state.error})
    if state.engine is None or state.agent is None:
        raise HTTPException(503, detail={"kind": "startup", "message": "engine not ready"})
    if scenario.needs_model and not state.model_rails:
        raise HTTPException(503, detail={
            "kind": "llm",
            "message": f"{scenario.title} needs a model. Set ANTHROPIC_API_KEY and restart.",
        })

    try:
        result = sc.run(scenario_id, state.engine, state.agent)
    except LLMError as exc:
        raise HTTPException(502, detail={"kind": "llm", "message": str(exc)}) from exc

    # Every trace a scenario produced belongs in the trace ring like any other.
    for step in result.steps:
        if step.trace:
            state.record(step.trace)
    return {"scenario": scenario.to_dict(), "result": result.to_dict()}
