"""`pii.action.agent_tool` and `pii.action.agent_data`, proven adjustable
end to end: Parameters UI/API -> validation -> overrides -> policy reload
-> the real PII rail -> the real agent/tool boundary.

Every test here drives the actual production path — a real `PATCH
/api/parameters` (validated against the registry, written to overrides,
reloaded), then a real `POST /api/agent/chat` (and, for a write tool,
`POST /api/agent/approve`) through the real `AgentRunner` and
`Engine.evaluate(..., Surface.AGENT_TOOL/AGENT_DATA, ...)`. Nothing here
reads a policy value directly or calls the rail in isolation — the point is
proving a config change made through the API changes what the deterministic
rail around a real tool call actually does. The model is scripted the same
way `test_agent.py` scripts it: this is not testing whether Claude picks
the right tool, only what the stack does around the call it makes.

`file_grievance` (a write tool) is used for both surfaces rather than
`check_claim_status`, deliberately: `CLM-88817766`'s seed note carries both
PII *and* an injection attempt, and injection scanning always runs on
`agent.data` regardless of severity matrix settings — mixing the two would
make a PII-specific action change invisible behind an injection block that
fires unconditionally. `file_grievance` lets each test supply a payload
that is only PII, isolating the surface actually under test.
"""

from __future__ import annotations

import pytest

from backend.guardrails.agent.runner import AgentRunner
from tests.test_agent import ScriptedClaude

SSN = "796-33-9021"
EMAIL = "meera.balan@example.com"


def _install_scripted_llm(script, **agentic_kwargs):
    """`state.reload()` — triggered by every PATCH — rebuilds `state.agent`
    from `state.engine.llm`, which is `None` without a live API key. The
    scripted model has to be reinstalled after every reload, the same way a
    real deployment re-points at its real model after a config change.

    `engine.llm` alone is not enough any more: `entity_rail` (like every
    other model-backed rail) captured its own `llm` reference when
    `_build_rails()` last ran — at `state.reload()` time, before this
    function ever sets `engine.llm` to the scripted model — so it keeps
    calling the stale one (`None` in tests) unless rebuilt. This never
    mattered while PII detection was deterministic; it does now that
    `entity_rail`'s own judge call is the only thing that can find anything.

    `agentic_kwargs` (`agentic_pii=`/`agentic_injection=`/`agentic_content=`)
    forward to `ScriptedClaude`, for `agent.data_check_mode="agentic"` tests."""
    from backend.server.state import state as app_state

    llm = ScriptedClaude(script, **agentic_kwargs)
    app_state.engine.llm = llm
    app_state.engine._build_rails()  # noqa: SLF001 — rebuild every rail against the new llm
    app_state.agent = AgentRunner(app_state.engine, llm)
    app_state.model_rails = True
    return llm


def _patch(client, values):
    resp = client.patch("/api/parameters", json={"values": values})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── the default itself, confirmed through the real API ──────────────────
def test_both_keys_default_to_mask_through_the_real_api(client):
    snapshot = client.get("/api/parameters").json()
    assert snapshot["current"]["pii.action.agent_tool"] == "mask"
    assert snapshot["current"]["pii.action.agent_data"] == "mask"


def test_both_keys_accept_every_documented_action_through_the_real_api(client):
    """ALLOW/pass, MASK, FLAG, BLOCK — the four the existing registry
    already permits for both keys, proven by round-tripping each through
    the real validation path rather than reading `options=[...]` in source."""
    for key in ("pii.action.agent_tool", "pii.action.agent_data"):
        for action in ("pass", "mask", "flag", "block"):
            patched = _patch(client, {key: action})
            assert patched["snapshot"]["current"][key] == action
    # leave both back at the documented default for the tests below
    _patch(client, {"pii.action.agent_tool": "mask", "pii.action.agent_data": "mask"})


def test_an_undocumented_action_is_rejected_by_validation(client):
    resp = client.patch("/api/parameters",
                        json={"values": {"pii.action.agent_tool": "redact"}})
    assert resp.status_code == 422
    # rejected before anything reloaded — the stored value is untouched
    assert client.get("/api/parameters").json()["current"]["pii.action.agent_tool"] == "mask"


# ── pii.action.agent_tool: PATCH -> reload -> the real args rail ────────
# `args_verdict` is the causal signal to prove here, not `args_preview`:
# the args scan governs whether the call proceeds at all (block) and what
# gets recorded for audit (the verdict itself) — unlike agent_data, this
# surface does not currently rewrite what the tool actually receives, only
# whether it runs, so `args_preview` stays the raw args regardless of
# action. A write tool always pauses for approval once it passes the args
# rail, so the call only lands in `calls[0]` after that approval resolves —
# except when blocked, which never reaches the approval gate at all.
@pytest.mark.parametrize("action", ["block", "mask", "flag", "pass"])
def test_agent_tool_action_changes_real_tool_call_behaviour(client, action):
    # agent.prefilter_mode and agent.data_check_mode both default to
    # "agentic" now — pinned back to the fixed pipeline here since this test
    # is isolating pii.action.agent_tool specifically, not either agentic path.
    _patch(client, {"pii.action.agent_tool": action,
                    "agent.prefilter_mode": "off", "agent.data_check_mode": "rail"})
    # US_SSN is judge-only now — no deterministic layer exists — so the
    # stub is scripted to report it the same way a real judge call would.
    _install_scripted_llm([
        ("tool", "file_grievance",
         {"subject": f"Billing dispute — SSN {SSN} on file",
          "details": "Please investigate the duplicate charge."}),
        ("answer", "Noted, thank you."),
    ], entities=[{"text": SSN, "kind": "US_SSN", "confidence": 0.9}])

    resp = client.post("/api/agent/chat",
                       json={"message": "file a grievance about a billing error"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    if action == "block":
        assert body["approval"] is None, "a blocked call must never reach the approval gate"
        call = body["calls"][0]
    else:
        assert body["approval"] is not None, \
            "a write tool that passed the args rail must still stop for approval"
        approve_resp = client.post("/api/agent/approve",
                                   json={"token": body["approval"]["token"], "approved": True})
        assert approve_resp.status_code == 200, approve_resp.text
        call = approve_resp.json()["calls"][0]

    assert call["name"] == "file_grievance"
    assert call["args_verdict"] == action, \
        f"the args rail's verdict must track the configured action, got {call!r}"
    if action == "block":
        assert call["blocked_reason"]


# ── pii.action.agent_data: PATCH -> reload -> the real tool-result rail ─
@pytest.mark.parametrize("action", ["block", "mask", "flag", "pass"])
def test_agent_data_action_changes_real_tool_result_behaviour(client, action):
    # Pinned to `pass` so the raw email survives into `_file_grievance`'s own
    # echoed JSON response undisturbed — isolating what happens to it on the
    # way *out* (agent.data) from what would otherwise already happen on the
    # way in (agent.tool). The email goes in `subject`, not `details`:
    # `_file_grievance` echoes `subject` back in its own JSON response but
    # never includes `details` at all — only `subject` actually reaches the
    # agent.data surface for this tool.
    _patch(client, {"pii.action.agent_tool": "pass", "pii.action.agent_data": action,
                    "agent.prefilter_mode": "off", "agent.data_check_mode": "rail"})
    # EMAIL_ADDRESS is judge-only now — no deterministic layer exists — so
    # the stub is scripted to report it the same way a real judge call would.
    _install_scripted_llm([
        ("tool", "file_grievance",
         {"subject": f"Billing dispute — contact {EMAIL}",
          "details": "Please investigate the duplicate charge."}),
        ("answer", "Noted, thank you."),
    ], entities=[{"text": EMAIL, "kind": "EMAIL_ADDRESS", "confidence": 0.9}])

    resp = client.post("/api/agent/chat",
                       json={"message": "file a grievance about a billing error"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["approval"] is not None, "file_grievance always stops for approval"
    token = body["approval"]["token"]

    approve_resp = client.post("/api/agent/approve",
                               json={"token": token, "approved": True})
    assert approve_resp.status_code == 200, approve_resp.text
    approved_body = approve_resp.json()
    call = approved_body["calls"][0]
    assert call["name"] == "file_grievance"
    assert call["result_verdict"] == action, \
        f"the result rail's verdict must track the configured action, got {call!r}"

    if action == "block":
        assert call["result_preview"] == "(withheld)"
        assert EMAIL not in str(approved_body.get("reply", ""))
    elif action == "mask":
        assert EMAIL not in call["result_preview"], \
            "a masked tool result must not still show the raw value in the trace"
    else:  # flag, pass — neither rewrites the result
        assert EMAIL in call["result_preview"]


# ── agent.data_check_mode: PATCH -> reload -> the agentic specialists ───
def test_agent_data_check_mode_agentic_through_the_real_api(client):
    """Same shape as `test_agent_data_action_changes_real_tool_result_behaviour`,
    but flips the *mode* rather than the fixed rail's own action: with
    `agent.data_check_mode=agentic`, pii_agent — not `engine.evaluate()` —
    decides what happens to the email in `file_grievance`'s echoed result."""
    _patch(client, {"pii.action.agent_tool": "pass", "agent.data_check_mode": "agentic",
                    "agent.prefilter_mode": "off"})
    # `agentic_pii="MASK"` scripts PIIAgent's own decision; separately,
    # PIICapabilities.execute()'s actual substitution still goes through
    # EntityRail (judge-only now), so the span itself is scripted too.
    _install_scripted_llm(
        [("tool", "file_grievance",
          {"subject": f"Billing dispute — contact {EMAIL}",
           "details": "Please investigate the duplicate charge."}),
         ("answer", "Noted, thank you.")],
        agentic_pii="MASK",
        entities=[{"text": EMAIL, "kind": "EMAIL_ADDRESS", "confidence": 0.9}],
    )

    resp = client.post("/api/agent/chat",
                       json={"message": "file a grievance about a billing error"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["approval"] is not None, "file_grievance always stops for approval"

    approve_resp = client.post("/api/agent/approve",
                               json={"token": body["approval"]["token"], "approved": True})
    assert approve_resp.status_code == 200, approve_resp.text
    call = approve_resp.json()["calls"][0]
    assert call["name"] == "file_grievance"
    assert call["result_verdict"] == "mask"
    assert EMAIL not in call["result_preview"]

    # leave the mode back at its default for tests below
    _patch(client, {"agent.data_check_mode": "rail"})


# ── the two surfaces are independently adjustable ────────────────────────
def test_agent_tool_and_agent_data_are_set_independently(client):
    """Changing one must not move the other — each is its own registry key,
    its own override, its own read at the call site."""
    _patch(client, {"pii.action.agent_tool": "block"})
    snap = client.get("/api/parameters").json()["current"]
    assert snap["pii.action.agent_tool"] == "block"
    assert snap["pii.action.agent_data"] == "mask", "unrelated key must be untouched"

    _patch(client, {"pii.action.agent_data": "flag"})
    snap = client.get("/api/parameters").json()["current"]
    assert snap["pii.action.agent_tool"] == "block", "unrelated key must still be untouched"
    assert snap["pii.action.agent_data"] == "flag"
