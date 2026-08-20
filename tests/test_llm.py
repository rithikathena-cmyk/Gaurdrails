"""Request shaping for the Claude client.

No network. These pin the per-model parameter selection, because getting it
wrong is silent in the worst way: the judge call 400s, the rail fails closed,
and a clean request gets blocked with 0ms on the clock.
"""

from __future__ import annotations

import pytest

from backend.guardrails.llm import _tuning, supports_adaptive


@pytest.mark.parametrize("model", [
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5",
])
def test_modern_models_take_adaptive_thinking(model):
    assert supports_adaptive(model) is True


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-5", "claude-opus-4-1",
])
def test_older_models_do_not(model):
    assert supports_adaptive(model) is False


def test_modern_model_gets_thinking_and_effort():
    kw = _tuning("claude-opus-5", "low", {"type": "json_schema", "schema": {}})
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"


def test_haiku_gets_neither_thinking_nor_effort():
    """Regression: sending these to Haiku 4.5 returns a 400, the rail fails
    closed, and a clean question is blocked in under a second."""
    kw = _tuning("claude-haiku-4-5", "low", {"type": "json_schema", "schema": {}})
    assert "thinking" not in kw
    assert "effort" not in kw["output_config"]
    assert kw["output_config"]["format"]["type"] == "json_schema"


def test_structured_output_survives_on_every_model():
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    for model in ("claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"):
        assert _tuning(model, "low", fmt)["output_config"]["format"] == fmt


def test_generation_shape_has_no_format_block():
    kw = _tuning("claude-opus-5", "medium")
    assert "format" not in kw["output_config"]
    assert kw["output_config"]["effort"] == "medium"


def test_older_model_generation_sends_no_output_config():
    assert _tuning("claude-haiku-4-5", "medium") == {}


def test_a_key_with_a_trailing_newline_still_works(monkeypatch):
    """Pasting a key into a hosting dashboard picks up a newline more often
    than not, and httpx will not put a newline in a header. It surfaced as
    `APIConnectionError` — indistinguishable from a firewall — and every rail
    failed closed, so the deployment refused every request over one invisible
    character."""
    from backend.guardrails.llm import Claude

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key\n")
    assert Claude().client.api_key == "sk-ant-test-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-test-key  ")
    assert Claude().client.api_key == "sk-ant-test-key"


def test_a_key_that_is_only_whitespace_is_no_key_at_all(monkeypatch):
    from backend.guardrails.llm import Claude, LLMError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "   \n")
    with pytest.raises(LLMError, match="not set"):
        Claude()
