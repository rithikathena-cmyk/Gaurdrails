"""Config loading, validation, and the overrides layer."""

from __future__ import annotations

import pytest
import yaml

from backend.guardrails import load
from backend.guardrails.config import ConfigError, reset_overrides, save_overrides


# ── loading ────────────────────────────────────────────────────────
def test_real_config_loads(policy):
    assert policy.get("prompt_attack.threshold") == 0.85
    assert policy.severity("pii", "user.prompt") == "high"


def test_unknown_key_is_fatal(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("content:\n  hate:\n    thresold: 0.7\n")   # typo
    with pytest.raises(ConfigError, match="unknown parameter"):
        load(f)


def test_locked_key_is_rejected_with_its_reason(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("policy:\n  verdict_precedence: pass > block\n")
    with pytest.raises(ConfigError, match="not adjustable") as exc:
        load(f)
    assert "safety invariant" in str(exc.value)
    assert "Why:" in str(exc.value)


def test_out_of_range_is_rejected(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("content:\n  hate:\n    threshold: 1.8\n")
    with pytest.raises(ConfigError, match="above the maximum"):
        load(f)


def test_bad_enum_is_rejected(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("words:\n  action: explode\n")
    with pytest.raises(ConfigError, match="is not one of"):
        load(f)


def test_custom_patterns_accepts_arbitrary_text_without_validation(tmp_path):
    """`pii.custom_regex` (compiled, validated at load time) was replaced by
    `pii.custom_patterns` — descriptive hints folded into the judge's prompt,
    not a compiled pattern, so there is nothing left here to reject at load
    time. Malformed-regex-looking text loads clean; recognising it (or not)
    is the judge's job now, not this config's."""
    f = tmp_path / "p.yaml"
    f.write_text("pii:\n  custom_patterns: ['([unclosed']\n")
    policy = load(f)
    assert policy.get("pii.custom_patterns") == ["([unclosed"]


def test_unknown_surface_is_rejected(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("severity_matrix:\n  pii:\n    user.propmt: high\n")
    with pytest.raises(ConfigError, match="unknown surface"):
        load(f)


def test_yaml_reads_bare_off_as_false_and_we_accept_it(tmp_path):
    """YAML 1.1 turns `off` into boolean False. `off` is a real level."""
    f = tmp_path / "p.yaml"
    f.write_text("severity_matrix:\n  pii:\n    retrieval: off\n")
    assert load(f).severity("pii", "retrieval") == "off"


def test_bare_on_gets_a_helpful_error(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("severity_matrix:\n  pii:\n    retrieval: on\n")
    with pytest.raises(ConfigError, match="Did you mean 'high'"):
        load(f)


# ── severity matrix ────────────────────────────────────────────────
def test_high_severity_tightens_the_threshold(policy):
    scaled = policy.threshold("content.hate.threshold", "content", "user.prompt")  # high
    assert scaled < float(policy.get("content.hate.threshold"))


def test_medium_severity_leaves_the_threshold_alone(policy):
    base = float(policy.get("content.hate.threshold"))
    scaled = policy.threshold("content.hate.threshold", "content", "user.feedback")  # medium
    assert scaled == pytest.approx(base)


def test_low_severity_loosens_the_threshold(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(
        "content:\n  hate:\n    threshold: 0.60\n"
        "severity_matrix:\n  content:\n    user.prompt: low\n"
    )
    p = load(f)
    assert p.threshold("content.hate.threshold", "content", "user.prompt") == pytest.approx(0.78)


def test_scaled_thresholds_are_clamped_to_one(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(
        "content:\n  hate:\n    threshold: 0.95\n"
        "severity_matrix:\n  content:\n    user.prompt: low\n"
    )
    assert load(f).threshold("content.hate.threshold", "content", "user.prompt") == 1.0


def test_off_disables_the_family(policy):
    assert policy.enabled("grounding", "user.prompt") is False
    assert policy.enabled("grounding", "llm.response") is True


def test_disabled_family_gets_an_unreachable_threshold(policy):
    assert policy.threshold("content.hate.threshold", "content", "retrieval") > 1.0


# ── overrides ──────────────────────────────────────────────────────
def test_save_writes_only_the_diff(sandbox, sandbox_policy):
    save_overrides(sandbox_policy, {"content.hate.threshold": 0.42})
    body = yaml.safe_load((sandbox / "overrides.yaml").read_text())
    assert body["values"] == {"content.hate.threshold": 0.42}


def test_baseline_file_is_never_machine_written(sandbox, sandbox_policy):
    before = (sandbox / "policy.yaml").read_text()
    save_overrides(sandbox_policy, {"content.hate.threshold": 0.42})
    assert (sandbox / "policy.yaml").read_text() == before


def test_overrides_apply_on_reload(sandbox, sandbox_policy):
    save_overrides(sandbox_policy, {"content.hate.threshold": 0.42})
    reloaded = load(sandbox / "policy.yaml")
    assert reloaded.get("content.hate.threshold") == 0.42
    assert "content.hate.threshold" in reloaded.overridden
    assert reloaded.baseline_values["content.hate.threshold"] == 0.70


def test_setting_a_value_back_to_baseline_removes_the_override(sandbox, sandbox_policy):
    save_overrides(sandbox_policy, {"content.hate.threshold": 0.42})
    p = load(sandbox / "policy.yaml")
    save_overrides(p, {"content.hate.threshold": 0.70})
    assert not (sandbox / "overrides.yaml").exists()


def test_matrix_overrides_round_trip(sandbox, sandbox_policy):
    save_overrides(sandbox_policy, matrix={"pii": {"user.prompt": "low"}})
    reloaded = load(sandbox / "policy.yaml")
    assert reloaded.severity("pii", "user.prompt") == "low"
    assert "pii.user.prompt" in reloaded.matrix_overridden


def test_save_reports_what_changed(sandbox_policy):
    summary = save_overrides(sandbox_policy, {"words.action": "block"})
    assert summary["changes"] == [{"key": "words.action", "from": "mask", "to": "block"}]


def test_invalid_change_is_rejected_before_anything_is_written(sandbox, sandbox_policy):
    with pytest.raises(ConfigError, match="above the maximum"):
        save_overrides(sandbox_policy, {"content.hate.threshold": 9.0})
    assert not (sandbox / "overrides.yaml").exists()


def test_locked_key_cannot_be_saved(sandbox_policy):
    with pytest.raises(ConfigError, match="not adjustable"):
        save_overrides(sandbox_policy, {"policy.verdict_precedence": "pass"})


def test_reset_removes_every_override(sandbox, sandbox_policy):
    save_overrides(sandbox_policy, {"content.hate.threshold": 0.42},
                   {"pii": {"user.prompt": "low"}})
    p = load(sandbox / "policy.yaml")
    assert p.overridden and p.matrix_overridden

    reset_overrides(p)
    clean = load(sandbox / "policy.yaml")
    assert clean.overridden == set()
    assert clean.get("content.hate.threshold") == 0.70
    assert clean.severity("pii", "user.prompt") == "high"
