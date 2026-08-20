"""Config loading, validation, and overrides.

Two files, deliberately:

  config/policy.yaml      the checked-in baseline. Hand-written, commented,
                          never machine-written.
  config/overrides.yaml   generated. Only the keys the console changed.

`load()` layers overrides on top of the baseline. That keeps the comments in
policy.yaml intact, makes "what did we change from baseline" a one-file diff,
and makes reset a delete.

Validation against `registry.py` is fatal in two cases, both on purpose:

  1. An unknown key — a typo'd threshold that silently does nothing is worse
     than a crash, because the rail looks configured and isn't.
  2. A locked parameter — with the message explaining why it is locked.

Note on `policy.runtime_override`: that lock forbids changing a rail through a
*request parameter*. Writing a validated config file, recording the change, and
reloading is the sanctioned path — it has an author, a diff, and an audit entry.
`save_overrides()` is that path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import (
    ADJUSTABLE,
    BY_KEY,
    LOCK_META,
    SEVERITY_KEYS,
    SEVERITY_SCALE,
    SURFACE_KEYS,
    defaults,
)


class ConfigError(ValueError):
    pass


OVERRIDES_HEADER = """\
# GENERATED FILE — written by the Guardrail Console.
#
# Only parameters that differ from config/policy.yaml appear here, so this file
# is the diff between your baseline and what is running right now.
#
# Safe to delete: removing it reverts every parameter to the baseline. Safe to
# commit: it is the record of what a deployment actually changed.
"""


@dataclass
class Policy:
    """Resolved, validated configuration."""

    values: dict[str, Any] = field(default_factory=dict)
    matrix: dict[str, dict[str, str]] = field(default_factory=dict)
    lexicons: dict[str, list[str]] = field(default_factory=dict)
    source: str = ""
    overrides_path: str = ""
    overridden: set[str] = field(default_factory=set)
    matrix_overridden: set[str] = field(default_factory=set)
    baseline_values: dict[str, Any] = field(default_factory=dict)
    baseline_matrix: dict[str, dict[str, str]] = field(default_factory=dict)

    # ---- access ------------------------------------------------------
    def get(self, key: str, fallback: Any = None) -> Any:
        if key in self.values:
            return self.values[key]
        if key in ADJUSTABLE:
            return ADJUSTABLE[key].default
        return fallback

    def severity(self, family: str, surface: str) -> str:
        return self.matrix.get(family, {}).get(surface, "medium")

    def severity_multiplier(self, family: str, surface: str) -> float:
        """`high` means stricter, so it lowers the effective threshold."""
        return SEVERITY_SCALE.get(self.severity(family, surface), 1.0)

    def enabled(self, family: str, surface: str) -> bool:
        return self.severity(family, surface) != "off"

    def threshold(self, key: str, family: str, surface: str) -> float:
        """Registry threshold scaled by the matrix cell, clamped to [0, 1]."""
        base = float(self.get(key))
        mult = self.severity_multiplier(family, surface)
        if mult < 0:
            return 2.0  # unreachable — family disabled on this surface
        return max(0.0, min(1.0, base * mult))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "matrix": self.matrix,
            "source": self.source,
            "overrides_path": self.overrides_path,
            "overridden": sorted(self.overridden),
            "matrix_overridden": sorted(self.matrix_overridden),
            "baseline": self.baseline_values,
            "baseline_matrix": self.baseline_matrix,
            "lexicon_sizes": {k: len(v) for k, v in self.lexicons.items()},
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Nested YAML → dotted keys, stopping at list/scalar leaves."""
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
    return out


def coerce(param_key: str, raw: Any) -> Any:
    """Validate and convert one value against its registry declaration."""
    if param_key in BY_KEY and param_key not in ADJUSTABLE:
        p = BY_KEY[param_key]
        meta = LOCK_META[p.lock.value]
        raise ConfigError(
            f"{param_key} is not adjustable — it is a {meta['label'].lower()} "
            f"fixed at {p.value!r}.\n  Why: {p.why}"
        )
    if param_key not in ADJUSTABLE:
        raise ConfigError(
            f"unknown parameter {param_key!r}. A typo'd key would silently do "
            "nothing, so it is rejected here instead."
        )

    p = ADJUSTABLE[param_key]
    try:
        if p.type == "float":
            v: Any = float(raw)
        elif p.type == "int":
            v = int(raw)
        elif p.type == "bool":
            v = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
        elif p.type in ("string[]", "regex[]", "enum[]", "set", "ruleset"):
            if raw is None:
                v = []
            elif isinstance(raw, list):
                v = [str(x) for x in raw]
            else:
                raise ConfigError(f"{param_key}: expected a list, got {type(raw).__name__}")
        else:
            v = raw
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{param_key}: cannot read {raw!r} as {p.type}") from exc

    if p.type in ("float", "int"):
        if p.minimum is not None and v < p.minimum:
            raise ConfigError(f"{param_key}: {v} is below the minimum {p.minimum}")
        if p.maximum is not None and v > p.maximum:
            raise ConfigError(f"{param_key}: {v} is above the maximum {p.maximum}")
    if p.type == "enum" and p.options and v not in p.options:
        raise ConfigError(f"{param_key}: {v!r} is not one of {p.options}")
    if p.type == "regex[]":
        import re

        for i, pattern in enumerate(v):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(f"{param_key}[{i}]: not a valid regex — {exc}") from exc
    return v


def _validate_matrix(matrix: dict[str, Any], where: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for family, row in (matrix or {}).items():
        if not isinstance(row, dict):
            raise ConfigError(f"{where}.{family}: expected a mapping of surface → level")
        clean: dict[str, str] = {}
        for surface, level in row.items():
            # YAML 1.1 reads a bare `off` as boolean False (and `on` as True).
            # `off` is a real severity level, so accept what the parser returns
            # rather than making everyone quote it.
            if level is False:
                level = "off"
            elif level is True:
                raise ConfigError(
                    f"{where}.{family}.{surface}: YAML read this as boolean true. "
                    "Did you mean 'high'? Quote the value for a literal string."
                )
            if surface not in SURFACE_KEYS:
                raise ConfigError(
                    f"{where}.{family}.{surface}: unknown surface. "
                    f"Valid surfaces: {', '.join(SURFACE_KEYS)}"
                )
            if level not in SEVERITY_KEYS:
                raise ConfigError(
                    f"{where}.{family}.{surface}: {level!r} is not one of {SEVERITY_KEYS}"
                )
            clean[surface] = level
        out[family] = clean
    return out


def _load_lexicon(base: Path, name: str) -> list[str]:
    path = base / "lexicons" / f"{name}.txt"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def overrides_path_for(policy_path: Path) -> Path:
    return policy_path.parent / "overrides.yaml"


def load(path: str | os.PathLike[str] | None = None) -> Policy:
    path = Path(path or os.getenv("GUARDRAIL_CONFIG", "config/policy.yaml"))
    if not path.exists():
        raise ConfigError(f"config not found: {path}")

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_matrix = doc.pop("severity_matrix", {}) or {}
    flat = _flatten(doc)

    values = defaults()
    for key, raw in flat.items():
        values[key] = coerce(key, raw)
    matrix = _validate_matrix(raw_matrix, "severity_matrix")

    baseline_values = dict(values)
    baseline_matrix = {f: dict(r) for f, r in matrix.items()}

    # ---- overrides layer -------------------------------------------------
    ov_path = overrides_path_for(path)
    overridden: set[str] = set()
    matrix_overridden: set[str] = set()
    if ov_path.exists():
        ov = yaml.safe_load(ov_path.read_text(encoding="utf-8")) or {}
        for key, raw in (ov.get("values") or {}).items():
            values[key] = coerce(key, raw)
            overridden.add(key)
        for family, row in _validate_matrix(
            ov.get("severity_matrix") or {}, "overrides.severity_matrix"
        ).items():
            matrix.setdefault(family, {}).update(row)
            for surface in row:
                matrix_overridden.add(f"{family}.{surface}")

    base = path.parent
    # words.profanity.enabled gates the shipped base lexicon only. Your own
    # custom terms are always loaded — turning off the shared baseline should
    # not silently discard rules you wrote yourself.
    baseline = _load_lexicon(base, "blocklist") if values.get("words.profanity.enabled") else []
    lexicons = {
        "blocklist": baseline
                     + list(values.get("words.custom_terms") or [])
                     + list(values.get("words.custom_phrases") or []),
        "allowlist": _load_lexicon(base, "allowlist") + list(values.get("words.allowlist") or []),
    }

    return Policy(
        values=values, matrix=matrix, lexicons=lexicons,
        source=str(path), overrides_path=str(ov_path),
        overridden=overridden, matrix_overridden=matrix_overridden,
        baseline_values=baseline_values, baseline_matrix=baseline_matrix,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def save_overrides(policy: Policy, values: dict[str, Any] | None = None,
                   matrix: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """Validate a change set and write it to overrides.yaml.

    Returns a summary of what changed. Every value is validated against the
    registry *before* anything is written — a rejected change leaves the running
    config untouched.

    A key set back to its baseline value is removed from the overrides file
    rather than recorded as an override, so the file stays a true diff.
    """
    ov_path = Path(policy.overrides_path or overrides_path_for(Path(policy.source)))
    existing = {}
    if ov_path.exists():
        existing = yaml.safe_load(ov_path.read_text(encoding="utf-8")) or {}

    out_values: dict[str, Any] = dict(existing.get("values") or {})
    out_matrix: dict[str, dict[str, str]] = {
        f: dict(r) for f, r in (existing.get("severity_matrix") or {}).items()
    }
    changes: list[dict[str, Any]] = []

    # Validate everything first.
    staged = {key: coerce(key, raw) for key, raw in (values or {}).items()}
    staged_matrix = _validate_matrix(matrix or {}, "severity_matrix")

    for key, value in staged.items():
        before = policy.get(key)
        baseline = policy.baseline_values.get(key, ADJUSTABLE[key].default)
        if value == baseline:
            out_values.pop(key, None)
        else:
            out_values[key] = value
        if value != before:
            changes.append({"key": key, "from": before, "to": value})

    for family, row in staged_matrix.items():
        for surface, level in row.items():
            before = policy.severity(family, surface)
            baseline = policy.baseline_matrix.get(family, {}).get(surface)
            if level == baseline:
                out_matrix.get(family, {}).pop(surface, None)
                if family in out_matrix and not out_matrix[family]:
                    out_matrix.pop(family)
            else:
                out_matrix.setdefault(family, {})[surface] = level
            if level != before:
                changes.append({
                    "key": f"severity_matrix.{family}.{surface}",
                    "from": before, "to": level,
                })

    body: dict[str, Any] = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if out_values:
        body["values"] = dict(sorted(out_values.items()))
    if out_matrix:
        body["severity_matrix"] = {f: dict(sorted(r.items())) for f, r in sorted(out_matrix.items())}

    ov_path.parent.mkdir(parents=True, exist_ok=True)
    if len(body) == 1:  # nothing but the timestamp — no overrides remain
        ov_path.unlink(missing_ok=True)
    else:
        ov_path.write_text(
            OVERRIDES_HEADER + "\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    return {
        "changes": changes,
        "overrides_remaining": len(out_values) + sum(len(r) for r in out_matrix.values()),
        "path": str(ov_path),
    }


def reset_overrides(policy: Policy) -> dict[str, Any]:
    """Delete overrides.yaml — every parameter returns to the baseline."""
    ov_path = Path(policy.overrides_path or overrides_path_for(Path(policy.source)))
    existed = ov_path.exists()
    ov_path.unlink(missing_ok=True)
    return {"reset": existed, "path": str(ov_path)}
