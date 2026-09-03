"""The `GuardrailSupervisor` MVP's flat tool allowlist.

Every function here wraps something that already exists — a deterministic
rail, or a `_private` function from one of the six specialist agents' own
tool modules (`tools.py`, `injection_tools.py`, `scope_tools.py`,
`content_tools.py`). Nothing here re-implements regex, checksum, NER, or
policy-lookup logic; this module's only job is to expose the exact six flat
names an autonomous-guardrail-supervisor spec asks for, in the single-hop
shape it asks for — one supervisor calling one tool directly, rather than
one supervisor selecting a specialist agent that then runs its own nested
PLAN/DECIDE loop before reaching the same detector.

`ALLOWED_GUARDRAIL_TOOLS` is the boundary. `call()` looks a name up in a
plain dict; there is no `getattr`, no `eval`, no dynamic import anywhere in
this file. An unknown name raises `ToolNotAllowed` — the exact exception
`agents/tools.py` already defines and every existing agent's tool module
already raises, reused rather than redefined, so a caller catching one
catches both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..engine import PII_ACTION_KEY, Engine
from ..types import RailResult, Surface, Verdict
from . import content_tools, injection_tools, scope_tools
from .tools import ToolNotAllowed
from .tools import _detect_pii_entities, _detect_pii_presidio
from .types import ToolResult


@dataclass(frozen=True)
class GuardrailTool:
    name: str
    fn: Callable[[dict, Engine], dict]


# ---------------------------------------------------------------------------
# The six tools. Each takes (args, engine) and returns a plain,
# JSON-compatible dict — structured evidence for the supervisor to reason
# over, never a raw matched value and never a decision.
# ---------------------------------------------------------------------------
def _detect_pii(args: dict, engine: Engine) -> dict:
    """Local NER (Presidio) + the free-form judge, merged — the same two
    layers `PIIAgent` calls separately, combined here into one flat result.
    No deterministic regex/checksum layer exists any more; every kind,
    including what used to have a fixed shape (an email, a phone number, a
    national ID), is judge-only now."""
    text = str(args.get("text", ""))
    presidio = _detect_pii_presidio({"text": text}, engine)
    entities = _detect_pii_entities({"text": text}, engine)
    findings = presidio["findings"] + entities["findings"]
    types = sorted({f["kind"] for f in findings})
    confidence = max((f["confidence"] for f in findings), default=0.0)
    return {
        "tool": "detect_pii", "detected": bool(types), "types": types,
        "confidence": round(confidence, 3),
    }


def _detect_prompt_injection(args: dict, engine: Engine) -> dict:
    """The deterministic pattern layer first (`content.INJECTION_PATTERNS`,
    via `injection_tools._detect_injection_patterns`) — free, and what
    catches the common case outright. Only if it finds nothing does this
    fall back to the local classifier's score, exactly the precedence the
    production `PromptAttackRail` already applies."""
    text = str(args.get("text", ""))
    patterns = injection_tools._detect_injection_patterns({"text": text}, engine)  # noqa: SLF001
    matches = patterns["matches"]
    if matches:
        confidence = max(m["confidence"] for m in matches)
        techniques = sorted({m["technique"] for m in matches})
        return {
            "tool": "detect_prompt_injection", "detected": True,
            "types": techniques, "confidence": round(confidence, 3),
        }
    local = injection_tools._classify_injection({"text": text}, engine)  # noqa: SLF001
    score = local.get("local_score")
    threshold = float(engine.policy.get("prompt_attack.threshold"))
    detected = score is not None and score >= threshold
    return {
        "tool": "detect_prompt_injection", "detected": bool(detected),
        "types": ["local_classifier"] if detected else [],
        "confidence": round(score, 3) if score is not None else 0.0,
    }


def _detect_destructive_intent(args: dict, engine: Engine) -> dict:
    """Re-runs the existing `PolicyRail` — `config/policy.yaml`'s named
    regex rule sets (`security_rules`, `use_case_rules`, ...) — unchanged.
    No new detection engine: a destructive or capability-escalation phrase
    is exactly what those rule sets already catch for nothing, on every
    deployment, with or without an API key."""
    text = str(args.get("text", ""))
    result = RailResult(rail=engine.policy_rail.name, engine=engine.policy_rail.engine,
                        verdict=Verdict.PASS)
    out = engine.policy_rail.evaluate(text, result)
    fired = sorted({d.kind for d in out.detections})
    detected = out.verdict is not Verdict.PASS
    return {
        "tool": "detect_destructive_intent", "detected": detected,
        "types": fired,
        # A rule either matched or it did not — the same reasoning
        # `registry.py`'s `adjudicator.deterministic_rails` already states
        # for why a regex hit is never a fuzzy score.
        "confidence": 1.0 if detected else 0.0,
    }


def _check_scope(args: dict, engine: Engine) -> dict:
    """The deterministic vocabulary pass — `ScopeRail._hits`, via
    `scope_tools._check_domain_vocabulary` — unchanged."""
    text = str(args.get("text", ""))
    out = scope_tools._check_domain_vocabulary({"text": text}, engine)  # noqa: SLF001
    return {
        "tool": "check_scope", "in_scope": bool(out["in_vocabulary"]),
        "matched_terms": out["matched_terms"],
    }


def _check_semantic_risk(args: dict, engine: Engine) -> dict:
    """The local content-safety classifier — `toxicity_check.score`, via
    `content_tools._score_content_categories` — unchanged. Reports per-category
    scores plus the worst one, never a verdict; deciding is the supervisor's
    job, not this tool's."""
    text = str(args.get("text", ""))
    out = content_tools._score_content_categories({"text": text}, engine)  # noqa: SLF001
    scores: dict[str, float] = out.get("scores") or {}
    worst_category, worst_score = max(scores.items(), key=lambda kv: kv[1], default=(None, 0.0))
    return {
        "tool": "check_semantic_risk", "available": bool(out.get("available")),
        "scores": scores, "worst_category": worst_category,
        "max_score": round(worst_score, 3),
    }


#: The recognized top-level policy keys `get_policy` understands. Not the
#: raw `config/policy.yaml` dotted-key namespace — a structured vocabulary
#: matched to the five other flat tools, so PLAN can ask "what does the PII
#: policy say" without knowing that means `pii.action.user_prompt` plus
#: `pii.mask_strategy` plus `pii.reversible` under the hood.
POLICY_KEYS: frozenset[str] = frozenset({
    "pii", "injection", "destructive_intent", "scope", "semantic_risk",
})


def _resolve_surface(raw: str) -> Surface:
    try:
        return Surface(raw)
    except ValueError:
        return Surface.USER_PROMPT


def _policy_pii(entity: str, surface: Surface, engine: Engine) -> dict:
    """Same configured facts `agents/tools.py`'s `_get_pii_policy` already
    exposes to the PII agent — read here for a `surface`-aware, optionally
    entity-scoped lookup rather than a single required `kind`."""
    action = str(engine.policy.get(PII_ACTION_KEY[surface]))
    entities = set(engine.policy.get("pii.entity_kinds") or [])
    out: dict = {
        "tool": "get_policy", "policy": f"pii.{entity}" if entity else "pii", "valid": True,
        "surface": surface.value, "action": action,
        "mask_strategy": str(engine.policy.get("pii.mask_strategy")),
        "reversible": bool(engine.policy.get("pii.reversible")),
    }
    if entity:
        out["entity"] = entity
        out["entity_enabled"] = entity in entities
    else:
        out["entities_enabled"] = sorted(entities)
    return out


def _policy_injection(engine: Engine) -> dict:
    """Same facts `agents/injection_tools.py`'s `_get_injection_policy`
    already exposes to the injection agent."""
    return {
        "tool": "get_policy", "policy": "injection", "valid": True,
        "threshold": float(engine.policy.get("prompt_attack.threshold")),
        "action": str(engine.policy.get("prompt_attack.action")),
        "engine_mode": str(engine.policy.get("prompt_attack.engine")),
    }


def _policy_destructive_intent(engine: Engine) -> dict:
    """`detect_destructive_intent` has no single adjustable threshold or
    action — `config/policy.yaml`'s `security_rules`/`use_case_rules` each
    carry their own `pattern => action`. What *is* genuinely configured, and
    genuinely worth reporting, is how many rules are loaded per set —
    `engine.policy_rail.counts`, computed once at startup, not re-derived
    here."""
    rail = engine.policy_rail
    return {
        "tool": "get_policy", "policy": "destructive_intent", "valid": True,
        "rule_sets_loaded": dict(rail.counts), "total_rules": len(rail.rules),
    }


def _policy_scope(engine: Engine) -> dict:
    """Same facts `agents/scope_tools.py`'s `_get_scope_policy` already
    exposes to the scope agent."""
    return {
        "tool": "get_policy", "policy": "scope", "valid": True,
        "threshold": float(engine.policy.get("scope.threshold")),
        "action": str(engine.policy.get("scope.action")),
        "domain_terms_configured": len(engine.policy.get("scope.domain_terms") or []),
    }


def _policy_semantic_risk(category: str, engine: Engine) -> dict:
    """Same facts `agents/content_tools.py`'s `_get_content_policy` already
    exposes to the content-safety agent, keyed the same optional-category
    way that function already is."""
    from ..rails.content import CATEGORIES

    enabled = [c for c in engine.policy.get("content.enabled_categories") or [] if c in CATEGORIES]
    if category and category not in CATEGORIES:
        return {
            "tool": "get_policy", "policy": f"semantic_risk.{category}", "valid": False,
            "error": f"{category!r} is not a recognized content category",
            "valid_categories": sorted(CATEGORIES),
        }
    out: dict = {
        "tool": "get_policy",
        "policy": f"semantic_risk.{category}" if category else "semantic_risk",
        "valid": True, "action": str(engine.policy.get("content.action.user_prompt")),
        "enabled_categories": enabled,
    }
    if category:
        out["category"] = category
        out["threshold"] = float(engine.policy.get(f"content.{category}.threshold"))
        out["enabled"] = category in enabled
    return out


def _get_policy(args: dict, engine: Engine) -> dict:
    """A structured, policy-aware lookup — `POLICY_KEYS` (optionally
    `<key>.<sub-key>`, e.g. `pii.US_SSN`), not a raw `config/policy.yaml`
    dotted path. Every branch reads the live `Policy` object directly
    (`engine.policy.get(...)`, `engine.policy_rail`); nothing here infers a
    policy from the request text, and there is no argument shape that can
    set a value — every path is a read.
    """
    policy = str(args.get("policy", "")).strip()
    surface = _resolve_surface(str(args.get("surface", "user.prompt")))

    if not policy:
        return {"tool": "get_policy", "policy": "", "valid": False,
                "error": "no policy key given", "valid_keys": sorted(POLICY_KEYS)}

    base, _, sub = policy.partition(".")
    if base == "pii":
        return _policy_pii(sub, surface, engine)
    if base == "injection":
        return _policy_injection(engine)
    if base == "destructive_intent":
        return _policy_destructive_intent(engine)
    if base == "scope":
        return _policy_scope(engine)
    if base == "semantic_risk":
        return _policy_semantic_risk(sub, engine)

    return {"tool": "get_policy", "policy": policy, "valid": False,
            "error": f"{policy!r} is not a recognized policy key",
            "valid_keys": sorted(POLICY_KEYS)}


GUARDRAIL_TOOLS: dict[str, GuardrailTool] = {
    "detect_pii": GuardrailTool("detect_pii", _detect_pii),
    "detect_prompt_injection": GuardrailTool("detect_prompt_injection", _detect_prompt_injection),
    "detect_destructive_intent": GuardrailTool("detect_destructive_intent", _detect_destructive_intent),
    "check_scope": GuardrailTool("check_scope", _check_scope),
    "check_semantic_risk": GuardrailTool("check_semantic_risk", _check_semantic_risk),
    "get_policy": GuardrailTool("get_policy", _get_policy),
}

#: The explicit allowlist. `GuardrailSupervisor` validates every tool name a
#: plan names against this before calling anything — a name outside it is
#: rejected in Python, never reached by `call()`.
ALLOWED_GUARDRAIL_TOOLS: frozenset[str] = frozenset(GUARDRAIL_TOOLS)


def call(name: str, args: dict, engine: Engine, call_id: str) -> ToolResult:
    """The only entry point. A name not in `GUARDRAIL_TOOLS` never reaches a
    function call — it raises `ToolNotAllowed` here, before any code the name
    might have named gets a chance to run."""
    tool = GUARDRAIL_TOOLS.get(name)
    if tool is None:
        raise ToolNotAllowed(name)
    began = time.perf_counter()
    try:
        result = tool.fn(args, engine)
        return ToolResult(call_id=call_id, tool=name, status="ok",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          result=result)
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        return ToolResult(call_id=call_id, tool=name, status="error",
                          duration_ms=(time.perf_counter() - began) * 1000,
                          error=f"{type(exc).__name__}: {exc}")
