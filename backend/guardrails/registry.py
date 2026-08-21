"""The parameter registry.

Every knob in the system is declared here exactly once, and every knob is
either **adjustable** (you set it per deployment, in policy.yaml) or **locked**
(you cannot set it, and the registry records *why*).

The lock reasons are not decoration. They are four genuinely different
conversations:

  MODEL   — fixed by the weights or architecture underneath. Changing it means
            swapping models and re-baselining every threshold you tuned.
  SAFETY  — deliberately not tunable. An adjustable version of this parameter
            would be a bypass, so the bypass does not exist.
  ARCH    — determined by pipeline topology. Changing it means rewiring the
            stack, not editing a setting.
  COMPLY  — required by a regulation or an audit obligation. "Off" is not a
            legal option.

`config.py` validates policy.yaml against this registry and refuses to start if
the config tries to set a locked parameter. The API serves this registry to the
Parameters page, so the UI can never drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Lock(str, Enum):
    MODEL = "model"
    SAFETY = "safety"
    ARCH = "arch"
    COMPLY = "comply"


LOCK_META: dict[str, dict[str, str]] = {
    "model": {
        "label": "Model-bound",
        "glyph": "▣",
        "token": "--ink-3",
        "blurb": "Fixed by the weights or architecture underneath. Changing it means swapping models.",
    },
    "safety": {
        "label": "Safety invariant",
        "glyph": "▲",
        "token": "--block",
        "blurb": "Deliberately not tunable. An adjustable version of this would be a bypass.",
    },
    "arch": {
        "label": "Architectural",
        "glyph": "■",
        "token": "--mask",
        "blurb": "Determined by pipeline topology. Would require rewiring the stack, not a setting.",
    },
    "comply": {
        "label": "Compliance",
        "glyph": "●",
        "token": "--pass",
        "blurb": "Required by a regulation or audit obligation. Off is not a legal option.",
    },
}

# Surfaces a rail can evaluate on. The frontend renders the severity matrix
# from this list — adding a surface here adds a column everywhere.
SURFACES: list[dict[str, str]] = [
    {"key": "user.prompt", "label": "Prompt", "blurb": "Text on the way in from the user."},
    {"key": "user.feedback", "label": "Feedback", "blurb": "Corrections and follow-ups."},
    {"key": "ingest.document", "label": "Ingest",
     "blurb": "Documents entering the knowledge base. Scanned once, before indexing."},
    {"key": "retrieval", "label": "Retrieval", "blurb": "Chunks returned by the knowledge base."},
    {"key": "llm.response", "label": "Response", "blurb": "Generated text on the way out."},
    {"key": "llm.ask_user", "label": "Ask user", "blurb": "Clarifying questions the model asks."},
    {"key": "agent.tool", "label": "Tool call",
     "blurb": "Arguments the agent is about to send to a tool."},
    {"key": "agent.data", "label": "Tool result",
     "blurb": "What a tool returned, before the agent is allowed to read it."},
]

# Severity levels and what each does to a family's thresholds.
SEVERITY_LEVELS: list[dict[str, Any]] = [
    {"key": "high", "label": "High", "multiplier": 0.70, "token": "--block",
     "blurb": "Stricter — thresholds × 0.70"},
    {"key": "medium", "label": "Medium", "multiplier": 1.00, "token": "--mask",
     "blurb": "Baseline"},
    {"key": "low", "label": "Low", "multiplier": 1.30, "token": "--flag",
     "blurb": "Looser — thresholds × 1.30"},
    {"key": "off", "label": "Off", "multiplier": -1.0, "token": "--ink-3",
     "blurb": "Family does not run on that surface"},
]

SURFACE_KEYS: list[str] = [s["key"] for s in SURFACES]
SEVERITY_KEYS: list[str] = [s["key"] for s in SEVERITY_LEVELS]
SEVERITY_SCALE: dict[str, float] = {s["key"]: s["multiplier"] for s in SEVERITY_LEVELS}


@dataclass
class Param:
    key: str
    family: str
    desc: str
    type: str
    # adjustable
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: list[str] = field(default_factory=list)
    # locked
    lock: Lock | None = None
    value: str = ""
    why: str = ""

    @property
    def adjustable(self) -> bool:
        return self.lock is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key": self.key,
            "family": self.family,
            "desc": self.desc,
            "type": self.type,
            "adjustable": self.adjustable,
        }
        if self.adjustable:
            d.update(
                default=self.default,
                min=self.minimum,
                max=self.maximum,
                step=self.step,
                options=self.options,
            )
        else:
            d.update(lock=self.lock.value, value=self.value, why=self.why)
        return d


def _a(key, family, desc, type_, default, *, minimum=None, maximum=None, step=None, options=None):
    return Param(
        key=key,
        family=family,
        desc=desc,
        type=type_,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        options=options or [],
    )


def _l(key, family, desc, type_, lock, value, why):
    return Param(key=key, family=family, desc=desc, type=type_, lock=lock, value=value, why=why)


FAMILIES: dict[str, dict[str, str]] = {
    "adjudicator": {
        "name": "Adjudication",
        "engine": "claude · margin-triggered second opinion",
    },
    "content": {
        "name": "Content Guardrails",
        "engine": "claude judge (structured output) · pattern detectors",
    },
    "words": {
        "name": "Word Guardrails",
        "engine": "aho–corasick automaton · custom lexicons",
    },
    "pii": {
        "name": "Sensitive Information Guardrails",
        "engine": "regex recognizers + checksums · aes-gcm token vault",
    },
    "grounding": {
        "name": "Grounding Guardrails",
        "engine": "claude judge (structured output) · lexical overlap",
    },
    "policy": {
        "name": "Custom Controls & Policy",
        "engine": "yaml policy sets · severity matrix · verdict precedence",
    },
    "scope": {
        "name": "Scope Guardrails",
        "engine": "domain vocabulary → claude judge",
    },
    "ingest": {
        "name": "Document Ingestion",
        "engine": "extract · chunk · ingest rails · bm25 index",
    },
    "agent": {
        "name": "Agent & Tools",
        "engine": "claude tool use · per-call rails · approval gate",
    },
}


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
PARAMS: list[Param] = [
    # ---------------- content ----------------
    _a("content.hate.threshold", "content",
       "Score above which a hate-speech classification fires.",
       "float", 0.70, minimum=0, maximum=1, step=0.01),
    _a("content.violence.threshold", "content",
       "Violence and violent-crime hazard categories.",
       "float", 0.65, minimum=0, maximum=1, step=0.01),
    _a("content.insults.threshold", "content",
       "Personal attacks and demeaning language.",
       "float", 0.75, minimum=0, maximum=1, step=0.01),
    _a("content.misconduct.threshold", "content",
       "Requests to facilitate non-violent crime or fraud.",
       "float", 0.70, minimum=0, maximum=1, step=0.01),
    _a("content.self_harm.threshold", "content",
       "Set low deliberately — a false positive here costs far less than a miss.",
       "float", 0.40, minimum=0, maximum=1, step=0.01),
    _a("content.sexual.threshold", "content",
       "Sexual content, scaled for a public-service audience.",
       "float", 0.60, minimum=0, maximum=1, step=0.01),
    _a("content.action.user_prompt", "content",
       "What happens when a content rail fires on the way in.",
       "enum", "block", options=["block", "mask", "flag", "pass"]),
    _a("content.action.llm_response", "content",
       "What happens when it fires on the way out.",
       "enum", "regenerate", options=["block", "regenerate", "flag", "pass"]),
    _a("prompt_attack.threshold", "content",
       "Injection detector cutoff. Above ~0.9 you start missing multi-turn attacks.",
       "float", 0.85, minimum=0, maximum=1, step=0.01),
    _a("prompt_attack.action", "content",
       "Verdict when an injection attempt is detected.",
       "enum", "block", options=["block", "flag", "pass"]),
    _a("content.judge_model", "content",
       "Model backing the content rails. Judge quality is the rail's ceiling.",
       "enum", "claude-sonnet-5",
       options=["claude-sonnet-5", "claude-haiku-4-5"]),
    _a("content.enabled_categories", "content",
       "Which hazard categories to evaluate at all.",
       "set", ["hate", "violence", "insults", "misconduct", "self_harm", "sexual"]),

    _a("content.engine", "content",
       "What scores content safety. The local classifier is a few milliseconds of CPU "
       "and no API call; the judge is slower but reads intent. Together, the local "
       "layer settles the confident cases and the judge is asked about the rest.",
       "enum", "local+judge",
       options=["local+judge", "local", "judge", "off"]),
    _a("content.local_block_threshold", "content",
       "How sure the local classifier must be before it blocks without asking the "
       "judge. High on purpose: a short-circuit skips the layer that reads intent, so "
       "it should only fire where the classifier is not in any doubt.",
       "float", 0.90, minimum=0.5, maximum=1, step=0.01),
    _a("prompt_attack.engine", "content",
       "What detects injection, after the deterministic pattern layer. Patterns run "
       "first either way — this chooses what happens when they find nothing. "
       "Defaults to `judge`: the local classifier was measured scoring a legitimate "
       "question at 0.991 and a real attack at 1.000, so no threshold separates them.",
       "enum", "judge",
       options=["local+judge", "local", "judge", "off"]),
    _a("prompt_attack.local_block_threshold", "content",
       "How sure the local injection classifier must be before it blocks without the "
       "judge. Kept high because this model reports injection on text that merely "
       "discusses prompts — including a citizen asking why they were refused.",
       "float", 0.90, minimum=0.5, maximum=1, step=0.01),

    _l("content.local_short_circuit_scope", "content",
       "Which categories a local classifier may settle on its own.",
       "const", Lock.SAFETY,
       "block-direction only; never for misconduct or self_harm",
       "A local classifier may end a request early only by blocking it. It can never "
       "return a clean verdict that skips the judge, because 'this model saw nothing' "
       "and 'there is nothing here' are different claims. Two categories are excluded "
       "even from blocking: the Jigsaw-family taxonomy these models are trained on has "
       "no label for misconduct at all, and its self_harm coverage is too weak to carry "
       "a category whose threshold is deliberately the lowest of the six because a miss "
       "costs more than a false positive. Both keep semantic judge coverage."),
    _l("content.hazard_taxonomy", "content",
       "The category set the judge is allowed to emit.",
       "enum[6]", Lock.MODEL,
       "hate · violence · insults · misconduct · self_harm · sexual",
       "Defined by the judge's structured-output schema. A seventh category needs a new schema, "
       "a new prompt, and a fresh calibration run — not a config edit."),
    _l("content.score_range", "content",
       "Range every content score is reported in.",
       "const", Lock.MODEL, "[0.0, 1.0]",
       "Thresholds are only comparable because every detector normalises to the same range. "
       "The range is the contract."),
    _l("content.eval_order", "content",
       "Prompt rails always complete before dispatch.",
       "const", Lock.ARCH, "prompt → dispatch → output",
       "Evaluating after dispatch would mean the model already saw the input. There is no "
       "ordering in which an input rail can run late and still be an input rail."),
    _l("content.judge_isolation", "content",
       "The judge never sees prior conversation turns.",
       "const", Lock.SAFETY, "single-turn, no history",
       "History is attacker-controlled. A judge that reads it can be argued out of its own "
       "verdict across turns."),
    _l("content.model_version_pinning", "content",
       "Judge model is pinned per deployment.",
       "const", Lock.SAFETY, "pinned at deploy",
       "Silent model upgrades invalidate every threshold you tuned. Version moves are a "
       "deliberate release with a re-baseline, never a background change."),

    # ---------------- words ----------------
    _a("words.profanity.enabled", "words",
       "Base profanity lexicon.", "bool", True),
    _a("words.custom_terms", "words",
       "Your own blocked terms. No practical ceiling.", "string[]", []),
    _a("words.custom_phrases", "words",
       "Multi-word sequences — embargoed products, legacy branding, competitor names.",
       "string[]", []),
    _a("words.allowlist", "words",
       "Exemptions — terms that would otherwise trip the blocklist in your domain.",
       "string[]", []),
    _a("words.match_mode", "words",
       "How loosely a term matches.", "enum", "word",
       options=["exact", "word", "substring"]),
    _a("words.case_sensitive", "words",
       "Whether casing is significant when matching.", "bool", False),
    _a("words.action", "words",
       "Verdict when a term matches.", "enum", "mask",
       options=["block", "mask", "flag"]),

    _l("words.normalization", "words",
       "Unicode fold applied before matching.",
       "const", Lock.SAFETY, "NFKC + homoglyph fold, always on",
       "This is the single most common filter bypass. Making it optional would make the "
       "filter itself optional — an attacker just asks for it to be off."),
    _l("words.match_engine", "words",
       "Single-pass automaton over every pattern at once.",
       "const", Lock.ARCH, "Aho–Corasick, O(n + matches)",
       "Match cost is independent of list size. That property is exactly why the term list "
       "is allowed to be unbounded."),
    _l("words.list_precedence", "words",
       "Order in which blocklist and allowlist are applied.",
       "const", Lock.SAFETY, "blocklist → allowlist",
       "Reversing it lets one allowlist entry silently disable a blocked term, with no diff "
       "on the blocklist to show for it."),
    _l("words.max_pattern_length", "words",
       "Longest single pattern accepted.",
       "int", Lock.ARCH, "256 characters",
       "Automaton node budget. Anything longer is a semantic claim, not a lexical one, and "
       "belongs in the content or grounding rails."),

    # ---------------- pii ----------------
    _a("pii.entities", "pii",
       "Which entity types to detect.", "enum[]",
       ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS",
        "DATE_OF_BIRTH", "AADHAAR", "PAN", "IBAN"]),
    _a("pii.confidence_threshold", "pii",
       "Minimum recognizer confidence to count as a detection.",
       "float", 0.50, minimum=0, maximum=1, step=0.01),
    _a("pii.mask_strategy", "pii",
       "How a detected value is replaced.", "enum", "vault-token",
       options=["redact", "replace", "hash", "partial", "vault-token"]),
    _a("pii.partial_reveal", "pii",
       "Trailing characters left visible under partial masking.",
       "int", 4, minimum=0, maximum=4, step=1),
    _a("pii.reversible", "pii",
       "Whether an authorized caller can unmask at egress.", "bool", True),
    _a("pii.custom_regex", "pii",
       "Domain identifiers the built-ins don't cover — claim numbers, file refs.",
       "regex[]", []),
    _a("pii.allowlist", "pii",
       "Published contacts that are exempt from masking — a department's own address or "
       "helpline. Matched case-insensitively against the whole text, not against the "
       "detected value: a recognizer slices a span to its own boundaries, so the phone "
       "pattern takes 800 425 1969 out of the published 1800 425 1969 and a value-based "
       "entry would never match it. An exempt value is still detected, counted and "
       "audited — only the rewrite is skipped.",
       "regex[]", []),

    _l("pii.allowlist_ordering", "pii",
       "Detection runs first; the allowlist only exempts what was already found.",
       "const", Lock.SAFETY, "detect → exempt → mask",
       "An allowlist consulted before detection would suppress the match itself, making an "
       "exemption indistinguishable from a recognizer that failed. Exempt values are still "
       "detected, still counted, and still written to the audit entry."),

    _a("pii.action.user_prompt", "pii",
       "Applied to text on the way in.", "enum", "mask",
       options=["block", "mask", "flag", "pass"]),
    _a("pii.action.retrieval", "pii",
       "Applied to text from the knowledge base, not just user input.",
       "enum", "mask", options=["mask", "flag", "pass"]),
    _a("pii.action.llm_response", "pii",
       "Egress scan on generated text.", "enum", "mask",
       options=["block", "mask", "flag", "pass"]),
    _a("pii.entity_kinds", "pii",
       "Named entities a model looks for, on top of the regex recognizers. These "
       "are the identifiers a pattern cannot find — a person, a street address.",
       "set", ["PERSON", "ADDRESS", "ORGANISATION"]),
    _a("pii.entity_engine", "pii",
       "What finds names and addresses. Presidio is local NER — about a second of CPU "
       "and no API call; the judge is slower but reads context. Together, the judge is "
       "asked only when Presidio finds nothing.",
       "enum", "presidio+judge",
       options=["presidio+judge", "presidio", "judge", "off"]),

    _a("pii.entity_confidence", "pii",
       "How sure the entity model must be before a span is masked.",
       "float", 0.60, minimum=0, maximum=1, step=0.01),
    _a("pii.action.ingest", "pii",
       "Applied to a document as it is ingested, before it reaches the index.",
       "enum", "mask", options=["block", "mask", "flag"]),
    _a("pii.action.agent_tool", "pii",
       "Applied to the arguments the agent is about to hand a tool.",
       "enum", "mask", options=["block", "mask", "flag", "pass"]),
    _a("pii.action.agent_data", "pii",
       "Applied to what a tool returned, before the agent reads it.",
       "enum", "mask", options=["block", "mask", "flag", "pass"]),
    _a("pii.agent.allow_masked_pii_response", "pii",
       "Whether the autonomous agent path may return a MASK recommendation at all. "
       "False hands it to a person instead of returning a value with a vault token.",
       "bool", True),
    _a("pii.agent.preserve_masked_tokens", "pii",
       "Whether a masked agent response keeps the reversible vault token, or a "
       "generic, non-reversible marker. Only the token can ever be resolved back.",
       "bool", True),
    _a("pii.vault.resolution", "pii",
       "Whether the agentic path's own egress step may resolve a vault token in an "
       "agent's response back to its real value for the entitled reader — the same "
       "check `vault.unmask` already runs for ordinary chat, applied here. "
       "'owner_only' still denies every reader the token was not minted for; "
       "'never' means this path does not attempt resolution for anyone, ever.",
       "enum", "owner_only", options=["owner_only", "never"]),

    _l("pii.checksum_validation", "pii",
       "Structural validation on card, SSN, IBAN, Aadhaar, and PAN candidates.",
       "const", Lock.SAFETY, "Luhn / mod-97 / Verhoeff, always on",
       "Disabling it wouldn't catch more PII — it would flood the queue with false "
       "positives until somebody turns the whole rail off to stop the noise."),
    _l("pii.audit_log", "pii",
       "Every detection recorded before masking.",
       "const", Lock.COMPLY, "append-only, hash-chained",
       "You must be able to prove what was detected and when. Stored under a separate ACL "
       "from the response log so reading transcripts doesn't mean reading raw PII."),
    _l("pii.vault_encryption", "pii",
       "Cipher protecting reversible mask tokens.",
       "const", Lock.COMPLY, "AES-256-GCM",
       "Set by the control baseline you're certifying against, not by preference."),
    _l("pii.token_determinism", "pii",
       "Token format for a masked value.",
       "const", Lock.SAFETY, "random per occurrence",
       "A deterministic token is a stable identifier. Reusing one across requests lets an "
       "observer correlate users without ever seeing the underlying value."),
    _l("pii.detection_order", "pii",
       "Detection always precedes masking.",
       "const", Lock.ARCH, "detect → audit → mask",
       "The audit record has to capture what was there. Masking first would leave nothing "
       "to record."),

    # ---------------- grounding ----------------
    _a("grounding.consistency.threshold", "grounding",
       "Minimum factual-consistency score against retrieved context.",
       "float", 0.50, minimum=0, maximum=1, step=0.01),
    _a("grounding.relevance.threshold", "grounding",
       "Minimum relevance between the response and the question.",
       "float", 0.35, minimum=0, maximum=1, step=0.01),
    _a("grounding.context_window", "grounding",
       "How many retrieved chunks the check considers.",
       "int", 4, minimum=1, maximum=20, step=1),
    _a("grounding.action_on_fail", "grounding",
       "What happens to an ungrounded response.", "enum", "regenerate",
       options=["regenerate", "flag", "human_review", "block"]),
    _a("grounding.max_regenerations", "grounding",
       "Retries before escalating. Each one costs a full model call.",
       "int", 2, minimum=0, maximum=3, step=1),
    _a("grounding.engine", "grounding",
       "What scores factual consistency. A local NLI model can check a claim against "
       "the retrieved chunks without an API call, but it scores entailment only — "
       "relevance and the verbatim unsupported-claim list still need the judge.",
       "enum", "local+judge",
       options=["local+judge", "local", "judge", "off"]),
    _a("grounding.require_citations", "grounding",
       "Reject responses that assert without pointing at a source.", "bool", False),

    _a("adjudicator.enabled", "adjudicator",
       "Send marginal decisions to a model for a second opinion.", "bool", True),
    _a("adjudicator.margin", "adjudicator",
       "How close to its threshold a rail must land to count as marginal. "
       "Wider means more model calls on more requests.",
       "float", 0.08, minimum=0, maximum=0.3, step=0.01),
    _a("adjudicator.min_confidence", "adjudicator",
       "Confidence the adjudicator needs before it may lower a verdict. "
       "Raising a verdict needs none.",
       "float", 0.6, minimum=0, maximum=1, step=0.05),
    _a("adjudicator.rails", "adjudicator",
       "Which scored rails may be adjudicated.", "list",
       ["content.safety", "prompt_attack", "scope.domain", "grounding.consistency"]),

    _l("adjudicator.downgrade_floor", "adjudicator",
       "The lowest verdict a downgrade may reach.",
       "const", Lock.SAFETY, "flag",
       "A lowered verdict still has to leave a record. If the adjudicator could reach "
       "`pass`, one confident model call would erase an incident an operator never saw."),
    _l("adjudicator.deterministic_rails", "adjudicator",
       "Regex and lexicon rails are never adjudicated.",
       "const", Lock.SAFETY, "excluded",
       "A pasted credential or `drop table` has no ambiguous band — it matched or it did "
       "not. A model permitted to overrule a regex is a bypass, not a nuance."),
    _l("adjudicator.error_rails", "adjudicator",
       "A rail that errored or timed out is never adjudicated.",
       "const", Lock.SAFETY, "excluded",
       "Its verdict is a fail-closed default, not a score. Letting the adjudicator soften "
       "it would undo the fail-closed guarantee exactly when the stack is least healthy."),

    _l("grounding.score_direction", "grounding",
       "Higher score means better grounded.",
       "const", Lock.MODEL, "P(consistent), ascending",
       "The judge emits a probability of consistency. Inverting the comparison would invert "
       "the rail while leaving the config looking correct."),
    _l("grounding.claim_segmentation", "grounding",
       "Response is split into claims before scoring.",
       "const", Lock.MODEL, "sentence-level",
       "One claim scored against one source at a time. Whole-response scoring averages a "
       "single fabricated sentence away to nothing."),
    _l("grounding.applies_to", "grounding",
       "Scope of the grounding rail.",
       "const", Lock.ARCH, "retrieval-backed responses only",
       "There is nothing to ground against when no context was retrieved. The rail no-ops "
       "rather than inventing a baseline to score against."),
    _l("grounding.retry_prompt_suffix", "grounding",
       "Instruction appended on a regeneration.",
       "const", Lock.ARCH, "cite only retrieved context",
       "The retry has to differ from the original call in a way that addresses the failure, "
       "or you are just paying twice for the same answer."),

    # ---------------- policy ----------------
    _a("policy.security_rules", "policy",
       "Security rules, each `regex => block|mask|flag`. Omit the action and it flags.",
       "ruleset", []),
    _a("policy.privacy_rules", "policy",
       "Privacy rules, same `regex => action` form.", "ruleset", []),
    _a("policy.compliance_rules", "policy",
       "Regulatory rule set — mapped to what you're certifying against.",
       "ruleset", []),
    _a("policy.use_case_rules", "policy",
       "Rules scoped to one application rather than the whole tenant.",
       "ruleset", []),
    _a("policy.latency_budget_ms", "policy",
       "How long the rails on one surface may take before the unfinished ones fail "
       "closed. They run concurrently, so this bounds the slowest single rail, not "
       "their sum. The ceiling was 10s when one model-backed rail ran here; a "
       "surface with three needs room above their normal variance, or the tripwire "
       "trips on ordinary traffic.",
       "int", 20_000, minimum=500, maximum=120_000, step=500),
    _a("policy.disclosure", "policy",
       "How much a user is told about why a rail fired. Higher is friendlier and "
       "more useful; it is also more information for someone probing the filters.",
       "enum", "category", options=["none", "minimal", "category", "detailed"]),
    _a("policy.human_review.trigger", "policy",
       "What lands in the review queue.", "enum", "repeat failures",
       options=["repeat failures", "any block", "any mask", "sampled 5%", "none"]),
    _a("policy.fail_mode", "policy",
       "Behaviour when a rail errors out.", "enum", "fail_closed",
       options=["fail_closed", "fail_open"]),

    _l("policy.disclosure.injection_cap", "policy",
       "Ceiling on what a user is told when the injection rail fires.",
       "const", Lock.SAFETY, "never above 'category'",
       "Naming the matched technique tells an attacker exactly which phrasing to vary "
       "next. Every other family respects policy.disclosure; this one is capped, so "
       "turning disclosure up for usability cannot turn the filter into a tutorial."),
    _l("policy.verdict_precedence", "policy",
       "How competing rail verdicts resolve.",
       "const", Lock.SAFETY, "block > mask > flag > pass",
       "The most restrictive verdict always wins. Any other ordering lets one permissive "
       "rail overrule a restrictive one, which is the whole failure mode this stack exists "
       "to prevent."),
    _l("policy.timeout_behavior", "policy",
       "Behaviour when the latency budget is exceeded.",
       "const", Lock.SAFETY, "fail closed",
       "An unevaluated request is not a safe request. This holds even when fail_mode is set "
       "to open — a timeout is not the same event as a rail returning an error."),
    _l("policy.rail_isolation", "policy",
       "Rails cannot read each other's intermediate state.",
       "const", Lock.ARCH, "isolated evaluation",
       "Shared state turns independent detectors into one correlated detector. You lose "
       "exactly the redundancy you paid for."),
    _l("policy.runtime_override", "policy",
       "Whether a rail can be disabled through the API at runtime.",
       "const", Lock.SAFETY, "not permitted",
       "Rails change through a versioned config release with an author and a diff — "
       "never through a request parameter that any caller can set."),
    _l("policy.audit_immutability", "policy",
       "Audit record mutability.",
       "const", Lock.COMPLY, "append-only, hash-chained",
       "A mutable audit trail is not an audit trail."),

    # ---------------- scope ----------------
    _a("scope.threshold", "scope",
       "How clearly in-scope a question must be. Below this the configured action "
       "fires. Higher is stricter.",
       "float", 0.40, minimum=0, maximum=1, step=0.01),
    _a("scope.action", "scope",
       "What an out-of-scope question does. `flag` records it and answers anyway; "
       "`block` refuses.",
       "enum", "flag", options=["block", "flag", "pass"]),
    _a("scope.domain_terms", "scope",
       "Vocabulary that settles the common case without a model call. A question "
       "containing any of these is in scope, full stop.",
       "set", []),

    _l("scope.judge_order", "scope",
       "When the semantic check runs.",
       "const", Lock.ARCH, "only after the vocabulary pass finds nothing",
       "Asking a model about every question would triple the cost of the ordinary "
       "ones to answer a question the keyword layer already settled."),

    # ---------------- ingest ----------------
    _a("ingest.chunk_size", "ingest",
       "Target characters per indexed chunk.", "int", 700,
       minimum=200, maximum=2000, step=50),
    _a("ingest.chunk_overlap", "ingest",
       "Characters repeated between neighbouring chunks, so a fact split across a "
       "boundary is still retrievable.", "int", 80, minimum=0, maximum=400, step=10),
    _a("ingest.max_document_chars", "ingest",
       "Largest document accepted in one upload.", "int", 200_000,
       minimum=1000, maximum=2_000_000, step=1000),
    _a("ingest.allowed_types", "ingest",
       "File extensions accepted for upload. Spreadsheets are parsed; images and "
       "scanned pages are transcribed by a model.",
       "set", ["txt", "md", "csv", "json", "pdf", "xlsx", "xlsm",
               "png", "jpg", "jpeg", "webp"]),
    _a("ingest.ocr_model", "ingest",
       "Model used to transcribe a scanned page or an image. Transcription is "
       "mechanical — the ceiling is legibility, not reasoning.",
       "string", "claude-sonnet-5"),
    _a("ingest.ocr_max_pages", "ingest",
       "Pages of one scanned document that will be transcribed. Beyond this the "
       "document is indexed with a note saying what was left out.",
       "int", 20, minimum=1, maximum=200, step=1),
    _a("ingest.latency_budget_ms", "ingest",
       "Budget for the rails that scan a whole document. Separate from "
       "policy.latency_budget_ms because a document is not a prompt: the same judge "
       "reads a hundred times more text, and one budget for both quarantines "
       "perfectly good uploads.",
       "int", 60_000, minimum=1000, maximum=300_000, step=1000),
    _a("ingest.min_chunk_score", "ingest",
       "Retrieval floor. A chunk scoring below this is not returned — a weak match "
       "gives the grounding rail irrelevant context to score against.",
       "float", 0.15, minimum=0.0, maximum=1.0, step=0.01),

    _l("ingest.mask_before_index", "ingest",
       "When masking happens relative to indexing.",
       "const", Lock.SAFETY, "before the chunk is written",
       "An index that stores raw values is a second copy of the data you just "
       "protected, in a store that answers search queries."),
    _l("ingest.quarantine_on_block", "ingest",
       "What happens to a document that fails an ingest rail.",
       "const", Lock.ARCH, "quarantined, never indexed",
       "Indexing it with a flag makes retrieval safety a matter of remembering to "
       "check the flag. Quarantine is the same decision made once."),
    _l("ingest.ocr_isolation", "ingest",
       "What the transcribing model is allowed to do with the page it reads.",
       "const", Lock.SAFETY, "transcribe only, never obey",
       "Transcription is the one point where a model sees a document before the "
       "rails do. It is told the page is data being copied, not instructions — and "
       "its output is then treated as an untrusted document like any other, so an "
       "injection printed on a scan is quarantined exactly as a pasted one is."),
    _l("ingest.injection_scan", "ingest",
       "Prompt-injection scanning of ingested documents.",
       "const", Lock.SAFETY, "always on",
       "Indirect injection is the whole reason ingestion is a trust boundary. A "
       "document is attacker-supplied text that the model will later be asked to "
       "follow instructions near."),

    # ---------------- agent ----------------
    _a("agent.max_steps", "agent",
       "Tool-use rounds before the agent must answer with what it has.",
       "int", 6, minimum=1, maximum=12, step=1),
    _a("agent.tools_enabled", "agent",
       "Tools the agent may call. Removing one here removes it from the model's "
       "tool list entirely — it cannot call what it cannot see.",
       "set", ["search_documents", "lookup_fee", "check_claim_status", "file_grievance"]),
    _a("agent.tool_timeout_ms", "agent",
       "How long a single tool may run before the step fails closed.",
       "int", 5000, minimum=250, maximum=30_000, step=250),
    _a("agent.max_tool_calls", "agent",
       "Total tool calls allowed across one request, across all steps.",
       "int", 10, minimum=1, maximum=40, step=1),
    _a("agent.masked_field_disclosure", "agent",
       "How the agent reports a retrieved field that arrived masked — show "
       "the token placeholder as-is, or explain in prose that it is protected.",
       "enum", "relay", options=["relay", "explain"]),

    _l("agent.approval_required_for", "agent",
       "Which tool calls stop and ask a person.",
       "const", Lock.SAFETY, "every write tool, always",
       "A tool that changes state outside this system asks a human. An adjustable "
       "version of this is a setting somebody turns off on a busy Friday."),
    _l("agent.tool_result_trust", "agent",
       "How a tool result is treated on the way back.",
       "const", Lock.SAFETY, "untrusted — crosses agent.data rails first",
       "A tool result is attacker-reachable text. Trusting it because it came from "
       "your own tool is how indirect injection lands."),
    _l("agent.vault_unmask_scope", "agent",
       "Which tools may see a raw value behind a vault token.",
       "const", Lock.COMPLY, "declared per tool, never model-chosen",
       "The model asks for a lookup; it does not decide who is entitled to the "
       "underlying identifier. That decision belongs to the tool definition."),
    _l("agent.tool_registry", "agent",
       "Where the callable tool set comes from.",
       "const", Lock.ARCH, "code-declared, config-filtered",
       "Tools are functions with side effects. Declaring one from config would mean "
       "config could name an endpoint nobody reviewed."),
]


BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}
ADJUSTABLE: dict[str, Param] = {p.key: p for p in PARAMS if p.adjustable}
LOCKED: dict[str, Param] = {p.key: p for p in PARAMS if not p.adjustable}


def defaults() -> dict[str, Any]:
    """Every adjustable parameter at its registry default."""
    return {k: p.default for k, p in ADJUSTABLE.items()}


def control_for(p: Param) -> str:
    """Which UI control this parameter needs.

    Declared here rather than inferred in the frontend, so a new parameter type
    is one edit in one file.
    """
    if not p.adjustable:
        return "locked"
    if p.type in ("float", "int"):
        return "range" if p.minimum is not None else "number"
    if p.type == "bool":
        return "toggle"
    if p.type == "enum":
        return "select"
    if p.type in ("string[]", "regex[]", "enum[]", "set", "ruleset"):
        return "tags"
    if p.type == "matrix":
        return "matrix"
    return "text"


def as_payload() -> dict[str, Any]:
    """Registry view for the API / Parameters page.

    Everything the frontend needs to render and edit the control surface —
    including the surface list, severity levels, lock colours, and which
    control each parameter type needs. The page hardcodes none of it.
    """
    return {
        "families": [
            {
                "key": key,
                "name": meta["name"],
                "engine": meta["engine"],
                "params": [
                    {**p.to_dict(), "control": control_for(p)}
                    for p in PARAMS if p.family == key
                ],
                "adjustable": sum(1 for p in PARAMS if p.family == key and p.adjustable),
                "locked": sum(1 for p in PARAMS if p.family == key and not p.adjustable),
            }
            for key, meta in FAMILIES.items()
        ],
        "locks": LOCK_META,
        "surfaces": SURFACES,
        "severity_levels": SEVERITY_LEVELS,
        "total": len(PARAMS),
        "total_adjustable": len(ADJUSTABLE),
        "total_locked": len(LOCKED),
    }
