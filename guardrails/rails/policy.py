"""Custom policy rules.

The four rule sets — security, privacy, compliance, use-case — were declared in
the registry long before anything read them. A parameter that looks configurable
and does nothing is exactly the failure this stack exists to prevent, so they
are now a real rail.

A rule is `pattern => action`:

    policy:
      security_rules:
        - 'password\\s*[:=]\\s*\\S+ => block'
        - 'api[_-]?key => flag'

`pattern` is a regex, matched case-insensitively against normalized text.
`action` is block | mask | flag. Omit ` => action` and the rule defaults to
flag — the least destructive choice, so a half-written rule cannot start
refusing traffic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Detection, RailResult, Verdict

ACTIONS = {"block", "mask", "flag"}
_SPLIT = re.compile(r"\s*=>\s*")


class RuleError(ValueError):
    pass


@dataclass
class Rule:
    source: str          # which rule set it came from
    pattern: re.Pattern[str]
    action: str
    raw: str

    @property
    def verdict(self) -> Verdict:
        return Verdict(self.action)


def parse(rule_set: str, rules: list[str]) -> list[Rule]:
    out: list[Rule] = []
    for i, raw in enumerate(rules or []):
        text = str(raw).strip()
        if not text or text.startswith("#"):
            continue
        parts = _SPLIT.split(text, maxsplit=1)
        pattern_src = parts[0].strip()
        action = parts[1].strip().lower() if len(parts) > 1 else "flag"
        if action not in ACTIONS:
            raise RuleError(
                f"policy.{rule_set}[{i}]: {action!r} is not one of {sorted(ACTIONS)}"
            )
        try:
            pattern = re.compile(pattern_src, re.IGNORECASE)
        except re.error as exc:
            raise RuleError(f"policy.{rule_set}[{i}]: not a valid regex — {exc}") from exc
        out.append(Rule(source=rule_set, pattern=pattern, action=action, raw=text))
    return out


class PolicyRail:
    """Evaluates every configured rule set in one pass."""

    name = "policy.rules"
    engine = "named rule sets · regex"

    def __init__(self, rule_sets: dict[str, list[str]]) -> None:
        self.rules: list[Rule] = []
        self.counts: dict[str, int] = {}
        for name, raw in (rule_sets or {}).items():
            parsed = parse(name, raw)
            self.rules.extend(parsed)
            if parsed:
                self.counts[name] = len(parsed)

    def __bool__(self) -> bool:
        return bool(self.rules)

    def evaluate(self, text: str, result: RailResult) -> RailResult:
        result.unit = "count"
        result.threshold = 1.0
        result.meta = {"rules_loaded": len(self.rules), "by_set": dict(self.counts)}

        if not self.rules:
            result.verdict = Verdict.PASS
            result.meta["skipped"] = "no policy rules configured"
            return result

        hits: list[tuple[Rule, re.Match[str]]] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                hits.append((rule, m))

        result.score = float(len(hits))
        result.detections = [
            Detection(kind=f"policy.{r.source}", value=m.group(0), start=m.start(),
                      end=m.end(), confidence=1.0, note=f"{r.raw} → {r.action}")
            for r, m in hits
        ]

        if not hits:
            result.verdict = Verdict.PASS
            return result

        # Precedence applies within the rail too: the strictest matching rule wins.
        result.verdict = max((r.verdict for r, _ in hits), key=lambda v: v.rank)
        result.meta["fired"] = sorted({r.source for r, _ in hits})

        if result.verdict is Verdict.MASK:
            out = list(text)
            for _, m in sorted(hits, key=lambda p: p[1].start(), reverse=True):
                out[m.start():m.end()] = list("*" * (m.end() - m.start()))
            result.text_out = "".join(out)
        return result
