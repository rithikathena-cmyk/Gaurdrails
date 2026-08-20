"""Word guardrails — Aho–Corasick over the full lexicon in one pass.

Pure Python, no build dependency. The automaton is what makes
`words.custom_terms` genuinely unbounded: match cost is O(n + matches) in the
length of the *input*, independent of how many patterns are loaded. That
property is why the registry locks `words.match_engine` — the unbounded term
list is a consequence of the algorithm, not a separate promise.
"""

from __future__ import annotations

from collections import deque

from ..types import Detection, RailResult, Verdict, action_verdict
from .normalize import normalize

MAX_PATTERN_LENGTH = 256  # registry: words.max_pattern_length (locked, arch)

_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class Automaton:
    """Aho–Corasick. Build once at startup, match many."""

    __slots__ = ("goto", "fail", "out", "_built")

    def __init__(self) -> None:
        self.goto: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.out: list[list[str]] = [[]]
        self._built = False

    def add(self, pattern: str) -> None:
        if not pattern or len(pattern) > MAX_PATTERN_LENGTH:
            return
        node = 0
        for ch in pattern:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
                self.goto[node][ch] = nxt
            node = nxt
        self.out[node].append(pattern)
        self._built = False

    def build(self) -> None:
        q: deque[int] = deque()
        for nxt in self.goto[0].values():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            node = q.popleft()
            for ch, nxt in self.goto[node].items():
                q.append(nxt)
                f = self.fail[node]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.goto[f].get(ch, 0) if f or ch in self.goto[0] else 0
                self.out[nxt].extend(self.out[self.fail[nxt]])
        self._built = True

    def search(self, text: str) -> list[tuple[int, int, str]]:
        """Yield (start, end, pattern) for every occurrence."""
        if not self._built:
            self.build()
        hits: list[tuple[int, int, str]] = []
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pat in self.out[node]:
                hits.append((i - len(pat) + 1, i + 1, pat))
        return hits


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return before not in _WORD_CHARS and after not in _WORD_CHARS


class WordRail:
    """Lexical matching with a locked blocklist → allowlist precedence."""

    name = "words.lexicon"
    engine = "aho–corasick · pure python"

    def __init__(self, blocklist: list[str], allowlist: list[str], *, case_sensitive: bool = False,
                 match_mode: str = "word") -> None:
        self.case_sensitive = case_sensitive
        self.match_mode = match_mode
        self.block = Automaton()
        self.allow = Automaton()
        self._n_block = 0
        for term in blocklist:
            t = term if case_sensitive else term.lower()
            self.block.add(t)
            self._n_block += 1
        for term in allowlist:
            self.allow.add(term if case_sensitive else term.lower())
        self.block.build()
        self.allow.build()

    def _filter(self, text: str, hits: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        if self.match_mode == "substring":
            return hits
        if self.match_mode == "word":
            return [h for h in hits if _is_word_boundary(text, h[0], h[1])]
        # exact: the whole input is the term
        return [h for h in hits if h[0] == 0 and h[1] == len(text)]

    def evaluate(self, text: str, action: str, result: RailResult) -> RailResult:
        norm, _ = normalize(text)
        haystack = norm if self.case_sensitive else norm.lower()

        hits = self._filter(haystack, self.block.search(haystack))

        # Locked precedence: blocklist runs first, allowlist only exempts what
        # the blocklist already caught. Reversing this lets one allowlist entry
        # silently disable a blocked term.
        allowed = self._filter(haystack, self.allow.search(haystack))
        allow_spans = [(a, b) for a, b, _ in allowed]
        kept = [
            h for h in hits
            if not any(a <= h[0] and h[1] <= b for a, b in allow_spans)
        ]

        result.unit = "count"
        result.score = float(len(kept))
        result.threshold = 1.0
        result.meta = {
            "patterns_loaded": self._n_block,
            "raw_hits": len(hits),
            "exempted_by_allowlist": len(hits) - len(kept),
            "match_mode": self.match_mode,
        }
        result.detections = [
            Detection(kind="blocked_term", value=pat, start=s, end=e, confidence=1.0)
            for s, e, pat in kept
        ]

        if not kept:
            result.verdict = Verdict.PASS
            return result

        result.verdict = action_verdict(action, Verdict.MASK)
        if result.verdict is Verdict.MASK:
            out = list(text)
            # Mask against the normalized offsets; lengths match because the
            # fold is character-for-character except for removals, which only
            # shorten. Clamp defensively.
            for s, e, _ in sorted(kept, reverse=True):
                s, e = min(s, len(out)), min(e, len(out))
                out[s:e] = list("*" * (e - s))
            result.text_out = "".join(out)
        return result
