"""Application state and engine lifecycle.

One place owns the engine, the session store, and the trace ring. Routes read
from here; they never build an engine themselves. That makes `reload()` — which
config edits depend on — a single, obvious operation.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from guardrails import (
    AgentRunner,
    AuditLog,
    Claude,
    ConfigError,
    Corpus,
    Engine,
    LLMError,
    PendingApproval,
    Policy,
    load,
)

log = logging.getLogger("guardrails.server")

MAX_TRACES = 50
MAX_HISTORY_TURNS = 12
MAX_PENDING = 32
PENDING_TTL_S = 30 * 60
CORPUS_PATH = Path(os.getenv("GUARDRAIL_CORPUS", "data/corpus.json"))


class AppState:
    def __init__(self) -> None:
        self.policy: Policy | None = None
        self.engine: Engine | None = None
        self.agent: AgentRunner | None = None
        self.error: str | None = None
        self.model_rails = False
        self.traces: deque[dict[str, Any]] = deque(maxlen=MAX_TRACES)
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.audit = AuditLog("audit.log")
        # The corpus outlives a config reload: reloading rails must not drop the
        # documents somebody ingested through them.
        self.corpus = Corpus(CORPUS_PATH)
        self.pending: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    def reload(self) -> None:
        """Rebuild the engine from config. Raises ConfigError on bad config.

        Held under a lock so an in-flight request never sees a half-swapped
        engine. Sessions and traces survive a reload; the rails do not.
        """
        with self._lock:
            path = os.getenv("GUARDRAIL_CONFIG", "config/policy.yaml")
            try:
                policy = load(path)
            except ConfigError as exc:
                self.error = str(exc)
                log.error("config rejected:\n%s", exc)
                raise

            llm = None
            if os.getenv("ANTHROPIC_API_KEY"):
                try:
                    llm = Claude(
                        model=os.getenv("GUARDRAIL_MODEL", "claude-opus-5"),
                        judge_model=str(policy.get("content.judge_model")),
                    )
                except LLMError as exc:
                    log.warning("model rails unavailable: %s", exc)

            self.policy = policy
            self.engine = Engine(policy, llm, self.audit, self.corpus)
            self.agent = AgentRunner(self.engine, llm)
            self.model_rails = llm is not None
            self.error = None
            log.info(
                "config loaded — %s%s, model rails %s",
                policy.source,
                f" (+{len(policy.overridden)} overrides)" if policy.overridden else "",
                "live" if llm else "offline",
            )

    def try_reload(self) -> None:
        """Reload, swallowing config errors into `self.error` for /api/health."""
        try:
            self.reload()
        except ConfigError:
            pass

    # -----------------------------------------------------------------
    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self.sessions.setdefault(session_id, [])

    def remember(self, session_id: str, user: str, assistant: str) -> None:
        h = self.history(session_id)
        h.append({"role": "user", "content": user})
        h.append({"role": "assistant", "content": assistant})
        del h[:-MAX_HISTORY_TURNS]

    def forget(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def record(self, trace: dict[str, Any]) -> None:
        self.traces.appendleft(trace)

    # -----------------------------------------------------------------
    def park(self, pending: PendingApproval) -> None:
        """Hold a paused write-tool call until a person answers.

        Bounded and expiring: an approval nobody answered is an approval that
        should not still be executable an hour later.
        """
        with self._lock:
            cutoff = time.time() - PENDING_TTL_S
            for token in [t for t, p in self.pending.items() if p.created_at < cutoff]:
                self.pending.pop(token, None)
            while len(self.pending) >= MAX_PENDING:
                self.pending.pop(next(iter(self.pending)))
            self.pending[pending.token] = pending

    def claim(self, token: str) -> PendingApproval | None:
        """Take a parked approval. One use only — an approval is not replayable."""
        with self._lock:
            pending = self.pending.pop(token, None)
        if pending is None:
            return None
        if time.time() - pending.created_at > PENDING_TTL_S:
            return None
        return pending


state = AppState()
