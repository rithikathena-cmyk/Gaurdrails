"""Conversation transcripts, kept per person.

Distinct from `state.sessions`, which is the short context window handed to the
model and is deliberately trimmed and thrown away. This is the durable record:
what somebody asked, what the desk answered, what the rails decided, and the
request id that ties each turn back to its full trace.

Two rules shape the whole file:

    a person sees their own conversations, and only their own. That is not a
    UI convenience — the routes resolve the owner from the session cookie, so
    asking for somebody else's transcript is a 403 rather than a filter that
    a crafted request could skip.

    a blocked turn is still recorded. The refusal *is* the interesting part
    when an operator is asking why somebody could not get an answer, and a
    history that quietly omits refusals is a history that misleads.

What is stored is the masked text — the same string the model was given. The
vault holds the real values and this store never sees them, so a transcript
read cannot become a way to recover what masking removed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("guardrails.server")

HISTORY_PATH = Path(os.getenv("GUARDRAIL_HISTORY_FILE", "data/history.json"))

#: Turns kept per person. Old ones fall off the end rather than growing a file
#: nobody prunes; an operator who needs more has the audit log.
MAX_TURNS_PER_USER = 400


class HistoryStore:
    """Append-only within a session, capped per person, persisted to disk."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else HISTORY_PATH
        self._turns: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._load()

    # ---- persistence -------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("history file unreadable — starting empty")
            return
        self._turns = {str(k): list(v) for k, v in (raw.get("turns") or {}).items()}

    def _save(self) -> None:
        """Called with the lock held."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "turns": self._turns}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---- writing -----------------------------------------------------
    def append(self, user: str, *, session_id: str, question: str, reply: str,
               verdict: str, request_id: str, mode: str = "chat",
               blocked: bool = False, refusal_reason: str = "",
               masked: int = 0, tokens: int = 0, cost_usd: float = 0.0,
               model: str = "") -> None:
        user = (user or "").strip().lower()
        if not user:
            return
        turn = {
            "at": time.time(),
            "session_id": session_id or "default",
            "mode": mode,
            "question": question,
            "reply": reply,
            "verdict": verdict,
            "blocked": bool(blocked),
            "refusal_reason": refusal_reason,
            "request_id": request_id,
            "masked": int(masked),
            "tokens": int(tokens),
            "cost_usd": round(float(cost_usd), 6),
            "model": model,
        }
        with self._lock:
            turns = self._turns.setdefault(user, [])
            turns.append(turn)
            del turns[:-MAX_TURNS_PER_USER]
            self._save()

    def forget_user(self, user: str) -> None:
        """Called when an account is removed — their transcripts go too."""
        with self._lock:
            if self._turns.pop((user or "").strip().lower(), None) is not None:
                self._save()

    # ---- reading -----------------------------------------------------
    def turns(self, user: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._turns.get((user or "").strip().lower(), []))

    def sessions(self, user: str) -> list[dict[str, Any]]:
        """One entry per conversation, newest first, summarised.

        The summary is what a list needs — when, how many turns, whether any
        were refused — so the list itself never has to carry every transcript.
        """
        grouped: dict[str, dict[str, Any]] = {}
        for t in self.turns(user):
            g = grouped.setdefault(t["session_id"], {
                "session_id": t["session_id"], "turns": 0, "blocked": 0,
                "started_at": t["at"], "last_at": t["at"], "opened_with": t["question"],
                "tokens": 0, "cost_usd": 0.0, "modes": set(),
            })
            g["turns"] += 1
            g["blocked"] += 1 if t["blocked"] else 0
            g["started_at"] = min(g["started_at"], t["at"])
            g["last_at"] = max(g["last_at"], t["at"])
            g["tokens"] += t.get("tokens", 0)
            g["cost_usd"] += t.get("cost_usd", 0.0)
            g["modes"].add(t.get("mode", "chat"))
        out = []
        for g in grouped.values():
            g["modes"] = sorted(g["modes"])
            g["cost_usd"] = round(g["cost_usd"], 6)
            out.append(g)
        out.sort(key=lambda g: g["last_at"], reverse=True)
        return out

    def session(self, user: str, session_id: str) -> list[dict[str, Any]]:
        return [t for t in self.turns(user) if t["session_id"] == session_id]

    def stats(self, user: str) -> dict[str, Any]:
        turns = self.turns(user)
        return {
            "turns": len(turns),
            "sessions": len({t["session_id"] for t in turns}),
            "blocked": sum(1 for t in turns if t["blocked"]),
            "masked": sum(t.get("masked", 0) for t in turns),
            "tokens": sum(t.get("tokens", 0) for t in turns),
            "cost_usd": round(sum(t.get("cost_usd", 0.0) for t in turns), 6),
            "last_at": max((t["at"] for t in turns), default=None),
        }

    @property
    def users(self) -> list[str]:
        with self._lock:
            return sorted(self._turns)


history = HistoryStore()
