"""Tracing.

Built in from the first commit, not retrofitted. Every rail is timed by the
tracer rather than timing itself, so a rail cannot forget to report and cannot
report a number it made up.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .types import RailResult, StageTrace, Trace, Verdict, precedence


class Tracer:
    def __init__(self, session_id: str = "") -> None:
        self.trace = Trace(session_id=session_id)
        self._t0 = time.perf_counter()
        self._stage: StageTrace | None = None

    # ---- timing ------------------------------------------------------
    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def stage(self, name: str, subtitle: str = "", kind: str = "rail") -> Iterator[StageTrace]:
        st = StageTrace(name=name, subtitle=subtitle, kind=kind, start_ms=self._elapsed_ms())
        self._stage = st
        began = time.perf_counter()
        try:
            yield st
        finally:
            st.duration_ms = (time.perf_counter() - began) * 1000
            st.verdict = precedence([r.verdict for r in st.rails])
            self.trace.stages.append(st)
            self._stage = None
            if kind in ("rail", "retry"):
                self.trace.guardrail_ms += st.duration_ms

    @contextmanager
    def rail(self, name: str, engine: str) -> Iterator[RailResult]:
        """Time one rail and attach it to the open stage.

        A rail that raises still gets recorded — with the configured fail mode
        applied by the caller — because an invisible failure is the worst kind.
        """
        res = RailResult(rail=name, engine=engine, verdict=Verdict.PASS)
        began = time.perf_counter()
        try:
            yield res
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
            res.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            res.duration_ms = (time.perf_counter() - began) * 1000
            if self._stage is not None:
                self._stage.rails.append(res)

    def note(self, text: str) -> None:
        """Attach a note to the open stage, or to the one that just closed.

        Notes are usually written *about* a stage that has already finished —
        "grounding failed, regenerating" is decided after the grounding stage
        exits. Requiring an open stage meant those notes were dropped on the
        floor: written by the engine, never seen in a trace.
        """
        stage = self._stage or (self.trace.stages[-1] if self.trace.stages else None)
        if stage is not None:
            stage.notes.append(text)

    # ---- finish ------------------------------------------------------
    def finish(self, verdict: Verdict) -> Trace:
        self.trace.verdict = verdict
        self.trace.total_ms = self._elapsed_ms()
        return self.trace


class AuditLog:
    """Append-only, hash-chained audit record.

    `policy.audit_immutability` in the registry is locked to this. The chain is
    what makes the log an audit trail rather than a file: each entry commits to
    the previous one, so a deletion or edit anywhere breaks verification.
    """

    def __init__(self, path: str | Path = "audit.log") -> None:
        self.path = Path(path)
        self._prev = self._tail_hash()
        # Reading the previous hash, hashing, appending, and advancing the
        # pointer have to be one step. Without this, two requests overlapping
        # in the thread pool both read the same `prev` and the chain forks —
        # and a chain that breaks on its own trains an operator to ignore the
        # alarm that matters.
        self._lock = threading.Lock()

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = ""
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return "0" * 64
        try:
            return json.loads(last)["hash"]
        except (json.JSONDecodeError, KeyError):
            return "0" * 64

    def write(self, trace: Trace, detections: list[dict]) -> str:
        body = {
            "request_id": trace.request_id,
            "session_id": trace.session_id,
            "ts": trace.created_at,
            "verdict": trace.verdict.value,
            "total_ms": round(trace.total_ms, 2),
            "guardrail_ms": round(trace.guardrail_ms, 2),
            "rails": [
                {"rail": r.rail, "verdict": r.verdict.value, "score": round(r.score, 4)}
                for r in trace.rails
            ],
            # Detections are recorded pre-masking. In a real deployment this
            # file lives under a separate ACL from the response log.
            "detections": detections,
        }
        return self._commit(body)

    def write_agent_run(self, *, request_id: str, who: str, status: str,
                        agents_selected: list[str], agent_decisions: dict[str, Any],
                        policy_decision: dict[str, Any] | None, final_action: str,
                        confidence: float, escalation_reason: str, duration_ms: float,
                        trace: list[dict[str, Any]], surface: str = "") -> str:
        """One entry per Supervisor run — `POST /api/agents/supervisor/run`,
        every status (completed, escalated, failed before a result existed).

        Same hash-chained file as `write()`, deliberately: one audit trail,
        not two. Deliberately narrow on content — no request `text`, no
        agent `rationale`/`evidence_summary`, no finding `evidence` (call_ids
        are fine; free text an agent wrote is not, since nothing stops a
        judge call from echoing a raw value into its own explanation). What
        it keeps is exactly what answers "who ran what, what did each agent
        and the policy engine decide, what was the outcome, and when" —
        structured fields and finding *kinds*, not prose.
        """
        body = {
            "kind": "agent_run",
            "request_id": request_id,
            "who": who,
            "ts": time.time(),
            "surface": surface,
            "status": status,
            "agents_selected": list(agents_selected),
            "agent_decisions": agent_decisions,
            "policy_decision": policy_decision,
            "final_action": final_action,
            "confidence": round(confidence, 4),
            "escalation_reason": escalation_reason,
            "duration_ms": round(duration_ms, 2),
            "trace": trace,
        }
        return self._commit(body)

    def write_guardrail_supervisor_run(self, *, request_id: str, who: str, status: str,
                                       hard_blocked: bool, tools_run: list[str],
                                       risk_score: float | None, judge_calls: int,
                                       policy_decision: dict[str, Any] | None,
                                       final_action: str, escalation_reason: str,
                                       duration_ms: float, trace: list[dict[str, Any]],
                                       surface: str = "") -> str:
        """One entry per `GuardrailSupervisor.run()` — the flat MVP loop in
        `guardrail_supervisor.py`. Same hash-chained file as `write()` and
        `write_agent_run()`, deliberately: one audit trail. `kind`
        distinguishes it from an ordinary chat entry or a `Supervisor` run.

        Deliberately narrow, the same reasoning `write_agent_run()` already
        documents: no request text, no model-authored free text beyond a
        rationale field the model does not control the length or content of
        arbitrarily. `hard_blocked` and `judge_calls` are what let an
        auditor answer "was the judge even asked" without reading the full
        trace.
        """
        body = {
            "kind": "guardrail_supervisor_run",
            "request_id": request_id,
            "who": who,
            "ts": time.time(),
            "surface": surface,
            "status": status,
            "hard_blocked": hard_blocked,
            "tools_run": list(tools_run),
            "risk_score": round(risk_score, 4) if risk_score is not None else None,
            "judge_calls": judge_calls,
            "policy_decision": policy_decision,
            "final_action": final_action,
            "escalation_reason": escalation_reason,
            "duration_ms": round(duration_ms, 2),
            "trace": trace,
        }
        return self._commit(body)

    def _commit(self, body: dict[str, Any]) -> str:
        with self._lock:
            body = {**body, "prev": self._prev}
            digest = hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            entry = {**body, "hash": digest}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._prev = digest
            return digest

    def verify(self) -> tuple[bool, str]:
        """Walk the chain. Returns (ok, message)."""
        import hashlib

        if not self.path.exists():
            return True, "no audit log yet"
        prev = "0" * 64
        n = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                claimed = entry.pop("hash")
                if entry.get("prev") != prev:
                    return False, f"chain broken at entry {i}: prev mismatch"
                digest = hashlib.sha256(
                    json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if digest != claimed:
                    return False, f"chain broken at entry {i}: content modified"
                prev = claimed
                n += 1
        return True, f"{n} entries verified"
