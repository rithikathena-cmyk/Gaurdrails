"""Loading a local model without making the first request pay for it.

Every local rail wants the same thing: build the model once, on demand, and let
the caller fall through to the judge if it is not there. `presidio_ner.py`
established that pattern. This adds the part that only shows up once the models
are actually in the request path.

**A model that is still loading is not a model that is available.** Measured on
a first request through the stack:

    prompt rails                       23781ms
      prompt_attack                    19024ms   <- 11s of it was torch loading
      content.safety                   19842ms   <- 7s of it was torch loading

against `policy.latency_budget_ms: 20000`. That is a legitimate request that
came within 160ms of being failed closed because a file was being read off
disk. At the 8000ms budget this parameter used to carry, it would simply have
been blocked.

So loading happens on a background thread and `get()` never waits for it. Until
it finishes, the rail sees `None` — which is precisely the case it already
handles correctly, because "the local model did not answer" and "the local
model is not installed" want the same behaviour: escalate to the judge. The
first request is a judge call, like it was before any of this existed, and
every request after it is fast.

`warm()` is the blocking form, for the evaluation harness and for a deployment
that would rather pay at startup — a measurement that includes a one-off model
load is measuring the disk, and an operator who wants a hot process should be
able to say so.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable


class LazyModel:
    """One model, built at most once, never on the calling thread.

    Failure is terminal and quiet: a model that could not be built will not be
    retried on every subsequent request, because a missing optional dependency
    does not become present under load, and retrying would put the import cost
    back in the request path it was moved out of.
    """

    def __init__(self, name: str, build: Callable[[], Any],
                 log: logging.Logger) -> None:
        self._name = name
        self._build = build
        self._log = log
        self._value: Any = None
        self._failed = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._value is not None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            value = self._build()
        except Exception as exc:  # noqa: BLE001 — any failure means "not available"
            with self._lock:
                self._failed = True
            self._log.warning("%s unavailable, deferring to the judge: %s", self._name, exc)
            return
        with self._lock:
            self._value = value
        self._log.info("%s ready", self._name)

    def _start(self) -> None:
        """Kick off a load if one is not already running or finished."""
        with self._lock:
            if self._value is not None or self._failed:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name=f"load-{self._name}", daemon=True,
            )
            self._thread.start()

    def get(self) -> Any:
        """The model if it is loaded, else None — never blocks.

        Returning None while the load is in flight is deliberate. The caller
        already treats "no local answer" as "ask the judge", which is the
        correct and safe reading of a model that has not finished loading.
        """
        self._start()
        return self._value

    def warm(self, timeout: float | None = None) -> Any:
        """Block until the model is loaded, or the timeout expires.

        For the evaluation harness and for deliberate startup warming. Never
        called from a request path.
        """
        self._start()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self._value

    def reset(self) -> None:
        """Drop the model and let it load again. Tests only."""
        with self._lock:
            self._value = None
            self._failed = False
            self._thread = None
