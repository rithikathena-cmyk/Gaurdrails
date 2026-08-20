"""Backend: the guardrail library and the HTTP layer that serves it.

`frontend/` is the other half. The split is by side of the wire, not by
feature — `guardrails/` holds no HTTP and `server/` holds no guardrail logic,
and this package is only the roof over both.
"""
