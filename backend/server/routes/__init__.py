"""HTTP routes, grouped by concern.

The permission on each group is declared here, once, next to the router it
guards — rather than sprinkled through handlers where a new endpoint can quietly
forget to ask.
"""

from fastapi import APIRouter, Depends

from ..auth import require
from . import (agent, agents, chat, documents, history, params, pipeline,
               scenarios, session, system, users)

api = APIRouter(prefix="/api")

# Public: the sign-in page needs health, and needs to sign you in.
api.include_router(system.router, tags=["system"])
api.include_router(session.router, tags=["auth"])

# Everything else requires a session and the right permission.
api.include_router(chat.router, tags=["chat"], dependencies=[Depends(require("chat"))])
api.include_router(agent.router, tags=["agent"], dependencies=[Depends(require("chat"))])
# `agents` (plural) — the autonomous Supervisor and its registered
# specialists, reachable directly. Distinct from `agent` (singular) above,
# the conversational tool-use loop `POST /api/chat` already runs; same split
# `guardrails.agent` and `guardrails.agents` already draw.
api.include_router(agents.router, tags=["agents"], dependencies=[Depends(require("agents"))])
# The real end-to-end pipeline `/summary` drives — chains GuardrailSupervisor,
# Supervisor, and Engine.converse() together. Same permission as `agents`:
# it runs the same autonomous agents directly, just composed into one call.
api.include_router(pipeline.router, tags=["pipeline"], dependencies=[Depends(require("agents"))])
# Transcripts authorise per request, not per router: the same path serves a
# citizen reading their own and an operator reading anyone's. `chat` is the
# floor; the owner check is in the handlers.
api.include_router(history.router, tags=["history"],
                   dependencies=[Depends(require("chat"))])
api.include_router(documents.router, tags=["documents"],
                   dependencies=[Depends(require("documents"))])
api.include_router(scenarios.router, tags=["scenarios"],
                   dependencies=[Depends(require("scenarios"))])
api.include_router(users.router, tags=["users"],
                   dependencies=[Depends(require("users"))])
api.include_router(params.router, tags=["parameters"],
                   dependencies=[Depends(require("parameters"))])

__all__ = ["api"]
