"""HTTP routes, grouped by concern.

The permission on each group is declared here, once, next to the router it
guards — rather than sprinkled through handlers where a new endpoint can quietly
forget to ask.
"""

from fastapi import APIRouter, Depends

from ..auth import require
from . import agent, chat, documents, params, scenarios, session, system

api = APIRouter(prefix="/api")

# Public: the sign-in page needs health, and needs to sign you in.
api.include_router(system.router, tags=["system"])
api.include_router(session.router, tags=["auth"])

# Everything else requires a session and the right permission.
api.include_router(chat.router, tags=["chat"], dependencies=[Depends(require("chat"))])
api.include_router(agent.router, tags=["agent"], dependencies=[Depends(require("chat"))])
api.include_router(documents.router, tags=["documents"],
                   dependencies=[Depends(require("documents"))])
api.include_router(scenarios.router, tags=["scenarios"],
                   dependencies=[Depends(require("scenarios"))])
api.include_router(params.router, tags=["parameters"],
                   dependencies=[Depends(require("parameters"))])

__all__ = ["api"]
