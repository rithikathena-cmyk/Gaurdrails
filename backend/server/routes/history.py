"""Transcripts: your own by default, anyone's if you hold `traces`.

The authorisation here is per request rather than per router, which is the
exception in this codebase and worth the explanation. Every other group has one
permission for the whole router — that is what keeps a new endpoint from quietly
forgetting to ask. This group cannot: the same path serves a citizen reading
their own conversations and an operator reading somebody else's, and those are
different rights over the same URL.

So the rule is written once, in `_may_read`, and every handler goes through it.
A citizen asking for another person's transcript gets 403, not a filtered list —
the check happens before anything is read, so there is no version of the request
that returns data it should not.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..auth import User, current_user, directory
from ..history import history

router = APIRouter()


def _may_read(me: User, whose: str) -> str:
    """Resolve and authorise the transcript owner, or raise.

    Returns the normalised username so handlers never re-derive it.
    """
    whose = (whose or "").strip().lower() or me.name
    if whose == me.name:
        return whose
    if not me.can("traces"):
        raise HTTPException(403, detail={
            "kind": "forbidden",
            "message": "You can only read your own conversations. Reading "
                       "somebody else's needs the traces permission.",
        })
    if whose not in directory.users:
        raise HTTPException(404, detail={"kind": "missing",
                                         "message": f"no user {whose!r}"})
    return whose


def _who(name: str) -> dict[str, Any]:
    user = directory.users.get(name)
    return {"name": name,
            "display": (user.display or name) if user else name,
            "role_label": user.to_dict()["role_label"] if user else "—"}


@router.get("/history")
def my_history(user: str = "", me: User = Depends(current_user)) -> dict[str, Any]:
    """Conversations, newest first. `user` is ignored unless you hold `traces`."""
    whose = _may_read(me, user)
    return {
        "whose": _who(whose),
        "mine": whose == me.name,
        "sessions": history.sessions(whose),
        "stats": history.stats(whose),
        # An operator gets the list of people to switch between; a citizen does
        # not, because for them there is nothing to switch to.
        "people": ([
            {**_who(n), **history.stats(n)}
            for n in sorted(directory.users)
        ] if me.can("traces") else []),
    }


@router.get("/history/{whose}/{session_id}")
def one_session(whose: str, session_id: str,
                me: User = Depends(current_user)) -> dict[str, Any]:
    """Every turn in one conversation, oldest first — the way it was had."""
    owner = _may_read(me, whose)
    turns = history.session(owner, session_id)
    if not turns:
        raise HTTPException(404, detail={
            "kind": "missing", "message": "no such conversation"})
    return {"whose": _who(owner), "mine": owner == me.name,
            "session_id": session_id, "turns": turns}
