"""Sign in, sign out, and who am I.

The cookie is HttpOnly so page scripts cannot read it — the console asks
`/api/auth/me` for its identity instead of parsing a token, which means an XSS
in the chat transcript cannot walk away with a session.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, Field

from ..auth import COOKIE, PERMISSIONS, ROLES, SESSION_TTL_S, current_user, directory

log = logging.getLogger("guardrails.server")
router = APIRouter()


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.get("/auth/roles")
def roles() -> dict[str, Any]:
    """What the sign-in page offers, described by the server that enforces it."""
    return {
        "roles": [
            {"key": key, **{k: v for k, v in meta.items()}}
            for key, meta in ROLES.items()
        ],
        "permissions": PERMISSIONS,
        "demo_accounts": [
            {"username": u.name, "display": u.display, "role": u.role,
             "password": u.name}
            for u in directory.users.values()
            if u.name in ("citizen", "admin")
        ],
    }


@router.post("/auth/login")
def login(body: Credentials, response: Response) -> dict[str, Any]:
    user = directory.authenticate(body.username, body.password)
    if user is None:
        # One message for both failures: which half was wrong is the attacker's
        # problem to work out, not ours to tell them.
        raise HTTPException(401, detail={
            "kind": "auth", "message": "That username and password do not match.",
        })
    token = directory.open_session(user)
    response.set_cookie(
        COOKIE, token, max_age=SESSION_TTL_S, httponly=True, samesite="lax", path="/",
    )
    log.info("signed in: %s (%s)", user.name, user.role)
    return {"ok": True, "user": user.to_dict()}


@router.post("/auth/logout")
def logout(response: Response, gc_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    directory.close_session(gc_session)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(gc_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Identity plus the permission set. The console renders its nav from this."""
    return {"user": current_user(gc_session).to_dict()}
