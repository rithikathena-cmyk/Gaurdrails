"""Sign-in, roles, and permissions.

Deliberately server-side. Hiding the trace tab from a citizen's sidebar is a
presentation choice; it is not access control, because `GET /api/traces` is one
curl away. So the role resolves to a permission set here, every route that
matters declares which permission it needs, and the frontend filters its nav
from the same answer the server enforces — one source, two consumers.

What this is **not**: an identity system. Users live in memory, there is no
password policy, no lockout, no rate limiting, and no revocation beyond the
process. It exists so the console has a principal to enforce against and the
audit log has a name to record. Put a real IdP in front of it before this
faces anything but localhost.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Cookie, HTTPException

COOKIE = "gc_session"
SESSION_TTL_S = 8 * 60 * 60
PBKDF2_ROUNDS = 240_000


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
# One permission per thing worth protecting, rather than a boolean `is_admin`.
# The difference shows the moment somebody needs a reviewer who can read traces
# but not edit thresholds.
PERMISSIONS: dict[str, str] = {
    "chat": "Ask the assistant, in chat or agent mode",
    "traces": "Read the full trace of any request",
    "documents": "Ingest documents and see what the rails found",
    "parameters": "Read and change the control surface",
    "scenarios": "Run the evaluation scenarios",
    "audit": "Read the policy and verify the audit chain",
}

ROLES: dict[str, dict[str, Any]] = {
    "user": {
        "label": "Citizen",
        "blurb": "Asks questions. Sees the answer and whether it was refused — "
                 "nothing about how the decision was made.",
        "permissions": ["chat"],
    },
    "admin": {
        "label": "Administrator",
        "blurb": "Everything: traces, the document corpus, the control surface, "
                 "the scenarios, and the audit chain.",
        "permissions": sorted(PERMISSIONS),
    },
}

# Which console views each permission unlocks. The frontend renders its nav from
# this rather than deciding for itself what an admin is.
VIEW_PERMISSION = {
    "chat": "chat",
    "trace": "traces",
    "docs": "documents",
    "params": "parameters",
}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt_hex), int(rounds))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


@dataclass
class User:
    name: str
    role: str
    password_hash: str
    display: str = ""

    @property
    def permissions(self) -> list[str]:
        return list(ROLES.get(self.role, {}).get("permissions", []))

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display": self.display or self.name,
            "role": self.role,
            "role_label": ROLES.get(self.role, {}).get("label", self.role),
            "permissions": self.permissions,
            "views": [v for v, p in VIEW_PERMISSION.items() if self.can(p)],
        }


def _default_users() -> dict[str, User]:
    """Demo accounts, or whatever GUARDRAIL_USERS declares.

    The demo passwords are printed on the sign-in page on purpose — a hidden
    default password is a password everybody shares and nobody changes.
    """
    raw = os.getenv("GUARDRAIL_USERS", "")
    if raw:
        users: dict[str, User] = {}
        for entry in json.loads(raw):
            users[entry["name"]] = User(
                name=entry["name"], role=entry.get("role", "user"),
                password_hash=entry.get("password_hash")
                or hash_password(entry["password"]),
                display=entry.get("display", ""),
            )
        return users
    return {
        "citizen": User("citizen", "user", hash_password("citizen"), "Meera B."),
        "admin": User("admin", "admin", hash_password("admin"), "Operations"),
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class Session:
    token: str
    user: str
    created_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > SESSION_TTL_S


class Directory:
    """Users and their live sessions. In memory: a restart signs everyone out."""

    def __init__(self) -> None:
        self.users = _default_users()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ---- authentication ------------------------------------------
    def authenticate(self, name: str, password: str) -> User | None:
        user = self.users.get((name or "").strip().lower())
        if user is None:
            # Spend the same time either way; a fast "no such user" is a user
            # enumeration oracle.
            hash_password(password or "")
            return None
        if not verify_password(password or "", user.password_hash):
            return None
        return user

    def open_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._prune()
            self._sessions[token] = Session(token=token, user=user.name)
        return token

    def close_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def resolve(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if session.expired:
                self._sessions.pop(token, None)
                return None
        return self.users.get(session.user)

    def _prune(self) -> None:
        for token in [t for t, s in self._sessions.items() if s.expired]:
            self._sessions.pop(token, None)

    @property
    def active(self) -> int:
        with self._lock:
            self._prune()
            return len(self._sessions)


directory = Directory()


# ---------------------------------------------------------------------------
# Route guards
# ---------------------------------------------------------------------------
def current_user(gc_session: str | None = Cookie(default=None)) -> User:
    user = directory.resolve(gc_session)
    if user is None:
        raise HTTPException(401, detail={"kind": "auth", "message": "sign in to continue"})
    return user


def require(permission: str):
    """A dependency that enforces one permission.

    The message names the permission that was missing, the way a good refusal
    does elsewhere in this stack — being told "denied" without being told what
    would have worked is how people file tickets.
    """
    def guard(gc_session: str | None = Cookie(default=None)) -> User:
        user = current_user(gc_session)
        if not user.can(permission):
            raise HTTPException(403, detail={
                "kind": "permission",
                "message": f"{user.to_dict()['role_label']} accounts do not hold "
                           f"'{permission}'. Sign in as an administrator.",
                "permission": permission,
            })
        return user

    return guard
