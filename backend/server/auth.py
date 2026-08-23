"""Sign-in, roles, and permissions.

Deliberately server-side. Hiding the trace tab from a citizen's sidebar is a
presentation choice; it is not access control, because `GET /api/traces` is one
curl away. So the role resolves to a permission set here, every route that
matters declares which permission it needs, and the frontend filters its nav
from the same answer the server enforces — one source, two consumers.

What this is **not**: an identity system. There is no password policy, no
lockout, and no rate limiting. Accounts and sessions are two JSON files under
`data/`, which makes that directory credential material — a token in
`sessions.json` is a signed-in browser, so treat it the way you would treat the
password hashes sitting beside it. It exists so the console has a principal to
enforce against and the audit log has a name to record. Put a real IdP in front
of it before this faces anything but localhost.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import json
import os
import secrets
import threading
import time
from pathlib import Path
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
log = logging.getLogger("guardrails.server")

#: What a million tokens costs, per model, input and output separately.
#:
#: These are defaults, not gospel — confirm them against your own contract and
#: override with GUARDRAIL_PRICING, a JSON object of
#: {"model-prefix": {"in": <usd per Mtok>, "out": <usd per Mtok>}}. Cost is
#: computed and stored in micro-dollars (integers), because accumulating a
#: float per request drifts, and a budget that drifts is an argument later.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-opus":   {"in": 15.0, "out": 75.0},
    "claude-sonnet": {"in": 3.0,  "out": 15.0},
    "claude-haiku":  {"in": 1.0,  "out": 5.0},
}


def _pricing() -> dict[str, dict[str, float]]:
    raw = os.getenv("GUARDRAIL_PRICING", "")
    if not raw:
        return dict(_DEFAULT_PRICING)
    try:
        return {k: {"in": float(v["in"]), "out": float(v["out"])}
                for k, v in json.loads(raw).items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        log.warning("GUARDRAIL_PRICING unreadable — using built-in rates")
        return dict(_DEFAULT_PRICING)


PRICING = _pricing()


def rate_for(model: str) -> dict[str, float]:
    """Longest matching prefix wins, so a dated model id resolves to its family."""
    best: dict[str, float] | None = None
    best_len = -1
    for prefix, rates in PRICING.items():
        if (model or "").startswith(prefix) and len(prefix) > best_len:
            best, best_len = rates, len(prefix)
    return best or {"in": 0.0, "out": 0.0}


def cost_micros(model: str, tokens_in: int, tokens_out: int) -> int:
    """Micro-dollars for one call. Integer arithmetic end to end."""
    r = rate_for(model)
    return round((tokens_in * r["in"] + tokens_out * r["out"]))


#: Models an operator may assign to a person. "" means the deployment default,
#: so an unset account follows the server rather than pinning a version forever.
ASSIGNABLE_MODELS: list[dict[str, str]] = [
    {"key": "", "label": "Deployment default", "note": "follows the server setting"},
    {"key": "claude-sonnet-5", "label": "Sonnet 5", "note": "balanced — the deployment default"},
    {"key": "claude-haiku-4-5", "label": "Haiku 4.5", "note": "fastest and cheapest"},
]
MODEL_KEYS = {m["key"] for m in ASSIGNABLE_MODELS}

PERMISSIONS: dict[str, str] = {
    "chat": "Ask the assistant, in chat or agent mode",
    "traces": "Read the full trace of any request",
    "documents": "Ingest documents and see what the rails found",
    "parameters": "Read and change the control surface",
    "scenarios": "Run the evaluation scenarios",
    "audit": "Read the policy and verify the audit chain",
    "users": "Add people and set what they may spend",
    "agents": "Run the autonomous guardrail agents directly, outside a chat turn",
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
    "people": "users",
    "history": "chat",
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
    #: Ceilings on model tokens, per window. 0 means no ceiling — an explicit
    #: choice, not an unset field, so it reads the same in the API.
    token_limit: int = 0        # all time
    daily_limit: int = 0
    monthly_limit: int = 0

    tokens_used: int = 0        # all time
    day_tokens: int = 0
    month_tokens: int = 0

    #: Micro-dollars, integer. A float accumulated per request drifts, and a
    #: number that drifts is an argument with finance later.
    cost_micros: int = 0
    day_cost_micros: int = 0
    month_cost_micros: int = 0

    #: Which day and month the counters above belong to. Rollover is lazy —
    #: checked when the counter is read or written — because a scheduler that
    #: has to fire at midnight is a second thing that can fail.
    day_stamp: str = ""
    month_stamp: str = ""
    #: "" follows the deployment default rather than pinning a model.
    model: str = ""

    # -- windows ---------------------------------------------------
    def roll(self, today: str = "", month: str = "") -> None:
        """Zero a window's counters when its period has turned over."""
        today = today or time.strftime("%Y-%m-%d")
        month = month or today[:7]
        if self.day_stamp != today:
            self.day_stamp, self.day_tokens, self.day_cost_micros = today, 0, 0
        if self.month_stamp != month:
            self.month_stamp, self.month_tokens, self.month_cost_micros = month, 0, 0

    @property
    def unlimited(self) -> bool:
        return self.token_limit <= 0

    @property
    def tokens_left(self) -> int | None:
        """None when there is no ceiling — not 0, which would read as spent."""
        return None if self.unlimited else max(0, self.token_limit - self.tokens_used)

    def windows(self) -> list[tuple[str, int, int]]:
        """(name, used, limit) for each window that has a ceiling."""
        self.roll()
        pairs = [("total", self.tokens_used, self.token_limit),
                 ("daily", self.day_tokens, self.daily_limit),
                 ("monthly", self.month_tokens, self.monthly_limit)]
        return [w for w in pairs if w[2] > 0]

    def breached_window(self) -> tuple[str, int, int] | None:
        """The first ceiling this person has reached, or None.

        Daily first: it is the one most likely to be a mistake rather than a
        policy, and the one an operator can clear most safely.
        """
        order = {"daily": 0, "monthly": 1, "total": 2}
        hit = [w for w in self.windows() if w[1] >= w[2]]
        hit.sort(key=lambda w: order.get(w[0], 9))
        return hit[0] if hit else None

    @property
    def over_budget(self) -> bool:
        return self.breached_window() is not None

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
            "token_limit": self.token_limit,
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit,
            "tokens_used": self.tokens_used,
            "day_tokens": self.day_tokens,
            "month_tokens": self.month_tokens,
            "tokens_left": self.tokens_left,
            "cost_usd": round(self.cost_micros / 1e6, 6),
            "day_cost_usd": round(self.day_cost_micros / 1e6, 6),
            "month_cost_usd": round(self.month_cost_micros / 1e6, 6),
            "windows": [{"name": n, "used": u, "limit": l} for n, u, l in self.windows()],
            "breached": (self.breached_window() or (None,))[0],
            "over_budget": self.over_budget,
            "model": self.model,
            "model_label": next(
                (m["label"] for m in ASSIGNABLE_MODELS if m["key"] == self.model),
                self.model or "Deployment default"),
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


#: Where added accounts and spend counters live. Under `data/`, which is
#: gitignored — password hashes are deployment state, not source.
USERS_PATH = Path(os.getenv("GUARDRAIL_USERS_FILE", "data/users.json"))

#: Live sessions. Beside the password hashes, and as sensitive as they are —
#: a token in this file is a signed-in browser. Delete it to sign everyone out.
SESSIONS_PATH = Path(os.getenv("GUARDRAIL_SESSIONS_FILE", "data/sessions.json"))


class Directory:
    """Users and their live sessions, both of which outlive the process.

    Sessions used to be memory-only, and the reasoning was sound: a session
    table that never touches disk cannot be stolen from disk. What that missed
    is that the cookie is a persistent one — `max_age=SESSION_TTL_S`, eight
    hours — so the browser goes on presenting a credential the server has
    already forgotten. Every restart turned every open tab into a redirect to
    the sign-in page, mid-task, with nothing to explain it. A deploy should not
    sign out the people using the thing.

    They live in `data/` beside the password hashes, so the file is already as
    sensitive as that directory: treat it as credential material, keep it off
    shared disks, and delete it to sign everyone out at once. Expiry is still
    enforced on load and on read, so a stale file grants nothing — the eight
    hours run from sign-in, not from restart.
    """

    def __init__(self) -> None:
        self.users = _default_users()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._load()
        self._load_sessions()

    # ---- persistence ---------------------------------------------
    def _load(self) -> None:
        if not USERS_PATH.exists():
            return
        try:
            raw = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("users file unreadable — falling back to defaults")
            return
        for entry in raw.get("users", []):
            name = str(entry.get("name", "")).strip().lower()
            if not name:
                continue
            self.users[name] = User(
                name=name,
                role=entry.get("role", "user"),
                password_hash=entry.get("password_hash", ""),
                display=entry.get("display", ""),
                token_limit=int(entry.get("token_limit", 0)),
                daily_limit=int(entry.get("daily_limit", 0)),
                monthly_limit=int(entry.get("monthly_limit", 0)),
                tokens_used=int(entry.get("tokens_used", 0)),
                day_tokens=int(entry.get("day_tokens", 0)),
                month_tokens=int(entry.get("month_tokens", 0)),
                cost_micros=int(entry.get("cost_micros", 0)),
                day_cost_micros=int(entry.get("day_cost_micros", 0)),
                month_cost_micros=int(entry.get("month_cost_micros", 0)),
                day_stamp=entry.get("day_stamp", ""),
                month_stamp=entry.get("month_stamp", ""),
                model=entry.get("model", ""),
            )

    def _save(self) -> None:
        """Called with the lock held."""
        USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "users": [
            {"name": u.name, "role": u.role, "password_hash": u.password_hash,
             "display": u.display, "token_limit": u.token_limit,
             "daily_limit": u.daily_limit, "monthly_limit": u.monthly_limit,
             "tokens_used": u.tokens_used, "day_tokens": u.day_tokens,
             "month_tokens": u.month_tokens, "cost_micros": u.cost_micros,
             "day_cost_micros": u.day_cost_micros,
             "month_cost_micros": u.month_cost_micros,
             "day_stamp": u.day_stamp, "month_stamp": u.month_stamp,
             "model": u.model}
            for u in self.users.values()
        ]}
        tmp = USERS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(USERS_PATH)

    # ---- sessions ------------------------------------------------
    def _load_sessions(self) -> None:
        """Restore live sessions, dropping the ones that expired while down.

        A session belonging to a user who no longer exists is dropped too —
        `remove_user` clears them, but a file written before that ran should not
        resurrect an account's access.
        """
        if not SESSIONS_PATH.exists():
            return
        try:
            raw = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("session file unreadable — everyone signs in again")
            return
        restored = 0
        for entry in raw.get("sessions", []):
            token, user = str(entry.get("token", "")), str(entry.get("user", ""))
            if not token or user not in self.users:
                continue
            session = Session(token=token, user=user,
                              created_at=float(entry.get("created_at", 0.0)))
            if session.expired:
                continue
            self._sessions[token] = session
            restored += 1
        if restored:
            log.info("restored %d session(s)", restored)

    def _save_sessions(self) -> None:
        """Called with the lock held. Same atomic replace the user file uses."""
        try:
            SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "sessions": [
                {"token": s.token, "user": s.user, "created_at": s.created_at}
                for s in self._sessions.values() if not s.expired
            ]}
            tmp = SESSIONS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(SESSIONS_PATH)
        except OSError as exc:
            # A read-only or full disk should not fail the sign-in that
            # triggered this. The session still works; it just will not
            # survive a restart, which is where this started.
            log.warning("could not persist sessions: %s", exc)

    # ---- administration ------------------------------------------
    def add_user(self, name: str, password: str, role: str, display: str = "",
                 token_limit: int = 0, model: str = "", daily_limit: int = 0,
                 monthly_limit: int = 0) -> User:
        name = (name or "").strip().lower()
        if not name:
            raise ValueError("a username is required")
        if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError("username may contain letters, digits, - _ . only")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} — one of {sorted(ROLES)}")
        if len(password or "") < 4:
            raise ValueError("password must be at least 4 characters")
        if model not in MODEL_KEYS:
            raise ValueError(f"unknown model {model!r}")
        with self._lock:
            if name in self.users:
                raise ValueError(f"{name} already exists")
            user = User(name=name, role=role, password_hash=hash_password(password),
                        display=(display or "").strip(),
                        token_limit=max(0, int(token_limit)), model=model,
                        daily_limit=max(0, int(daily_limit)),
                        monthly_limit=max(0, int(monthly_limit)))
            self.users[name] = user
            self._save()
        return user

    def remove_user(self, name: str) -> bool:
        name = (name or "").strip().lower()
        with self._lock:
            if name not in self.users:
                return False
            self.users.pop(name)
            # Their live sessions go with them, or a deleted account keeps working
            # until its cookie expires.
            for token in [t for t, sess in self._sessions.items() if sess.user == name]:
                self._sessions.pop(token, None)
            self._save_sessions()
            self._save()
        return True

    def set_limits(self, name: str, *, total: int | None = None,
                   daily: int | None = None, monthly: int | None = None) -> User | None:
        """Set any of the three ceilings. Absent windows are left alone."""
        with self._lock:
            user = self.users.get((name or "").strip().lower())
            if user is None:
                return None
            if total is not None:
                user.token_limit = max(0, int(total))
            if daily is not None:
                user.daily_limit = max(0, int(daily))
            if monthly is not None:
                user.monthly_limit = max(0, int(monthly))
            self._save()
        return user

    def set_model(self, name: str, model: str) -> User | None:
        if model not in MODEL_KEYS:
            raise ValueError(f"unknown model {model!r}")
        with self._lock:
            user = self.users.get((name or "").strip().lower())
            if user is None:
                return None
            user.model = model
            self._save()
        return user

    def set_password(self, name: str, password: str) -> User | None:
        if len(password or "") < 4:
            raise ValueError("password must be at least 4 characters")
        with self._lock:
            user = self.users.get((name or "").strip().lower())
            if user is None:
                return None
            user.password_hash = hash_password(password)
            self._save()
        return user

    def reset_usage(self, name: str, window: str = "all") -> User | None:
        """Zero one window's counters.

        Per window on purpose: clearing a year of accumulated history to
        unblock somebody for the afternoon is not what an operator meant by
        "reset", and it destroys the only record of what was spent.
        """
        with self._lock:
            user = self.users.get((name or "").strip().lower())
            if user is None:
                return None
            user.roll()
            if window in ("all", "total"):
                user.tokens_used = user.cost_micros = 0
            if window in ("all", "daily"):
                user.day_tokens = user.day_cost_micros = 0
            if window in ("all", "monthly"):
                user.month_tokens = user.month_cost_micros = 0
            self._save()
        return user

    def spend(self, name: str, calls: list[tuple[str, int, int]]) -> None:
        """Record what a request cost, per window. Never refuses — the check
        happens before the request runs, not after it has been paid for."""
        if not calls:
            return
        with self._lock:
            user = self.users.get((name or "").strip().lower())
            if user is None:
                return
            user.roll()
            for model, tin, tout in calls:
                tokens = tin + tout
                micros = cost_micros(model or "", tin, tout)
                user.tokens_used += tokens
                user.day_tokens += tokens
                user.month_tokens += tokens
                user.cost_micros += micros
                user.day_cost_micros += micros
                user.month_cost_micros += micros
            self._save()

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
            self._save_sessions()
        return token

    def close_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)
            self._save_sessions()

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

    def sessions_for(self, name: str) -> int:
        """How many live sessions this account has open."""
        name = (name or "").strip().lower()
        with self._lock:
            self._prune()
            return sum(1 for s in self._sessions.values() if s.user == name)

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
