"""Reversible masking — AES-256-GCM, scoped to an owner.

Split out of `pii.py` when its regex/checksum detection layer was removed:
this module has never been about detection, only about what happens to a
value once *some* rail — now `entities.py` alone — decides to mask it
reversibly. Everything here is detector-agnostic.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------
#: A token must outlive the request that minted it, and an agent run can sit at
#: an approval gate for `PENDING_TTL_S` (30 minutes) before the approved write
#: unmasks its arguments. An hour clears that with room to spare; anything
#: shorter turns a slow human approval into a failed unmask.
DEFAULT_VAULT_TTL_S = 60 * 60

#: Owner for vault tokens minted while a document is being ingested.
#:
#: Deliberately not `""`. The empty owner is the single-tenant bucket the CLI
#: and library callers use, so a value masked out of a document under it
#: unmasked again at egress for any caller that simply had no principal —
#: `run.py --ask "the office number"` printed a resident's phone number
#: straight back out of the corpus it had just been masked into.
#:
#: A token minted here belongs to the corpus, not to a person, and nothing that
#: signs in can claim it: `@` is not a legal username character, so no account
#: can hold this name. Corpus tokens are therefore never revealed to anybody,
#: which is the point — a document is masked so that it stays masked.
CORPUS_OWNER = "@corpus"

#: Owner for vault tokens minted from content the current caller did not
#: themselves supply: a model's own generated reply, a tool result, or the
#: arguments the agent is about to hand a tool. `Surface.USER_PROMPT` and
#: `Surface.USER_FEEDBACK` are the only surfaces where a caller's own text is
#: being scanned, so they are the only surfaces that mint under `principal`.
#: Everywhere else — retrieval, llm.response, agent.tool, agent.data — a new
#: detection belongs to no signed-in caller, for the same reason ingestion
#: does not: it was not this caller's value to begin with, so it must not
#: unmask for them just because they were the one who triggered the scan.
#: `@` keeps it unreachable by `add_user`, same as `CORPUS_OWNER`.
SYSTEM_OWNER = "@system"


@dataclass(frozen=True)
class VaultEntry:
    owner: str
    entity: str
    blob: bytes
    created_at: float


class Vault:
    """Reversible mask tokens, AES-256-GCM, scoped to an owner.

    Registry: `pii.vault_encryption` (compliance-locked) and
    `pii.token_determinism` (safety-locked — random per occurrence).

    **A token is not a capability.** Holding one proves only that you saw a
    masked reply; it says nothing about whether you are the person whose value
    was masked. So every entry records the principal it was minted for, and
    `reveal()` requires that principal back. A token that leaks into another
    user's transcript, a log, or a screenshot reveals nothing to whoever finds
    it.

    `owner` is the *authenticated* principal — `User.name`, resolved server-side
    from the session cookie. Deliberately not the chat `session_id`, which is a
    client-supplied field in the request body and therefore forgeable: scoping
    to it would let a caller name someone else's session and read their values.

    An empty owner is the single-tenant bucket used by the CLI and by library
    callers that have no principal. It is a real owner value like any other —
    `""` cannot read `alice`'s tokens and `alice` cannot read `""`'s.

    Denials are recorded rather than raised, so an unmask that was refused
    reaches the trace and therefore the audit chain instead of vanishing into a
    `None` return.
    """

    #: Denials kept in memory before the oldest is dropped. Bounded because an
    #: attacker enumerating tokens must not be able to grow this without limit.
    MAX_DENIALS = 256

    def __init__(self, key: bytes | None = None, *,
                 ttl_s: float = DEFAULT_VAULT_TTL_S) -> None:
        self._store: dict[str, VaultEntry] = {}
        self._denials: deque[dict[str, Any]] = deque(maxlen=self.MAX_DENIALS)
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._key = key or os.urandom(32)
        self._aead = None
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._aead = AESGCM(self._key)
        except ImportError:  # pragma: no cover - optional dependency
            self._aead = None

    @property
    def encrypted(self) -> bool:
        return self._aead is not None

    def _deny(self, token_id: str, owner: str, reason: str) -> None:
        """Record a refused reveal.

        The token id is truncated and the value never appears — a denial record
        that quoted what was behind the token would hand an attacker the thing
        the denial just protected.
        """
        self._denials.append({
            "token": token_id[:4] + "…" if len(token_id) > 4 else "…",
            "owner": owner or "(none)",
            "reason": reason,
            "at": time.time(),
        })

    def take_denials(self) -> list[dict[str, Any]]:
        """Drain recorded denials so the caller can put them in the trace."""
        with self._lock:
            out = list(self._denials)
            self._denials.clear()
        return out

    def store(self, entity: str, value: str, owner: str) -> str:
        """Mint a token owned by `owner`.

        `owner` is required rather than defaulted. A default would mean a call
        site that forgot to pass one silently minted a token anybody could read,
        and the registry's own rule applies here: a control that looks
        configured but does nothing is worse than a crash.
        """
        token_id = secrets.token_hex(6)
        if self._aead is not None:
            nonce = os.urandom(12)
            # `owner` is bound as AEAD associated data, so the ciphertext will
            # not decrypt under a different owner even if the entry map is
            # tampered with. The check below is the gate; this is the backstop.
            blob = nonce + self._aead.encrypt(nonce, value.encode(), owner.encode())
        else:
            blob = value.encode()
        with self._lock:
            self._store[token_id] = VaultEntry(
                owner=owner, entity=entity, blob=blob, created_at=time.time(),
            )
        return token_id

    def reveal(self, token_id: str, owner: str) -> str | None:
        """Return the value behind `token_id`, or None if `owner` may not see it.

        Every refusal is the same `None` from the caller's side — an error that
        distinguished "wrong owner" from "no such token" would confirm which
        tokens exist to whoever is guessing.
        """
        with self._lock:
            entry = self._store.get(token_id)
            if entry is None:
                self._deny(token_id, owner, "unknown")
                return None

            if self._ttl_s > 0 and (time.time() - entry.created_at) > self._ttl_s:
                # Drop it: an expired entry has no further legitimate use, and
                # keeping it around only widens the window if the key leaks.
                self._store.pop(token_id, None)
                self._deny(token_id, owner, "expired")
                return None

            # Constant-time compare. The margin is small, but a principal name
            # is guessable and there is no reason to leak its prefix in timing.
            if not secrets.compare_digest(entry.owner, owner):
                self._deny(token_id, owner, "owner_mismatch")
                return None

            blob = entry.blob

        if self._aead is None:
            return blob.decode()
        try:
            return self._aead.decrypt(blob[:12], blob[12:], owner.encode()).decode()
        except Exception:
            # InvalidTag, or anything else the AEAD raises on a mangled blob.
            # Fail closed and record it — a tamper attempt is worth an audit
            # line even though the caller only sees None.
            self._deny(token_id, owner, "tampered")
            return None
