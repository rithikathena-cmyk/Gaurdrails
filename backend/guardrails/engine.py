"""The engine.

One `evaluate()` per surface, and one `converse()` that walks a request through
the whole stack. Everything the registry locks is enforced here, in one place:

  verdict precedence   most restrictive wins, always
  timeout behaviour    fail closed, even when fail_mode is open
  rail isolation       rails receive text and config, never each other's state
  detect → audit → mask ordering
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import knowledge
from .config import Policy
from .explain import Violation, explain, summarise
from .knowledge import Corpus, Document, chunk_text, new_document_id
from .knowledge import retrieve
from .llm import Claude, LLMError, Refusal
from .rails.adjudicator import Adjudicator
from .rails import toxicity_check
from .rails.content import CATEGORIES, ContentRail, PromptAttackRail
from .rails.entities import EntityRail
from .rails.scope import ScopeRail, requires_retrieval
from .rails.grounding import GroundingRail
from .rails.normalize import normalize
from .rails.pii import CORPUS_OWNER, SYSTEM_OWNER, PIIRail, Vault
from .rails.policy import PolicyRail
from .rails.words import WordRail
from .tracing import AuditLog, Tracer
from .types import (
    Detection,
    EvaluationResult,
    RailResult,
    Surface,
    Trace,
    Verdict,
    precedence,
)

log = logging.getLogger("guardrails.engine")

SYSTEM_PROMPT = """You are a document-grounded assistant. Your knowledge of anything \
specific comes from what has been ingested into your knowledge base, not from general \
training knowledge — you have no fixed subject area of your own; it is whatever has \
been uploaded.

Answer only from the CONTEXT provided in the user turn. If the context does not \
contain what is needed, say so plainly and name what you would need — do not fill \
the gap from general knowledge, and never invent a fee, deadline, form number, or \
eligibility rule.

Some values in the user's message may appear as masked tokens like \
<US_SSN:a1b2c3>. That is expected — the guardrail layer removed them before the \
message reached you. Do not ask the user to repeat a masked value; work with what \
you have or ask for a different identifier.

Be direct and concrete. Lead with the answer.

Format so it can be read at a glance:
- Open with the answer itself, in bold, on one line. Not a preamble.
- Bold every specific value that carries weight: an amount, a deadline, a form number, a document name, a reference.
- Use a markdown pipe table whenever you are giving three or more items that share attributes: documents and what they are for, fees by category, offices and their portals, steps and their timelines. Two columns is usually right; never more than four.
- Short bullets for anything sequential or list-shaped. One level, no nesting.
- Never a heading in a short answer. Use `###` only when the reply genuinely has two or more sections.
- Say what is missing in a final line, plainly, rather than padding the answer."""


@dataclass
class ConversationResult:
    reply: str
    trace: Trace
    blocked: bool = False
    refusal_reason: str = ""
    chunks: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    human_review: bool = False
    review_reason: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IngestResult:
    """What one document ingestion produced."""

    document: Document
    trace: Trace
    detections: list[dict[str, Any]] = field(default_factory=list)
    quarantined: bool = False
    reason: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)


# Explicit surface → action-key map. This used to be built with an f-string
# (`f"pii.action.{surface.split('.')[-1]}"`), which produced `pii.action.response`
# for llm.response — a key that does not exist. It silently fell back to the
# inbound action, so `pii.action.llm_response` could never take effect. A literal
# map cannot miss, and a new surface fails loudly instead of falling back.
PII_ACTION_KEY = {
    Surface.USER_PROMPT: "pii.action.user_prompt",
    Surface.USER_FEEDBACK: "pii.action.user_prompt",
    Surface.INGEST: "pii.action.ingest",
    Surface.RETRIEVAL: "pii.action.retrieval",
    Surface.LLM_RESPONSE: "pii.action.llm_response",
    Surface.LLM_ASK_USER: "pii.action.llm_response",
    Surface.AGENT_TOOL: "pii.action.agent_tool",
    Surface.AGENT_DATA: "pii.action.agent_data",
}

# Surfaces where prompt-injection scanning runs whatever the severity matrix
# says. `ingest.injection_scan` and `agent.tool_result_trust` are both locked
# on, and a matrix cell must not be able to switch off a lock.
INJECTION_ALWAYS = (Surface.INGEST, Surface.AGENT_DATA)
INJECTION_BY_MATRIX = (Surface.USER_PROMPT, Surface.USER_FEEDBACK)

# Scope is a question about what the *user* asked. Applying it to a retrieved
# chunk or a tool result would ask "is this document on topic", which is a
# different question and not this rail's job.
SCOPE_SURFACES = (Surface.USER_PROMPT, Surface.USER_FEEDBACK)

REFUSAL_FALLBACK = (
    "That request was stopped before it reached the model. If this was unintended, "
    "rephrase it and try again — reference {rid}."
)

REVIEW_TEMPLATE = (
    "I could not produce an answer I can stand behind from the sources available, so "
    "this has gone to a person to review. Reference {rid}."
)

# `scope.domain` no longer decides whether a question's topic is covered here
# (see rails/scope.py) — retrieval does. A question `requires_retrieval()`
# that came back with nothing is refused here, before paying for a model call
# that would otherwise be asked to answer from "(nothing retrieved)" and
# relying on the grounding rail to catch it after the fact.
NOT_FOUND_TEMPLATE = (
    "I don't have anything in the knowledge base that answers this. Try rephrasing, "
    "or ask what this service can help with — reference {rid}."
)


class Engine:
    def __init__(self, policy: Policy, llm: Claude | None = None,
                 audit: AuditLog | None = None, corpus: Corpus | None = None) -> None:
        self.policy = policy
        self.llm = llm
        self.audit = audit or AuditLog()
        self.vault = Vault()
        # One store backs both ingestion and retrieval. Binding it here means a
        # document cannot be ingested into one corpus and searched in another.
        self.corpus = corpus if corpus is not None else Corpus(seed=True)
        knowledge.use(self.corpus)
        self._build_rails()
        self.reseed_builtin_rails()

    # -----------------------------------------------------------------
    def reseed_builtin_rails(self) -> None:
        """Replace every built-in document that has never actually been
        rail-checked with the output of running it through `ingest()` for
        real — the exact path an uploaded document already takes.

        Public, and called from two places: here, at construction, and again
        by `POST /api/documents/reset` after `Corpus.reset()` — `reset()` is
        a plain `Corpus` method with no `Engine` in reach, so it puts the
        built-ins straight back the same unrailed way `seed_builtin()`
        always has. A reset that silently undid this fix would be worse than
        never having it.

        `Corpus(seed=True)` loads its built-ins straight into the store,
        synchronously, with no `Engine` involved, because plenty of callers
        (tests, `--ask`, anything that just wants a searchable corpus) want
        exactly that and have no rails to run anyway. An `Engine` has rails,
        though, and this is the one place worth paying their cost for: a
        built-in that was never actually scanned had `verdict="pass"` as a
        hardcoded literal, not a finding, and its stored text was the raw
        seed string, PII and all — `agent.data`/`retrieval` catch that on the
        way *out* today, but a direct read of the document (the `documents`
        permission, not `chat`) had no such rail in front of it at all.

        Runs once per document, ever: `rails_applied` is persisted, so a
        second boot against the same `data/corpus.json` finds nothing left
        to do.

        Gated on the corpus actually being disk-backed (`self.corpus.path`
        is set) — an ephemeral, in-memory `Corpus(seed=True)` with no path,
        which is what the overwhelming majority of tests construct, gets
        nothing durable out of paying this cost, so it does not pay it. The
        real deployment (`server/state.py` always passes a real path) is the
        one place this matters, and the one place it runs.

        Forces `entity_rail` to judge-only for the duration, restored after.
        Presidio's NER "loads on first use, not at import" (`presidio_ner.py`)
        — for an ordinary request that is a background cost paid once, well
        after startup. Here it would be the *first* thing that ever triggers
        it, synchronously, inside `Engine.__init__`, on a real PDF-sized seed
        document — on a 512MB deployment that is exactly the combination
        (a cold spaCy load plus a full NER pass over ~150,000 characters,
        blocking the process before it can even answer a health check) that
        pushed one real deploy over the limit before this guard existed. The
        judge alone still masks everything Presidio would have; it costs
        more model calls, not less protection.
        """
        if self.corpus.path is None:
            return

        from .knowledge.seed import CORPUS as SEED_CORPUS

        by_id = {f"seed:{d['id']}": d for d in SEED_CORPUS}
        pending = [doc for doc in self.corpus.all()
                  if doc.id in by_id and not doc.rails_applied]
        if not pending:
            return

        original_engine_mode = self.entity_rail.engine_mode if self.entity_rail else None
        if self.entity_rail:
            self.entity_rail.engine_mode = "judge"
        try:
            for doc in pending:
                seed = by_id[doc.id]
                self.ingest(seed["title"], seed["text"], source="built-in", kind="txt",
                           method="seed", doc_id=doc.id)
        finally:
            if self.entity_rail:
                self.entity_rail.engine_mode = original_engine_mode

    # -----------------------------------------------------------------
    def _build_rails(self) -> None:
        p = self.policy
        self.word_rail = WordRail(
            blocklist=p.lexicons.get("blocklist", []),
            allowlist=p.lexicons.get("allowlist", []),
            case_sensitive=bool(p.get("words.case_sensitive")),
            match_mode=str(p.get("words.match_mode")),
        )
        self.pii_rail = PIIRail(
            entities=list(p.get("pii.entities") or []),
            confidence_threshold=float(p.get("pii.confidence_threshold")),
            mask_strategy=str(p.get("pii.mask_strategy")),
            partial_reveal=int(p.get("pii.partial_reveal")),
            partial_reveal_prefix=int(p.get("pii.partial_reveal_prefix")),
            custom_regex=list(p.get("pii.custom_regex") or []),
            vault=self.vault,
            allowlist=list(p.get("pii.allowlist") or []),
        )
        self.policy_rail = PolicyRail({
            "security_rules": list(p.get("policy.security_rules") or []),
            "privacy_rules": list(p.get("policy.privacy_rules") or []),
            "compliance_rules": list(p.get("policy.compliance_rules") or []),
            "use_case_rules": list(p.get("policy.use_case_rules") or []),
        })
        # The injection rail's pattern layer is deterministic and must run with
        # or without an API key — it is the cheapest rail in the stack and the
        # one that catches the most common attack outright. Only its judge
        # layer needs a model.
        self.attack_rail = PromptAttackRail(
            self.llm,
            threshold=float(p.get("prompt_attack.threshold")),
            use_judge=self.llm is not None,
            engine_mode=str(p.get("prompt_attack.engine")),
            local_block_threshold=float(p.get("prompt_attack.local_block_threshold")),
        )
        # Scope is two-layer like the injection rail: the vocabulary pass is
        # deterministic and runs with or without a key; only the semantic
        # fallback needs a model.
        self.scope_rail = ScopeRail(
            self.llm,
            threshold=float(p.get("scope.threshold")),
            terms=list(p.get("scope.domain_terms") or []),
            use_judge=self.llm is not None,
            hard_block_threshold=float(p.get("scope.hard_block_threshold")),
        )
        # Names and addresses have no shape for a regex to match, so this one
        # is model-only. It shares the vault with pii.detect.
        self.entity_rail = EntityRail(
            self.llm, self.vault,
            confidence_threshold=float(p.get("pii.entity_confidence")),
            mask_strategy=str(p.get("pii.mask_strategy")),
            kinds=list(p.get("pii.entity_kinds") or []),
            engine_mode=str(p.get("pii.entity_engine")),
            allowlist=list(p.get("pii.allowlist") or []),
            partial_reveal=int(p.get("pii.partial_reveal")),
            partial_reveal_prefix=int(p.get("pii.partial_reveal_prefix")),
        )
        # The adjudicator is not a rail: it reviews what the rails decided,
        # and only when one of them landed within a margin of its threshold.
        self.adjudicator = Adjudicator(
            self.llm,
            margin=float(p.get("adjudicator.margin")),
            rails=list(p.get("adjudicator.rails") or []),
            min_confidence=float(p.get("adjudicator.min_confidence")),
            enabled=bool(p.get("adjudicator.enabled")) and self.llm is not None,
        )
        # Grounding genuinely cannot run without a model — there is no
        # deterministic way to check whether a claim is supported.
        self.grounding_rail = (
            GroundingRail(
                self.llm,
                consistency_threshold=float(p.get("grounding.consistency.threshold")),
                relevance_threshold=float(p.get("grounding.relevance.threshold")),
                context_window=int(p.get("grounding.context_window")),
                require_citations=bool(p.get("grounding.require_citations")),
                engine_mode=str(p.get("grounding.engine")),
            )
            if (self.llm is not None
                or str(p.get("grounding.engine")) in ("local", "local+judge"))
            else None
        )

    def _content_rail(self, surface: str) -> ContentRail | None:
        p = self.policy
        mode = str(p.get("content.engine"))
        if mode == "off":
            return None
        # The rail used to exist only when a key did. It now runs whenever
        # *either* layer can: with a local classifier and no key the four
        # categories it covers are still checked, which is the difference
        # between an unkeyed deployment being unguarded and being partly
        # guarded. What it cannot cover is reported as unevaluated, not passed.
        has_judge = self.llm is not None and mode in ("judge", "local+judge")
        has_local = mode in ("local", "local+judge") and toxicity_check.available()
        if not (has_judge or has_local):
            return None
        thresholds = {
            c: p.threshold(f"content.{c}.threshold", "content", surface) for c in CATEGORIES
        }
        return ContentRail(
            self.llm, thresholds, list(p.get("content.enabled_categories") or CATEGORIES),
            engine_mode=mode,
            local_block_threshold=float(p.get("content.local_block_threshold")),
        )

    # -----------------------------------------------------------------
    def _run(self, tracer: Tracer, name: str, engine_label: str, fn) -> RailResult:
        """Run one rail, applying the configured fail mode if it raises.

        A rail that errors must never silently pass. `fail_closed` is the
        default for exactly this case.
        """
        fail_open = str(self.policy.get("policy.fail_mode")) == "fail_open"
        try:
            with tracer.rail(name, engine_label) as res:
                fn(res)
            return res
        # Deliberately broad. `policy.fail_mode` used to cover only LLM errors,
        # so a rail raising anything else — a bad regex, a None where a string
        # was expected — escaped the fail-mode contract and took down the whole
        # request instead of failing closed.
        except Exception as exc:  # noqa: BLE001
            res = RailResult(
                rail=name, engine=engine_label,
                verdict=Verdict.PASS if fail_open else Verdict.BLOCK,
                error=str(exc),
                meta={"fail_mode": "open" if fail_open else "closed"},
            )
            if tracer._stage is not None:  # noqa: SLF001 — same package
                tracer._stage.rails.append(res)  # noqa: SLF001
            log.warning("rail %s failed (%s mode): %s", name, res.meta["fail_mode"], exc)
            return res

    # -----------------------------------------------------------------
    def _pii_spans(self, text: str):
        """Where the deterministic rail would find identifiers in this text.

        Recomputed rather than shared, because the rails run concurrently and
        the entity rail cannot wait on a sibling's result. A second regex pass
        costs a tenth of a millisecond; a dependency between two concurrent
        jobs costs a deadlock the first time somebody reorders them.
        """
        try:
            return [d for d, _ in self.pii_rail._detect(text)]  # noqa: SLF001
        except Exception:  # noqa: BLE001 — an overlap hint is never worth a failure
            return []

    def evaluate(self, text: str, surface: Surface, tracer: Tracer,
                 stage_name: str, subtitle: str = "",
                 owner: str = "") -> EvaluationResult:
        """Run every rail configured for one surface, concurrently.

        `owner` is the authenticated principal any vault token minted on this
        surface belongs to. It is threaded in rather than read from the tracer
        because masking authorization must not depend on an observability
        object — a trace can be swapped or omitted; the owner cannot.
        """
        p = self.policy
        s = surface.value
        began = time.perf_counter()

        with tracer.stage(stage_name, subtitle or f"{s} · concurrent"):
            jobs: list[tuple[str, str, Any]] = []

            if p.enabled("words", s):
                action = str(p.get("words.action"))
                jobs.append((
                    self.word_rail.name, self.word_rail.engine,
                    lambda r, t, a=action: self.word_rail.evaluate(t, a, r),
                ))

            if p.enabled("pii", s):
                action = str(p.get(PII_ACTION_KEY[surface]))
                jobs.append((
                    self.pii_rail.name, self.pii_rail.engine,
                    lambda r, t, a=action: self.pii_rail.evaluate(t, a, r, owner),
                ))

            if p.enabled("scope", s) and self.scope_rail and surface in SCOPE_SURFACES:
                action = str(p.get("scope.action"))
                jobs.append((
                    self.scope_rail.name, self.scope_rail.engine,
                    lambda r, t, a=action: self.scope_rail.evaluate(t, a, r),
                ))

            if p.enabled("pii", s) and self.entity_rail:
                action = str(p.get(PII_ACTION_KEY[surface]))
                jobs.append((
                    self.entity_rail.name, self.entity_rail.engine,
                    lambda r, t, a=action: self.entity_rail.evaluate(
                        t, a, r, prior=self._pii_spans(t), owner=owner),
                ))

            if p.enabled("policy", s) and self.policy_rail:
                jobs.append((
                    self.policy_rail.name, self.policy_rail.engine,
                    lambda r, t: self.policy_rail.evaluate(t, r),
                ))

            if self.attack_rail and (
                surface in INJECTION_ALWAYS
                or (surface in INJECTION_BY_MATRIX and p.enabled("content", s))
            ):
                action = str(p.get("prompt_attack.action"))
                jobs.append((
                    self.attack_rail.name, self.attack_rail.engine,
                    lambda r, t, a=action: self.attack_rail.evaluate(t, a, r),
                ))

            content_rail = self._content_rail(s)
            if p.enabled("content", s) and content_rail:
                key = ("content.action.user_prompt" if surface in
                       (Surface.USER_PROMPT, Surface.USER_FEEDBACK)
                       else "content.action.llm_response")
                action = str(p.get(key))
                action = "block" if action == "regenerate" else action
                jobs.append((
                    content_rail.name, content_rail.engine,
                    lambda r, t, a=action: content_rail.evaluate(t, a, r),
                ))

            # A whole document gets its own budget. The prompt budget is sized
            # for a few hundred characters; applying it to a 30,000-character
            # upload means the content judge times out and fails closed, and a
            # legitimate document is quarantined for being long.
            budget_key = ("ingest.latency_budget_ms" if surface is Surface.INGEST
                          else "policy.latency_budget_ms")
            budget = float(p.get(budget_key)) / 1000.0
            results: list[RailResult] = []

            if jobs:
                with futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    pending = {
                        pool.submit(self._run, tracer, name, eng,
                                    lambda r, f=fn: f(r, text)): name
                        for name, eng, fn in jobs
                    }
                    try:
                        for fut in futures.as_completed(pending, timeout=budget):
                            results.append(fut.result())
                    except futures.TimeoutError:
                        # Locked: policy.timeout_behavior — fail closed regardless
                        # of fail_mode. An unevaluated request is not a safe one.
                        unfinished = [n for f, n in pending.items() if not f.done()]
                        for name in unfinished:
                            res = RailResult(
                                rail=name, engine="—", verdict=Verdict.BLOCK,
                                error=f"latency budget exceeded ({budget * 1000:.0f}ms)",
                                meta={"timeout": True},
                            )
                            results.append(res)
                            if tracer._stage is not None:  # noqa: SLF001
                                tracer._stage.rails.append(res)  # noqa: SLF001
                        tracer.note(
                            f"latency budget exceeded — {len(unfinished)} rail(s) unevaluated, "
                            "failed closed"
                        )
                        tracer.trace.fail_mode_triggered = True

        verdict = precedence([r.verdict for r in results])

        # detect → audit → mask (locked ordering), and masking *composes*.
        #
        # Each rail computes its replacement from the text it was given, so two
        # masking rails on one request used to produce two rewrites of the
        # original and the last one silently won — a blocked word next to an SSN
        # came out unmasked. A later rail is now re-run against the text as it
        # stands. Only deterministic rails rewrite text, so the second pass
        # costs no model call.
        order = {name: i for i, (name, _, _) in enumerate(jobs)}
        fns = {name: fn for name, _, fn in jobs}
        out = text
        for r in sorted((x for x in results if x.text_out is not None),
                        key=lambda x: order.get(x.rail, 99)):
            out = self._remask(out, r, fns.get(r.rail)) if out != text else r.text_out

        return EvaluationResult(
            verdict=verdict, text=out, results=results,
            duration_ms=(time.perf_counter() - began) * 1000,
        )

    @staticmethod
    def _remask(current: str, r: RailResult, fn) -> str:
        """A later masking rail, re-run against the text as it now stands."""
        if fn is None:
            return r.text_out if r.text_out is not None else current
        redo = RailResult(rail=r.rail, engine=r.engine, verdict=Verdict.PASS)
        try:
            fn(redo, current)
        except Exception:  # noqa: BLE001 — the first pass already has a verdict
            return current
        return redo.text_out if redo.text_out is not None else current

    # -----------------------------------------------------------------
    def ingest(self, title: str, text: str, *, source: str = "upload",
               kind: str = "txt", session_id: str = "",
               method: str = "text", doc_id: str = "") -> IngestResult:
        """Take one document into the knowledge base.

            extract → normalize → ingest rails → chunk → index

        The order is the contract. Rails run on the whole document *before* it
        is chunked and written, so a masked value is masked in the index rather
        than at read time (`ingest.mask_before_index`, locked). A document that
        fails a rail is quarantined, not indexed with a flag
        (`ingest.quarantine_on_block`, locked) — a flag makes retrieval safety
        something a future caller has to remember.

        `doc_id`, when given, replaces whatever entry already holds that id
        instead of minting a fresh one — the one caller that needs this is
        `Engine.__init__`'s own re-ingest of a seed document that was placed in
        the store before rails existed to check it, and it has to land on the
        exact same id the corpus already indexes it under.
        """
        p = self.policy
        tracer = Tracer(session_id=session_id)
        title = (title or "Untitled document").strip()[:160]

        with tracer.stage("Ingest ingress", "accept the document", kind="rail"):
            with tracer.rail("document.accept", "in-process") as r:
                limit = int(p.get("ingest.max_document_chars"))
                r.verdict = Verdict.PASS
                r.score = float(len(text))
                r.unit = "count"
                r.meta = {"title": title, "source": source, "kind": kind,
                          "chars": len(text), "limit": limit, "extracted_by": method}
                if len(text) > limit:
                    r.verdict = Verdict.BLOCK
                    r.error = f"document is {len(text)} chars, limit is {limit}"
            with tracer.rail("vault.open", "aes-256-gcm") as r:
                r.verdict = Verdict.PASS
                r.meta = {"encrypted": self.vault.encrypted}

        oversized = tracer.trace.stages[0].verdict is Verdict.BLOCK

        with tracer.stage("Normalize", "NFKC · homoglyph fold", kind="rail"):
            with tracer.rail("unicode.normalize", "locked — safety invariant") as r:
                normalized, changed = normalize(text)
                r.verdict = Verdict.PASS
                r.score = float(changed)
                r.unit = "count"
                r.meta = {"characters_changed": changed, "can_be_disabled": False}

        scan = self.evaluate(normalized, Surface.INGEST, tracer, "Ingest rails",
                             "the document is untrusted input",
                             owner=CORPUS_OWNER)
        detections = [
            {"stage": "ingest", "rail": r.rail, **d.redacted()}
            for r in scan.results for d in r.detections
        ]
        masked_count = sum(
            len(r.detections) for r in scan.results if r.verdict is Verdict.MASK
        )

        # Chunk the *masked* text. Nothing else is ever written.
        with tracer.stage("Chunk", "paragraph packing with overlap", kind="rail"):
            with tracer.rail("document.chunk", f"{p.get('ingest.chunk_size')} chars") as r:
                chunks = chunk_text(
                    scan.text,
                    size=int(p.get("ingest.chunk_size")),
                    overlap=int(p.get("ingest.chunk_overlap")),
                )
                r.verdict = Verdict.PASS
                r.score = float(len(chunks))
                r.unit = "count"
                r.meta = {"chunks": len(chunks),
                          "chunk_size": int(p.get("ingest.chunk_size")),
                          "overlap": int(p.get("ingest.chunk_overlap"))}

        quarantined = scan.blocked or oversized or not chunks
        reason = ""
        if oversized:
            reason = "document exceeds ingest.max_document_chars"
        elif scan.blocked:
            reason = self._reason(scan.results) or "failed an ingest rail"
        elif not chunks:
            reason = "no text to index"

        doc = Document(
            id=doc_id or new_document_id(title), title=title, source=source, kind=kind,
            chars=len(text), chunks=chunks,
            status="quarantined" if quarantined else "indexed",
            verdict=scan.verdict.value, masked=masked_count,
            findings=detections, reason=reason, request_id=tracer.trace.request_id,
            method=method, rails_applied=True,
        )

        with tracer.stage("Index", "bm25 over chunks", kind="rail"):
            with tracer.rail("corpus.index", "in-memory + json") as r:
                self.corpus.add(doc)
                r.verdict = Verdict.BLOCK if quarantined else Verdict.PASS
                r.score = float(0 if quarantined else len(chunks))
                r.unit = "count"
                r.meta = {
                    "indexed": not quarantined,
                    "chunks_written": 0 if quarantined else len(chunks),
                    "corpus": self.corpus.stats(),
                }
                if quarantined:
                    r.error = f"quarantined — {reason}"
                    tracer.note("quarantined: stored for review, never returned by search")

        trace = tracer.finish(Verdict.BLOCK if quarantined else scan.verdict)
        self.audit.write(trace, detections)
        violations = self._explain([r for r in scan.results if r.verdict is not Verdict.PASS],
                                   "document")
        return IngestResult(
            document=doc, trace=trace, detections=detections,
            quarantined=quarantined, reason=reason,
            violations=[v.to_dict() for v in violations],
        )

    # -----------------------------------------------------------------
    def converse(self, question: str, history: list[dict[str, Any]] | None = None,
                 session_id: str = "", *, model: str | None = None,
                 principal: str = "") -> ConversationResult:
        """Run a request and apply the human-review trigger.

        `_converse` has several early returns — a blocked prompt, a model
        refusal, an exhausted regeneration budget. Deciding review here rather
        than at each of those sites means a new return path cannot forget to
        consult the trigger.

        `principal` owns any vault token this request mints, and is the only
        identity permitted to unmask them at egress. Callers with a real
        identity (the server, from the session cookie) must pass it; the CLI and
        library callers leave it empty, which is its own single-tenant owner
        rather than a wildcard.
        """
        result = self._converse(question, history, session_id, model=model,
                                principal=principal)
        result.human_review, result.review_reason = self._review(
            result.trace, result.trace.verdict
        )
        if result.human_review and result.trace.stages:
            result.trace.stages[-1].notes.append(
                f"queued for human review — {result.review_reason}"
            )
        return result

    def _converse(self, question: str, history: list[dict[str, Any]] | None = None,
                  session_id: str = "", *, model: str | None = None,
                  principal: str = "") -> ConversationResult:
        p = self.policy
        tracer = Tracer(session_id=session_id)
        history = history or []
        all_detections: list[dict[str, Any]] = []

        # --- ingress ---------------------------------------------------
        with tracer.stage("Ingress", "bind session, open vault", kind="rail"):
            with tracer.rail("session.bind", "in-process") as r:
                r.verdict = Verdict.PASS
                r.meta = {"session": session_id or "anonymous", "policy": p.source,
                          "principal": principal or "(none)"}
            with tracer.rail("vault.open", "aes-256-gcm") as r:
                r.verdict = Verdict.PASS
                r.meta = {"encrypted": self.vault.encrypted,
                          "owner": principal or "(none)"}

        # --- normalize (never optional) --------------------------------
        with tracer.stage("Normalize", "NFKC · homoglyph fold", kind="rail"):
            with tracer.rail("unicode.normalize", "locked — safety invariant") as r:
                normalized, changed = normalize(question)
                r.verdict = Verdict.PASS
                r.score = float(changed)
                r.unit = "count"
                r.meta = {"characters_changed": changed, "can_be_disabled": False}
        question_n = normalized

        # --- prompt rails ---------------------------------------------
        ingress = self.evaluate(question_n, Surface.USER_PROMPT, tracer, "Prompt rails",
                                owner=principal)
        all_detections += [
            {"stage": "prompt", "rail": r.rail, **d.redacted()}
            for r in ingress.results for d in r.detections
        ]

        with tracer.stage("Policy decision", "verdict precedence", kind="rail"):
            with tracer.rail("precedence.resolve", "locked — block > mask > flag > pass") as r:
                r.verdict = ingress.verdict
                r.meta = {
                    "verdicts": [x.verdict.value for x in ingress.results],
                    "resolved": ingress.verdict.value,
                }

            # The agentic step. Almost always a no-op: it fires only when a
            # scored rail landed within a margin of its own threshold, which is
            # to say when the number did not really decide anything.
            _adj_began = time.perf_counter()
            ruling = self.adjudicator.review(ingress.text, ingress.results, ingress.verdict)
            _adj_ms = (time.perf_counter() - _adj_began) * 1000
            if ruling is not None:
                with tracer.rail(self.adjudicator.name, self.adjudicator.engine) as r:
                    r.verdict = ruling.verdict
                    r.score = ruling.confidence
                    r.threshold = self.adjudicator.min_confidence
                    r.higher_is_better = True
                    r.meta = ruling.to_dict()
                # The model call happens before the rail context opens, so the
                # timer inside it would report ~0ms and hide the real cost in
                # the stage total. The stage holds this same object.
                r.duration_ms = _adj_ms
                # It joins the rail results, not just the trace: a block it
                # raised has to be able to explain itself in the refusal, and
                # `_refusal` reads this list.
                ingress.results.append(r)
                if ruling.changed:
                    ingress.verdict = ruling.verdict
                    tracer.note(
                        f"adjudicator {ruling.direction} the verdict "
                        f"{ruling.original.value} → {ruling.verdict.value}: {ruling.rationale}"
                    )

        if ingress.blocked:
            trace = tracer.finish(Verdict.BLOCK)
            self.audit.write(trace, all_detections)
            reply, violations = self._refusal(ingress.results, trace.request_id)
            return ConversationResult(
                reply=reply, trace=trace, blocked=True,
                refusal_reason=self._reason(ingress.results),
                detections=all_detections,
                violations=[v.to_dict() for v in violations],
            )

        prompt_for_model = ingress.text

        # --- retrieval -------------------------------------------------
        chunks: list[str] = []
        with tracer.stage("Retrieval", "bm25 over the index, gated on term coverage", kind="retrieval"):
            with tracer.rail("corpus.search", "bm25 + coverage gate") as r:
                chunks = retrieve(
                    question_n,
                    k=int(p.get("grounding.context_window")),
                    min_score=float(p.get("ingest.min_chunk_score")),
                )
                r.verdict = Verdict.PASS
                r.score = float(len(chunks))
                r.unit = "count"
                r.meta = {"chunks": len(chunks)}

        # Hoisted so the violations built at the end can say a masked value came
        # out of a retrieved document rather than out of the reader's own
        # message. Those are other people's details, and that distinction is the
        # whole reason for telling them at all.
        rag_results: list[RailResult] = []
        if chunks and p.enabled("pii", "retrieval"):
            # A real chunk's own text already contains blank lines — a PDF
            # table or a paragraph break — so "\n\n" cannot double as both the
            # join separator and the split point back afterward without
            # shattering every chunk that happens to contain one into several.
            # \x1e (ASCII record separator) is not a character extraction, a
            # rail, or a human ever produces, so joining and splitting on it
            # round-trips exactly N chunks in, N chunks out — unlike "\n\n",
            # which turned 6 retrieved chunks into 20-40+ fragments here,
            # every one of them handed to the grounding judge as if it were
            # its own chunk.
            sep = "\x1e"
            joined = sep.join(chunks)
            # `owner=CORPUS_OWNER`, not `principal`: a value found here was
            # quoted out of the corpus, not supplied by the caller asking the
            # question, so a token minted here must not unmask for them just
            # because they are the one who triggered the scan.
            rag = self.evaluate(joined, Surface.RETRIEVAL, tracer, "Retrieval rails",
                                "scanning retrieved context", owner=CORPUS_OWNER)
            rag_results = rag.results
            if rag.blocked:
                chunks = []
                tracer.note("retrieved context blocked — proceeding without it")
            elif rag.text != joined:
                chunks = rag.text.split(sep)
            all_detections += [
                {"stage": "retrieval", "rail": r.rail, **d.redacted()}
                for r in rag.results for d in r.detections
            ]

        if self.llm is None:
            trace = tracer.finish(ingress.verdict)
            return ConversationResult(
                reply="No API key configured — rails ran, generation skipped.",
                trace=trace, chunks=chunks, detections=all_detections,
            )

        # --- retrieval-relevance domain gate ----------------------------
        # The actual "is this in scope" answer, for anything the prompt rails
        # did not already settle: did the corpus have anything relevant?
        # `requires_retrieval` reads what `scope.domain` already computed —
        # PASS or FLAG alike — and excludes only a bare greeting and the
        # "nothing configured" skip, neither of which classified anything.
        # Checked here, after the no-key path above: with no model there is
        # no generation to protect from hallucinating, and that branch
        # already gives the honest reason nothing was answered.
        if not chunks and requires_retrieval(question_n, ingress.results):
            trace = tracer.finish(Verdict.BLOCK)
            self.audit.write(trace, all_detections)
            return ConversationResult(
                reply=NOT_FOUND_TEMPLATE.format(rid=trace.request_id),
                trace=trace, blocked=True, refusal_reason="retrieval_not_found",
                detections=all_detections,
            )

        # --- generation + output rails, with regeneration --------------
        max_regen = int(p.get("grounding.max_regenerations"))
        on_fail = str(p.get("grounding.action_on_fail"))
        reply = ""
        attempt = 0

        while True:
            attempt += 1
            label = "Generation" if attempt == 1 else f"Regeneration · attempt {attempt}"
            suffix = "" if attempt == 1 else (
                "\n\nYour previous answer contained claims the retrieved context does not "
                "support. Answer again using only the context. If the context is "
                "insufficient, say so."
            )
            with tracer.stage(label, self.llm.model, kind="model" if attempt == 1 else "retry"):
                with tracer.rail("llm.call", self.llm.model) as r:
                    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
                    user_turn = (
                        f"CONTEXT:\n{context or '(nothing retrieved)'}\n\n"
                        f"QUESTION:\n{prompt_for_model}{suffix}"
                    )
                    try:
                        gen = self.llm.generate(
                            SYSTEM_PROMPT, [*history, {"role": "user", "content": user_turn}],
                            model=model,
                        )
                        reply = gen.text
                        r.verdict = Verdict.PASS
                        r.meta = {
                            "model": gen.model,
                            "input_tokens": gen.input_tokens,
                            "output_tokens": gen.output_tokens,
                            "fell_back": gen.fell_back,
                            "attempt": attempt,
                        }
                    except Refusal as exc:
                        r.verdict = Verdict.BLOCK
                        r.error = str(exc)
                        r.meta = {"refusal_category": exc.category}
                        trace = tracer.finish(Verdict.BLOCK)
                        self.audit.write(trace, all_detections)
                        return ConversationResult(
                            reply=REFUSAL_FALLBACK.format(rid=trace.request_id),
                            trace=trace, blocked=True,
                            refusal_reason=f"model declined ({exc.category})",
                            chunks=chunks, detections=all_detections,
                        )

            # `owner=SYSTEM_OWNER`: anything newly detected here surfaced in
            # generated text, not in what the caller sent — see the note on the
            # retrieval call above. Values the caller supplied are already
            # vault tokens by the time the model sees them, so re-detecting
            # here means the value came from somewhere else.
            egress = self.evaluate(reply, Surface.LLM_RESPONSE, tracer,
                                   f"Output rails{'' if attempt == 1 else f' · attempt {attempt}'}",
                                   owner=SYSTEM_OWNER)
            all_detections += [
                {"stage": f"output.{attempt}", "rail": r.rail, **d.redacted()}
                for r in egress.results for d in r.detections
            ]

            grounding_failed = False
            if self.grounding_rail and p.enabled("grounding", "llm.response"):
                with tracer.stage(
                    f"Grounding{'' if attempt == 1 else f' · attempt {attempt}'}",
                    "claim-level consistency", kind="rail",
                ):
                    gr = self._run(
                        tracer, self.grounding_rail.name, self.grounding_rail.engine,
                        lambda r: self.grounding_rail.evaluate(
                            question_n, reply, chunks,
                            "flag" if on_fail == "flag" else "block", r,
                        ),
                    )
                grounding_failed = gr.verdict is Verdict.BLOCK

            if egress.blocked and not grounding_failed:
                trace = tracer.finish(Verdict.BLOCK)
                self.audit.write(trace, all_detections)
                reply, violations = self._refusal(egress.results, trace.request_id)
                return ConversationResult(
                    reply=reply, trace=trace, blocked=True,
                    refusal_reason=self._reason(egress.results),
                    chunks=chunks, detections=all_detections,
                    violations=[v.to_dict() for v in violations],
                )

            if not grounding_failed:
                reply = egress.text
                break

            if on_fail in ("flag", "pass"):
                tracer.note("grounding failed — flagged, response delivered")
                reply = egress.text
                break

            if attempt > max_regen:
                tracer.note(f"grounding failed {attempt}× — escalating to human review")
                tracer.trace.regenerations = attempt - 1
                trace = tracer.finish(Verdict.BLOCK)
                self.audit.write(trace, all_detections)
                return ConversationResult(
                    reply=REVIEW_TEMPLATE.format(rid=trace.request_id),
                    trace=trace, blocked=True,
                    refusal_reason="ungrounded after maximum regenerations",
                    chunks=chunks, detections=all_detections,
                )

            tracer.note(f"grounding failed — regenerating (attempt {attempt + 1})")
            tracer.trace.regenerations = attempt

        # --- egress ----------------------------------------------------
        with tracer.stage("Egress", "unmask for authorized caller", kind="rail"):
            with tracer.rail("vault.unmask", "aes-256-gcm") as r:
                revealed = 0
                if bool(p.get("pii.reversible")):
                    import re as _re

                    def _reveal(m):
                        nonlocal revealed
                        # Scoped to the principal that minted the token. A token
                        # this caller does not own stays masked rather than
                        # raising — the reply is still deliverable, it simply
                        # does not carry someone else's value.
                        val = self.vault.reveal(m.group(2), principal)
                        if val is None:
                            return m.group(0)
                        revealed += 1
                        return val

                    reply = _re.sub(r"<([A-Z_0-9]+):([0-9a-f]{12})(?:\s…[^>]*)?>", _reveal, reply)
                denials = self.vault.take_denials()
                r.verdict = Verdict.PASS
                r.score = float(revealed)
                r.unit = "count"
                r.meta = {
                    "tokens_revealed": revealed,
                    "reversible": bool(p.get("pii.reversible")),
                    "principal": principal or "(none)",
                }
                # A refused unmask is a security event: it means a token reached
                # a caller that does not own it. Surfaced here so it lands in the
                # trace and therefore in the hash-chained audit log.
                if denials:
                    r.meta["unmask_denied"] = len(denials)
                    r.meta["denial_reasons"] = sorted({d["reason"] for d in denials})
                    tracer.note(
                        f"{len(denials)} vault token(s) refused to '{principal or '(none)'}' — "
                        + ", ".join(sorted({d["reason"] for d in denials}))
                    )

            with tracer.rail("audit.write", "append-only, hash-chained") as r:
                r.verdict = Verdict.PASS
                r.meta = {"detections_recorded": len(all_detections)}

        final = precedence([ingress.verdict, egress.verdict])
        trace = tracer.finish(final)
        digest = self.audit.write(trace, all_detections)
        trace.stages[-1].rails[-1].meta["hash"] = digest[:16]

        # A delivered reply still carries violations: a masked SSN is something
        # the user should be told about even though nothing was refused.
        violations = (
            self._explain([r for r in ingress.results if r.verdict is not Verdict.PASS],
                          "prompt")
            + self._explain([r for r in rag_results if r.verdict is not Verdict.PASS],
                            "retrieved")
            + self._explain([r for r in egress.results if r.verdict is not Verdict.PASS],
                            "reply")
        )
        return ConversationResult(
            reply=reply, trace=trace, chunks=chunks, detections=all_detections,
            violations=[v.to_dict() for v in violations],
        )

    def _explain(self, rails: list[RailResult], origin: str = "prompt") -> list[Violation]:
        return explain(rails, str(self.policy.get("policy.disclosure")), origin)

    def _refusal(self, rails: list[RailResult], request_id: str) -> tuple[str, list[Violation]]:
        """Build a refusal that says what happened, at the configured disclosure."""
        violations = self._explain(rails)
        message = summarise(violations, request_id, blocked=True)
        return (message or REFUSAL_FALLBACK.format(rid=request_id)), violations

    def _review(self, trace: Trace, verdict: Verdict) -> tuple[bool, str]:
        """Does this request go to the human review queue?

        `policy.human_review.trigger` used to be declared and never read. It is
        consulted here, once, at the end of every request — including blocked
        ones, since a block is often exactly what a reviewer needs to see.
        """
        trigger = str(self.policy.get("policy.human_review.trigger"))
        if trigger == "none":
            return False, ""
        if trigger == "any block":
            return verdict is Verdict.BLOCK, "verdict was block"
        if trigger == "any mask":
            return verdict in (Verdict.MASK, Verdict.BLOCK), f"verdict was {verdict.value}"
        if trigger == "sampled 5%":
            # Deterministic on the request id — no RNG, so a trace can be
            # replayed and land in the queue the same way it did in production.
            sampled = int(trace.request_id[-2:], 16) < 13  # 13/256 ≈ 5%
            return sampled, "sampled for review"
        # "repeat failures" — the default
        if trace.regenerations > 0:
            return True, f"{trace.regenerations} regeneration(s) before delivery"
        failed = [r for r in trace.rails if r.error]
        if failed:
            return True, f"{len(failed)} rail(s) errored"
        return False, ""

    @staticmethod
    def _reason(results: list[RailResult]) -> str:
        worst = [r for r in results if r.verdict is Verdict.BLOCK]
        if not worst:
            return ""
        r = worst[0]
        if r.error:
            return f"{r.rail}: {r.error}"
        tech = r.meta.get("technique") or r.meta.get("worst_category") or ""
        return f"{r.rail}" + (f" ({tech})" if tech else "")
