"""The evaluation harness.

Three sections, because "is this stack any good" is three questions that fail
independently:

  **retrieval** — did the index surface the right document? Deterministic, no
  model, no cost. Recall@k, precision@k, MRR, and the out-of-corpus cases where
  the correct answer is *nothing*.

  **rails** — did the guardrails fire when they should, and stay quiet when they
  shouldn't? Reported as two separate rates, because they trade against each
  other and an aggregate accuracy hides which way a change moved things. Most
  cases run without an API key.

  **answers** — is the reply grounded, and does it contain what the corpus
  actually says? Needs a model, so it is opt-in.

A grounding score of 0.94 across thirty ad-hoc requests is an observation. This
is the thing that turns it into a measurement: fixed questions, labelled
expectations, and a number that moves when the system gets worse.
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..engine import Engine
from ..types import Surface, Verdict


class EvalError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass
class RetrievalCase:
    id: str
    question: str
    relevant: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RailCase:
    id: str
    text: str
    expect: str = "pass"
    surface: str = "user.prompt"
    kinds: list[str] = field(default_factory=list)
    note: str = ""
    #: Only a judge (or a local model) can catch this one. Without either, the
    #: case is skipped rather than counted as a miss — a deterministic run that
    #: reported a false negative for a semantic case would make the
    #: no-API-key regression gate impossible to keep green, and a gate nobody
    #: can keep green is a gate that gets deleted.
    needs_model: bool = False


@dataclass
class AnswerCase:
    id: str
    question: str
    must_include: list[str] = field(default_factory=list)
    must_admit_gap: bool = False
    min_consistency: float = 0.0
    note: str = ""


@dataclass
class Suite:
    retrieval: list[RetrievalCase] = field(default_factory=list)
    rails: list[RailCase] = field(default_factory=list)
    answers: list[AnswerCase] = field(default_factory=list)
    source: str = ""


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.exists():
        raise EvalError(f"evaluation suite not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Suite(
            retrieval=[RetrievalCase(**c) for c in doc.get("retrieval") or []],
            rails=[RailCase(**c) for c in doc.get("rails") or []],
            answers=[AnswerCase(**c) for c in doc.get("answers") or []],
            source=str(path),
        )
    except TypeError as exc:
        raise EvalError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class Row:
    id: str
    ok: bool
    summary: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ok": self.ok, "summary": self.summary, "detail": self.detail}


@dataclass
class Section:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    rows: list[Row] = field(default_factory=list)
    skipped: str = ""

    @property
    def failures(self) -> list[Row]:
        return [r for r in self.rows if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "metrics": self.metrics, "skipped": self.skipped,
            "cases": len(self.rows), "failures": len(self.failures),
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class Report:
    sections: list[Section] = field(default_factory=list)
    suite: str = ""
    elapsed_ms: float = 0.0
    model_rails: bool = False

    @property
    def failures(self) -> int:
        return sum(len(s.failures) for s in self.sections)

    @property
    def cases(self) -> int:
        return sum(len(s.rows) for s in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_rails": self.model_rails,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cases": self.cases,
            "failures": self.failures,
            "sections": [s.to_dict() for s in self.sections],
        }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def run_retrieval(suite: Suite, engine: Engine, k: int = 4) -> Section:
    """Recall, precision, and MRR against labelled relevant documents.

    Out-of-corpus questions are scored the other way round: returning nothing is
    correct, and any hit is a false positive. Without them a retriever that
    always answers looks perfect.
    """
    section = Section("retrieval")
    min_score = float(engine.policy.get("ingest.min_chunk_score"))
    recalls, precisions, rr = [], [], []
    in_corpus = out_corpus = out_clean = 0

    for case in suite.retrieval:
        hits = engine.corpus.search(case.question, k=k, min_coverage=min_score)
        got = []
        for h in hits:                       # de-duplicate: chunks, not documents
            if h.doc_id not in got:
                got.append(h.doc_id)
        relevant = set(case.relevant)

        if not relevant:
            out_corpus += 1
            ok = not got
            out_clean += int(ok)
            section.rows.append(Row(
                case.id, ok,
                "returned nothing, correctly" if ok else f"{len(got)} spurious hits",
                "" if ok else ", ".join(got[:3]),
            ))
            continue

        in_corpus += 1
        found = [d for d in got if d in relevant]
        recall = len(found) / len(relevant)
        precision = len(found) / len(got) if got else 0.0
        rank = next((i + 1 for i, d in enumerate(got) if d in relevant), 0)
        recalls.append(recall)
        precisions.append(precision)
        rr.append(1 / rank if rank else 0.0)
        ok = rank == 1                       # the right document, first
        section.rows.append(Row(
            case.id, ok,
            f"rank {rank}" if rank else "missed",
            f"expected {', '.join(sorted(relevant))} · got {', '.join(got) or 'nothing'}",
        ))

    section.metrics = {
        "questions": len(suite.retrieval),
        "in_corpus": in_corpus,
        "recall_at_k": round(statistics.mean(recalls), 3) if recalls else None,
        "precision_at_k": round(statistics.mean(precisions), 3) if precisions else None,
        "mrr": round(statistics.mean(rr), 3) if rr else None,
        "hit_at_1": round(sum(1 for x in rr if x == 1.0) / len(rr), 3) if rr else None,
        "out_of_corpus": out_corpus,
        "out_of_corpus_silent": out_clean,
        "k": k,
    }
    return section


# ---------------------------------------------------------------------------
# Rails
# ---------------------------------------------------------------------------
SURFACES = {s.value: s for s in Surface}
RANK = {"pass": 0, "flag": 1, "mask": 2, "block": 3}


def run_rails(suite: Suite, engine: Engine) -> Section:
    """False positives and false negatives, reported separately.

    They are separate because they trade against each other: one aggregate
    accuracy number lets a change that blocks twice as much legitimate traffic
    look like an improvement.
    """
    from ..tracing import Tracer

    section = Section("rails")
    fp = fn = exact = 0
    positives = negatives = 0
    kind_hits = kind_total = 0
    skipped = 0

    # A semantic case needs something that can read meaning. Either layer will
    # do; with neither, the case is not evidence about this run.
    from ..rails import toxicity_check
    semantic_available = engine.llm is not None or toxicity_check.classifier() is not None

    for case in suite.rails:
        surface = SURFACES.get(case.surface)
        if surface is None:
            section.rows.append(Row(case.id, False, f"unknown surface {case.surface}"))
            continue

        if case.needs_model and not semantic_available:
            skipped += 1
            section.rows.append(Row(case.id, True, "skipped",
                                    "needs a judge or a local model"))
            continue

        tracer = Tracer(session_id="eval")
        result = engine.evaluate(case.text, surface, tracer, "eval")
        actual = result.verdict.value
        expected = case.expect
        found_kinds = {d.kind for r in result.results for d in r.detections}

        if expected == "pass":
            negatives += 1
            ok = actual == "pass"
            fp += int(not ok)
        else:
            positives += 1
            # Catching it harder than labelled is not a miss; letting it through is.
            ok = RANK[actual] >= RANK[expected]
            fn += int(actual == "pass")
        exact += int(actual == expected)

        if case.kinds:
            kind_total += 1
            missing = [k for k in case.kinds if k not in found_kinds]
            if missing:
                ok = False
            else:
                kind_hits += 1
        else:
            missing = []

        detail = f"expected {expected}, got {actual}"
        if missing:
            detail += f" · missing {', '.join(missing)}"
        if case.note and not ok:
            detail += f" · {case.note}"
        section.rows.append(Row(case.id, ok, actual, detail))

    section.metrics = {
        "cases": len(suite.rails),
        "scored": len(suite.rails) - skipped,
        # Named rather than silently folded in: a rate computed over fewer
        # cases than the suite contains should say so.
        "skipped_needs_model": skipped,
        "should_fire": positives,
        "should_stay_quiet": negatives,
        "false_positive_rate": round(fp / negatives, 3) if negatives else None,
        "false_negative_rate": round(fn / positives, 3) if positives else None,
        "exact_verdict_match": round(exact / len(suite.rails), 3) if suite.rails else None,
        "detection_kinds_matched": f"{kind_hits}/{kind_total}" if kind_total else None,
    }
    return section


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d[\d,]*")


def _figures(text: str) -> set[str]:
    """Digit strings, comma-normalised, two digits or more.

    Two or more because a numbered list is not a claim: 1. and 2. are
    formatting, and counting them as figures made every structured answer
    look unfaithful."""
    out = set()
    for raw in _NUMBER.findall(text or ""):
        digits = raw.replace(",", "").strip()
        if len(digits) >= 2:
            out.add(digits)
    return out
_GAP = re.compile(
    r"\b(?:does not|doesn't|do not|don't|no |not )\b.{0,40}"
    r"\b(?:contain|include|cover|say|specify|state|have|mention|find)\b|"
    r"\b(?:i (?:would )?(?:need|can't|cannot)|not in the (?:context|source|material))\b",
    re.I,
)


def run_answers(suite: Suite, engine: Engine) -> Section:
    """Faithfulness and coverage on generated answers.

    Coverage is a substring check on purpose: the facts being looked for are
    figures and form numbers, and a fuzzy match on "1,200" is not more
    informative than an exact one — it is just harder to explain when it fails.
    """
    section = Section("answers")
    if engine.llm is None:
        section.skipped = "no model configured — set ANTHROPIC_API_KEY"
        return section

    coverage, consistency, relevance = [], [], []
    unsupported_total: list[str] = []
    for case in suite.answers:
        result = engine.converse(case.question, session_id="eval")
        reply = result.reply
        trace = result.trace.to_dict()
        grounding = next(
            (r for s in trace["stages"] for r in s["rails"]
             if r["rail"] == "grounding.consistency"), None,
        )
        if grounding:
            consistency.append(grounding["meta"].get("consistency", grounding["score"]))
            # `relevance` is legitimately `None` — key present, not just
            # missing — when grounding short-circuited on the local NLI
            # layer alone (`relevance_scored: False`): a natural-language-
            # inference model has no opinion on "does this answer the
            # question", only on "is this claim entailed", so nothing scored
            # it. `.get(..., 0.0)` only substitutes a default for a *missing*
            # key, not a key whose value is `None`, so it let a real `None`
            # through to `statistics.mean()` and crashed the whole eval run.
            case_relevance = grounding["meta"].get("relevance")
            if case_relevance is not None:
                relevance.append(case_relevance)

        problems: list[str] = []
        if case.must_include:
            present = [f for f in case.must_include if str(f).lower() in reply.lower()]
            coverage.append(len(present) / len(case.must_include))
            missing = [f for f in case.must_include if f not in present]
            if missing:
                problems.append("missing " + ", ".join(str(m) for m in missing))
        # Every figure in an answer should trace to the context it was given,
        # or to the question itself. This replaced a 'does the reply contain a
        # number' check, which flagged 1,200 and 2,400 — figures the corpus
        # does contain — and called a correct answer a failure.
        allowed = _figures(" ".join(result.chunks)) | _figures(case.question)
        unsupported = sorted(_figures(reply) - allowed)
        unsupported_total.extend(unsupported)
        if unsupported:
            problems.append("figures not in the retrieved context: "
                            + ", ".join(unsupported[:4]))
        if case.must_admit_gap and not _GAP.search(reply):
            problems.append("did not say what was missing")
        if case.min_consistency and grounding and grounding["score"] < case.min_consistency:
            problems.append(f"consistency {grounding['score']:.2f} < {case.min_consistency}")
        if result.blocked:
            problems.append("blocked")

        section.rows.append(Row(
            case.id, not problems,
            f"consistency {grounding['score']:.2f}" if grounding else "no grounding",
            "; ".join(problems),
        ))

    section.metrics = {
        "questions": len(suite.answers),
        "fact_coverage": round(statistics.mean(coverage), 3) if coverage else None,
        "consistency_mean": round(statistics.mean(consistency), 3) if consistency else None,
        "relevance_mean": round(statistics.mean(relevance), 3) if relevance else None,
        "unsupported_figures": len(unsupported_total),
    }
    return section


# ---------------------------------------------------------------------------
def run(suite: Suite, engine: Engine, *, answers: bool = False, k: int = 4) -> Report:
    began = time.perf_counter()
    report = Report(suite=suite.source, model_rails=engine.llm is not None)
    report.sections.append(run_retrieval(suite, engine, k=k))
    report.sections.append(run_rails(suite, engine))
    if answers:
        report.sections.append(run_answers(suite, engine))
    else:
        skipped = Section("answers", skipped="not requested — pass --answers")
        report.sections.append(skipped)
    report.elapsed_ms = (time.perf_counter() - began) * 1000
    return report
