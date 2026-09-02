#!/usr/bin/env python
"""Entrypoint.

    python run.py            start the server
    python run.py --check    validate config against the registry and exit
    python run.py --ask "…"  one request through the stack, printed as a trace
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles default to cp1252, which mangles the arrows and dashes in
# config error messages. Force UTF-8 so a validation error is readable.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-tty
        pass

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")


def cmd_check(path: str) -> int:
    from backend.guardrails import ConfigError, load
    from backend.guardrails.registry import ADJUSTABLE, LOCKED

    try:
        policy = load(path)
    except ConfigError as exc:
        print(f"\n  config rejected\n\n{exc}\n", file=sys.stderr)
        return 1

    print(f"\n  {policy.source} is valid\n")
    print(f"  {len(ADJUSTABLE)} adjustable parameters, {len(LOCKED)} fixed")
    print(f"  lexicons: {', '.join(f'{k}={len(v)}' for k, v in policy.lexicons.items())}")
    # Every surface the registry declares, not a hardcoded three — a new surface
    # that does not appear here is a column nobody reviews.
    print("\n  severity matrix")
    from backend.guardrails.registry import SURFACES

    width = max(len(s["label"]) for s in SURFACES) + 2
    print("    " + "family".ljust(12) + "".join(s["label"].ljust(width) for s in SURFACES))
    for family in sorted(policy.matrix):        # whatever the config declares
        row = "    " + family.ljust(12)
        row += "".join(policy.severity(family, s["key"]).ljust(width) for s in SURFACES)
        print(row)
    print()
    return 0


def cmd_ask(path: str, question: str) -> int:
    from backend.guardrails import AuditLog, Claude, Engine, LLMError, load

    policy = load(path)
    llm = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llm = Claude(judge_model=str(policy.get("content.judge_model")))
        except LLMError as exc:
            print(f"  {exc}\n", file=sys.stderr)
    else:
        print("  ANTHROPIC_API_KEY not set - deterministic rails only\n")

    engine = Engine(policy, llm, AuditLog("audit.log"))
    result = engine.converse(question, session_id="cli")
    t = result.trace

    print(f"\n  {t.request_id}  verdict={t.verdict.value}  "
          f"{t.total_ms:.0f}ms total, {t.guardrail_ms:.0f}ms in rails\n")
    for i, stage in enumerate(t.stages, 1):
        print(f"  {i:02d}  {stage.name:<34} {stage.duration_ms:>7.1f}ms  {stage.verdict.value}")
        for r in stage.rails:
            if r.threshold in (0.0, 1.0) and r.score == 0.0:
                measure = ""
            elif r.unit == "count":
                measure = f"{r.score:.0f} hit{'' if r.score == 1 else 's'}"
            else:
                measure = f"{r.score:.2f} / {r.threshold:.2f}"
            flag = "   <- " + r.error if r.error else ""
            print(f"        {r.rail:<28} {measure:<16} {r.duration_ms:>7.1f}ms  "
                  f"{r.verdict.value}{flag}")
        for note in stage.notes:
            print(f"        - {note}")
    print(f"\n  {result.reply}\n")
    return 0



def cmd_eval(path: str, suite_path: str, answers: bool, json_out: str) -> int:
    """Score the stack against a labelled suite.

    Retrieval and rails are deterministic and free; answers cost a model call
    per question, so they are opt-in rather than a surprise.
    """
    from backend.guardrails import AuditLog, Claude, Corpus, Engine, LLMError, load
    from backend.guardrails.evaluation import EvalError, load_suite, run

    policy = load(path)
    try:
        suite = load_suite(suite_path)
    except EvalError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1

    llm = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llm = Claude(judge_model=str(policy.get("content.judge_model")))
        except LLMError as exc:
            print(f"  {exc}\n", file=sys.stderr)

    # A scratch corpus: the suite is labelled against the built-in documents, so
    # scoring must not depend on whatever happens to be uploaded today.
    engine = Engine(policy, llm, AuditLog("audit.log"), Corpus(seed=True))
    # `force=True`: an in-memory corpus has no disk path, so `Engine.__init__`'s
    # own call already skipped this — leaving the seed document permanently
    # "unprocessed" from retrieval's PII-freshness check, which would silently
    # score the slow always-rescan fallback path on every question instead of
    # what a real deployment (a disk-backed corpus) actually does.
    engine.reseed_builtin_rails(force=True)
    report = run(suite, engine, answers=answers)

    rails_state = "model rails live" if report.model_rails else "deterministic rails only"
    print(f"\n  {suite.source}   {report.cases} cases   {rails_state}\n")

    for section in report.sections:
        if section.skipped:
            print(f"  {section.name:<10} skipped — {section.skipped}\n")
            continue
        print(f"  {section.name.upper()}")
        for key, value in section.metrics.items():
            if value is None:
                continue
            print(f"      {key.replace('_', ' '):<26} {value}")
        failures = section.failures
        if failures:
            print(f"      {'failing cases':<26} {len(failures)}")
            for row in failures:
                print(f"        x {row.id:<24} {row.summary}")
                if row.detail:
                    print(f"          {row.detail}")
        else:
            print(f"      {'failing cases':<26} none")
        print()

    verdict = "clean" if report.failures == 0 else f"{report.failures} failing"
    print(f"  {report.cases} cases in {report.elapsed_ms / 1000:.1f}s — {verdict}\n")

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"  report written to {json_out}\n")

    return 1 if report.failures else 0


def cmd_compare(path: str, suite_path: str) -> int:
    """Measure the local layer against the judge on the same labelled cases.

    Exits non-zero on a regression, so this can gate a change rather than
    merely describe one.
    """
    from backend.guardrails import AuditLog, Claude, LLMError, load
    from backend.guardrails.evaluation import compare
    from backend.guardrails.evaluation.suite import load_suite

    suite = load_suite(suite_path)
    policy = load(path)
    llm = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llm = Claude(judge_model=str(policy.get("content.judge_model")))
        except LLMError as exc:
            print(f"  {exc}\n", file=sys.stderr)

    print(f"\n  {suite.source} · {len(suite.rails)} rail cases · both arms\n")
    result = compare.run(suite, path, llm=llm, audit=AuditLog("audit.log"))
    print(compare.render(result))

    # Written out because this run costs real judge calls: the next question
    # about the numbers should be answerable without paying for them twice.
    out = Path("eval/comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"  written to {out}\n")
    return 1 if result.regressions else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Guardrail Console")
    ap.add_argument("--config", default=os.getenv("GUARDRAIL_CONFIG", "config/policy.yaml"))
    ap.add_argument("--check", action="store_true", help="validate config and exit")
    ap.add_argument("--ask", metavar="TEXT", help="run one request through the stack")
    ap.add_argument("--compare", action="store_true",
                    help="judge-only vs local+judge over the evaluation suite")
    ap.add_argument("--eval", action="store_true", help="score against a labelled suite")
    ap.add_argument("--suite", default="eval/suite.yaml", help="the suite to score against")
    ap.add_argument("--answers", action="store_true",
                    help="include the generated-answer section (one model call per question)")
    ap.add_argument("--json", metavar="PATH", default="", help="write the report as JSON")
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    os.environ["GUARDRAIL_CONFIG"] = args.config

    if args.check:
        return cmd_check(args.config)
    if args.ask:
        return cmd_ask(args.config, args.ask)
    if args.compare:
        return cmd_compare(args.config, args.suite)
    if args.eval:
        return cmd_eval(args.config, args.suite, args.answers, args.json)

    if cmd_check(args.config) != 0:
        return 1
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  ANTHROPIC_API_KEY not set - model rails will be skipped.")
        print("  Copy .env.example to .env and add your key.\n")

    import uvicorn

    print(f"  http://{args.host}:{args.port}\n")
    uvicorn.run("backend.server.app:app", host=args.host, port=args.port, reload=args.reload,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
