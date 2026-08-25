# Standard Operating Procedures — Guardrail Console

Operational procedures for running, testing, configuring, and maintaining this
project day to day. **`README.md`** explains what the system does and how the
pipeline works; **`DEPLOY.md`** covers hosting options in depth. This document
is the "how do I actually do X" reference — each section is a checklist you
can follow without having read the rest of the codebase first.

---

## 1. Prerequisites

- Python 3.11+ (verified against 3.14 in this environment)
- An Anthropic API key for anything model-backed — copy `.env.example` to
  `.env` and set `ANTHROPIC_API_KEY`. Everything deterministic (PII regex +
  checksums, the word automaton, the injection pattern layer, verdict
  precedence, the audit chain) runs and is testable with **no key at all**.
- `.env` is git-ignored (`.gitignore:1`). Never commit it — verify with
  `git check-ignore -v .env` before any commit that touches it.

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add ANTHROPIC_API_KEY
```

---

## 2. Running the app locally

```bash
python run.py                 # start the server — http://127.0.0.1:8000
python run.py --check         # validate config/policy.yaml against the registry, then exit
python run.py --ask "..."     # one request through the full pipeline, printed as a trace
python run.py --eval          # score against eval/suite.yaml (retrieval + rails, no API key)
python run.py --eval --answers      # adds generated-answer scoring — needs a model
```

`HOST` / `PORT` env vars override the bind address. `GUARDRAIL_CONFIG`
overrides which policy file is loaded (`--config` does the same from the CLI).

**Sign-in for manual testing:** demo accounts `admin`/`admin` (all
permissions) and `citizen`/`citizen` (`chat` only) exist in every fresh
`data/users.json`. Rotate or remove them before any URL is shared beyond your
own machine (§13).

---

## 3. Testing procedure

### 3.1 Full suite (run before every merge)

```bash
python -m pytest tests/ -q
```

All ~950+ tests run with **no API key required** — model-backed judge calls
are exercised through scripted/stubbed LLM doubles (see `tests/conftest.py`
and the `Scripted*`/`Stub*` classes in the newer test files), so the suite is
fully deterministic and safe for CI. `test_agentic_eval.py`'s live-model test
is marked `agentic_eval` and skips cleanly without `ANTHROPIC_API_KEY` — it
does not fail the run.

Expect roughly 6–7 minutes on this machine (spaCy/Presidio model loading
dominates). Run a narrower slice while iterating:

```bash
python -m pytest tests/test_guardrail_supervisor.py -q     # one file
python -m pytest tests/ -k "pipeline" -q                   # by keyword
python -m pytest tests/ -m "not presidio" -q                # skip the slow local-NER cases
```

### 3.2 Evaluation suite (regression gate, not unit tests)

```bash
python run.py --eval                       # retrieval + rails — deterministic
python run.py --eval --answers             # + generated-answer faithfulness — needs a model
python run.py --eval --json eval/report.json
```

Reports three independent numbers (recall/precision/MRR, false-positive vs.
false-negative rate, and fact coverage/grounding consistency) rather than one
aggregate score — see `README.md#measuring-it` for why they're kept apart.

### 3.3 Live end-to-end scenarios (needs `ANTHROPIC_API_KEY`)

Six scenarios drive the **real** engine and agent, not a recording, and are
runnable two ways:

```bash
# via HTTP, against a running `python run.py` instance, signed in as admin
POST /api/scenarios/{id}/run
# ids: clean, pii, injection, poisoned-doc, agentic-claim, resident-record
```

```python
# or directly, in a script/REPL
from backend.guardrails import Engine, AgentRunner, Claude, load
from backend.guardrails.evaluation import scenarios as sc

engine = Engine(load("config/policy.yaml"), Claude())
agent = AgentRunner(engine)
result = sc.run("agentic-claim", engine, agent)   # or "poisoned-doc", etc.
print(result.passed, [c.to_dict() for c in result.checks])
```

`poisoned-doc` and `agentic-claim` are the two **complex** scenarios — a
payload caught on two separate trust boundaries, and a full agent run with a
masked identifier, an entitled tool, and an approval-gated write,
respectively. Both need a live model (`needs_model=True`); without a key the
scenario reports an error rather than crashing.

### 3.4 The new pipeline route, end to end

`POST /api/pipeline/run` (added on top of the shipping `/api/agents/*`
routes — unmodified) chains, in order: `GuardrailSupervisor` (fast,
mostly-deterministic, hard-block precheck) → `Supervisor` (six specialist
agents) → `PolicyEngine.decide()` (combined floor) → `Engine.converse()`
(only if nothing upstream already stopped the request). Requires the
`agents` permission (admin, or an account granted it). Body:

```json
{"text": "...", "surface": "user.prompt"}
```

The response's `stopped_at` field tells you which stage ended the request
(`"guardrail_supervisor"`, `"policy_engine"`, or `null` if it reached
conversation) — check that first when debugging an unexpected block.

---

## 4. Changing guardrail configuration

**Never hand-edit `config/overrides.yaml`.** It is machine-written — the
diff between `config/policy.yaml` (the checked-in baseline) and whatever is
currently running.

1. Preferred path: the Parameters view in the console (`/console`, needs the
   `parameters` permission), or `PATCH /api/parameters` directly:
   ```json
   {"values": {"content.hate.threshold": 0.55}, "author": "your-name"}
   ```
2. Validation runs **before** anything is written — a rejected change (out of
   range, or a locked/safety-invariant key) leaves the running config
   untouched and returns `422` with the reason.
3. A successful change: writes `config/overrides.yaml`, appends one JSON line
   to `config-changes.log` (author + diff), reloads the engine in place.
4. **Rollback:** set the value back to baseline (removes it from the diff) or
   `POST /api/parameters/reset` to wipe all overrides back to baseline.
   Review who changed what with `GET /api/parameters/changes?limit=50`.
5. To change the checked-in baseline itself (not a runtime override), edit
   `config/policy.yaml` directly and run `python run.py --check` before
   committing — startup treats an unknown/invalid key as fatal by design.

38 parameters are **fixed** (model-bound, safety-invariant, architectural, or
compliance-required) and will always reject a `PATCH` — see
`README.md#adjustable-vs-fixed` for the categories and rationale before
assuming a parameter is a bug rather than a locked invariant.

---

## 5. Document ingestion

1. Upload via the console's Documents view, or:
   ```
   POST /api/documents/upload   multipart: file, title (form field, optional)
   POST /api/documents          {"title": "...", "text": "..."}   — pasted text
   ```
2. Every document crosses `ingest.document` rails before indexing — check the
   response's `quarantined` field. `true` means it failed a rail (most often
   the injection scanner, which is locked on for ingestion regardless of the
   severity matrix) and was **not** indexed; it is retained for review, not
   silently dropped.
3. Inspect any document, including its chunks as actually stored, with
   `GET /api/documents/{doc_id}`.
4. Remove one document (built-in or uploaded): `DELETE /api/documents/{doc_id}`.
5. Full reset back to the 36-document built-in seed, dropping every upload:
   `POST /api/documents/reset`.

---

## 6. User & access management

All under `require("users")` — the `users` permission (admin role by
default).

| Action | Call |
|---|---|
| List accounts + budgets | `GET /api/users` |
| Create an account | `POST /api/users` `{name, password, role, token_limit?, daily_limit?, monthly_limit?, model?}` |
| Change budget/model | `PATCH /api/users/{name}` |
| **Reset someone else's password** (no current password needed) | `PATCH /api/users/{name}/password` `{password}` |
| Grant/revoke one permission | `PATCH /api/users/{name}/permissions` `{permission, held}` |
| Reset usage counters | `POST /api/users/{name}/reset-usage?window=all\|total\|daily\|monthly` |
| Delete an account | `DELETE /api/users/{name}` — refuses to delete the account you're signed in as |

A person changing **their own** password (needs the current one):
`POST /api/auth/password` `{current_password, new_password}`.

A token limit of `0` means unlimited — it is the default for every new
account, not "may spend nothing."

To seed accounts at boot instead of the two demo ones, set `GUARDRAIL_USERS`:
```bash
GUARDRAIL_USERS='[{"name":"ravi","password":"…","role":"admin","display":"Ravi K."}]'
```

---

## 7. Agent turns and write-action approvals

1. `POST /api/agent/chat` `{message, session_id}` runs the tool loop. If the
   model wants to run a **write** tool (currently `file_grievance`), the call
   pauses and the response's `approval` field carries a one-use `token` and a
   human-readable (already-unmasked) `summary` of exactly what will happen.
2. Show the summary to a person. Nothing is filed until they answer:
   ```
   POST /api/agent/approve   {token, approved: true|false, session_id}
   ```
3. An approval not answered within 30 minutes expires and must be re-requested
   — it is held in process memory, so a server restart also drops any pending
   approval rather than executing it later.

---

## 8. Incident response / troubleshooting a blocked or unexpected request

1. Check `GET /api/health` first — `ok`, `model_rails`, `config` (which
   policy file loaded), and `corpus` stats. A `model_rails: false` with no
   error means content/injection-judge/grounding rails are running
   deterministic-pattern-only; that is expected without `ANTHROPIC_API_KEY`,
   not a fault.
2. Every request produces a full trace (`GET /api/traces`, or the `trace`
   field on any chat/agent/pipeline response). Read it stage by stage —
   `verdict`, which rail fired, and its `score`/`threshold`/`meta`. This is
   the source of truth; do not guess from the reply text alone.
3. For a pipeline run specifically, check `stopped_at` and
   `guardrail_supervisor.hard_blocked` first — a hard block (prompt
   injection or destructive-intent pattern match) short-circuits before any
   judge call and before `Supervisor` or `Engine.converse()` ever run.
4. Verify the audit chain hasn't been tampered with:
   `GET /api/audit/verify` (needs the `audit` permission). It hash-chains
   every request; a broken chain is reported, not silently accepted.
5. If a scenario or eval regression appears, re-run just that piece narrowly
   (§3.1–3.3) before assuming the whole suite is broken — the sections
   (retrieval / rails / answers, or per-scenario checks) are reported
   separately on purpose so you know which one moved.

---

## 9. Release / change-management checklist

Before opening a PR or merging to `main`:

- [ ] `python run.py --check` — config still validates against the registry
- [ ] `python -m pytest tests/ -q` — full suite green, no API key needed
- [ ] `python run.py --eval` — retrieval/rails regression gate still passes
- [ ] If touching a rail, a scenario, or the pipeline route: run the relevant
      live scenario(s) from §3.3 once with a real key before merging —
      the stubbed pytest suite proves composition and routing, not that the
      real judge still agrees with the scripted verdicts.
- [ ] `git status` / `git diff` reviewed — no stray files, no secrets
      (`.env`, `data/users.json`, `data/history.json` are all git-ignored;
      double-check before `git add -A`)
- [ ] For a config change: `config-changes.log` and `config/overrides.yaml`
      reflect only intended changes, or were reset

---

## 10. Deployment

Full detail in `DEPLOY.md`. Summary:

| Target | Command | Notes |
|---|---|---|
| Render | Dashboard → New → Blueprint (`render.yaml`) | free tier, no persistent disk, spins down after 15 min idle |
| Google Cloud Run | `gcloud run deploy ... --memory 1Gi --timeout 300 --set-secrets ANTHROPIC_API_KEY=...` | scales to zero, same ephemeral-disk caveat |
| Docker, locally | `docker build -t guardrail-console . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -v "$PWD/data:/app/data" guardrail-console` | the only option with real persistence |

Health check for any target: `/api/health` returning `200` with `ok: true`
means more than "the port is open" — it means config loaded and reports its
own model-rail status.

**Before handing out any URL:** rotate/remove the `admin`/`admin` and
`citizen`/`citizen` demo accounts (§6), and know that nothing rate-limits the
chat/agent/pipeline endpoints — each request spends real API budget.

---

## 11. Data & secrets handling

- `data/` (gitignored) holds `corpus.json`, `users.json`, `history.json` —
  runtime state, not source. On Render/Cloud Run without a mounted volume it
  resets on every deploy; only Docker with `-v $PWD/data:/app/data` persists
  it.
- The vault (PII → token mapping) is in-process only — it does not survive a
  restart. Do not depend on unmasking working across a redeploy.
- `.env`, `config-changes.log` (may contain values changed via the API), and
  anything under `data/` should never be committed. Run
  `git status` after any broad `git add` and inspect anything unfamiliar
  before staging.

---

## 12. Command reference

```bash
# setup
pip install -r requirements.txt
cp .env.example .env

# run
python run.py
python run.py --check
python run.py --ask "..."
python frontend/demo/flow.py --sample injection

# test
python -m pytest tests/ -q
python -m pytest tests/test_guardrail_supervisor.py tests/test_pipeline_http.py -q
python run.py --eval
python run.py --eval --answers

# operate (HTTP, signed in)
POST /api/scenarios/{id}/run
POST /api/pipeline/run           {"text": "...", "surface": "user.prompt"}
GET  /api/health
GET  /api/audit/verify
PATCH /api/parameters            {"values": {...}, "author": "..."}
POST /api/parameters/reset
```
