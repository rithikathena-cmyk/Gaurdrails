# Deploying

Two free options, both running the **base install** — `requirements.txt` only, no
local classifiers.

## What "free" covers, and what it does not

Hosting is free on either platform below. **The Anthropic API is not.** There is no
free API tier, and a chat turn makes about six judge calls, so with Haiku and prompt
caching expect roughly **$0.002–0.005 per request** — a few dollars per thousand. Cheap,
but not zero, and worth knowing before you hand the URL to anyone.

Without a key the app still starts and still does real work:

| Runs | Skipped |
|---|---|
| PII detection with checksums, the vault, masking and unmasking | content safety |
| the injection pattern layer | semantic injection |
| the word automaton, policy rule sets | scope, grounding |
| normalization, verdict precedence, the audit chain | |

The console says the model rails are off rather than showing green. That is a genuine
demonstration of about half the system, at no cost at all.

## Why no local models

Measured on this app, not estimated:

```
process RSS, base install         97 MB
process RSS, local models loaded   1,090 MB
```

The free tiers below give 512 MB. The local layer also needs ~1.6 GB of weights on
disk, and the evaluation put its contribution at **4.6% of judge calls — inside
run-to-run noise**. It is a poor trade even where it fits.

The Dockerfile therefore installs `requirements.txt` and never
`requirements-local.txt`. With `transformers` absent, `available()` returns `False`,
the content, injection and grounding rails are built judge-only, and everything behaves
as the README's base path describes. Nothing is silently degraded.

---

## Render

```bash
# 1. Push the repo to GitHub (already done if you are reading this in git).
# 2. Render dashboard -> New -> Blueprint -> pick the repo.
#    render.yaml is picked up automatically.
# 3. It will prompt for ANTHROPIC_API_KEY. Paste it there — never commit it.
```

Health check is `/api/health`, which reports config errors and whether the model rails
are live, so a green check means more than "the port is open".

**Two free-plan limits that matter here.**

*No persistent disk.* `data/` holds accounts, live sessions, the corpus and the
transcripts, and it resets on every deploy. The built-in demo accounts come back and the
corpus reseeds, so the app works — but anything a user added is gone. Add a paid disk
mounted at `/app/data` if that is not acceptable.

*Spin-down after 15 minutes idle.* The next request pays a cold start on top of this
app's own 20–50s judge latency, so the first one after a quiet period can take a minute.
When demoing, ask a cheap question first to wake it.

---

## Google Cloud Run

```bash
PROJECT=your-project-id
gcloud run deploy guardrail-console \
  --source . \
  --project "$PROJECT" \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

Create the secret once first:

```bash
printf '%s' "sk-ant-..." | gcloud secrets create anthropic-api-key --data-file=-
```

Three flags are load-bearing:

- **`--timeout 300`.** The default is 300s and this app's requests take 20–50s. Do not
  lower it.
- **`--memory 1Gi`.** 512Mi is enough for the process at 97 MB, but leaves nothing for
  a request spike. 1Gi is still inside the free allowance for light traffic.
- **`--set-secrets`, not `--set-env-vars`.** An env var set on the command line ends up
  in your shell history and in the service description.

Cloud Run scales to zero, so the same cold-start caveat applies. The filesystem is
ephemeral in the same way — `data/` does not survive a new revision.

---

## Locally, with Docker

```bash
docker build -t guardrail-console .
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -v "$PWD/data:/app/data" \
  guardrail-console
```

The volume mount is what gives you persistence: accounts, sessions and the corpus
survive a container restart. Neither free tier above offers the equivalent.

---

## Before you hand out the URL

The README is explicit that this is not an identity system: no password policy, no
lockout, no rate limiting, and the demo accounts `admin/admin` and `citizen/citizen`
exist in every fresh deployment.

- **Change or remove the demo accounts.** They are created whenever `data/users.json`
  is absent, which on an ephemeral filesystem is every deploy.
- **Nothing rate-limits the chat endpoint.** Each request costs you money at the API.
  A public URL with no limiter is a public URL that spends your balance.
- **Config editing has no separate authorisation** beyond the `parameters` permission.
  An operator account can change any adjustable rail.

None of that is a problem for a demo you are showing someone. All of it is a problem
for a URL you leave up.

---

## Verified, and not

Checked on this machine:

- `HOST=0.0.0.0 PORT=9123 python run.py` binds and answers `/api/health` with 200 —
  the env-var path both platforms rely on.
- `toxicity_check.available()` returns `False` with `transformers` absent, and the
  content rail is then built judge-only.
- The full suite passes: 476 tests.

The image builds and runs:

```
docker build         858 MB image, exit 0
container healthy    ~6s from start
RAM serving traffic  178 MB   — inside Render's 512 MB
a live PII request   verdict=mask, 25 rails, 15.8s, no raw value in the reply
/ · /summary · /console   200, 200, 200
```

The first clean build found a real bug: `python-multipart` was missing from
`requirements.txt`. FastAPI only raises about it when the upload route is *built*, so it
does not fail at import — it fails at startup, and only where the package is genuinely
absent. Locally it was present as somebody else's transitive dependency, so nothing had
ever noticed. It is pinned now.
