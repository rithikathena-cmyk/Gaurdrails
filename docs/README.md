# docs

| | |
|---|---|
| **`guardrails-flow-script.html` / `.pdf`** | One page: the same pipeline diagram the product's own home page shows (Supervisor Agent → six specialists → tools/RAG → LLM → Output Supervisor Agent → human/response), a table of which entry point (`/api/chat`, `/api/agent/chat`, `/api/pipeline/run`) runs it and how, and a nine-card "meet the agents" roster. No prose deep-dive — see the explainer below for that. Note: it corrects one thing the live diagram's own caption in `home.html` hasn't caught up to yet — both prefilter flags now default to `"agentic"` (on), not `"off"`. |
| **`guardrails-agents-and-pii-explainer.html` / `.pdf`** | The older, broader internals reference — same territory as the flow script above plus a deeper PII-methods section, but that PII section (§07) still describes the deterministic regex/checksum layer that has since been removed. Treat `guardrails-flow-script.html` as the current source for PII specifics; this one's structure (agents, flow, file map) otherwise still holds. |
| **`guardrails-demo-script.html` / `.pdf`** | A presenter's runbook for giving a live demo — the flow diagram explained first (ingestion, chat turn, agent turn, each mapped to the beats that exercise it), a plain numbered technical walkthrough of the same pipeline, then nine beats with a line to say, a copy-ready input, and what should happen. Three timing plans (5/15/30+ min), a closing line, and a Q&A backup section. |
| **`guardrails-manual-test-questions.html` / `.pdf`** | No narration — just the questions to paste into the console (or click as a chip), who to sign in as, and the exact result each one produced when run live against this instance on 2026-09-01. Use this to check the stack still behaves, or as a quick reference while demoing. |

The PDF is rendered from the HTML, which is the source. Regenerate after editing:

```bash
chrome --headless --no-pdf-header-footer \
  --print-to-pdf=docs/guardrails-demo-script.pdf \
  file:///ABSOLUTE/PATH/docs/guardrails-demo-script.html
```

Every figure in it is measured on a running deployment rather than estimated, so
re-measure before changing a number — `python run.py --compare` and the trace view are
where they come from.
