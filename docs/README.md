# docs

| | |
|---|---|
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
