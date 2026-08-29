# docs

| | |
|---|---|
| **`project-demo-review.html` / `.pdf`** | Demo-review focused: the end-to-end flow (ingestion, chat, agent), the concrete technique behind every checkpoint, and both agent layers side by side — `AgentRunner`'s tool loop and the `GuardrailSupervisor` → `Supervisor` (six specialists) → `PolicyEngine` reasoning pipeline, with a traced, measured example and a demo script. |
| **`guardrails-explained.html` / `.pdf`** | How the stack works and how to demo it: the pipeline stage by stage, how a rail decides, what every parameter does and how to change it, how the agent's boundaries work, and a step-by-step script for walking someone through it. Twelve pages. |
| `architecture.html` / `.pdf` | Earlier architecture write-up. |
| **`local-testing-briefing.html` / `.pdf`** | Findings from a local test pass against the real seeded corpus: rail logic scores perfectly (39/39, zero false positives/negatives), but a retrieval-side PII latency budget silently drops real answers under sustained load — root cause, evidence, and a prioritized fix list. |
| `guardrails-test-charter.html` / `.pdf` | A 51-case end-to-end question set for manual or scripted testing — grounded Q&A, PII masking (every entity type plus checksum-based false-positive controls), prompt injection, secrets, destructive intent, content safety, and known gaps, each with a copy-ready input and expected verdict. |

The PDF is rendered from the HTML, which is the source. Regenerate after editing:

```bash
chrome --headless --no-pdf-header-footer \
  --print-to-pdf=docs/guardrails-explained.pdf \
  file:///ABSOLUTE/PATH/docs/guardrails-explained.html
```

Every figure in it is measured on a running deployment rather than estimated, so
re-measure before changing a number — `python run.py --compare` and the trace view are
where they come from.
