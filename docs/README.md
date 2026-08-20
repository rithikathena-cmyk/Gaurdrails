# docs

| | |
|---|---|
| **`guardrails-explained.html` / `.pdf`** | How the stack works and how to demo it: the pipeline stage by stage, how a rail decides, what every parameter does and how to change it, how the agent's boundaries work, and a step-by-step script for walking someone through it. Twelve pages. |
| `architecture.html` / `.pdf` | Earlier architecture write-up. |

The PDF is rendered from the HTML, which is the source. Regenerate after editing:

```bash
chrome --headless --no-pdf-header-footer \
  --print-to-pdf=docs/guardrails-explained.pdf \
  file:///ABSOLUTE/PATH/docs/guardrails-explained.html
```

Every figure in it is measured on a running deployment rather than estimated, so
re-measure before changing a number — `python run.py --compare` and the trace view are
where they come from.
