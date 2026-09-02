"""Generate the pipeline flowchart embedded in web/home.html.

Run it, then paste the emitted SVG over the existing `<svg class="flow">`
block in web/home.html. Kept as a script rather than hand-authored markup:

coordinates are computed rather than hand-placed, so every node declares its
height, the cursor walks down, and edges are drawn between what the cursor
recorded. Hand-authoring absolutely-positioned SVG boxes is how diagrams
drift out of alignment the first time one label grows.

Colours are class names, never literals, so the same SVG serves both themes.

The simplified, slide-ready version: five stages plus the two things fanned
out underneath the agent. Deliberately drops what a slide doesn't need —
Gate 1-4 numbering, the seven-rail enumeration, the adjudicator, the
regenerate loop, document ingestion — in favour of one box per real
decision point. `GuardrailSupervisor` and the six-specialist `Supervisor`
(see agents/supervisor.py, guardrail_supervisor.py) are collapsed into one
"Guardrail Supervisor" box here on purpose; the full split lives in the
code and in git history, not on this diagram.
"""

from pathlib import Path

W = 1400          # viewBox width
CX = 640          # spine centre
DENY_X = 1268     # the red refusal rail runs down here
GAP = 46           # vertical gap between nodes

out: list[str] = []
y = 54
anchors: dict[str, dict] = {}


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def node(nid, kind, title, lines, w, h, cx=CX, gate_label=None, agent=False, tag=None):
    """One box. `kind` picks the class; gates are left-aligned like the reference.

    `tag`, e.g. ("AGENTIC", "agentic"), marks whether a gate's own checklist is
    planned per request by a judge, or is the same fixed job list every time —
    the distinction is in engine.py's job list (fixed) versus
    guardrail_supervisor.py's _plan() (a judge decides what even runs).
    """
    global y
    x = cx - w / 2
    out.append(f'<g class="node {kind}" data-node="{nid}">')
    out.append(f'  <rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="11"/>')
    if agent:
        out.append(f'  <text class="agent-tag" x="{x + w - 16:.0f}" y="{y + 18:.0f}">AGENT</text>')
    if tag:
        tag_text, tag_kind = tag
        out.append(f'  <text class="gate-tag {tag_kind}" x="{x + w - 16:.0f}" y="{y + 18:.0f}">{esc(tag_text)}</text>')
    if gate_label:
        tx = x + 20
        out.append(f'  <text class="gate-label" x="{tx:.0f}" y="{y + 26:.0f}">{esc(gate_label)}</text>')
        out.append(f'  <text class="title left" x="{tx:.0f}" y="{y + 51:.0f}">{esc(title)}</text>')
        for i, line in enumerate(lines):
            out.append(f'  <text class="mono left" x="{tx:.0f}" y="{y + 73 + i * 17:.0f}">{esc(line)}</text>')
    else:
        out.append(f'  <text class="title" x="{cx}" y="{y + (30 if lines else h / 2 + 5):.0f}">{esc(title)}</text>')
        for i, line in enumerate(lines):
            out.append(f'  <text class="mono" x="{cx}" y="{y + 52 + i * 18:.0f}">{esc(line)}</text>')
    out.append("</g>")
    anchors[nid] = {"x": cx, "top": y, "bottom": y + h, "left": x, "right": x + w, "h": h}
    y += h
    return anchors[nid]


def arrow(label=None, length=GAP, cx=CX):
    """Vertical connector down the spine."""
    global y
    y2 = y + length
    out.append(f'<path class="edge" d="M{cx} {y:.0f} L{cx} {y2 - 9:.0f}" marker-end="url(#arrow)"/>')
    if label:
        out.append(f'<text class="edge-label" x="{cx + 14}" y="{y + length / 2 + 4:.0f}">{esc(label)}</text>')
    y = y2


def fanout(nid, items, w, h, gap=14):
    """The parallel rails: a bus out, one box each, a bus back in."""
    global y
    total = len(items) * w + (len(items) - 1) * gap
    x0 = CX - total / 2
    bus_y = y + 20
    out.append(f'<path class="edge" d="M{CX} {y:.0f} L{CX} {bus_y:.0f}"/>')
    out.append(f'<path class="edge" d="M{x0 + w / 2:.0f} {bus_y:.0f} L{x0 + total - w / 2:.0f} {bus_y:.0f}"/>')
    top = bus_y + 26
    for i, (title, sub, kind) in enumerate(items):
        cx = x0 + i * (w + gap) + w / 2
        out.append(f'<path class="edge" d="M{cx:.0f} {bus_y:.0f} L{cx:.0f} {top - 9:.0f}" marker-end="url(#arrow)"/>')
        out.append(f'<g class="node {kind}">')
        out.append(f'  <rect x="{cx - w / 2:.0f}" y="{top:.0f}" width="{w}" height="{h}" rx="10"/>')
        out.append(f'  <text class="title small" x="{cx:.0f}" y="{top + h * 0.42:.0f}">{esc(title)}</text>')
        out.append(f'  <text class="mono" x="{cx:.0f}" y="{top + h * 0.76:.0f}">{esc(sub)}</text>')
        out.append("</g>")
    bottom = top + h
    join_y = bottom + 24
    out.append(f'<path class="edge" d="M{x0 + w / 2:.0f} {bottom:.0f} L{x0 + w / 2:.0f} {join_y:.0f}"/>')
    out.append(f'<path class="edge" d="M{x0 + total - w / 2:.0f} {bottom:.0f} L{x0 + total - w / 2:.0f} {join_y:.0f}"/>')
    for i in range(1, len(items) - 1):
        cx = x0 + i * (w + gap) + w / 2
        out.append(f'<path class="edge" d="M{cx:.0f} {bottom:.0f} L{cx:.0f} {join_y:.0f}"/>')
    out.append(f'<path class="edge" d="M{x0 + w / 2:.0f} {join_y:.0f} L{x0 + total - w / 2:.0f} {join_y:.0f}"/>')
    anchors[nid] = {"top": top, "bottom": join_y, "x": CX}
    y = join_y


def deny(from_node, label):
    """A refusal leaving a gate for the rail on the right."""
    a = anchors[from_node]
    mid = a["top"] + a["h"] / 2
    out.append(f'<path class="deny" d="M{a["right"]:.0f} {mid:.0f} L{DENY_X} {mid:.0f}"/>')
    out.append(f'<text class="deny-label" x="{a["right"] + 16:.0f}" y="{mid - 10:.0f}">{esc(label)}</text>')
    return mid


# ══════════════════════════════════════════════════════════════════
# the simplified pipeline — five stages, the way you'd say it out loud
# on a slide: Guardrail Supervisor, Input Guardrails, Agent (with its own
# tool-call rails and tools fanned out beneath it), Authorization,
# Output & Grounding. Everything that survives here is a real decision
# point; everything cut is still in the detailed diagram's git history.
# ══════════════════════════════════════════════════════════════════
node("user-in", "terminal", "Citizen or operator", [
    "asks a question",
], 440, 74)
arrow(None, 44)

gs = node("guardrail-supervisor", "gate", "Quick risk pre-check", [
    "a judge plans which checks even run — guardrail_supervisor.py",
    "then up to six specialists: pii · injection · content · scope · authorization · grounding",
], 600, 98, gate_label="1. GUARDRAIL SUPERVISOR", tag=("AGENTIC", "agentic"))
deny_y1 = deny("guardrail-supervisor", "BLOCK / ESCALATE — sent to a person")
arrow("risk & policy", 44)

ig = node("input-guardrails", "gate", "Is the question allowed through?", [
    "PII · Injection · Scope · Content · Policy — the same fixed job list, every time",
], 600, 80, gate_label="2. INPUT GUARDRAILS", tag=("FIXED", "fixed"))
deny_y2 = deny("input-guardrails", "BLOCK / REDACT")
arrow("allowed", 44)

node("agent", "model", "The agent plans", [
    "uses tools; the same fixed rails check the call and its result",
], 600, 80, agent=True)
arrow("the same fixed rails, run again on tool traffic", 40)

# Not the Supervisor's six specialist agents — agent/runner.py never imports
# agents/supervisor.py. This is engine.evaluate() again, just on the
# AGENT_TOOL / AGENT_DATA surfaces, so the active families differ from Gate 2:
# scope and grounding are `off` on both (severity matrix), content and
# injection are result-only (content.action_key + INJECTION_ALWAYS in
# engine.py), and authorization isn't a rail at all here — it's its own
# dedicated stage below, a resource_owner lookup with no rail and no model.
fanout("tool-rails", [
    ("pii", "args & result", "solid"),
    ("policy", "args & result", "solid"),
    ("content", "result only", "solid"),
    ("injection", "result only — locked on", "solid"),
], 170, 60, gap=16)
arrow("then the tool runs", 40)

fanout("tools", [
    ("search_documents", "the knowledge base", "solid"),
    ("lookup_fee", "a fee schedule", "solid"),
    ("case / grievance", "status, or file one", "solid"),
], 260, 60, gap=20)
arrow(None, 44)

auth = node("authorization", "gate", "Can this user access this resource or action?", [
    "a resource_owner lookup, not a rail or a model call — agent/runner.py",
], 600, 80, gate_label="4. AUTHORIZATION", tag=("FIXED", "fixed"))
deny_y3 = deny("authorization", "DENIED — not this caller's record")
arrow("resource + action", 44)

og = node("output-grounding", "gate", "Is the answer safe, and true to its sources?", [
    "the same fixed checks again, then every claim against what was retrieved",
], 600, 80, gate_label="5. OUTPUT & GROUNDING", tag=("FIXED", "fixed"))
deny_y4 = deny("output-grounding", "BLOCK / HUMAN — sent to a person")
arrow(None, 44)

node("response", "terminal", "The response reaches the user", [
    "only what every stage above let through",
], 470, 74)
arrow(None, 44)

trace = node("trace", "data", "TRACE", [
    "every check, decision and action — hash-chained",
], 700, 80)

# the refusal rail: down the right margin, into the trace log
audit_mid = trace["top"] + trace["h"] / 2
out.append(f'<path class="deny" d="M{DENY_X} {deny_y1:.0f} L{DENY_X} {audit_mid:.0f} '
           f'L{trace["right"] + 10:.0f} {audit_mid:.0f}" marker-end="url(#arrow-deny)"/>')


# Relative to this file, so the script runs from any working directory
# and on any machine.
OUT = Path(__file__).resolve().parent.parent / "web" / "_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="The guardrail pipeline, five stages">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
