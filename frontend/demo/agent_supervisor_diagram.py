"""Generate the six-specialist Supervisor flowchart embedded in web/home.html.

Same tool as `home_diagram.py`/`supervisor_diagram.py`, same vocabulary
(`node`/`arrow`/`fanout`/`deny`), copied rather than imported so each diagram
stays a single, runnable file.

`supervisor_diagram.py` draws `GuardrailSupervisor` — the flat MVP that calls
six *tools* directly. This one draws its sibling, `Supervisor`
(agents/supervisor.py) — the one that selects among six *specialist agents*,
each of which runs its own nested PLAN/DECIDE loop before reaching a
detector. They are different classes; this file exists because the summary
page only ever showed the flat one collapsed into a single box.

Run it, then paste the emitted SVG into web/home.html, under a new
"Inside the Guardrail Supervisor" figure.
"""

from pathlib import Path

W = 1400
CX = 640
DENY_X = 1268
GAP = 46

out: list[str] = []
y = 54
anchors: dict[str, dict] = {}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node(nid, kind, title, lines, w, h, cx=CX, gate_label=None):
    global y
    x = cx - w / 2
    out.append(f'<g class="node {kind}" data-node="{nid}">')
    out.append(f'  <rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="11"/>')
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
    global y
    y2 = y + length
    out.append(f'<path class="edge" d="M{cx} {y:.0f} L{cx} {y2 - 9:.0f}" marker-end="url(#arrow)"/>')
    if label:
        out.append(f'<text class="edge-label" x="{cx + 14}" y="{y + length / 2 + 4:.0f}">{esc(label)}</text>')
    y = y2


def fanout(nid, items, w, h, gap=14):
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
        out.append(f'  <text class="title small" x="{cx:.0f}" y="{top + 29:.0f}">{esc(title)}</text>')
        out.append(f'  <text class="mono" x="{cx:.0f}" y="{top + 50:.0f}">{esc(sub)}</text>')
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
    a = anchors[from_node]
    mid = a["top"] + a["h"] / 2
    out.append(f'<path class="deny" d="M{a["right"]:.0f} {mid:.0f} L{DENY_X} {mid:.0f}"/>')
    out.append(f'<text class="deny-label" x="{a["right"] + 16:.0f}" y="{mid - 10:.0f}">{esc(label)}</text>')
    return mid


# ══════════════════════════════════════════════════════════════════
# The six-specialist Supervisor — PLAN -> SELECT/EXECUTE -> OBSERVE ->
# DECIDE -> POLICY -> ACT -> TRACE. `agents/supervisor.py`'s own docstring
# names these same seven phases; this diagram is that loop, drawn.
#
# PLAN is one judge call, not two — "which agents are relevant" is answered
# in the same call that decides whether another round is needed, never a
# separate "understand the request" step first.
#
# EXECUTE only ever runs the agents PLAN actually named — never all six by
# default, and never more than `max_agent_calls`. Each specialist is its own
# full autonomous agent, run unmodified, exactly as it runs standalone,
# including its own nested PLAN/DECIDE and its own POLICY step — this
# diagram does not re-draw that; the six-tool version in
# `supervisor_diagram.py` is closer to what one specialist's own detector
# layer actually looks like.
# ══════════════════════════════════════════════════════════════════
node("req-in", "terminal", "A request reaches the Supervisor", [
    "the six-specialist path — agents/supervisor.py",
], 560, 74)
arrow("before any specialist is asked anything", 48)

plan = node("plan", "model", "PLAN — which specialist agents are relevant", [
    "one judge call: which agents to run, and whether another round will be needed",
], 600, 80)
deny_y1 = deny("plan", "no agent selected — ALLOW")
arrow("only the agents it actually named", 44)

fanout("select-execute", [
    ("pii", "own PLAN + DECIDE", "solid"),
    ("injection", "own PLAN + DECIDE", "solid"),
    ("content", "own PLAN + DECIDE", "solid"),
    ("scope", "own PLAN + DECIDE", "solid"),
    ("authorization", "resource_owner lookup", "solid"),
    ("grounding", "chunks, if supplied", "solid"),
], 170, 66, gap=12)
arrow("OBSERVE — each agent's own structured AgentResult, nothing else", 46)

decide = node("decide", "gate", "DECIDE — weigh what the agents found", [
    "one agent ran: its own decision carries — re-deciding from a distance is worse",
    "more than one: a second judge call weighs them against each other",
], 600, 96, gate_label="DECIDE")
arrow(None, 44)

policy = node("policy", "gate", "POLICY — the deterministic floor", [
    "PolicyEngine.decide(): floor = the strictest action any selected agent's own",
    "policy step already enforced — can only raise the recommendation, never lower it",
], 600, 96, gate_label="POLICY ENGINE")
arrow(None, 44)

node("act", "solid", "ACT", [
    "PIICapabilities.execute() — the same six-value GuardrailAction, always",
], 600, 66)
arrow(None, 44)

trace = node("trace-out", "data", "TRACE — hash-chained, nothing fabricated", [
    "every phase, every agent's own result, judge_calls counted",
], 700, 80)
arrow(None, 44)

node("req-out", "terminal", "The final action, and which agent decided it", [
    "ALLOW · MASK · REDACT · BLOCK · FLAG · ESCALATE — one of six, always",
], 470, 74)

# The "no agent selected" rail still reaches POLICY — an empty plan is
# reconciled as ALLOW with no findings, not skipped past reconciliation
# entirely. A second, unlabelled rail (drawn the same way
# `supervisor_diagram.py` draws its own two skip-the-judge paths landing in
# one place) covers everything that exits early for a different reason —
# a timeout, an exhausted agent-call budget, a malformed judge reply, or
# `max_iterations` running out without PLAN declaring itself done — all of
# which `Supervisor.run()` resolves the same way: `_escalate()`, straight to
# TRACE with ESCALATE as the final action.
p = anchors["policy"]
policy_mid = p["top"] + p["h"] / 2
out.append(f'<path class="deny" d="M{DENY_X} {deny_y1:.0f} L{DENY_X} {policy_mid:.0f} '
           f'L{p["right"] + 10:.0f} {policy_mid:.0f}" marker-end="url(#arrow-deny)"/>')

d = anchors["decide"]
decide_mid = d["top"] + d["h"] / 2
out.append(f'<path class="deny" d="M{d["right"]:.0f} {decide_mid:.0f} L{DENY_X} {decide_mid:.0f}"/>')
out.append(f'<text class="deny-label" x="{d["right"] + 16:.0f}" y="{decide_mid - 26:.0f}">'
           f'{esc("timeout · budget exhausted ·")}</text>')
out.append(f'<text class="deny-label" x="{d["right"] + 16:.0f}" y="{decide_mid - 10:.0f}">'
           f'{esc("malformed output — ESCALATE")}</text>')

t = anchors["trace-out"]
trace_mid = t["top"] + t["h"] / 2
out.append(f'<path class="deny" d="M{DENY_X} {decide_mid:.0f} L{DENY_X} {trace_mid:.0f} '
           f'L{t["right"] + 10:.0f} {trace_mid:.0f}" marker-end="url(#arrow-deny)"/>')


OUT = Path(__file__).resolve().parent.parent / "web" / "_agent_supervisor_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="The six-specialist Supervisor, PLAN through TRACE">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
