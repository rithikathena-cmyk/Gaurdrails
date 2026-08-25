"""Generate the Guardrail Supervisor flowchart embedded in web/home.html.

Same tool as `home_diagram.py`, same vocabulary (`node`/`arrow`/`fanout`/
`deny`), copied rather than imported so each diagram stays a single,
runnable file — the point of the original was never having to keep two
scripts' import paths in sync just to hand-splice one `<svg>` block.

Run it, then paste the emitted SVG over the second `<svg class="flow">`
block in web/home.html (the one under "The Guardrail Supervisor").
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
# the Guardrail Supervisor — PLAN -> SELECT -> EXECUTE -> OBSERVE ->
# DECIDE -> ENFORCE -> TRACE. Two places skip the judge entirely: the
# PRECHECK gate (an obvious case, before PLAN ever runs) and the DECIDE
# gate's two tails (risk_proxy outside the configured band). Both peel
# off to the same right-margin rail and land in ENFORCE, exactly the
# way home_diagram.py's four refusal gates all land in one audit box —
# the same node reached three different ways is the point: ENFORCE has
# the last word regardless of how a request got there.
# ══════════════════════════════════════════════════════════════════
node("req-in", "terminal", "A request reaches the Guardrail Supervisor", [
    "the flat MVP path — one supervisor, six tools, no nested agent",
], 560, 74)
arrow("before a model is asked anything", 48)

precheck = node("precheck", "gate", "PRECHECK — is this obviously dangerous?", [
    "detect_prompt_injection + detect_destructive_intent, deterministically",
], 600, 88, gate_label="PRECHECK")
deny_y1 = deny("precheck", "HARD_BLOCK — 0 judge calls, straight to ENFORCE")
arrow("clear — nothing obvious matched", 44)

node("plan", "model", "PLAN — which checks are worth running", [
    "one judge call: risk categories, tools, and which policies to look up",
], 600, 80)
arrow("the tools it named, and any policy keys", 44)

fanout("select-execute", [
    ("detect_pii", "regex + presidio", "solid"),
    ("detect_prompt_injection", "pattern layer", "solid"),
    ("detect_destructive_intent", "policy rules", "solid"),
    ("check_scope", "domain vocabulary", "solid"),
    ("check_semantic_risk", "content categories", "solid"),
    ("get_policy", "pii · pii.<entity> · injection …", "solid"),
], 190, 66, gap=12)
arrow("OBSERVE — what the tools actually found", 46)

decide = node("decide", "gate", "DECIDE — risk_proxy vs. the configured band", [
    "max(tool confidence): below risk_low or above risk_high skips the judge",
], 600, 88, gate_label="DECIDE — THE RISK GATE")
deny_y2 = deny("decide", "outside the band — ALLOW or a deterministic action · 0 judge calls")
arrow("only the band between the two thresholds reaches here", 50)

node("decide-judge", "model", "Ask the judge", [
    "a recommendation only — action, risk_score, evidence, never the final word",
], 600, 80)
arrow(None, 40)

enforce = node("enforce", "gate", "ENFORCE — the deterministic floor", [
    "PolicyEngine: combines the recommendation with config/policy.yaml's floor",
    "can only raise the action, never lower it — RBAC checked here too",
], 600, 96, gate_label="ENFORCE")
arrow(None, 44)

trace = node("trace-out", "data", "TRACE — hash-chained, nothing fabricated", [
    "every phase, every tool call, judge_calls counted — the same audit log",
], 700, 84)
arrow(None, 44)

node("req-out", "terminal", "The final action, and the trace behind it", [
    "ALLOW · MASK · REDACT · BLOCK · FLAG · ESCALATE — one of six, always",
], 470, 74)

# the two skip-the-judge rails, both landing in ENFORCE — one right-margin
# rail carries both, the same pattern home_diagram.py uses for its four
# refusal gates converging on the audit box.
e = anchors["enforce"]
enforce_mid = e["top"] + e["h"] / 2
out.append(f'<path class="deny" d="M{DENY_X} {deny_y1:.0f} L{DENY_X} {enforce_mid:.0f} '
           f'L{e["right"] + 10:.0f} {enforce_mid:.0f}" marker-end="url(#arrow-deny)"/>')
out.append(f'<path class="deny" d="M{DENY_X} {deny_y2:.0f} L{DENY_X} {deny_y1:.0f}"/>')


OUT = Path(__file__).resolve().parent.parent / "web" / "_supervisor_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="The Guardrail Supervisor, PLAN through TRACE">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
