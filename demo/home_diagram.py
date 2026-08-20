"""Generate the pipeline flowchart embedded in web/home.html.

Run it, then paste the emitted SVG over the existing `<svg class="flow">`
block in web/home.html. Kept as a script rather than hand-authored markup:

coordinates are computed rather than hand-placed, so every node declares its
height, the cursor walks down, and edges are drawn between what the cursor
recorded. Hand-authoring 25 absolutely-positioned SVG boxes is how diagrams
drift out of alignment the first time one label grows.

Colours are class names, never literals, so the same SVG serves both themes.
"""

from pathlib import Path

W = 1400          # viewBox width
CX = 640          # spine centre
DENY_X = 1268     # the red refusal rail runs down here
GAP = 46          # vertical gap between nodes

out: list[str] = []
y = 54
anchors: dict[str, dict] = {}


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def node(nid, kind, title, lines, w, h, cx=CX, gate_label=None):
    """One box. `kind` picks the class; gates are left-aligned like the reference."""
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
    """A refusal leaving a gate for the rail on the right."""
    a = anchors[from_node]
    mid = a["top"] + a["h"] / 2
    out.append(f'<path class="deny" d="M{a["right"]:.0f} {mid:.0f} L{DENY_X} {mid:.0f}"/>')
    out.append(f'<text class="deny-label" x="{a["right"] + 16:.0f}" y="{mid - 10:.0f}">{esc(label)}</text>')
    return mid


# ══════════════════════════════════════════════════════════════════
# the pipeline
#
# Nine steps, one line of explanation each. An earlier draft had twenty boxes
# and two lines apiece: more accurate, and harder to walk somebody through.
# What was cut is still in the trace and in /demo/stages — this is the version
# you can say out loud.
# ══════════════════════════════════════════════════════════════════
node("user-in", "terminal", "Someone asks a question", [
    "signed in — a citizen sees less than an operator",
], 440, 74)
arrow("the question, and who is asking", 48)

gate1 = node("gate1", "gate", "Is the question allowed through?", [
    "clean up the text first, then seven checks at once",
], 600, 88, gate_label="GATE 1 — THE QUESTION")
deny_y1 = deny("gate1", "REFUSED — the model is never called")

# Seven now, not five. Dashed means the check is a model with nothing
# deterministic underneath it; `injection` and `scope` are solid because each
# has a free layer that settles most traffic before any model is asked.
fanout("rails", [
    ("Banned words", "words", "solid"),
    ("Personal details", "pii", "solid"),
    ("Names, addresses", "entities", "model"),
    ("House rules", "policy", "solid"),
    ("Attacks", "injection", "solid"),
    ("Off-topic", "scope", "solid"),
    ("Harmful content", "content", "model"),
], 158, 66, gap=12)

arrow("seven opinions, one decision", 44)

node("gate2", "gate", "The strictest answer wins", [
    "block beats mask beats flag beats pass",
], 600, 84, gate_label="GATE 2 — THE DECISION")
deny_y2 = deny("gate2", "REFUSED — written down, then explained")
arrow("unless it was a close call", 44)

# The only box on this page that usually does not run. It is drawn on the spine
# rather than off to one side because when it does run, it decides.
node("adjudicate", "model", "Was it too close to call?", [
    "only when a score sits near its line — otherwise skipped",
], 600, 80)
arrow("personal details are now tokens", 46)

retrieval = node("retrieval", "solid", "Look up the facts", [
    "search the documents, and check what comes back",
    "the office's own address stays readable; yours does not",
], 600, 96)
arrow("the question, plus what was found", 46)

node("generate", "model", "Ask the model", [
    "it only sees the tokens, and only the facts found above",
], 600, 80)
deny_y3 = deny("generate", "MODEL DECLINED")
arrow("a draft nobody has seen yet", 44)

node("gate3", "gate", "Is the answer allowed out?", [
    "the same checks again, on the way back",
], 600, 84, gate_label="GATE 3 — THE ANSWER")
deny("gate3", "REFUSED — the draft is never shown")
arrow(None, 40)

gate4 = node("gate4", "gate", "Does it match the sources?", [
    "every claim checked against what was found",
], 600, 84, gate_label="GATE 4 — IS IT TRUE")
deny_y4 = deny("gate4", "SENT TO A PERSON")
arrow(None, 44)

audit = node("audit", "data", "Put the real values back, and write it all down", [
    "the reader gets their own data; the log cannot be edited",
], 700, 84)
arrow(None, 44)

node("user-out", "terminal", "The answer, and the trace behind it", [
    "every check, its verdict, and how long it took",
], 470, 74)

# the refusal rail: down the right margin, into the audit log
a = anchors["audit"]
audit_mid = a["top"] + a["h"] / 2
out.append(f'<path class="deny" d="M{DENY_X} {deny_y1:.0f} L{DENY_X} {audit_mid:.0f} '
           f'L{a["right"] + 10:.0f} {audit_mid:.0f}" marker-end="url(#arrow-deny)"/>')

# the regeneration loop, down the left margin back into generation
g = anchors["generate"]
LOOP_X = 208   # far enough left that its label clears the spine boxes
out.append(f'<path class="loop" d="M{anchors["gate4"]["left"]:.0f} '
           f'{anchors["gate4"]["top"] + anchors["gate4"]["h"] / 2:.0f} '
           f'L{LOOP_X} {anchors["gate4"]["top"] + anchors["gate4"]["h"] / 2:.0f} '
           f'L{LOOP_X} {g["top"] + g["h"] / 2:.0f} L{g["left"] - 10:.0f} {g["top"] + g["h"] / 2:.0f}" '
           f'marker-end="url(#arrow-loop)"/>')
out.append(f'<text class="loop-label" x="{LOOP_X + 12}" '
           f'y="{(g["top"] + anchors["gate4"]["top"]) / 2:.0f}">regenerate ≤ 2×</text>')

# ── the ingestion feed, left of retrieval ────────────────────────
# Aligned on retrieval's centre line so the join is one straight arrow, and
# far enough left that the quarantine stub has room. The first cut ran the
# stub off the left edge of the viewBox, which clipped its label.
r = anchors["retrieval"]
r_mid = r["top"] + r["h"] / 2
IX = 170
GATE_W, GATE_H = 250, 62
gate_y = r_mid - GATE_H / 2
doc_y = gate_y - 58 - 30

out.append('<g class="node solid">')
out.append(f'  <rect x="{IX - 100}" y="{doc_y:.0f}" width="200" height="58" rx="10"/>')
out.append(f'  <text class="title small" x="{IX}" y="{doc_y + 25:.0f}">A document arrives</text>')
out.append(f'  <text class="mono" x="{IX}" y="{doc_y + 43:.0f}">uploaded, pasted, or scanned</text>')
out.append("</g>")
out.append(f'<path class="edge" d="M{IX} {doc_y + 58:.0f} L{IX} {gate_y - 9:.0f}" '
           f'marker-end="url(#arrow)"/>')

out.append('<g class="node gate">')
out.append(f'  <rect x="{IX - GATE_W / 2:.0f}" y="{gate_y:.0f}" width="{GATE_W}" '
           f'height="{GATE_H}" rx="10"/>')
out.append(f'  <text class="gate-label" x="{IX - GATE_W / 2 + 18:.0f}" '
           f'y="{gate_y + 23:.0f}">GATE — A DOCUMENT</text>')
out.append(f'  <text class="mono left" x="{IX - GATE_W / 2 + 18:.0f}" '
           f'y="{gate_y + 45:.0f}">checked before it is stored</text>')
out.append("</g>")

# quarantine leaves downward, so its label never reaches the canvas edge
out.append(f'<path class="deny" d="M{IX} {gate_y + GATE_H:.0f} L{IX} {gate_y + GATE_H + 24:.0f}"/>')
out.append(f'<text class="deny-label mid" x="{IX}" y="{gate_y + GATE_H + 40:.0f}">quarantined — no search finds it</text>')

# one straight run into retrieval
out.append(f'<path class="edge" d="M{IX + GATE_W / 2:.0f} {r_mid:.0f} '
           f'L{r["left"] - 9:.0f} {r_mid:.0f}" marker-end="url(#arrow)"/>')


# Relative to this file, so the script runs from any working directory
# and on any machine.
OUT = Path(__file__).resolve().parent.parent / "web" / "_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="The guardrail pipeline, gate by gate">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
