"""Generate the simplified pipeline flowchart embedded in web/home.html.

Same tool as `home_diagram.py`/`supervisor_diagram.py`, same vocabulary
(`node`/`arrow`/`fanout`/`deny`), copied rather than imported so each diagram
stays a single, runnable file.

Redraws the top-level pipeline around the shape actually asked for: one
Supervisor Agent that plans/routes/delegates/decides, a bank of specialist
agents it delegates to, then tools/RAG/the model, then a second Output
Supervisor Agent that checks the reply before it ever reaches the user.

Grounded in the same code as every other diagram on this page — nothing
here is invented:

    SUPERVISOR AGENT   Supervisor.run() — agents/supervisor.py. PLAN (one
                        judge call, which specialist agents are relevant) ->
                        SELECT/EXECUTE -> OBSERVE -> DECIDE -> POLICY -> ACT.
    MULTI-AGENTS        the six specialist agents Supervisor can delegate
                        to — pii_agent.py, injection_agent.py, scope_agent.py,
                        authorization_agent.py, content_safety_agent.py,
                        grounding_agent.py. All six genuinely run their own
                        PLAN + DECIDE, calling the judge from their own
                        `_plan()`/`_decide()` — grounding_agent.py's own
                        docstring says so explicitly ("same shape as every
                        other agent"). Two have a real short-circuit that
                        looks like a flat check from a distance: Authorization
                        skips a tool call when the request is plainly about
                        the caller's own data, and Grounding skips PLAN
                        entirely when there are no retrieved chunks to check —
                        neither is a sign they lack their own agent loop.
    TOOLS / RAG          agent/tools.py's three tools; search_documents is
                        the RAG one — BM25, optional local rerank
                        (knowledge/ingest.py's search_with_rerank()).
    LLM                  the next Agent step in agent/runner.py's _loop() —
                        reads tool/RAG results, drafts the reply.
    OUTPUT SUPERVISOR    engine.py's evaluate() on Surface.LLM_RESPONSE,
    AGENT                plus grounding_rail.evaluate() checking every claim
                        against the retrieved chunks ("citations" below).

Run it, then paste the emitted SVG into web/home.html, over the existing
pipeline `<svg class="flow">` block.
"""

from pathlib import Path

W = 1400
CX = 640
GAP = 46

out: list[str] = []
y = 54
anchors: dict[str, dict] = {}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node(nid, kind, title, lines, w, h, cx=CX, gate_label=None, tag=None):
    global y
    x = cx - w / 2
    out.append(f'<g class="node {kind}" data-node="{nid}">')
    out.append(f'  <rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="11"/>')
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
    global y
    y2 = y + length
    out.append(f'<path class="edge" d="M{cx} {y:.0f} L{cx} {y2 - 9:.0f}" marker-end="url(#arrow)"/>')
    if label:
        out.append(f'<text class="edge-label" x="{cx + 14}" y="{y + length / 2 + 4:.0f}">{esc(label)}</text>')
    y = y2


def fanout(nid, items, w, h, gap=14, agent_tag=False):
    """`items` entries are `(title, sub, kind)` or `(title, sub, kind, sub2)`
    — `sub2`, when given and non-empty, renders as a third line at a fixed
    16px step below `sub` rather than a fraction of `h`, so a taller box
    (passed via `h`) never has to guess where the extra line lands; the
    caller is responsible for passing an `h` tall enough for however many
    lines the tallest item in this fanout actually has (2 lines fit in the
    same `h=64` this diagram always used before; 3 lines need `h=80`+)."""
    global y
    total = len(items) * w + (len(items) - 1) * gap
    x0 = CX - total / 2
    bus_y = y + 20
    out.append(f'<path class="edge" d="M{CX} {y:.0f} L{CX} {bus_y:.0f}"/>')
    out.append(f'<path class="edge" d="M{x0 + w / 2:.0f} {bus_y:.0f} L{x0 + total - w / 2:.0f} {bus_y:.0f}"/>')
    top = bus_y + 26
    for i, item in enumerate(items):
        title, sub, kind = item[0], item[1], item[2]
        sub2 = item[3] if len(item) > 3 else ""
        cx = x0 + i * (w + gap) + w / 2
        out.append(f'<path class="edge" d="M{cx:.0f} {bus_y:.0f} L{cx:.0f} {top - 9:.0f}" marker-end="url(#arrow)"/>')
        out.append(f'<g class="node {kind}">')
        out.append(f'  <rect x="{cx - w / 2:.0f}" y="{top:.0f}" width="{w}" height="{h}" rx="10"/>')
        if agent_tag:
            out.append(f'  <text class="agent-tag small" x="{cx + w / 2 - 10:.0f}" y="{top + 14:.0f}">AGENT</text>')
        out.append(f'  <text class="title small" x="{cx:.0f}" y="{top + 26:.0f}">{esc(title)}</text>')
        out.append(f'  <text class="mono small" x="{cx:.0f}" y="{top + 42:.0f}">{esc(sub)}</text>')
        if sub2:
            out.append(f'  <text class="mono small nested" x="{cx:.0f}" y="{top + 58:.0f}">{esc(sub2)}</text>')
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


# ══════════════════════════════════════════════════════════════════
node("user-in", "terminal", "Citizen or operator", [
    "asks a question — USER",
], 440, 70)
arrow(None, 40)

node("input", "solid", "INPUT — normalize", [
    "NFKC + homoglyph fold, locked — the same first step every surface uses",
], 600, 66)
arrow("handed to the Supervisor Agent", 44)

sup = node("supervisor-agent", "gate", "Plan / Route / Delegate / Decide", [
    "PLAN — one judge call: which specialist agents are relevant — agents/supervisor.py",
    "DECIDE — one agent's own decision carries alone; more than one specialist,",
    "a second judge call weighs them",
], 600, 113, gate_label="SUPERVISOR AGENT", tag=("AGENTIC", "agentic"))
arrow("delegates to whichever specialists PLAN actually named", 46)

fanout("multi-agents", [
    ("PII", "own PLAN + DECIDE", "solid", "+ nested NER agent"),
    ("Security", "own PLAN + DECIDE", "solid", "+ nested DeBERTa agent"),
    ("Scope", "own PLAN + DECIDE", "solid"),
    ("Authorization", "own PLAN + DECIDE", "solid"),
    ("Content Safety", "own PLAN + DECIDE", "solid", "+ nested Toxic-BERT agent"),
    ("Grounding", "own PLAN + DECIDE", "solid", "+ nested NLI agent"),
], 180, 90, gap=12, agent_tag=True)
arrow("PolicyEngine combines every verdict — floor can only rise, never fall", 48)

node("tools", "solid", "TOOLS", [
    "agent/tools.py — search_documents · lookup_fee · case/grievance",
], 600, 66)
arrow("search_documents is the retrieval tool", 40)

node("rag", "solid", "RAG", [
    "BM25 lexical search, optional local rerank — knowledge/ingest.py",
], 600, 66)
arrow("candidate chunks join this turn's context", 44)

node("llm", "model", "Generate Response", [
    "the next Agent step reads tool results + RAG chunks, drafts the reply",
], 600, 80, gate_label="LLM")
arrow("the drafted reply, before it reaches anyone", 46)

out_sup = node("output-supervisor", "gate", "Plan / Check the reply", [
    "PII · Safety · Policy · Grounding · Citations — every claim against",
    "what was actually retrieved, engine.py's evaluate() + grounding_rail",
], 600, 96, gate_label="OUTPUT SUPERVISOR AGENT", tag=("AGENTIC", "agentic"))
arrow(None, 40)

# a real two-way split, side by side — REJECT and ALLOW are equally live
# outcomes here, not a main path with a rail escaping off the margin.
branch_y = y
bus_y = y + 20
LX, RX = CX - 220, CX + 220
out.append(f'<path class="edge" d="M{CX} {branch_y:.0f} L{CX} {bus_y:.0f}"/>')
out.append(f'<path class="edge" d="M{LX} {bus_y:.0f} L{RX} {bus_y:.0f}"/>')
out.append(f'<path class="deny" d="M{LX:.0f} {bus_y:.0f} L{LX:.0f} {bus_y + 27:.0f}" marker-end="url(#arrow-deny)"/>')
out.append(f'<path class="edge" d="M{RX:.0f} {bus_y:.0f} L{RX:.0f} {bus_y + 27:.0f}" marker-end="url(#arrow)"/>')
out.append(f'<text class="deny-label" x="{LX:.0f}" y="{bus_y - 8:.0f}" text-anchor="middle">REJECT</text>')
out.append(f'<text class="edge-label" x="{RX:.0f}" y="{bus_y - 8:.0f}" text-anchor="middle">ALLOW</text>')

box_top = bus_y + 27
out.append('<g class="node solid" data-node="human">')
out.append(f'  <rect x="{LX - 180:.0f}" y="{box_top:.0f}" width="360" height="66" rx="10"/>')
out.append(f'  <text class="title" x="{LX:.0f}" y="{box_top + 29:.0f}">HUMAN</text>')
out.append(f'  <text class="mono" x="{LX:.0f}" y="{box_top + 50:.0f}">a person reviews — never reaches the user unmoderated</text>')
out.append('</g>')

out.append('<g class="node terminal" data-node="response">')
out.append(f'  <rect x="{RX - 180:.0f}" y="{box_top:.0f}" width="360" height="66" rx="10"/>')
out.append(f'  <text class="title" x="{RX:.0f}" y="{box_top + 29:.0f}">RESPONSE</text>')
out.append(f'  <text class="mono" x="{RX:.0f}" y="{box_top + 50:.0f}">only what both supervisors let through</text>')
out.append('</g>')

merge_y = box_top + 66 + 30
out.append(f'<path class="edge" d="M{RX:.0f} {box_top + 66:.0f} L{RX:.0f} {merge_y - 9:.0f} '
            f'L{CX} {merge_y - 9:.0f}" marker-end="url(#arrow)"/>')
y = merge_y

node("user-out", "terminal", "USER", [
    "receives the response",
], 300, 60)

OUT = Path(__file__).resolve().parent.parent / "web" / "_pipeline_v2_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="The guardrail pipeline: Supervisor Agent, multi-agents, tools, RAG, LLM, Output Supervisor Agent">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
