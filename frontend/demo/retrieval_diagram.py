"""Generate the "Inside search_documents" flowchart embedded in web/home.html.

Same tool as `home_diagram.py`/`supervisor_diagram.py`/`agent_supervisor_diagram.py`,
same vocabulary (`node`/`arrow`/`fanout`/`deny`), copied rather than imported so
each diagram stays a single, runnable file.

`home_diagram.py`'s top-level pipeline draws `search_documents` as one box. This
unpacks that box — what `agent/tools.py`'s `_search_documents()` and
`knowledge/ingest.py`'s `search_with_rerank()` actually do to turn a query into
retrieved chunks — and then closes the loop: those chunks are a tool result, not
a final answer, so the diagram keeps going into `agent/runner.py`'s `_loop()`
far enough to show the model actually reading them and producing the reply,
before handing off to Output rails / Grounding / Egress (Stage 5 in the
top-level diagram — not redrawn here, since grounding checks the reply against
these exact chunks and already has its own box there).

Deliberately NOT drawn here, because the code does not do it: no query
rewrite/expand/classify step, no vector search, no metadata/RBAC filter inside
retrieval (authorization is its own separate, later stage — see gate 4 in the
top-level diagram), and no cross-encoder. Retrieval is BM25 lexical search,
optionally reordered by a local sentence-embedding model over BM25's own
candidates — nothing else. Adding stages this diagram doesn't have would make
it prettier and wrong.

Three rails use the plain "edge" class, not "deny": skipping the embedding
rerank (engine defaults to plain "bm25"), finding zero hits, and the reranker
model not being warm yet are normal, expected outcomes here, not a guardrail
refusal — the red "deny" styling is reserved for the other diagrams' actual
BLOCK/ESCALATE paths, and reusing it here would misrepresent an empty result
or a config default as a security event.

Run it, then paste the emitted SVG into web/home.html, under a new
"Inside search_documents — how a chunk actually gets retrieved" figure.
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


def deny(from_node, label):
    a = anchors[from_node]
    mid = a["top"] + a["h"] / 2
    out.append(f'<path class="deny" d="M{a["right"]:.0f} {mid:.0f} L{DENY_X} {mid:.0f}"/>')
    out.append(f'<text class="deny-label" x="{a["right"] + 16:.0f}" y="{mid - 10:.0f}">{esc(label)}</text>')
    return mid


def skip(from_node, label):
    """Like `deny`, but the plain "edge" class — a normal, expected alternate
    path (reranking off by default, zero hits), not a guardrail refusal.
    """
    a = anchors[from_node]
    mid = a["top"] + a["h"] / 2
    out.append(f'<path class="edge" d="M{a["right"]:.0f} {mid:.0f} L{DENY_X} {mid:.0f}"/>')
    out.append(f'<text class="edge-label" x="{a["right"] + 16:.0f}" y="{mid - 10:.0f}">{esc(label)}</text>')
    return mid


# ══════════════════════════════════════════════════════════════════
# Inside search_documents — BM25 first, an optional local reranker second,
# nothing else. Three "skip" rails (all normal outcomes, not refusals) merge
# into one right-margin lane and rejoin at the final chunk list, the same
# multi-origin-merge pattern the other three diagrams use for their own
# deny rails — here drawn in the neutral edge colour instead of deny's red.
#
# Past the chunk list, the diagram keeps going one more hop, into
# `_loop()`'s next `Agent step` — the chunks are a tool result, appended to
# `messages`, and it is THAT next `llm.converse()` call that actually reads
# them and drafts the reply. Everything after the model stops calling tools
# (Output rails, Grounding, Egress) is Stage 5 on the top-level diagram, not
# redrawn here.
# ══════════════════════════════════════════════════════════════════
node("req-in", "terminal", "The agent calls search_documents", [
    "one of three tools — agent/tools.py's _search_documents()",
], 600, 74)
arrow("query, plus ctx.k=4 and ctx.min_score=0.15 — ToolContext defaults", 50)

bm25 = node("bm25", "solid", "BM25 search — Corpus.search()", [
    "lexical term-matching only, no semantic layer — K1=1.5, B=0.75",
    "coverage ≥ ingest.min_chunk_score (0.15) · ≥2 matched terms if query >3 words",
], 600, 92)
skip_y1 = skip("bm25", "below the gate — 0 hits; the tool says what's missing, not a block")
arrow("candidates BM25 already thinks are plausible", 44)

engine = node("engine-toggle", "solid", "retrieval.engine — is a reranker even configured?", [
    '"bm25" (default): today\'s exact behaviour, lexical order kept',
    '"bm25+embedding": pool = max(k, embedding_candidates=12) candidates go on',
], 600, 92)
skip_y2 = skip("engine-toggle", 'default "bm25" — order kept unchanged, 0 extra model calls')
arrow('only when retrieval.engine = "bm25+embedding"', 44)

rerank = node("rerank", "model", "Embedding rerank — a local MiniLM model", [
    "sentence-transformers/all-MiniLM-L6-v2 · cosine similarity, loaded lazily",
    "reorders the candidate pool, keeps the top k — additive, never a second index",
], 600, 92)
skip_y3 = skip("rerank", "not warm yet → None — BM25 order kept, retrieval is never worse")
arrow(None, 44)

final_chunks = node("final-chunks", "solid", "Top k chunks — the same k Corpus.search() used", [
    "k=4 by default · each hit appended to ctx.chunks as this turn's context",
], 600, 80)
arrow('formatted "[i] title — text", one line per hit', 44)

node("tool-result", "solid", "Chunks return as this tool's result", [
    "joins the step's tool_result messages — agent/runner.py's _loop() keeps going",
], 620, 74)
arrow("the same llm.converse() call, tool results now in messages", 46)

node("llm-answer", "model", "Agent step N+1 — the model reads the chunks, drafts a reply", [
    "grounded in what search_documents actually returned, nothing assumed",
    "may call another tool instead — the loop keeps going until it doesn't",
], 640, 92)
arrow("once the model stops calling tools — turn.wants_tools is False", 46)

node("response-out", "terminal", "Reply moves to Output rails → Grounding → Egress", [
    "Stage 5 on the top-level diagram — grounding checks every claim against these chunks",
], 660, 74)

# the three skip rails, all normal outcomes, merged into one lane and back
# into the spine at the final chunk list.
fc = anchors["final-chunks"]
fc_mid = fc["top"] + fc["h"] / 2
out.append(f'<path class="edge" d="M{DENY_X} {skip_y1:.0f} L{DENY_X} {skip_y2:.0f}"/>')
out.append(f'<path class="edge" d="M{DENY_X} {skip_y2:.0f} L{DENY_X} {skip_y3:.0f}"/>')
out.append(f'<path class="edge" d="M{DENY_X} {skip_y3:.0f} L{DENY_X} {fc_mid:.0f} '
           f'L{fc["right"] + 10:.0f} {fc_mid:.0f}" marker-end="url(#arrow)"/>')


OUT = Path(__file__).resolve().parent.parent / "web" / "_retrieval_diagram.svg.part"

HEIGHT = y + 40
svg = "\n".join(out)

OUT.write_text(
    f'<svg viewBox="0 0 {W} {HEIGHT:.0f}" class="flow" role="img" '
    f'aria-label="Inside search_documents, BM25 through the model reading the chunks">\n{svg}\n</svg>',
    encoding="utf-8",
)
print(f"diagram generated: {W} x {HEIGHT:.0f}, {len(anchors)} anchored nodes -> {OUT}")
