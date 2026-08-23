/** Documents view — ingestion, and what the ingest rails found.

    The report after an upload is the point of this screen. An ingest that just
    says "done" teaches the operator nothing about what their corpus now holds;
    this one says what was found, what was masked before writing, and whether
    the document was indexed or quarantined. */

import { api } from "./api.js";
import { $, $$, esc } from "./dom.js";
import { addTrace, showTrace } from "./trace.js";

let loaded = false;

/** How the text was obtained. Worth showing: a transcribed scan is a document a
    model has already read once, and the operator should know which ones those are. */
const METHOD = {
  paste: "pasted",
  text: "text file",
  sheet: "spreadsheet parsed",
  "pdf.text": "PDF text layer",
  "pdf.ocr": "scan transcribed",
  "pdf.mixed": "part scan, transcribed",
  "image.ocr": "image transcribed",
};

export function docsLoaded() {
  return loaded;
}

export function initDocs() {
  $("#d-reset").addEventListener("click", resetCorpus);
  $("#d-file").addEventListener("change", uploadFile);
}

export async function loadDocs() {
  try {
    const body = await api.documents();
    loaded = true;
    renderStats(body.stats);
    renderList(body.documents);
  } catch (err) {
    $("#d-list").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
}

/* ── header numbers ── */
function renderStats(s) {
  const cells = [
    ["indexed", s.indexed, "searchable right now"],
    ["chunks", s.chunks, "what retrieval ranks over"],
    ["uploaded", s.uploaded, "beyond the built-in corpus"],
    ["quarantined", s.quarantined, "stored, never returned"],
    ["masked", s.masked_values, "values removed before writing"],
  ];
  $("#d-stats").innerHTML = cells.map(([label, value, hint]) => `
    <div class="d-stat${label === "quarantined" && value ? " warn" : ""}" title="${esc(hint)}">
      <b class="num">${value}</b><span>${label}</span>
    </div>`).join("");
  $("#d-count").textContent =
    `${s.documents} document${s.documents === 1 ? "" : "s"} · ${s.chunks} indexed chunks`;
}

/* ── the list ── */
function renderList(docs) {
  if (!docs.length) {
    $("#d-list").innerHTML = `<div class="empty">Nothing ingested yet.</div>`;
    return;
  }
  $("#d-list").innerHTML = docs.map((d) => `
    <div class="d-item${d.status === "quarantined" ? " quarantined" : ""}" data-id="${esc(d.id)}">
      <div class="d-item-main">
        <div class="d-item-title">
          ${esc(d.title)}
          ${d.built_in ? `<span class="chip mute">built-in</span>` : ""}
          ${d.status === "quarantined"
            ? `<span class="chip block">quarantined</span>`
            : `<span class="chip pass">indexed</span>`}
          ${d.masked ? `<span class="chip mask">${d.masked} masked</span>` : ""}
        </div>
        <div class="d-item-sub">
          <span class="mono">${esc(d.source)}</span>
          <span class="how">${esc(METHOD[d.method] || d.method || "text")}</span>
          <span>${d.n_chunks} chunk${d.n_chunks === 1 ? "" : "s"}</span>
          <span>${d.chars.toLocaleString()} chars</span>
          ${d.reason ? `<span class="why">${esc(d.reason)}</span>` : ""}
        </div>
      </div>
      <div class="d-item-tools">
        ${d.request_id ? `<button class="link-btn" data-trace="${esc(d.request_id)}">trace</button>` : ""}
        <button class="link-btn" data-peek="${esc(d.id)}">chunks</button>
        <button class="link-btn danger" data-del="${esc(d.id)}">delete</button>
      </div>
    </div>`).join("");

  $$("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    try {
      await api.deleteDocument(b.dataset.del);
      await loadDocs();
    } catch (err) {
      note(err.message, true);
      b.disabled = false;
    }
  }));
  $$("[data-trace]").forEach((b) =>
    b.addEventListener("click", () => showTrace(b.dataset.trace)));
  $$("[data-peek]").forEach((b) => b.addEventListener("click", () => peek(b.dataset.peek)));
}

/** What is actually stored — the honest way to show that masking happened
    before indexing rather than at read time. */
async function peek(id) {
  const row = $(`.d-item[data-id="${CSS.escape(id)}"]`);
  const open = row.querySelector(".d-chunks");
  if (open) { open.remove(); return; }
  try {
    const { document: doc } = await api.document(id);
    const node = document.createElement("div");
    node.className = "d-chunks";
    node.innerHTML = (doc.chunks || []).map((c, i) => `
      <div class="d-chunk"><span class="n">${i + 1}</span><p>${esc(c)}</p></div>`).join("")
      || `<div class="empty">No chunks — nothing was indexed.</div>`;
    row.appendChild(node);
  } catch (err) {
    note(err.message, true);
  }
}

/* ── ingesting ── */
function note(text, isError = false) {
  const el = $("#d-note");
  el.textContent = text;
  el.className = "d-note" + (isError ? " err" : "");
}

async function uploadFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const title = $("#d-title").value.trim();
  await run(() => api.upload(file, title));
  event.target.value = "";          // so the same file can be re-chosen
}

async function run(call) {
  note("running the ingest rails…");
  try {
    const body = await call();
    report(body);
    addTrace(body.trace);
    $("#d-title").value = "";
    note("");
    await loadDocs();
  } catch (err) {
    note(err.message, true);
  }
}

/** The ingest report: what happened, in the order it happened. */
function report(body) {
  const d = body.document;
  const kinds = [...new Set((body.detections || []).map((x) => x.kind))];
  const stages = (body.trace.stages || []).map((s) => `
    <div class="d-stage">
      <span class="chip ${s.verdict}">${s.verdict}</span>
      <b>${esc(s.name)}</b>
      <span class="ms num">${Math.round(s.duration_ms)}ms</span>
    </div>`).join("");

  $("#d-report").innerHTML = `
    <section class="card d-result ${body.quarantined ? "bad" : "good"}">
      <div class="card-head">
        <h3>${body.quarantined ? "Quarantined" : "Indexed"} — ${esc(d.title)}</h3>
        <span class="eyebrow mono">${esc(d.request_id)}</span>
      </div>
      <div class="d-body">
        <p class="d-how">Read as: <b>${esc(METHOD[d.method] || d.method)}</b>${
          d.method && d.method.includes("ocr")
            ? " — a model transcribed this page before the rails saw it, so what it "
              + "returned was treated as an untrusted document like any other."
            : ""}</p>
        <p class="d-verdict">
          ${body.quarantined
            ? `This document failed an ingest rail (<b>${esc(body.reason)}</b>). It is stored
               for review and indexed nowhere — no query can return it.`
            : `${d.n_chunks} chunk${d.n_chunks === 1 ? "" : "s"} written to the index.` +
              (d.masked
                ? ` <b>${d.masked} value${d.masked === 1 ? "" : "s"} masked before writing</b> —
                    the index never held them.`
                : " Nothing needed masking.")}
        </p>
        ${kinds.length ? `<div class="d-kinds">${kinds.map((k) =>
          `<span class="chip mask">${esc(k)}</span>`).join("")}</div>` : ""}
        <div class="d-stages">${stages}</div>
        <div class="turn-tools">
          <button class="link-btn" data-open-trace="${esc(d.request_id)}">View the full trace</button>
        </div>
      </div>
    </section>`;
  $("[data-open-trace]").addEventListener("click", (e) =>
    showTrace(e.target.dataset.openTrace));
}

async function resetCorpus() {
  if (!confirm("Drop every uploaded document and restore all twenty-five built-in ones, "
               + "including any you have deleted?")) return;
  $("#d-reset").disabled = true;
  try {
    await api.resetDocuments();
    $("#d-report").innerHTML = "";
    await loadDocs();
  } catch (err) {
    note(err.message, true);
  } finally {
    $("#d-reset").disabled = false;
  }
}
