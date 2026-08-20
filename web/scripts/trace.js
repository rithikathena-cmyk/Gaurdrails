/** Request trace view — stage waterfall and per-rail detail. */

import { $, $$, esc } from "./dom.js";

const state = { traces: [], active: null };

export function addTrace(trace) {
  state.traces.unshift(trace);
  state.traces = state.traces.slice(0, 20);
  state.active = trace.request_id;
  renderRecent();
}

export function showTrace(requestId) {
  state.active = requestId;
  document.dispatchEvent(new CustomEvent("nav", { detail: "trace" }));
}

export function renderRecent() {
  const box = $("#recent");
  if (!state.traces.length) {
    box.innerHTML = `<div class="recent-empty">Nothing yet.</div>`;
    return;
  }
  box.innerHTML = state.traces.map((t) => `
    <button class="recent-item" data-trace="${esc(t.request_id)}">
      <span class="chip ${t.verdict}" style="padding:1px 5px">${t.verdict}</span>
      <span class="rid">${esc(t.request_id.replace("req_", ""))}</span>
      <span class="ms">${Math.round(t.total_ms)}ms</span>
    </button>`).join("");
  $$(".recent-item", box).forEach((b) =>
    b.addEventListener("click", () => showTrace(b.dataset.trace)));
}

export function renderTrace() {
  const body = $("#trace-body");
  if (!state.traces.length) {
    body.innerHTML = `<div class="card"><div class="empty">
      Send a message and the full trace lands here — every rail, its score against its
      threshold, and what it cost.</div></div>`;
    return;
  }

  const t = state.traces.find((x) => x.request_id === state.active) || state.traces[0];
  const total = Math.max(t.total_ms, 1);
  const c = t.rail_count;

  body.innerHTML = `
    <div class="trace-picker">
      ${state.traces.map((x) => `
        <button class="trace-pill" data-t="${esc(x.request_id)}"
                aria-pressed="${x.request_id === t.request_id}">
          ${esc(x.request_id.replace("req_", ""))} · ${x.verdict}
        </button>`).join("")}
    </div>

    <div class="stat-row">
      ${stat("Wall clock", Math.round(t.total_ms).toLocaleString(), "ms",
             "send → response delivered")}
      ${stat("Guardrail overhead", Math.round(t.guardrail_ms).toLocaleString(), "ms",
             `${t.guardrail_pct}% of wall clock`, "var(--accent)")}
      ${stat("Rails evaluated", t.rails_evaluated, "",
             `${c.pass} pass · ${c.mask} mask · ${c.flag} flag · ${c.block} block`)}
      ${stat("Final verdict", t.verdict.toUpperCase(), "",
             t.regenerations ? `${t.regenerations} regeneration(s)` : "delivered first attempt",
             `var(--${t.verdict})`, true)}
    </div>

    <div class="card">
      <div class="card-head"><h3>Stage waterfall</h3>
        <div class="right"><span class="eyebrow">click a stage to expand</span></div></div>
      <div>${t.stages.map((s, i) => stage(s, i, total)).join("")}</div>
      <div class="legend">
        <span><i style="background:var(--accent)"></i> guardrail evaluation</span>
        <span><i style="background:var(--ink-3)"></i> model / retrieval</span>
        <span><i style="background:var(--mask)"></i> retry caused by a rail</span>
      </div>
    </div>

    <div class="note">
      <b>Reading the overhead.</b> Rails inside a stage run concurrently, so a stage costs
      as much as its slowest rail, not the sum — the cheap rails hide entirely behind the
      model-backed ones they run alongside.
      ${t.fail_mode_triggered ? `<br><br><b style="color:var(--block)">The latency budget
        was exceeded on this request.</b> Unevaluated rails failed closed, which is locked
        behaviour — an unevaluated request is not a safe request.` : ""}
    </div>`;

  $$(".trace-pill", body).forEach((b) =>
    b.addEventListener("click", () => { state.active = b.dataset.t; renderTrace(); }));
  $$(".wf-head", body).forEach((h) =>
    h.addEventListener("click", () => {
      const open = h.parentElement.classList.toggle("open");
      h.setAttribute("aria-expanded", String(open));
    }));
}

function stat(label, value, unit, sub, color = "", small = false) {
  return `<div class="card stat">
    <span class="eyebrow">${esc(label)}</span>
    <b style="${color ? `color:${color};` : ""}${small ? "font-size:19px;" : ""}">${esc(value)}${
      unit ? `<small>${unit}</small>` : ""}</b>
    <span>${esc(sub)}</span></div>`;
}

function stage(s, i, total) {
  const left = (s.start_ms / total) * 100;
  const width = Math.max(0.6, (s.duration_ms / total) * 100);
  return `
    <div class="wf-stage">
      <button class="wf-head" aria-expanded="false">
        <span class="wf-idx num">${String(i + 1).padStart(2, "0")}</span>
        <span class="wf-name">${esc(s.name)}<small>${esc(s.subtitle)}</small></span>
        <span class="wf-track"><i class="${s.kind}"
              style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"></i></span>
        <span class="wf-ms">${s.duration_ms.toFixed(1)}ms</span>
        <span class="wf-vd"><span class="chip ${s.verdict}">${s.verdict}</span></span>
      </button>
      <div class="wf-detail">
        ${s.rails.map(rail).join("") ||
          `<div class="rail-why" style="padding:8px 0">No rails in this stage.</div>`}
        ${s.notes.map((n) => `<div class="stage-note">${esc(n)}</div>`).join("")}
      </div>
    </div>`;
}

function rail(r) {
  const isCount = r.unit === "count";
  const value = isCount ? r.score.toFixed(0) : r.score.toFixed(2);
  const threshold = isCount
    ? `min ${r.threshold.toFixed(0)}`
    : `thr ${r.higher_is_better ? "≥" : "<"} ${r.threshold.toFixed(2)}`;
  const pct = isCount ? Math.min(100, r.score * 50) : Math.min(100, r.score * 100);

  return `
    <div class="rail-row">
      <span class="rail-name">${esc(r.rail)}<small>${esc(r.engine)}</small>
        <span class="meter"><i style="width:${pct}%;background:var(--${r.verdict})"></i></span>
      </span>
      <span class="rail-score">${value}<small>${threshold}</small></span>
      <span class="rail-why">${why(r)}</span>
      <span class="rail-ms">${r.duration_ms.toFixed(1)}ms</span>
      <span class="rail-vd"><span class="chip ${r.verdict}">${r.verdict}</span></span>
    </div>`;
}

function why(r) {
  if (r.error) return `<span class="err">${esc(r.error)}</span>`;
  if (r.meta?.rationale) return esc(r.meta.rationale);
  if (r.meta?.skipped) return esc(r.meta.skipped);
  if (r.detections?.length) {
    const kinds = r.detections.slice(0, 4).map((d) => esc(d.kind)).join(", ");
    return kinds + (r.detections.length > 4 ? ` +${r.detections.length - 4}` : "");
  }
  if (r.meta?.breached?.length) return "breached: " + esc(r.meta.breached.join(", "));
  if (r.meta && Object.keys(r.meta).length) {
    return esc(Object.entries(r.meta).slice(0, 2)
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v).slice(0, 40) : v}`)
      .join(" · "));
  }
  return "";
}
