/**
 * Parameters view.
 *
 * Rendered entirely from /api/parameters — families, surfaces, severity levels,
 * lock metadata, and which control each parameter needs all come from the
 * Python registry. Nothing about the control surface is hardcoded here.
 *
 * Edits are batched and PATCHed. That is not a runtime override:
 * `policy.runtime_override` stays locked. The server validates the change set,
 * writes config/overrides.yaml, records it, and reloads the engine.
 */

import { api } from "./api.js";
import { $, $$, debounce, el, esc, fmt, toast } from "./dom.js";

const state = {
  data: null,
  filter: "all",
  search: "",
  pendingValues: {},
  pendingMatrix: {},
  saving: false,
};

const token = (name) => `var(${name})`;
const lockMeta = (key) => state.data.locks[key];

/* ─────────────────────────── loading ─────────────────────────── */
export async function loadParams() {
  try {
    state.data = await api.parameters();
    render();
  } catch (err) {
    $("#params-body").innerHTML =
      `<div class="card"><div class="empty">Could not load parameters: ${esc(err.message)}</div></div>`;
  }
}

function applySnapshot(snapshot) {
  state.data = { ...state.data, ...snapshot };
  render();
}

/* ─────────────────────────── saving ─────────────────────────── */
const flush = debounce(async () => {
  const values = state.pendingValues;
  const matrix = state.pendingMatrix;
  if (!Object.keys(values).length && !Object.keys(matrix).length) return;

  state.pendingValues = {};
  state.pendingMatrix = {};
  state.saving = true;
  setSaveState("saving");

  try {
    const res = await api.patchParameters(values, matrix);
    applySnapshot(res.snapshot);
    setSaveState("saved");
    const n = res.changes.length;
    toast(n === 1
      ? `${res.changes[0].key} → ${fmt(res.changes[0].to)}`
      : `${n} parameters updated · engine reloaded`);
  } catch (err) {
    setSaveState("error");
    toast(err.message, "err");
    // Re-render from the server's truth so the UI never shows a value that
    // was rejected.
    loadParams();
  } finally {
    state.saving = false;
  }
}, 450);

function queueValue(key, value) {
  state.pendingValues[key] = value;
  setSaveState("pending");
  flush();
}

function queueMatrix(family, surface, level) {
  (state.pendingMatrix[family] ||= {})[surface] = level;
  setSaveState("pending");
  flush();
}

function setSaveState(kind) {
  const node = $("#save-state");
  if (!node) return;
  const label = {
    pending: "unsaved…",
    saving: "saving…",
    saved: "saved · engine reloaded",
    error: "rejected",
    idle: "",
  }[kind];
  node.textContent = label;
  node.className = `save-state ${kind}`;
}

/* ─────────────────────────── render ─────────────────────────── */
function render() {
  const d = state.data;
  if (!d) return;

  renderToolbar(d);
  renderLockKey(d);
  renderMatrix(d);
  renderFamilies(d);
  applyFilter();
  wire();
}

function renderToolbar(d) {
  const overrides = (d.overridden?.length || 0) + (d.matrix_overridden?.length || 0);
  $("#p-meta").innerHTML = `
    <span class="chip mute">${d.total} parameters</span>
    <span class="chip accent">${d.total_adjustable} adjustable</span>
    <span class="chip mute">${d.total_locked} fixed</span>
    ${overrides
      ? `<span class="chip mask">${overrides} changed from baseline</span>`
      : `<span class="chip pass">matching baseline</span>`}
    <span class="save-state" id="save-state"></span>`;
  $("#p-reset").disabled = overrides === 0;
}

function renderLockKey(d) {
  $("#lock-key").innerHTML = `
    <div style="--lk:${token("--accent")}">
      <b>◆ Adjustable</b>
      <p>Editable here. The change is written to <code>overrides.yaml</code>, recorded,
         and the engine reloads.</p>
    </div>
    ${Object.entries(d.locks).map(([key, m]) => `
      <div style="--lk:${token(m.token)}">
        <b>${m.glyph} ${esc(m.label)}</b>
        <p>${esc(m.blurb)}</p>
      </div>`).join("")}`;
}

/* ── severity matrix ── */
function renderMatrix(d) {
  const surfaces = d.surfaces || [];
  const levels = d.severity_levels || [];
  const matrix = d.matrix || {};
  const changed = new Set(d.matrix_overridden || []);

  const rows = Object.entries(matrix).map(([family, row]) => {
    const famName = d.families.find((f) => f.key === family)?.name || family;
    return `
      <tr>
        <td><span class="m-fam">${esc(famName)}</span><small>${esc(family)}</small></td>
        ${surfaces.map((s) => {
          const level = row[s.key] || "medium";
          const isChanged = changed.has(`${family}.${s.key}`);
          return `<td><button class="cell ${level}${isChanged ? " changed" : ""}"
                      data-family="${family}" data-surface="${s.key}" data-level="${level}"
                      title="Click to cycle · ${esc(s.blurb)}">${level}</button></td>`;
        }).join("")}
      </tr>`;
  }).join("");

  $("#matrix-block").innerHTML = !rows ? "" : `
    <section class="card fam" id="matrix-card" data-section="matrix">
      <div class="fam-head">
        <h3>Severity matrix</h3>
        <span class="src">guardrail family × surface — click any cell to cycle</span>
        <span class="counts"><span class="chip accent">drives every threshold below</span></span>
      </div>
      <div class="scroll-x">
        <table class="matrix">
          <thead><tr><th>family</th>${surfaces.map((s) =>
            `<th title="${esc(s.blurb)}">${esc(s.label)}</th>`).join("")}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="legend">
        ${levels.map((l) => `
          <span><span class="cell ${l.key}" style="pointer-events:none">${l.key}</span>
          ${esc(l.blurb)}</span>`).join("")}
      </div>
    </section>`;
}

/* ── families ── */
function renderFamilies(d) {
  $("#params-body").innerHTML = d.families.map((f) => `
    <section class="card fam" data-section="${f.key}">
      <div class="fam-head">
        <h3>${esc(f.name)}</h3>
        <span class="src">${esc(f.engine)}</span>
        <span class="counts">
          <span class="chip accent">${f.adjustable} adjustable</span>
          <span class="chip mute">${f.locked} fixed</span>
        </span>
      </div>
      <div class="scroll-x">
        <table class="ptable">
          <thead>
            <tr><th>Parameter</th><th>Type</th><th>Value</th><th>Why it's fixed</th></tr>
          </thead>
          <tbody>${f.params.map((p) => row(p, d)).join("")}</tbody>
        </table>
      </div>
    </section>`).join("");
}

function row(p, d) {
  const hay = `${p.key} ${p.desc}`.toLowerCase();
  if (p.adjustable) {
    const value = d.current?.[p.key] ?? p.default;
    const baseline = d.baseline?.[p.key] ?? p.default;
    const changed = JSON.stringify(value) !== JSON.stringify(baseline);
    return `
      <tr class="adj" data-kind="adj" data-hay="${esc(hay)}" data-key="${esc(p.key)}">
        <td class="pkey">${esc(p.key)}${changed ? ` <span class="dot-changed" title="changed from baseline"></span>` : ""}
            <small>${esc(p.desc)}</small></td>
        <td class="ptype">${esc(p.type)}</td>
        <td class="pctrl">${control(p, value)}</td>
        <td class="why tune">
          ${changed
            ? `<button class="revert" data-revert="${esc(p.key)}"
                 title="Revert to baseline">↺ baseline ${esc(fmt(baseline))}</button>`
            : `◆ tune per deployment`}
        </td>
      </tr>`;
  }
  const m = lockMeta(p.lock);
  return `
    <tr class="locked" data-kind="locked" data-hay="${esc(hay)}" style="--lk:${token(m.token)}">
      <td class="pkey">${esc(p.key)}<small>${esc(p.desc)}</small></td>
      <td class="ptype">${esc(p.type)}</td>
      <td>
        <span class="fixed-val">${LOCK_SVG}<span>${esc(p.value)}</span></span>
        <span class="lock-tag" style="color:${token(m.token)};
              background:color-mix(in srgb, ${token(m.token)} 12%, transparent)">
          ${m.glyph} ${esc(m.label)}</span>
      </td>
      <td class="why">${esc(p.why)}</td>
    </tr>`;
}

/* ─────────────────────────── controls ─────────────────────────── */
const LOCK_SVG = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
  <rect x="2" y="5.2" width="8" height="6" rx="1.2" stroke="currentColor" stroke-width="1.2"/>
  <path d="M4 5.2V3.6a2 2 0 1 1 4 0v1.6" stroke="currentColor" stroke-width="1.2"/></svg>`;

function control(p, value) {
  const k = esc(p.key);
  switch (p.control) {
    case "range": {
      const dec = p.step && p.step < 1 ? 2 : 0;
      return `
        <div class="ctrl">
          <input type="range" data-input="${k}" min="${p.min}" max="${p.max}"
                 step="${p.step ?? 1}" value="${value}" aria-label="${k}">
          <output data-out="${k}">${Number(value).toFixed(dec)}</output>
        </div>
        <small class="ctrl-hint">range ${p.min}–${p.max}</small>`;
    }
    case "number":
      return `<div class="ctrl">
                <input type="number" class="num-input" data-input="${k}"
                       value="${value}" aria-label="${k}"></div>`;
    case "toggle":
      return `<div class="ctrl">
                <button class="sw" role="switch" data-input="${k}"
                        aria-checked="${!!value}" aria-label="${k}"></button>
                <span class="swl" data-out="${k}">${value ? "on" : "off"}</span>
              </div>`;
    case "select":
      return `<div class="ctrl">
                <select data-input="${k}" aria-label="${k}">
                  ${(p.options || []).map((o) =>
                    `<option${o === value ? " selected" : ""}>${esc(o)}</option>`).join("")}
                </select></div>`;
    case "tags": {
      const items = Array.isArray(value) ? value : [];
      return `
        <div class="tags" data-tags="${k}">
          ${items.map((t, i) => `
            <span class="tag">${esc(t)}
              <button class="tag-x" data-remove="${k}" data-index="${i}"
                      aria-label="Remove ${esc(t)}">×</button></span>`).join("")}
          <input class="tag-add" data-add="${k}" placeholder="add…"
                 aria-label="Add to ${k}" size="8">
        </div>
        <small class="ctrl-hint">${items.length} item${items.length === 1 ? "" : "s"} · Enter to add</small>`;
    }
    case "matrix":
      return `<span class="pval dim">edited in the matrix above</span>`;
    default:
      return `<div class="ctrl">
                <input type="text" class="text-input" data-input="${k}"
                       value="${esc(fmt(value))}" aria-label="${k}"></div>`;
  }
}

/* ─────────────────────────── wiring ─────────────────────────── */
function wire() {
  // sliders / numbers / text
  $$('#params-body input[type="range"], #params-body .num-input, #params-body .text-input')
    .forEach((input) => {
      const key = input.dataset.input;
      const out = $(`#params-body [data-out="${CSS.escape(key)}"]`);
      input.addEventListener("input", () => {
        if (out) {
          const dec = Number(input.step) && Number(input.step) < 1 ? 2 : 0;
          out.textContent = Number(input.value).toFixed(dec);
        }
      });
      input.addEventListener("change", () => {
        const raw = input.type === "range" || input.classList.contains("num-input")
          ? Number(input.value) : input.value;
        queueValue(key, raw);
      });
    });

  // toggles
  $$("#params-body .sw").forEach((sw) => {
    sw.addEventListener("click", () => {
      const on = sw.getAttribute("aria-checked") !== "true";
      sw.setAttribute("aria-checked", String(on));
      const out = $(`#params-body [data-out="${CSS.escape(sw.dataset.input)}"]`);
      if (out) out.textContent = on ? "on" : "off";
      queueValue(sw.dataset.input, on);
    });
  });

  // selects
  $$("#params-body select").forEach((sel) => {
    sel.addEventListener("change", () => queueValue(sel.dataset.input, sel.value));
  });

  // tag add / remove
  $$("#params-body .tag-add").forEach((input) => {
    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      const key = input.dataset.add;
      queueValue(key, [...(state.data.current[key] || []), text]);
      input.value = "";
    });
  });
  $$("#params-body .tag-x").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.remove;
      const next = [...(state.data.current[key] || [])];
      next.splice(Number(btn.dataset.index), 1);
      queueValue(key, next);
    });
  });

  // revert one parameter to baseline
  $$("#params-body .revert").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.revert;
      queueValue(key, state.data.baseline[key]);
    });
  });

  // matrix cells cycle through the levels the server declared
  const levels = (state.data.severity_levels || []).map((l) => l.key);
  $$("#matrix-block .cell[data-family]").forEach((cell) => {
    cell.addEventListener("click", () => {
      const next = levels[(levels.indexOf(cell.dataset.level) + 1) % levels.length];
      cell.dataset.level = next;
      cell.className = `cell ${next} changed`;
      cell.textContent = next;
      queueMatrix(cell.dataset.family, cell.dataset.surface, next);
    });
  });
}

/* ─────────────────────────── filter ─────────────────────────── */
export function applyFilter() {
  const f = state.filter;
  const q = state.search.trim().toLowerCase();

  const rows = $$("#params-body tbody tr");
  rows.forEach((r) => {
    const kindOk = f === "all" || r.dataset.kind === f;
    const searchOk = !q || r.dataset.hay.includes(q);
    r.classList.toggle("hide", !(kindOk && searchOk));
  });

  // Collapse any family with nothing left to show.
  $$("#params-body .fam").forEach((sec) => {
    const any = $$("tbody tr", sec).some((r) => !r.classList.contains("hide"));
    sec.classList.toggle("hide", !any);
  });

  // The matrix is an adjustable control, so it has no place in the Fixed view,
  // and it isn't a parameter row so a search should hide it too.
  const matrixCard = $("#matrix-card");
  if (matrixCard) matrixCard.classList.toggle("hide", f === "locked" || Boolean(q));

  const shown = rows.filter((r) => !r.classList.contains("hide")).length;
  const total = rows.length;
  const note = $("#p-showing");
  if (note) {
    note.textContent = shown === total
      ? `showing all ${total}`
      : `showing ${shown} of ${total}`;
    note.classList.toggle("filtered", shown !== total);
  }
  const emptyState = $("#p-empty");
  if (emptyState) emptyState.classList.toggle("hide", shown > 0);
}

export function initParams() {
  $$("#p-filters [data-filter]").forEach((b) => {
    b.addEventListener("click", () => {
      state.filter = b.dataset.filter;
      $$("#p-filters [data-filter]").forEach((x) =>
        x.setAttribute("aria-pressed", String(x === b)));
      applyFilter();
    });
  });

  const search = $("#p-search");
  search.addEventListener("input", debounce(() => {
    state.search = search.value;
    applyFilter();
  }, 120));
  search.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { search.value = ""; state.search = ""; applyFilter(); }
  });

  $("#p-reset").addEventListener("click", async () => {
    if (!confirm("Drop every override and return to the checked-in baseline?")) return;
    try {
      const res = await api.resetParameters();
      applySnapshot(res.snapshot);
      toast(`${res.reverted} parameter${res.reverted === 1 ? "" : "s"} reverted to baseline`);
    } catch (err) {
      toast(err.message, "err");
    }
  });
}

export const paramsLoaded = () => Boolean(state.data);
