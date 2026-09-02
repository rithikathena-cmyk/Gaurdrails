/** Tiny DOM helpers. Shared by every view. */

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const html = (strings, ...values) =>
  strings.reduce((out, s, i) => out + s + (i < values.length ? values[i] : ""), "");

/** Build an element from an HTML string. */
export function el(markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  return t.content.firstElementChild;
}

export function chip(kind, text) {
  return `<span class="chip ${kind}">${esc(text)}</span>`;
}

/** Format a config value for display. */
export function fmt(v) {
  if (Array.isArray(v)) return v.length ? v.join(", ") : "—";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (v === null || v === undefined || v === "") return "—";
  return String(v);
}

/** A duration for a human to read. Judge calls run 1-100+s; showing that in
 *  milliseconds (`"126872ms"`) is unreadable. Sub-second rails (most
 *  deterministic ones) stay in ms, where a decimal second would be the
 *  awkward choice instead. */
export function fmtMs(ms) {
  const n = Number(ms) || 0;
  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${Math.round(n)}ms`;
}

export function debounce(fn, ms = 350) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Transient status message, bottom-centre. */
let toastTimer;
export function toast(message, kind = "ok") {
  let node = $("#toast");
  if (!node) {
    node = el(`<div id="toast" class="toast" role="status" aria-live="polite"></div>`);
    document.body.appendChild(node);
  }
  node.className = `toast show ${kind}`;
  node.textContent = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2600);
}
