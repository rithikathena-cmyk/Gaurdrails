/* Conversations — yours, or anyone's if you hold `traces`.
 *
 * Two panes: a list of conversations on the left, the selected transcript on
 * the right. An operator gets a person picker above the list; a citizen does
 * not, because for them there is nothing to switch to — and the server would
 * refuse anyway, so the absence is honest rather than decorative.
 */

import { api } from "./api.js";
import { $, $$, esc } from "./dom.js";
import { renderMarkdown } from "./markdown.js";
import { showTrace } from "./trace.js";

let snapshot = null;
let whose = "";
let openSession = null;

export const historyLoaded = () => snapshot !== null;

const fmt = (n) => Number(n || 0).toLocaleString();
const money = (v) => (Number(v) < 0.01 && Number(v) > 0
  ? `$${Number(v).toFixed(4)}` : `$${Number(v || 0).toFixed(2)}`);

function when(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { day: "numeric", month: "short" }) +
      " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── list ── */
function renderPeople() {
  const box = $("#h-people");
  const people = snapshot.people || [];
  box.hidden = people.length === 0;
  if (!people.length) return;
  box.innerHTML = people.map((p) => `
    <button class="h-person${p.name === whose ? " on" : ""}" data-who="${esc(p.name)}">
      <b>${esc(p.display)}</b>
      <span>${fmt(p.turns)} turn${p.turns === 1 ? "" : "s"}${p.blocked ? ` · ${p.blocked} refused` : ""}</span>
    </button>`).join("");
}

function renderList() {
  const s = snapshot.stats || {};
  $("#h-turns").textContent = fmt(s.turns);
  $("#h-sessions").textContent = fmt(s.sessions);
  $("#h-blocked").textContent = fmt(s.blocked);
  $("#h-masked").textContent = fmt(s.masked);
  $("#h-cost").textContent = money(s.cost_usd);

  $("#h-scope-note").textContent = snapshot.people.length
    ? "You hold the traces permission, so you can read anyone's."
    : "You are reading your own, and only your own.";

  $("#h-whose").textContent = snapshot.mine
    ? "Your conversations"
    : `${snapshot.whose.display} · ${snapshot.whose.role_label}`;

  const list = snapshot.sessions || [];
  $("#h-empty").hidden = list.length > 0;
  $("#h-list").innerHTML = list.map((g) => `
    <button class="h-item${g.session_id === openSession ? " on" : ""}"
            data-session="${esc(g.session_id)}">
      <div class="h-item-top">
        <span class="h-when">${when(g.last_at)}</span>
        ${g.blocked ? `<span class="h-chip block">${g.blocked} refused</span>` : ""}
        ${g.modes.includes("agent") ? `<span class="h-chip">agent</span>` : ""}
      </div>
      <p class="h-open">${esc(g.opened_with)}</p>
      <span class="h-meta">${g.turns} turn${g.turns === 1 ? "" : "s"} · ${fmt(g.tokens)} tokens · ${money(g.cost_usd)}</span>
    </button>`).join("");
}

/* ── transcript ── */
function renderTurns(payload) {
  const head = $("#h-detail-head");
  // Title the transcript with what was actually asked. `session_id` is a
  // machine handle the browser minted (`web-64uvm9`) — it identifies the
  // conversation to the server but tells a reader nothing about which one this
  // is, and the list already titles every entry this way. Kept as metadata
  // because support requests still start from it.
  const opened = (payload.turns[0] && payload.turns[0].question) || payload.session_id;
  head.innerHTML = `
    <b title="${esc(opened)}">${esc(opened)}</b>
    <span>${payload.turns.length} turn${payload.turns.length === 1 ? "" : "s"} ·
      ${esc(payload.whose.display)} ·
      <code>${esc(payload.session_id)}</code></span>`;

  $("#h-turnlist").innerHTML = payload.turns.map((t) => `
    <article class="h-turn${t.blocked ? " refused" : ""}">
      <header>
        <span class="h-when">${when(t.at)}</span>
        <span class="h-chip ${esc(t.verdict)}">${esc(t.verdict)}</span>
        ${t.mode === "agent" ? `<span class="h-chip">agent</span>` : ""}
        ${t.masked ? `<span class="h-chip mask">${t.masked} masked</span>` : ""}
        <span class="h-meta">${fmt(t.tokens)} tokens · ${money(t.cost_usd)}${t.model ? ` · ${esc(t.model)}` : ""}</span>
        ${t.request_id ? `<button class="h-trace" data-request="${esc(t.request_id)}"
             title="Open the full trace for this turn">trace ↗</button>` : ""}
      </header>
      <div class="h-q">${esc(t.question)}</div>
      <div class="h-a">${t.blocked
        ? `<span class="h-refusal">${esc(t.refusal_reason || "refused")}</span>${renderMarkdown(t.reply || "")}`
        : renderMarkdown(t.reply || "")}</div>
    </article>`).join("");
  $("#h-detail").hidden = false;
  $("#h-detail-empty").hidden = true;
}

async function openConversation(sessionId) {
  openSession = sessionId;
  renderList();
  try {
    const payload = await api.historySession(snapshot.whose.name, sessionId);
    renderTurns(payload);
  } catch (err) {
    $("#h-detail").hidden = true;
    $("#h-detail-empty").hidden = false;
    $("#h-detail-empty").textContent = err.message || "Could not open that conversation.";
  }
}

/* ── data ── */
export async function loadHistory(who = "") {
  try {
    snapshot = await api.history(who);
    whose = snapshot.whose.name;
    openSession = null;
    renderPeople();
    renderList();
    $("#h-detail").hidden = true;
    $("#h-detail-empty").hidden = false;
    $("#h-detail-empty").textContent = snapshot.sessions.length
      ? "Pick a conversation to read it."
      : "Nothing here yet.";
  } catch (err) {
    $("#h-empty").hidden = false;
    $("#h-empty").textContent = err.message || "Could not load conversations.";
  }
}

/* ── wiring ── */
export function initHistory() {
  $("#h-people").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-who]");
    if (btn) loadHistory(btn.dataset.who);
  });

  $("#h-list").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-session]");
    if (btn) openConversation(btn.dataset.session);
  });

  // A turn links to its own trace, which is the point of keeping the id.
  $("#h-turnlist").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-request]");
    if (!btn) return;
    showTrace(btn.dataset.request);
  });

  $("#h-refresh").addEventListener("click", () => loadHistory(whose));
}
