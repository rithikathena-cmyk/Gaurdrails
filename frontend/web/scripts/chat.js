/** Chat view. */

import { api } from "./api.js";
import { $, $$, esc } from "./dom.js";
import { renderMarkdown } from "./markdown.js";
import { addTrace, showTrace } from "./trace.js";


/* Badge glyphs. The sample declares which one it wants, so a new sample is a
   server edit rather than a frontend edit. */
const SAMPLE_ICON = {
  check:  'M2.8 8.4 6 11.6l7.2-7.2',
  shield: 'M8 1.8 13.6 4.2v4.3c0 3.2-2.3 6-5.6 6.8-3.3-.8-5.6-3.6-5.6-6.8V4.2L8 1.8Z',
  alert:  'M8 2.4 14.6 13.6H1.4L8 2.4Zm0 4.2v3.1m0 2h.01',
  search: 'M7.2 11.6a4.4 4.4 0 1 0 0-8.8 4.4 4.4 0 0 0 0 8.8Zm3.3.1 3 3',
  tool:   'M2.6 4.6h10.8M2.6 8h10.8M2.6 11.4h6.4',
  key:    'M9.6 6.4a2.8 2.8 0 1 0-2.7 2.8L6 10.1v1.4H4.6v1.6H2.4v-2.2l4.2-4.2a2.8 2.8 0 0 0 3-0.3Z',
  pen:    'M3 13h2.6l6.2-6.2-2.6-2.6L3 10.4V13Zm8-9.6 1.6 1.6',
};
const badge = (name) => {
  const path = SAMPLE_ICON[name] || SAMPLE_ICON.check;
  return `<span class="badge"><svg width="15" height="15" viewBox="0 0 16 16" fill="none"
    aria-hidden="true"><path d="${path}" stroke="currentColor" stroke-width="1.5"
    stroke-linecap="round" stroke-linejoin="round"/></svg></span>`;
};

const SESSION = "web-" + Math.random().toString(36).slice(2, 8);
let busy = false;
// One flow. The agent is the pipeline; there is nothing to switch to.
let agentSamples = [];

export async function initChat() {
  const input = $("#input");
  const form = $("#composer");

  const autogrow = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  };
  input.addEventListener("input", autogrow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });

  $("#new-chat").addEventListener("click", async () => {
    await api.resetSession(SESSION);
    $("#messages").innerHTML = "";
    $("#greeting").classList.remove("hide");
    input.value = "";
    autogrow();
    document.dispatchEvent(new CustomEvent("nav", { detail: "chat" }));
  });


  await loadSamples();
  renderSamples();
  autogrow();
  input.focus();
}

/** The prompts offered under an empty composer. They come from the agent's own
    tool list, so a suggestion can never name a tool the config has disabled. */
async function loadSamples() {
  try {
    agentSamples = (await api.agentTools()).samples || [];
  } catch { agentSamples = []; }
}

function renderSamples() {
  const samples = agentSamples;
  $("#suggestions").innerHTML = samples.map((s) => `
    <button class="suggestion" data-fill="${esc(s.text)}">
      ${badge(s.icon)}
      <span class="suggestion-body">
        <b>${esc(s.title)}</b><span>${esc(s.blurb)}</span>
      </span>
    </button>`).join("");
  $$(".suggestion").forEach((b) =>
    b.addEventListener("click", () => {
      const input = $("#input");
      input.value = b.dataset.fill;
      input.dispatchEvent(new Event("input"));
      input.focus();
    }));
}

async function send() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text || busy) return;

  $("#greeting").classList.add("hide");
  addUser(text);
  input.value = "";
  input.dispatchEvent(new Event("input"));
  setBusy(true);

  const pending = document.createElement("div");
  pending.className = "turn assistant";
  pending.innerHTML = `
    <div class="turn-meta"><span class="who">agent</span></div>
    <div class="thinking"><span class="pulse"></span> ${
      "planning, calling tools…"}</div>`;
  $("#messages").appendChild(pending);
  scroll();

  try {
    const data = await api.agentChat(text, SESSION);
    pending.remove();
    addAssistant(data);
    addTrace(data.trace);
  } catch (err) {
    pending.remove();
    addError(err.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

const IDLE_HINT = "Tools run behind rails · writes ask first";

function setBusy(on) {
  busy = on;
  $("#send").disabled = on;
  $("#composer-hint").textContent = on
    ? "planning…"
    : IDLE_HINT;
}

function addUser(text) {
  const node = document.createElement("div");
  node.className = "turn user";
  node.innerHTML = `<div>
      <div class="turn-meta"><span class="who">you</span></div>
      <div class="bubble">${esc(text)}</div></div>`;
  $("#messages").appendChild(node);
  scroll();
}

function addAssistant(data) {
  const t = data.trace;
  const chips = [`<span class="chip ${data.verdict}">${data.verdict}</span>`];

  // Every surface, not just the prompt. The chip counted `stage === "prompt"`
  // while the violations panel reported the response, so one turn showed
  // "2 values masked" beside a panel saying 4 — both true, neither reconcilable
  // by the reader. The chip is now the turn's total and the panel breaks it
  // down by where each came from, so the numbers add up.
  const masked = (data.detections || []).filter(
    (d) => !["blocked_term", "unsupported_claim"].includes(d.kind));
  if (masked.length) {
    chips.push(`<span class="chip mask">${masked.length} value${masked.length > 1 ? "s" : ""} masked</span>`);
  }
  if (t.regenerations) {
    chips.push(`<span class="chip flag">${t.regenerations} regeneration${t.regenerations > 1 ? "s" : ""}</span>`);
  }
  chips.push(`<span class="chip mute">${Math.round(t.total_ms)}ms · ${Math.round(t.guardrail_ms)}ms rails</span>`);

  const node = document.createElement("div");
  node.className = "turn assistant";
  node.innerHTML = `
    <div class="turn-meta">
      <span class="who">${data.calls || data.approval ? "agent" : "assistant"}</span>${chips.join("")}
    </div>
    ${violations(data)}
    ${approvalCard(data)}
    ${data.reply
      ? `<div class="body md${data.blocked ? " refused" : ""}">${renderMarkdown(data.reply)}</div>`
      : ""}
    <div class="turn-tools">
      <button class="link-btn">View trace</button>
      <span class="eyebrow">${t.rails_evaluated} rails · ${esc(t.request_id)}</span>
    </div>`;
  node.querySelector(".link-btn").addEventListener("click", () => showTrace(t.request_id));
  if (data.approval) wireApproval(node, data.approval.token);
  $("#messages").appendChild(node);
  scroll();
}

// The per-tool-call block that used to render here — name, arguments, and the
// `args`/`data` verdicts — has been removed from the chat turn. It is working
// detail, not an answer, and on a two-step run it pushed the reply below the
// fold. Nothing is lost: every tool call, both surface verdicts, and the
// arguments are still in the trace behind "View trace", and anything a rail
// actually acted on still appears above the reply via `violations()`.

/** A write tool is waiting on a person. Say what will happen, in words. */
function approvalCard(data) {
  if (!data.approval) return "";
  const a = data.approval;
  return `<div class="approval" data-token="${esc(a.token)}">
    <div class="approval-head">
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="M8 1.6 13.4 4v4.2c0 3.2-2.3 5.9-5.4 6.7-3.1-.8-5.4-3.5-5.4-6.7V4L8 1.6Z"
              stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>
      </svg>
      <b>This needs your approval</b>
    </div>
    <p class="approval-what">${esc(a.summary)}</p>
    <p class="approval-why">${esc(a.why)} — nothing has been filed yet.</p>
    <div class="approval-row">
      <button class="btn small" data-approve>Approve and continue</button>
      <button class="btn ghost small" data-decline>Decline</button>
    </div>
  </div>`;
}

function wireApproval(node, token) {
  const card = node.querySelector(".approval");
  const decide = async (approved) => {
    card.querySelectorAll("button").forEach((b) => { b.disabled = true; });
    card.insertAdjacentHTML("beforeend",
      `<div class="thinking"><span class="pulse"></span> ${
        approved ? "approved — finishing…" : "declining…"}</div>`);
    try {
      const data = await api.approve(token, approved, SESSION);
      card.remove();
      addAssistant(data);
      addTrace(data.trace);
    } catch (err) {
      addError(err.message);
    }
  };
  card.querySelector("[data-approve]").addEventListener("click", () => decide(true));
  card.querySelector("[data-decline]").addEventListener("click", () => decide(false));
}

const FAMILY_ICON = {
  pii:       "M8 1.6 13.4 4v4.2c0 3.2-2.3 5.9-5.4 6.7-3.1-.8-5.4-3.5-5.4-6.7V4L8 1.6Z",
  words:     "M3 4h10M3 8h10M3 12h6",
  content:   "M8 2.2 14.4 13H1.6L8 2.2Zm0 4.3v3.1M8 11.2h.01",
  injection: "M8 1.6 13.4 4v4.2c0 3.2-2.3 5.9-5.4 6.7-3.1-.8-5.4-3.5-5.4-6.7V4L8 1.6Z",
  policy:    "M4 2h8v12l-4-2.4L4 14V2Z",
  grounding: "M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0Zm5.5-3v3.3l2.2 1.3",
};

/** Tell the user what tripped, at the disclosure level the server chose. */
function violations(data) {
  const list = data.violations || [];
  if (!list.length) return "";
  return `<div class="violations">${list.map((v) => `
    <div class="violation ${esc(v.verdict)}">
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="${FAMILY_ICON[v.family] || FAMILY_ICON.content}"
              stroke="currentColor" stroke-width="1.4"
              stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div class="violation-body">
        <b>${esc(v.title)}</b>
        <p>${esc(v.detail)}</p>
        ${v.items && v.items.length
          ? `<div class="violation-items">${v.items.map((i) =>
              `<span class="vitem">${esc(i)}</span>`).join("")}</div>`
          : ""}
      </div>
      <span class="chip ${esc(v.verdict)}">${esc(v.verdict)}</span>
    </div>`).join("")}</div>`;
}

function addError(message) {
  const node = document.createElement("div");
  node.className = "turn assistant";
  node.innerHTML = `
    <div class="turn-meta"><span class="who">system</span>
      <span class="chip block">error</span></div>
    <div class="body refused">${esc(message)}</div>`;
  $("#messages").appendChild(node);
  scroll();
}

function scroll() {
  const c = $("#conversation");
  c.scrollTop = c.scrollHeight;
}
