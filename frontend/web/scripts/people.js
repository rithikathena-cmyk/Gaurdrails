/* People — who exists, what they may spend, and which model they get.
 *
 * The table is the point, so it carries state rather than describing it: a bar
 * for the budget, a chip when someone is over it. Everything here is a thin
 * skin over /api/users, which is where the rules actually live — this file
 * decides nothing.
 */

import { api } from "./api.js";
import { $, $$, esc } from "./dom.js";

let snapshot = null;

export const peopleLoaded = () => snapshot !== null;

const fmt = (n) => Number(n || 0).toLocaleString();

function say(kind, text) {
  const box = $("#u-msg");
  if (!box) return;
  box.className = `u-msg ${kind}`;
  box.textContent = text;
  box.hidden = !text;
}

/* ── rendering ── */
const money = (n) => {
  const v = Number(n || 0);
  if (v === 0) return "$0.00";
  // Sub-cent amounts are normal here; rounding them to $0.00 makes the column
  // look broken on a light day.
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
};

/* One row per window: the bar and the input that governs it, together. Split
   across two columns they were 40% wider and you had to look twice to see which
   number moved which bar. */
function windowRow(label, key, used, limit, name, step) {
  const capped = limit > 0;
  const pct = capped ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  const tone = capped && used >= limit ? "over" : pct >= 80 ? "near" : "ok";
  const bar = capped
    ? `<div class="u-bar ${tone}" role="img"
            aria-label="${label}: ${pct}% of ${limit} tokens used"><i style="width:${pct}%"></i></div>`
    : `<span class="u-nobar">no ceiling</span>`;
  return `
    <div class="u-win">
      <span class="u-win-name">${label}</span>
      ${bar}
      <span class="u-win-num">${fmt(used)}</span>
      <input class="u-lim" data-w="${key}" type="number" min="0" step="${step}"
             value="${limit}" data-name="${esc(name)}"
             aria-label="${label} token limit for ${esc(name)}">
    </div>`;
}

function budgetCell(u) {
  return [
    windowRow("day", "daily", u.day_tokens, u.daily_limit, u.name, 1000),
    windowRow("month", "monthly", u.month_tokens, u.monthly_limit, u.name, 10000),
    windowRow("total", "total", u.tokens_used, u.token_limit, u.name, 10000),
  ].join("");
}

/* The cost is a button. Collapsed it answers "how much"; open it answers
   "how much, over what, and against which ceiling" — which is three numbers
   per window and does not belong permanently in a table cell. */
function costCell(u) {
  const line = (label, used, limit, cost) => {
    const cap = limit > 0 ? `of ${fmt(limit)}` : `<span class="u-dim">no ceiling</span>`;
    const pct = limit > 0 ? ` · ${Math.min(100, Math.round((used / limit) * 100))}%` : "";
    return `<tr><th>${label}</th><td>${fmt(used)} ${cap}${pct}</td><td>${money(cost)}</td></tr>`;
  };
  return `
    <button class="u-cost-btn" data-cost="${esc(u.name)}" aria-expanded="false"
            title="Show the breakdown by window">
      <b class="u-cost">${money(u.cost_usd)}</b>
      <span class="u-sub">${money(u.day_cost_usd)} today ▾</span>
    </button>
    <div class="u-cost-detail" hidden>
      <table>
        <thead><tr><th></th><th>tokens</th><th>cost</th></tr></thead>
        <tbody>
          ${line("day", u.day_tokens, u.daily_limit, u.day_cost_usd)}
          ${line("month", u.month_tokens, u.monthly_limit, u.month_cost_usd)}
          ${line("total", u.tokens_used, u.token_limit, u.cost_usd)}
        </tbody>
      </table>
      <span class="u-sub">${esc(u.model_label)}${u.breached ? ` · over the ${esc(u.breached)} ceiling` : ""}</span>
    </div>`;
}

function row(u, models) {
  const opts = models.map((m) =>
    `<option value="${esc(m.key)}"${m.key === u.model ? " selected" : ""}>${esc(m.label)}</option>`
  ).join("");
  return `
  <tr data-user="${esc(u.name)}">
    <td>
      <div class="u-who">
        <span class="u-avatar${u.role === "admin" ? " admin" : ""}">${esc((u.display || u.name).slice(0, 2).toUpperCase())}</span>
        <span>
          <b>${esc(u.display || u.name)}</b>
          <span class="u-sub">${esc(u.name)}</span>
        </span>
      </div>
    </td>
    <td><span class="u-role${u.role === "admin" ? " admin" : ""}">${esc(u.role_label)}</span></td>
    <td>
      <select class="u-model" data-name="${esc(u.name)}" aria-label="Model for ${esc(u.name)}">
        ${opts}
      </select>
    </td>
    <td class="u-budget">${budgetCell(u)}</td>
    <td class="u-costcell">${costCell(u)}</td>
    <td>
      <div class="u-actions">
        <button class="u-btn" data-act="limit" data-name="${esc(u.name)}">Save</button>
        <button class="u-btn ghost" data-act="reset" data-name="${esc(u.name)}"
                title="Zero every counter">Reset</button>
        <button class="u-btn danger" data-act="delete" data-name="${esc(u.name)}">Remove</button>
      </div>
      ${u.active_sessions ? `<span class="u-sub">${u.active_sessions} signed in</span>` : ""}
    </td>
  </tr>`;
}

function render() {
  if (!snapshot) return;
  const { users, models } = snapshot;
  $("#u-total").textContent = snapshot.total;
  $("#u-admins").textContent = snapshot.by_role.admin ?? 0;
  $("#u-citizens").textContent = snapshot.by_role.user ?? 0;
  $("#u-spent").textContent = fmt(snapshot.tokens_spent);
  $("#u-cost").textContent = money(snapshot.cost_usd);
  $("#u-month-cost").textContent = money(snapshot.month_cost_usd);
  $("#u-capped").textContent = snapshot.capped;

  const rates = $("#u-rates");
  if (rates && snapshot.pricing) {
    rates.innerHTML = snapshot.pricing.map((r) =>
      `<span><b>${esc(r.model)}</b> $${r.input_per_mtok}/$${r.output_per_mtok} per Mtok</span>`
    ).join("");
  }

  const over = $("#u-over");
  over.hidden = snapshot.over_budget === 0;
  over.textContent = `${snapshot.over_budget} over budget`;

  $("#u-rows").innerHTML = users.map((u) => row(u, models)).join("");

  const sel = $("#nu-role");
  if (sel && !sel.dataset.filled) {
    sel.innerHTML = snapshot.roles
      .map((r) => `<option value="${esc(r.key)}">${esc(r.label)}</option>`).join("");
    sel.dataset.filled = "1";
  }
  const msel = $("#nu-model");
  if (msel && !msel.dataset.filled) {
    msel.innerHTML = models
      .map((m) => `<option value="${esc(m.key)}">${esc(m.label)}</option>`).join("");
    msel.dataset.filled = "1";
  }
}

/* ── data ── */
export async function loadPeople() {
  try {
    snapshot = await api.users();
    say("", "");
    render();
  } catch (err) {
    say("err", err.message || "Could not load the people list.");
  }
}

async function apply(fn, ok) {
  try {
    snapshot = await fn();
    render();
    say("ok", ok);
  } catch (err) {
    say("err", err.message || "That did not work.");
  }
}

function closeCostPanels() {
  $$(".u-cost-detail").forEach((d) => { d.hidden = true; });
  $$("[data-cost]").forEach((b) => b.setAttribute("aria-expanded", "false"));
}

/* ── wiring ── */
export function initPeople() {
  // A popover that only closes by clicking its own trigger is a popover people
  // leave open by accident.
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".u-costcell")) closeCostPanels();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeCostPanels();
  });

  $("#u-rows").addEventListener("click", (e) => {
    const cost = e.target.closest("button[data-cost]");
    if (cost) {
      const panel = cost.nextElementSibling;
      const open = panel.hidden;
      closeCostPanels();               // one open at a time, or they overlap
      panel.hidden = !open;
      cost.setAttribute("aria-expanded", String(open));
      return;
    }
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const name = btn.dataset.name;
    if (btn.dataset.act === "limit") {
      const row = btn.closest("tr");
      const read = (w) => {
        const el = row.querySelector(`.u-lim[data-w="${w}"]`);
        return Math.max(0, parseInt(el.value, 10) || 0);
      };
      const body = { daily_limit: read("daily"), monthly_limit: read("monthly"),
                     token_limit: read("total") };
      const set = Object.entries(body).filter(([, v]) => v > 0).length;
      apply(() => api.setUser(name, body),
            set === 0 ? `${name} has no ceiling on any window.`
                      : `${name}: ${set} ceiling${set > 1 ? "s" : ""} saved.`);
    }
    if (btn.dataset.act === "reset") {
      const w = btn.dataset.window || "all";
      apply(() => api.resetUsage(name, w),
            `${name}'s ${w === "all" ? "" : w + " "}usage is back to zero.`);
    }
    if (btn.dataset.act === "delete") {
      // A destructive action gets one deliberate confirmation, naming what goes.
      if (!confirm(`Remove ${name}? Their account and any open session go with it.`)) return;
      apply(() => api.deleteUser(name), `${name} removed.`);
    }
  });

  $("#u-rows").addEventListener("change", (e) => {
    const sel = e.target.closest("select.u-model");
    if (!sel) return;
    const label = sel.options[sel.selectedIndex].textContent;
    apply(() => api.setUser(sel.dataset.name, { model: sel.value }),
          `${sel.dataset.name} now uses ${label}.`);
  });

  $("#new-user").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      name: $("#nu-name").value.trim(),
      password: $("#nu-pass").value,
      display: $("#nu-display").value.trim(),
      role: $("#nu-role").value,
      model: $("#nu-model").value,
      token_limit: Math.max(0, parseInt($("#nu-limit").value, 10) || 0),
      daily_limit: Math.max(0, parseInt($("#nu-daily").value, 10) || 0),
      monthly_limit: Math.max(0, parseInt($("#nu-monthly").value, 10) || 0),
    };
    try {
      snapshot = await api.addUser(body);
      render();
      e.target.reset();
      say("ok", `${body.name} added.`);
    } catch (err) {
      say("err", err.message || "Could not add that person.");
    }
  });
}
