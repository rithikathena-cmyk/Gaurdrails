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
function budgetCell(u) {
  if (u.token_limit <= 0) {
    return `<span class="u-dim">no ceiling</span>
            <span class="u-sub">${fmt(u.tokens_used)} spent</span>`;
  }
  const pct = Math.min(100, Math.round((u.tokens_used / u.token_limit) * 100));
  const tone = u.over_budget ? "over" : pct >= 80 ? "near" : "ok";
  return `
    <div class="u-bar ${tone}" role="img"
         aria-label="${pct}% of budget used"><i style="width:${pct}%"></i></div>
    <span class="u-sub">${fmt(u.tokens_used)} / ${fmt(u.token_limit)} · ${pct}%</span>`;
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
    <td>
      <div class="u-actions">
        <input class="u-limit" type="number" min="0" step="1000" value="${u.token_limit}"
               data-name="${esc(u.name)}" aria-label="Token limit for ${esc(u.name)}">
        <button class="u-btn" data-act="limit" data-name="${esc(u.name)}">Set</button>
        <button class="u-btn ghost" data-act="reset" data-name="${esc(u.name)}"
                title="Set tokens used back to zero">Reset</button>
        <button class="u-btn danger" data-act="delete" data-name="${esc(u.name)}"
                title="Remove this account">Remove</button>
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
  $("#u-capped").textContent = snapshot.capped;

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

/* ── wiring ── */
export function initPeople() {
  $("#u-rows").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const name = btn.dataset.name;
    if (btn.dataset.act === "limit") {
      const input = $(`.u-limit[data-name="${CSS.escape(name)}"]`);
      const limit = Math.max(0, parseInt(input.value, 10) || 0);
      apply(() => api.setUser(name, { token_limit: limit }),
            limit === 0 ? `${name} has no ceiling.` : `${name} capped at ${fmt(limit)} tokens.`);
    }
    if (btn.dataset.act === "reset") {
      apply(() => api.resetUsage(name), `${name}'s usage is back to zero.`);
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
