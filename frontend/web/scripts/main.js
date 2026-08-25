/** Boot and navigation. */

import { api } from "./api.js";
import { $, $$ } from "./dom.js";
import { initChat } from "./chat.js";
import { initDocs, loadDocs, docsLoaded } from "./docs.js";
import { initParams, loadParams, paramsLoaded } from "./params.js";
import { renderRecent, renderTrace } from "./trace.js";
import { initPeople, loadPeople, peopleLoaded } from "./people.js";
import { initHistory, loadHistory, historyLoaded, refreshSidebarChats } from "./history.js";

/* ── theme ── */
const savedTheme = localStorage.getItem("gc-theme");
if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);

$("#theme").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const isDark = current
    ? current === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = isDark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("gc-theme", next);
});

function setSidebar(collapsed) {
  $(".shell").classList.toggle("collapsed", collapsed);
  localStorage.setItem("gc-sidebar", collapsed ? "collapsed" : "open");
  // Focus follows the control that replaces the one you just used, so a
  // keyboard user is not left with focus on a hidden button.
  ($(collapsed ? "#expand" : "#collapse"))?.focus({ preventScroll: true });
}
$("#collapse").addEventListener("click", () => setSidebar(true));
$("#expand").addEventListener("click", () => setSidebar(false));
if (localStorage.getItem("gc-sidebar") === "collapsed") {
  $(".shell").classList.add("collapsed");
}

/* ── navigation ── */
function navigate(view) {
  $$(".nav-item").forEach((n) => {
    const on = n.dataset.view === view;
    n.classList.toggle("active", on);
    n.setAttribute("aria-selected", String(on));
  });
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));

  if (view === "params" && !paramsLoaded()) loadParams();
  if (view === "docs" && !docsLoaded()) loadDocs();
  if (view === "trace") renderTrace();
  if (view === "people" && !peopleLoaded()) loadPeople();
  if (view === "history" && !historyLoaded()) loadHistory();
}

$$(".nav-item").forEach((b) =>
  b.addEventListener("click", () => navigate(b.dataset.view)));
document.addEventListener("nav", (e) => navigate(e.detail));

/* The home page links straight at a view — /console#docs. Anything else in the
   hash is ignored rather than left on a blank screen. */
const VIEWS = new Set(["chat", "trace", "docs", "params", "people", "history"]);
const fromHash = () => {
  const view = location.hash.replace("#", "");
  if (VIEWS.has(view)) navigate(view);
};
addEventListener("hashchange", fromHash);

/* ── status ── */
async function refreshStatus() {
  const dot = $("#status-dot");
  const text = $("#status-text");
  try {
    const h = await api.health();
    if (!h.ok) {
      dot.className = "dot err";
      text.textContent = "config rejected";
      $("#foot-note").textContent = h.error || "";
      $("#send").disabled = true;
      return;
    }
    if (!h.model_rails) {
      dot.className = "dot warn";
      text.textContent = "deterministic rails only";
      $("#foot-note").textContent = h.note || "";
    } else {
      dot.className = "dot ok";
      text.textContent = "all rails live";
      $("#foot-note").textContent = "";
    }
  } catch {
    dot.className = "dot err";
    text.textContent = "server unreachable";
  }
}

/* ── identity ──
   The nav is rendered from the permission set the server enforces, so the
   sidebar cannot drift from what the API will actually allow. Hiding a tab is
   presentation; the 403 behind it is the control. */
async function applyIdentity() {
  let user;
  try {
    user = (await api.me()).user;
  } catch {
    return null;                       // api.js has already sent us to /login
  }
  const allowed = new Set(user.views);
  const held = new Set(user.permissions || []);
  $$(".nav-item").forEach((n) => {
    // A tab switches a view and is gated on that view. A link leaves the app,
    // so it is shown unless it names a permission this role does not hold —
    // without this branch a link, having no data-view, would hide for everyone.
    n.hidden = n.dataset.view
      ? !allowed.has(n.dataset.view)
      : Boolean(n.dataset.perm) && !held.has(n.dataset.perm);
  });

  // The recent-requests rail is a trace reader's tool; without the permission
  // it is a heading over an empty box.
  $("#recent-section").hidden = !allowed.has("trace");

  // The chat list, unlike the trace rail above, is not a specialist's tool —
  // anyone who can chat can see their own titles, so it loads unconditionally
  // rather than waiting for the History tab to be clicked.
  if (allowed.has("history")) refreshSidebarChats();

  const box = $("#side-user");
  box.hidden = false;
  $("#who-avatar").textContent = (user.display || user.name).slice(0, 2).toUpperCase();
  $("#who-name").textContent = user.display || user.name;
  $("#who-role").textContent = user.role_label;
  box.classList.toggle("admin", user.role === "admin");
  $("#sign-out").addEventListener("click", async () => {
    await api.logout();
    location.href = "/";
  });
  return user;
}

/* ── boot ── */
const identity = await applyIdentity();
await initChat();
initPeople();
initHistory();
if (identity?.views.includes("params")) initParams();
if (identity?.views.includes("docs")) initDocs();
fromHash();
if (identity?.views.includes("trace")) renderRecent();
refreshStatus();
