/** The only module that talks to the server. */

// Every response this server ever sends is either empty or genuine JSON — an
// HTML body (a platform gateway's own 502/504 page, never something this
// FastAPI app produces) means something between the browser and the app
// failed, not that the app answered badly. Parsed once, here, so `request()`
// and `upload()` cannot drift out of sync on it the way they already have.
async function parseBody(res) {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;   // present but unparseable — distinct from "empty"
  }
}

function bodyError(res, data) {
  if (data === undefined) {
    return new Error(
      `The server sent back something that wasn't JSON (${res.status} ${res.statusText}) — `
      + "likely a gateway timeout or an outage between you and the app, not this app's own error.");
  }
  const message = data?.error?.message
    || (res.ok ? "The server sent an empty response." : `${res.status} ${res.statusText}`);
  const err = new Error(message);
  err.status = res.status;
  err.kind = data?.error?.kind;
  return err;
}

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  // A dead session should return you to the door, not to a broken screen.
  if (res.status === 401 && !path.startsWith("/api/auth")) {
    location.href = `/?next=${encodeURIComponent(location.pathname)}`;
    return new Promise(() => {});
  }
  const data = await parseBody(res);
  if (!res.ok || data === undefined) throw bodyError(res, data);
  return data;
}

export const api = {
  health:      ()               => request("/api/health"),
  me:          ()               => request("/api/auth/me"),
  logout:      ()               => request("/api/auth/logout", { method: "POST" }),
  samples:     ()               => request("/api/samples"),
  parameters:  ()               => request("/api/parameters"),
  changes:     ()               => request("/api/parameters/changes"),
  traces:      ()               => request("/api/traces"),
  auditVerify: ()               => request("/api/audit/verify"),

  /* conversations */
  history:     (who = "")       => request(
                                    `/api/history${who ? `?user=${encodeURIComponent(who)}` : ""}`),
  historySession: (who, sid)    => request(
                                    `/api/history/${encodeURIComponent(who)}/${encodeURIComponent(sid)}`),

  /* people */
  users:       ()               => request("/api/users"),
  addUser:     (body)           => request("/api/users", {
                                    method: "POST", body: JSON.stringify(body) }),
  setUser:     (name, body)     => request(`/api/users/${encodeURIComponent(name)}`, {
                                    method: "PATCH", body: JSON.stringify(body) }),
  resetUsage:  (name)           => request(
                                    `/api/users/${encodeURIComponent(name)}/reset-usage`,
                                    { method: "POST" }),
  setPassword: (name, password) => request(
                                    `/api/users/${encodeURIComponent(name)}/password`,
                                    { method: "PATCH", body: JSON.stringify({ password }) }),
  deleteUser:  (name)           => request(`/api/users/${encodeURIComponent(name)}`, {
                                    method: "DELETE" }),
  setPermission: (name, permission, held) => request(
                                    `/api/users/${encodeURIComponent(name)}/permissions`,
                                    { method: "PATCH", body: JSON.stringify({ permission, held }) }),

  chat: (message, sessionId) =>
    request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId }),
    }),

  resetSession: (sessionId) =>
    request(`/api/session/reset?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    }),

  patchParameters: (values = {}, matrix = {}) =>
    request("/api/parameters", {
      method: "PATCH",
      body: JSON.stringify({ values, matrix, author: "console" }),
    }),

  resetParameters: () =>
    request("/api/parameters/reset", { method: "POST" }),

  /* ── agent ── */
  agentTools:  ()               => request("/api/agent/tools"),

  agentChat: (message, sessionId) =>
    request("/api/agent/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId }),
    }),

  approve: (token, approved, sessionId) =>
    request("/api/agent/approve", {
      method: "POST",
      body: JSON.stringify({ token, approved, session_id: sessionId }),
    }),

  /* ── documents ── */
  documents:   ()               => request("/api/documents"),
  document:    (id)             => request(`/api/documents/${encodeURIComponent(id)}`),

  ingest: (title, text) =>
    request("/api/documents", {
      method: "POST",
      body: JSON.stringify({ title, text }),
    }),

  // Multipart: no Content-Type header, so the browser sets the boundary —
  // the one call that cannot go through request() itself, but it still
  // parses its response the same safe way, via the same shared helpers.
  upload: (file, title = "") => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return fetch("/api/documents/upload", { method: "POST", body: form })
      .then(async (res) => {
        const data = await parseBody(res);
        if (!res.ok || data === undefined) throw bodyError(res, data);
        return data;
      });
  },

  deleteDocument: (id) =>
    request(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),

  resetDocuments: () => request("/api/documents/reset", { method: "POST" }),
};
