/** The only module that talks to the server. */

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  // A dead session should return you to the door, not to a broken screen.
  if (res.status === 401 && !path.startsWith("/api/auth")) {
    location.href = `/login?next=${encodeURIComponent(location.pathname)}`;
    return new Promise(() => {});
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const message = data?.error?.message || `${res.status} ${res.statusText}`;
    const err = new Error(message);
    err.status = res.status;
    err.kind = data?.error?.kind;
    throw err;
  }
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

  // Multipart: no Content-Type header, so the browser sets the boundary.
  upload: (file, title = "") => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return fetch("/api/documents/upload", { method: "POST", body: form })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
          const err = new Error(data?.error?.message || res.statusText);
          err.kind = data?.error?.kind;
          throw err;
        }
        return data;
      });
  },

  deleteDocument: (id) =>
    request(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),

  resetDocuments: () => request("/api/documents/reset", { method: "POST" }),
};
