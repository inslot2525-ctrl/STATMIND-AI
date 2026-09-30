// API client: auth token handling, JSON/form requests, background-job polling
// and signed-URL downloads. Every backend call goes through apiFetch.

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const TOKEN_KEY = "statmind_token";
const USER_KEY = "statmind_user";

// Storage can throw (private mode, blocked site data); treat that as logged out
function readStorage(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export function getToken() {
  return readStorage(TOKEN_KEY);
}

export function getStoredUser() {
  try {
    return JSON.parse(readStorage(USER_KEY) || "null");
  } catch {
    return null;
  }
}

function saveSession({ access_token, user }) {
  writeStorage(TOKEN_KEY, access_token);
  writeStorage(USER_KEY, JSON.stringify(user));
  return user;
}

export function clearSession() {
  writeStorage(TOKEN_KEY, null);
  writeStorage(USER_KEY, null);
}

let unauthorizedHandler = () => {};

// Called when the server rejects our token (expired or revoked)
export function onUnauthorized(handler) {
  unauthorizedHandler = handler;
}

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: [...], msg: "..."}]
    return detail.map((d) => `${(d.loc || []).slice(-1)[0] || "field"}: ${d.msg}`).join("; ");
  }
  return JSON.stringify(detail);
}

export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  } catch {
    throw new Error("Cannot reach the StatMind server. Check that the backend is running.");
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      if (response.ok) throw new Error("Backend did not return valid JSON.");
    }
  }

  if (response.status === 401 && token) {
    clearSession();
    unauthorizedHandler();
  }

  if (!response.ok) {
    throw new Error(formatDetail(data?.detail) || `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

export function postForm(path, formData) {
  return apiFetch(path, { method: "POST", body: formData });
}

export function postJson(path, body) {
  return apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ---- auth ---------------------------------------------------------------

export async function login(email, password) {
  const body = new URLSearchParams({ username: email, password });
  return saveSession(await apiFetch("/api/auth/login", { method: "POST", body }));
}

export async function register(email, password) {
  return saveSession(await postJson("/api/auth/register", { email, password }));
}

// ---- background jobs ----------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Start a job endpoint (returns 202 + job) and poll until it finishes.
// onProgress receives the job ({status, progress, message}) on every poll.
export async function runJob(path, formData, onProgress = () => {}) {
  let job = await postForm(path, formData);
  let delay = 700;

  while (job.status === "queued" || job.status === "running") {
    onProgress(job);
    await sleep(delay);
    delay = Math.min(delay * 1.3, 3000);
    job = await apiFetch(`/api/jobs/${job.id}`);
  }

  onProgress(job);
  if (job.status !== "succeeded") {
    throw new Error(job.error || "The job failed.");
  }
  return job.result;
}

// ---- downloads ----------------------------------------------------------

// Ask for a short-lived signed URL, then let the browser download it
export async function downloadArtifact(artifactId) {
  const { url } = await apiFetch(`/api/artifacts/${artifactId}/download-url`);
  window.location.assign(url.startsWith("/") ? `${API_BASE_URL}${url}` : url);
}
