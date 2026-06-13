/**
 * Thin client for the UrbanLab backend (see backend/openapi.json).
 * Handles JWT auth + chat sessions + the SSE message stream.
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "urbanlab_token";

export const getToken = (): string | null =>
  typeof window === "undefined" ? null : localStorage.getItem(TOKEN_KEY);

export const setToken = (token: string): void => localStorage.setItem(TOKEN_KEY, token);

export const clearToken = (): void => localStorage.removeItem(TOKEN_KEY);

const authHeaders = (): Record<string, string> => {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

async function asError(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => null);
  const detail =
    body?.detail && typeof body.detail === "string" ? body.detail : fallback;
  return new Error(detail);
}

// ---- auth ----

export type AuthUser = { id: number; email: string; created_at: string };

async function authenticate(
  path: "login" | "register",
  email: string,
  password: string,
): Promise<string> {
  const res = await fetch(`${API_URL}/api/auth/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw await asError(res, `${path} failed`);
  const { access_token } = (await res.json()) as { access_token: string };
  setToken(access_token);
  return access_token;
}

export const login = (email: string, password: string) =>
  authenticate("login", email, password);
export const register = (email: string, password: string) =>
  authenticate("register", email, password);

export async function me(): Promise<AuthUser> {
  const res = await fetch(`${API_URL}/api/auth/me`, { headers: authHeaders() });
  if (!res.ok) throw await asError(res, "Not authenticated");
  return res.json();
}

// ---- chat ----

export async function createSession(): Promise<string> {
  const res = await fetch(`${API_URL}/api/chat/sessions`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw await asError(res, "Could not create session");
  const { id } = (await res.json()) as { id: string };
  return id;
}

/** POST a user message; returns the raw SSE Response for the caller to stream. */
export async function sendMessage(
  sessionId: string,
  content: string,
  signal?: AbortSignal,
): Promise<Response> {
  const res = await fetch(`${API_URL}/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal,
  });
  if (!res.ok || !res.body) throw await asError(res, `Request failed (${res.status})`);
  return res;
}
