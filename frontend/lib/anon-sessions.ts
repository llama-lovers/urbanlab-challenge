/**
 * Client-side registry of anonymous chat sessions.
 *
 * The backend ties sessions to a user via `user_id`, and `GET /sessions` only
 * lists sessions for the authenticated user — anonymous sessions (`user_id`
 * NULL) are never returned. They are, however, readable by id (the backend
 * grants access to any session without an owner), so remembering their ids in
 * localStorage is enough to rebuild the thread list after a reload.
 *
 * For logged-in users this registry is unused — the backend is the source of
 * truth — so every helper is a no-op when an auth token is present.
 */
import { getToken } from '@/lib/api'

const KEY = 'urbanlab_anon_sessions'

export type AnonSession = {
  id: string
  title: string | null
  updatedAt: string
}

const isBrowser = (): boolean => typeof window !== 'undefined'

const read = (): AnonSession[] => {
  if (!isBrowser()) return []
  try {
    const raw = localStorage.getItem(KEY)
    const parsed = raw ? (JSON.parse(raw) as unknown) : []
    return Array.isArray(parsed) ? (parsed as AnonSession[]) : []
  } catch {
    return []
  }
}

const write = (sessions: AnonSession[]): void => {
  if (!isBrowser()) return
  localStorage.setItem(KEY, JSON.stringify(sessions))
}

/** Mirror of the backend's `_make_title`: first line, truncated to 60 chars. */
const makeTitle = (content: string, maxLen = 60): string => {
  const trimmed = content.trim()
  if (trimmed.length <= maxLen) return trimmed
  return trimmed.slice(0, maxLen).replace(/\s+\S*$/, '') + '…'
}

/** Anonymous sessions, most-recently-updated first. */
export const listAnonSessions = (): AnonSession[] =>
  [...read()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))

/**
 * Record activity on a session: bump its `updatedAt` and, on the first message,
 * derive a title from the user's text (matching the backend). No-op when the
 * user is authenticated.
 */
export const recordAnonSession = (id: string, firstUserText: string): void => {
  if (getToken()) return
  const sessions = read()
  const now = new Date().toISOString()
  const existing = sessions.find((s) => s.id === id)
  if (existing) {
    existing.updatedAt = now
    if (!existing.title && firstUserText.trim()) {
      existing.title = makeTitle(firstUserText)
    }
  } else {
    sessions.push({
      id,
      title: firstUserText.trim() ? makeTitle(firstUserText) : null,
      updatedAt: now,
    })
  }
  write(sessions)
}

export const removeAnonSession = (id: string): void => {
  write(read().filter((s) => s.id !== id))
}
