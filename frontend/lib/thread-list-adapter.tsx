'use client'

import { useMemo, type FC, type PropsWithChildren } from 'react'
import {
  ExportedMessageRepository,
  RuntimeAdapterProvider,
  useAui,
  type RemoteThreadListAdapter,
  type ThreadHistoryAdapter,
  type ThreadMessageLike,
} from '@assistant-ui/react'
import {
  createSession,
  deleteSession,
  getSessionMessages,
  listSessions,
  type StoredMessage,
} from '@/lib/api'
import { sourcesToMarkdown } from '@/lib/chat-adapter'

const toThreadMessage = (m: StoredMessage): ThreadMessageLike => {
  const sources = m.role === 'assistant' ? (m.sources ?? []) : []
  return {
    id: String(m.id),
    role: m.role,
    content: [{ type: 'text', text: m.content + sourcesToMarkdown(sources) }],
    createdAt: new Date(m.created_at),
  }
}

/**
 * Loads a thread's history from the backend. The backend already persists every
 * turn during the SSE stream, so `append` is a no-op — we only ever read.
 */
const useBackendHistoryAdapter = (): ThreadHistoryAdapter => {
  const aui = useAui()
  return useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        const remoteId = aui.threadListItem().getState().remoteId
        if (!remoteId) return { messages: [] }
        const messages = await getSessionMessages(remoteId)
        return ExportedMessageRepository.fromArray(messages.map(toThreadMessage))
      },
      async append() {
        // backend persists messages itself during the message stream
      },
    }),
    [aui],
  )
}

const HistoryProvider: FC<PropsWithChildren> = ({ children }) => {
  const history = useBackendHistoryAdapter()
  const adapters = useMemo(() => ({ history }), [history])
  return <RuntimeAdapterProvider adapters={adapters}>{children}</RuntimeAdapterProvider>
}

/**
 * Backs the assistant-ui thread list with the UrbanLab backend so sessions
 * survive a page reload. `remoteId` is the backend chat-session id throughout.
 */
export const useBackendThreadListAdapter = (): RemoteThreadListAdapter =>
  useMemo<RemoteThreadListAdapter>(
    () => ({
      async list() {
        const sessions = await listSessions()
        return {
          threads: sessions.map((s) => ({
            status: 'regular' as const,
            remoteId: s.id,
            title: s.title ?? undefined,
            lastMessageAt: new Date(s.updated_at),
          })),
        }
      },
      async initialize() {
        const remoteId = await createSession()
        return { remoteId, externalId: undefined }
      },
      async delete(remoteId) {
        await deleteSession(remoteId)
      },
      // The backend auto-titles sessions from the first message; there are no
      // rename/archive endpoints, so these are intentional no-ops.
      async rename() {},
      async archive() {},
      async unarchive() {},
      async generateTitle() {
        return new ReadableStream()
      },
      async fetch(remoteId) {
        const sessions = await listSessions()
        const found = sessions.find((s) => s.id === remoteId)
        return {
          status: 'regular',
          remoteId,
          title: found?.title ?? undefined,
          lastMessageAt: found ? new Date(found.updated_at) : undefined,
        }
      },
      unstable_Provider: HistoryProvider,
    }),
    [],
  )
