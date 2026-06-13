import type { ChatModelAdapter, ChatModelRunOptions, ThreadMessage } from '@assistant-ui/react'
import { sendMessage } from '@/lib/api'

// Extract the text + first image (as a base64 data URL) from the latest user turn.
const lastUserInput = (messages: readonly ThreadMessage[]): { text: string; image?: string } => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role !== 'user') continue
    const text = m.content
      .map((p) => (p.type === 'text' ? p.text : ''))
      .join('')
      .trim()
    let image: string | undefined
    for (const p of m.content) {
      if (p.type === 'image' && p.image) {
        image = p.image
        break
      }
    }
    return { text, image }
  }
  return { text: '' }
}

export type Source = { title?: string; url?: string }

export const sourcesToMarkdown = (sources: Source[]): string => {
  if (!sources.length) return ''
  const items = sources
    .map((s) => {
      const title = s.title ?? s.url ?? 'source'
      return s.url ? `- [${title}](${s.url})` : `- ${title}`
    })
    .join('\n')
  return `\n\n---\n**Źródła:**\n${items}`
}

/**
 * assistant-ui runtime backed by the UrbanLab backend.
 *
 * The backend chat session id (the thread's `remoteId`) is resolved by the
 * thread-list runtime — see `resolveSessionId`, wired up in `assistant.tsx`.
 * This adapter just sends the latest user message (the backend keeps history +
 * does RAG) and renders the SSE stream (`delta` / `sources` / `done` / `error`).
 */
export class UrbanLabAdapter implements ChatModelAdapter {
  /** Resolves the backend session id for the active thread (idempotent). */
  constructor(private readonly resolveSessionId: () => Promise<string>) {}

  async *run({ messages, abortSignal }: ChatModelRunOptions) {
    const { text: content, image } = lastUserInput(messages)
    if (!content && !image) return

    const sessionId = await this.resolveSessionId()
    const response = await sendMessage(sessionId, content, image, abortSignal)

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let text = ''
    let sources: Source[] = []

    const emit = () => ({
      content: [{ type: 'text' as const, text: text + sourcesToMarkdown(sources) }],
    })

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames are separated by a blank line.
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        let event = 'message'
        let data = ''
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) data += line.slice(5).trim()
        }
        if (!data) continue

        if (event === 'delta') {
          text += (JSON.parse(data) as { text: string }).text
          yield emit()
        } else if (event === 'sources') {
          sources = JSON.parse(data) as Source[]
          yield emit()
        } else if (event === 'error') {
          const { detail } = JSON.parse(data) as { detail: string }
          throw new Error(detail)
        } else if (event === 'done') {
          return
        }
      }
    }
  }
}
