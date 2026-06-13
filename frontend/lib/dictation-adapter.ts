import type { DictationAdapter } from '@assistant-ui/react'
import { transcribeAudio } from '@/lib/api'

/**
 * Dictation via our Whisper backend: record mic audio with MediaRecorder, then
 * on stop send it to Whisper's /asr endpoint and emit the transcript, which the
 * composer inserts into the text field. Batch (no interim results).
 */
export class WhisperDictationAdapter implements DictationAdapter {
  listen(): DictationAdapter.Session {
    const speech = new Set<(r: DictationAdapter.Result) => void>()
    const speechStart = new Set<() => void>()
    const speechEnd = new Set<(r: DictationAdapter.Result) => void>()

    let status: DictationAdapter.Status = { type: 'starting' }
    let recorder: MediaRecorder | null = null
    let stream: MediaStream | null = null
    const chunks: Blob[] = []

    const stopTracks = () => stream?.getTracks().forEach((t) => t.stop())

    void (async () => {
      try {
        // Microphone access requires a secure context. Over plain HTTP on a
        // non-localhost host (e.g. http://10.8.47.10:3000) the browser leaves
        // `navigator.mediaDevices` undefined, so getUserMedia would throw at
        // once — surface an actionable message instead of ending silently.
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error(
            window.isSecureContext
              ? 'Ta przeglądarka nie udostępnia mikrofonu.'
              : 'Nagrywanie wymaga HTTPS lub http://localhost — otwórz aplikację przez HTTPS, aby używać mikrofonu.',
          )
        }
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        recorder = new MediaRecorder(stream)
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data)
        }
        recorder.start()
        status = { type: 'running' }
        speechStart.forEach((cb) => cb())
      } catch (err) {
        status = { type: 'ended', reason: 'error' }
        stopTracks()
        if (typeof window !== 'undefined') {
          window.alert(err instanceof Error ? err.message : 'Nie udało się uruchomić mikrofonu.')
        }
      }
    })()

    return {
      get status() {
        return status
      },
      stop: async () => {
        if (!recorder || recorder.state === 'inactive') {
          status = { type: 'ended', reason: 'error' }
          stopTracks()
          return
        }
        const active = recorder
        await new Promise<void>((resolve) => {
          active.onstop = () => resolve()
          active.stop()
        })
        stopTracks()
        try {
          const blob = new Blob(chunks, { type: active.mimeType || 'audio/webm' })
          const transcript = (await transcribeAudio(blob)).trim()
          // onSpeech (isFinal) is what the composer appends to the input.
          if (transcript) speech.forEach((cb) => cb({ transcript, isFinal: true }))
          speechEnd.forEach((cb) => cb({ transcript, isFinal: true }))
          status = { type: 'ended', reason: 'stopped' }
        } catch (err) {
          status = { type: 'ended', reason: 'error' }
          throw err
        }
      },
      cancel: () => {
        try {
          recorder?.stop()
        } catch {
          /* ignore */
        }
        stopTracks()
        status = { type: 'ended', reason: 'cancelled' }
      },
      onSpeech: (cb) => {
        speech.add(cb)
        return () => void speech.delete(cb)
      },
      onSpeechStart: (cb) => {
        speechStart.add(cb)
        return () => void speechStart.delete(cb)
      },
      onSpeechEnd: (cb) => {
        speechEnd.add(cb)
        return () => void speechEnd.delete(cb)
      },
    }
  }
}
