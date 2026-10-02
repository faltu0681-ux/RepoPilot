/**
 * Subscribe to a run's event stream.
 *
 * The backend replays stored history on connect, so this hook can attach to a run at any point
 * and end up with the whole story. Three things make that hold up:
 *
 * 1. The cursor advances with the highest sequence seen, and is sent as `after_sequence` on every
 *    reconnect, so a flaky connection never re-renders events the user already has.
 * 2. A `resync` frame means the server dropped events for us. We refetch them over REST from the
 *    sequence it names and splice them in, rather than showing a stream with a hole in it.
 * 3. Events are deduped by sequence on the way in, so a duplicate delivery is a no-op instead of a
 *    doubled line.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { getRun, getRunEvents } from './api'
import type { ExecutionEvent, Frame, RunStatus } from '@repopilot/contracts'

export type StreamState =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'finished'
  | 'reconnecting'
  | 'failed'

const MAX_BACKOFF_MS = 10_000

function socketUrl(runId: string, afterSequence: number): string {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const query = new URLSearchParams({
    run_id: runId,
    after_sequence: String(afterSequence),
  })
  return `${scheme}://${window.location.host}/ws/events?${query}`
}

export interface RunStream {
  events: ExecutionEvent[]
  status: RunStatus | null
  state: StreamState
  detail: string | null
  clear: () => void
}

export function useRunStream(runId: string | null): RunStream {
  const [events, setEvents] = useState<ExecutionEvent[]>([])
  const [status, setStatus] = useState<RunStatus | null>(null)
  const [state, setState] = useState<StreamState>('idle')
  const [detail, setDetail] = useState<string | null>(null)

  const cursor = useRef(0)
  const seen = useRef<Set<number>>(new Set())

  const add = useCallback((incoming: ExecutionEvent[]) => {
    const fresh = incoming.filter((event) => !seen.current.has(event.sequence))
    if (fresh.length === 0) return

    for (const event of fresh) {
      seen.current.add(event.sequence)
      if (event.sequence > cursor.current) cursor.current = event.sequence
    }

    setEvents((previous) => [...previous, ...fresh].sort((a, b) => a.sequence - b.sequence))
  }, [])

  const resync = useCallback(
    async (fromSequence: number) => {
      try {
        add(await getRunEvents(runId as string, fromSequence))
      } catch {
        // The socket will close and reconnect, which replays from the cursor anyway.
      }
    },
    [add, runId],
  )

  const clear = useCallback(() => {
    cursor.current = 0
    seen.current = new Set()
    setEvents([])
    setStatus(null)
    setState('idle')
    setDetail(null)
  }, [])

  useEffect(() => {
    if (!runId) return

    let socket: WebSocket | null = null
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    let disposed = false
    let attempt = 0
    let finished = false

    const connect = () => {
      if (disposed || finished) return
      setState(attempt === 0 ? 'connecting' : 'reconnecting')

      socket = new WebSocket(socketUrl(runId, cursor.current))

      socket.onopen = () => {
        attempt = 0
        setState('streaming')
        setDetail(null)
      }

      socket.onmessage = (message) => {
        let frame: Frame
        try {
          frame = JSON.parse(message.data as string) as Frame
        } catch {
          return
        }

        switch (frame.type) {
          case 'hello':
            break
          case 'event':
            add([frame.event])
            break
          case 'status':
            setStatus(frame.status)
            break
          case 'resync':
            setDetail(`${frame.dropped} events dropped, refetching`)
            void resync(frame.from_sequence)
            break
          case 'error':
            setDetail(frame.detail)
            break
          case 'bye':
            // A run-scoped socket only says goodbye when the run is over, so this is final.
            finished = true
            setState('finished')
            setDetail(frame.reason)
            void getRun(runId)
              .then((run) => setStatus(run.status))
              .catch(() => undefined)
            break
        }
      }

      socket.onerror = () => {
        // `onclose` always follows, and that is where the retry is scheduled.
      }

      socket.onclose = () => {
        if (disposed || finished) return
        attempt += 1
        const delay = Math.min(500 * 2 ** (attempt - 1), MAX_BACKOFF_MS)
        setState('reconnecting')
        setDetail(`connection lost, retrying in ${Math.round(delay / 100) / 10}s`)
        retryTimer = setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      disposed = true
      if (retryTimer) clearTimeout(retryTimer)
      socket?.close()
    }
  }, [add, resync, runId])

  return { events, status, state, detail, clear }
}
