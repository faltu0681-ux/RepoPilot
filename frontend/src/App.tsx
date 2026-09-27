import { useEffect, useMemo, useRef, useState } from 'react'

import { createRun } from './api'
import { useRunStream } from './useRunStream'
import type { ExecutionEvent, RunStatus } from './types'

const SAMPLE_REPOSITORY = 'octocat/Hello-World'
const SAMPLE_TASK = 'Add a /health endpoint that reports the service uptime'

export default function App() {
  const [repository, setRepository] = useState(SAMPLE_REPOSITORY)
  const [task, setTask] = useState(SAMPLE_TASK)
  const [runId, setRunId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const stream = useRunStream(runId)

  const start = async (event: React.FormEvent) => {
    event.preventDefault()
    if (submitting) return

    setSubmitting(true)
    setError(null)
    stream.clear()
    try {
      const run = await createRun(repository.trim(), task.trim())
      setRunId(run.id)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSubmitting(false)
    }
  }

  const restart = () => {
    setRunId(null)
    stream.clear()
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          Repo<span>Pilot</span>
        </h1>
        <p className="tagline">Point it at a repository and a task. Watch it work.</p>
      </header>

      <form className="controls" onSubmit={start}>
        <label>
          Repository
          <input
            value={repository}
            onChange={(e) => setRepository(e.target.value)}
            placeholder="owner/name"
            spellCheck={false}
            autoComplete="off"
            required
          />
        </label>
        <label className="grow">
          Task
          <input
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="What should it do?"
            required
          />
        </label>
        <button type="submit" disabled={submitting || !repository.trim() || !task.trim()}>
          {submitting ? 'Starting' : 'Start run'}
        </button>
        {runId && (
          <button type="button" className="ghost" onClick={restart}>
            New run
          </button>
        )}
      </form>

      {error && <p className="banner error">{error}</p>}

      <Terminal
        runId={runId}
        events={stream.events}
        status={stream.status}
        state={stream.state}
        detail={stream.detail}
      />
    </div>
  )
}

interface TerminalProps {
  runId: string | null
  events: ExecutionEvent[]
  status: RunStatus | null
  state: string
  detail: string | null
}

function Terminal({ runId, events, status, state, detail }: TerminalProps) {
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  // Follow the tail unless the reader has scrolled up to look at something.
  useEffect(() => {
    const element = scroller.current
    if (element && pinned.current) element.scrollTop = element.scrollHeight
  }, [events])

  const summary = useMemo(() => {
    const steps = new Map<string, number>()
    for (const event of events) {
      if (event.step) steps.set(event.step, (steps.get(event.step) ?? 0) + 1)
    }
    return steps.size
  }, [events])

  const onScroll = () => {
    const element = scroller.current
    if (!element) return
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight
    pinned.current = distance < 40
  }

  if (!runId) {
    return (
      <section className="terminal idle">
        <p>No run yet. Start one above.</p>
      </section>
    )
  }

  return (
    <section className="terminal">
      <div className="terminal-bar">
        <span className={`dot ${state}`} />
        <span className="run-id">{runId}</span>
        <span className="spacer" />
        {summary > 0 && <span className="meta">{summary} steps</span>}
        {status && <span className={`badge ${status}`}>{status}</span>}
        <span className={`meta ${state}`}>{detail ?? state}</span>
      </div>

      <div className="terminal-body" ref={scroller} onScroll={onScroll}>
        {events.length === 0 && <p className="muted">Waiting for the first event</p>}
        {events.map((event) => (
          <Line key={event.sequence} event={event} />
        ))}
      </div>
    </section>
  )
}

function Line({ event }: { event: ExecutionEvent }) {
  return (
    <div className={`line ${event.level}`}>
      <span className="seq">{String(event.sequence).padStart(3, ' ')}</span>
      <span className="type">{event.type}</span>
      {event.step && <span className="step">{event.step}</span>}
      <span className="message">{event.message}</span>
    </div>
  )
}
