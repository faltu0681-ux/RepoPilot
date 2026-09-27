import { useEffect, useMemo, useRef, useState } from 'react'

import { createRun } from './api'
import { useRunStream } from './useRunStream'
import type {
  ExecutionEvent,
  ImplementationPlan,
  RunStatus,
  SelectedFile,
} from './types'

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

      <Workspace events={stream.events} />
    </div>
  )
}

/**
 * Renders what the run found: the files it considered, and the plan it produced.
 *
 * Everything here is derived from the event stream, so a reconnecting client rebuilds the
 * whole panel from the replayed events alone. There is no second source of truth to keep in
 * sync, and nothing is fetched that the backend did not actually report.
 */
function Workspace({ events }: { events: ExecutionEvent[] }) {
  const files = useMemo(() => collectSelectedFiles(events), [events])
  const plan = useMemo(() => findPlan(events), [events])

  if (files.length === 0 && plan === null) return null

  return (
    <div className="workspace">
      {files.length > 0 && (
        <section className="panel">
          <h2>
            Discovered files <span className="count">{files.length}</span>
          </h2>
          <ul className="files">
            {files.map((file) => (
              <li key={file.path}>
                <span className={`score${file.is_test ? ' test' : ''}`}>
                  {file.score.toFixed(1)}
                </span>
                <span className="path">{file.path}</span>
                <span className="facts">
                  {file.language ?? 'unknown'} · {file.lines} lines
                  {file.is_test ? ' · test' : ''}
                </span>
                {file.reasons.length > 0 && (
                  <span className="because">{file.reasons.join('; ')}</span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {plan && <PlanPanel plan={plan} />}
    </div>
  )
}

function PlanPanel({ plan }: { plan: ImplementationPlan }) {
  const filesToChange = plan.files_to_change.length

  return (
    <section className="panel">
      <h2>
        Implementation plan{' '}
        {plan.model && <span className="model">{plan.model}</span>}
      </h2>

      <p className="summary">{plan.summary}</p>

      {plan.approach && (
        <div className="block">
          <h3>Approach</h3>
          <p>{plan.approach}</p>
        </div>
      )}

      {plan.steps.length > 0 && (
        <div className="block">
          <h3>Steps</h3>
          <ol className="plan-steps">
            {plan.steps.map((step) => (
              <li key={step.order}>
                <span className="title">{step.title}</span>
                {step.files.length > 0 && <span className="path">{step.files.join(', ')}</span>}
                {step.detail && <p>{step.detail}</p>}
                {step.rationale && <p className="rationale">{step.rationale}</p>}
              </li>
            ))}
          </ol>
        </div>
      )}

      {plan.relevant_files.length > 0 && (
        <div className="block">
          <h3>Files ({filesToChange} to change)</h3>
          <ul className="plan-files">
            {plan.relevant_files.map((file) => (
              <li key={file.path}>
                <span className={`role ${file.role}`}>{file.role}</span>
                <span className="path">{file.path}</span>
                {file.reason && <span className="because">{file.reason}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {plan.test_strategy && (
        <div className="block">
          <h3>Test strategy</h3>
          <p>{plan.test_strategy}</p>
        </div>
      )}

      {plan.risks.length > 0 && (
        <div className="block">
          <h3>Risks</h3>
          <ul className="risks">
            {plan.risks.map((risk) => (
              <li key={risk}>{risk}</li>
            ))}
          </ul>
        </div>
      )}

      {plan.prompt_tokens !== null && plan.completion_tokens !== null && (
        <p className="tokens">
          {plan.prompt_tokens} in / {plan.completion_tokens} out tokens
        </p>
      )}
    </section>
  )
}

/** Collect the shortlisted files, in the order the backend announced them. */
function collectSelectedFiles(events: ExecutionEvent[]): SelectedFile[] {
  const files: SelectedFile[] = []
  const seen = new Set<string>()

  for (const event of events) {
    if (event.type !== 'file.selected') continue
    const path = event.data.path
    if (typeof path !== 'string' || seen.has(path)) continue
    seen.add(path)
    files.push({
      path,
      score: asNumber(event.data.score),
      reasons: asStrings(event.data.reasons),
      language: asString(event.data.language),
      lines: asNumber(event.data.lines),
      is_test: event.data.is_test === true,
    })
  }

  return files
}

function findPlan(events: ExecutionEvent[]): ImplementationPlan | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event.type !== 'plan.created') continue
    const plan = event.data.plan
    if (plan && typeof plan === 'object') return plan as ImplementationPlan
  }
  return null
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function asNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
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
