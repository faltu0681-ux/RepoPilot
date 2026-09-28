import { useEffect, useMemo, useRef, useState } from 'react'

import { createRun } from './api'
import { useRunStream } from './useRunStream'
import type {
  ExecutionEvent,
  FailureDiagnosis,
  ImplementationPlan,
  RunStatus,
  SelectedFile,
  TestFailure,
  TestOutcome,
  WrittenFile,
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
 * Renders what the run found and what it changed: the files it considered, the plan it made,
 * and the files it actually wrote.
 *
 * Everything here is derived from the event stream, so a reconnecting client rebuilds the
 * whole panel from the replayed events alone. There is no second source of truth to keep in
 * sync, and nothing is fetched that the backend did not actually report.
 */
function Workspace({ events }: { events: ExecutionEvent[] }) {
  const files = useMemo(() => collectSelectedFiles(events), [events])
  const plan = useMemo(() => findPlan(events), [events])
  const branch = useMemo(() => findBranch(events), [events])
  const changes = useMemo(() => collectWrittenFiles(events), [events])
  const verification = useMemo(() => readVerification(events), [events])

  if (files.length === 0 && plan === null && changes.length === 0 && !verification) return null

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

      {changes.length > 0 && <ChangesPanel changes={changes} branch={branch} />}

      {verification && <VerificationPanel verification={verification} />}
    </div>
  )
}

/** What the run wrote, on which branch. The bodies are not in the event stream, by design. */
function ChangesPanel({
  changes,
  branch,
}: {
  changes: WrittenFile[]
  branch: string | null
}) {
  const created = changes.filter((file) => file.created).length

  return (
    <section className="panel">
      <h2>
        Changes applied <span className="count">{changes.length}</span>
        {branch && <span className="model">{branch}</span>}
      </h2>

      <ul className="files changes">
        {changes.map((file) => (
          <li key={file.path}>
            <span className={`score ${file.created ? 'new' : 'edit'}`}>
              {file.created ? 'new' : 'edit'}
            </span>
            <span className="path">{file.path}</span>
            <span className="facts">
              {file.lines} lines · {file.bytes} bytes
            </span>
          </li>
        ))}
      </ul>

      {created > 0 && (
        <p className="summary">
          {created} file{created === 1 ? '' : 's'} created, {changes.length - created} rewritten.
          Changes are uncommitted on this branch, waiting for the test run.
        </p>
      )}
    </section>
  )
}

/**
 * Everything known about testing, rebuilt from the event stream.
 *
 * Only the last `test.completed` counts as the verdict; earlier ones are the rounds that led to
 * it, and a run that repaired a failure and then passed should read as passed, not as failed
 * with a later footnote.
 */
interface Verification {
  framework: string | null
  command: string[]
  suiteReason: string
  skipped: string | null
  outcome: TestOutcome | null
  runs: number
  failures: TestFailure[]
  diagnosis: { summary: string; findings: FailureDiagnosis[]; round: number } | null
  repairs: string[]
  attempts: number
}

const VERDICT_LABELS: Record<string, string> = {
  passed: 'passing',
  failed: 'failing',
  skipped: 'not run',
  timeout: 'timed out',
  unavailable: 'unavailable',
  cancelled: 'stopped',
}

function VerificationPanel({ verification }: { verification: Verification }) {
  const { framework, command, skipped, outcome, failures, diagnosis, repairs, attempts } =
    verification
  const status = outcome?.status ?? 'unknown'
  const counts = outcome
    ? [
        outcome.passed !== null ? `${outcome.passed} passed` : null,
        outcome.failed !== null ? `${outcome.failed} failed` : null,
        outcome.errors ? `${outcome.errors} errors` : null,
        outcome.skipped ? `${outcome.skipped} skipped` : null,
      ].filter((part): part is string => part !== null)
    : []

  return (
    <section className="panel">
      <h2>
        Verification{' '}
        {framework && <span className={`verdict ${status}`}>{VERDICT_LABELS[status] ?? status}</span>}
        {attempts > 1 && <span className="model">attempt {attempts}</span>}
      </h2>

      {skipped ? (
        <p className="summary">{skipped}</p>
      ) : (
        <>
          <p className="summary">
            <code className="command">{command.join(' ')}</code>
            {framework && ` · ${framework}`}
            {outcome && outcome.duration_seconds > 0 && ` · ${outcome.duration_seconds.toFixed(1)}s`}
            {verification.runs > 1 && ` · ${verification.runs} runs`}
          </p>

          {counts.length > 0 && (
            <ul className="counts">
              {counts.map((count) => (
                <li key={count}>{count}</li>
              ))}
            </ul>
          )}

          {failures.length > 0 && (
            <div className="block">
              <h3>Failing tests ({failures.length})</h3>
              <ul className="failures">
                {failures.map((failure) => (
                  <li key={failure.test}>
                    <span className="test">{failure.test}</span>
                    {failure.message && <p>{failure.message}</p>}
                    {failure.excerpt && <pre>{failure.excerpt}</pre>}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {diagnosis && (
            <div className="block">
              <h3>
                Diagnosis{diagnosis.round > 1 ? ` (round ${diagnosis.round})` : ''}
              </h3>
              <p>{diagnosis.summary}</p>
              <ul className="findings">
                {diagnosis.findings.map((finding) => (
                  <li key={`${finding.test}:${finding.path}`}>
                    <span className={`tag ${finding.pre_existing ? 'existing' : 'repair'}`}>
                      {finding.pre_existing ? 'pre-existing' : 'repair'}
                    </span>
                    <span className="test">{finding.test}</span>
                    <span className="path">{finding.path}</span>
                    <p>{finding.cause}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {repairs.length > 0 && (
            <p className="summary">
              Repaired {repairs.join(', ')} and ran the suite again.
            </p>
          )}

          {outcome?.output && (
            <details className="output">
              <summary>Test output</summary>
              <pre>{outcome.output}</pre>
            </details>
          )}
        </>
      )}
    </section>
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

/** The branch the run created, if it got that far. */
function findBranch(events: ExecutionEvent[]): string | null {
  for (const event of events) {
    if (event.type !== 'branch.created') continue
    const branch = asString(event.data.branch)
    if (branch) return branch
  }
  return null
}

/** Collect the written files, in the order the backend announced them. */
function collectWrittenFiles(events: ExecutionEvent[]): WrittenFile[] {
  const changes: WrittenFile[] = []
  const seen = new Set<string>()

  for (const event of events) {
    if (event.type !== 'file.written') continue
    const path = asString(event.data.path)
    if (!path || seen.has(path)) continue
    seen.add(path)
    const role = event.data.role
    changes.push({
      path,
      role: role === 'add' ? 'add' : 'modify',
      bytes: asNumber(event.data.bytes),
      lines: asNumber(event.data.lines),
      created: event.data.created === true,
      branch: asString(event.data.branch),
    })
  }

  return changes
}

/**
 * Rebuilds the verification panel from the event stream alone.
 *
 * Returns null until the run has said anything about testing, so a run that never reaches the
 * test step shows no panel at all rather than an empty one. Every field is read defensively: a
 * panel that throws on an unexpected payload would take the live view down with it.
 */
function readVerification(events: ExecutionEvent[]): Verification | null {
  let framework: string | null = null
  let command: string[] = []
  let suiteReason = ''
  let skipped: string | null = null
  let outcome: TestOutcome | null = null
  let runs = 0
  let diagnosis: Verification['diagnosis'] = null
  const repairs: string[] = []
  let attempts = 1

  for (const event of events) {
    if (event.type === 'run.started') {
      // The backend counts attempts from one, and the highest one it announced is where we are.
      attempts = Math.max(attempts, asNumber(event.data.attempt) || 1)
      continue
    }

    if (event.type === 'test.suite_detected') {
      framework = asString(event.data.framework) ?? framework
      command = asStrings(event.data.command)
      suiteReason = asString(event.data.reason) ?? suiteReason
      skipped = event.data.skipped === true ? (asString(event.data.reason) ?? 'no suite') : null
      continue
    }

    if (event.type === 'test.skipped') {
      skipped = asString(event.data.reason) ?? 'no suite'
      continue
    }

    if (event.type === 'test.completed') {
      runs += 1
      outcome = {
        framework: asString(event.data.framework) ?? framework ?? 'unknown',
        command: asStrings(event.data.command).length ? asStrings(event.data.command) : command,
        status: asString(event.data.status) ?? 'unknown',
        exit_code: typeof event.data.exit_code === 'number' ? event.data.exit_code : null,
        duration_seconds: asNumber(event.data.duration_seconds),
        reason: asString(event.data.reason) ?? '',
        output: asString(event.data.output) ?? '',
        passed: asOptionalNumber(event.data.passed),
        failed: asOptionalNumber(event.data.failed),
        errors: asOptionalNumber(event.data.errors),
        skipped: asOptionalNumber(event.data.skipped),
        total: asOptionalNumber(event.data.total),
        failures: readFailures(event.data.failures),
      }
      continue
    }

    if (event.type === 'failure.diagnosed') {
      diagnosis = {
        summary: asString(event.data.summary) ?? '',
        findings: readFindings(event.data.findings),
        round: Math.max(1, asNumber(event.data.round)),
      }
      continue
    }

    if (event.type === 'file.written' && event.data.repair === true) {
      const path = asString(event.data.path)
      if (path && !repairs.includes(path)) repairs.push(path)
    }
  }

  if (framework === null && skipped === null && outcome === null && diagnosis === null) return null

  return {
    framework,
    command,
    suiteReason,
    skipped,
    outcome,
    runs,
    failures: outcome?.failures ?? [],
    diagnosis,
    repairs,
    attempts,
  }
}

function readFailures(value: unknown): TestFailure[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => {
      if (typeof item !== 'object' || item === null) return null
      const record = item as Record<string, unknown>
      const test = asString(record.test)
      if (!test) return null
      return {
        test,
        message: asString(record.message) ?? '',
        excerpt: asString(record.excerpt) ?? '',
      }
    })
    .filter((failure): failure is TestFailure => failure !== null)
}

function readFindings(value: unknown): FailureDiagnosis[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => {
      if (typeof item !== 'object' || item === null) return null
      const record = item as Record<string, unknown>
      const test = asString(record.test)
      const path = asString(record.path)
      if (!test || !path) return null
      return {
        test,
        path,
        cause: asString(record.cause) ?? '',
        pre_existing: record.pre_existing === true,
        bytes: asNumber(record.bytes),
        actionable: record.actionable !== false,
      }
    })
    .filter((finding): finding is FailureDiagnosis => finding !== null)
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

/** A count the backend reported as "not measurable" stays absent rather than becoming a zero. */
function asOptionalNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
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
