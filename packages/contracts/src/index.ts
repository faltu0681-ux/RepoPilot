/**
 * The wire types the backend speaks.
 *
 * These mirror `apps/api/repopilot/domain/models.py` and `apps/api/repopilot/api/ws.py`, which are
 * the source of truth. The frame `type` field is a discriminant, so a switch on it narrows the
 * payload.
 *
 * This file is its own package rather than a file inside the web app because the contract is the
 * one thing in this repository that both stacks depend on conceptually: changing an event name or
 * a payload field here without changing it in Python breaks the UI silently, and TypeScript
 * cannot catch that for us. Keeping it in one named place at least gives it one name to grep for.
 */

export type RunStatus =
  | 'pending'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export const TERMINAL_RUN_STATUSES: readonly RunStatus[] = [
  'succeeded',
  'failed',
  'cancelled',
]

export type EventType =
  | 'run.created'
  | 'run.started'
  | 'run.finished'
  | 'step.started'
  | 'step.progress'
  | 'step.completed'
  | 'step.failed'
  | 'log'
  | 'file.selected'
  | 'plan.created'
  | 'branch.created'
  | 'file.written'
  | 'test.suite_detected'
  | 'test.started'
  | 'test.completed'
  | 'test.skipped'
  | 'failure.diagnosed'
  | 'change.committed'
  | 'pull_request.opened'
  | 'run.summarized'
  | 'sandbox.ready'

/**
 * Where a run's untrusted commands execute, and what that environment enforces.
 *
 * The payload of `sandbox.ready`. A reader deciding whether to trust a run needs to know which
 * boundary its test suite ran behind: `local` means a child process on the host with no filesystem
 * or network isolation, and `docker` means a throwaway container with the run's checkout as its
 * only mount. The `isolation` and `resources` fields are descriptions, not guarantees.
 */
export type SandboxDescription = {
  name: 'local' | 'docker'
  isolation: string
  network: string
  filesystem: string
  resources: string
  /** Present only for the container backend: the image the commands ran in. */
  image?: string
}

export type EventLevel = 'debug' | 'info' | 'warn' | 'error'

/** Mirrors `repopilot.analysis.relevance.CandidateFile`. */
export interface SelectedFile {
  path: string
  score: number
  reasons: string[]
  language: string | null
  lines: number
  is_test: boolean
  root?: string
}

/** Mirrors `repopilot.agent.schema`. */
export type FileRole = 'modify' | 'add' | 'read'

export interface PlanFile {
  path: string
  reason: string
  role: FileRole
}

export interface PlanStep {
  order: number
  title: string
  detail: string
  files: string[]
  rationale: string
}

export interface ImplementationPlan {
  summary: string
  approach: string
  relevant_files: PlanFile[]
  steps: PlanStep[]
  risks: string[]
  test_strategy: string
  files_to_change: string[]
  model: string
  prompt_tokens: number | null
  completion_tokens: number | null
}

/** One entry of the `file.written` event, which reports a file without carrying its body. */
export interface WrittenFile {
  path: string
  role: FileRole
  bytes: number
  lines: number
  created: boolean
  branch: string | null
  /** Set on a file the diagnosis step repaired after a failing suite (Phase 2C). */
  repair?: boolean
  round?: number
}

/**
 * What the run decided to do about testing, read from the `test.*` events rather than from a
 * second API. Kept deliberately flat: the events are the record, and a panel that has to be
 * reconciled with them is a panel that can quietly disagree with them.
 */
export type TestVerdict = 'passed' | 'failed' | 'skipped' | 'timeout' | 'unavailable' | 'cancelled'

/** Mirrors `repopilot.analysis.testing.TestFailure`. */
export interface TestFailure {
  test: string
  message: string
  excerpt: string
}

/** Mirrors `repopilot.analysis.testing.TestOutcome`, the payload of `test.completed`. */
export interface TestOutcome {
  framework: string
  command: string[]
  status: TestVerdict | string
  exit_code: number | null
  duration_seconds: number
  reason: string
  /** The tail of the run's output, trimmed by the backend. */
  output: string
  passed: number | null
  failed: number | null
  errors: number | null
  skipped: number | null
  total: number | null
  failures: TestFailure[]
}

/** Mirrors `repopilot.agent.schema.FailureDiagnosis`, as carried by `failure.diagnosed`. */
export interface FailureDiagnosis {
  test: string
  path: string
  cause: string
  pre_existing: boolean
  bytes: number
  actionable: boolean
}

export interface ExecutionEvent {
  run_id: string
  sequence: number
  type: EventType
  level: EventLevel
  message: string
  timestamp: string
  step: string | null
  step_index: number | null
  progress: number | null
  data: Record<string, unknown>
}

/**
 * What the run established about its own change.
 *
 * Three values, not two, because `unverified` and `failed` are different promises. `verified`
 * means a suite passed. `failed` means a suite ran and did not pass. `unverified` means no suite
 * ran at all, so nothing is claimed either way. Collapsing the last two into a single "not
 * verified" would report a repository with no test runner as a failure, and would report a timeout
 * as broken code.
 */
export type RunVerdict = 'verified' | 'failed' | 'unverified'

/** Mirrors `repopilot.domain.models.RunSummary`, the payload of `run.summarized`. */
export interface RunSummary {
  verdict: RunVerdict
  /** The one sentence explaining the verdict, written to be read on its own. */
  reason: string
  task: string
  plan_summary: string
  files_created: string[]
  files_modified: string[]
  test_framework: string | null
  test_command: string[]
  test_status: string | null
  tests_passed: number | null
  tests_failed: number | null
  tests_skipped: number | null
  test_output: string
  unverified_reason: string | null
  repair_rounds: number
  diagnosis_summary: string
  pre_existing_failures: number
  branch: string | null
  commit: string | null
  commit_message: string
  pull_request_url: string | null
  pull_request_number: number | null
  /** Why there is no pull request, when there is not one. Absent means one was opened. */
  pull_request_skipped_reason: string | null
}

export interface Run {
  id: string
  repository: string
  branch: string
  task: string
  status: RunStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  duration_seconds: number | null
  total_steps: number
  completed_steps: number
  progress: number
  attempts: number
  error: string | null
  /**
   * Present once the run has finished reporting. `null` while it is still running, and on a run
   * that never reached the summary step, so a client must treat absent as "not reported yet"
   * rather than as a failed run.
   */
  summary: RunSummary | null
}

/** A folder the backend can be pointed at with `local://<name>`. */
export interface LocalRepository {
  /** Folder name, and the target of the `local://` reference. */
  name: string
  /** The `local://<name>` string to submit as the repository. */
  reference: string
  path: string
}

/** Response of `GET /api/runs/repositories`. */
export interface LocalRepositories {
  /** Where the folders were read from, so the UI can show the real path. */
  repositoriesDir: string
  repositories: LocalRepository[]
}

export interface HelloFrame {
  type: 'hello'
  version: number
  run_id: string | null
  after_sequence: number
}

export interface EventFrame {
  type: 'event'
  event: ExecutionEvent
}

export interface StatusFrame {
  type: 'status'
  run_id: string
  status: RunStatus
}

export interface ResyncFrame {
  type: 'resync'
  run_id: string | null
  from_sequence: number
  dropped: number
  reason: string
}

export interface ErrorFrame {
  type: 'error'
  code: string
  detail: string
}

export interface ByeFrame {
  type: 'bye'
  reason: string
}

export type Frame =
  | HelloFrame
  | EventFrame
  | StatusFrame
  | ResyncFrame
  | ErrorFrame
  | ByeFrame
