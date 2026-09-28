/**
 * The wire types the backend speaks.
 *
 * These mirror `backend/repopilot/domain/models.py` and `backend/repopilot/api/ws.py`. The frame
 * `type` field is a discriminant, so a switch on it narrows the payload.
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
