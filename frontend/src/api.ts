/** REST calls against the FastAPI backend. */

import type { ExecutionEvent, Run } from './types'

const BASE = '/api'

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
  })

  if (!response.ok) {
    // The backend returns {"detail": "..."} for HTTPException and
    // {"message": "..."} for domain errors, so read whichever is there.
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      detail = body?.detail ?? body?.message ?? detail
    } catch {
      // Body was not JSON; the status line is the best we have.
    }
    throw new ApiError(detail, response.status)
  }

  return response.json() as Promise<T>
}

export function createRun(repository: string, task: string): Promise<Run> {
  return request<Run>('/runs', {
    method: 'POST',
    body: JSON.stringify({ repository, task }),
  })
}

export function getRun(runId: string): Promise<Run> {
  return request<Run>(`/runs/${runId}`)
}

export function getRunEvents(runId: string, afterSequence = 0): Promise<ExecutionEvent[]> {
  const query = new URLSearchParams({ after_sequence: String(afterSequence) })
  return request<ExecutionEvent[]>(`/runs/${runId}/events?${query}`)
}

export function cancelRun(runId: string): Promise<Run> {
  return request<Run>(`/runs/${runId}`, { method: 'DELETE' })
}
