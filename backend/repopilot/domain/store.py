"""In-memory persistence for runs and their event streams.

This is the single place that owns run state. The API and the engine both go through it, so
there is exactly one view of what a run is doing.

Two properties matter for the rest of the system and are tested:

1. **Ordering.** Events get a per-run, gap-free, monotonic ``sequence`` starting at 1. A
   client can therefore resume from a cursor and detect loss.
2. **Isolation.** Reads return deep copies. A caller can never reach in and mutate stored
   state, which would desynchronise the run from its event stream.

Swapping this for Postgres later means reimplementing this one class; nothing above it knows
runs live in a dict.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Any

from repopilot.core.errors import InvalidTransitionError, NotFoundError
from repopilot.core.ids import new_id
from repopilot.core.time import utcnow
from repopilot.domain.models import (
    TERMINAL_RUN_STATUSES,
    EventLevel,
    EventType,
    ExecutionEvent,
    Run,
    RunStatus,
)

DEFAULT_MAX_EVENTS_PER_RUN = 5_000


class RunStore:
    """Thread- and task-safe in-memory store for :class:`Run` and :class:`ExecutionEvent`."""

    def __init__(self, max_events_per_run: int = DEFAULT_MAX_EVENTS_PER_RUN) -> None:
        self._max_events = max_events_per_run
        self._runs: dict[str, Run] = {}
        self._events: dict[str, deque[ExecutionEvent]] = {}
        self._sequence: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ runs

    async def create_run(
        self,
        *,
        repository: str,
        task: str,
        branch: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Run:
        """Create and register a new run in the ``pending`` state."""
        run = Run(
            repository=repository,  # type: ignore[arg-type]
            task=task,
            branch=branch,
            metadata=metadata or {},
        )
        async with self._lock:
            self._runs[run.id] = run
            self._events[run.id] = deque(maxlen=self._max_events)
            self._sequence[run.id] = 0
        return run.model_copy(deep=True)

    async def get_run(self, run_id: str) -> Run | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run else None

    async def require_run(self, run_id: str) -> Run:
        run = await self.get_run(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} does not exist")
        return run

    async def list_runs(self, *, status: RunStatus | None = None, limit: int = 50) -> list[Run]:
        """Newest first."""
        async with self._lock:
            runs: Iterable[Run] = reversed(list(self._runs.values()))
            if status is not None:
                runs = (r for r in runs if r.status == status)
            selected = list(runs)[:limit]
        return [r.model_copy(deep=True) for r in selected]

    async def update_run(self, run_id: str, **changes: Any) -> Run:
        """Patch arbitrary run fields, rejecting illegal status changes."""
        async with self._lock:
            run = self._require_locked(run_id)
            if "status" in changes:
                return self._transition_locked(run, changes.pop("status"), **changes)
            updated = run.model_copy(update=changes, deep=True)
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def transition(
        self, run_id: str, status: RunStatus, *, error: str | None = None, **changes: Any
    ) -> Run:
        """Move a run to a new status, enforcing the state machine."""
        async with self._lock:
            run = self._require_locked(run_id)
            return self._transition_locked(run, status, error=error, **changes)

    # ---------------------------------------------------------------- events

    async def append_event(
        self,
        run_id: str,
        *,
        type: EventType,
        message: str,
        level: EventLevel = EventLevel.INFO,
        step: str | None = None,
        step_index: int | None = None,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        """Append one event to a run's stream and return it with its assigned sequence."""
        async with self._lock:
            self._require_locked(run_id)
            sequence = self._sequence[run_id] + 1
            self._sequence[run_id] = sequence
            event = ExecutionEvent(
                id=new_id("evt"),
                run_id=run_id,
                sequence=sequence,
                type=type,
                level=level,
                message=message,
                step=step,
                step_index=step_index,
                progress=progress,
                data=data or {},
            )
            self._events[run_id].append(event)
        return event

    async def get_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int | None = None
    ) -> list[ExecutionEvent]:
        """Return events with ``sequence > after_sequence``, oldest first."""
        async with self._lock:
            self._require_locked(run_id)
            events = [e for e in self._events[run_id] if e.sequence > after_sequence]
        if limit is not None:
            events = events[:limit]
        return events

    async def latest_sequence(self, run_id: str) -> int:
        async with self._lock:
            self._require_locked(run_id)
            return self._sequence[run_id]

    # ----------------------------------------------------------------- stats

    async def counts_by_status(self) -> dict[str, int]:
        async with self._lock:
            counts: dict[str, int] = {status.value: 0 for status in RunStatus}
            for run in self._runs.values():
                counts[run.status.value] += 1
            return counts

    async def purge_terminal(self) -> int:
        """Drop finished runs. Useful for long-lived dev servers; not used in production paths."""
        async with self._lock:
            doomed = [
                run_id for run_id, run in self._runs.items() if run.status in TERMINAL_RUN_STATUSES
            ]
            for run_id in doomed:
                del self._runs[run_id]
                del self._events[run_id]
                del self._sequence[run_id]
            return len(doomed)

    # -------------------------------------------------------------- internal

    def _require_locked(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} does not exist")
        return run

    def _transition_locked(
        self, run: Run, status: RunStatus, *, error: str | None = None, **changes: Any
    ) -> Run:
        if not run.can_transition_to(status):
            raise InvalidTransitionError(run.id, run.status.value, status.value)

        now = utcnow()
        updates: dict[str, Any] = {"status": status, **changes}

        if status == RunStatus.RUNNING and run.started_at is None:
            updates["started_at"] = now
        if status in TERMINAL_RUN_STATUSES:
            updates["finished_at"] = now
            updates["current_step"] = None
            updates["current_step_index"] = None
        updates["error"] = error

        updated = run.model_copy(update=updates, deep=True)
        self._runs[run.id] = updated
        return updated.model_copy(deep=True)
