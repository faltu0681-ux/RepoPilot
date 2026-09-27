"""The run engine.

Owns the lifecycle of every in-flight run: creates it, drives the step pipeline, records every
observation as an event, and translates step outcomes into run status changes.

Runs execute as background asyncio tasks, so ``POST /api/runs`` returns immediately and the
event stream is the only way to watch the work. The engine is the only writer of run status
after creation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from repopilot.config import Settings
from repopilot.core.errors import NotFoundError
from repopilot.domain.models import (
    EventLevel,
    EventType,
    ExecutionEvent,
    Run,
    RunStatus,
)
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.steps import (
    RunServices,
    Step,
    StepContext,
    StepFailed,
    StepOutcome,
    default_pipeline,
)
from repopilot.github.client import GitHubClient

logger = logging.getLogger("repopilot.engine")

#: How many times a run may be restarted by a step asking to retry.
MAX_ATTEMPTS = 3


class RunEngine:
    """Drives runs through the step pipeline."""

    def __init__(
        self,
        *,
        store: RunStore,
        bus: EventBus,
        settings: Settings,
        github: GitHubClient | None = None,
        steps: tuple[Step, ...] | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._settings = settings
        self._github = github or GitHubClient(
            token=settings.github_token.get_secret_value() if settings.github_token else None,
            api_url=settings.github_api_url,
        )
        self._steps = steps if steps is not None else default_pipeline()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    @property
    def steps(self) -> tuple[Step, ...]:
        return self._steps

    @property
    def store(self) -> RunStore:
        return self._store

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def github(self) -> GitHubClient:
        return self._github

    @property
    def active_runs(self) -> list[str]:
        return [run_id for run_id, task in self._tasks.items() if not task.done()]

    # ------------------------------------------------------------------ public

    async def start(
        self,
        *,
        repository: str,
        task: str,
        branch: str | None = None,
        metadata: dict | None = None,
    ) -> Run:
        """Create a run and begin executing it in the background."""
        run = await self._store.create_run(
            repository=repository,
            task=task,
            branch=branch,
            metadata=metadata or {},
        )
        await self._store.update_run(run.id, total_steps=len(self._steps))
        await self._emit(
            run.id,
            EventType.RUN_CREATED,
            f"Run created for {run.repository.slug}",
            data={
                "run_id": run.id,
                "repository": run.repository.slug,
                "task": run.task,
                "branch": run.branch,
                "steps": [step.name for step in self._steps],
            },
        )
        self._tasks[run.id] = asyncio.create_task(
            self._execute(run.id), name=f"repopilot-run-{run.id}"
        )
        logger.info("started run %s for %s", run.id, run.repository.slug)
        return await self._store.require_run(run.id)

    async def cancel(self, run_id: str) -> Run:
        """Ask a run to stop. Safe to call on a run that has already finished."""
        run = await self._store.require_run(run_id)
        if run.is_terminal:
            return run
        self._cancelled.add(run_id)
        await self._emit(run_id, EventType.LOG, "Cancellation requested", level=EventLevel.WARN)
        return await self._finish(run_id, RunStatus.CANCELLED, error="cancelled by user")

    async def wait_for(self, run_id: str, seconds: float | None = 30.0) -> None:
        """Wait for a run to finish. Used by tests and by the synchronous CLI."""
        task = self._tasks.get(run_id)
        if task is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), seconds)

    async def shutdown(self) -> None:
        """Cancel every in-flight run and release the GitHub client."""
        for run_id in list(self._tasks):
            self._cancelled.add(run_id)
            with contextlib.suppress(Exception):
                await self._store.transition(run_id, RunStatus.CANCELLED, error="server shut down")
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        await self._bus.shutdown()
        await self._github.aclose()

    # ----------------------------------------------------------------- private

    async def _execute(self, run_id: str) -> None:
        attempt = 0
        try:
            while True:
                await self._store.transition(run_id, RunStatus.RUNNING)
                await self._emit(
                    run_id,
                    EventType.RUN_STARTED,
                    f"Starting{'' if attempt == 0 else f' (attempt {attempt + 1})'}",
                    data={"attempt": attempt + 1, "steps": [s.name for s in self._steps]},
                )

                outcome, reason = await self._run_pipeline(run_id, attempt)

                if outcome is StepOutcome.OK:
                    await self._finish(run_id, RunStatus.SUCCEEDED)
                    return
                if outcome is StepOutcome.ABORT:
                    return  # the failure path already finalised the run
                if attempt + 1 >= MAX_ATTEMPTS:
                    await self._fail(run_id, f"{reason} (no attempts left)")
                    return

                attempt += 1
                await self._store.update_run(run_id, attempts=attempt)
                await self._emit(
                    run_id,
                    EventType.LOG,
                    f"{reason} Retrying (attempt {attempt + 1} of {MAX_ATTEMPTS})",
                    level=EventLevel.WARN,
                )
        except asyncio.CancelledError:
            logger.info("run %s cancelled", run_id)
            with contextlib.suppress(Exception):
                await self._finish(run_id, RunStatus.CANCELLED, error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - the engine must never die silently
            logger.exception("run %s failed", run_id)
            with contextlib.suppress(Exception):
                await self._store.transition(run_id, RunStatus.FAILED, error=str(exc))
            await self._emit(
                run_id,
                EventType.RUN_FINISHED,
                f"Run failed: {exc}",
                level=EventLevel.ERROR,
                data={"error": str(exc), "type": type(exc).__name__},
            )
            await self._publish_status(run_id, RunStatus.FAILED)

    async def _run_pipeline(self, run_id: str, attempt: int) -> tuple[StepOutcome, str]:
        """Execute every step in order. Returns the outcome and a human-readable reason."""
        run = await self._store.require_run(run_id)
        services = RunServices(
            github=self._github, workspace=self._settings.workspace_dir.resolve()
        )
        state: dict = {}
        reason = "pipeline completed"

        for index, step in enumerate(self._steps):
            if self._is_cancelled(run_id):
                return StepOutcome.ABORT, "cancelled by user"

            run = await self._store.update_run(
                run_id, current_step=step.name, current_step_index=index
            )
            ctx = StepContext(
                run=run,
                store=self._store,
                services=services,
                emit=self._make_emitter(run_id, step.name, index, len(self._steps)),
                is_cancelled=lambda: self._is_cancelled(run_id),
                attempt=attempt,
                state=state,
            )
            try:
                outcome = await step.execute(ctx)
            except StepFailed as exc:
                await self._emit(
                    run_id,
                    EventType.STEP_FAILED,
                    f"Step '{step.name}' failed: {exc}",
                    level=EventLevel.ERROR,
                    step=step.name,
                    step_index=index,
                    data={"error": str(exc), "retryable": exc.retryable},
                )
                if exc.retryable and attempt + 1 < MAX_ATTEMPTS:
                    return StepOutcome.RETRY, f"Step '{step.name}' failed: {exc}"
                await self._fail(run_id, f"Step '{step.name}' failed: {exc}")
                return StepOutcome.ABORT, str(exc)
            except Exception as exc:  # noqa: BLE001 - any step may fail any way
                await self._emit(
                    run_id,
                    EventType.STEP_FAILED,
                    f"Step '{step.name}' raised {type(exc).__name__}: {exc}",
                    level=EventLevel.ERROR,
                    step=step.name,
                    step_index=index,
                    data={"error": str(exc), "type": type(exc).__name__},
                )
                await self._fail(run_id, f"Step '{step.name}' raised {type(exc).__name__}: {exc}")
                return StepOutcome.ABORT, str(exc)

            if outcome is StepOutcome.ABORT:
                await self._fail(run_id, f"Step '{step.name}' aborted the run")
                return StepOutcome.ABORT, "aborted"
            if outcome is StepOutcome.RETRY:
                return StepOutcome.RETRY, f"Step '{step.name}' requested a retry"

            await self._store.update_run(run_id, completed_steps=index + 1)

        return StepOutcome.OK, reason

    def _make_emitter(self, run_id: str, step_name: str, step_index: int, total: int):
        """Build the ``ctx.emit`` coroutine for one step, with bookkeeping filled in."""

        async def emit(
            event_type: EventType | str,
            message: str,
            *,
            level: EventLevel | str = EventLevel.INFO,
            step: str | None = None,
            progress: float | None = None,
            data: dict | None = None,
        ) -> ExecutionEvent:
            return await self._emit(
                run_id,
                EventType(event_type),
                message,
                level=EventLevel(level),
                step=step or step_name,
                step_index=step_index,
                progress=progress if progress is not None else step_index / max(total, 1),
                data=data,
            )

        return emit

    def _is_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled

    async def _emit(
        self,
        run_id: str,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        step: str | None = None,
        step_index: int | None = None,
        progress: float | None = None,
        data: dict | None = None,
    ) -> ExecutionEvent:
        """Write an event to the store (the record) then fan it out on the bus (the fast path)."""
        try:
            event = await self._store.append_event(
                run_id,
                type=event_type,
                message=message,
                level=level,
                step=step,
                step_index=step_index,
                progress=progress,
                data=data,
            )
        except NotFoundError:
            logger.warning("dropping event for unknown run %s", run_id)
            raise
        await self._bus.publish_event(event)
        return event

    async def _fail(self, run_id: str, error: str) -> None:
        run = await self._store.transition(run_id, RunStatus.FAILED, error=error)
        await self._emit(
            run_id,
            EventType.RUN_FINISHED,
            f"Run failed: {error}",
            level=EventLevel.ERROR,
            data={"error": error},
        )
        await self._publish_status(run_id, run.status)

    async def _finish(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        run = await self._store.require_run(run_id)
        if run.is_terminal:
            return
        level = {
            RunStatus.SUCCEEDED: EventLevel.INFO,
            RunStatus.CANCELLED: EventLevel.WARN,
            RunStatus.FAILED: EventLevel.ERROR,
        }[status]
        message = {
            RunStatus.SUCCEEDED: "Run finished successfully",
            RunStatus.CANCELLED: "Run cancelled",
            RunStatus.FAILED: f"Run failed: {error}",
        }[status]
        # Settle the state first: an observer told the run is over must be able to read a
        # terminal status straight afterwards, with no window where the two disagree.
        run = await self._store.transition(run_id, status, error=error)
        await self._emit(
            run_id,
            EventType.RUN_FINISHED,
            message,
            level=level,
            data={"status": status.value, "error": error},
        )
        await self._publish_status(run_id, run.status)

    async def _publish_status(self, run_id: str, status: RunStatus) -> None:
        await self._bus.publish_status(run_id, status)
