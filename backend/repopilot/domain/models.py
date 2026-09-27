"""Core domain types.

This module imports nothing from the rest of the project. Domain knowledge stays here so the
execution layer and the API layer can both depend on it without depending on each other.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from repopilot.core.ids import new_id
from repopilot.core.time import utcnow
from repopilot.domain.repository import RepositoryRef


class RunStatus(StrEnum):
    """Where a run is in its lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED})

#: The run state machine. Anything not listed here is rejected.
ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.RUNNING, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


class EventType(StrEnum):
    """What an execution event represents."""

    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    STEP_STARTED = "step.started"
    STEP_PROGRESS = "step.progress"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    LOG = "log"
    # Phase 2. Discovery and planning facts, streamed so the terminal can render them.
    FILE_SELECTED = "file.selected"
    PLAN_CREATED = "plan.created"


class EventLevel(StrEnum):
    """Severity of an execution event, used for terminal colouring."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Run(BaseModel):
    """A single unit of agent work against one repository and one task."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: new_id("run"))
    repository: RepositoryRef
    task: str
    #: Defaults to a per-run branch name; the API layer may supply an explicit branch.
    branch: str | None = None
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    current_step: str | None = None
    current_step_index: int | None = None
    total_steps: int | None = None
    completed_steps: int = 0
    #: Incremented by the retry/fix loop. Zero until a first diagnosis-driven retry.
    attempts: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("repository", mode="before")
    @classmethod
    def _parse_repository(cls, value: Any) -> Any:
        if isinstance(value, (str, RepositoryRef)):
            return RepositoryRef.parse(value)
        return value

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        task = value.strip()
        if not task:
            raise ValueError("task cannot be empty")
        return task

    @field_serializer("repository")
    def _serialize_repository(self, value: RepositoryRef) -> str:
        return value.slug

    def model_post_init(self, context: Any, /) -> None:
        if self.branch is None:
            short_id = self.id.split("_", 1)[-1][:8]
            self.branch = f"repopilot/{short_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or utcnow()
        return (end - self.started_at).total_seconds()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress(self) -> float:
        """Fraction of steps finished, in ``0.0..1.0``. Zero when the plan is unknown."""
        if not self.total_steps:
            return 0.0
        if self.status == RunStatus.SUCCEEDED:
            return 1.0
        return min(self.completed_steps / self.total_steps, 1.0)

    def can_transition_to(self, status: RunStatus) -> bool:
        return status in ALLOWED_TRANSITIONS[self.status]


class ExecutionEvent(BaseModel):
    """One immutable observation emitted while a run executes.

    Events are the product of the whole system: the engine produces them, the store orders
    them per run, and the WebSocket and the terminal UI are both pure views over the stream.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    #: Monotonic per run, starting at 1. Lets a client detect gaps and resume from a cursor.
    sequence: int = Field(ge=1)
    type: EventType
    level: EventLevel = EventLevel.INFO
    message: str
    timestamp: datetime = Field(default_factory=utcnow)
    step: str | None = None
    step_index: int | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    data: dict[str, Any] = Field(default_factory=dict)
