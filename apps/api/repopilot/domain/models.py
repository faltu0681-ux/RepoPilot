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


class RunVerdict(StrEnum):
    """What a finished run proved about its own change.

    Three states rather than two, because "the tests passed" and "the tests did not pass" are not
    the only two things a run can end with. A repository that declares no test runner, or one whose
    runner is not installed, was never checked either way, and reporting that as either a pass or a
    failure would be a claim the run has no evidence for. :mod:`repopilot.execution.steps` has
    always reported those cases as ``verified: False`` rather than as a green run; this enum is
    that distinction given a name.
    """

    #: The suite ran and passed on the code this run produced.
    VERIFIED = "verified"
    #: The suite ran and did not pass. The run says why, and what it tried.
    FAILED = "failed"
    #: Nothing proved or disproved the change: no runner was found, or the runner could not run.
    UNVERIFIED = "unverified"


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
    # Phase 2B. The plan turning into real files in the run's isolated workspace.
    BRANCH_CREATED = "branch.created"
    FILE_WRITTEN = "file.written"
    # Phase 2C. The change being proved, and what its failures mean.
    TEST_SUITE_DETECTED = "test.suite_detected"
    TEST_STARTED = "test.started"
    TEST_COMPLETED = "test.completed"
    TEST_SKIPPED = "test.skipped"
    FAILURE_DIAGNOSED = "failure.diagnosed"
    # Phase 2D. The change landing somewhere and being reported on.
    CHANGE_COMMITTED = "change.committed"
    PULL_REQUEST_OPENED = "pull_request.opened"
    RUN_SUMMARIZED = "run.summarized"
    #: V1. Where this run's untrusted commands execute, and what that environment enforces.
    #: Emitted once per run so a reader never has to assume which boundary was in force.
    SANDBOX_READY = "sandbox.ready"


class EventLevel(StrEnum):
    """Severity of an execution event, used for terminal colouring."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class RunSummary(BaseModel):
    """What one run did, stated once, plainly, at the end.

    The whole point of this type is that a reader does not have to reconstruct it. The plan, the
    written files, the test counts, the diagnosis, and the commit already exist as separate events,
    and a client that wants the shape of a run has to know that shape and fold ten event types
    into one answer. This is that answer, computed once by the pipeline and carried on the run
    itself, so ``GET /api/runs/{id}`` answers "what happened" without a client reading the stream.

    Every field is optional because a run can end before reaching any of them. A run that failed in
    the plan step has a verdict and nothing else, and that is a truthful summary rather than a
    summary padded with placeholders.
    """

    model_config = ConfigDict(frozen=True)

    #: What the run proved about its change. Always present, because a run always ends with one of
    #: these three and guessing is not an option.
    verdict: RunVerdict = RunVerdict.UNVERIFIED
    #: One line saying why the verdict is what it is, in the run's own terms.
    reason: str = ""

    #: What the model set out to do.
    task: str = ""
    plan_summary: str = ""
    #: The files the run changed, split by whether it created or rewrote them.
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)

    #: What proved it. Absent when no runner was found, which is a different fact from a runner
    #: that was found and failed, so it is not flattened into a null with a string next to it.
    test_framework: str | None = None
    test_command: list[str] = Field(default_factory=list)
    test_status: str | None = None
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_skipped: int | None = None
    test_output: str = ""
    #: Why nothing was verified, when the answer is UNVERIFIED.
    unverified_reason: str | None = None

    #: What the run did about a failure, if there was one.
    repair_rounds: int = 0
    diagnosis_summary: str = ""
    pre_existing_failures: int = 0

    #: Where the change ended up. A local run has no remote, so these stay absent rather than
    #: being filled with a branch that exists only on this machine.
    branch: str | None = None
    commit: str | None = None
    commit_message: str = ""
    pull_request_url: str | None = None
    pull_request_number: int | None = None
    #: Why no pull request was opened, when there is none. The honest reason: no remote, no token,
    #: or nothing to propose.
    pull_request_skipped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """The shape an event carries, so a client can render it without a second call."""
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "task": self.task,
            "plan_summary": self.plan_summary,
            "files_created": list(self.files_created),
            "files_modified": list(self.files_modified),
            "test_framework": self.test_framework,
            "test_command": list(self.test_command),
            "test_status": self.test_status,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "test_output": self.test_output,
            "unverified_reason": self.unverified_reason,
            "repair_rounds": self.repair_rounds,
            "diagnosis_summary": self.diagnosis_summary,
            "pre_existing_failures": self.pre_existing_failures,
            "branch": self.branch,
            "commit": self.commit,
            "commit_message": self.commit_message,
            "pull_request_url": self.pull_request_url,
            "pull_request_number": self.pull_request_number,
            "pull_request_skipped_reason": self.pull_request_skipped_reason,
        }


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
    #: Written by the summarise step at the end of a run, and absent until then. See
    #: :class:`RunSummary` for why this exists when the same facts are already in the event stream.
    summary: RunSummary | None = None

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
