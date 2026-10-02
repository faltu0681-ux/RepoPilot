"""The structured artefacts Phase 2 produces.

These are plain pydantic models so that a malformed model reply is rejected at the boundary
with a clear message, rather than being coerced into something that looks plausible and is
wrong. Every model is also serialisable to the shape the event stream carries, because the
terminal UI and Phase 2D's run summary both read the plan back out of events.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: A repair round names at most this many failing tests. Past that the suite is failing for one
#: reason, and twenty findings is already more than a prompt should carry.
MAX_DIAGNOSED_FAILURES = 20

#: Matches the writer's own ceiling in :mod:`repopilot.agent.implement`, so a body the model
#: produced is never rejected later for being too large to be a file.
MAX_DIAGNOSED_FILE_CHARS = 200_000


class PlanStep(BaseModel):
    """One thing to do, in the order it should be done."""

    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(default="", max_length=4000)
    files: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=1000)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "title": self.title,
            "detail": self.detail,
            "files": list(self.files),
            "rationale": self.rationale,
        }


class FileSelection(BaseModel):
    """A file the model judged relevant, with why."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    reason: str = Field(default="", max_length=1000)
    role: Literal["modify", "add", "read"] = "modify"

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason": self.reason, "role": self.role}


class ImplementationPlan(BaseModel):
    """A validated plan for solving the task in this repository."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    approach: str = Field(default="", max_length=4000)
    relevant_files: list[FileSelection] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    test_strategy: str = Field(default="", max_length=4000)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @field_validator("steps")
    @classmethod
    def _renumber_steps(cls, steps: list[PlanStep]) -> list[PlanStep]:
        """Force a dense 1..n ordering so the terminal never renders a gap."""
        for index, step in enumerate(steps, start=1):
            step.order = index
        return steps

    @property
    def files_to_change(self) -> list[str]:
        """Files the plan intends to edit or create, de-duplicated and ordered."""
        ordered: list[str] = []
        for selection in self.relevant_files:
            if selection.role in ("modify", "add") and selection.path not in ordered:
                ordered.append(selection.path)
        return ordered

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "approach": self.approach,
            "relevant_files": [f.as_dict() for f in self.relevant_files],
            "steps": [s.as_dict() for s in self.steps],
            "risks": list(self.risks),
            "test_strategy": self.test_strategy,
            "files_to_change": self.files_to_change,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class FileChange(BaseModel):
    """The new contents of one file the plan asked for, validated but not yet written.

    Phase 2B asks the model for one file at a time, so the path and role are decided by the
    run rather than by the reply: the model is only ever asked for the body. That keeps a
    hallucinated path from turning into a file, because there is no path in the reply to
    hallucinate.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    role: Literal["modify", "add"]
    content: str

    @property
    def byte_size(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def line_count(self) -> int:
        return len(self.content.splitlines())

    def as_dict(self) -> dict[str, Any]:
        """The metadata an event may carry.

        Deliberately not the file body: events are persisted, replayed to every connected
        client, and kept for the life of the run, so a source file has no business in one.
        """
        return {
            "path": self.path,
            "role": self.role,
            "bytes": self.byte_size,
            "lines": self.line_count,
        }


class FailureDiagnosis(BaseModel):
    """One failing test, what the model thinks caused it, and the file that would fix it.

    ``pre_existing`` is the field that stops the repair loop from chasing a failure the run did
    not cause: a repository that was already broken stays reported rather than being rewritten
    twice in a row in the hope of fixing someone else's bug. A finding marked pre-existing is
    expected to carry no ``content``, because there is nothing to write.
    """

    model_config = ConfigDict(extra="forbid")

    test: str = Field(min_length=1, max_length=400)
    path: str = Field(min_length=1)
    cause: str = Field(min_length=1, max_length=2000)
    pre_existing: bool = False
    content: str = Field(default="", max_length=MAX_DIAGNOSED_FILE_CHARS)

    @property
    def actionable(self) -> bool:
        return not self.pre_existing and self.content.strip() != ""

    def as_dict(self) -> dict[str, Any]:
        """The metadata an event may carry. The corrected body stays on disk, not in the stream."""
        return {
            "test": self.test,
            "path": self.path,
            "cause": self.cause,
            "pre_existing": self.pre_existing,
            "bytes": len(self.content.encode("utf-8")),
            "actionable": self.actionable,
        }


class TestFailureReport(BaseModel):
    """What the model concluded about a failing suite, for one repair round."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    findings: list[FailureDiagnosis] = Field(max_length=MAX_DIAGNOSED_FAILURES)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def actionable(self) -> list[FailureDiagnosis]:
        """The findings that name a fix, in the order the model listed them."""
        return [finding for finding in self.findings if finding.actionable]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [finding.as_dict() for finding in self.findings],
            "actionable": len(self.actionable()),
            "pre_existing": sum(1 for finding in self.findings if finding.pre_existing),
            "model": self.model,
        }
