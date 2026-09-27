"""The structured artefacts Phase 2 produces.

These are plain pydantic models so that a malformed model reply is rejected at the boundary
with a clear message, rather than being coerced into something that looks plausible and is
wrong. Every model is also serialisable to the shape the event stream carries, because the
terminal UI and Phase 2D's run summary both read the plan back out of events.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
