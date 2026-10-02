"""Tests for the domain value types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from repopilot.domain.models import (
    ALLOWED_TRANSITIONS,
    TERMINAL_RUN_STATUSES,
    EventType,
    ExecutionEvent,
    Run,
    RunStatus,
)
from repopilot.domain.repository import RepositoryRef

ALL_STATUSES = list(RunStatus)


class TestRepositoryRef:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("pallets/flask", "pallets/flask"),
            ("pallets/flask.git", "pallets/flask"),
            ("pallets/flask/", "pallets/flask"),
            ("https://github.com/pallets/flask", "pallets/flask"),
            ("https://github.com/pallets/flask.git", "pallets/flask"),
            ("https://github.com/pallets/flask/tree/main/src", "pallets/flask"),
            ("git@github.com:pallets/flask.git", "pallets/flask"),
            ("  pallets/flask  ", "pallets/flask"),
        ],
    )
    def test_parses_every_common_form(self, raw: str, expected: str) -> None:
        assert RepositoryRef.parse(raw).slug == expected

    def test_exposes_urls(self) -> None:
        ref = RepositoryRef.parse("pallets/flask")

        assert ref.https_url == "https://github.com/pallets/flask"
        assert ref.clone_url == "https://github.com/pallets/flask.git"
        assert str(ref) == "pallets/flask"

    def test_is_idempotent(self) -> None:
        ref = RepositoryRef.parse("pallets/flask")
        assert RepositoryRef.parse(ref) is ref

    @pytest.mark.parametrize("raw", ["", "   ", "flask", "a/b/c/d", "https://github.com/flask"])
    def test_rejects_nonsense(self, raw: str) -> None:
        with pytest.raises(ValueError):
            RepositoryRef.parse(raw)

    def test_rejects_illegal_characters(self) -> None:
        with pytest.raises(ValidationError):
            RepositoryRef(owner="pall ets", name="flask")

    def test_is_frozen(self) -> None:
        ref = RepositoryRef.parse("pallets/flask")
        with pytest.raises(ValidationError):
            ref.owner = "other"  # type: ignore[misc]


class TestRun:
    def test_defaults(self) -> None:
        run = Run(repository="pallets/flask", task="Add a health check")

        assert run.id.startswith("run_")
        assert run.status is RunStatus.PENDING
        assert run.branch == f"repopilot/{run.id.split('_', 1)[1][:8]}"
        assert run.started_at is None
        assert run.finished_at is None
        assert run.attempts == 0
        assert run.is_terminal is False
        assert run.duration_seconds is None

    def test_accepts_an_explicit_branch(self) -> None:
        run = Run(repository="pallets/flask", task="t", branch="feat/custom")
        assert run.branch == "feat/custom"

    def test_task_is_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            Run(repository="pallets/flask", task="   ")

    def test_repository_serialises_as_a_slug(self) -> None:
        run = Run(repository="https://github.com/pallets/flask", task="t")
        assert run.model_dump()["repository"] == "pallets/flask"
        assert run.model_dump_json()[:200].count("pallets/flask") == 1

    def test_progress_before_a_plan_exists(self) -> None:
        assert Run(repository="a/b", task="t").progress == 0.0

    def test_progress_tracks_completed_steps(self) -> None:
        run = Run(repository="a/b", task="t", total_steps=4, completed_steps=1)
        assert run.progress == 0.25

    def test_progress_is_one_when_succeeded(self) -> None:
        run = Run(repository="a/b", task="t", total_steps=4, completed_steps=1)
        run.status = RunStatus.SUCCEEDED
        assert run.progress == 1.0

    def test_terminal_statuses_have_no_outgoing_transitions(self) -> None:
        for status in TERMINAL_RUN_STATUSES:
            assert ALLOWED_TRANSITIONS[status] == frozenset()

    def test_pending_can_start_but_not_succeed_directly(self) -> None:
        run = Run(repository="a/b", task="t")

        assert run.can_transition_to(RunStatus.RUNNING) is True
        assert run.can_transition_to(RunStatus.SUCCEEDED) is False

    def test_transition_table_covers_every_status(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(ALL_STATUSES)
        for targets in ALLOWED_TRANSITIONS.values():
            assert targets <= set(ALL_STATUSES)


class TestExecutionEvent:
    def test_defaults(self) -> None:
        event = ExecutionEvent(run_id="run_abc", sequence=1, type=EventType.LOG, message="hello")

        assert event.id.startswith("evt_")
        assert event.step is None
        assert event.progress is None
        assert event.data == {}

    def test_is_frozen(self) -> None:
        event = ExecutionEvent(run_id="run_abc", sequence=1, type=EventType.LOG, message="hello")
        with pytest.raises(ValidationError):
            event.message = "mutated"  # type: ignore[misc]

    def test_sequence_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionEvent(run_id="run_abc", sequence=0, type=EventType.LOG, message="x")

    def test_timestamps_are_timezone_aware(self) -> None:
        event = ExecutionEvent(run_id="run_abc", sequence=1, type=EventType.LOG, message="x")
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.utcoffset().total_seconds() == 0
