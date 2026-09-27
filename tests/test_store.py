"""Tests for the in-memory run store.

The store is the only owner of run state, so these tests focus on the two guarantees the rest
of the system depends on: gap-free event ordering, and isolation of stored objects.
"""

from __future__ import annotations

import pytest

from repopilot.core.errors import InvalidTransitionError, NotFoundError
from repopilot.domain.models import EventLevel, EventType, Run, RunStatus
from repopilot.domain.store import RunStore


@pytest.fixture
def store() -> RunStore:
    return RunStore()


async def make_run(store: RunStore, **kwargs) -> Run:
    return await store.create_run(
        repository=kwargs.pop("repository", "pallets/flask"),
        task=kwargs.pop("task", "Add a health check"),
        **kwargs,
    )


class TestCreateAndRead:
    async def test_create_returns_a_pending_run(self, store: RunStore) -> None:
        run = await make_run(store)

        assert run.status is RunStatus.PENDING
        assert run.repository.slug == "pallets/flask"
        assert run.id.startswith("run_")

    async def test_get_run_returns_a_copy(self, store: RunStore) -> None:
        created = await make_run(store)

        fetched = await store.get_run(created.id)
        assert fetched is not None
        assert fetched.id == created.id

        fetched.task = "mutated by the caller"
        refetched = await store.require_run(created.id)
        assert refetched.task == "Add a health check"

    async def test_get_run_returns_none_when_missing(self, store: RunStore) -> None:
        assert await store.get_run("run_nope") is None

    async def test_require_run_raises_when_missing(self, store: RunStore) -> None:
        with pytest.raises(NotFoundError):
            await store.require_run("run_nope")

    async def test_ids_are_unique(self, store: RunStore) -> None:
        ids = {(await make_run(store)).id for _ in range(50)}
        assert len(ids) == 50

    async def test_list_runs_is_newest_first(self, store: RunStore) -> None:
        first = await make_run(store)
        second = await make_run(store)
        third = await make_run(store)

        assert [r.id for r in await store.list_runs()] == [
            third.id,
            second.id,
            first.id,
        ]

    async def test_list_runs_filters_by_status(self, store: RunStore) -> None:
        await make_run(store)
        running = await make_run(store)
        await store.transition(running.id, RunStatus.RUNNING)

        assert [r.id for r in await store.list_runs(status=RunStatus.RUNNING)] == [running.id]
        assert len(await store.list_runs(status=RunStatus.PENDING)) == 1

    async def test_list_runs_honours_limit(self, store: RunStore) -> None:
        for _ in range(5):
            await make_run(store)

        assert len(await store.list_runs(limit=3)) == 3

    async def test_counts_by_status(self, store: RunStore) -> None:
        await make_run(store)
        done = await make_run(store)
        await store.transition(done.id, RunStatus.RUNNING)
        await store.transition(done.id, RunStatus.SUCCEEDED)

        counts = await store.counts_by_status()
        assert counts == {
            "pending": 1,
            "running": 0,
            "succeeded": 1,
            "failed": 0,
            "cancelled": 0,
        }


class TestTransitions:
    async def test_running_stamps_started_at(self, store: RunStore) -> None:
        run = await make_run(store)
        assert run.started_at is None

        running = await store.transition(run.id, RunStatus.RUNNING)

        assert running.status is RunStatus.RUNNING
        assert running.started_at is not None
        assert running.finished_at is None

    async def test_terminal_stamps_finished_at_and_clears_step(self, store: RunStore) -> None:
        run = await make_run(store)
        await store.transition(
            run.id, RunStatus.RUNNING, current_step="clone", current_step_index=0
        )

        finished = await store.transition(run.id, RunStatus.SUCCEEDED, error=None)

        assert finished.finished_at is not None
        assert finished.current_step is None
        assert finished.current_step_index is None
        assert finished.is_terminal is True
        assert finished.duration_seconds is not None

    async def test_failure_records_the_error(self, store: RunStore) -> None:
        run = await make_run(store)
        await store.transition(run.id, RunStatus.RUNNING)

        failed = await store.transition(run.id, RunStatus.FAILED, error="tests failed on step 3")

        assert failed.error == "tests failed on step 3"

    async def test_illegal_transition_is_rejected(self, store: RunStore) -> None:
        run = await make_run(store)
        await store.transition(run.id, RunStatus.RUNNING)
        await store.transition(run.id, RunStatus.SUCCEEDED)

        with pytest.raises(InvalidTransitionError) as excinfo:
            await store.transition(run.id, RunStatus.RUNNING)

        assert excinfo.value.current == "succeeded"
        assert excinfo.value.requested == "running"

    async def test_cannot_skip_running(self, store: RunStore) -> None:
        run = await make_run(store)
        with pytest.raises(InvalidTransitionError):
            await store.transition(run.id, RunStatus.SUCCEEDED)

    async def test_pending_can_be_cancelled(self, store: RunStore) -> None:
        run = await make_run(store)
        cancelled = await store.transition(run.id, RunStatus.CANCELLED)
        assert cancelled.status is RunStatus.CANCELLED

    async def test_update_run_also_enforces_transitions(self, store: RunStore) -> None:
        run = await make_run(store)
        await store.transition(run.id, RunStatus.RUNNING)
        await store.transition(run.id, RunStatus.FAILED)

        with pytest.raises(InvalidTransitionError):
            await store.update_run(run.id, status=RunStatus.RUNNING)

    async def test_update_run_patches_other_fields(self, store: RunStore) -> None:
        run = await make_run(store)

        updated = await store.update_run(
            run.id, total_steps=6, completed_steps=2, current_step="plan"
        )

        assert (updated.total_steps, updated.completed_steps) == (6, 2)
        assert updated.current_step == "plan"

    async def test_transition_on_missing_run_raises(self, store: RunStore) -> None:
        with pytest.raises(NotFoundError):
            await store.transition("run_nope", RunStatus.RUNNING)


class TestEvents:
    async def test_sequences_start_at_one_and_are_gap_free(self, store: RunStore) -> None:
        run = await make_run(store)

        for i in range(5):
            event = await store.append_event(run.id, type=EventType.LOG, message=f"line {i}")
            assert event.sequence == i + 1

        assert await store.latest_sequence(run.id) == 5

    async def test_sequences_are_per_run(self, store: RunStore) -> None:
        a = await make_run(store)
        b = await make_run(store)

        await store.append_event(a.id, type=EventType.LOG, message="a1")
        await store.append_event(a.id, type=EventType.LOG, message="a2")
        event = await store.append_event(b.id, type=EventType.LOG, message="b1")

        assert event.sequence == 1
        assert len(await store.get_events(a.id)) == 2

    async def test_get_events_supports_a_cursor(self, store: RunStore) -> None:
        run = await make_run(store)
        for i in range(5):
            await store.append_event(run.id, type=EventType.LOG, message=f"line {i}")

        tail = await store.get_events(run.id, after_sequence=3)

        assert [e.sequence for e in tail] == [4, 5]
        assert [e.message for e in tail] == ["line 3", "line 4"]

    async def test_get_events_supports_a_limit(self, store: RunStore) -> None:
        run = await make_run(store)
        for i in range(5):
            await store.append_event(run.id, type=EventType.LOG, message=str(i))

        assert len(await store.get_events(run.id, limit=2)) == 2

    async def test_event_carries_optional_context(self, store: RunStore) -> None:
        run = await make_run(store)

        event = await store.append_event(
            run.id,
            type=EventType.STEP_PROGRESS,
            message="cloning",
            level=EventLevel.DEBUG,
            step="clone",
            step_index=0,
            progress=0.25,
            data={"bytes": 1024},
        )

        assert event.type is EventType.STEP_PROGRESS
        assert event.level is EventLevel.DEBUG
        assert event.step == "clone"
        assert event.step_index == 0
        assert event.progress == 0.25
        assert event.data == {"bytes": 1024}

    async def test_events_for_a_missing_run_raise(self, store: RunStore) -> None:
        with pytest.raises(NotFoundError):
            await store.append_event("run_nope", type=EventType.LOG, message="x")

    async def test_history_is_capped(self) -> None:
        store = RunStore(max_events_per_run=10)
        run = await make_run(store)
        for i in range(25):
            await store.append_event(run.id, type=EventType.LOG, message=str(i))

        events = await store.get_events(run.id)

        assert len(events) == 10
        assert events[0].sequence == 16
        assert await store.latest_sequence(run.id) == 25


class TestPurge:
    async def test_purge_removes_only_terminal_runs(self, store: RunStore) -> None:
        active = await make_run(store)
        done = await make_run(store)
        await store.transition(done.id, RunStatus.RUNNING)
        await store.transition(done.id, RunStatus.SUCCEEDED)

        assert await store.purge_terminal() == 1
        assert await store.get_run(active.id) is not None
        assert await store.get_run(done.id) is None

    async def test_purged_run_rejects_events(self, store: RunStore) -> None:
        run = await make_run(store)
        await store.transition(run.id, RunStatus.RUNNING)
        await store.transition(run.id, RunStatus.FAILED)
        await store.purge_terminal()

        with pytest.raises(NotFoundError):
            await store.append_event(run.id, type=EventType.LOG, message="x")
