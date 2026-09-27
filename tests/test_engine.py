"""Tests for the step pipeline and the run engine.

The GitHub client is mocked at the transport layer and the clone step is pointed at a local
repository. That way the pipeline under test is the real pipeline doing real git and real
filesystem work, without touching the network.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import httpx
import pytest

from repopilot.config import Settings
from repopilot.core.errors import NotFoundError
from repopilot.domain.models import EventLevel, EventType, Run, RunStatus
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.engine import RunEngine
from repopilot.execution.steps import (
    CloneRepositoryStep,
    InspectRepositoryStep,
    ResolveRepositoryStep,
    RunServices,
    Step,
    StepContext,
    StepFailed,
    StepOutcome,
    ValidateRequestStep,
    _inspect_tree,
    default_pipeline,
)
from repopilot.github.client import GitHubClient, RepositoryInfo

REPO_PAYLOAD = {
    "full_name": "acme/widgets",
    "description": "A test repository",
    "default_branch": "main",
    "private": False,
    "archived": False,
    "language": "Python",
    "stargazers_count": 42,
    "open_issues_count": 3,
    "size": 12,
    "license": {"spdx_id": "MIT"},
    "pushed_at": "2026-09-01T10:00:00Z",
    "clone_url": "https://github.com/acme/widgets.git",
    "html_url": "https://github.com/acme/widgets",
}

#: Files the fixture repository really contains, once node_modules is ignored.
FIXTURE_FILE_COUNT = 7
FIXTURE_PYTHON_FILES = 3


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A small but realistic Python repository, committed and ready to clone."""
    root = tmp_path / "source"
    (root / "src" / "widgets").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "node_modules" / "junk").mkdir(parents=True)

    (root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (root / "README.md").write_text("# widgets\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "widgets"\ndependencies = ["pytest"]\n', encoding="utf-8"
    )
    (root / "src" / "widgets" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "widgets" / "core.py").write_text(
        "def run():\n    return 1\n", encoding="utf-8"
    )
    (root / "src" / "widgets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "tests" / "test_core.py").write_text("def test_run():\n    pass\n", encoding="utf-8")
    (root / "node_modules" / "junk" / "big.js").write_text("x" * 10_000, encoding="utf-8")

    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "test@example.com", cwd=root)
    git("config", "user.name", "test", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        cors_origins=["http://testserver"],
    )


def make_engine(
    settings: Settings,
    *,
    steps: tuple[Step, ...] | None = None,
    github_status: int = 200,
) -> RunEngine:
    def handler(request: httpx.Request) -> httpx.Response:
        if github_status != 200:
            return httpx.Response(github_status, json={})
        return httpx.Response(200, json=REPO_PAYLOAD)

    return RunEngine(
        store=RunStore(),
        bus=EventBus(),
        settings=settings,
        github=GitHubClient(transport=httpx.MockTransport(handler)),
        steps=steps,
    )


class LocalCloneStep(CloneRepositoryStep):
    """Clone from a local path instead of a remote. Real git, no network."""

    def __init__(self, source: Path) -> None:
        self.source = source

    async def execute(self, ctx: StepContext) -> StepOutcome:
        info: RepositoryInfo = ctx.state["repository_info"]
        ctx.state["repository_info"] = info.model_copy(update={"clone_url": str(self.source)})
        return await super().execute(ctx)


def full_pipeline(source: Path) -> tuple[Step, ...]:
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        LocalCloneStep(source=source),
        InspectRepositoryStep(),
    )


def make_context(engine: RunEngine, run: Run) -> StepContext:
    """A context whose events are recorded in the store but not faked in memory."""

    async def emit(event_type, message, **kwargs):
        kwargs.setdefault("type", EventType(event_type))
        kwargs.setdefault("level", EventLevel.INFO)
        kwargs.setdefault("step", "test")
        return await engine.store.append_event(run.id, message=message, **kwargs)

    return StepContext(
        run=run,
        store=engine.store,
        services=RunServices(
            github=engine.github, workspace=engine._settings.workspace_dir.resolve()
        ),
        emit=emit,
    )


# ------------------------------------------------------------------ unit: inspect


class TestInspectTree:
    def test_counts_files_and_ignores_noise(self, fake_repo: Path) -> None:
        summary = _inspect_tree(fake_repo)

        assert summary["file_count"] == FIXTURE_FILE_COUNT
        assert summary["languages"] == {"Python": FIXTURE_PYTHON_FILES}
        assert "node_modules" not in summary["top_level"]
        assert "src" in summary["top_level"]
        assert "pytest" in summary["test_frameworks"]
        assert summary["total_bytes"] > 0

    def test_detects_npm_test(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "jest"}, "devDependencies": {"jest": "^29"}}',
            encoding="utf-8",
        )

        summary = _inspect_tree(tmp_path)

        assert {"jest", "npm test"} <= summary["test_frameworks"]

    def test_reports_no_frameworks_when_there_are_none(self, tmp_path: Path) -> None:
        (tmp_path / "main.rs").write_text("fn main() {}\n", encoding="utf-8")

        summary = _inspect_tree(tmp_path)

        assert summary["test_frameworks"] == set()
        assert summary["languages"] == {"Rust": 1}

    def test_tolerates_an_empty_directory(self, tmp_path: Path) -> None:
        summary = _inspect_tree(tmp_path)

        assert summary["file_count"] == 0
        assert summary["languages"] == {}

    def test_counts_javascript_and_typescript_separately(self, tmp_path: Path) -> None:
        (tmp_path / "a.ts").write_text("const a = 1\n", encoding="utf-8")
        (tmp_path / "b.tsx").write_text("export const B = 1\n", encoding="utf-8")
        (tmp_path / "c.js").write_text("const c = 1\n", encoding="utf-8")
        (tmp_path / "d.jsx").write_text("export const D = 1\n", encoding="utf-8")

        summary = _inspect_tree(tmp_path)

        assert summary["languages"] == {"TypeScript": 2, "JavaScript": 2}


class TestPipelineShape:
    def test_pipeline_names_are_stable(self) -> None:
        assert [step.name for step in default_pipeline()] == [
            "validate",
            "resolve",
            "clone",
            "inspect",
        ]

    def test_pipeline_has_no_duplicate_names(self) -> None:
        names = [step.name for step in default_pipeline()]
        assert len(names) == len(set(names))

    def test_every_step_implements_execute(self) -> None:
        for step in default_pipeline():
            assert isinstance(step, Step)
            assert callable(step.execute)


# --------------------------------------------------------------- integration: engine


class TestEngineHappyPath:
    async def test_run_reaches_succeeded(self, settings: Settings, fake_repo: Path) -> None:
        engine = make_engine(settings, steps=full_pipeline(fake_repo))

        run = await engine.start(repository="acme/widgets", task="Add a health check")
        await engine.wait_for(run.id, 60)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.SUCCEEDED
        assert final.error is None
        assert final.started_at is not None
        assert final.finished_at is not None
        assert final.completed_steps == 4
        assert final.total_steps == 4
        assert final.progress == 1.0

    async def test_the_full_event_story_is_recorded(
        self, settings: Settings, fake_repo: Path
    ) -> None:
        engine = make_engine(settings, steps=full_pipeline(fake_repo))

        run = await engine.start(repository="acme/widgets", task="Do a thing")
        await engine.wait_for(run.id, 60)
        await engine.shutdown()

        events = await engine.store.get_events(run.id)
        types = [event.type for event in events]

        assert types[0] is EventType.RUN_CREATED
        assert EventType.RUN_STARTED in types
        assert types[-1] is EventType.RUN_FINISHED
        assert types.count(EventType.STEP_STARTED) == 4
        assert types.count(EventType.STEP_COMPLETED) == 4
        assert EventType.STEP_FAILED not in types
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))

        clone = next(e for e in events if e.step == "clone" and e.type is EventType.STEP_COMPLETED)
        assert clone.data["head"] == await _head_of(settings.workspace_dir / "acme__widgets")

        inspect = next(
            e for e in events if e.step == "inspect" and e.type is EventType.STEP_COMPLETED
        )
        assert inspect.data["file_count"] == FIXTURE_FILE_COUNT
        assert inspect.data["languages"] == {"Python": FIXTURE_PYTHON_FILES}
        assert "pytest" in inspect.data["test_frameworks"]

    async def test_events_carry_increasing_progress(
        self, settings: Settings, fake_repo: Path
    ) -> None:
        engine = make_engine(settings, steps=full_pipeline(fake_repo))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 60)
        await engine.shutdown()

        events = await engine.store.get_events(run.id)
        progresses = [e.progress for e in events if e.step_index is not None]

        assert progresses == sorted(progresses)
        assert progresses[0] == 0.0
        assert max(progresses) == 0.75

    async def test_live_subscribers_see_events_as_they_happen(
        self, settings: Settings, fake_repo: Path
    ) -> None:
        engine = make_engine(settings, steps=full_pipeline(fake_repo))

        async with engine.bus.subscribe() as subscription:
            run = await engine.start(repository="acme/widgets", task="t")
            await engine.wait_for(run.id, 60)
            received = []
            while not subscription.queue.empty():
                received.append(subscription.queue.get_nowait())

        await engine.shutdown()

        assert len(received) >= 6
        assert received[0].type is EventType.RUN_CREATED
        assert received[-1] is RunStatus.SUCCEEDED

    async def test_a_failed_step_marks_the_run_failed(
        self, settings: Settings, fake_repo: Path
    ) -> None:
        class FailingInspect(InspectRepositoryStep):
            async def execute(self, ctx: StepContext) -> StepOutcome:
                raise StepFailed("the clone vanished")

        engine = make_engine(
            settings,
            steps=(
                ValidateRequestStep(),
                ResolveRepositoryStep(),
                LocalCloneStep(source=fake_repo),
                FailingInspect(),
            ),
        )

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 60)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "the clone vanished" in (final.error or "")
        assert final.completed_steps == 3


class TestEngineFailurePaths:
    async def test_github_failure_fails_the_run(self, settings: Settings) -> None:
        engine = make_engine(
            settings,
            steps=(ValidateRequestStep(), ResolveRepositoryStep()),
            github_status=500,
        )

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "resolve" in (final.error or "")

        events = await engine.store.get_events(run.id)
        assert any(e.type is EventType.STEP_FAILED for e in events)
        assert any(e.type is EventType.RUN_FINISHED and e.level is EventLevel.ERROR for e in events)

    async def test_github_404_fails_the_run(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(ResolveRepositoryStep(),), github_status=404)

        run = await engine.start(repository="acme/nope", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "acme/nope" in (final.error or "")

    async def test_cloning_a_missing_source_fails_the_run(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        engine = make_engine(
            settings,
            steps=(
                ValidateRequestStep(),
                ResolveRepositoryStep(),
                LocalCloneStep(source=tmp_path / "does-not-exist"),
            ),
        )

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 60)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "git clone exited" in (final.error or "")

    async def test_an_unexpected_exception_fails_the_run(self, settings: Settings) -> None:
        class ExplodingStep(Step):
            name = "explode"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                raise ValueError("something went very wrong")

        engine = make_engine(settings, steps=(ExplodingStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "ValueError" in (final.error or "")

    async def test_abort_outcome_fails_the_run(self, settings: Settings) -> None:
        class AbortingStep(Step):
            name = "abort"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                return StepOutcome.ABORT

        engine = make_engine(settings, steps=(AbortingStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "aborted" in (final.error or "")

    async def test_an_empty_task_is_rejected_by_the_step(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(ValidateRequestStep(),))
        run = await engine.store.create_run(repository="acme/widgets", task="placeholder")
        blank = run.model_copy(update={"task": "   "}, deep=True)

        with pytest.raises(StepFailed, match="task is empty"):
            await ValidateRequestStep().execute(make_context(engine, blank))

        await engine.shutdown()

    async def test_cloning_before_resolving_is_refused(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(CloneRepositoryStep(),))
        run = await engine.store.create_run(repository="acme/widgets", task="t")
        ctx = make_context(engine, run)

        with pytest.raises(StepFailed, match="resolved"):
            await CloneRepositoryStep().execute(ctx)

        await engine.shutdown()

    async def test_inspecting_without_a_clone_is_refused(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(InspectRepositoryStep(),))
        run = await engine.store.create_run(repository="acme/widgets", task="t")
        ctx = make_context(engine, run)

        with pytest.raises(StepFailed, match="clone step must run first"):
            await InspectRepositoryStep().execute(ctx)

        await engine.shutdown()

    async def test_an_archived_repository_is_refused(
        self, settings: Settings, fake_repo: Path
    ) -> None:
        payload = {**REPO_PAYLOAD, "archived": True}
        github = GitHubClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        )
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=github,
            steps=(ResolveRepositoryStep(), CloneRepositoryStep()),
        )

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "archived" in (final.error or "")


class TestEngineRetryLoop:
    async def test_retryable_failures_are_retried(self, settings: Settings) -> None:
        attempts = {"count": 0}

        class FlakyStep(Step):
            name = "flaky"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise StepFailed("transient network wobble", retryable=True)
                await ctx.emit(EventType.LOG, "recovered on a later attempt")
                return StepOutcome.OK

        engine = make_engine(settings, steps=(FlakyStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert attempts["count"] == 3
        assert final.attempts == 2
        assert final.status is RunStatus.SUCCEEDED

    async def test_retries_are_capped(self, settings: Settings) -> None:
        attempts = {"count": 0}

        class AlwaysFlakyStep(Step):
            name = "always-flaky"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                attempts["count"] += 1
                raise StepFailed("never works", retryable=True)

        engine = make_engine(settings, steps=(AlwaysFlakyStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert attempts["count"] == 3
        assert final.attempts == 2
        assert final.status is RunStatus.FAILED

    async def test_a_retry_resets_step_bookkeeping(self, settings: Settings) -> None:
        seen: list[str] = []

        class RecordingStep(Step):
            name = "record"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                seen.append(ctx.run.current_step or "")
                if len(seen) == 1:
                    raise StepFailed("first try always fails", retryable=True)
                return StepOutcome.OK

        engine = make_engine(settings, steps=(RecordingStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        assert seen == ["record", "record"]
        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED

    async def test_an_explicit_retry_outcome_is_honoured(self, settings: Settings) -> None:
        calls = {"count": 0}

        class AskingStep(Step):
            name = "ask-retry"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                calls["count"] += 1
                return StepOutcome.RETRY if calls["count"] == 1 else StepOutcome.OK

        engine = make_engine(settings, steps=(AskingStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert calls["count"] == 2
        assert final.status is RunStatus.SUCCEEDED


class TestEngineCancellation:
    async def test_cancelling_a_running_run_stops_it(self, settings: Settings) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingStep(Step):
            name = "block"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                started.set()
                while not ctx.is_cancelled():  # noqa: ASYNC110 - polling a cancellation flag
                    await asyncio.sleep(0.01)
                cancelled.set()
                return StepOutcome.OK

        engine = make_engine(settings, steps=(BlockingStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await asyncio.wait_for(started.wait(), 5)

        await engine.cancel(run.id)
        await asyncio.wait_for(cancelled.wait(), 5)
        await engine.wait_for(run.id, 10)
        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.CANCELLED
        assert final.error == "cancelled by user"

    async def test_cancelling_a_finished_run_is_a_noop(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(ValidateRequestStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        await engine.wait_for(run.id, 30)
        await engine.cancel(run.id)
        await engine.shutdown()

        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED

    async def test_cancelling_an_unknown_run_raises(self, settings: Settings) -> None:
        engine = make_engine(settings)
        with pytest.raises(NotFoundError):
            await engine.cancel("run_nope")
        await engine.shutdown()


class TestEngineLifecycle:
    async def test_start_returns_before_the_run_finishes(self, settings: Settings) -> None:
        release = asyncio.Event()
        running = asyncio.Event()

        class SlowStep(Step):
            name = "slow"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                running.set()
                await release.wait()
                return StepOutcome.OK

        engine = make_engine(settings, steps=(SlowStep(),))

        run = await engine.start(repository="acme/widgets", task="t")
        assert run.status is RunStatus.PENDING, "start() must not block on the pipeline"

        await asyncio.wait_for(running.wait(), 5)
        assert run.id in engine.active_runs

        release.set()
        await engine.wait_for(run.id, 10)
        await engine.shutdown()

        assert engine.active_runs == []

    async def test_shutdown_cancels_in_flight_runs(self, settings: Settings) -> None:
        started = asyncio.Event()

        class BlockingStep(Step):
            name = "block"

            async def execute(self, ctx: StepContext) -> StepOutcome:
                started.set()
                await asyncio.sleep(30)
                return StepOutcome.OK

        engine = make_engine(settings, steps=(BlockingStep(),))
        run = await engine.start(repository="acme/widgets", task="t")
        await asyncio.wait_for(started.wait(), 5)

        await engine.shutdown()

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.CANCELLED

    async def test_two_runs_do_not_interfere(self, settings: Settings) -> None:
        engine = make_engine(settings, steps=(ValidateRequestStep(),))

        first = await engine.start(repository="acme/widgets", task="first")
        second = await engine.start(repository="acme/other", task="second")
        await engine.wait_for(first.id, 30)
        await engine.wait_for(second.id, 30)
        await engine.shutdown()

        first_events = await engine.store.get_events(first.id)
        second_events = await engine.store.get_events(second.id)

        assert {e.run_id for e in first_events} == {first.id}
        assert {e.run_id for e in second_events} == {second.id}
        assert [e.sequence for e in first_events] == list(range(1, len(first_events) + 1))
        assert [e.sequence for e in second_events] == list(range(1, len(second_events) + 1))

    async def test_the_default_pipeline_is_used_when_none_is_given(
        self, settings: Settings
    ) -> None:
        engine = make_engine(settings, steps=None)

        assert [step.name for step in engine.steps] == [
            "validate",
            "resolve",
            "clone",
            "inspect",
        ]

        await engine.shutdown()


# ----------------------------------------------------------------------- helpers


async def _head_of(path: Path) -> str | None:
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(path),
        "rev-parse",
        "HEAD",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    return stdout.decode().strip() or None
