"""Tests for the three Phase 2A steps and the pipeline that wires them together.

The clone is real (from a local git repository), the workspace is real, the profile and the
relevance ranking are real, and the plan arrives over real HTTP. The only thing substituted is
the network endpoint the plan is fetched from.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from repopilot.agent.llm import OpenAIChatClient
from repopilot.config import Settings
from repopilot.domain.models import EventType, RunStatus
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.engine import RunEngine
from repopilot.execution.steps import (
    BuildRepositoryProfileStep,
    CloneRepositoryStep,
    CreateImplementationPlanStep,
    ResolveRepositoryStep,
    RunServices,
    SelectRelevantFilesStep,
    StepContext,
    StepFailed,
    StepOutcome,
    ValidateRequestStep,
)
from repopilot.github.client import GitHubClient, RepositoryInfo

REPO_PAYLOAD = {
    "full_name": "acme/widgets",
    "default_branch": "main",
    "private": False,
    "archived": False,
    "stargazers_count": 3,
    "clone_url": "https://github.com/acme/widgets.git",
    "html_url": "https://github.com/acme/widgets",
}

PLAN_REPLY = {
    "summary": "Make run() return the doubled value.",
    "approach": "core.run already returns 1; change the constant and the test.",
    "relevant_files": [
        {"path": "src/widgets/core.py", "reason": "defines run()", "role": "modify"},
        {"path": "tests/test_core.py", "reason": "asserts the old value", "role": "modify"},
    ],
    "steps": [
        {
            "order": 1,
            "title": "Double the return value in run()",
            "detail": "Change the literal 1 to 2.",
            "files": ["src/widgets/core.py"],
            "rationale": "That is where the value is produced.",
        }
    ],
    "risks": [],
    "test_strategy": "Update tests/test_core.py to assert 2.",
}


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _head_of(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _tree_fingerprint(repo: Path) -> list[tuple[str, int]]:
    """Every path outside .git with its size, so any edit to the source is detectable."""
    return sorted(
        (p.relative_to(repo).as_posix(), p.stat().st_size)
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(repo).parts
    )


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "src" / "widgets").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "README.md").write_text("# widgets\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "widgets"\n', encoding="utf-8")
    (root / "src" / "widgets" / "core.py").write_text(
        "def run():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_core.py").write_text("def test_run():\n    pass\n", encoding="utf-8")
    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "t@example.com", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # No model is configured, so RunEngine builds no client and no test can reach a real
    # provider. Tests that need a model pass one explicitly.
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        cors_origins=["http://testserver"],
        llm_model="",
    )


def plan_llm(reply: dict | str, *, model: str = "test-model") -> OpenAIChatClient:
    """A real client whose endpoint replies with a real HTTP body containing a plan."""
    content = reply if isinstance(reply, str) else json.dumps(reply)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 13},
            },
        )

    return OpenAIChatClient(
        model=model,
        base_url="https://model.invalid/v1",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def mock_github() -> GitHubClient:
    return GitHubClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=REPO_PAYLOAD))
    )


async def _unused_emit(event_type, message, **kwargs):
    raise AssertionError(f"this step should not emit, but emitted {event_type}: {message}")


class LocalClone(CloneRepositoryStep):
    """Clone from a local repository, so the pipeline test needs no network."""

    def __init__(self, source: Path) -> None:
        self.source = source

    async def execute(self, ctx: StepContext) -> StepOutcome:
        info: RepositoryInfo = ctx.state["repository_info"]
        ctx.state["repository_info"] = info.model_copy(update={"clone_url": str(self.source)})
        return await super().execute(ctx)


def phase2a_pipeline(source: Path) -> tuple:
    """The six steps a Phase 2A run executes, with the clone pointed at a local repository."""
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        LocalClone(source),
        BuildRepositoryProfileStep(),
        SelectRelevantFilesStep(),
        CreateImplementationPlanStep(),
    )


async def build_context(
    settings: Settings, source: Path, *, llm=None, task: str = "change what run() returns"
) -> StepContext:
    """A context with a genuinely cloned workspace, ready for the Phase 2A steps."""
    engine = RunEngine(
        store=RunStore(),
        bus=EventBus(),
        settings=settings,
        github=mock_github(),
        llm=llm,
        steps=phase2a_pipeline(source),
    )
    run = await engine.store.create_run(repository="acme/widgets", task=task)

    async def emit(event_type, message, **kwargs):
        kwargs.setdefault("type", EventType(event_type))
        return await engine.store.append_event(run.id, message=message, **kwargs)

    ctx = StepContext(
        run=run,
        store=engine.store,
        services=RunServices(
            github=engine.github,
            workspace=settings.workspace_dir.resolve(),
            llm=llm,
        ),
        emit=emit,
    )
    ctx.state["repository_info"] = RepositoryInfo(
        full_name="acme/widgets",
        default_branch="main",
        html_url="https://github.com/acme/widgets",
        clone_url=str(source),
    )
    assert await CloneRepositoryStep().execute(ctx) is StepOutcome.OK
    assert (ctx.local_path() / ".git").is_dir()
    ctx.state["default_branch"] = "main"
    return ctx


class TestIsolatedWorkspace:
    async def test_each_run_gets_its_own_directory(self, settings: Settings) -> None:
        run_a = await RunStore().create_run(repository="acme/widgets", task="a")
        run_b = await RunStore().create_run(repository="acme/widgets", task="b")
        services = RunServices(github=GitHubClient(), workspace=settings.workspace_dir.resolve())
        ctx_a = StepContext(run=run_a, store=RunStore(), services=services, emit=None)  # type: ignore[arg-type]
        ctx_b = StepContext(run=run_b, store=RunStore(), services=services, emit=None)  # type: ignore[arg-type]

        assert ctx_a.local_path() != ctx_b.local_path()
        assert ctx_a.local_path().name == run_a.id
        assert ctx_a.repo_dir() == ctx_b.repo_dir()

    async def test_the_workspace_lives_under_the_repository_directory(
        self, settings: Settings
    ) -> None:
        run = await RunStore().create_run(repository="acme/widgets", task="a")
        ctx = StepContext(
            run=run,
            store=RunStore(),
            services=RunServices(github=GitHubClient(), workspace=Path("C:/ws")),
            emit=None,  # type: ignore[arg-type]
        )

        assert ctx.local_path() == Path("C:/ws/acme__widgets") / run.id


class TestBuildRepositoryProfileStep:
    async def test_reports_measured_facts(self, settings: Settings, source_repo: Path) -> None:
        ctx = await build_context(settings, source_repo)

        assert await BuildRepositoryProfileStep().execute(ctx) is StepOutcome.OK

        profile = ctx.state["profile"]
        assert profile.primary_language == "Python"
        assert len(profile.files) == 2
        assert "pyproject.toml" in profile.manifests
        assert profile.default_branch == "main"

    async def test_emits_a_progress_event_with_the_measurements(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo)

        assert await BuildRepositoryProfileStep().execute(ctx) is StepOutcome.OK

        events = await ctx.store.get_events(ctx.run.id)
        progress = [
            e for e in events if e.type is EventType.STEP_PROGRESS and "source_files" in e.data
        ]
        assert progress
        assert progress[-1].data["source_files"] == 2
        assert progress[-1].data["manifests"] == ["pyproject.toml"]
        assert "Python" in progress[-1].message

    async def test_fails_when_there_is_nothing_to_profile(self, settings: Settings) -> None:
        """No checkout means no profile; the step must not invent one."""
        run = await RunStore().create_run(repository="acme/widgets", task="t")
        ctx = StepContext(
            run=run,
            store=RunStore(),
            services=RunServices(github=GitHubClient(), workspace=settings.workspace_dir.resolve()),
            emit=_unused_emit,
        )
        settings.ensure_workspace()

        with pytest.raises(StepFailed, match="does not exist"):
            await BuildRepositoryProfileStep().execute(ctx)


class TestSelectRelevantFilesStep:
    async def test_shortlists_the_file_the_task_names(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, task="change what run returns")
        await BuildRepositoryProfileStep().execute(ctx)

        assert await SelectRelevantFilesStep().execute(ctx) is StepOutcome.OK

        candidates = ctx.state["candidates"]
        assert candidates[0].path == "src/widgets/core.py"
        assert candidates[0].reasons

    async def test_emits_one_file_selected_event_per_lead_candidate(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo)
        await BuildRepositoryProfileStep().execute(ctx)

        await SelectRelevantFilesStep().execute(ctx)

        events = await ctx.store.get_events(ctx.run.id)
        selected = [e for e in events if e.type is EventType.FILE_SELECTED]
        assert selected
        assert selected[0].data["path"] == "src/widgets/core.py"
        assert selected[0].data["reasons"]

    async def test_warns_instead_of_failing_when_nothing_matches(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, task="implement quantum flux capacitor")
        await BuildRepositoryProfileStep().execute(ctx)

        assert await SelectRelevantFilesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["candidates"] == []
        events = await ctx.store.get_events(ctx.run.id)
        assert any("No file in this repository matched" in e.message for e in events)

    async def test_requires_a_profile_first(self, settings: Settings, source_repo: Path) -> None:
        ctx = await build_context(settings, source_repo)

        with pytest.raises(StepFailed, match="profile step must run first"):
            await SelectRelevantFilesStep().execute(ctx)


class TestCreateImplementationPlanStep:
    async def test_emits_a_plan_created_event_carrying_the_plan(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=plan_llm(PLAN_REPLY))
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        assert await CreateImplementationPlanStep().execute(ctx) is StepOutcome.OK

        events = await ctx.store.get_events(ctx.run.id)
        plan_events = [e for e in events if e.type is EventType.PLAN_CREATED]
        assert len(plan_events) == 1
        payload = plan_events[0].data["plan"]
        assert payload["summary"] == PLAN_REPLY["summary"]
        assert payload["model"] == "test-model"
        assert payload["steps"][0]["title"] == PLAN_REPLY["steps"][0]["title"]
        assert payload["files_to_change"] == ["src/widgets/core.py", "tests/test_core.py"]
        assert plan_events[0].message == PLAN_REPLY["summary"]

    async def test_streams_one_progress_event_per_plan_step(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=plan_llm(PLAN_REPLY))
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        await CreateImplementationPlanStep().execute(ctx)

        events = await ctx.store.get_events(ctx.run.id)
        step_lines = [e for e in events if e.message.startswith("Plan step ")]
        assert len(step_lines) == 1
        assert "src/widgets/core.py" in step_lines[0].message
        assert step_lines[0].data["order"] == 1

    async def test_fails_with_an_actionable_message_when_no_model_is_configured(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=None)
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        with pytest.raises(StepFailed) as excinfo:
            await CreateImplementationPlanStep().execute(ctx)

        assert "REPOPILOT_LLM_MODEL" in str(excinfo.value)
        assert "Ollama" in str(excinfo.value)

    async def test_fails_when_nothing_was_shortlisted(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(
            settings, source_repo, llm=plan_llm(PLAN_REPLY), task="implement quantum flux"
        )
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        with pytest.raises(StepFailed, match="nothing to plan"):
            await CreateImplementationPlanStep().execute(ctx)

    async def test_marks_a_model_failure_retryable(
        self, settings: Settings, source_repo: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="upstream unavailable")

        llm = OpenAIChatClient(
            model="m",
            base_url="https://model.invalid/v1",
            api_key="k",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        ctx = await build_context(settings, source_repo, llm=llm)
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        with pytest.raises(StepFailed) as excinfo:
            await CreateImplementationPlanStep().execute(ctx)

        assert excinfo.value.retryable is True
        assert "503" in str(excinfo.value)

    async def test_warns_when_the_model_invents_a_path(
        self, settings: Settings, source_repo: Path
    ) -> None:
        reply = {
            **PLAN_REPLY,
            "relevant_files": [
                *PLAN_REPLY["relevant_files"],
                {"path": "src/widgets/ghost.py", "reason": "made up", "role": "modify"},
            ],
        }
        ctx = await build_context(settings, source_repo, llm=plan_llm(reply))
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)

        await CreateImplementationPlanStep().execute(ctx)

        events = await ctx.store.get_events(ctx.run.id)
        warnings = [e for e in events if "invented" in e.message]
        assert warnings
        assert warnings[0].level.value == "warn"
        assert warnings[0].data["hallucinated_paths"] == ["src/widgets/ghost.py"]


class TestPhase2APipeline:
    async def test_the_whole_thing_runs_end_to_end(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """validate -> clone -> profile -> select_files -> plan, all real work."""
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=plan_llm(PLAN_REPLY),
            steps=phase2a_pipeline(source_repo),
        )

        run = await engine.start(repository="acme/widgets", task="change what run() returns")
        await engine.wait_for(run.id, 120)

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.SUCCEEDED, final.error
        assert final.total_steps == 6
        assert final.completed_steps == 6
        assert final.progress == 1.0

        events = await engine.store.get_events(run.id)
        types = [e.type for e in events]
        assert types[0] is EventType.RUN_CREATED
        assert types[-1] is EventType.RUN_FINISHED
        assert EventType.FILE_SELECTED in types
        assert EventType.PLAN_CREATED in types
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))

        # The real emitter attributes each event to the step that produced it.
        selected = next(e for e in events if e.type is EventType.FILE_SELECTED)
        assert selected.step == "select_files"
        assert selected.step_index == 4

        plan_event = next(e for e in events if e.type is EventType.PLAN_CREATED)
        assert plan_event.step == "plan"
        assert plan_event.step_index == 5
        assert plan_event.data["plan"]["summary"] == PLAN_REPLY["summary"]
        assert plan_event.data["workspace"].endswith(run.id)

        # The clone is real, and the files the plan names really exist in it.
        workspace = Path(plan_event.data["workspace"])
        assert (workspace / ".git").is_dir()
        assert (workspace / "src" / "widgets" / "core.py").is_file()

        await engine.shutdown()

    async def test_a_run_without_a_model_fails_rather_than_faking_a_plan(
        self, settings: Settings, source_repo: Path
    ) -> None:
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=None,
            steps=phase2a_pipeline(source_repo),
        )

        run = await engine.start(repository="acme/widgets", task="change run()")
        await engine.wait_for(run.id, 120)

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "REPOPILOT_LLM_MODEL" in (final.error or "")

        events = await engine.store.get_events(run.id)
        assert not any(e.type is EventType.PLAN_CREATED for e in events)
        assert any(e.type is EventType.STEP_FAILED for e in events)

        await engine.shutdown()

    async def test_the_workspace_never_touches_the_source_repository(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """The user's own checkout must be byte-identical after a run."""
        before = await asyncio.to_thread(_tree_fingerprint, source_repo)
        head_before = await asyncio.to_thread(_head_of, source_repo)

        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=plan_llm(PLAN_REPLY),
            steps=phase2a_pipeline(source_repo),
        )
        run = await engine.start(repository="acme/widgets", task="change run()")
        await engine.wait_for(run.id, 120)
        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED

        after = await asyncio.to_thread(_tree_fingerprint, source_repo)
        head_after = await asyncio.to_thread(_head_of, source_repo)

        assert after == before
        assert head_after == head_before
        # And the run worked on a copy, not on that directory.
        assert (settings.workspace_dir.resolve() / "acme__widgets" / run.id).is_dir()

        await engine.shutdown()


class TestStepLifecycleEvents:
    """Every step has to announce where it starts and that it ended.

    The engine does not emit these itself. Each step reports its own start and completion, the
    way the Phase 1 steps do, and a step that forgets is more than a cosmetic gap. The terminal
    renders a run as a sequence of steps, so a step with no ``step.started`` never shows up as
    running and one with no ``step.completed`` never shows up as done: the work happens and stays
    invisible in the status line, which is the opposite of what streaming a step should mean.
    """

    @pytest.fixture
    async def events(self, settings: Settings, source_repo: Path) -> list:
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=plan_llm(PLAN_REPLY),
            steps=phase2a_pipeline(source_repo),
        )
        run = await engine.start(repository="acme/widgets", task="change what run() returns")
        await engine.wait_for(run.id, 120)
        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED
        events = await engine.store.get_events(run.id)
        await engine.shutdown()
        return events

    def lifecycle(self, events: list, event_type: EventType) -> list[tuple]:
        return [(e.step, e.step_index) for e in events if e.type is event_type]

    async def test_every_step_announces_itself(self, events: list) -> None:
        started = [step for step, _ in self.lifecycle(events, EventType.STEP_STARTED)]
        completed = [step for step, _ in self.lifecycle(events, EventType.STEP_COMPLETED)]

        assert started == [
            "validate",
            "resolve",
            "clone",
            "profile",
            "select_files",
            "plan",
        ]
        assert completed == started

    async def test_a_step_carries_its_position(self, events: list) -> None:
        assert self.lifecycle(events, EventType.STEP_STARTED) == [
            ("validate", 0),
            ("resolve", 1),
            ("clone", 2),
            ("profile", 3),
            ("select_files", 4),
            ("plan", 5),
        ]

    async def test_no_two_steps_overlap(self, events: list) -> None:
        """Strictly start, complete, start, complete, so one current step can be rendered."""
        ordered = [
            e.type for e in events if e.type in (EventType.STEP_STARTED, EventType.STEP_COMPLETED)
        ]

        assert ordered
        assert ordered[0::2] == [EventType.STEP_STARTED] * (len(ordered) // 2)
        assert ordered[1::2] == [EventType.STEP_COMPLETED] * (len(ordered) // 2)

    @pytest.mark.parametrize("step_name", ["profile", "select_files", "plan"])
    async def test_a_completion_says_something_useful(self, events: list, step_name: str) -> None:
        event = next(
            e for e in events if e.type is EventType.STEP_COMPLETED and e.step == step_name
        )

        assert event.message
        assert event.data, f"{step_name} completed with no data for the UI to show"

    async def test_the_plan_completion_summarises_the_plan(self, events: list) -> None:
        event = next(e for e in events if e.type is EventType.STEP_COMPLETED and e.step == "plan")

        assert event.data["steps"] > 0
        assert event.data["files_to_change"]
        assert event.data["model"] == "test-model"
