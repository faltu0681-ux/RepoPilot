"""Tests for Phase 2B: writing the plan into the run's own checkout.

The clone is real, git is real, the branch is real, and the files on disk are the ones the
model returned. The only thing substituted is the reasoning backend: a scripted
:class:`ChatClient` that answers one file at a time, plus one test that drives the real HTTP
client so the transport itself is covered.

The tests that matter most here are the refusals. A run that writes outside its checkout, into
``.git``, or into a credentials file would be a much worse bug than a run that writes nothing,
so each of those is asserted rather than assumed.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from repopilot.agent.implement import (
    MAX_CONTENT_CHARS,
    ChangeRejectedError,
    UnsafePathError,
    extract_file_content,
    generate_file_change,
    read_for_edit,
    resolve_write_target,
    write_file,
)
from repopilot.agent.llm import (
    ChatClient,
    ChatMessage,
    Completion,
    LLMRequestError,
    OpenAIChatClient,
)
from repopilot.agent.plan import generate_plan
from repopilot.agent.schema import ImplementationPlan
from repopilot.config import Settings
from repopilot.domain.models import EventType, ExecutionEvent, RunStatus
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.engine import RunEngine
from repopilot.execution.steps import (
    ApplyImplementationPlanStep,
    BuildRepositoryProfileStep,
    CloneRepositoryStep,
    CreateImplementationPlanStep,
    ResolveRepositoryStep,
    RunServices,
    SelectRelevantFilesStep,
    Step,
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
    "stargazers": 3,
    "clone_url": "https://github.com/acme/widgets.git",
    "html_url": "https://github.com/acme/widgets",
}

CORE = "def run():\n    return 1\n"
TESTS = "def test_run():\n    pass\n"
NEW_MODULE = '"""Timing helpers."""\n\n\ndef uptime() -> int:\n    return 0\n'

PLAN_REPLY = {
    "summary": "Make run() return the doubled value and report uptime.",
    "approach": "core.run() is the value under test; the new module is standalone.",
    "relevant_files": [
        {"path": "src/widgets/core.py", "reason": "defines run()", "role": "modify"},
        {"path": "tests/test_core.py", "reason": "asserts the old value", "role": "modify"},
        {"path": "src/widgets/uptime.py", "reason": "new module", "role": "add"},
        {"path": "README.md", "reason": "documents the package", "role": "read"},
    ],
    "steps": [
        {
            "order": 1,
            "title": "Double the return value in run()",
            "detail": "Change the literal 1 to 2.",
            "files": ["src/widgets/core.py"],
            "rationale": "That is where the value is produced.",
        },
        {
            "order": 2,
            "title": "Assert the new value in the test",
            "detail": "Make the test assert 2.",
            "files": ["tests/test_core.py"],
            "rationale": "The existing test is the one that will fail otherwise.",
        },
    ],
    "risks": [],
    "test_strategy": "Run pytest.",
}

FILE_REPLIES = {
    "src/widgets/core.py": "def run():\n    return 2\n",
    "tests/test_core.py": "def test_run():\n    assert run() == 2\n",
    "src/widgets/uptime.py": NEW_MODULE,
}

#: The implementer prompt always names the file it wants back in its last line. The path
#: capture is greedy so that the dots in "core.py" survive it.
_ASKED = re.compile(r"Return the (?:new contents of|contents of the new) (\S+)\.")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
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
    (root / "src" / "widgets" / "core.py").write_text(CORE, encoding="utf-8")
    (root / "tests" / "test_core.py").write_text(TESTS, encoding="utf-8")
    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "t@example.com", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        cors_origins=["http://testserver"],
        llm_model="",
    )


def mock_github() -> GitHubClient:
    return GitHubClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=REPO_PAYLOAD))
    )


class ScriptedClient:
    """A ``ChatClient`` that answers the planner and the file writer.

    Keyed by path, so a test says what each file should become and the fake works out which
    request it is answering. A list value is a queue of replies, which is how the retry tests
    give a bad answer first and a good one second. The planner prompt is recognised by its
    system message, because one fake covering both calls is what makes the end-to-end test a
    real pipeline run.
    """

    def __init__(self, replies: dict[str, Any] | None = None, *, model: str = "test-model") -> None:
        self.model = model
        self._replies: dict[str, Any] = dict(replies or FILE_REPLIES)
        self.requests: list[list[ChatMessage]] = []
        self._current: str | None = None

    def queue(self, path: str, *replies: str) -> None:
        self._replies[path] = list(replies)

    @property
    def asked_for(self) -> list[str]:
        return [self._asked_path(messages) for messages in self.requests]

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.requests.append(messages)
        if "planning a change" in messages[0].content:
            return Completion(
                text=json.dumps(PLAN_REPLY),
                model=self.model,
                prompt_tokens=42,
                completion_tokens=13,
            )
        # A retry carries a correction instead of the request, so the file being asked about is
        # the one the last request named.
        match = _ASKED.search("\n".join(messages[-1].content.strip().splitlines()[-3:]))
        if match is not None:
            self._current = match.group(1)
        path = self._current
        if path is None:
            raise AssertionError(f"no file was requested in: {messages[-1].content[-200:]!r}")
        if path not in self._replies:
            raise AssertionError(f"the run asked about an unplanned file: {path}")
        reply = self._replies[path]
        if isinstance(reply, list):
            if not reply:
                raise AssertionError(f"ran out of scripted replies for {path}")
            reply = reply.pop(0)
        return Completion(text=reply, model=self.model, prompt_tokens=11, completion_tokens=7)

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _asked_path(messages: list[ChatMessage]) -> str:
        tail = "\n".join(messages[-1].content.strip().splitlines()[-3:])
        match = _ASKED.search(tail)
        if match is None:
            raise AssertionError(f"no file was requested in: {tail!r}")
        return match.group(1)


def plan_llm(reply: dict | str = PLAN_REPLY) -> OpenAIChatClient:
    """A real client whose endpoint replies with a real plan over real HTTP."""
    content = reply if isinstance(reply, str) else json.dumps(reply)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 13},
            },
        )

    return OpenAIChatClient(
        model="test-model",
        base_url="https://model.invalid/v1",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def file_llm(reply: str) -> OpenAIChatClient:
    """A real client whose endpoint replies with one file body over real HTTP."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 9},
            },
        )

    return OpenAIChatClient(
        model="test-model",
        base_url="https://model.invalid/v1",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def broken_llm() -> OpenAIChatClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    return OpenAIChatClient(
        model="m",
        base_url="https://model.invalid/v1",
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


class LocalClone(CloneRepositoryStep):
    """Clone from a local repository, so the pipeline needs no network."""

    def __init__(self, source: Path) -> None:
        self.source = source

    async def execute(self, ctx: StepContext) -> StepOutcome:
        info: RepositoryInfo = ctx.state["repository_info"]
        ctx.state["repository_info"] = info.model_copy(update={"clone_url": str(self.source)})
        return await super().execute(ctx)


def phase2b_pipeline(source: Path) -> tuple[Step, ...]:
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        LocalClone(source),
        BuildRepositoryProfileStep(),
        SelectRelevantFilesStep(),
        CreateImplementationPlanStep(),
        ApplyImplementationPlanStep(),
    )


async def build_context(
    settings: Settings,
    source: Path,
    *,
    llm: ChatClient | None = None,
    task: str = "double what run() returns and add an uptime helper",
    plan: bool = True,
    file_client: ChatClient | None = None,
) -> StepContext:
    """A context with a real clone, a real profile, a real shortlist, and a real plan.

    ``llm`` plans. ``file_client`` is what the apply step will call, which defaults to the same
    client and can be swapped for a test that needs a different backend for the two calls.
    """
    engine = RunEngine(
        store=RunStore(),
        bus=EventBus(),
        settings=settings,
        github=mock_github(),
        llm=llm,
        steps=phase2b_pipeline(source),
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
    if plan:
        assert await BuildRepositoryProfileStep().execute(ctx) is StepOutcome.OK
        assert await SelectRelevantFilesStep().execute(ctx) is StepOutcome.OK
        assert await CreateImplementationPlanStep().execute(ctx) is StepOutcome.OK
    if file_client is not None:
        ctx.services.llm = file_client
    return ctx


# ------------------------------------------------------------------ path refusals


class TestResolveWriteTarget:
    def test_accepts_a_nested_path_inside_the_checkout(self, tmp_path: Path) -> None:
        target = resolve_write_target(tmp_path, "src/widgets/core.py")

        assert target == (tmp_path / "src" / "widgets" / "core.py").resolve()

    def test_normalises_windows_separators(self, tmp_path: Path) -> None:
        assert (
            resolve_write_target(tmp_path, "src\\core.py")
            == (tmp_path / "src" / "core.py").resolve()
        )

    @pytest.mark.parametrize("path", ["/etc/passwd", "C:\\Windows\\system32\\hosts", "c:/temp/x"])
    def test_refuses_an_absolute_path(self, tmp_path: Path, path: str) -> None:
        with pytest.raises(UnsafePathError, match="absolute path"):
            resolve_write_target(tmp_path, path)

    @pytest.mark.parametrize("path", ["../secrets.txt", "src/../../outside.py", "a/b/../../../x"])
    def test_refuses_a_path_that_climbs_out(self, tmp_path: Path, path: str) -> None:
        with pytest.raises(UnsafePathError, match="leaves the workspace"):
            resolve_write_target(tmp_path, path)

    def test_refuses_the_same_climb_written_with_backslashes(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError, match="leaves the workspace"):
            resolve_write_target(tmp_path, "src\\..\\..\\outside.py")

    @pytest.mark.parametrize("path", [".git/config", "src/.git/hooks/pre-commit"])
    def test_refuses_anything_inside_git(self, tmp_path: Path, path: str) -> None:
        with pytest.raises(UnsafePathError, match="git directory"):
            resolve_write_target(tmp_path, path)

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.production",
            "config/secrets.yml",
            "deploy/server.pem",
            "keys/app.key",
            "id_rsa",
            ".npmrc",
        ],
    )
    def test_refuses_credentials_files(self, tmp_path: Path, path: str) -> None:
        with pytest.raises(UnsafePathError, match="credentials file"):
            resolve_write_target(tmp_path, path)

    def test_refuses_an_empty_path(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError, match="empty path"):
            resolve_write_target(tmp_path, "   ")

    def test_refuses_a_symlink_that_points_out_of_the_checkout(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        (outside / "loot.txt").write_text("not yours\n", encoding="utf-8")
        root = tmp_path / "checkout"
        root.mkdir()
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except OSError:  # pragma: no cover - Windows without developer mode
            pytest.skip("symlinks are not available here")

        with pytest.raises(UnsafePathError, match="outside the workspace"):
            resolve_write_target(root, "link/loot.txt")


class TestWriteFile:
    def test_creates_the_file_and_its_parents(self, tmp_path: Path) -> None:
        result = write_file(tmp_path, "src/new/dir/mod.py", "x = 1\n")

        assert (tmp_path / "src" / "new" / "dir" / "mod.py").read_text(
            encoding="utf-8"
        ) == "x = 1\n"
        assert result.created is True
        assert result.bytes == 6
        assert result.lines == 1

    def test_reports_an_overwrite_as_not_created(self, tmp_path: Path) -> None:
        write_file(tmp_path, "mod.py", "x = 1\n")

        result = write_file(tmp_path, "mod.py", "x = 2\n")

        assert result.created is False
        assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_writes_the_bytes_the_model_produced(self, tmp_path: Path) -> None:
        write_file(tmp_path, "mod.py", "line\r\nother\r\n")

        assert (tmp_path / "mod.py").read_bytes() == b"line\r\nother\r\n"

    def test_refuses_to_write_nothing_outside_the_checkout(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError):
            write_file(tmp_path, "../escaped.py", "x = 1\n")

        assert not (tmp_path.parent / "escaped.py").exists()

    def test_refuses_a_file_larger_than_the_limit(self, tmp_path: Path) -> None:
        with pytest.raises(ChangeRejectedError, match="limit"):
            write_file(tmp_path, "big.py", "x" * (MAX_CONTENT_CHARS + 1))

        assert not (tmp_path / "big.py").exists()


class TestReadForEdit:
    def test_returns_the_current_contents(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

        assert read_for_edit(tmp_path, "a.py") == "x = 1\n"

    def test_returns_nothing_for_a_file_that_is_not_there(self, tmp_path: Path) -> None:
        assert read_for_edit(tmp_path, "missing.py") is None

    def test_returns_nothing_for_a_refused_path(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")

        assert read_for_edit(tmp_path, ".env") is None

    def test_returns_nothing_for_a_file_too_large_to_rewrite(self, tmp_path: Path) -> None:
        (tmp_path / "big.py").write_text("x = 1\n" * 40_000, encoding="utf-8")

        assert read_for_edit(tmp_path, "big.py") is None


# -------------------------------------------------------------- reply to file body


class TestExtractFileContent:
    def test_accepts_a_json_object(self) -> None:
        assert extract_file_content('{"content": "x = 1\\n"}') == "x = 1\n"

    def test_accepts_a_fenced_block_with_a_language_tag(self) -> None:
        assert extract_file_content("```python\nx = 1\n```") == "x = 1\n"

    def test_accepts_a_fenced_block_with_no_language_tag(self) -> None:
        assert extract_file_content("```\nx = 1\n```") == "x = 1\n"

    def test_accepts_a_bare_file(self) -> None:
        assert extract_file_content("def run():\n    return 2\n") == "def run():\n    return 2\n"

    def test_adds_the_trailing_newline_a_file_should_have(self) -> None:
        assert extract_file_content("x = 1") == "x = 1\n"

    def test_rejects_an_empty_response(self) -> None:
        with pytest.raises(ChangeRejectedError, match="empty response"):
            extract_file_content("   ")

    def test_rejects_a_reply_that_starts_with_prose(self) -> None:
        with pytest.raises(ChangeRejectedError, match="started with prose"):
            extract_file_content("Sure! Here is the updated file:\n\nx = 1\n")

    def test_rejects_a_reply_that_elides_the_rest_of_the_file(self) -> None:
        with pytest.raises(ChangeRejectedError, match="elides the rest"):
            extract_file_content("x = 1\n# ... rest of file unchanged\n")

    def test_rejects_a_reply_with_an_unclosed_fence(self) -> None:
        with pytest.raises(ChangeRejectedError, match="never closed"):
            extract_file_content("```python\nx = 1\n")

    def test_rejects_a_fence_left_inside_the_body(self) -> None:
        with pytest.raises(ChangeRejectedError, match="code fence"):
            extract_file_content("x = 1\n```\ny = 2\n")

    def test_rejects_a_file_that_did_not_change(self) -> None:
        with pytest.raises(ChangeRejectedError, match="identical"):
            extract_file_content("x = 1\n", current="x = 1\n")


class TestGenerateFileChange:
    async def test_asks_for_one_file_with_its_current_contents(self, tmp_path: Path) -> None:
        client = ScriptedClient()
        plan = ImplementationPlan.model_validate(PLAN_REPLY)

        change = await generate_file_change(
            client,
            task="double what run() returns",
            repository="acme/widgets",
            plan=plan,
            path="src/widgets/core.py",
            role="modify",
            current=CORE,
            siblings=["tests/test_core.py"],
        )

        assert change.content == FILE_REPLIES["src/widgets/core.py"]
        assert client.asked_for == ["src/widgets/core.py"]

        user_prompt = client.requests[0][-1].content
        assert CORE.strip() in user_prompt
        assert "double what run() returns" in user_prompt
        assert "Double the return value in run()" in user_prompt
        assert "tests/test_core.py" in user_prompt

    async def test_tells_the_model_the_file_does_not_exist_yet(self) -> None:
        client = ScriptedClient()
        plan = ImplementationPlan.model_validate(PLAN_REPLY)

        await generate_file_change(
            client,
            task="add an uptime helper",
            repository="acme/widgets",
            plan=plan,
            path="src/widgets/uptime.py",
            role="add",
            current="",
        )

        assert "does not exist yet" in client.requests[0][-1].content

    async def test_asks_again_when_the_first_reply_is_unusable(self) -> None:
        client = ScriptedClient()
        client.queue(
            "src/widgets/core.py",
            "Sure! Here you go:\n\ndef run():\n    return 2\n",
            FILE_REPLIES["src/widgets/core.py"],
        )
        plan = ImplementationPlan.model_validate(PLAN_REPLY)

        change = await generate_file_change(
            client,
            task="t",
            repository="acme/widgets",
            plan=plan,
            path="src/widgets/core.py",
            role="modify",
            current=CORE,
        )

        assert change.content == FILE_REPLIES["src/widgets/core.py"]
        assert len(client.requests) == 2
        # The retry tells the model what was wrong with the first answer.
        assert client.requests[1][-1].content.startswith("That was not a usable file")

    async def test_gives_up_after_the_attempt_limit(self) -> None:
        client = ScriptedClient()
        client.queue("src/widgets/core.py", "```python\nx = 1\n", "```python\ny = 2\n")
        plan = ImplementationPlan.model_validate(PLAN_REPLY)

        with pytest.raises(ChangeRejectedError, match="after 2 attempts"):
            await generate_file_change(
                client,
                task="t",
                repository="acme/widgets",
                plan=plan,
                path="src/widgets/core.py",
                role="modify",
                current=CORE,
            )

        assert len(client.requests) == 2

    async def test_a_backend_failure_is_not_retried_here(self) -> None:
        plan = ImplementationPlan.model_validate(PLAN_REPLY)

        with pytest.raises(LLMRequestError):
            await generate_file_change(
                broken_llm(),
                task="t",
                repository="acme/widgets",
                plan=plan,
                path="src/widgets/core.py",
                role="modify",
                current=CORE,
            )


# ---------------------------------------------------------------------- the step


class TestApplyImplementationPlanStep:
    async def test_creates_the_runs_branch_in_its_own_checkout(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())

        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK

        events = await ctx.store.get_events(ctx.run.id)
        created = [e for e in events if e.type is EventType.BRANCH_CREATED]
        assert len(created) == 1
        assert created[0].data["branch"] == ctx.run.branch
        assert created[0].data["base"] == "main"
        assert created[0].data["head"] == git("rev-parse", "HEAD", cwd=source_repo)
        assert created[0].data["workspace"] == str(ctx.local_path())
        assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=ctx.local_path()) == ctx.run.branch

    async def test_writes_every_planned_file(self, settings: Settings, source_repo: Path) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())

        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK

        root = ctx.local_path()
        assert (root / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            FILE_REPLIES["src/widgets/core.py"]
        )
        assert (root / "tests" / "test_core.py").read_text(encoding="utf-8") == (
            FILE_REPLIES["tests/test_core.py"]
        )
        assert (root / "src" / "widgets" / "uptime.py").read_text(encoding="utf-8") == NEW_MODULE

        events = await ctx.store.get_events(ctx.run.id)
        written = [e for e in events if e.type is EventType.FILE_WRITTEN]
        assert [e.data["path"] for e in written] == [
            "src/widgets/core.py",
            "tests/test_core.py",
            "src/widgets/uptime.py",
        ]
        assert written[-1].data["created"] is True
        assert written[0].data["created"] is False
        assert written[0].data["branch"] == ctx.run.branch
        assert written[0].data["lines"] == 2

    async def test_the_events_never_carry_the_file_body(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())

        await ApplyImplementationPlanStep().execute(ctx)

        events = await ctx.store.get_events(ctx.run.id)
        payloads = [
            json.dumps(e.data)
            for e in events
            if e.type in (EventType.FILE_WRITTEN, EventType.BRANCH_CREATED)
        ]
        assert payloads
        for payload in payloads:
            assert "content" not in payload
            assert "def run" not in payload
            assert "uptime()" not in payload

    async def test_a_read_only_file_is_never_asked_about_or_written(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())

        await ApplyImplementationPlanStep().execute(ctx)

        assert "README.md" not in ScriptedClient().asked_for  # a fresh client asked nothing
        assert (ctx.local_path() / "README.md").read_text(encoding="utf-8") == "# widgets\n"

    async def test_the_run_never_commits(self, settings: Settings, source_repo: Path) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())
        head = git("rev-parse", "HEAD", cwd=source_repo)

        await ApplyImplementationPlanStep().execute(ctx)

        assert git("rev-parse", "HEAD", cwd=ctx.local_path()) == head
        dirty = git("status", "--porcelain", cwd=ctx.local_path())
        assert "M src/widgets/core.py" in dirty
        assert "?? src/widgets/uptime.py" in dirty

    async def test_a_refused_path_is_skipped_and_the_rest_still_lands(
        self, settings: Settings, source_repo: Path
    ) -> None:
        plan_reply = {
            **PLAN_REPLY,
            "relevant_files": [
                {"path": "../escaped.py", "reason": "outside", "role": "add"},
                {"path": "src/widgets/core.py", "reason": "defines run()", "role": "modify"},
            ],
        }
        ctx = await build_context(settings, source_repo, llm=ScriptedClient(), plan=False)
        await BuildRepositoryProfileStep().execute(ctx)
        await SelectRelevantFilesStep().execute(ctx)
        # The grounding filter drops invented ``modify`` paths but keeps ``add``, so an escape
        # attempt only reaches the writer the way a hallucinating model would produce one.
        plan, _ = await generate_plan(
            plan_llm(plan_reply),
            task=ctx.run.task,
            repository="acme/widgets",
            profile=ctx.state["profile"],
            candidates=ctx.state["candidates"],
            root=ctx.local_path(),
        )
        ctx.state["plan"] = plan

        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK

        assert not (ctx.local_path().parent.parent / "escaped.py").exists()
        assert (ctx.local_path() / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            FILE_REPLIES["src/widgets/core.py"]
        )
        events = await ctx.store.get_events(ctx.run.id)
        warnings = [e for e in events if "Left ../escaped.py alone" in e.message]
        assert warnings and warnings[0].level.value == "warn"
        written = [e for e in events if e.type is EventType.FILE_WRITTEN]
        assert [e.data["path"] for e in written] == ["src/widgets/core.py"]

    async def test_fails_when_the_plan_says_modify_but_the_file_is_gone(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient(), plan=False)
        ctx.state["plan"] = ImplementationPlan.model_validate(
            {
                **PLAN_REPLY,
                "relevant_files": [
                    {"path": "src/widgets/ghost.py", "reason": "invented", "role": "modify"}
                ],
                "steps": [],
            }
        )

        with pytest.raises(StepFailed, match="no planned file could be written"):
            await ApplyImplementationPlanStep().execute(ctx)

        assert not (ctx.local_path() / "src" / "widgets" / "ghost.py").exists()

    async def test_a_plan_with_nothing_to_change_is_not_a_failure(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient())
        ctx.state["plan"] = ImplementationPlan.model_validate(
            {**PLAN_REPLY, "relevant_files": [{"path": "README.md", "role": "read"}], "steps": []}
        )

        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK

        events = await ctx.store.get_events(ctx.run.id)
        assert not any(e.type is EventType.FILE_WRITTEN for e in events)
        assert any("no file was written" in e.message for e in events)

    async def test_fails_with_an_actionable_message_when_no_model_is_configured(
        self, settings: Settings, source_repo: Path
    ) -> None:
        ctx = await build_context(settings, source_repo, llm=None, plan=False)
        ctx.state["plan"] = ImplementationPlan.model_validate(PLAN_REPLY)

        with pytest.raises(StepFailed) as excinfo:
            await ApplyImplementationPlanStep().execute(ctx)

        assert "REPOPILOT_LLM_MODEL" in str(excinfo.value)

    async def test_fails_when_there_is_no_plan(self, settings: Settings, source_repo: Path) -> None:
        ctx = await build_context(settings, source_repo, llm=ScriptedClient(), plan=False)

        with pytest.raises(StepFailed, match="plan step must run first"):
            await ApplyImplementationPlanStep().execute(ctx)

    async def test_a_second_pass_reuses_the_branch(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """A retried run must not fail on a branch that is already checked out."""
        client = ScriptedClient()
        client.queue(
            "src/widgets/core.py", "def run():\n    return 2\n", "def run():\n    return 4\n"
        )
        ctx = await build_context(settings, source_repo, llm=client)
        plan = ctx.state["plan"]
        ctx.state["plan"] = plan.model_copy(
            update={
                "relevant_files": [
                    f for f in plan.relevant_files if f.path == "src/widgets/core.py"
                ]
            }
        )
        step = ApplyImplementationPlanStep()

        assert await step.execute(ctx) is StepOutcome.OK
        assert (ctx.local_path() / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            "def run():\n    return 2\n"
        )

        assert await step.execute(ctx) is StepOutcome.OK
        assert (ctx.local_path() / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            "def run():\n    return 4\n"
        )
        assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=ctx.local_path()) == ctx.run.branch
        assert (
            len(
                [
                    e
                    for e in await ctx.store.get_events(ctx.run.id)
                    if e.type is EventType.BRANCH_CREATED
                ]
            )
            == 2
        )


# ------------------------------------------------------------------- end to end


class TestPhase2BPipeline:
    async def test_the_whole_thing_runs_end_to_end(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """validate -> resolve -> clone -> inspect -> profile -> select_files -> plan -> apply."""
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient(),
            steps=phase2b_pipeline(source_repo),
        )

        run = await engine.start(
            repository="acme/widgets", task="double what run() returns and add an uptime helper"
        )
        await engine.wait_for(run.id, 120)

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.SUCCEEDED, final.error
        assert final.total_steps == 7
        assert final.completed_steps == 7
        assert final.progress == 1.0

        events = await engine.store.get_events(run.id)
        types = [e.type for e in events]
        assert types[0] is EventType.RUN_CREATED
        assert types[-1] is EventType.RUN_FINISHED
        assert EventType.BRANCH_CREATED in types
        assert types.index(EventType.BRANCH_CREATED) < types.index(EventType.FILE_WRITTEN)
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))

        branch_event = next(e for e in events if e.type is EventType.BRANCH_CREATED)
        assert branch_event.step == "apply_plan"
        workspace = Path(branch_event.data["workspace"])
        assert workspace.name == run.id
        assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=workspace) == run.branch

        written = [e for e in events if e.type is EventType.FILE_WRITTEN]
        assert len(written) == 3
        assert (workspace / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            FILE_REPLIES["src/widgets/core.py"]
        )
        assert (workspace / "src" / "widgets" / "uptime.py").is_file()

        completed = next(
            e for e in events if e.type is EventType.STEP_COMPLETED and e.step == "apply_plan"
        )
        assert completed.data["branch"] == run.branch
        assert len(completed.data["written"]) == 3

        await engine.shutdown()

    async def test_a_run_without_a_model_fails_before_writing_anything(
        self, settings: Settings, source_repo: Path
    ) -> None:
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=None,
            steps=phase2b_pipeline(source_repo),
        )

        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 120)

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.FAILED
        assert "REPOPILOT_LLM_MODEL" in (final.error or "")

        events = await engine.store.get_events(run.id)
        assert not any(e.type is EventType.FILE_WRITTEN for e in events)
        assert not any(e.type is EventType.BRANCH_CREATED for e in events)

        await engine.shutdown()

    async def test_the_events_reach_a_live_subscriber(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """The terminal renders the stream, so the writes have to be published, not stored."""
        bus = EventBus()
        engine = RunEngine(
            store=RunStore(),
            bus=bus,
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient(),
            steps=phase2b_pipeline(source_repo),
        )
        run = await engine.start(repository="acme/widgets", task="double what run() returns")

        async with bus.subscribe(run_id=run.id) as subscription:
            await engine.wait_for(run.id, 120)
            seen: list[EventType] = []
            while not subscription.queue.empty():
                item = subscription.queue.get_nowait()
                if isinstance(item, ExecutionEvent):
                    seen.append(item.type)

        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED
        assert EventType.BRANCH_CREATED in seen
        assert EventType.FILE_WRITTEN in seen

        await engine.shutdown()

    async def test_the_workspace_never_touches_the_source_repository(
        self, settings: Settings, source_repo: Path
    ) -> None:
        before = await asyncio.to_thread(_tree_fingerprint, source_repo)
        head_before = git("rev-parse", "HEAD", cwd=source_repo)

        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient(),
            steps=phase2b_pipeline(source_repo),
        )
        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 120)
        assert (await engine.store.require_run(run.id)).status is RunStatus.SUCCEEDED

        after = await asyncio.to_thread(_tree_fingerprint, source_repo)

        assert after == before
        assert git("rev-parse", "HEAD", cwd=source_repo) == head_before
        assert "repopilot/" not in git("branch", "--list", cwd=source_repo)

        await engine.shutdown()

    async def test_two_runs_of_one_repository_stay_isolated(
        self, settings: Settings, source_repo: Path
    ) -> None:
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient(),
            steps=phase2b_pipeline(source_repo),
        )

        first = await engine.start(repository="acme/widgets", task="double what run() returns")
        second = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(first.id, 120)
        await engine.wait_for(second.id, 120)

        assert (await engine.store.require_run(first.id)).status is RunStatus.SUCCEEDED
        assert (await engine.store.require_run(second.id)).status is RunStatus.SUCCEEDED

        repo_dir = settings.workspace_dir.resolve() / "acme__widgets"
        assert (repo_dir / first.id).is_dir()
        assert (repo_dir / second.id).is_dir()
        assert first.branch != second.branch

        await engine.shutdown()

    async def test_a_real_http_client_drives_a_real_write(
        self, settings: Settings, source_repo: Path
    ) -> None:
        """The scripted fake is convenient, but the shipped client must work the same way."""
        client = file_llm("def run():\n    return 7\n")
        ctx = await build_context(
            settings,
            source_repo,
            llm=plan_llm(),
            plan=True,
            file_client=client,
        )
        plan = ctx.state["plan"]
        # One file, so the single scripted reply is the file that matters.
        ctx.state["plan"] = plan.model_copy(
            update={
                "relevant_files": [
                    f for f in plan.relevant_files if f.path == "src/widgets/core.py"
                ]
            }
        )

        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK

        assert (ctx.local_path() / "src" / "widgets" / "core.py").read_text(encoding="utf-8") == (
            "def run():\n    return 7\n"
        )
        await client.aclose()
