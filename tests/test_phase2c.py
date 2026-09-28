"""Tests for Phase 2C: running the repository's tests, and repairing what the run broke.

The clone is real, the branch is real, and where a test is a Python repository the suite that
runs is a real ``python -m pytest`` in a real checkout. The reasoning backend is a scripted
:class:`ChatClient`. Nothing is mocked about the thing that matters most here: the command is
executed, its output is parsed, and the repair is proved by running the suite again.

The refusals are the tests that matter. A run that repairs a file it never wrote, that accepts a
partial file as a repair, that calls a missing runner a failing test, or that loops forever
because nobody bounded it, would each be far worse than a run that reports the failure and stops.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from repopilot.agent.diagnose import DiagnosisRejectedError, generate_diagnosis
from repopilot.agent.llm import ChatClient, ChatMessage, Completion
from repopilot.agent.schema import TestFailureReport as FailureReport
from repopilot.analysis.profile import RepositoryProfile, build_profile
from repopilot.analysis.testing import (
    MAX_REPORTED_FAILURES,
    collected_nothing,
    detect_test_suite,
    parse_test_output,
    tail_output,
)
from repopilot.analysis.testing import TestOutcome as Outcome
from repopilot.analysis.testing import TestTally as Tally
from repopilot.config import Settings
from repopilot.domain.models import EventType, ExecutionEvent, RunStatus
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.engine import MAX_ATTEMPTS, RunEngine
from repopilot.execution.sandbox import CommandResult, ShellSandbox
from repopilot.execution.steps import (
    ApplyImplementationPlanStep,
    BuildRepositoryProfileStep,
    CloneRepositoryStep,
    CreateImplementationPlanStep,
    DiagnoseFailuresStep,
    ResolveRepositoryStep,
    RunServices,
    RunTestSuiteStep,
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
#: A test that fails against ``CORE`` and passes against the doubled return value, which is what
#: gives the repair loop something real to fix. The package sits at the repository root because
#: ``python -m pytest`` puts the working directory on ``sys.path`` and nothing here installs the
#: package it is testing.
ASSERTING_TESTS = "from widgets.core import run\n\n\ndef test_run():\n    assert run() == 2\n"
BROKEN_CORE = "def run(:\n"

PLAN_REPLY = {
    "summary": "Double the value run() returns and make the test assert it.",
    "approach": "core.run() is the value under test.",
    "relevant_files": [
        {"path": "widgets/core.py", "reason": "defines run()", "role": "modify"},
        {"path": "tests/test_core.py", "reason": "asserts the value", "role": "modify"},
    ],
    "steps": [
        {
            "order": 1,
            "title": "Double the return value",
            "detail": "Change the literal 1 to 2.",
            "files": ["widgets/core.py"],
            "rationale": "That is where the value is produced.",
        }
    ],
    "risks": [],
    "test_strategy": "Run pytest.",
}

#: What the plan writes first: a change that breaks the test on purpose, so the diagnoser has a
#: real failure to explain and a real fix to apply.
BROKEN_CHANGE = "def run():\n    return 3\n"
REPAIRED_CORE = "def run():\n    return 2\n"

#: A change that does what it says, for the tests where the suite is expected to go green.
REPAIRED_REPLIES = {
    "widgets/core.py": REPAIRED_CORE,
    "tests/test_core.py": TESTS,
}

#: The files every fixture repository is made of, for the case that needs a package with a real
#: failing test: a change to ``widgets/core.py`` breaks the assertion in the test.
PACKAGE_FILES = {
    "pyproject.toml": '[project]\nname = "widgets"\n\n[tool.pytest.ini_options]\n',
    "widgets/__init__.py": "",
    "widgets/core.py": CORE,
    "tests/test_core.py": ASSERTING_TESTS,
}


def package_repo(where: Path) -> Path:
    return write_repo(where, dict(PACKAGE_FILES))


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fingerprint(repo: Path) -> list[tuple[str, int]]:
    """Every path outside .git with its size, so any edit to the source is detectable."""
    return sorted(
        (path.relative_to(repo).as_posix(), path.stat().st_size)
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(repo).parts
    )


def write_repo(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "t@example.com", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return root


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """A python package with a pytest suite, a real one that runs in this environment."""
    return write_repo(
        tmp_path / "source",
        {
            "pyproject.toml": '[project]\nname = "widgets"\n\n[tool.pytest.ini_options]\n',
            "widgets/__init__.py": "",
            "widgets/core.py": CORE,
            "tests/test_core.py": TESTS,
        },
    )


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


def profile_of(root: Path) -> RepositoryProfile:
    return build_profile(root)


#: A phrase from the diagnoser's own instructions, used to tell its requests from the other two.
DIAGNOSIS_MARKER = "you return a diagnosis"

#: The implementer prompt always names the file it wants back in its last line. The path capture
#: is greedy so that the dots in "core.py" survive it.
_ASKED = re.compile(r"Return the (?:new contents of|contents of the new) (\S+)\.")


class ScriptedClient:
    """A ``ChatClient`` that answers the planner, the file writer, and the diagnoser.

    Keyed the way the prompts identify the work, so one fake can serve a whole pipeline run: the
    planner is recognised by its system message, the implementer by the file it is asked for, and
    the diagnoser by its own system message. A list value is a queue, which is how a test gives
    the first repair round a wrong answer and the second a right one.
    """

    def __init__(self, replies: dict[str, Any] | None = None, *, model: str = "test-model") -> None:
        self.model = model
        self._replies: dict[str, Any] = dict(replies or {})
        self.requests: list[list[ChatMessage]] = []
        self._current: str | None = None

    def queue(self, key: str, *replies: Any) -> None:
        self._replies[key] = list(replies)

    def _next(self, key: str) -> Any:
        reply = self._replies.get(key)
        if isinstance(reply, list):
            if not reply:
                raise AssertionError(f"ran out of scripted replies for {key}")
            reply = reply.pop(0)
        if reply is None:
            raise AssertionError(f"the run asked for {key!r}, which was not scripted")
        return reply

    def _asked_path(self, messages: list[ChatMessage]) -> str:
        tail = "\n".join(messages[-1].content.strip().splitlines()[-3:])
        match = _ASKED.search(tail)
        if match is None:
            raise AssertionError(f"no file was requested in: {tail!r}")
        return match.group(1)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.requests.append(messages)
        system = messages[0].content
        if "planning a change" in system:
            return Completion(
                text=json.dumps(PLAN_REPLY),
                model=self.model,
                prompt_tokens=40,
                completion_tokens=12,
            )
        if DIAGNOSIS_MARKER in system:
            return Completion(
                text=json.dumps(self._next("diagnosis")),
                model=self.model,
                prompt_tokens=90,
                completion_tokens=30,
            )
        # A correction carries the previous request rather than a new one, so the file being asked
        # about is the one the last request named.
        if "Return the" in messages[-1].content:
            self._current = self._asked_path(messages)
        path = self._current or "file"
        key = path if path in self._replies else "file"
        return Completion(
            text=str(self._next(key)), model=self.model, prompt_tokens=11, completion_tokens=7
        )

    async def aclose(self) -> None:
        return None


def diagnosis(*findings: dict[str, Any], summary: str = "The change broke the value.") -> dict:
    return {"summary": summary, "findings": list(findings)}


class LocalClone(CloneRepositoryStep):
    """Clone from a local repository, so nothing in these tests needs a network."""

    def __init__(self, source: Path) -> None:
        self.source = source

    async def execute(self, ctx: StepContext) -> StepOutcome:
        info: RepositoryInfo = ctx.state["repository_info"]
        ctx.state["repository_info"] = info.model_copy(update={"clone_url": str(self.source)})
        return await super().execute(ctx)


def phase2c_pipeline(source: Path) -> tuple[Step, ...]:
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        LocalClone(source),
        BuildRepositoryProfileStep(),
        SelectRelevantFilesStep(),
        CreateImplementationPlanStep(),
        ApplyImplementationPlanStep(),
        RunTestSuiteStep(),
        DiagnoseFailuresStep(),
    )


class FakeSandbox:
    """A sandbox that answers from a script, for the cases a real command cannot produce.

    Real processes are used everywhere else; this exists only to simulate a runner that is
    missing, that hangs, or that prints something no runner on this machine would print.
    """

    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, ...]] = []

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None = None,
        is_cancelled: Any = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        self.calls.append(tuple(argv))
        if not self.results:
            raise AssertionError(f"unexpected command: {list(argv)}")
        return self.results.pop(0)


def passed(exit_code: int = 0, output: str = "2 passed in 0.10s") -> CommandResult:
    return CommandResult(argv=("pytest",), cwd=Path("."), exit_code=exit_code, stdout=output)


PYTEST_FAILING = """\
FAILURES
=================================== FAILURES ===================================
_________________________________ test_run _____________________________________

    def test_run():
>       assert run() == 2
E       assert 3 == 2

tests/test_core.py:6: AssertionError
=========================== short test summary info ============================
FAILED tests/test_core.py::test_run - assert 3 == 2
========================= 1 failed, 1 passed in 0.21s =========================
"""


def failing() -> CommandResult:
    return CommandResult(
        argv=("pytest",), cwd=Path("."), exit_code=1, stdout=PYTEST_FAILING, duration_seconds=0.4
    )


class Context:
    """A StepContext with a real clone, so the steps under test work on a real checkout."""

    def __init__(self, ctx: StepContext) -> None:
        self._ctx = ctx
        self.events: list[tuple[EventType, str, dict]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    async def emit(self, event_type: Any, message: str, **kwargs: Any) -> Any:
        self.events.append((EventType(event_type), message, kwargs.get("data") or {}))
        return None

    def of(self, event_type: EventType) -> list[tuple[str, dict]]:
        return [(message, data) for kind, message, data in self.events if kind is event_type]

    def messages(self) -> list[str]:
        return [message for _, message, _ in self.events]


async def build_context(
    source: Path,
    *,
    llm: ChatClient | None = None,
    settings: Settings,
    attempt: int = 0,
    max_attempts: int = MAX_ATTEMPTS,
    apply: bool = True,
) -> Context:
    """A context past the clone, with a real profile and, by default, a real applied plan."""
    engine = RunEngine(
        store=RunStore(),
        bus=EventBus(),
        settings=settings,
        github=mock_github(),
        llm=llm,
        steps=(ValidateRequestStep(), ResolveRepositoryStep(), LocalClone(source)),
    )
    run = await engine.store.create_run(repository="acme/widgets", task="double what run() returns")
    ctx = StepContext(
        run=run,
        store=engine.store,
        services=RunServices(
            github=engine.github, workspace=settings.workspace_dir.resolve(), llm=llm
        ),
        emit=lambda *a, **k: asyncio.sleep(0),
        attempt=attempt,
        max_attempts=max_attempts,
    )
    ctx.state["repository_info"] = RepositoryInfo(
        full_name="acme/widgets",
        default_branch="main",
        html_url="https://github.com/acme/widgets",
        clone_url=str(source),
    )
    assert await CloneRepositoryStep().execute(ctx) is StepOutcome.OK
    ctx.state["default_branch"] = "main"
    if apply:
        assert await BuildRepositoryProfileStep().execute(ctx) is StepOutcome.OK
        assert await SelectRelevantFilesStep().execute(ctx) is StepOutcome.OK
        assert await CreateImplementationPlanStep().execute(ctx) is StepOutcome.OK
        assert await ApplyImplementationPlanStep().execute(ctx) is StepOutcome.OK
    else:
        assert await BuildRepositoryProfileStep().execute(ctx) is StepOutcome.OK
    return Context(ctx)


# ----------------------------------------------------------------------- the sandbox


class TestShellSandbox:
    async def test_runs_a_command_and_captures_both_streams(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run(
            (
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
            ),
            cwd=tmp_path,
        )

        assert result.exit_code == 3
        assert not result.ok
        assert result.ran
        assert "out" in result.stdout
        assert "err" in result.stderr
        assert "out" in result.output and "err" in result.output

    async def test_runs_in_the_directory_it_is_given(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run(
            (sys.executable, "-c", "import os; print(os.getcwd())"), cwd=tmp_path
        )

        reported = Path(result.stdout.strip())
        assert reported.name == tmp_path.name

    async def test_a_missing_binary_is_reported_not_raised(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run(("definitely-not-a-real-binary",), cwd=tmp_path)

        assert not result.ran
        assert not result.ok
        assert "definitely-not-a-real-binary" in result.error

    async def test_an_empty_command_is_reported_not_raised(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run((), cwd=tmp_path)

        assert not result.ran
        assert result.error

    async def test_a_command_that_overruns_is_killed(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            timeout_seconds=1.0,
        )

        assert result.timed_out
        assert not result.ran
        assert not result.ok
        assert "1s" in result.error
        assert result.duration_seconds < 20

    async def test_cancelling_stops_the_command(self, tmp_path: Path) -> None:
        cancelled = False

        def is_cancelled() -> bool:
            return cancelled

        async def flip() -> None:
            nonlocal cancelled
            await asyncio.sleep(0.4)
            cancelled = True

        flipper = asyncio.create_task(flip())
        result = await ShellSandbox().run(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            timeout_seconds=25.0,
            is_cancelled=is_cancelled,
        )
        await flipper

        assert result.cancelled
        assert not result.ok
        assert result.duration_seconds < 20

    async def test_output_is_bounded_to_a_tail(self, tmp_path: Path) -> None:
        result = await ShellSandbox(max_output_chars=500).run(
            (sys.executable, "-c", "print('x' * 100000)"),
            cwd=tmp_path,
        )

        assert len(result.stdout) <= 500
        assert "summary at the end" in result.stdout or len(result.stdout) <= 500

    async def test_a_timeout_does_not_wait_for_the_process_after_killing_it(
        self, tmp_path: Path
    ) -> None:
        result = await ShellSandbox().run(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            timeout_seconds=1.0,
        )

        assert result.timed_out
        assert result.duration_seconds < 20

    async def test_repo_pilot_configuration_never_reaches_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REPOPILOT_LLM_API_KEY", "sk-do-not-leak")
        monkeypatch.setenv("REPOPILOT_GITHUB_TOKEN", "ghp-do-not-leak")

        result = await ShellSandbox().run(
            (sys.executable, "-c", "import os; print(os.getenv('REPOPILOT_LLM_API_KEY'))"),
            cwd=tmp_path,
        )

        assert result.stdout.strip() == "None"

    async def test_a_child_can_be_given_its_own_environment(self, tmp_path: Path) -> None:
        result = await ShellSandbox().run(
            (sys.executable, "-c", "import os; print(os.environ['WIDGETS_TEST'])"),
            cwd=tmp_path,
            env={"WIDGETS_TEST": "yes"},
        )

        assert result.stdout.strip() == "yes"

    def test_it_works_on_a_loop_that_cannot_spawn_subprocesses(self, tmp_path: Path) -> None:
        """uvicorn's Windows reload mode runs a SelectorEventLoop, which spawns nothing."""

        async def scenario() -> CommandResult:
            return await ShellSandbox().run(
                (sys.executable, "-c", "print('on the selector loop')"), cwd=tmp_path
            )

        loop = asyncio.SelectorEventLoop()
        try:
            result = loop.run_until_complete(scenario())
        finally:
            loop.close()

        assert result.exit_code == 0
        assert "on the selector loop" in result.stdout


# ----------------------------------------------------------------------- detection


class TestDetection:
    def test_finds_pytest_from_its_configuration(self, python_repo: Path) -> None:
        suite = detect_test_suite(profile_of(python_repo))

        assert suite is not None
        assert suite.framework == "pytest"
        assert suite.command[0] == sys.executable
        assert "-m" in suite.command and "pytest" in suite.command
        assert suite.language == "python"
        assert suite.test_files == 1

    def test_pytest_probes_for_itself(self, python_repo: Path) -> None:
        suite = detect_test_suite(profile_of(python_repo))

        assert suite is not None
        assert suite.probe
        assert "pytest" in suite.probe_reason

    def test_finds_npm_jest_from_the_test_script(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "node",
            {
                "package.json": json.dumps(
                    {"scripts": {"test": "jest --ci"}, "devDependencies": {"jest": "^29"}}
                ),
                "src/app.test.js": "test('a', () => {});",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "jest"
        assert suite.command == ("npm", "test")
        assert not suite.probe

    def test_uses_pnpm_when_the_lockfile_says_so(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "pnpm",
            {
                "package.json": json.dumps({"scripts": {"test": "vitest run"}}),
                "pnpm-lock.yaml": "lockfileVersion: 9\n",
                "src/app.test.ts": "it('a', () => {});",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "vitest"
        assert suite.command == ("pnpm", "test")

    def test_finds_go(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "go",
            {"go.mod": "module acme/widgets\n\ngo 1.22\n", "core_test.go": "package core\n"},
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "go"
        assert suite.command == ("go", "test", "./...")

    def test_finds_cargo(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "rs",
            {
                "Cargo.toml": '[package]\nname = "widgets"\n',
                "src/lib.rs": "pub fn run() -> i32 { 1 }\n",
                "tests/core_test.rs": "#[test]\nfn run() { assert_eq!(1, 1); }\n",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "cargo"
        assert suite.command == ("cargo", "test")

    def test_finds_maven(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "mvn",
            {
                "pom.xml": "<project><modelVersion>4.0.0</modelVersion></project>",
                "src/test/java/WidgetsTest.java": "class WidgetsTest {}",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "maven"
        assert suite.command == ("mvn", "-B", "test")

    def test_finds_gradle(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "gradle",
            {
                "build.gradle": "plugins { id 'java' }\n",
                "src/test/java/WidgetsTest.java": "class WidgetsTest {}",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "gradle"

    def test_falls_back_to_unittest_for_python_tests_with_no_runner(self, tmp_path: Path) -> None:
        """A test module only unittest's own discovery pattern would pick up."""
        root = write_repo(
            tmp_path / "stdlib",
            {
                "pyproject.toml": '[project]\nname = "widgets"\n',
                "widgets.py": "def run():\n    return 1\n",
                "tests/testcore.py": (
                    "import unittest\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_it(self):\n        self.assertTrue(True)\n"
                ),
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "unittest"
        assert suite.command[1:] == ("-m", "unittest", "discover", "-v")

    def test_a_repository_with_no_tests_gets_no_command(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "quiet",
            {"README.md": "# widgets\n", "main.py": "print('hello')\n"},
        )

        assert detect_test_suite(profile_of(root)) is None

    def test_a_manifest_without_a_test_script_gets_no_command(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "noscript",
            {
                "package.json": json.dumps({"scripts": {"build": "tsc"}}),
                "src/app.test.js": "test('a', () => {});",
            },
        )

        assert detect_test_suite(profile_of(root)) is None

    def test_the_ecosystem_with_the_most_tests_wins(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "mixed",
            {
                "go.mod": "module acme/widgets\n",
                "core_test.go": "package core\n",
                "core2_test.go": "package core\n",
                "core3_test.go": "package core\n",
                "pyproject.toml": "[tool.pytest.ini_options]\n",
                "tests/test_core.py": "def test_a():\n    pass\n",
            },
        )

        suite = detect_test_suite(profile_of(root))

        assert suite is not None
        assert suite.framework == "go"

    def test_the_answer_does_not_depend_on_which_detector_ran_first(self, tmp_path: Path) -> None:
        root = write_repo(
            tmp_path / "tie",
            {
                "go.mod": "module acme/widgets\n",
                "core_test.go": "package core\n",
                "pyproject.toml": "[tool.pytest.ini_options]\n",
                "tests/test_core.py": "def test_a():\n    pass\n",
            },
        )
        profile = profile_of(root)

        assert detect_test_suite(profile) == detect_test_suite(profile)

    def test_the_command_is_shown_the_way_a_person_would_type_it(self, python_repo: Path) -> None:
        suite = detect_test_suite(profile_of(python_repo))

        assert suite is not None
        # The interpreter path is quoted when it needs to be, so the display can be copied.
        assert suite.as_command_text() == " ".join(
            f'"{part}"' if (" " in part or "\\" in part) else part for part in suite.command
        )
        assert "-m pytest" in suite.as_command_text()


# -------------------------------------------------------------------------- parsing


class TestParsing:
    def test_reads_pytest_counts_and_names_the_failures(self) -> None:
        tally = parse_test_output("pytest", PYTEST_FAILING, exit_code=1)

        assert tally.failed == 1
        assert tally.passed == 1
        assert len(tally.failures) == 1
        assert tally.failures[0].test == "tests/test_core.py::test_run"
        assert "assert 3 == 2" in tally.failures[0].excerpt

    def test_a_total_is_only_reported_when_every_count_is_known(self) -> None:
        assert parse_test_output("pytest", "2 passed in 0.05s", exit_code=0).total is None
        assert (
            parse_test_output(
                "maven", "Tests run: 4, Failures: 0, Errors: 0, Skipped: 0", exit_code=0
            ).total
            == 4
        )

    def test_a_failing_run_with_no_named_test_still_carries_evidence(self) -> None:
        tally = parse_test_output("cargo", "error: could not compile `widgets`", exit_code=101)

        assert len(tally.failures) == 1
        assert "could not compile" in tally.failures[0].excerpt
        assert "101" in tally.failures[0].test

    def test_a_passing_run_is_not_given_a_failure(self) -> None:
        tally = parse_test_output("pytest", "3 passed in 0.05s", exit_code=0)

        assert tally.failures == ()
        assert tally.passed == 3

    def test_counts_stay_unknown_when_the_runner_prints_none(self) -> None:
        tally = parse_test_output("npm", "npm error code ELIFECYCLE", exit_code=1)

        assert tally.passed is None
        assert tally.failed is None
        assert len(tally.failures) == 1

    def test_reads_unittest(self) -> None:
        output = (
            "FAIL: test_run (core.T)\nERROR: test_other (core.T)\nFAILED (failures=1, errors=1)\n"
        )

        tally = parse_test_output("unittest", output, exit_code=1)

        assert tally.failed == 1
        assert tally.errors == 1
        assert [f.test for f in tally.failures] == ["test_run", "test_other"]

    def test_reads_jest(self) -> None:
        output = "✕ doubles the value (5 ms)\nTests: 1 failed, 2 passed, 3 total\n"

        tally = parse_test_output("jest", output, exit_code=1)

        assert tally.failed == 1
        assert tally.passed == 2
        assert tally.failures[0].test == "doubles the value"

    def test_reads_go(self) -> None:
        output = "--- FAIL: TestRun (0.00s)\nok  \tacme/widgets/other\t0.2s\nFAIL\n"

        tally = parse_test_output("go", output, exit_code=1)

        assert tally.failed == 1
        assert tally.passed == 1
        assert tally.failures[0].test == "TestRun"

    def test_reads_cargo(self) -> None:
        output = (
            "test core::tests::run ... FAILED\ntest result: FAILED. 1 passed; 1 failed; 0 ignored\n"
        )

        tally = parse_test_output("cargo", output, exit_code=101)

        assert tally.passed == 1
        assert tally.failed == 1
        assert tally.failures[0].test == "core::tests::run"

    def test_reads_maven(self) -> None:
        output = (
            "Tests run: 5, Failures: 1, Errors: 0, Skipped: 1\n"
            "core.WidgetsTest.testRun -- Time elapsed: 0.1 s <<< FAILURE!\n"
        )

        tally = parse_test_output("maven", output, exit_code=1)

        assert tally.failed == 1
        assert tally.skipped == 1
        assert tally.passed == 4
        assert tally.failures[0].test == "core.WidgetsTest.testRun"

    def test_reads_gradle(self) -> None:
        tally = parse_test_output("gradle", "3 tests completed, 1 failed\n", exit_code=1)

        assert tally.failed == 1
        assert tally.passed == 2

    def test_failures_are_capped(self) -> None:
        output = "\n".join(f"FAILED tests/test_x.py::test_{i} - boom" for i in range(200))

        tally = parse_test_output("pytest", f"{output}\n200 failed in 1.0s\n", exit_code=1)

        assert len(tally.failures) == MAX_REPORTED_FAILURES
        assert tally.failed == 200

    def test_pytest_collecting_nothing_is_not_a_failure(self) -> None:
        assert collected_nothing("pytest", 5)
        assert not collected_nothing("pytest", 1)
        assert not collected_nothing("cargo", 101)

    def test_a_runner_that_collected_nothing_is_not_a_passing_suite(self) -> None:
        output = "Ran 0 tests in 0.001s\n\nOK\n"

        assert collected_nothing("unittest", 0, output)
        assert not collected_nothing("unittest", 0, "Ran 3 tests in 0.001s\n\nOK\n")

    def test_the_tail_keeps_the_end_of_a_long_log(self) -> None:
        output = "\n".join(f"line {i}" for i in range(5000))

        tail = tail_output(output, max_chars=200)

        assert len(tail) <= 200
        assert tail.endswith("line 4999")
        assert "line 0" not in tail

    def test_an_empty_output_has_an_empty_tail(self) -> None:
        assert tail_output("") == ""


# -------------------------------------------------------------------- the test step


class TestRunTestSuiteStep:
    async def test_runs_a_real_suite_in_the_runs_own_checkout(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        outcome = ctx.state["test_result"]
        assert outcome.status == "passed"
        assert outcome.tally.passed == 1
        assert [event for event, _, _ in ctx.events].count(EventType.TEST_SUITE_DETECTED) == 1
        started, completed = ctx.of(EventType.TEST_STARTED), ctx.of(EventType.TEST_COMPLETED)
        assert started and completed
        assert completed[0][1]["status"] == "passed"
        assert ctx.of(EventType.STEP_COMPLETED)[0][1]["verified"] is True

    async def test_a_repository_with_no_suite_is_skipped_not_failed(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        source = write_repo(tmp_path / "quiet", {"README.md": "# quiet\n", "main.py": "x = 1\n"})

        ctx = await build_context(source, settings=settings, apply=False)

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["test_result"] is None
        assert len(ctx.of(EventType.TEST_SKIPPED)) == 1
        assert not ctx.of(EventType.TEST_COMPLETED)
        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.OK

    async def test_a_missing_runner_is_reported_as_unavailable_not_as_a_failure(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)
        ctx.services.sandbox = FakeSandbox(
            CommandResult(argv=("py",), cwd=ctx.local_path(), exit_code=1, error="no pytest")
        )

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert len(ctx.of(EventType.TEST_SKIPPED)) == 1
        message, data = ctx.of(EventType.TEST_SKIPPED)[0]
        assert data["reason"] == "pytest is not installed in the environment running RepoPilot"
        assert "Could not run" in message
        # The probe was a check on the environment, so the suite itself was never started.
        assert len(ctx.services.sandbox.calls) == 1

    async def test_a_failing_suite_is_handed_on_rather_than_failing_the_run(
        self, settings: Settings
    ) -> None:
        source = write_repo(
            settings.workspace_dir.parent / "broken",
            {
                "pyproject.toml": "[tool.pytest.ini_options]\n",
                "widgets.py": "def run():\n    return 1\n",
                "tests/test_core.py": "from widgets import run\n\n\ndef test_run():\n"
                "    assert run() == 2\n",
            },
        )
        ctx = await build_context(source, settings=settings, apply=False)

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        outcome = ctx.state["test_result"]
        assert outcome.status == "failed"
        assert outcome.exit_code == 1
        assert outcome.tally.failures[0].test.endswith("test_run")
        assert len(ctx.of(EventType.TEST_COMPLETED)) == 1
        assert any("diagnosed next" in message for message in ctx.messages())

    async def test_the_suite_runs_where_the_run_owns_the_files(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)
        seen: list[Path] = []

        class Watcher(FakeSandbox):
            async def run(self, argv, *, cwd, **kwargs):  # type: ignore[no-untyped-def]
                seen.append(cwd)
                return await super().run(argv, cwd=cwd, **kwargs)

        ctx.services.sandbox = Watcher(
            CommandResult(argv=("p",), cwd=ctx.local_path(), exit_code=0),
            CommandResult(argv=("p",), cwd=ctx.local_path(), exit_code=0, stdout="1 passed"),
        )

        await RunTestSuiteStep().execute(ctx)

        assert seen
        assert all(path == ctx.local_path() for path in seen)

    async def test_a_hung_suite_is_reported_as_a_timeout(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)
        ctx.services.sandbox = FakeSandbox(
            CommandResult(argv=("p",), cwd=ctx.local_path(), exit_code=0),
            CommandResult(
                argv=("p",),
                cwd=ctx.local_path(),
                timed_out=True,
                error="the command was still running after 5s",
            ),
        )

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        outcome = ctx.state["test_result"]
        assert outcome.status == "timeout"
        assert not outcome.suite_ran
        assert (
            "did not finish" in ctx.of(EventType.STEP_COMPLETED)[0][1]
            or "stopped" in (ctx.of(EventType.STEP_COMPLETED)[0][0])
        )

    async def test_a_cancelled_suite_does_not_fail_the_run(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)
        ctx.services.sandbox = FakeSandbox(
            CommandResult(argv=("p",), cwd=ctx.local_path(), exit_code=0),
            CommandResult(argv=("p",), cwd=ctx.local_path(), cancelled=True, error="cancelled"),
        )

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK
        assert ctx.state["test_result"].status == "cancelled"

    async def test_no_profile_is_an_error_not_a_skip(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = write_repo(tmp_path / "bare", {"README.md": "# bare\n", "main.py": "x = 1\n"})
        ctx = await build_context(source, settings=settings, apply=False)
        ctx.state.pop("profile")

        with pytest.raises(StepFailed, match="profile step must run first"):
            await RunTestSuiteStep().execute(ctx)


# ----------------------------------------------------------------- the diagnose step


class TestDiagnoseFailuresStep:
    async def test_nothing_to_do_does_not_call_a_model(
        self, python_repo: Path, settings: Settings
    ) -> None:
        ctx = await build_context(python_repo, settings=settings, apply=False)
        llm = ScriptedClient()
        ctx.services.llm = llm
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.OK

        assert llm.requests == []
        assert any("Nothing to diagnose" in message for message in ctx.messages())

    async def test_a_real_failure_with_no_model_says_what_to_do(self, settings: Settings) -> None:
        source = write_repo(
            settings.workspace_dir.parent / "broken",
            {
                "pyproject.toml": "[tool.pytest.ini_options]\n",
                "widgets.py": "def run():\n    return 1\n",
                "tests/test_core.py": "from widgets import run\n\n\ndef test_run():\n"
                "    assert run() == 2\n",
            },
        )
        ctx = await build_context(source, settings=settings, apply=False)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        with pytest.raises(StepFailed, match="no reasoning backend is configured"):
            await DiagnoseFailuresStep().execute(ctx)

    async def test_it_repairs_what_the_run_broke_and_the_suite_then_passes(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """The whole point: a real pytest failure, a real repair, a real re-run."""
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "run() returns 3 but the test expects 2",
                        "content": REPAIRED_CORE,
                    }
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)

        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK
        assert ctx.state["test_result"].status == "failed"

        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["test_result"].status == "passed"
        assert (ctx.local_path() / "widgets" / "core.py").read_text(
            encoding="utf-8"
        ) == REPAIRED_CORE
        diagnosed = ctx.of(EventType.FAILURE_DIAGNOSED)
        assert len(diagnosed) == 1
        assert diagnosed[0][1]["findings"][0]["path"] == "widgets/core.py"
        written = [data for _, data in ctx.of(EventType.FILE_WRITTEN) if data.get("repair")]
        assert written and written[0]["path"] == "widgets/core.py"
        assert any("The suite passes after 1 repair round" in m for m in ctx.messages())

    async def test_it_refuses_to_repair_a_file_the_run_never_wrote(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "tests/test_core.py",
                        "cause": "the test asserts the wrong thing",
                        "content": "def test_run():\n    assert True\n",
                    }
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        # The diagnosis was honest and out of scope, so the run asks the engine for another
        # attempt rather than writing a file it never touched.
        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.RETRY

        # The file on disk is the file the run wrote: an out-of-diff repair never happened.
        assert (ctx.local_path() / "tests" / "test_core.py").read_text(encoding="utf-8") == (
            ASSERTING_TESTS
        )
        assert not [d for _, d in ctx.of(EventType.FILE_WRITTEN) if d.get("repair")]
        assert any("did not change" in message for message in ctx.messages())

    async def test_it_refuses_a_partial_file_as_a_repair(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "wrong value",
                        "content": "# just the diff\n... rest of the file unchanged\n",
                    }
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        # A partial "repair" is refused rather than written, so the suite is still failing and
        # the run asks for another attempt.
        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.RETRY

        assert any("Refused the proposed fix" in m for m in ctx.messages())
        assert (ctx.local_path() / "widgets" / "core.py").read_text(
            encoding="utf-8"
        ) == BROKEN_CHANGE

    async def test_a_pre_existing_failure_is_reported_and_not_repaired(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "the module has never returned 2",
                        "pre_existing": True,
                    }
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.OK

        assert not [d for _, d in ctx.of(EventType.FILE_WRITTEN) if d.get("repair")]
        assert any("already there before this change" in m for m in ctx.messages())

    async def test_the_repair_loop_is_bounded_and_then_asks_to_retry(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "still wrong",
                        "content": "def run():\n    return 5\n",
                    }
                ),
            }
        )
        # Each round proposes a different value, so both rounds really write a file rather than
        # the second being refused as a no-op.
        llm.queue(
            "diagnosis",
            diagnosis(
                {
                    "test": "tests/test_core.py::test_run",
                    "path": "widgets/core.py",
                    "cause": "wrong value",
                    "content": "def run():\n    return 5\n",
                }
            ),
            diagnosis(
                {
                    "test": "tests/test_core.py::test_run",
                    "path": "widgets/core.py",
                    "cause": "wrong value again",
                    "content": "def run():\n    return 4\n",
                }
            ),
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.RETRY

        # Two rounds, each with one diagnosis and one repair, and no third.
        assert len(ctx.of(EventType.FAILURE_DIAGNOSED)) == 2
        assert len([d for _, d in ctx.of(EventType.FILE_WRITTEN) if d.get("repair")]) == 2
        assert sum(1 for m in ctx.messages() if "Giving up" in m) == 1
        assert (ctx.local_path() / "widgets" / "core.py").read_text(encoding="utf-8") == (
            "def run():\n    return 4\n"
        )

    async def test_on_the_last_attempt_a_failure_ends_the_run_with_the_reason(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "still wrong",
                        "content": "def run():\n    return 5\n",
                    }
                ),
            }
        )
        ctx = await build_context(
            source,
            settings=settings,
            llm=llm,
            apply=True,
            attempt=MAX_ATTEMPTS - 1,
            max_attempts=MAX_ATTEMPTS,
        )
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        assert not ctx.can_retry
        with pytest.raises(StepFailed, match="still fails after 2 repair round"):
            await DiagnoseFailuresStep().execute(ctx)

    async def test_a_cancelled_run_stops_diagnosing(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "t",
                        "path": "widgets/core.py",
                        "cause": "c",
                        "content": REPAIRED_CORE,
                    }
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK
        ctx._ctx.is_cancelled = lambda: True

        assert await DiagnoseFailuresStep().execute(ctx) is StepOutcome.OK

        assert llm.requests == [] or "diagnos" not in llm.requests[-1][0].content.lower()
        assert not [d for _, d in ctx.of(EventType.FILE_WRITTEN) if d.get("repair")]

    async def test_a_model_that_cannot_be_parsed_ends_the_run(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                # A reply that never becomes a report at all, however many times it is read.
                "diagnosis": {"summary": "I could not be bothered"},
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        # A reply that never becomes a report is not retried forever, and the run says so.
        with pytest.raises(StepFailed, match="could not explain the test failure"):
            await DiagnoseFailuresStep().execute(ctx)

    async def test_the_prompt_carries_the_failure_and_only_the_runs_own_files(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "repair")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {"test": "t", "path": "widgets/core.py", "cause": "c", "pre_existing": True}
                ),
            }
        )
        ctx = await build_context(source, settings=settings, llm=llm, apply=True)
        assert await RunTestSuiteStep().execute(ctx) is StepOutcome.OK

        await DiagnoseFailuresStep().execute(ctx)

        diagnosis_prompt = next(
            messages[-1].content
            for messages in llm.requests
            if DIAGNOSIS_MARKER in messages[0].content
        )
        assert "tests/test_core.py::test_run" in diagnosis_prompt
        assert "assert 3 == 2" in diagnosis_prompt
        assert "widgets/core.py" in diagnosis_prompt
        assert "only files you may return contents for" in diagnosis_prompt
        assert "double what run() returns" in diagnosis_prompt


# ------------------------------------------------------------------------ diagnosis


class TestGenerateDiagnosis:
    async def test_a_transport_error_is_not_asked_again(self, python_repo: Path) -> None:
        from repopilot.agent.llm import LLMError

        class Broken:
            model = "m"

            async def complete(self, *a: Any, **k: Any) -> Completion:
                raise LLMError("the model is unreachable")

        with pytest.raises(LLMError):
            await generate_diagnosis(
                Broken(),  # type: ignore[arg-type]
                task="t",
                repository="acme/widgets",
                outcome=Outcome(
                    framework="pytest",
                    command=("pytest",),
                    status="failed",
                    exit_code=1,
                    tally=Tally(failures=(), failed=1),
                ),
                files=[],
                round_number=1,
            )

    async def test_a_reply_that_is_not_json_is_corrected_once(self) -> None:
        class Chatty:
            model = "m"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, messages: Any, **k: Any) -> Completion:
                self.calls += 1
                text = (
                    "here is my answer, in prose"
                    if self.calls == 1
                    else json.dumps(diagnosis(summary="run() returns the wrong value"))
                )
                return Completion(text=text, model=self.model)

        chatty = Chatty()
        report = await generate_diagnosis(
            chatty,  # type: ignore[arg-type]
            task="t",
            repository="acme/widgets",
            outcome=Outcome(framework="pytest", command=("pytest",), status="failed", exit_code=1),
            files=[],
            round_number=1,
        )

        assert isinstance(report, FailureReport)
        assert chatty.calls == 2
        assert report.model == "m"

    async def test_a_reply_with_the_wrong_shape_is_rejected(self) -> None:
        class Wrong:
            model = "m"

            async def complete(self, *a: Any, **k: Any) -> Completion:
                return Completion(text=json.dumps({"summary": "no findings key"}), model="m")

        with pytest.raises(DiagnosisRejectedError, match="did not return a usable diagnosis"):
            await generate_diagnosis(
                Wrong(),  # type: ignore[arg-type]
                task="t",
                repository="acme/widgets",
                outcome=Outcome(
                    framework="pytest", command=("pytest",), status="failed", exit_code=1
                ),
                files=[],
                round_number=1,
            )

    async def test_a_file_too_large_to_quote_is_named_without_its_contents(self) -> None:
        seen: list[str] = []

        class Watcher:
            model = "m"

            async def complete(self, messages: Any, **k: Any) -> Completion:
                seen.append(messages[-1].content)
                return Completion(text=json.dumps(diagnosis(summary="a big file")), model="m")

        await generate_diagnosis(
            Watcher(),  # type: ignore[arg-type]
            task="t",
            repository="acme/widgets",
            outcome=Outcome(framework="pytest", command=("pytest",), status="failed", exit_code=1),
            files=[("big.py", None)],
            round_number=1,
        )

        assert "big.py (too large to quote here" in seen[0]


# ------------------------------------------------------------------- the whole run


class TestPhase2CPipeline:
    async def test_the_whole_thing_runs_end_to_end(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = package_repo(tmp_path / "e2e")
        llm = ScriptedClient(
            {
                "widgets/core.py": BROKEN_CHANGE,
                "tests/test_core.py": ASSERTING_TESTS,
                "diagnosis": diagnosis(
                    {
                        "test": "tests/test_core.py::test_run",
                        "path": "widgets/core.py",
                        "cause": "run() returned 3",
                        "content": REPAIRED_CORE,
                    }
                ),
            }
        )
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=llm,
            steps=phase2c_pipeline(source),
        )

        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 300)

        final = await engine.store.require_run(run.id)
        assert final.status is RunStatus.SUCCEEDED, final.error
        assert final.total_steps == 9
        assert final.progress == 1.0

        events = await engine.store.get_events(run.id)
        types = [event.type for event in events]
        assert types[0] is EventType.RUN_CREATED
        assert types[-1] is EventType.RUN_FINISHED
        assert types.index(EventType.TEST_SUITE_DETECTED) < types.index(EventType.TEST_STARTED)
        assert types.index(EventType.TEST_STARTED) < types.index(EventType.TEST_COMPLETED)
        assert types.index(EventType.TEST_COMPLETED) < types.index(EventType.FAILURE_DIAGNOSED)
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))

        completed = [e for e in events if e.type is EventType.TEST_COMPLETED]
        assert [e.data["status"] for e in completed] == ["failed", "passed"]
        diagnosed = next(e for e in events if e.type is EventType.FAILURE_DIAGNOSED)
        assert diagnosed.step == "diagnose"
        # The event carries the cause, never the file body it proposes to write.
        assert "content" not in diagnosed.data["findings"][0]
        assert diagnosed.data["findings"][0]["cause"]

        await engine.shutdown()

    async def test_a_repository_with_no_suite_still_succeeds(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        source = write_repo(
            tmp_path / "nosuite",
            {
                "pyproject.toml": '[project]\nname = "widgets"\n',
                "widgets/__init__.py": "",
                "widgets/core.py": CORE,
                "widgets/uptime.py": "def uptime() -> int:\n    return 0\n",
            },
        )
        llm = ScriptedClient(
            {
                "widgets/core.py": REPAIRED_CORE,
                "widgets/uptime.py": "def uptime() -> int:\n    return 1\n",
            }
        )
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=llm,
            steps=(
                ValidateRequestStep(),
                ResolveRepositoryStep(),
                LocalClone(source),
                BuildRepositoryProfileStep(),
                SelectRelevantFilesStep(),
                CreateImplementationPlanStep(),
                ApplyImplementationPlanStep(),
                RunTestSuiteStep(),
                DiagnoseFailuresStep(),
            ),
        )

        run = await engine.start(repository="acme/widgets", task="add an uptime helper")
        await engine.wait_for(run.id, 300)

        final = await engine.store.require_run(run.id)
        events = await engine.store.get_events(run.id)
        assert EventType.TEST_SKIPPED in [event.type for event in events]
        assert EventType.TEST_COMPLETED not in [event.type for event in events]
        assert final.status is not RunStatus.FAILED, final.error

        await engine.shutdown()

    async def test_the_workspace_never_touches_the_source_repository(
        self, settings: Settings, python_repo: Path
    ) -> None:
        before = _fingerprint(python_repo)
        llm = ScriptedClient(REPAIRED_REPLIES)

        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=llm,
            steps=phase2c_pipeline(python_repo),
        )
        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 300)

        assert _fingerprint(python_repo) == before
        assert (python_repo / "widgets" / "core.py").read_text(encoding="utf-8") == CORE

        await engine.shutdown()

    async def test_the_events_reach_a_live_subscriber(
        self, settings: Settings, python_repo: Path
    ) -> None:
        bus = EventBus()
        engine = RunEngine(
            store=RunStore(),
            bus=bus,
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient(REPAIRED_REPLIES),
            steps=phase2c_pipeline(python_repo),
        )

        async with bus.subscribe() as subscription:
            run = await engine.start(repository="acme/widgets", task="double what run() returns")
            await engine.wait_for(run.id, 300)

            seen: list[EventType] = []
            while not subscription.queue.empty():
                item = subscription.queue.get_nowait()
                if isinstance(item, ExecutionEvent):
                    seen.append(item.type)

        assert EventType.TEST_SUITE_DETECTED in seen
        assert EventType.TEST_STARTED in seen
        assert EventType.TEST_COMPLETED in seen
        # The verdict a client renders is in the event, not only in the run's final status.
        completed = await engine.store.get_events(run.id)
        result = next(event for event in completed if event.type is EventType.TEST_COMPLETED)
        assert result.data["status"] == "passed"

        await engine.shutdown()
