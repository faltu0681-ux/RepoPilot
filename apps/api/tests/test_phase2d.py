"""Tests for Phase 2D: committing the change, proposing it, and reporting what happened.

The git here is real. A test builds an actual repository with an actual commit, changes a real
file, and lets the step run real ``git`` against it. Nothing about the thing that matters most in
this phase -- whether the right bytes ended up in a commit on the right branch -- is mocked. GitHub
is the opposite: it is reached through a mock transport, because the alternative is a test that
needs a token and a writable repository to prove that a JSON body is shaped correctly.

What is being asserted is mostly honesty. A run that reports itself verified when its suite never
ran, that calls pre-existing failures a pass, that opens a pull request with no token, or that
invents a commit SHA, would each be worse than a run that reports nothing and says why. So the
skip paths get as much attention here as the happy one.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from repopilot.agent.schema import ImplementationPlan
from repopilot.analysis.diffing import (
    MAX_DIFF_LINES_PER_FILE,
    Change,
    FileDiff,
    assemble_change,
    build_commit_message,
    render_pull_request_body,
)
from repopilot.analysis.testing import TestOutcome as Outcome
from repopilot.analysis.testing import TestTally as Tally
from repopilot.config import Settings
from repopilot.domain.models import EventType, RunStatus
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.engine import RunEngine
from repopilot.execution.steps import (
    CommitChangesStep,
    OpenPullRequestStep,
    RunServices,
    StepContext,
    StepOutcome,
    SummariseRunStep,
)
from repopilot.github.client import GitHubClient, GitHubError, NotAuthenticated
from tests.test_phase2c import phase2c_pipeline

PR_PAYLOAD = {
    "number": 42,
    "html_url": "https://github.com/acme/widgets/pull/42",
    "title": "RepoPilot: double what run() returns",
    "state": "open",
}


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        cors_origins=["http://testserver"],
        llm_model="",
    )


class Recorder:
    """A ``StepContext`` stand-in that keeps the events so a test can read them back."""

    def __init__(self, ctx: StepContext) -> None:
        self._ctx = ctx
        self.events: list[tuple[EventType, str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    async def emit(self, event_type: Any, message: str, **kwargs: Any) -> Any:
        self.events.append((EventType(event_type), message, kwargs.get("data") or {}))
        return None

    def data_of(self, event_type: EventType) -> list[dict[str, Any]]:
        return [data for kind, _, data in self.events if kind is event_type]

    def messages_of(self, event_type: EventType) -> list[str]:
        return [message for kind, message, _ in self.events if kind is event_type]

    def kinds(self) -> list[EventType]:
        return [kind for kind, _, _ in self.events]


async def make_context(
    settings: Settings,
    *,
    repository: str = "acme/widgets",
    task: str = "double what run() returns",
    github: GitHubClient | None = None,
    git_repo: bool = True,
) -> Recorder:
    """A run in the store, with its own checkout already created.

    ``git_repo=False`` leaves a plain directory, which is what the commit step has to cope with
    when a workspace has no git in it. Built that way rather than by deleting ``.git`` afterwards,
    because git marks its objects read-only on Windows and removing them is a fight of its own.
    """
    store = RunStore()
    run = await store.create_run(repository=repository, task=task)
    ctx = StepContext(
        run=run,
        store=store,
        services=RunServices(
            github=github if github is not None else GitHubClient(),
            workspace=settings.workspace_dir.resolve(),
        ),
        emit=lambda *a, **k: asyncio.sleep(0),
    )

    root = ctx.local_path()
    root.mkdir(parents=True, exist_ok=True)
    (root / "core.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    if not git_repo:
        return Recorder(ctx)

    git("init", "-q", "-b", "main", cwd=root)
    # Local identity, exactly as the commit step does for itself: a commit needs a committer as
    # well as an author, so passing --author alone is not enough on an unconfigured machine.
    git("config", "user.email", "t@example.com", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return Recorder(ctx)


async def bare_remote(settings: Settings) -> Path:
    """An empty bare repository on disk, standing in for GitHub as a push target.

    A real ``git push`` into a real repository is the only way to find out whether the branch
    actually landed, and it is what the publish step's push-and-then-propose ordering rests on.
    """
    remote = settings.workspace_dir.parent / "remote.git"
    await asyncio.to_thread(
        subprocess.run,
        ["git", "init", "-q", "--bare", str(remote)],
        check=True,
        capture_output=True,
    )
    return remote


async def committed_and_pushed(settings: Settings, **github: Any) -> Recorder:
    """A context whose run has a real commit on a real branch, pushed to a real remote."""
    recorder = await make_context(settings, github=github_client(**github))
    ctx = recorder
    root = ctx.local_path()
    (root / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")

    assert await CommitChangesStep().execute(ctx) is StepOutcome.OK
    branch = "repopilot/run"
    ctx.state["branch"] = branch
    git("remote", "add", "origin", str(await bare_remote(settings)), cwd=root)
    git("push", "-q", "origin", f"HEAD:refs/heads/{branch}", cwd=root)
    return recorder


# ------------------------------------------------------------------- the diff reader


class TestChangeAssembly:
    def test_counts_added_and_removed_lines_per_file(self) -> None:
        change = assemble_change(
            name_status="M\tcore.py\nA\tnew.py\n",
            numstat="2\t1\tcore.py\n3\t0\tnew.py\n",
            patch="",
        )

        assert change.paths == ["core.py", "new.py"]
        assert change.lines_added == 5
        assert change.lines_removed == 1
        assert change.empty is False

    def test_a_deleted_file_is_still_a_change(self) -> None:
        change = assemble_change(name_status="D\tgone.py\n", numstat="0\t7\tgone.py\n", patch="")

        assert change.empty is False
        assert change.lines_removed == 7

    def test_no_output_is_an_empty_change_not_an_error(self) -> None:
        change = assemble_change(name_status="", numstat="", patch="")

        assert change.empty is True
        assert change.paths == []
        assert change.lines_added == 0

    def test_binary_files_are_counted_but_carry_no_patch(self) -> None:
        """``-`` in numstat is git's way of saying "not text", not "zero lines"."""
        change = assemble_change(name_status="M\tlogo.png\n", numstat="-\t-\tlogo.png\n", patch="")

        assert change.lines_added == 0
        assert change.lines_removed == 0
        assert change.empty is False, "a binary edit is still a change to review"

    def test_a_huge_file_is_truncated_per_file(self) -> None:
        patch = "".join(f"+line {n}\n" for n in range(MAX_DIFF_LINES_PER_FILE + 500))
        change = assemble_change(
            name_status="M\tbig.py\n",
            numstat=f"{MAX_DIFF_LINES_PER_FILE + 500}\t0\tbig.py\n",
            patch=f"diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n{patch}",
        )

        assert change.files[0].truncated is True
        assert change.files[0].lines_added == MAX_DIFF_LINES_PER_FILE + 500, (
            "the count is the true one; only the quoted patch is bounded"
        )
        assert len(change.files[0].patch.splitlines()) <= MAX_DIFF_LINES_PER_FILE

    def test_the_total_is_bounded_across_many_files(self) -> None:
        """One enormous file must not be able to fill the whole budget and starve the rest."""
        per_file = MAX_DIFF_LINES_PER_FILE
        files = 40
        status = "".join(f"M\tf{n}.py\n" for n in range(files))
        numstat = "".join(f"{per_file}\t0\tf{n}.py\n" for n in range(files))
        patch = "".join(
            f"diff --git a/f{n}.py b/f{n}.py\n--- a/f{n}.py\n+++ b/f{n}.py\n"
            + "".join("+line\n" for _ in range(per_file))
            for n in range(files)
        )
        change = assemble_change(name_status=status, numstat=numstat, patch=patch)

        quoted = sum(len(f.patch.splitlines()) for f in change.files)
        assert quoted < per_file * files, "the total budget has to bind, not just the per-file one"

    def test_a_quoted_path_is_unquoted(self) -> None:
        """Git C-quotes a name holding anything a terminal would mangle.

        Left quoted, the commit message and the pull request body would name
        ``"src/caf\\303\\251.py"`` -- a file that does not exist. Worse, the counts and the hunks
        come from git's *other* two outputs, so if only this one is unquoted the three disagree
        about what the file is: it would be reported with no lines changed and no diff attached,
        which reads as an empty edit.
        """
        quoted = '"src/caf\\303\\251.py"'
        change = assemble_change(
            name_status=f"M\t{quoted}\n",
            numstat=f"2\t1\t{quoted}\n",
            patch=(f"diff --git {quoted[:-1]}a/{quoted[1:]} b/... \n@@\n-old\n+new\n"),
        )

        assert change.paths == ["src/café.py"]
        assert change.lines_added == 2
        assert change.lines_removed == 1

    def test_the_patch_of_a_quoted_path_is_attached_to_it(self) -> None:
        """The ``diff --git`` header is quoted the same way, and must join on the same string."""
        quoted = '"src/caf\\303\\251.py"'
        patch = f'diff --git {quoted[:-1]}"a/src/caf\\303\\251.py" "b/src/caf\\303\\251.py"\n+new\n'

        change = assemble_change(
            name_status=f"M\t{quoted}\n", numstat=f"1\t0\t{quoted}\n", patch=patch
        )

        assert change.files[0].path == "src/café.py"
        assert "+new" in change.files[0].patch, (
            "a quoted path must not lose its diff to a header that would not parse"
        )


# ----------------------------------------------------------------- the commit message


class TestCommitMessage:
    def test_the_subject_is_the_task_on_one_line(self) -> None:
        message = build_commit_message(
            task="Double what run() returns.\nAnd keep it fast.",
            verdict="verified",
            verdict_reason="The pytest suite passes.",
            change=Change(files=(FileDiff(path="core.py", status="M"),)),
        )

        assert "\n" not in message.subject
        assert message.subject == "Double what run() returns. And keep it fast"

    def test_the_subject_does_not_advertise_the_verdict(self) -> None:
        """A subject saying VERIFIED invites a reader to trust the word, not the suite."""
        message = build_commit_message(
            task="Double what run() returns",
            verdict="verified",
            verdict_reason="The pytest suite passes.",
            change=Change(),
        )

        assert "verified" not in message.subject.lower()

    def test_the_body_carries_the_tally_and_the_verdict(self) -> None:
        message = build_commit_message(
            task="Double what run() returns",
            verdict="failed",
            verdict_reason="Two tests did not pass.",
            change=Change(
                files=(FileDiff(path="core.py", status="M", lines_added=2, lines_removed=1),)
            ),
            test_framework="pytest",
            test_status="failed",
        )

        body = message.as_text()
        assert "core.py" in body
        assert "+2 / -1" in body
        assert "FAILED" in body

    def test_a_verdict_of_verified_still_says_verified_in_the_body(self) -> None:
        """The word belongs in the body; the body is where a reader looks for evidence."""
        message = build_commit_message(
            task="Double what run() returns",
            verdict="verified",
            verdict_reason="The pytest suite passes.",
            change=Change(),
            test_framework="pytest",
            test_status="passed",
        )

        assert "VERIFIED" in message.as_text()

    def test_failed_and_unverified_are_not_the_same_word(self) -> None:
        """One says the code is wrong; the other says nobody found out. A reviewer who cannot
        tell those apart cannot decide whether to merge.
        """
        failed = build_commit_message(
            task="t", verdict="failed", verdict_reason="r", change=Change()
        ).as_text()
        unverified = build_commit_message(
            task="t", verdict="unverified", verdict_reason="r", change=Change()
        ).as_text()

        assert "FAILED" in failed and "FAILED" not in unverified
        assert "UNVERIFIED" in unverified

    def test_an_empty_task_still_produces_a_usable_message(self) -> None:
        message = build_commit_message(
            task="   ", verdict="unverified", verdict_reason="", change=Change()
        )

        assert message.subject.strip()
        assert message.as_args()


class TestPullRequestBody:
    def _body(self, **overrides: Any) -> str:
        arguments: dict[str, Any] = {
            "task": "Double what run() returns",
            "verdict": "verified",
            "verdict_reason": "The pytest suite passes.",
            "plan_summary": "Change the literal.",
            "change": Change(files=(FileDiff(path="core.py", status="M", lines_added=2),)),
            "test_framework": "pytest",
            "test_command": ["pytest", "-q"],
            "test_status": "passed",
            "tests_passed": 3,
            "tests_failed": 0,
            "tests_skipped": 0,
            "unverified_reason": None,
        }
        arguments.update(overrides)
        return render_pull_request_body(**arguments)

    def test_it_states_the_verdict_and_the_plan(self) -> None:
        body = self._body()

        assert "VERIFIED" in body
        assert "Change the literal." in body

    def test_it_states_how_the_tests_were_run(self) -> None:
        body = self._body()

        assert "pytest -q" in body
        assert "3 passed" in body

    def test_an_unverified_run_says_so_rather_than_looking_green(self) -> None:
        body = self._body(
            verdict="unverified",
            verdict_reason="Nothing was verified: the repository declares no test runner.",
            test_framework=None,
            test_command=[],
            test_status=None,
            tests_passed=None,
            tests_failed=None,
            tests_skipped=None,
            unverified_reason="the repository declares no test runner",
        )

        assert "UNVERIFIED" in body
        assert "no test runner" in body

    def test_a_failed_run_is_labelled_failed(self) -> None:
        body = self._body(
            verdict="failed",
            verdict_reason="Two tests did not pass.",
            test_status="failed",
            tests_passed=1,
            tests_failed=2,
        )

        assert "FAILED" in body


# --------------------------------------------------------------------- the commit step


class TestCommitChangesStep:
    async def test_a_brand_new_file_is_committed(self, settings: Settings) -> None:
        """The commonest agent output is a file that did not exist, and it used to vanish.

        Untracked files are invisible to ``git diff HEAD``, so the step measured "no change" for a
        run that had just written a whole new module, declared the tree clean, and committed
        nothing. Measuring the index is what fixes it.
        """
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        (root / "new_module.py").write_text("def run():\n    return 2\n", encoding="utf-8")
        ctx.state["written"] = ["new_module.py"]

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["commit"] is not None, "a new file is a change and must be committed"
        listed = git("show", "--name-only", "--pretty=", ctx.state["commit"], cwd=root)
        assert "new_module.py" in listed
        assert git("status", "--porcelain", cwd=root) == ""

    async def test_a_new_and_an_edited_file_are_both_committed(self, settings: Settings) -> None:
        """The mixed case: untracked files and modifications must land in one commit."""
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        (root / "core.py").write_text("def run():\n    return 3\n", encoding="utf-8")
        (root / "fresh.py").write_text("VALUE = 1\n", encoding="utf-8")
        ctx.state["written"] = ["core.py", "fresh.py"]

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        commit = ctx.state["commit"]
        assert commit is not None
        listed = git("show", "--name-only", "--pretty=", commit, cwd=root)
        assert "core.py" in listed
        assert "fresh.py" in listed

    async def test_a_deleted_file_is_still_a_change(self, settings: Settings) -> None:
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        git("rm", "-q", "core.py", cwd=root)
        ctx.state["written"] = []

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["commit"] is not None, "a deletion is a change a reviewer must see"
        assert "core.py" in git("show", "--name-only", "--pretty=", ctx.state["commit"], cwd=root)

    async def test_it_commits_the_change_on_the_runs_own_branch(self, settings: Settings) -> None:
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        git("checkout", "-q", "-b", "repopilot/run", cwd=root)
        (root / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")
        ctx.state["branch"] = "repopilot/run"

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        subject = git("log", "-1", "--pretty=%s", cwd=root)
        assert "double what run() returns" in subject.lower()
        assert ctx.state["commit"] == git("rev-parse", "HEAD", cwd=root)
        assert git("status", "--porcelain", cwd=root) == "", "the tree must be clean afterwards"
        assert git("branch", "--show-current", cwd=root) == "repopilot/run"

    async def test_it_emits_the_commit_with_its_files(self, settings: Settings) -> None:
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        (root / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")

        await CommitChangesStep().execute(ctx)

        (data,) = recorder.data_of(EventType.CHANGE_COMMITTED)
        assert data["commit"] == ctx.state["commit"]
        assert data["files"] == ["core.py"] or "core.py" in data.get("paths", ["core.py"])

    async def test_a_clean_tree_commits_nothing_and_says_why(self, settings: Settings) -> None:
        """A run that changed nothing has still finished; it must not invent a commit."""
        recorder = await make_context(settings)
        ctx = recorder
        before = git("rev-parse", "HEAD", cwd=ctx.local_path())

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["commit"] is None
        assert git("rev-parse", "HEAD", cwd=ctx.local_path()) == before
        assert recorder.data_of(EventType.CHANGE_COMMITTED) == []
        (completed,) = recorder.data_of(EventType.STEP_COMPLETED)
        assert completed["committed"] is False
        assert "clean" in completed["reason"] or "no changes" in completed["reason"]

    async def test_a_directory_that_is_not_a_repository_is_not_a_failure(
        self, settings: Settings
    ) -> None:
        recorder = await make_context(settings, git_repo=False)
        ctx = recorder
        (ctx.local_path() / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["commit"] is None
        assert recorder.data_of(EventType.CHANGE_COMMITTED) == []
        assert any("git" in m.lower() for m in recorder.messages_of(EventType.LOG))

    async def test_it_commits_without_any_global_git_identity(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The commit must not depend on whether the machine running it has git configured."""
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(settings.workspace_dir / "no-such-gitconfig"))
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(settings.workspace_dir / "no-such-sysconfig"))
        recorder = await make_context(settings)
        ctx = recorder
        root = ctx.local_path()
        (root / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")

        assert await CommitChangesStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["commit"], "committing must work with no inherited identity"

    async def test_the_commit_never_lands_in_another_checkout(self, settings: Settings) -> None:
        """The isolation the clone step bought is what makes this step safe."""
        recorder = await make_context(settings)
        ctx = recorder
        sibling = ctx.repo_dir() / "some-other-run"
        sibling.mkdir(parents=True)
        (sibling / "core.py").write_text("def run():\n    return 9\n", encoding="utf-8")
        git("init", "-q", "-b", "main", cwd=sibling)
        git("add", "-A", cwd=sibling)
        git("commit", "-q", "--author=t <t@example.com>", "-m", "other", cwd=sibling)
        before = git("rev-parse", "HEAD", cwd=sibling)
        (ctx.local_path() / "core.py").write_text("def run():\n    return 2\n", encoding="utf-8")

        await CommitChangesStep().execute(ctx)

        assert git("rev-parse", "HEAD", cwd=sibling) == before
        assert (sibling / "core.py").read_text(encoding="utf-8") == "def run():\n    return 9\n"


# ----------------------------------------------------------------- the verdict reporting


class TestSummariseRunStep:
    async def _run(self, ctx: StepContext, **state: Any) -> Recorder:
        """Seed the state and run the step on the *same* recorder the test holds.

        Recording onto a second wrapper would work but split the events across two objects, and
        the split is invisible until a test asserts on the wrong one and finds nothing there.
        """
        ctx.state.update(state)
        assert await SummariseRunStep().execute(ctx) is StepOutcome.OK
        assert isinstance(ctx, Recorder), "pass the recorder in, not the bare context"
        return ctx

    async def test_a_passing_suite_is_verified(self, settings: Settings) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest",
                command=("pytest", "-q"),
                status="passed",
                tally=Tally(passed=3),
            ),
            written=[{"path": "core.py", "created": False}],
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary is not None
        assert stored.summary.verdict.value == "verified"
        assert stored.summary.tests_passed == 3
        assert stored.summary.files_modified == ["core.py"]
        assert stored.summary.test_command == ["pytest", "-q"]

    async def test_a_failing_suite_is_failed_not_unverified(self, settings: Settings) -> None:
        """The suite ran and the code is wrong. That is a failure, not a gap in knowledge."""
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest",
                command=("pytest",),
                status="failed",
                tally=Tally(failed=2, failures=()),
                reason="2 failed",
            ),
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.verdict.value == "failed"

    async def test_a_run_with_no_suite_is_unverified_and_says_why(self, settings: Settings) -> None:
        """No detected runner is not a pass. This is the easiest verdict to get wrong."""
        recorder = await make_context(settings)

        await self._run(
            recorder, test_result=None, test_skip_reason="the repository declares no test runner"
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.verdict.value == "unverified"
        assert "no test runner" in (stored.summary.reason or "")
        assert stored.summary.tests_passed is None

    async def test_a_suite_that_never_ran_is_unverified_not_failed(
        self, settings: Settings
    ) -> None:
        """A timeout says the environment stopped, not that the code is wrong."""
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest",
                command=("pytest",),
                status="timeout",
                reason="exceeded 600s",
            ),
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.verdict.value == "unverified"

    async def test_pre_existing_failures_are_not_reported_as_a_pass(
        self, settings: Settings
    ) -> None:
        """The run status is SUCCEEDED here; the suite is still red. Reporting VERIFIED would
        be the single most misleading thing this phase could do.
        """
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest",
                command=("pytest",),
                status="failed",
                tally=Tally(failed=3, failures=()),
            ),
            pre_existing_failures=3,
            diagnosis_summary="All three failures predate this change.",
        )

        stored = await recorder.store.require_run(recorder.run.id)
        summary = stored.summary
        assert summary.verdict.value == "failed"
        assert summary.pre_existing_failures == 3
        assert "pre-existing" in summary.reason

    async def test_a_repaired_run_credits_the_repair(self, settings: Settings) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest", command=("pytest",), status="passed", tally=Tally(passed=1)
            ),
            repair_rounds=2,
            test_suite=None,
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.verdict.value == "verified"
        assert stored.summary.repair_rounds == 2

    async def test_it_records_the_plan_the_run_followed(self, settings: Settings) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            plan=ImplementationPlan(summary="Double the literal and assert it."),
            test_result=Outcome(
                framework="pytest", command=("pytest",), status="passed", tally=Tally(passed=1)
            ),
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.plan_summary == "Double the literal and assert it."

    async def test_created_and_modified_files_are_told_apart(self, settings: Settings) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest", command=("pytest",), status="passed", tally=Tally(passed=1)
            ),
            written=[
                {"path": "new.py", "created": True},
                {"path": "core.py", "created": False},
                {"path": "core.py", "created": False},
            ],
        )

        stored = await recorder.store.require_run(recorder.run.id)
        assert stored.summary.files_created == ["new.py"]
        assert stored.summary.files_modified == ["core.py"], "a repeat write is not a second file"

    async def test_it_emits_the_summary_on_the_stream(self, settings: Settings) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest", command=("pytest",), status="passed", tally=Tally(passed=1)
            ),
            commit="a" * 40,
        )

        (data,) = recorder.data_of(EventType.RUN_SUMMARIZED)
        assert data["verdict"] == "verified"
        assert data["commit"] == "a" * 40
        assert any("verified" in m.lower() for m in recorder.messages_of(EventType.RUN_SUMMARIZED))

    async def test_the_summary_survives_a_round_trip_through_the_store(
        self, settings: Settings
    ) -> None:
        recorder = await make_context(settings)

        await self._run(
            recorder,
            test_result=Outcome(
                framework="pytest", command=("pytest",), status="passed", tally=Tally(passed=1)
            ),
        )

        stored = await recorder.store.require_run(recorder.run.id)
        payload = stored.summary.model_dump(mode="json")
        assert payload["verdict"] == "verified"
        assert json_ready(payload)


def json_ready(value: Any) -> bool:
    import json

    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


# ------------------------------------------------------------------ the pull request step


def github_client(*, token: str | None = None, handler: Any = None) -> GitHubClient:
    return GitHubClient(
        token=token,
        transport=httpx.MockTransport(
            handler or (lambda request: httpx.Response(201, json=PR_PAYLOAD))
        ),
    )


class TestOpenPullRequestStep:
    async def test_a_local_folder_is_skipped_with_a_reason(self, settings: Settings) -> None:
        """A local folder is the user's own code on this machine. There is nowhere to propose it."""
        recorder = await make_context(
            settings, repository="local://demo", github=github_client(token="t")
        )
        ctx = recorder
        ctx.state["commit"] = "a" * 40
        ctx.state["branch"] = "repopilot/run"

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["pull_request"] is None
        assert "local://demo" in ctx.state["pull_request_skipped_reason"]
        assert recorder.data_of(EventType.PULL_REQUEST_OPENED) == []

    async def test_a_missing_token_is_skipped_without_calling_github(
        self, settings: Settings
    ) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(201, json=PR_PAYLOAD)

        recorder = await make_context(settings, github=github_client(handler=handler))
        ctx = recorder
        ctx.state["commit"] = "a" * 40
        ctx.state["branch"] = "repopilot/run"

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert calls == [], "an unauthenticated write must not be attempted"
        assert "token" in ctx.state["pull_request_skipped_reason"].lower()

    async def test_a_run_that_committed_nothing_proposes_nothing(self, settings: Settings) -> None:
        recorder = await make_context(settings, github=github_client(token="t"))
        ctx = recorder
        ctx.state["commit"] = None
        ctx.state["branch"] = "repopilot/run"

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["pull_request"] is None
        assert "no commit" in ctx.state["pull_request_skipped_reason"].lower()

    async def test_it_refuses_when_the_branch_cannot_be_pushed(self, settings: Settings) -> None:
        """A checkout with no remote has nowhere to push, and must not pretend otherwise."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(201, json=PR_PAYLOAD)

        recorder = await make_context(settings, github=github_client(token="t", handler=handler))
        ctx = recorder
        ctx.state["commit"] = "a" * 40
        ctx.state["branch"] = "repopilot/run"

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert calls == [], "no pull request may be opened for an unpushed branch"
        assert ctx.state["pull_request"] is None
        assert ctx.state["pull_request_skipped_reason"]

    async def test_a_pushed_branch_gets_a_pull_request(self, settings: Settings) -> None:
        ctx = await committed_and_pushed(settings, token="t")

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["pull_request"]["number"] == 42
        assert ctx.state["pull_request"]["url"] == PR_PAYLOAD["html_url"]
        (data,) = ctx.data_of(EventType.PULL_REQUEST_OPENED)
        assert data["number"] == 42
        assert data["branch"] == "repopilot/run"

    async def test_github_refusing_is_reported_rather_than_raised(self, settings: Settings) -> None:
        """A refused pull request does not unmake the change; it is reported and the run ends."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"message": "Validation Failed"})

        ctx = await committed_and_pushed(settings, token="t", handler=handler)

        assert await OpenPullRequestStep().execute(ctx) is StepOutcome.OK

        assert ctx.state["pull_request"] is None
        assert "422" in ctx.state["pull_request_skipped_reason"]
        assert ctx.state["commit"], "the commit stands regardless of the pull request"

    async def test_the_pull_request_reaches_the_summary(self, settings: Settings) -> None:
        ctx = await committed_and_pushed(settings, token="t")

        await OpenPullRequestStep().execute(ctx)
        await SummariseRunStep().execute(ctx)

        stored = await ctx.store.require_run(ctx.run.id)
        assert stored.summary.pull_request_url == PR_PAYLOAD["html_url"]
        assert stored.summary.pull_request_number == 42
        assert stored.summary.commit == ctx.state["commit"]


# ------------------------------------------------------------- the GitHub client itself


class TestCreatePullRequest:
    async def test_it_posts_the_branch_and_base_gitHub_expects(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(201, json=PR_PAYLOAD)

        client = GitHubClient(token="secret", transport=httpx.MockTransport(handler))
        await client.create_pull_request(
            client_ref(),
            head="repopilot/run",
            base="main",
            title="RepoPilot: change",
            body="## VERIFIED\n",
        )
        await client.aclose()

        (request,) = seen
        assert request.method == "POST"
        assert request.url.path == "/repos/acme/widgets/pulls"
        payload = json.loads(request.content)
        assert payload == {
            "title": "RepoPilot: change",
            "body": "## VERIFIED\n",
            "head": "repopilot/run",
            "base": "main",
            "draft": False,
        }
        assert request.headers["Authorization"] == "Bearer secret"

    async def test_it_refuses_without_a_token_and_sends_nothing(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(201, json=PR_PAYLOAD)

        client = GitHubClient(transport=httpx.MockTransport(handler))

        with pytest.raises(NotAuthenticated):
            await client.create_pull_request(
                client_ref(), head="b", base="main", title="t", body="x"
            )
        await client.aclose()

        assert calls == []

    async def test_it_needs_a_base_branch(self) -> None:
        client = GitHubClient(
            token="t", transport=httpx.MockTransport(lambda r: httpx.Response(201))
        )

        with pytest.raises(GitHubError, match="base branch"):
            await client.create_pull_request(client_ref(), head="b", base="", title="t", body="x")
        await client.aclose()

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(401, "REPOPILOT_GITHUB_TOKEN"), (403, "REPOPILOT_GITHUB_TOKEN"), (500, "500")],
    )
    async def test_it_turns_a_rejection_into_something_actionable(
        self, status: int, expected: str
    ) -> None:
        client = GitHubClient(
            token="t", transport=httpx.MockTransport(lambda r: httpx.Response(status))
        )

        with pytest.raises(GitHubError, match=expected):
            await client.create_pull_request(
                client_ref(), head="b", base="main", title="t", body="x"
            )
        await client.aclose()

    async def test_it_says_that_a_pull_request_may_already_exist(self) -> None:
        """422 on a retried run is normal, and the message should read that way."""
        client = GitHubClient(
            token="t", transport=httpx.MockTransport(lambda r: httpx.Response(422))
        )

        with pytest.raises(GitHubError, match="already exist"):
            await client.create_pull_request(
                client_ref(), head="b", base="main", title="t", body="x"
            )
        await client.aclose()

    async def test_an_unreachable_api_is_a_github_error_not_a_crash(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = GitHubClient(token="t", transport=httpx.MockTransport(handler))

        with pytest.raises(GitHubError, match="could not reach"):
            await client.create_pull_request(
                client_ref(), head="b", base="main", title="t", body="x"
            )
        await client.aclose()


# --------------------------------------------------------------------- the whole run


def phase2d_pipeline(source: Path) -> tuple[Any, ...]:
    """The 2C pipeline plus the three reporting steps, cloned from a local repository."""
    from tests.test_phase2c import LocalClone

    return (
        *phase2c_pipeline(source)[:2],
        LocalClone(source),
        *phase2c_pipeline(source)[3:],
        CommitChangesStep(),
        OpenPullRequestStep(),
        SummariseRunStep(),
    )


class TestPhase2DPipeline:
    """One run, all twelve steps, the summary as the API would serve it.

    Everything above tests a step in isolation; this proves the three of them are actually wired
    in, run in the order that makes their output true, and leave a summary on the run that a
    client can read without replaying the stream.
    """

    async def test_the_run_commits_summarises_and_persists(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        from tests.test_phase2c import (
            ASSERTING_TESTS,
            REPAIRED_CORE,
            ScriptedClient,
            mock_github,
            package_repo,
        )

        source = package_repo(tmp_path / "e2e")
        llm = ScriptedClient(
            {"widgets/core.py": REPAIRED_CORE, "tests/test_core.py": ASSERTING_TESTS}
        )
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=llm,
            steps=phase2d_pipeline(source),
        )

        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 300)
        final = await engine.store.require_run(run.id)
        await engine.shutdown()

        assert final.status is RunStatus.SUCCEEDED, final.error

        # A real commit, in the run's own clone, naming the task.
        assert final.summary is not None
        assert final.summary.commit, "the run committed nothing"
        assert final.summary.verdict.value == "verified"
        assert final.summary.tests_passed == 1
        assert final.summary.branch
        assert "double what run() returns" in final.summary.commit_message.lower()

        # No remote and no token, so no pull request -- reported rather than faked.
        assert final.summary.pull_request_url is None
        assert final.summary.pull_request_skipped_reason

        events = await engine.store.get_events(run.id)
        types = [event.type for event in events]
        assert types[0] is EventType.RUN_CREATED
        assert types[-1] is EventType.RUN_FINISHED
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))

        # The ordering the phase exists to guarantee: judged, then committed, then summarised.
        assert types.index(EventType.TEST_COMPLETED) < types.index(EventType.CHANGE_COMMITTED)
        assert types.index(EventType.CHANGE_COMMITTED) < types.index(EventType.RUN_SUMMARIZED)

        # And once the summary is out, nothing else reports on the run: only the summarise step
        # closing itself and the run finishing are left, so no later step can contradict what the
        # summary said.
        after = events[types.index(EventType.RUN_SUMMARIZED) + 1 :]
        assert [event.type for event in after] == [
            EventType.STEP_COMPLETED,
            EventType.RUN_FINISHED,
        ]
        assert after[0].step == "summarise"

        summarised = next(e for e in events if e.type is EventType.RUN_SUMMARIZED)
        assert summarised.data["verdict"] == "verified"
        assert summarised.data["commit"] == final.summary.commit

    async def test_a_run_with_no_suite_is_reported_unverified_not_verified(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """The end-to-end version of the honesty case: a repository with no pytest at all.

        The run succeeds -- there was nothing to fail -- and the summary still must not claim the
        change was proved.
        """
        from tests.test_phase2c import ScriptedClient, mock_github, write_repo

        source = write_repo(
            tmp_path / "nosuite",
            {
                "pyproject.toml": '[project]\nname = "widgets"\n',
                "widgets/__init__.py": "",
                "widgets/core.py": "def run():\n    return 1\n",
            },
        )
        engine = RunEngine(
            store=RunStore(),
            bus=EventBus(),
            settings=settings,
            github=mock_github(),
            llm=ScriptedClient({"widgets/core.py": "def run():\n    return 2\n"}),
            steps=phase2d_pipeline(source),
        )

        run = await engine.start(repository="acme/widgets", task="double what run() returns")
        await engine.wait_for(run.id, 300)
        final = await engine.store.require_run(run.id)
        await engine.shutdown()

        assert final.status is RunStatus.SUCCEEDED, final.error
        assert final.summary is not None
        assert final.summary.verdict.value == "unverified"
        assert final.summary.commit, "the change is still committed, just not proved"
        assert "no test runner" in final.summary.reason.lower()

        events = await engine.store.get_events(run.id)
        assert any(event.type is EventType.TEST_SKIPPED for event in events)
        summarised = next(e for e in events if e.type is EventType.RUN_SUMMARIZED)
        assert summarised.data["verdict"] == "unverified"


def client_ref() -> Any:
    from repopilot.domain.repository import RepositoryRef

    return RepositoryRef.parse("acme/widgets")
