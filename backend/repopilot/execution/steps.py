"""The step pipeline: the seam where the agent loop plugs in.

A run is a list of :class:`Step` objects executed in order. The engine knows nothing about what
any step does; it only knows how to start one, watch it emit, and decide what to do when it
raises.

The Phase 1 pipeline does real work: validate the request, ask GitHub about the repository,
clone it, and inspect what was cloned. The agent steps were appended to the same list as each
phase landed:

    validate -> resolve -> clone -> inspect -> profile -> select_files -> plan -> apply_plan
    -> test -> diagnose

Opening the pull request (2D) is still to come. Everything that step needs is already in place:

* :class:`StepContext` carries the run, settings, the store, the workspace, a run-scoped
  ``state`` dict for passing results forward, and a bound ``emit`` coroutine.
* :class:`StepOutcome` separates ``ok`` from ``retry`` from ``abort``, and ``Run.attempts``
  counts retries, so the fix loop becomes engine policy instead of a rewrite.
* :class:`RunServices` is the container for the LLM client and for the shell sandbox the test
  step runs a suite in. Steps take collaborators from here, never by importing them, which is
  what keeps them substitutable in tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import stat
import subprocess
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from repopilot.agent.diagnose import DiagnosisRejectedError, generate_diagnosis
from repopilot.agent.implement import (
    ChangeRejectedError,
    UnsafePathError,
    generate_file_change,
    read_for_edit,
    resolve_write_target,
    validate_whole_file,
    write_file,
)
from repopilot.agent.llm import ChatClient, LLMError, LLMNotConfiguredError
from repopilot.agent.plan import PlanRejectedError, generate_plan
from repopilot.agent.prompts import DIAGNOSIS_SNIPPET_CHARS
from repopilot.agent.schema import FileSelection, ImplementationPlan, TestFailureReport
from repopilot.analysis.profile import (
    IGNORED_DIRECTORIES as _IGNORED_DIRECTORIES,
)
from repopilot.analysis.profile import (
    ProfileError,
    RepositoryProfile,
    build_profile,
)
from repopilot.analysis.relevance import (
    DEFAULT_CANDIDATE_LIMIT,
    CandidateFile,
    rank_candidates,
    tokenize,
)
from repopilot.analysis.testing import (
    TestOutcome,
    TestSuite,
    collected_nothing,
    detect_test_suite,
    parse_test_output,
    tail_output,
)
from repopilot.core.errors import RepopilotError
from repopilot.domain.models import EventLevel, EventType, Run
from repopilot.domain.repository import RepositoryRef
from repopilot.domain.store import RunStore
from repopilot.execution.sandbox import CommandResult, ShellSandbox
from repopilot.github.client import GitHubClient, RepositoryInfo

__all__ = [
    "IGNORED_DIRECTORIES",
    "IGNORED_SUFFIXES",
    "ApplyImplementationPlanStep",
    "BuildRepositoryProfileStep",
    "CloneRepositoryStep",
    "CreateImplementationPlanStep",
    "DiagnoseFailuresStep",
    "GitResult",
    "InspectRepositoryStep",
    "ResolveRepositoryStep",
    "RunServices",
    "RunTestSuiteStep",
    "SelectRelevantFilesStep",
    "Step",
    "StepContext",
    "StepFailed",
    "StepOutcome",
    "ValidateRequestStep",
    "default_pipeline",
]

#: Directories never worth descending into when inspecting a repository.
#: Re-exported from :mod:`repopilot.analysis.profile` so the Phase 1 inspect step and the
#: Phase 2 analysis pass cannot drift apart.
IGNORED_DIRECTORIES = _IGNORED_DIRECTORIES

IGNORED_SUFFIXES = frozenset(
    {
        ".lock",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".woff",
        ".woff2",
    }
)

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
    ".swift": "Swift",
    ".sh": "Shell",
}

#: Markers that reveal test tooling, mapped to the framework they indicate.
FRAMEWORK_MARKERS: dict[str, tuple[str, ...]] = {
    "pytest": ("pytest",),
    "unittest": ("unittest",),
    "jest": ('"jest"',),
    "vitest": ("vitest",),
    "mocha": ('"mocha"',),
    "testify": ("testify",),
}

MANIFEST_FILES = frozenset(
    {
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "Cargo.toml",
    }
)

EmitFn = Callable[..., Awaitable[Any]]

#: Wall-clock ceiling for any single git invocation.
GIT_TIMEOUT_SECONDS = 600.0


class StepOutcome(StrEnum):
    """What a step is telling the engine to do next."""

    OK = "ok"
    #: Ask the engine to run the pipeline again with an incremented attempt counter.
    RETRY = "retry"
    #: Stop immediately and mark the run failed.
    ABORT = "abort"


class StepFailed(RepopilotError):
    """A step failed in a way that should be shown to the user."""

    status_code = 500

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(slots=True)
class RunServices:
    """Long-lived collaborators available to every step in a run.

    ``github`` is the GitHub REST client. ``llm`` is the reasoning backend, which is ``None``
    when no model is configured; steps that need it raise ``LLMNotConfiguredError`` rather than
    inventing output. ``sandbox`` runs a command in the run's own checkout, which Phase 2C needs
    and Phase 2D will need again to describe a diff. The diff writer arrives with 2D.
    """

    github: GitHubClient
    workspace: Path
    llm: ChatClient | None = None
    sandbox: ShellSandbox = field(default_factory=ShellSandbox)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepContext:
    """Everything a step is allowed to touch."""

    run: Run
    store: RunStore
    services: RunServices
    emit: EmitFn
    is_cancelled: Callable[[], bool] = lambda: False
    attempt: int = 0
    #: The engine's own retry limit, so a step can tell whether asking to retry would be heard.
    #: Zero means "not supplied", which reads as no further attempts.
    max_attempts: int = 0
    #: Scratch space for passing results from one step to the next.
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def repository(self) -> RepositoryRef:
        return self.run.repository

    @property
    def attempt_label(self) -> str:
        return "" if self.attempt == 0 else f" (attempt {self.attempt + 1})"

    @property
    def can_retry(self) -> bool:
        """True when the engine would honour another attempt at this pipeline.

        A step that wants to loop has to ask this question rather than assume it: without it the
        only way to bound a loop is to guess at the engine's policy, and a guess that is wrong
        turns a failing test into an endless run.
        """
        return self.max_attempts > 0 and self.attempt + 1 < self.max_attempts

    def repo_dir(self) -> Path:
        """The per-repository directory that holds one isolated workspace per run.

        Phase 1 cloned every run of a repository into a single shared directory. Phase 2
        writes code, so two runs of the same repository must not be able to see each other's
        edits, and a run must never touch a checkout the user made themselves.
        """
        return self.services.workspace / self.run.repository.slug.replace("/", "__")

    def local_path(self) -> Path:
        """This run's isolated checkout: ``<workspace>/<owner>__<repo>/<run_id>``."""
        return self.repo_dir() / self.run.id


class Step(ABC):
    """One unit of run work."""

    name: ClassVar[str]
    title: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, ctx: StepContext) -> StepOutcome:
        """Do the work.

        Emit progress with ``await ctx.emit(...)``. Raise :class:`StepFailed` to fail the run.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------- the steps


class ValidateRequestStep(Step):
    """Reject work that cannot succeed, before spending a single network call."""

    name = "validate"
    title = "Validate request"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        ref = ctx.repository
        await ctx.emit(EventType.STEP_STARTED, f"Validating request for {ref.slug}")

        if not ctx.run.task.strip():
            raise StepFailed("the task is empty")

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Task accepted ({len(ctx.run.task)} characters)",
            data={
                "task": ctx.run.task,
                "repository": ctx.repository.slug,
                "branch": ctx.run.branch,
                "clone_url": ctx.repository.clone_url,
            },
        )
        return StepOutcome.OK


class ResolveRepositoryStep(Step):
    """Ask GitHub what this repository actually is."""

    name = "resolve"
    title = "Resolve repository"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        ref = ctx.repository
        await ctx.emit(EventType.STEP_STARTED, f"Querying GitHub for {ref.slug}")

        if not ctx.services.github.authenticated:
            await ctx.emit(
                EventType.LOG,
                "No GitHub token configured; private repositories will not resolve",
                level=EventLevel.WARN,
            )

        info = await ctx.services.github.get_repository(ref)
        ctx.state["repository_info"] = info

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Resolved {info.full_name} on '{info.default_branch}'",
            data={
                "full_name": info.full_name,
                "default_branch": info.default_branch,
                "private": info.private,
                "fork": info.fork,
                "language": info.language,
                "stars": info.stargazers_count,
                "open_issues": info.open_issues_count,
                "license": info.license,
                "archived": info.archived,
                "size_kb": info.size_kb,
            },
        )
        if info.archived:
            await ctx.emit(
                EventType.LOG,
                "This repository is archived; nothing will be able to merge",
                level=EventLevel.WARN,
            )
        return StepOutcome.OK


class CloneRepositoryStep(Step):
    """Clone the repository into the workspace. Real work, not a simulation."""

    name = "clone"
    title = "Clone repository"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        info: RepositoryInfo | None = ctx.state.get("repository_info")
        if info is None:
            raise StepFailed("cannot clone before the repository has been resolved")
        if info.archived:
            raise StepFailed(f"{info.full_name} is archived and cannot be worked on")

        target = ctx.local_path()
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)

        if target.exists():
            await ctx.emit(
                EventType.LOG,
                f"Replacing the existing checkout at {target}",
                level=EventLevel.DEBUG,
            )
            try:
                await asyncio.to_thread(_force_rmtree, target)
            except OSError as exc:
                raise StepFailed(
                    f"Could not clear the previous checkout at {target}: {exc}"
                ) from exc

        await ctx.emit(
            EventType.STEP_STARTED,
            f"Cloning {info.clone_url} (branch {info.default_branch})",
            data={"target": str(target), "expected_bytes": info.size_kb * 1024},
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await _run_git(
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            info.default_branch,
            info.clone_url,
            str(target),
        )

        if result.returncode != 0:
            raise StepFailed(
                f"git clone exited {result.returncode}: {_tail(result.stderr)}", retryable=True
            )

        elapsed = round(loop.time() - started, 3)
        head = await _git_head(target)
        ctx.state["head_sha"] = head
        ctx.state["clone_seconds"] = elapsed

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Cloned {info.full_name} at {(head or '?')[:8]} in {elapsed}s",
            data={
                "path": str(target),
                "head": head,
                "seconds": elapsed,
                "branch": info.default_branch,
            },
        )
        return StepOutcome.OK


class InspectRepositoryStep(Step):
    """Walk the clone and report what is in it. The first piece of repository analysis."""

    name = "inspect"
    title = "Inspect repository"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        root = ctx.local_path()
        if not root.is_dir():
            raise StepFailed(f"{root} does not exist; the clone step must run first")

        await ctx.emit(EventType.STEP_STARTED, f"Inspecting {ctx.repository.slug}")

        summary = await asyncio.to_thread(_inspect_tree, root)
        ctx.state["inspection"] = summary

        languages: Counter[str] = summary["languages"]
        top_language = languages.most_common(1)[0][0] if languages else "unknown"
        await ctx.emit(
            EventType.LOG,
            f"{summary['file_count']} files, "
            f"{summary['total_bytes'] / 1024:.0f} KiB, mostly {top_language}",
            data={
                "file_count": summary["file_count"],
                "total_bytes": summary["total_bytes"],
                "top_level": summary["top_level"],
            },
        )
        for language, count in languages.most_common(5):
            await ctx.emit(
                EventType.LOG,
                f"{language}: {count} files",
                level=EventLevel.DEBUG,
                data={"language": language, "files": count},
            )

        frameworks: set[str] = summary["test_frameworks"]
        if frameworks:
            await ctx.emit(EventType.LOG, f"Test tooling detected: {', '.join(sorted(frameworks))}")
        else:
            await ctx.emit(
                EventType.LOG,
                "No test framework detected in the manifests",
                level=EventLevel.WARN,
            )

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Inspected {summary['file_count']} files across {len(languages)} language(s)",
            data={
                "file_count": summary["file_count"],
                "languages": dict(languages.most_common(10)),
                "test_frameworks": sorted(frameworks),
                "top_level": summary["top_level"],
            },
        )
        return StepOutcome.OK


# ------------------------------------------------------------------- inspection


def _inspect_tree(root: Path) -> dict[str, Any]:
    """Count files and infer languages. Blocking; the caller runs it in a thread."""
    languages: Counter[str] = Counter()
    frameworks: set[str] = set()
    file_count = 0
    total_bytes = 0
    top_level: set[str] = set()

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if len(relative.parts) == 1:
            top_level.add(relative.parts[0])
        if path.is_dir():
            continue

        file_count += 1
        with contextlib.suppress(OSError):
            total_bytes += path.stat().st_size

        if path.name in MANIFEST_FILES:
            frameworks |= _frameworks_in(path)
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
        if language:
            languages[language] += 1

    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "languages": languages,
        "test_frameworks": frameworks,
        "top_level": sorted(top_level),
    }


def _frameworks_in(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return set()
    found = {
        framework
        for framework, markers in FRAMEWORK_MARKERS.items()
        if any(marker in text for marker in markers)
    }
    if path.name == "package.json" and '"test"' in text:
        found.add("npm test")
    return found


# ---------------------------------------------------------------------- helpers


def _force_rmtree(path: Path, attempts: int = 4) -> None:
    """Delete a tree even when Git has marked parts of it read-only.

    Git writes its pack files read-only on Windows, so a plain :func:`shutil.rmtree` raises
    ``PermissionError`` and leaves a half-deleted directory behind. The next clone then fails
    with "destination path already exists and is not an empty directory", which reads like a
    repository problem but is really a leftover one. Clearing the read-only bit and retrying
    briefly also absorbs the short-lived locks held by antivirus and file indexers.
    """
    last: OSError | None = None
    for attempt in range(attempts):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_force_unlink)
            return
        except OSError as exc:  # noqa: PERF203 - the retry is the point
            last = exc
            time.sleep(0.1 * (attempt + 1))
    raise OSError(f"could not remove {path}: {last}")


def _force_unlink(func: Callable[[str], Any], target: str, _exc: OSError) -> None:
    """Clear the read-only bit, then retry the delete that just failed."""
    os.chmod(target, stat.S_IWRITE)
    func(target)


async def _git_head(path: Path) -> str | None:
    result = await _run_git("-C", str(path), "rev-parse", "HEAD")
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str


async def _run_git(*args: str) -> GitResult:
    """Run git in a worker thread rather than through the event loop.

    On Windows, uvicorn hands a reload-mode server a ``SelectorEventLoop`` so that its own
    reloader can fork, and a selector loop cannot spawn subprocesses at all. Anything that shells
    out to git therefore dies with ``NotImplementedError`` the moment you run the app the way the
    README says to, and only on Windows. Driving ``subprocess`` from a thread keeps the clone step
    working under every loop policy, and keeps the event loop free while git works.
    """

    def call() -> GitResult:
        process = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            # Never let git stop and ask for a password on an unattended clone.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return GitResult(process.returncode, process.stdout, process.stderr)

    try:
        async with asyncio.timeout(GIT_TIMEOUT_SECONDS):
            return await asyncio.to_thread(call)
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        raise StepFailed(f"git {args[0]} timed out after {GIT_TIMEOUT_SECONDS}s") from exc


def _tail(raw: str, limit: int = 400) -> str:
    text = raw.strip()
    if not text:
        return "no output"
    lines = [line for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-3:])[-limit:]


def _truncate(text: str, limit: int) -> str:
    """Shorten a message for a single terminal line."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"


class BuildRepositoryProfileStep(Step):
    """Measure the cloned workspace before anything reasons about it.

    The Phase 1 inspect step counts files; this one reads them. It produces a
    :class:`~repopilot.analysis.profile.RepositoryProfile` with per-file language and line
    counts, and hands it to the next step through ``ctx.state``.
    """

    name = "profile"
    title = "Profile repository"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        root = ctx.local_path()
        if not root.is_dir():
            raise StepFailed(f"{root} does not exist; the clone step must run first")

        await ctx.emit(EventType.STEP_STARTED, f"Profiling {ctx.repository.slug}")

        try:
            profile = await asyncio.to_thread(
                build_profile, root, default_branch=ctx.state.get("default_branch")
            )
        except ProfileError as exc:
            raise StepFailed(str(exc)) from exc

        ctx.state["profile"] = profile

        languages = ", ".join(f"{name} {count}" for name, count in profile.languages[:5]) or "none"
        await ctx.emit(
            EventType.STEP_PROGRESS,
            f"Profiled {len(profile.files)} source files ({languages})",
            progress=0.5,
            data={
                "root": str(root),
                "total_files": profile.total_files,
                "source_files": len(profile.files),
                "skipped_files": profile.skipped_files,
                "languages": [name for name, _ in profile.languages[:8]],
                "manifests": list(profile.manifests),
                "test_configs": list(profile.test_configs),
                "test_files": len(profile.test_files),
                "top_level": list(profile.top_level[:30]),
            },
        )

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Profiled {profile.total_files} files, {len(profile.files)} of them source",
            data={
                "source_files": len(profile.files),
                "languages": [name for name, _ in profile.languages[:8]],
                "manifests": list(profile.manifests),
                "test_files": len(profile.test_files),
            },
        )
        return StepOutcome.OK


class SelectRelevantFilesStep(Step):
    """Rank the repository's files against the task and stream the shortlist.

    This is deterministic on purpose: the model only ever sees files that a reproducible
    heuristic already found, and every ``file.selected`` event carries the reasons it was
    chosen, so a run can always explain itself.
    """

    name = "select_files"
    title = "Select relevant files"
    candidate_limit = DEFAULT_CANDIDATE_LIMIT

    async def execute(self, ctx: StepContext) -> StepOutcome:
        profile: RepositoryProfile | None = ctx.state.get("profile")
        if profile is None:
            raise StepFailed("no repository profile; the profile step must run first")

        await ctx.emit(
            EventType.STEP_STARTED, f"Selecting files for '{_truncate(ctx.run.task, 60)}'"
        )

        candidates = await asyncio.to_thread(
            rank_candidates,
            profile,
            ctx.run.task,
            root=ctx.local_path(),
            limit=self.candidate_limit,
        )
        ctx.state["candidates"] = candidates

        if not candidates:
            keywords = ", ".join(tokenize(ctx.run.task)[:8]) or "(none)"
            await ctx.emit(
                EventType.LOG,
                f"No file in this repository matched the task (keywords: {keywords})",
                level=EventLevel.WARN,
                data={"keywords": tokenize(ctx.run.task), "candidates": 0},
            )
            await ctx.emit(
                EventType.STEP_COMPLETED, "No candidate files found", data={"candidates": 0}
            )
            return StepOutcome.OK

        # Only the strongest candidates are worth an event each; the rest go in one summary.
        announced = candidates[:10]
        for candidate in announced:
            await ctx.emit(
                EventType.FILE_SELECTED,
                f"{candidate.path} ({candidate.language or 'unknown'}, {candidate.lines} lines)",
                data={**candidate.as_dict(), "root": str(ctx.local_path())},
            )
        if len(candidates) > len(announced):
            await ctx.emit(
                EventType.STEP_PROGRESS,
                f"+{len(candidates) - len(announced)} more candidate files",
                progress=1.0,
                data={"remaining": len(candidates) - len(announced)},
            )

        await ctx.emit(
            EventType.LOG,
            f"Shortlisted {len(candidates)} of {profile.total_files} files for "
            f"'{_truncate(ctx.run.task, 60)}'",
            data={"candidates": len(candidates), "total_files": profile.total_files},
        )
        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Shortlisted {len(candidates)} files",
            data={
                "candidates": len(candidates),
                "total_files": profile.total_files,
                "paths": [candidate.path for candidate in announced],
            },
        )
        return StepOutcome.OK


class CreateImplementationPlanStep(Step):
    """Ask the configured model for a structured plan, grounded in the profile and shortlist.

    This is the first step that needs a reasoning backend. With no model configured the run
    fails with a message that says exactly which settings are missing, because a run that
    appears to plan while doing nothing is the failure mode worth spending code to prevent.
    """

    name = "plan"
    title = "Create implementation plan"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        client = ctx.services.llm
        if client is None:
            raise StepFailed(
                "no reasoning backend is configured, so no implementation plan can be "
                "generated. Set REPOPILOT_LLM_MODEL and REPOPILOT_LLM_BASE_URL (plus "
                "REPOPILOT_LLM_API_KEY for hosted providers) in .env, or start a local "
                "OpenAI-compatible server such as Ollama."
            )

        profile: RepositoryProfile | None = ctx.state.get("profile")
        if profile is None:
            raise StepFailed("no repository profile; the profile step must run first")

        candidates: list[CandidateFile] = ctx.state.get("candidates") or []
        if not candidates:
            raise StepFailed(
                "no file in this repository looks related to the task, so there is nothing "
                "to plan. Repositories in unfamiliar languages may need the profile step's "
                "file list widened."
            )

        await ctx.emit(
            EventType.STEP_STARTED,
            f"Planning against {len(candidates)} files with {client.model}",
            data={"model": client.model, "candidates": len(candidates)},
        )
        await ctx.emit(
            EventType.STEP_PROGRESS,
            f"Asking {client.model} to plan against {len(candidates)} files",
            progress=0.25,
            data={"model": client.model, "candidates": len(candidates)},
        )

        try:
            plan, dropped = await generate_plan(
                client,
                task=ctx.run.task,
                repository=ctx.run.repository.slug,
                profile=profile,
                candidates=candidates,
                default_branch=ctx.state.get("default_branch"),
                root=ctx.local_path(),
            )
        except PlanRejectedError as exc:
            raise StepFailed(str(exc), retryable=True) from exc
        except LLMNotConfiguredError as exc:
            raise StepFailed(str(exc)) from exc
        except LLMError as exc:
            # Transport, auth, and rate-limit problems are worth one more attempt.
            raise StepFailed(f"reasoning backend failed: {exc}", retryable=True) from exc

        ctx.state["plan"] = plan

        if dropped:
            await ctx.emit(
                EventType.LOG,
                f"Ignored {len(dropped)} path(s) the model invented: {', '.join(dropped[:5])}",
                level=EventLevel.WARN,
                data={"hallucinated_paths": dropped},
            )

        await ctx.emit(
            EventType.PLAN_CREATED,
            plan.summary,
            data={
                "plan": plan.as_dict(),
                "dropped_paths": dropped,
                "model": plan.model,
                "workspace": str(ctx.local_path()),
            },
        )
        for step in plan.steps:
            targets = ", ".join(step.files) or "no files yet"
            await ctx.emit(
                EventType.STEP_PROGRESS,
                f"Plan step {step.order}/{len(plan.steps)}: {step.title} -> {targets}",
                progress=0.4 + 0.6 * (step.order / max(len(plan.steps), 1)),
                data={"order": step.order, "title": step.title, "files": step.files},
            )

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Planned {len(plan.steps)} step(s) across {len(plan.files_to_change)} file(s)",
            data={
                "steps": len(plan.steps),
                "files_to_change": plan.files_to_change,
                "model": plan.model,
                "dropped_paths": dropped,
            },
        )
        return StepOutcome.OK


@dataclass(slots=True)
class _FileAttempt:
    """What happened to one planned file: written, deliberately skipped, or failed."""

    status: Literal["written", "skipped", "failed"]
    reason: str = ""
    payload: dict[str, Any] | None = None


class ApplyImplementationPlanStep(Step):
    """Write the plan into the run's checkout: a branch, then one file at a time.

    Every file the plan marks ``modify`` or ``add`` is generated and written; ``read`` files are
    left alone, because reading them is not the same as being asked to change them. The model is
    asked for one file per call so that a reply too large to finish, or wrong about one file,
    costs that file alone.

    Nothing here touches a file the run was not given. Paths are re-checked against the
    checkout on the way in (:func:`~repopilot.agent.implement.resolve_write_target`), the branch
    is created inside the run's own clone, and a path the plan never proposed cannot be written
    at all. The changes are left uncommitted on the branch: Phase 2C runs the tests against this
    working tree, and Phase 2D opens the pull request.
    """

    name = "apply_plan"
    title = "Apply implementation plan"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        client = ctx.services.llm
        if client is None:
            raise StepFailed(
                "no reasoning backend is configured, so the plan cannot be applied. Set "
                "REPOPILOT_LLM_MODEL and REPOPILOT_LLM_BASE_URL (plus REPOPILOT_LLM_API_KEY for "
                "hosted providers) in .env, or start a local OpenAI-compatible server."
            )

        plan: ImplementationPlan | None = ctx.state.get("plan")
        if plan is None:
            raise StepFailed("no implementation plan; the plan step must run first")

        root = ctx.local_path()
        if not root.is_dir():
            raise StepFailed(f"{root} does not exist; the clone step must run first")

        targets = _writable_selections(plan)
        branch = await self._create_branch(ctx, root)

        await ctx.emit(
            EventType.STEP_STARTED,
            f"Applying {len(targets)} planned change(s) to {branch}"
            if targets
            else f"Nothing to apply; the plan proposed no file changes on {branch}",
            data={"branch": branch, "files": len(targets), "workspace": str(root)},
        )

        if not targets:
            await ctx.emit(
                EventType.LOG,
                "The plan names no file to change, so no file was written",
                level=EventLevel.WARN,
                data={"files_to_change": []},
            )
            await ctx.emit(
                EventType.STEP_COMPLETED,
                "No file changes to apply",
                data={"branch": branch, "written": [], "skipped": [], "failed": []},
            )
            return StepOutcome.OK

        written: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []

        for index, selection in enumerate(targets):
            if ctx.is_cancelled():
                # The engine finalised the run when cancellation was requested, so returning
                # here leaves the run cancelled instead of overwriting that with a failure.
                await ctx.emit(
                    EventType.LOG,
                    f"Stopped after {len(written)} of {len(targets)} file(s): the run was "
                    "cancelled",
                    level=EventLevel.WARN,
                    data={"written": len(written), "total": len(targets)},
                )
                return StepOutcome.OK

            path = selection.path
            await ctx.emit(
                EventType.STEP_PROGRESS,
                f"Writing {path} ({index + 1} of {len(targets)})",
                progress=index / len(targets),
                data={
                    "path": path,
                    "role": selection.role,
                    "index": index + 1,
                    "total": len(targets),
                },
            )

            attempt = await self._write_one(ctx, root, plan, selection, branch, index, targets)
            if attempt.payload is not None:
                written.append(attempt.payload)
            elif attempt.status == "skipped":
                skipped.append({"path": path, "reason": attempt.reason})
            else:
                failed.append({"path": path, "reason": attempt.reason})

        for problem in skipped:
            await ctx.emit(
                EventType.LOG,
                f"Left {problem['path']} alone: {problem['reason']}",
                level=EventLevel.WARN,
                data=problem,
            )
        for problem in failed:
            await ctx.emit(
                EventType.LOG,
                f"Could not write {problem['path']}: {problem['reason']}",
                level=EventLevel.ERROR,
                data=problem,
            )

        if not written:
            raise StepFailed(
                f"no planned file could be written ({len(failed)} rejected, {len(skipped)} "
                "refused); the workspace is unchanged"
            )

        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"Wrote {len(written)} of {len(targets)} planned file(s) on {branch}",
            data={
                "branch": branch,
                "written": written,
                "skipped": skipped,
                "failed": failed,
                "workspace": str(root),
            },
        )
        return StepOutcome.OK

    async def _write_one(
        self,
        ctx: StepContext,
        root: Path,
        plan: ImplementationPlan,
        selection: FileSelection,
        branch: str,
        index: int,
        targets: list[FileSelection],
    ) -> _FileAttempt:
        """Generate and write one planned file, reporting what happened to it.

        A failure on one file is recorded and the run moves on to the next: a plan that touches
        six files should not be thrown away because the fourth one was refused. The step only
        fails outright when nothing at all could be written, so a run never claims to have
        applied a plan it did not.
        """
        client = ctx.services.llm
        assert client is not None  # the caller checked this
        path = selection.path

        try:
            target = resolve_write_target(root, path)
        except UnsafePathError as exc:
            return _FileAttempt(status="skipped", reason=str(exc))

        exists = target.is_file()
        if selection.role == "modify" and not exists:
            return _FileAttempt(
                status="skipped",
                reason="the plan says to modify it, but it is not in the checkout",
            )

        current = read_for_edit(root, path) or ""
        if selection.role == "modify" and not current:
            return _FileAttempt(
                status="skipped",
                reason=(
                    f"it is {target.stat().st_size} bytes, too large to rewrite safely"
                    if exists
                    else "it could not be read from the checkout"
                ),
            )

        siblings = [other.path for other in targets if other.path != path]
        try:
            change = await generate_file_change(
                client,
                task=ctx.run.task,
                repository=ctx.run.repository.slug,
                plan=plan,
                path=path,
                role=selection.role,
                current=current,
                siblings=siblings,
            )
        except ChangeRejectedError as exc:
            return _FileAttempt(status="failed", reason=str(exc))
        except LLMError as exc:
            return _FileAttempt(status="failed", reason=f"reasoning backend failed: {exc}")

        if current and change.content == current:
            return _FileAttempt(
                status="skipped", reason="the model returned it unchanged, so nothing was written"
            )

        try:
            result = await asyncio.to_thread(write_file, root, path, change.content)
        except (OSError, UnsafePathError) as exc:
            return _FileAttempt(status="failed", reason=f"could not write it: {exc}")

        payload = {
            **change.as_dict(),
            "created": result.created,
            "branch": branch,
            "workspace": str(root),
            "index": index + 1,
            "total": len(targets),
        }
        await ctx.emit(
            EventType.FILE_WRITTEN,
            f"{'Created' if result.created else 'Rewrote'} {result.path} "
            f"({result.lines} lines, {result.bytes} bytes)",
            progress=(index + 1) / len(targets),
            data=payload,
        )
        ctx.state.setdefault("written", []).append(payload)
        return _FileAttempt(status="written", payload=payload)

    async def _create_branch(self, ctx: StepContext, root: Path) -> str:
        """Check out this run's branch inside its own clone and report where it started."""
        branch = ctx.run.branch or f"repopilot/{ctx.run.id}"
        base = await _git_head(root)
        exists = await _run_git(
            "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"
        )
        # ``checkout -b`` fails on a branch that is already there, which is what a retried run
        # sees. Checking first keeps the step idempotent instead of turning a retry into a
        # failure about branch names.
        args = ("checkout", branch) if exists.returncode == 0 else ("checkout", "-b", branch)
        result = await _run_git("-C", str(root), *args)
        if result.returncode != 0:
            raise StepFailed(
                f"git {' '.join(args)} failed in {root}: {_tail(result.stderr)}", retryable=True
            )

        ctx.state["branch"] = branch
        await ctx.emit(
            EventType.BRANCH_CREATED,
            f"Created branch {branch} at {(base or '?')[:8]}",
            data={
                "branch": branch,
                "base": ctx.state.get("default_branch"),
                "head": base,
                "workspace": str(root),
            },
        )
        return branch


def _writable_selections(plan: ImplementationPlan) -> list[FileSelection]:
    """The plan's files to write, in plan order and without repeats.

    ``read`` is excluded because the plan marked those as context, not as work. The first role
    wins for a repeated path, which keeps a plan that marks the same file ``add`` and
    ``modify`` from writing it twice.
    """
    ordered: list[FileSelection] = []
    seen: set[str] = set()
    for selection in plan.relevant_files:
        if selection.role not in ("modify", "add") or selection.path in seen:
            continue
        seen.add(selection.path)
        ordered.append(selection)
    return ordered


# ------------------------------------------------------------------ verification (2C)

#: How many times one attempt will ask the model to fix what its own change broke. A repair round
#: costs a model call and a full suite run, so this is deliberately small: the point is to catch
#: the ordinary mistake (a signature the caller still uses, an import that moved), not to let a
#: model iterate against a suite until it goes green.
MAX_REPAIR_ROUNDS = 2


class RunTestSuiteStep(Step):
    """Run the repository's own tests against the change, in the run's own checkout.

    The command comes from the Phase 2A profile rather than from a guess: a repository that
    declares no runner is reported as having no suite and the run carries on, because inventing
    ``pytest`` for a Go repository produces a failure that looks like a broken change. The suite
    runs in ``ctx.local_path()``, the isolated clone this run owns, and never anywhere else.

    This step never decides to fix anything. It reports what happened, and a failure is passed to
    :class:`DiagnoseFailuresStep` rather than repaired here, so the two questions stay separate:
    did the change work, and if not, why.
    """

    name = "test"
    title = "Run the test suite"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        profile: RepositoryProfile | None = ctx.state.get("profile")
        if profile is None:
            raise StepFailed("no repository profile; the profile step must run first")

        root = ctx.local_path()
        if not root.is_dir():
            raise StepFailed(f"{root} does not exist; the clone step must run first")

        await ctx.emit(
            EventType.STEP_STARTED,
            "Looking for a test suite to run",
            data={"workspace": str(root)},
        )

        suite = detect_test_suite(profile)
        if suite is None:
            ctx.state["test_result"] = None
            await ctx.emit(
                EventType.TEST_SKIPPED,
                f"No test suite detected in {ctx.run.repository.slug}, so nothing was verified",
                data={"reason": "the repository declares no test runner", **profile_facts(profile)},
            )
            await ctx.emit(
                EventType.STEP_COMPLETED,
                "No test suite to run",
                data={"verified": False},
            )
            return StepOutcome.OK

        await ctx.emit(
            EventType.TEST_SUITE_DETECTED,
            f"Running the {suite.framework} suite: {suite.as_command_text()}",
            data=suite.as_dict(),
        )

        unavailable = await self._probe(ctx, suite, root)
        if unavailable is not None:
            await ctx.emit(
                EventType.TEST_SKIPPED,
                f"Could not run the {suite.framework} suite: {unavailable}",
                level=EventLevel.WARN,
                # The suite's own facts first: "reason" below is about the environment, which is
                # a different reason from the evidence that chose the runner.
                data={**suite.as_dict(), "reason": unavailable},
            )
            await ctx.emit(
                EventType.STEP_COMPLETED,
                f"The {suite.framework} suite is not available in this environment",
                data={"verified": False},
            )
            # Not a failure of the change and not something another attempt fixes, so the run
            # says what is missing and carries on. A green run that was never verified is
            # reported as unverified rather than as a pass.
            return StepOutcome.OK

        outcome = await run_suite(ctx, suite, root)
        ctx.state["test_result"] = outcome
        ctx.state["test_suite"] = suite

        if outcome.status == "skipped":
            # Reported as a skip rather than a pass: the suite declared a runner and it collected
            # nothing, so nothing here says the change works.
            return await self._closed(
                ctx, outcome, "Skipped the suite: nothing was collected", False
            )
        if not outcome.suite_ran:
            return await self._closed(ctx, outcome, _verdict_message(suite, outcome), False)

        await ctx.emit(
            EventType.TEST_COMPLETED,
            _verdict_message(suite, outcome),
            level=EventLevel.ERROR if outcome.status == "failed" else EventLevel.INFO,
            data=outcome.as_dict(),
        )

        if outcome.status == "failed":
            # Not a StepFailed: the diagnoser is next in the pipeline and a failure here is the
            # normal case it exists for. Raising would end the run before it can be explained.
            await ctx.emit(
                EventType.LOG,
                f"{_failure_count(outcome)} of the repository's tests did not pass; the failure "
                "will be diagnosed next",
                level=EventLevel.WARN,
                data={"framework": suite.framework, "exit_code": outcome.exit_code},
            )

        return await self._closed(
            ctx, outcome, _verdict_message(suite, outcome), outcome.status == "passed"
        )

    async def _closed(
        self, ctx: StepContext, outcome: TestOutcome, message: str, verified: bool
    ) -> StepOutcome:
        await ctx.emit(
            EventType.STEP_COMPLETED,
            message,
            data={"verified": verified, "status": outcome.status},
        )
        return StepOutcome.OK

    async def _probe(self, ctx: StepContext, suite: TestSuite, root: Path) -> str | None:
        """Return why the runner is unusable here, or ``None`` when it is fine.

        Only the python suites need this: their command is the interpreter, which always exists,
        so a missing pytest would otherwise arrive as a failing suite and send the diagnoser to
        repair code that was never broken.
        """
        if not suite.probe:
            return None
        result = await ctx.services.sandbox.run(
            suite.probe, cwd=root, is_cancelled=ctx.is_cancelled
        )
        if result.cancelled:
            return "the run was cancelled before the suite started"
        return None if result.ok else suite.probe_reason


async def run_suite(ctx: StepContext, suite: TestSuite, root: Path) -> TestOutcome:
    """Run one suite once and turn what it did into a reportable outcome.

    A module-level function rather than a method, because the diagnosis step re-runs the same
    suite after a repair and must get an identical reading of the same runner: a repair loop that
    interpreted the output differently the second time would compare two different things.
    """
    await ctx.emit(
        EventType.TEST_STARTED,
        f"Running {suite.as_command_text()} in {root.name}",
        data=suite.as_dict(),
    )
    result = await ctx.services.sandbox.run(suite.command, cwd=root, is_cancelled=ctx.is_cancelled)

    if result.cancelled:
        return _outcome_from(suite, result, status="cancelled")
    if result.timed_out:
        # Its own status, not a generic "unavailable": a suite that hangs is a fact about the
        # change worth reporting, and a repair round would only hang again in the same place.
        return _outcome_from(suite, result, status="timeout")
    if not result.ran:
        return _outcome_from(suite, result, status="unavailable")
    if collected_nothing(suite.framework, result.exit_code, result.output):
        return _outcome_from(
            suite,
            result,
            status="skipped",
            reason=f"the {suite.framework} runner collected no tests",
        )

    tally = parse_test_output(suite.framework, result.output, exit_code=result.exit_code)
    return TestOutcome(
        framework=suite.framework,
        command=suite.command,
        status="passed" if result.ok and not tally.failures else "failed",
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        tally=tally,
        output=tail_output(result.output),
    )


class DiagnoseFailuresStep(Step):
    """Explain a failing suite, repair what this run broke, and re-run to prove it.

    The loop is deliberately two-sided, and both sides are bounded:

    * **Within the attempt**, the run asks the model what is wrong, writes the fix through the
      same path-safe writer Phase 2B used, and runs the suite again, at most
      :data:`MAX_REPAIR_ROUNDS` times. This is where a failure usually gets fixed, because the
      repair is applied to the tree that produced the failure.
    * **Across attempts**, a suite that is still failing hands back to the engine as
      :attr:`StepOutcome.RETRY`, which re-runs the whole pipeline with an incremented
      ``Run.attempts`` and a fresh plan. Whether there is room for that is the engine's call, and
      this step asks :attr:`StepContext.can_retry` rather than assuming a number.

    Two things stop the loop before it starts. A finding marked pre-existing is reported and not
    repaired, because a repository that was already broken is not this run's to fix. And a
    finding naming a file this run did not write is refused with the reason shown, so a diagnosis
    can talk about a test that ought to change without the run reaching outside its own diff.
    """

    name = "diagnose"
    title = "Diagnose test failures"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        outcome: TestOutcome | None = ctx.state.get("test_result")
        if outcome is None or not outcome.tally.failures:
            await ctx.emit(
                EventType.STEP_STARTED,
                "No test failure to diagnose",
                data={"verified": outcome is not None},
            )
            await ctx.emit(
                EventType.STEP_COMPLETED,
                "Nothing to diagnose; the suite did not report a failure",
                data={"rounds": 0, "repaired": []},
            )
            return StepOutcome.OK

        client = ctx.services.llm
        if client is None:
            raise StepFailed(
                f"the {outcome.framework} suite fails and no reasoning backend is configured to "
                "explain why. Set REPOPILOT_LLM_MODEL and REPOPILOT_LLM_BASE_URL in .env to let "
                "the run attempt a repair, or read the failure output above and fix it yourself."
            )

        root = ctx.local_path()
        allowed = self._repairable(ctx)
        await ctx.emit(
            EventType.STEP_STARTED,
            f"Diagnosing {_failure_count(outcome)} failing test(s) from the "
            f"{outcome.framework} suite",
            data={
                "framework": outcome.framework,
                "failures": len(outcome.tally.failures),
                "repairable": [path for path, _ in allowed],
            },
        )
        if not allowed:
            await ctx.emit(
                EventType.LOG,
                "This run changed no file the diagnosis could repair, so the failures are "
                "reported as they are",
                level=EventLevel.WARN,
                data={"failures": len(outcome.tally.failures)},
            )
            return await self._give_up(ctx, outcome, reason="the run changed nothing to repair")

        summary = ""
        for round_number in range(1, MAX_REPAIR_ROUNDS + 1):
            if ctx.is_cancelled():
                await ctx.emit(
                    EventType.LOG,
                    "Stopped diagnosing: the run was cancelled",
                    level=EventLevel.WARN,
                    data={"round": round_number},
                )
                return StepOutcome.OK

            report = await self._diagnose(ctx, client, outcome, allowed, round_number, summary)
            summary = report.summary
            repaired = await self._apply(ctx, root, report, allowed, round_number)
            if not repaired:
                if not report.actionable():
                    return await self._accepted_as_pre_existing(ctx, outcome, report)
                return await self._give_up(
                    ctx,
                    outcome,
                    reason="no finding named a fix this run may make",
                    rounds=round_number - 1,
                )

            outcome = await run_suite(ctx, ctx.state["test_suite"], root)
            ctx.state["test_result"] = outcome
            if not outcome.suite_ran:
                # The suite stopped, hung, or was cancelled. That is not a verdict on the repair,
                # so it is not asked for twice: a timeout that survived one round would only
                # survive the next.
                return await self._closed(
                    ctx, outcome, _verdict_message(ctx.state["test_suite"], outcome), False
                )
            await ctx.emit(
                EventType.TEST_COMPLETED,
                _verdict_message(ctx.state["test_suite"], outcome),
                level=EventLevel.ERROR if outcome.status == "failed" else EventLevel.INFO,
                data=outcome.as_dict(),
            )
            if outcome.status == "passed":
                await ctx.emit(
                    EventType.STEP_COMPLETED,
                    f"The suite passes after {round_number} repair round(s)",
                    data={"rounds": round_number, "repaired": repaired, "verified": True},
                )
                return StepOutcome.OK

        return await self._give_up(ctx, outcome, reason=summary, rounds=MAX_REPAIR_ROUNDS)

    def _repairable(self, ctx: StepContext) -> list[tuple[str, str | None]]:
        """The files this run wrote, with their current contents for the prompt.

        This list is the boundary for the whole step. A path that is not in it cannot be written,
        whatever the model says, and a file too large to quote is passed as ``None`` so the model
        knows it exists without being handed a truncated version to copy.
        """
        repaired: list[tuple[str, str | None]] = []
        for entry in ctx.state.get("written", []):
            path = entry.get("path")
            if not isinstance(path, str) or path in {known for known, _ in repaired}:
                continue
            root = ctx.local_path()
            try:
                current = read_for_edit(root, path, max_chars=DIAGNOSIS_SNIPPET_CHARS)
            except OSError:  # pragma: no cover - a file that vanished mid-run
                current = None
            repaired.append((path, current))
        return repaired

    async def _diagnose(
        self,
        ctx: StepContext,
        client: ChatClient,
        outcome: TestOutcome,
        files: list[tuple[str, str | None]],
        round_number: int,
        previous: str,
    ) -> TestFailureReport:
        try:
            return await generate_diagnosis(
                client,
                task=ctx.run.task,
                repository=ctx.run.repository.slug,
                outcome=outcome,
                files=files,
                round_number=round_number,
                previous=previous,
            )
        except LLMError as exc:
            # A transport failure is worth one more attempt, and the engine already knows how to
            # count them. A reply that will not parse at all is not: asking again is how a run
            # burns three attempts on a model that is answering the wrong question.
            raise StepFailed(
                f"the reasoning backend failed while diagnosing the test failure: {exc}",
                retryable=True,
            ) from exc
        except DiagnosisRejectedError as exc:
            raise StepFailed(
                f"the model could not explain the test failure: {exc}. The failure output is in "
                "the test.completed event above."
            ) from exc

    async def _apply(
        self,
        ctx: StepContext,
        root: Path,
        report: TestFailureReport,
        allowed: list[tuple[str, str | None]],
        round_number: int,
    ) -> list[str]:
        """Write the repairs the model proposed. Returns the paths actually rewritten."""
        permitted = {path for path, _ in allowed}
        current = {path: body for path, body in allowed}
        repaired: list[str] = []

        await ctx.emit(
            EventType.FAILURE_DIAGNOSED,
            report.summary or f"{len(report.findings)} failing test(s) diagnosed",
            data={**report.as_dict(), "round": round_number},
        )

        for finding in report.findings:
            if finding.pre_existing:
                await ctx.emit(
                    EventType.LOG,
                    f"{finding.test} looks like it was already failing: {finding.cause}",
                    level=EventLevel.WARN,
                    data=finding.as_dict(),
                )
                continue
            if finding.path not in permitted:
                await ctx.emit(
                    EventType.LOG,
                    f"The fix for {finding.test} belongs in {finding.path}, which this run did "
                    f"not change, so it was left alone: {finding.cause}",
                    level=EventLevel.WARN,
                    data=finding.as_dict(),
                )
                continue
            if not finding.content.strip():
                await ctx.emit(
                    EventType.LOG,
                    f"No corrected contents came back for {finding.path}, so it was left alone",
                    level=EventLevel.WARN,
                    data=finding.as_dict(),
                )
                continue

            try:
                content = validate_whole_file(
                    finding.content, current=current.get(finding.path) or "", path=finding.path
                )
            except ChangeRejectedError as exc:
                await ctx.emit(
                    EventType.LOG,
                    f"Refused the proposed fix for {finding.path}: {exc}",
                    level=EventLevel.ERROR,
                    data=finding.as_dict(),
                )
                continue

            try:
                written = await asyncio.to_thread(write_file, root, finding.path, content)
            except (OSError, UnsafePathError) as exc:  # pragma: no cover - disk-level failure
                await ctx.emit(
                    EventType.LOG,
                    f"Could not write the repair to {finding.path}: {exc}",
                    level=EventLevel.ERROR,
                    data=finding.as_dict(),
                )
                continue

            repaired.append(written.path)
            await ctx.emit(
                EventType.FILE_WRITTEN,
                f"Repaired {written.path} for {finding.test} ({written.lines} lines)",
                data={
                    "path": written.path,
                    "role": "modify",
                    "bytes": written.bytes,
                    "lines": written.lines,
                    "created": False,
                    "branch": ctx.state.get("branch"),
                    "repair": True,
                    "round": round_number,
                },
            )
        return repaired

    async def _accepted_as_pre_existing(
        self, ctx: StepContext, outcome: TestOutcome, report: TestFailureReport
    ) -> StepOutcome:
        """Every failure was judged pre-existing: report it and let the change stand."""
        await ctx.emit(
            EventType.STEP_COMPLETED,
            f"The {outcome.framework} suite still fails, but every failure looks like it was "
            "already there before this change",
            data={"rounds": 1, "repaired": [], "pre_existing": len(report.findings)},
        )
        return StepOutcome.OK

    async def _give_up(
        self, ctx: StepContext, outcome: TestOutcome, *, reason: str, rounds: int = 0
    ) -> StepOutcome:
        """Report a failure the run cannot fix, then either ask to retry or end the run.

        Asking is the engine's decision to make, so this consults the bound it publishes. On the
        last attempt there is nothing to ask for, and the run ends with the failure spelled out
        rather than with a message about having run out of attempts.
        """
        detail = reason or (outcome.tally.failures[0].test if outcome.tally.failures else "unknown")
        await ctx.emit(
            EventType.LOG,
            f"Giving up on the {outcome.framework} suite after {rounds} repair round(s): {detail}",
            level=EventLevel.ERROR,
            data={"framework": outcome.framework, "rounds": rounds},
        )
        if ctx.can_retry:
            return StepOutcome.RETRY
        raise StepFailed(
            f"the {outcome.framework} suite still fails after {rounds} repair round(s) and "
            f"{ctx.attempt} attempt(s): {detail}"
        )


def _outcome_from(
    suite: TestSuite, result: CommandResult, *, status: str, reason: str = ""
) -> TestOutcome:
    """An outcome for a run that never really ran, carrying whatever the runner did say."""
    return TestOutcome(
        framework=suite.framework,
        command=suite.command,
        status=status,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        output=tail_output(result.output),
        reason=reason or result.error,
    )


def _failure_count(outcome: TestOutcome) -> int:
    return len(outcome.tally.failures) or (outcome.tally.failed or 0)


def _verdict_message(suite: TestSuite, outcome: TestOutcome) -> str:
    """One line, the way a person reading CI would say it."""
    if outcome.status == "passed":
        passed = outcome.tally.passed
        count = f"{passed} passed" if passed is not None else "the suite passed"
        return f"{count} in {outcome.duration_seconds:.1f}s"
    if outcome.status == "skipped":
        return f"Skipped the {suite.framework} suite: {outcome.reason or 'nothing to run'}"
    if outcome.status == "unavailable":
        return f"Could not run the {suite.framework} suite: {outcome.reason or 'unavailable'}"
    if outcome.status == "cancelled":
        return f"Stopped the {suite.framework} suite: the run was cancelled"
    if outcome.status == "timeout":
        return f"The {suite.framework} suite was stopped: {outcome.reason or 'it did not finish'}"
    tally = outcome.tally
    parts = [
        f"{tally.failed} failed" if tally.failed is not None else None,
        f"{tally.passed} passed" if tally.passed is not None else None,
        f"{tally.errors} errors" if tally.errors else None,
        f"{tally.skipped} skipped" if tally.skipped else None,
    ]
    detail = ", ".join(part for part in parts if part) or f"exit status {outcome.exit_code}"
    return f"{_failure_count(outcome)} of the {suite.framework} tests did not pass ({detail})"


def profile_facts(profile: RepositoryProfile) -> dict[str, Any]:
    """The measured facts worth quoting when no suite could be found."""
    return {
        "languages": [language for language, _ in profile.languages[:3]],
        "test_files": len(profile.test_files),
        "manifests": list(profile.manifests),
        "test_configs": list(profile.test_configs),
    }


def default_pipeline() -> tuple[Step, ...]:
    """The default pipeline: Phase 1's four steps, then 2A's, 2B's, and 2C's.

    Every phase is additive, so the four Phase 1 steps still run first and still mean what they
    meant. Phase 2C adds the test run and the diagnosis that can act on it; the pull request
    (2D) is appended here when it exists.
    """
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        CloneRepositoryStep(),
        InspectRepositoryStep(),
        BuildRepositoryProfileStep(),
        SelectRelevantFilesStep(),
        CreateImplementationPlanStep(),
        ApplyImplementationPlanStep(),
        RunTestSuiteStep(),
        DiagnoseFailuresStep(),
    )
