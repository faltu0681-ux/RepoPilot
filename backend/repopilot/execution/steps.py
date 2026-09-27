"""The step pipeline: the seam where the agent loop plugs in.

A run is a list of :class:`Step` objects executed in order. The engine knows nothing about what
any step does; it only knows how to start one, watch it emit, and decide what to do when it
raises.

The Phase 1 pipeline does real work: validate the request, ask GitHub about the repository,
clone it, and inspect what was cloned. The agent steps arrive later and are appended to the
same list:

    validate -> resolve -> clone -> inspect -> plan -> modify -> test -> diagnose -> pull request

Everything those later steps need is already in place:

* :class:`StepContext` carries the run, settings, the store, the workspace, a run-scoped
  ``state`` dict for passing results forward, and a bound ``emit`` coroutine.
* :class:`StepOutcome` separates ``ok`` from ``retry`` from ``abort``, and ``Run.attempts``
  counts retries, so the fix loop becomes engine policy instead of a rewrite.
* :class:`RunServices` is the container for the LLM client, the shell sandbox, and the diff
  writer when they arrive. Steps take collaborators from here, never by importing them, which
  is what keeps them substitutable in tests.
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
from typing import Any, ClassVar

from repopilot.agent.llm import ChatClient, LLMError, LLMNotConfiguredError
from repopilot.agent.plan import PlanRejectedError, generate_plan
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
from repopilot.core.errors import RepopilotError
from repopilot.domain.models import EventLevel, EventType, Run
from repopilot.domain.repository import RepositoryRef
from repopilot.domain.store import RunStore
from repopilot.github.client import GitHubClient, RepositoryInfo

__all__ = [
    "IGNORED_DIRECTORIES",
    "IGNORED_SUFFIXES",
    "BuildRepositoryProfileStep",
    "CloneRepositoryStep",
    "CreateImplementationPlanStep",
    "GitResult",
    "InspectRepositoryStep",
    "ResolveRepositoryStep",
    "RunServices",
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
    inventing output. The shell sandbox and the diff writer arrive in later phases.
    """

    github: GitHubClient
    workspace: Path
    llm: ChatClient | None = None
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
    #: Scratch space for passing results from one step to the next.
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def repository(self) -> RepositoryRef:
        return self.run.repository

    @property
    def attempt_label(self) -> str:
        return "" if self.attempt == 0 else f" (attempt {self.attempt + 1})"

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


def default_pipeline() -> tuple[Step, ...]:
    """The default pipeline: Phase 1's four steps followed by Phase 2A discovery and planning.

    Phase 2A is additive, so the four Phase 1 steps still run first and still mean what they
    meant. Steps that write code (2B) and run tests (2C) are appended here when they exist.
    """
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        CloneRepositoryStep(),
        InspectRepositoryStep(),
        BuildRepositoryProfileStep(),
        SelectRelevantFilesStep(),
        CreateImplementationPlanStep(),
    )
