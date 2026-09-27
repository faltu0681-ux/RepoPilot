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
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from repopilot.core.errors import RepopilotError
from repopilot.domain.models import EventLevel, EventType, Run
from repopilot.domain.repository import RepositoryRef
from repopilot.domain.store import RunStore
from repopilot.github.client import GitHubClient, RepositoryInfo

#: Directories never worth descending into when inspecting a repository.
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        "vendor",
        ".next",
        ".tox",
        "htmlcov",
        ".idea",
        ".gradle",
    }
)

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

    Phase 1 ships only ``github``. The LLM client, the shell sandbox, and the diff writer are
    added to this container as they are implemented.
    """

    github: GitHubClient
    workspace: Path
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

    def local_path(self) -> Path:
        """Where this repository is cloned inside the workspace."""
        return self.services.workspace / self.run.repository.slug.replace("/", "__")


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
            await asyncio.to_thread(shutil.rmtree, target, True)

        await ctx.emit(
            EventType.STEP_STARTED,
            f"Cloning {info.clone_url} (branch {info.default_branch})",
            data={"target": str(target), "expected_bytes": info.size_kb * 1024},
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        process = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            info.default_branch,
            info.clone_url,
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        _stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise StepFailed(
                f"git clone exited {process.returncode}: {_tail(stderr)}", retryable=True
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


async def _git_head(path: Path) -> str | None:
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
    if process.returncode != 0:
        return None
    return stdout.decode().strip() or None


def _tail(raw: bytes, limit: int = 400) -> str:
    text = raw.decode(errors="replace").strip()
    if not text:
        return "no output"
    lines = [line for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-3:])[-limit:]


def default_pipeline() -> tuple[Step, ...]:
    """The Phase 1 pipeline. Agent steps are appended here as they are implemented."""
    return (
        ValidateRequestStep(),
        ResolveRepositoryStep(),
        CloneRepositoryStep(),
        InspectRepositoryStep(),
    )
