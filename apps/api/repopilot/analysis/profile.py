"""Deterministic inspection of a cloned repository.

This is the "look at the code" half of Phase 2A and it never calls a model. Every fact it
reports is measured from the files on disk, which is what makes it useful as grounding: the
plan in :mod:`repopilot.agent.plan` is only allowed to talk about what is in here.

Keeping analysis separate from reasoning also means the expensive, cacheable half of a run
is free, testable, and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Directories that are never worth reading: VCS metadata, build output, vendored trees,
#: and the run's own artefacts.
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        "out",
        "target",
        "vendor",
        "third_party",
        "coverage",
        "htmlcov",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".gradle",
        ".terraform",
        "bower_components",
        ".repopilot",
    }
)

#: Extension -> language, for the languages a public repo is likely to be written in.
LANGUAGES_BY_SUFFIX: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".c": "C",
    ".h": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".hpp": "C++",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".sql": "SQL",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".lua": "Lua",
    ".dart": "Dart",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".r": "R",
    ".jl": "Julia",
    ".pl": "Perl",
    ".hs": "Haskell",
}

#: Files that usually describe how to build, test, and package a project.
MANIFEST_NAMES = frozenset(
    {
        "package.json",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "pipfile",
        "poetry.lock",
        "uv.lock",
        "pdm.lock",
        "environment.yml",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "gemfile",
        "composer.json",
        "makefile",
        "dockerfile",
        "docker-compose.yml",
    }
)

#: Files that usually describe how the project is tested or linted.
TEST_CONFIG_NAMES = frozenset(
    {
        "pytest.ini",
        "tox.ini",
        "noxfile.py",
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.js",
        "vitest.config.ts",
        "karma.conf.js",
        "playwright.config.ts",
        "cypress.config.js",
        "vitest.workspace.ts",
        "phpunit.xml",
        "rspec",
        ".rspec",
    }
)

DOC_NAMES = frozenset({"readme.md", "readme.rst", "readme.txt", "readme", "contributing.md"})

#: Paths that conventionally hold tests, checked against the path components.
TEST_PATH_MARKERS = ("test", "tests", "spec", "specs", "__tests__")

#: Per-file ceiling when reading content for analysis. Real source files are far smaller, and
#: the cap keeps one pathological minified blob from eating the model's context.
MAX_FILE_BYTES = 128_000

#: Hard ceiling on how many files a single profile will report.
MAX_FILES_SCANNED = 4_000


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One readable file in the repository."""

    path: str  # repo-relative, always forward-slashed
    language: str | None
    size_bytes: int
    lines: int

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def directory(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    @property
    def is_test(self) -> bool:
        return looks_like_test(self.path)

    @property
    def extension(self) -> str:
        return "." + self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """Everything measured about a repository before any reasoning happens."""

    root: Path
    files: tuple[SourceFile, ...]
    languages: tuple[tuple[str, int], ...]
    manifests: tuple[str, ...]
    test_configs: tuple[str, ...]
    docs: tuple[str, ...]
    top_level: tuple[str, ...]
    total_files: int
    skipped_files: int
    default_branch: str | None = None

    @property
    def primary_language(self) -> str | None:
        return self.languages[0][0] if self.languages else None

    @property
    def test_files(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files if f.is_test)

    def by_path(self, path: str) -> SourceFile | None:
        for source in self.files:
            if source.path == path:
                return source
        return None

    def extension_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source in self.files:
            if source.extension:
                counts[source.extension] = counts.get(source.extension, 0) + 1
        return counts


class ProfileError(Exception):
    """The workspace could not be inspected."""


def looks_like_test(path: str) -> bool:
    """Heuristic test detection, used only to prioritise files, never to decide anything."""
    lowered = path.lower()
    parts = lowered.split("/")
    if any(part in TEST_PATH_MARKERS for part in parts[:-1]):
        return True
    name = parts[-1]
    if name.startswith(("test_", "spec_", "test.", "conftest")):
        return True
    if name.endswith(("_test.py", "_test.go", "_test.rs", "_spec.rb")):
        return True
    # "core.test.ts" and "core.spec.js": an infix segment, not a prefix or a suffix.
    segments = name.split(".")
    return any(segment in ("test", "spec") for segment in segments[1:-1])


def is_ignored(path: Path, root: Path) -> bool:
    """True when any path component is a directory we never descend into."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in IGNORED_DIRECTORIES for part in parts[:-1])


def iter_source_files(
    root: Path, *, limit: int = MAX_FILES_SCANNED
) -> tuple[list[SourceFile], int]:
    """Walk ``root`` and measure every candidate file. Returns the files and the skip count."""
    files: list[SourceFile] = []
    skipped = 0

    for path in sorted(root.rglob("*")):
        if len(files) >= limit:
            break
        if is_ignored(path, root):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in MANIFEST_NAMES or path.name in TEST_CONFIG_NAMES:
            continue  # recorded separately; not source to edit
        if path.suffix.lower() not in LANGUAGES_BY_SUFFIX:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            skipped += 1
            continue
        if size > MAX_FILE_BYTES:
            skipped += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped += 1
            continue

        relative = path.relative_to(root).as_posix()
        files.append(
            SourceFile(
                path=relative,
                language=LANGUAGES_BY_SUFFIX.get(path.suffix.lower()),
                size_bytes=size,
                lines=count_lines(text),
            )
        )

    return files, skipped


def count_lines(text: str) -> int:
    """Lines in a file, counting a trailing newline as a terminator rather than a line."""
    if not text:
        return 0
    newlines = text.count("\n")
    return newlines if text.endswith("\n") else newlines + 1


def existing_names(root: Path, candidates: frozenset[str]) -> tuple[str, ...]:
    """Top-level files whose name matches ``candidates`` case-insensitively.

    The lookup is case-insensitive because the tables are lowercase, but the *recorded* name
    must be the real one on disk: a run must never end up holding a path such as
    ``cargo.toml`` when the repository actually contains ``Cargo.toml``.
    """
    found = [
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and entry.name.lower() in candidates
    ]
    return tuple(sorted(found))


def build_profile(root: Path, *, default_branch: str | None = None) -> RepositoryProfile:
    """Inspect a checkout and measure it. Raises ``ProfileError`` if there is nothing there."""
    if not root.is_dir():
        raise ProfileError(f"workspace does not exist: {root}")

    files, skipped = iter_source_files(root)
    top_level = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    manifests = existing_names(root, MANIFEST_NAMES)
    test_configs = existing_names(root, TEST_CONFIG_NAMES)

    if not files and not manifests and not test_configs:
        raise ProfileError(f"no source files or manifests found under {root}")

    languages: dict[str, int] = {}
    for source in files:
        if source.language:
            languages[source.language] = languages.get(source.language, 0) + 1

    return RepositoryProfile(
        root=root,
        files=tuple(files),
        languages=tuple(sorted(languages.items(), key=lambda kv: (-kv[1], kv[0]))),
        manifests=manifests,
        test_configs=test_configs,
        docs=tuple(
            sorted(p.name for p in root.iterdir() if p.is_file() and p.name.lower() in DOC_NAMES)
        ),
        top_level=tuple(top_level),
        total_files=len(files) + skipped,
        skipped_files=skipped,
        default_branch=default_branch,
    )


def read_source(root: Path, relative: str, *, max_bytes: int = MAX_FILE_BYTES) -> str | None:
    """Read a repo-relative file, refusing anything outside the workspace.

    The containment check is what keeps analysis (and later, editing) from touching the user's
    own repository or anything else on the machine.
    """
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        return None
    if not candidate.is_file():
        return None
    try:
        if candidate.stat().st_size > max_bytes:
            return None
        return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
