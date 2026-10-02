"""Working out how to run a repository's tests, and reading what came back.

This is the deterministic half of Phase 2C and it never calls a model. Two jobs, kept apart from
the step that actually runs anything:

* **Detection** turns the profile Phase 2A already built into one command. It reads only the
  manifests and test configs the profile recorded, so the tree is never walked twice and the
  framework is never one the repository does not actually declare. A repository that declares no
  runner gets no command: the step then says so instead of guessing, because a guessed test
  command on a repository with no tests fails in ways that look like real failures.
* **Reading the output** turns whatever the runner printed into counts and a list of failures.
  Every runner formats its results differently and any of them can be told to print nothing, so
  an unparsed count is ``None`` rather than zero and the exit code stays the authority on whether
  the suite passed.

Nothing in here executes code, and nothing here touches a file: the command is data until
:class:`repopilot.execution.steps.RunTestSuiteStep` hands it to the sandbox, and only ever
pointed at the run's own isolated checkout.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from repopilot.analysis.profile import RepositoryProfile, read_source

#: How many failures are reported and handed to the diagnoser. A suite with hundreds of failures
#: is one broken assumption, not hundreds of separate problems, and every failure is kept in the
#: event stream and replayed to every connected client.
MAX_REPORTED_FAILURES = 20

#: How much of a failure's assertion detail is kept, per failure.
MAX_EXCERPT_CHARS = 600

#: How much of the tail of the run's output is kept, for the event stream and the diagnoser.
MAX_OUTPUT_CHARS = 8_000

#: The clearest evidence is at the end of the log: runners put their summary last.
MAX_OUTPUT_LINES = 400

#: Manifest contents are read to look for a test script or a runner section. They are small.
MAX_MANIFEST_CHARS = 200_000


@dataclass(frozen=True, slots=True)
class TestSuite:
    """One command that runs a repository's tests, and why it was chosen."""

    framework: str
    command: tuple[str, ...]
    language: str | None
    reason: str
    test_files: int = 0
    #: A cheap command that proves the runner exists before the real one is blamed for failing.
    probe: tuple[str, ...] = ()
    probe_reason: str = ""

    def as_command_text(self) -> str:
        """The command as a person would type it, for display only.

        Never parsed back: the run always executes ``command`` as an argument vector.
        """
        return " ".join(
            part if re.fullmatch(r"[\w./:=+-]+", part) else f'"{part}"' for part in self.command
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "command": list(self.command),
            "language": self.language,
            "reason": self.reason,
            "test_files": self.test_files,
        }


@dataclass(frozen=True, slots=True)
class TestFailure:
    """One test that did not pass, as far as the runner's own output said."""

    test: str
    message: str = ""
    excerpt: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"test": self.test, "message": self.message, "excerpt": self.excerpt}


@dataclass(frozen=True, slots=True)
class TestTally:
    """What a runner's output said. Every count is optional: not every runner prints one."""

    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    failures: tuple[TestFailure, ...] = ()

    @property
    def total(self) -> int | None:
        counted = [
            c for c in (self.passed, self.failed, self.errors, self.skipped) if c is not None
        ]
        return sum(counted) if len(counted) == 4 else None

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "failures": [failure.as_dict() for failure in self.failures],
        }


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """One execution of a test suite, ready to be reported.

    ``status`` is the verdict the step acts on, and only three of the six values are ever a
    failure of the *code*: ``failed`` and ``errored`` say the tests did not pass, while
    ``unavailable``, ``timeout`` and ``cancelled`` say the suite never really ran, and retrying a
    plan will not change that.
    """

    framework: str
    command: tuple[str, ...]
    status: str
    exit_code: int | None = None
    duration_seconds: float = 0.0
    tally: TestTally = field(default_factory=TestTally)
    output: str = ""
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def suite_ran(self) -> bool:
        """False when the run tells us nothing about the code, only about the environment."""
        return self.status not in ("unavailable", "timeout", "cancelled")

    @property
    def broken(self) -> bool:
        """True when the tests did not pass, whatever the reason the run gives."""
        return self.status in ("failed", "errored")

    def as_command_text(self) -> str:
        """The command as a person would type it, for display and for the diagnosis prompt.

        Never parsed back: the sandbox only ever runs the argument vector.
        """
        return " ".join(
            part if re.fullmatch(r"[\w./:=+-]+", part) else f'"{part}"' for part in self.command
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "command": list(self.command),
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "reason": self.reason,
            "output": self.output,
            **self.tally.as_dict(),
        }


# --------------------------------------------------------------------------- detection


def detect_test_suite(profile: RepositoryProfile) -> TestSuite | None:
    """The one command that should run this repository's tests, or ``None`` if it declares none.

    Where a repository declares more than one runner, the ecosystem holding the most test files
    wins, because that is the one the change under test actually belongs to. Ties fall back to a
    fixed order so the answer never depends on directory iteration.
    """
    found = [
        suite
        for suite in (
            _pytest_suite(profile),
            _node_suite(profile),
            _go_suite(profile),
            _cargo_suite(profile),
            _maven_suite(profile),
            _gradle_suite(profile),
            _unittest_suite(profile),
        )
        if suite is not None
    ]
    if not found:
        return None
    return max(
        found, key=lambda suite: (suite.test_files, -_DETECTION_ORDER.index(suite.framework))
    )


#: Ties are broken by this order, so a repository declaring two runners equally is decided the
#: same way every time rather than by whichever detector happened to run first.
_DETECTION_ORDER = (
    "pytest",
    "jest",
    "vitest",
    "mocha",
    "npm",
    "pnpm",
    "yarn",
    "go",
    "cargo",
    "maven",
    "gradle",
    "unittest",
)


def _manifest_text(profile: RepositoryProfile, name: str) -> str:
    """The contents of a top-level manifest the profile already found, or an empty string."""
    if name not in profile.manifests:
        return ""
    return (read_source(profile.root, name, max_bytes=MAX_MANIFEST_CHARS) or "").strip()


def _test_file_count(profile: RepositoryProfile, suffixes: tuple[str, ...]) -> int:
    return sum(1 for path in profile.test_files if path.endswith(suffixes))


def _pytest_suite(profile: RepositoryProfile) -> TestSuite | None:
    """pytest, when the repository configures it or is written the way pytest expects.

    The interpreter is RepoPilot's own: a target repository's dependencies are not installed in
    this environment, and ``python -m pytest`` at least finds a pytest if one exists here, which
    a bare ``pytest`` on PATH would not.
    """
    files = _test_file_count(profile, (".py",))
    configured = any(
        name.lower() in ("pytest.ini", "tox.ini") for name in profile.test_configs
    ) or "[tool.pytest" in _manifest_text(profile, "pyproject.toml")
    pytest_shaped = any(
        Path(path).name.startswith(("test_",)) or path.endswith("_test.py")
        for path in profile.test_files
        if path.endswith(".py")
    )
    if not files or not (configured or pytest_shaped):
        return None

    evidence = []
    if configured:
        evidence.append("a pytest configuration")
    if pytest_shaped:
        evidence.append("pytest-style test files")
    return TestSuite(
        framework="pytest",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ),
        language="python",
        reason=f"the repository has {' and '.join(evidence)}",
        test_files=files,
        # A missing pytest is an environment problem, not a failing test, and the difference
        # decides whether this run tries to repair code or says what to install.
        probe=(sys.executable, "-c", "import pytest"),
        probe_reason="pytest is not installed in the environment running RepoPilot",
    )


def _unittest_suite(profile: RepositoryProfile) -> TestSuite | None:
    """The standard library runner, for a python repository with tests and nothing else."""
    files = _test_file_count(profile, (".py",))
    if not files or _pytest_suite(profile) is not None:
        return None
    if not any(
        path.endswith(".py") and path.split("/")[-1].startswith("test")
        for path in profile.test_files
    ):
        return None
    return TestSuite(
        framework="unittest",
        command=(sys.executable, "-m", "unittest", "discover", "-v"),
        language="python",
        reason="the repository has python tests and declares no other runner",
        test_files=files,
    )


def _node_suite(profile: RepositoryProfile) -> TestSuite | None:
    """The node package manager's own ``test`` script, which is the only runner it promises."""
    files = _test_file_count(
        profile, (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte")
    )
    if not files:
        return None

    text = _manifest_text(profile, "package.json")
    if not text:
        return None
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        return None
    script = ((manifest.get("scripts") or {}).get("test")) if isinstance(manifest, dict) else None
    if not isinstance(script, str) or not script.strip():
        return None

    lockfiles = {name.lower() for name in profile.top_level}
    if "pnpm-lock.yaml" in lockfiles:
        manager, framework = "pnpm", "pnpm"
    elif "yarn.lock" in lockfiles:
        manager, framework = "yarn", "yarn"
    else:
        manager, framework = "npm", "npm"
    # The script names the real runner, and the runner is what formats the output worth parsing.
    for candidate in ("jest", "vitest", "mocha", "ava", "tap"):
        if candidate in script.lower():
            framework = candidate
            break
    return TestSuite(
        framework=framework,
        command=(manager, "test"),
        language="javascript",
        reason=f'package.json defines a test script: "{script.strip()[:120]}"',
        test_files=files,
    )


def _go_suite(profile: RepositoryProfile) -> TestSuite | None:
    files = _test_file_count(profile, (".go",))
    if not files or not _manifest_text(profile, "go.mod"):
        return None
    return TestSuite(
        framework="go",
        command=("go", "test", "./..."),
        language="go",
        reason="the repository has a go.mod and Go test files",
        test_files=files,
    )


def _cargo_suite(profile: RepositoryProfile) -> TestSuite | None:
    files = _test_file_count(profile, (".rs",))
    if not files or not _manifest_text(profile, "Cargo.toml"):
        return None
    return TestSuite(
        framework="cargo",
        command=("cargo", "test"),
        language="rust",
        reason="the repository has a Cargo.toml and Rust test files",
        test_files=files,
    )


def _maven_suite(profile: RepositoryProfile) -> TestSuite | None:
    files = _test_file_count(profile, (".java", ".kt"))
    if not files or not _manifest_text(profile, "pom.xml"):
        return None
    return TestSuite(
        framework="maven",
        command=("mvn", "-B", "test"),
        language="java",
        reason="the repository has a pom.xml and Java or Kotlin test files",
        test_files=files,
    )


def _gradle_suite(profile: RepositoryProfile) -> TestSuite | None:
    files = _test_file_count(profile, (".java", ".kt"))
    if not files or not any(
        _manifest_text(profile, name) for name in ("build.gradle", "build.gradle.kts")
    ):
        return None
    return TestSuite(
        framework="gradle",
        command=("gradle", "test", "--console=plain", "--no-daemon"),
        language="java",
        reason="the repository has a build.gradle and Java or Kotlin test files",
        test_files=files,
    )


# -------------------------------------------------------------------------- the output


#: Exit codes that mean "there was nothing to run" rather than "something failed". pytest uses 5
#: when collection found no tests at all, which says nothing about the change under test.
NO_TESTS_EXIT_CODES: dict[str, frozenset[int]] = {"pytest": frozenset({5})}

#: Runners that exit 0 after collecting nothing, so only their own words can be trusted. unittest
#: reports "Ran 0 tests" and still says OK, which would otherwise read as a passing suite.
_NO_TESTS_OUTPUT: dict[str, re.Pattern[str]] = {
    "unittest": re.compile(r"^Ran 0 tests\b", re.MULTILINE),
}


def collected_nothing(framework: str, exit_code: int | None, output: str = "") -> bool:
    """True when the runner found no tests to run rather than that tests failed.

    Both signals are needed and neither is enough. A non-zero exit code is how pytest says it
    collected nothing, and a runner that exits 0 having collected nothing has to be believed
    because it printed so: reporting "passed" for a suite that never ran a test is the one
    unrecoverable way to report a change as verified.
    """
    if exit_code in NO_TESTS_EXIT_CODES.get(framework, frozenset()):
        return True
    marker = _NO_TESTS_OUTPUT.get(framework)
    return marker is not None and marker.search(output) is not None


def parse_test_output(framework: str, output: str, *, exit_code: int | None) -> TestTally:
    """Read a runner's own output.

    Best effort by design. When the output says nothing countable, the counts stay ``None`` and
    the step falls back on the exit code; when the run failed and no failure could be named, one
    failure carrying the tail of the log is synthesised, because a diagnoser with no evidence
    would be guessing.
    """
    parser = _PARSERS.get(framework, _parse_unknown)
    tally = parser(output)
    if exit_code == 0 or tally.failures:
        return tally
    # A failed run whose failure could not be named still has to be diagnosable, and a
    # diagnoser handed no evidence would be guessing. The tail of the log is the evidence.
    return TestTally(
        passed=tally.passed,
        failed=tally.failed,
        errors=tally.errors,
        skipped=tally.skipped,
        failures=(
            TestFailure(
                test=f"the {framework} suite exited with status {exit_code}",
                message="the runner did not name the failing test",
                excerpt=output[-MAX_EXCERPT_CHARS:].strip(),
            ),
        ),
    )


#: The counts line pytest ends with, e.g. "=== 2 failed, 5 passed, 1 skipped in 0.42s ===".
_PYTEST_COUNTS = re.compile(r"\d+ (?:failed|passed|error|skipped)")
_PYTEST_HEADER = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?:\s+-\s+(.*))?$")
#: A new failure block starts, or the FAILURES section does: either way the excerpt restarts.
#: The "short test summary" separator deliberately does not, because pytest prints the assertion
#: detail *before* it and the ``FAILED`` line that names it *after* it.
_PYTEST_BOUNDARY = re.compile(r"^_{3,}|^={3,}\s*(?:FAILURES|ERRORS)")


def _parse_pytest(output: str) -> TestTally:
    """pytest prints an ``E``-prefixed assertion per failure and one counts line at the end."""
    lines = output.splitlines()
    failures: list[TestFailure] = []
    excerpt: list[str] = []

    for line in lines:
        stripped = line.strip()
        header = _PYTEST_HEADER.match(stripped)
        if header is not None:
            failures.append(
                TestFailure(
                    test=header.group(1),
                    message=(header.group(2) or "").strip(),
                    excerpt="\n".join(excerpt)[-MAX_EXCERPT_CHARS:].strip(),
                )
            )
            excerpt = []
            continue
        if _PYTEST_BOUNDARY.match(stripped):
            excerpt = []
        elif stripped.startswith("E "):
            excerpt.append(stripped[2:].rstrip())

    counts = next((line for line in reversed(lines) if _PYTEST_COUNTS.search(line)), "")
    return TestTally(
        passed=_count_in(counts, "passed"),
        failed=_count_in(counts, "failed"),
        errors=_count_in(counts, "error"),
        skipped=_count_in(counts, "skipped"),
        failures=tuple(failures[:MAX_REPORTED_FAILURES]),
    )


def _count_in(line: str, label: str) -> int | None:
    """A count from one line of a runner's summary, or ``None`` when it is not printed."""
    match = re.search(rf"(\d+) {label}s?\b", line)
    return int(match.group(1)) if match else None


_UNITTEST_HEADER = re.compile(r"^(?:FAIL|ERROR): (\S+)")
_UNITTEST_SUMMARY = re.compile(
    r"^(?:FAILED|OK) \((?:failures=(?P<failures>\d+))?(?:, )?(?:errors=(?P<errors>\d+))?"
)


def _parse_unittest(output: str) -> TestTally:
    """unittest prints ``FAIL: name (class)`` per failure and ``FAILED (failures=1, errors=0)``."""
    failures = [
        TestFailure(test=match.group(1))
        for line in output.splitlines()
        if (match := _UNITTEST_HEADER.match(line.strip())) is not None
    ]
    summary = next(
        (
            line.strip()
            for line in reversed(output.splitlines())
            if _UNITTEST_SUMMARY.match(line.strip())
        ),
        "",
    )
    counts = _UNITTEST_SUMMARY.match(summary)
    return TestTally(
        failed=_group(counts, "failures"),
        errors=_group(counts, "errors"),
        skipped=_assigned(summary, "skipped"),
        failures=tuple(failures[:MAX_REPORTED_FAILURES]),
    )


def _group(match: re.Match[str] | None, name: str) -> int | None:
    if match is None:
        return None
    value = match.group(name)
    return int(value) if value is not None else None


def _assigned(line: str, name: str) -> int | None:
    """A ``name=N`` figure, which is how unittest and pytest report some counts."""
    match = re.search(rf"\b{name}=(\d+)", line)
    return int(match.group(1)) if match else None


#: jest and vitest print a duration in brackets on the failure line itself, which is noise to
#: anything matching the failure by name.
_JEST_DURATION = re.compile(r"\s*\(\d+\s*m?s\)$")


def _parse_jest(output: str) -> TestTally:
    """jest, vitest and mocha all print a ``Tests:`` line and name failures with a marker."""
    failures = [
        TestFailure(test=_JEST_DURATION.sub("", line.strip("✕×✗ ").strip()))
        for line in output.splitlines()
        if line.strip().startswith(("✕", "×", "✗"))
    ]
    summary = next(
        (line for line in reversed(output.splitlines()) if line.strip().startswith("Tests:")),
        None,
    )

    def count(label: str) -> int | None:
        match = re.search(rf"(\d+) {label}", summary) if summary else None
        return int(match.group(1)) if match else None

    return TestTally(
        passed=count("passed"),
        failed=count("failed"),
        skipped=count("skipped"),
        failures=tuple(failures[:MAX_REPORTED_FAILURES]),
    )


_GO_FAIL = re.compile(r"^--- FAIL: (\S+)", re.MULTILINE)


def _parse_go(output: str) -> TestTally:
    """``go test`` names each failure and prints one line per package, never a total."""
    failures = [TestFailure(test=name) for name in _GO_FAIL.findall(output)[:MAX_REPORTED_FAILURES]]
    return TestTally(
        failed=len(_GO_FAIL.findall(output)) or None,
        passed=len(re.findall(r"^ok\s", output, re.MULTILINE)) or None,
        failures=tuple(failures),
    )


_CARGO_RESULT = re.compile(r"test result: (ok|FAILED)\..*?(\d+) passed; (\d+) failed")
_CARGO_FAIL = re.compile(r"^test (\S+) \.\.\. FAILED", re.MULTILINE)


def _parse_cargo(output: str) -> TestTally:
    summary = next(iter(_CARGO_RESULT.finditer(output)), None)
    return TestTally(
        passed=int(summary.group(2)) if summary else None,
        failed=int(summary.group(3)) if summary else None,
        skipped=_cargo_ignored(output),
        failures=tuple(
            TestFailure(test=name) for name in _CARGO_FAIL.findall(output)[:MAX_REPORTED_FAILURES]
        ),
    )


def _cargo_ignored(output: str) -> int | None:
    match = re.search(r"(\d+) ignored", output)
    return int(match.group(1)) if match else None


_MAVEN_RUNS = re.compile(
    r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", re.MULTILINE
)
_MAVEN_FAIL = re.compile(r"^(\S+)\s*--\s*Time elapsed.*?<<<\s*(?:FAILURE|ERROR)!", re.MULTILINE)


def _parse_maven(output: str) -> TestTally:
    summary = list(_MAVEN_RUNS.finditer(output))
    last = summary[-1] if summary else None
    return TestTally(
        passed=int(last.group(1)) - int(last.group(2)) - int(last.group(3)) if last else None,
        failed=int(last.group(2)) if last else None,
        errors=int(last.group(3)) if last else None,
        skipped=int(last.group(4)) if last else None,
        failures=tuple(
            TestFailure(test=match.group(1))
            for match in list(_MAVEN_FAIL.finditer(output))[:MAX_REPORTED_FAILURES]
        ),
    )


_GRADLE_RUNS = re.compile(r"(\d+) tests completed, (\d+) failed")


def _parse_gradle(output: str) -> TestTally:
    summary = list(_GRADLE_RUNS.finditer(output))
    last = summary[-1] if summary else None
    failed = int(last.group(2)) if last else None
    return TestTally(
        failed=failed,
        passed=int(last.group(1)) - failed if last and failed is not None else None,
        failures=(),
    )


def _parse_unknown(output: str) -> TestTally:
    """Nothing recognised: the exit code is the whole answer, and the tail is the evidence."""
    return TestTally()


_PARSERS: dict[str, Callable[[str], TestTally]] = {
    "pytest": _parse_pytest,
    "unittest": _parse_unittest,
    "jest": _parse_jest,
    "vitest": _parse_jest,
    "mocha": _parse_jest,
    "ava": _parse_jest,
    "tap": _parse_jest,
    "go": _parse_go,
    "cargo": _parse_cargo,
    "maven": _parse_maven,
    "gradle": _parse_gradle,
}


def tail_output(output: str, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """The end of a run's output, which is where the summary and the failure are.

    Read from the end because a test suite prints its progress first and its verdict last, and
    the head of a long log is the part nobody needs.
    """
    if not output:
        return ""
    lines = output.splitlines()[-MAX_OUTPUT_LINES:]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text
