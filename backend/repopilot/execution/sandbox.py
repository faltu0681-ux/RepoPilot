"""Running a command in the run's checkout, without blocking the event loop.

Phase 2C runs a target repository's test suite, which is the first thing this project executes
code it did not write. Three properties matter more than features here:

* **Off the loop.** The blocking read happens in a worker thread, never on the event loop. This
  is the same reason :func:`repopilot.execution.steps._run_git` shells out from a thread: under
  uvicorn's Windows reload mode the loop is a ``SelectorEventLoop``, which cannot spawn
  subprocesses at all, and a test step that worked everywhere else would fail only there. The
  loop stays free to stream events while a suite runs.
* **Interruptible.** A suite that hangs must not hold a run open forever, and a cancelled run
  must stop promptly rather than at the end of the suite. The process is polled, and a timeout or
  a cancellation kills the whole process *tree*, because ``npm test`` and ``mvn test`` leave
  children behind that would otherwise outlive the run.
* **Bounded.** Output is kept as a tail rather than accumulated, so a suite that prints without
  end cannot exhaust memory, and only the tail is ever reported.

This is a working-directory and environment boundary, not a security one. Running a repository's
tests runs that repository's code with this process's user privileges, which is inherent to
verifying a change and is why the workspace is a throwaway clone rather than a checkout of the
user's own work.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("repopilot.sandbox")

#: How long a command may run before it is killed and reported as a timeout.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: How often the running process is checked against the deadline and the cancellation flag.
POLL_INTERVAL_SECONDS = 0.2

#: How long the output readers are given to finish once the process is gone. A suite that leaves
#: a grandchild holding the pipe open would otherwise hang the step while collecting output that
#: is only ever shown to a person.
READER_JOIN_SECONDS = 5.0

#: Kept per stream. Enough for a summary and its failures, far short of an unbounded log.
DEFAULT_MAX_OUTPUT_CHARS = 200_000

#: RepoPilot's own configuration never reaches the code being tested. A test suite has no business
#: seeing the run's LLM or GitHub credentials, and a repository that read them would be a run that
#: quietly exfiltrated secrets into a third party's CI.
_HIDDEN_ENV_PREFIX = "REPOPILOT_"

#: Set for the child so its output stays readable when piped and its tree stays clean of caches.
_CHILD_ENV = {
    "NO_COLOR": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


#: Windows has no ``setsid``, so a new console process group is the equivalent. Either way the
#: child leads its own group, which is what makes :meth:`ShellSandbox._kill_tree` possible.
if os.name == "nt":  # pragma: no cover - platform specific
    _PROCESS_GROUP = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
else:  # pragma: no cover - platform specific
    _PROCESS_GROUP = {"start_new_session": True}


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one command did. Never an exception for ordinary failure: a non-zero exit is data."""

    argv: tuple[str, ...]
    cwd: Path
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        """True only for a command that ran to completion and succeeded."""
        return self.exit_code == 0 and not (self.timed_out or self.cancelled or self.error)

    @property
    def ran(self) -> bool:
        """False when the command never really ran, so its exit code says nothing about the code."""
        return not (self.timed_out or self.cancelled or self.error)

    @property
    def output(self) -> str:
        """Both streams together, stdout first, as a person running the command would see it."""
        return "\n".join(part for part in (self.stdout, self.stderr) if part.strip())

    def as_dict(self) -> dict[str, object]:
        """Metadata for an event. The output is not here; it is trimmed and sent deliberately."""
        return {
            "command": list(self.argv),
            "cwd": str(self.cwd),
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "error": self.error,
        }


class _TailBuffer:
    """A bounded buffer that keeps the end of a stream, because that is where the verdict is."""

    def __init__(self, max_chars: int) -> None:
        self._max = max_chars
        self._chunks: list[str] = []
        self._size = 0
        self.dropped_chars = 0
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            # A single line can exceed the whole budget on its own: a minified bundle, a
            # one-line stack trace, a loop with a print in it.
            if len(text) > self._max:
                self.dropped_chars += len(text) - self._max
                text = text[-self._max :]
            self._chunks.append(text)
            self._size += len(text)
            while self._size > self._max and len(self._chunks) > 1:
                dropped = self._chunks.pop(0)
                self.dropped_chars += len(dropped)
                self._size -= len(dropped)

    def read(self) -> str:
        with self._lock:
            return "".join(self._chunks)


class ShellSandbox:
    """Runs a command in a directory, off the event loop, and reports what happened.

    Deliberately small and deliberately not a jail: the working directory is the boundary, the
    environment is filtered, the process is interruptible, and everything else is the caller's
    responsibility. One instance serves every step of every run, because it holds no state.
    """

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run ``argv`` in ``cwd`` and return the result. Never raises for an ordinary failure.

        A missing binary, a permission problem or a timeout are all reported in the result
        rather than raised, because every one of them is something the user should read about
        the run instead of a stack trace the engine would turn into a generic failure.
        """
        command = tuple(str(part) for part in argv)
        limit = self.default_timeout if timeout_seconds is None else timeout_seconds
        if not command:
            return CommandResult(argv=(), cwd=cwd, error="no command to run")

        started = time.monotonic()
        try:
            process = await asyncio.to_thread(self._spawn, command, cwd, env)
        except OSError as exc:
            return CommandResult(
                argv=command,
                cwd=cwd,
                duration_seconds=time.monotonic() - started,
                error=f"could not start {command[0]!r}: {exc}",
            )

        stdout = _TailBuffer(self.max_output_chars)
        stderr = _TailBuffer(self.max_output_chars)
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        cancelled = False
        while True:
            if process.poll() is not None:
                break
            if is_cancelled is not None and is_cancelled():
                cancelled = True
                break
            if time.monotonic() - started >= limit:
                timed_out = True
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if cancelled or timed_out:
            await asyncio.to_thread(self._kill_tree, process)
        exit_code = await asyncio.to_thread(process.wait)
        for reader in readers:
            # Bounded: a surviving grandchild may still hold the pipe open, and its output is
            # only ever shown to a person, so there is nothing to gain by waiting for it.
            await asyncio.to_thread(reader.join, READER_JOIN_SECONDS)

        duration = time.monotonic() - started
        if cancelled:
            return CommandResult(
                argv=command,
                cwd=cwd,
                exit_code=exit_code,
                stdout=stdout.read(),
                stderr=stderr.read(),
                duration_seconds=duration,
                cancelled=True,
                error="the run was cancelled while the command was still going",
            )
        if timed_out:
            return CommandResult(
                argv=command,
                cwd=cwd,
                exit_code=exit_code,
                stdout=stdout.read(),
                stderr=stderr.read(),
                duration_seconds=duration,
                timed_out=True,
                error=f"the command was still running after {limit:g}s",
            )
        return CommandResult(
            argv=command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=stdout.read(),
            stderr=stderr.read(),
            duration_seconds=duration,
        )

    def _spawn(
        self, argv: tuple[str, ...], cwd: Path, env: dict[str, str] | None
    ) -> subprocess.Popen[str]:
        return subprocess.Popen(  # noqa: S603 - a fixed argv, never a shell string
            list(argv),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=self._child_env(env),
            shell=False,
            # A new process group or console group is what makes the tree killable, and it also
            # stops a test that reads from the terminal from hanging on an inherited handle.
            **_PROCESS_GROUP,
        )

    def _child_env(self, overrides: dict[str, str] | None) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(_HIDDEN_ENV_PREFIX)
        }
        env.update(_CHILD_ENV)
        if overrides:
            env.update(overrides)
        return env

    def _kill_tree(self, process: subprocess.Popen[str]) -> None:
        """Kill the command and everything it started.

        ``npm test`` spawns a shell, which spawns the runner, which may spawn a compiler. Killing
        only the process that was launched leaves those behind, still holding the pipes and still
        writing into a workspace the run is about to delete.
        """
        if process.poll() is not None:
            return
        if os.name == "nt":
            with contextlib.suppress(OSError):
                subprocess.run(  # noqa: S603 - fixed argv
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    timeout=READER_JOIN_SECONDS,
                    check=False,
                )
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=READER_JOIN_SECONDS)


def _drain(stream: object, sink: _TailBuffer) -> None:
    """Copy a pipe into a buffer until EOF. Runs in a thread; a test suite is not polite."""
    if stream is None:  # pragma: no cover - only when a caller redirects the streams
        return
    for chunk in iter(stream.readline, ""):  # type: ignore[attr-defined]
        sink.write(chunk)
    with contextlib.suppress(OSError, ValueError):
        stream.close()  # type: ignore[attr-defined]
