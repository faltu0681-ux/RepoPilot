"""Running a command in a throwaway container instead of on the host.

This is the V1 answer to "the agent writes code, and that code has to run somewhere". It is a
*better* boundary than the shell backend, not a perfect one, and the difference is stated here
rather than discovered later.

**What the container runtime actually enforces.** For each command, RepoPilot starts one container
that mounts exactly this run's checkout and nothing else. That gives a kernel-enforced filesystem
boundary where the shell backend has only a convention; a container cannot open a file outside its
mount without the runtime refusing. On top of that RepoPilot asks for no network, no privilege
escalation, no Linux capabilities, and hard caps on memory, CPU and process count. Those are real
limits enforced by the runtime, and they are what stops a runaway test suite from taking the
machine with it.

**What it does not enforce, stated plainly.** A container is not a security boundary against a
determined attacker, and this one is no better than any other:

* Anyone who can talk to the Docker daemon, or who can start containers, effectively has root on the
  host. Docker's own documentation says this and this project does not pretend otherwise.
* A kernel or runtime vulnerability defeats the whole thing.
* The mounted workspace is host-writable, so anything the container writes is written by the host
  user. That is how the agent's changes come back out, which is the entire point, and it is also
  why a container can corrupt its own workspace.
* Mounting the user's ``.ssh`` or forwarding a credential would remove the boundary entirely. This
  module never mounts a host path other than the run's checkout, and never forwards a credential.

``docs/architecture.md`` carries the full threat model. The short version: this protects a
developer running RepoPilot against a repository they do not fully trust. It is not a multi-tenant
hosting boundary, and it is not a substitute for reviewing what a model wrote.

**Why one container per command.** Nothing is kept alive between commands. What carries state from
one command to the next is the mounted workspace on the host, which is why a suite that fails, gets
repaired, and runs again sees its own edits. The cost is a second of startup per command. The
benefit is that there is no long-lived container to leak, no stale state to clean up, and no
cleanup path that can itself fail.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from repopilot.execution.environment import ISOLATION_SUMMARY, ExecutionBackend, SandboxError
from repopilot.execution.sandbox import (
    DEFAULT_MAX_OUTPUT_CHARS,
    POLL_INTERVAL_SECONDS,
    READER_JOIN_SECONDS,
    CommandResult,
    TailBuffer,
    drain_stream,
)

logger = logging.getLogger("repopilot.docker")

#: Where the run's checkout appears inside the container. A fixed path, because the only thing
#: mounted is the checkout itself, and everything the agent does is relative to it.
CONTAINER_WORKSPACE = "/workspace"

#: Only ever one mount target. A second mount is how sandbox escapes start, so the argv builder
#: takes a directory and refuses anything it was not given.
CONTAINER_TMP = "/tmp"

#: Default image. Deliberately a pinned minor tag rather than ``latest``: a run's verdict should not
#: change because someone republished a tag, and a digest belongs here once V1 picks a stack.
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"

#: Environment variables passed from the host when they are set and on the allowlist. Note what is
#: *not* here: ``PATH``, ``JAVA_HOME``, ``GOPATH`` and friends are host paths that do not exist
#: inside the container, and forwarding them is a footgun that produces a confusing "command not
#: found" instead of a clear one. The image supplies its own toolchain.
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = ("LANG", "LC_ALL", "TZ", "CI", "TERM")

#: Never forwarded, whatever the allowlist says. This is the credential boundary: the agent's code
#: must not be able to read RepoPilot's own secrets, so the sandbox gets an environment built from
#: scratch rather than an inherited one with things removed.
NEVER_FORWARDED = ("REPOPILOT_GITHUB_TOKEN", "REPOPILOT_LLM_API_KEY")

#: How long to wait for the daemon to answer "do you have this image?". Purely a local question, so
#: it gets a short ceiling and is never what the operator configured. Downloading the image is a
#: different, much slower operation, and that one is ``sandbox_startup_timeout_seconds``.
IMAGE_PROBE_TIMEOUT_SECONDS = 30.0


class DockerSandbox:
    """Runs each command in a fresh, disposable container with the run's workspace mounted."""

    name = ExecutionBackend.DOCKER.value

    def __init__(
        self,
        *,
        workspace_root: Path,
        image: str = DEFAULT_SANDBOX_IMAGE,
        network: str = "none",
        memory: str = "2g",
        cpus: float = 2.0,
        pids_limit: int = 512,
        default_timeout: float = 600.0,
        startup_timeout: float = 300.0,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        docker_binary: str = "docker",
        env_allowlist: Sequence[str] = DEFAULT_ENV_ALLOWLIST,
        user: str = "",
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.image = image
        #: ``"none"`` is the default and should stay that way. Anything else is opt-in, because
        #: network access is the difference between "the agent builds code" and "the agent can talk
        #: to the internet while building code".
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.default_timeout = default_timeout
        self.startup_timeout = startup_timeout
        self.max_output_chars = max_output_chars
        self.docker_binary = docker_binary
        self.env_allowlist = tuple(env_allowlist)
        #: Empty means "the image's own user". Setting it to a non-root uid is better; it is not the
        #: default because it silently breaks images that need to write to root-owned paths, and a
        #: default that breaks things gets overridden.
        self.user = user
        self._counter = itertools.count()
        self._run_prefix = f"repopilot-{uuid.uuid4().hex[:10]}"
        self._docker_checked = False
        self._image_checked = False

    def describe(self) -> dict[str, Any]:
        """What this backend enforces. Reported on the run so nobody has to guess."""
        return {
            "name": self.name,
            "isolation": ISOLATION_SUMMARY[ExecutionBackend.DOCKER],
            "image": self.image,
            "network": self.network,
            "filesystem": f"only this run's checkout is mounted, at {CONTAINER_WORKSPACE}",
            "resources": (
                f"memory {self.memory}, {self.cpus:g} cpu, "
                f"{self.pids_limit} processes, {self.default_timeout:g}s command timeout, "
                f"{self.startup_timeout:g}s image startup"
            ),
        }

    async def aclose(self) -> None:
        """Nothing is kept between commands, so there is nothing to release."""
        return None

    # --------------------------------------------------------------------- run

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        command = tuple(str(part) for part in argv)
        limit = self.default_timeout if timeout_seconds is None else timeout_seconds
        if not command:
            return CommandResult(argv=(), cwd=cwd, error="no command to run")

        container_cwd = self._container_path(cwd)
        container_name = f"{self._run_prefix}-{next(self._counter)}"
        docker_argv = self._docker_argv(command, container_cwd, container_name, env)

        started = time.monotonic()
        available, detail = await asyncio.to_thread(self._docker_available)
        if not available:
            return CommandResult(
                argv=command,
                cwd=cwd,
                duration_seconds=time.monotonic() - started,
                error=f"the docker execution backend is unavailable: {detail}",
            )

        # The image has to exist before the command's clock starts, or a cold first run spends the
        # suite's own timeout on a download and reports a *test* timeout for it.
        ready, detail = await asyncio.to_thread(self._ensure_image)
        if not ready:
            return CommandResult(
                argv=command,
                cwd=cwd,
                duration_seconds=time.monotonic() - started,
                error=f"the docker sandbox image is unavailable: {detail}",
            )
        # From here on the suite timeout means what the operator thinks it means: the suite.
        started = time.monotonic()

        try:
            process = await asyncio.to_thread(self._spawn, docker_argv, container_name, cwd)
        except (OSError, SandboxError) as exc:
            # A RepoPilot defect (a path outside the workspace) raises out of ``_container_path``
            # above and stays loud. A binary that vanished between the check and the spawn is an
            # environment condition, and belongs in the result rather than in a stack trace.
            return CommandResult(
                argv=command,
                cwd=cwd,
                duration_seconds=time.monotonic() - started,
                error=str(exc),
            )

        stdout = TailBuffer(self.max_output_chars)
        stderr = TailBuffer(self.max_output_chars)
        readers = [
            threading.Thread(target=drain_stream, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=drain_stream, args=(process.stderr, stderr), daemon=True),
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
            # Killing the client is not enough: the container keeps running unless it is removed.
            # This is the one cleanup step that exists, and it is why the container is named.
            await asyncio.to_thread(self._remove, container_name)
            with contextlib.suppress(OSError):
                process.kill()

        exit_code = await asyncio.to_thread(process.wait)
        for reader in readers:
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
            error=self._error_for(exit_code, stdout.read(), stderr.read(), command),
        )

    # ----------------------------------------------------------------- private

    def _container_path(self, cwd: Path) -> str:
        """Map a host directory into the container, refusing anything outside the checkout.

        This is the check that stops a bug in RepoPilot from mounting the user's home directory, and
        it is a refusal rather than a clamp: a path outside the run's own checkout has no business
        being executed, so quietly rewriting it to the workspace root would hide a real defect.
        """
        resolved = cwd.expanduser().resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise SandboxError(
                f"{resolved} is outside this run's workspace ({self.workspace_root}); the docker "
                "backend refuses to run a command against a directory it did not mount"
            )
        relative = resolved.relative_to(self.workspace_root)
        if str(relative) == ".":
            return CONTAINER_WORKSPACE
        return f"{CONTAINER_WORKSPACE}/{relative.as_posix()}"

    def _docker_argv(
        self,
        command: tuple[str, ...],
        container_cwd: str,
        container_name: str,
        env: dict[str, str] | None,
    ) -> list[str]:
        """Build the ``docker run`` argv. Readable on purpose: this is the security policy."""
        argv: list[str] = [self.docker_binary, "run", "--rm", "--name", container_name]

        # The isolation controls, each one a deliberate line rather than a bundled flag list.
        argv += ["--network", self.network]
        argv += ["--memory", self.memory, "--memory-swap", self.memory]
        argv += ["--cpus", f"{self.cpus:g}"]
        argv += ["--pids-limit", str(self.pids_limit)]
        argv += ["--cap-drop", "ALL"]
        argv += ["--security-opt", "no-new-privileges"]
        # The image's own filesystem stays read-only. The two writable places are the mounted
        # workspace and tmpfs /tmp, which is what most build tools need to exist.
        argv += ["--read-only"]
        argv += ["--tmpfs", f"{CONTAINER_TMP}:rw,nosuid,nodev,size=256m"]
        if self.user:
            argv += ["--user", self.user]

        # The single mount. No ssh agent, no docker socket, no home directory, no config file.
        argv += ["--volume", f"{self.workspace_root}:{CONTAINER_WORKSPACE}"]
        argv += ["--workdir", container_cwd]

        # HOME points into the tmpfs, because the root filesystem is read-only and most tools want
        # somewhere to put a cache.
        argv += ["--env", f"HOME={CONTAINER_TMP}"]
        for key, value in self._container_env(env).items():
            argv += ["--env", f"{key}={value}"]

        argv += [self.image, *command]
        return argv

    def _container_env(self, overrides: dict[str, str] | None) -> dict[str, str]:
        """The sandbox's environment, built rather than inherited.

        Built from scratch on purpose. The shell backend can afford to filter an inherited
        environment; here, an allowlist is the whole control, so a host variable that nobody
        thought about cannot ride along.
        """
        env = {key: os.environ[key] for key in self.env_allowlist if key in os.environ}
        for key in NEVER_FORWARDED:
            env.pop(key, None)
        if overrides:
            for key, value in overrides.items():
                if key.upper().startswith("REPOPILOT_"):
                    continue
                env[key] = value
        return env

    def _spawn(
        self, docker_argv: list[str], container_name: str, cwd: Path
    ) -> subprocess.Popen[str]:
        try:
            return subprocess.Popen(  # noqa: S603 - fixed argv, never a shell string
                docker_argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                shell=False,
            )
        except OSError as exc:
            raise SandboxError(
                f"could not start {self.docker_binary!r}: {exc}. Install Docker Desktop or the "
                "docker engine, or set REPOPILOT_EXECUTION_BACKEND=local."
            ) from exc

    def _docker_available(self) -> tuple[bool, str]:
        """Check the daemon once per sandbox and remember, because this runs per command."""
        if self._docker_checked:
            return True, ""
        binary = shutil.which(self.docker_binary)
        if binary is None:
            return False, (
                f"{self.docker_binary!r} is not on PATH. Install Docker, or set "
                "REPOPILOT_EXECUTION_BACKEND=local to run commands on the host instead."
            )
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv
                [self.docker_binary, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"could not reach the docker daemon: {exc}"

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit {completed.returncode}"
            return False, f"the docker daemon is not responding ({reason})"

        self._docker_checked = True
        return True, ""

    def _ensure_image(self) -> tuple[bool, str]:
        """Get the image onto the host before the command clock starts, and remember that.

        Without this the first command of a run is also the ``docker pull``, so the suite's timeout
        is spent downloading and a pull slower than the suite kills the run mid-download and blames
        the test suite for it. Startup gets its own budget instead, which is what
        ``sandbox_startup_timeout_seconds`` is for. The check is local and therefore short: it asks
        the daemon whether it has the image, and only downloads when it genuinely does not.
        """
        if self._image_checked:
            return True, ""

        try:
            probe = subprocess.run(  # noqa: S603 - fixed argv
                [self.docker_binary, "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=IMAGE_PROBE_TIMEOUT_SECONDS,
                check=False,
            )
            if probe.returncode == 0:
                self._image_checked = True
                return True, ""

            pull = subprocess.run(  # noqa: S603 - fixed argv
                [self.docker_binary, "pull", self.image],
                capture_output=True,
                text=True,
                timeout=self.startup_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, (
                f"{self.image!r} could not be made ready within {self.startup_timeout:g}s. A large "
                "image on a slow link, or a daemon that has stopped answering."
            )
        except OSError as exc:
            return False, f"could not run {self.docker_binary!r}: {exc}"

        if pull.returncode != 0:
            detail = (pull.stderr or pull.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit {pull.returncode}"
            return False, (
                f"{self.image!r} is not available locally and could not be pulled ({reason}). "
                "Build it, or point REPOPILOT_SANDBOX_IMAGE at an image that exists."
            )

        self._image_checked = True
        return True, ""

    def _remove(self, container_name: str) -> None:
        """Force-remove a container that outlived its command. Best effort by design.

        Nothing here may raise: this runs while a run is being torn down, and a cleanup failure must
        not become the run's failure. The container is ``--rm``, so docker removes it on exit too.
        """
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed argv
                [self.docker_binary, "rm", "--force", "--volumes", container_name],
                capture_output=True,
                timeout=READER_JOIN_SECONDS,
                check=False,
            )

    def _error_for(
        self, exit_code: int | None, stdout: str, stderr: str, command: tuple[str, ...]
    ) -> str:
        """Explain a failure the runtime caused, rather than reporting it as a test failure.

        A missing image, a missing daemon, and an unmountable path all surface as a non-zero exit
        with a distinctive message. Without this, the run would report "the tests failed" when in
        fact no test ever ran, which is exactly the lie the verdict system exists to prevent.
        """
        if exit_code in (125, 126, 127):
            combined = f"{stdout}\n{stderr}".lower()
            # Docker's wording varies by version and by whether a pull was attempted, so all of
            # these have to be recognised: matching only one of them means the common case falls
            # through to the generic message and the operator loses the remedy.
            if any(
                phrase in combined
                for phrase in (
                    "unable to find image",
                    "no such image",
                    "pull access denied",
                    "repository does not exist",
                    "manifest unknown",
                )
            ):
                return (
                    f"sandbox image {self.image!r} is not available locally and could not be "
                    "pulled; build it or point REPOPILOT_SANDBOX_IMAGE at an image that exists"
                )
            if "cannot connect to the docker daemon" in combined:
                return "the docker daemon stopped responding during the command"
            if "permission denied" in combined and "mount" in combined:
                return f"the docker daemon refused to mount the workspace: {stderr.strip()[:400]}"
            return f"the sandbox could not run {command[0]!r} ({stderr.strip()[:400]})"
        return ""


__all__ = [
    "CONTAINER_WORKSPACE",
    "DEFAULT_ENV_ALLOWLIST",
    "DEFAULT_SANDBOX_IMAGE",
    "IMAGE_PROBE_TIMEOUT_SECONDS",
    "NEVER_FORWARDED",
    "DockerSandbox",
]
