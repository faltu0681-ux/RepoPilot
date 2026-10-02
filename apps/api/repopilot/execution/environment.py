"""Where a run's commands execute.

RepoPilot runs code it did not write: a target repository's test suite, then whatever the model
asks for while repairing a failure. *Where* that code runs is an architectural decision, not an
implementation detail, so it is named here rather than implied by whichever class happens to be
constructed.

**One interface, two backends.** :class:`ExecutionEnvironment` is a :class:`~typing.Protocol`, so
the shell backend that has existed since Phase 2C satisfies it without inheriting from anything and
without a single line changed at its call sites. A container backend is a second implementation of
the same two-method contract. Kubernetes, when it is ever needed, is a third implementation of the
same contract, and the agent engine cannot tell the difference: it asks for a directory and a
command and gets back a :class:`~repopilot.execution.sandbox.CommandResult`.

**The workspace is durable; the sandbox is not.** This is the decision that keeps V1 small.
RepoPilot does not keep a container alive across a run. Each command gets a fresh,
``--rm``, and what survives between commands is the *workspace directory on the host*, which is
mounted in. Editing a file, running a suite, and repairing it all operate on the same files because
they are the same files, not because anything was kept alive. The consequences are deliberate:

* A crashed or killed container cannot leak into the next command. There is no cleanup step that
  can fail, because there is no long-lived thing to clean up.
* A per-command container costs roughly a second of startup, which a test suite does not care about.
* An interactive shell that spans commands would need a different lifecycle. V1 has no such
  requirement, so it does not pay for one. See the architecture document for when that changes.

**Isolation is not a promise, it is a description.** :meth:`ExecutionEnvironment.describe` returns
what a backend actually enforces, and the run's summary and the UI report exactly that. A backend
that cannot enforce something says so rather than implying protection it does not have. In
particular ``local`` enforces a working directory and a filtered environment and nothing else; it is
not a jail. ``docs/architecture.md`` carries the threat model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from repopilot.core.errors import RepopilotError
from repopilot.execution.sandbox import CommandResult, ShellSandbox

logger = logging.getLogger("repopilot.environment")


class SandboxError(RepopilotError):
    """An execution environment could not be created or used."""

    status_code = 500


class SandboxUnavailable(SandboxError):
    """The configured execution backend is not usable on this machine.

    Distinct from :class:`SandboxError` because it is almost always a setup problem with a clear
    remedy, and a run that hits it should say which remedy rather than fail vaguely.
    """

    status_code = 503


class ExecutionBackend(StrEnum):
    """The environments V1 can run commands in."""

    #: A child process on this host, confined by working directory and environment.
    LOCAL = "local"
    #: A throwaway container, confined by the container runtime's own boundary.
    DOCKER = "docker"


#: What each backend actually restricts. Rendered into run events and shown in the UI, because a
#: reader deciding whether to trust a run needs to know which boundary produced it.
ISOLATION_SUMMARY: dict[ExecutionBackend, str] = {
    ExecutionBackend.LOCAL: (
        "host process: restricted to the run's workspace directory and a filtered environment, "
        "with no filesystem, network, or resource isolation"
    ),
    ExecutionBackend.DOCKER: (
        "ephemeral container: the run's workspace is the only mounted path, networking is off by "
        "default, and CPU, memory, and process counts are capped"
    ),
}


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Somewhere a command can be run and its outcome reported.

    Deliberately one method. Anything a backend needs in order to be useful to the agent engine —
    how to name itself, how to describe its own boundary — is either data on the object or a second
    trivially-implemented method. The engine asks for work and reads a result; it never manages a
    container, which is what keeps the Kubernetes path from becoming an engine rewrite.
    """

    #: Which backend this is, so a run can report where its commands ran.
    name: str

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run ``argv`` in ``cwd`` and report what happened. Never raises for an ordinary failure.

        A non-zero exit is data, not an exception. So are a missing binary, a timeout, and a
        cancellation: each is a fact about the run that belongs in the terminal and the summary
        rather than in a stack trace.
        """
        ...

    async def aclose(self) -> None:
        """Release anything held across commands. A no-op for stateless backends."""

    def describe(self) -> dict[str, Any]:
        """What this environment enforces, for events and the UI. Never a security guarantee."""
        ...


def build_execution_environment(
    settings: Any, *, scope: Path | None = None
) -> ExecutionEnvironment:
    """Construct the backend the configuration asked for.

    ``scope`` is the one directory a backend is allowed to touch: this run's own checkout. The
    container backend mounts exactly that, so one run cannot see another's workspace, and a run of
    one repository cannot reach a different repository's checkout on the same machine. The shell
    backend ignores it, because a working directory is not a mount and there is nothing to confine
    to.

    Fails loudly and specifically when a backend is configured but cannot run, because silently
    degrading to the host shell would turn "the agent's code runs in a container" into "the agent's
    code runs on your machine" without anyone being told. The error names the remedy.
    """
    backend = ExecutionBackend(str(settings.execution_backend).strip().lower())

    if backend is ExecutionBackend.LOCAL:
        return ShellSandbox(default_timeout=settings.test_timeout_seconds)

    if scope is None:
        raise SandboxUnavailable(
            "the docker execution backend needs a workspace directory to mount, and this build "
            "did not provide one"
        )

    # Imported here rather than at module scope so that a local-only install never touches the
    # container module, and so the one-way dependency (environment -> docker -> sandbox) is obvious
    # from this function alone.
    from repopilot.execution.docker import DockerSandbox

    logger.info("using the %s execution backend for untrusted commands", backend.value)
    return DockerSandbox(
        workspace_root=scope,
        image=settings.sandbox_image,
        network=settings.sandbox_network,
        memory=settings.sandbox_memory,
        cpus=settings.sandbox_cpus,
        pids_limit=settings.sandbox_pids_limit,
        default_timeout=settings.test_timeout_seconds,
        startup_timeout=settings.sandbox_startup_timeout_seconds,
        env_allowlist=settings.sandbox_env_allowlist,
        user=settings.sandbox_user,
        docker_binary=settings.sandbox_docker_binary,
    )


__all__ = [
    "ISOLATION_SUMMARY",
    "ExecutionBackend",
    "ExecutionEnvironment",
    "SandboxError",
    "SandboxUnavailable",
    "build_execution_environment",
]
