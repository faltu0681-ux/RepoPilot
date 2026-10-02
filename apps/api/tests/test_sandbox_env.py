"""The execution environment boundary: what a run's untrusted commands can reach.

Two questions are asked here, and they are deliberately different questions.

**"What did we ask for?"** is answered by inspecting the argv the container backend builds. That
argv *is* the security policy: the mounts, the network, the limits, the environment. Asserting on it
directly is more honest than trying to infer a policy from a container that happened to misbehave,
and it needs no container runtime, so these tests run on a machine with no Docker at all.

**"What happens when it goes wrong?"** is answered by a fake docker CLI that records what it was
asked to do and returns outcomes a working daemon never would: a suite that hangs, a command that
fails, a runtime that cannot start, a daemon that is not there. The code under test runs its real
path all the way down to the process spawn and the cleanup call, so the lifecycle is genuinely
exercised rather than mocked away.

The local backend is included deliberately. It is the default, and a run must be able to say
honestly that its commands ran on the host with no isolation rather than implying otherwise.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from repopilot.config import Settings
from repopilot.domain.models import EventType
from repopilot.domain.store import RunStore
from repopilot.execution.bus import EventBus
from repopilot.execution.docker import (
    CONTAINER_WORKSPACE,
    DEFAULT_ENV_ALLOWLIST,
    DockerSandbox,
)
from repopilot.execution.engine import RunEngine
from repopilot.execution.environment import (
    ExecutionBackend,
    ExecutionEnvironment,
    SandboxError,
    build_execution_environment,
)
from repopilot.execution.sandbox import ShellSandbox
from repopilot.execution.steps import Step, StepContext, StepOutcome
from tests.docker_fakes import (
    MARK_EXIT,
    MARK_RUNTIME_ERROR,
    MARK_SLEEP,
    FakeDocker,
    requires_docker,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """This run's own checkout, standing in for ``<workspace>/<owner>__<repo>/<run_id>``."""
    root = tmp_path / "workspace" / "acme__widgets" / "run_123"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def docker(checkout: Path, fake_docker: FakeDocker) -> DockerSandbox:
    return DockerSandbox(
        workspace_root=checkout,
        docker_binary=str(fake_docker.binary),
        default_timeout=10.0,
    )


def _engine(
    settings: Settings,
    *,
    steps: tuple[Step, ...],
    execution_backend: str = "local",
    docker_binary: str = "docker",
) -> RunEngine:
    """An engine whose pipeline is whatever the test wants, so only the wiring is exercised.

    Built with the same ``Settings`` the app would use, because the backend choice comes from
    configuration and a test that constructed the backend directly would prove nothing about it.
    """
    configured = settings.model_copy(
        update={
            "execution_backend": execution_backend,
            "sandbox_docker_binary": docker_binary,
        }
    )
    return RunEngine(
        store=RunStore(),
        bus=EventBus(),
        settings=configured,
        steps=steps,
    )


def _value_of(argv: list[str], flag: str) -> str | None:
    """The argument after ``flag``, or None. The tests read like the policy they assert."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def _env_pairs(argv: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for index, token in enumerate(argv):
        if token == "--env" and index + 1 < len(argv):
            key, _, value = argv[index + 1].partition("=")
            pairs[key] = value
    return pairs


# ------------------------------------------------------------ the interface itself


class TestExecutionEnvironmentInterface:
    """Both backends are the same thing to the engine, which is what makes V2 additive."""

    def test_both_backends_satisfy_the_protocol(self, checkout: Path) -> None:
        settings = Settings(_env_file=None)
        local = build_execution_environment(settings)
        container = build_execution_environment(
            settings.model_copy(update={"execution_backend": "docker"}), scope=checkout
        )

        assert isinstance(local, ExecutionEnvironment)
        assert isinstance(container, ExecutionEnvironment)

    def test_the_local_backend_is_the_default(self, checkout: Path) -> None:
        """A fresh clone must be able to run a test without Docker installed."""
        assert build_execution_environment(Settings(_env_file=None)).name == "local"

    def test_the_docker_backend_needs_a_scope(self) -> None:
        settings = Settings(_env_file=None, execution_backend="docker")
        with pytest.raises(SandboxError, match="workspace directory to mount"):
            build_execution_environment(settings)

    def test_an_unknown_backend_is_rejected_at_configuration_time(self) -> None:
        """A typo must fail at startup, not halfway through a run that looks like a broken repo."""
        with pytest.raises(ValueError, match="unsupported execution_backend"):
            Settings(_env_file=None, execution_backend="kubernets")

    def test_the_backend_name_is_case_and_space_insensitive(self, checkout: Path) -> None:
        settings = Settings(_env_file=None, execution_backend=" Docker ")
        built = build_execution_environment(settings, scope=checkout)
        assert built.name == ExecutionBackend.DOCKER.value

    def test_the_local_backend_describes_its_own_weaknesses(self) -> None:
        """A run must not be able to imply isolation it does not have."""
        described = ShellSandbox().describe()
        assert described["name"] == "local"
        assert "no filesystem" in str(described["isolation"])
        assert "unrestricted" in str(described["network"])
        assert "working directory only" in str(described["filesystem"])


# ------------------------------------------------------- the docker security policy


class TestDockerIsolationPolicy:
    """These assertions are the security policy, read straight off the argv."""

    def test_only_this_runs_checkout_is_mounted(
        self, docker: DockerSandbox, checkout: Path
    ) -> None:
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)

        volumes = [argv[i + 1] for i, token in enumerate(argv) if token == "--volume"]
        assert volumes == [f"{checkout}:{CONTAINER_WORKSPACE}"]

    def test_nothing_else_from_the_host_is_reachable(self, docker: DockerSandbox) -> None:
        """No home directory, no ssh agent, no docker socket, no credentials file."""
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)
        joined = " ".join(argv).lower()

        for forbidden in (".ssh", "docker.sock", "id_rsa", "/etc/passwd", ".netrc", ".aws"):
            assert forbidden not in joined, f"{forbidden} must never reach the container"

    def test_the_network_is_off_by_default(self, docker: DockerSandbox) -> None:
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)
        assert _value_of(argv, "--network") == "none"

    def test_a_chosen_network_is_honoured(self, checkout: Path, fake_docker: FakeDocker) -> None:
        """Opting in works, so the restrictive default is a decision rather than a hard limit."""
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            network="bridge",
        )
        argv = sandbox._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)
        assert _value_of(argv, "--network") == "bridge"

    def test_resources_are_capped(self, docker: DockerSandbox) -> None:
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)

        assert _value_of(argv, "--memory") == "2g"
        # Swap equal to memory disables overcommit, so a fork bomb cannot page the host to death.
        assert _value_of(argv, "--memory-swap") == _value_of(argv, "--memory")
        assert _value_of(argv, "--cpus") == "2"
        assert _value_of(argv, "--pids-limit") == "512"

    def test_privilege_escalation_is_refused(self, docker: DockerSandbox) -> None:
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)

        assert "--read-only" in argv
        assert _value_of(argv, "--cap-drop") == "ALL"
        assert _value_of(argv, "--security-opt") == "no-new-privileges"
        assert "--privileged" not in argv

    def test_the_container_is_removed_when_it_exits(self, docker: DockerSandbox) -> None:
        """``--rm`` plus the name is what makes cleanup possible at all."""
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)
        assert "--rm" in argv
        assert _value_of(argv, "--name") == "c1"


# ------------------------------------------------------- the credential boundary


class TestSecretsStayOutOfTheSandbox:
    """The control plane holds the GitHub token. The agent's code must never see it."""

    def test_repopilot_credentials_are_never_forwarded(
        self, docker: DockerSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REPOPILOT_GITHUB_TOKEN", "ghp_supersecret")
        monkeypatch.setenv("REPOPILOT_LLM_API_KEY", "sk-supersecret")

        env = docker._container_env(None)

        assert "REPOPILOT_GITHUB_TOKEN" not in env
        assert "REPOPILOT_LLM_API_KEY" not in env
        assert "supersecret" not in "".join(f"{k}={v}" for k, v in env.items())

    def test_the_environment_is_built_not_inherited(
        self, docker: DockerSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allowlist means a host variable nobody considered cannot ride along."""
        monkeypatch.setenv("SOME_UNDISCUSSED_HOST_SECRET", "leaked")

        env = docker._container_env(None)

        assert "SOME_UNDISCUSSED_HOST_SECRET" not in env
        assert set(env) <= set(DEFAULT_ENV_ALLOWLIST)

    def test_host_tool_paths_are_not_forwarded(
        self, docker: DockerSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``PATH`` and ``JAVA_HOME`` name host locations that do not exist in a container.

        Forwarding them produces a confusing "command not found" instead of an honest missing
        runner, so they are excluded by default and the image supplies its own toolchain.
        """
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
        monkeypatch.setenv("JAVA_HOME", r"C:\Program Files\Java")

        env = docker._container_env(None)

        assert "PATH" not in env
        assert "JAVA_HOME" not in env

    def test_a_caller_cannot_sneak_a_credential_past_the_filter(
        self, docker: DockerSandbox
    ) -> None:
        env = docker._container_env({"REPOPILOT_GITHUB_TOKEN": "ghp_leak", "SAFE": "yes"})

        assert "REPOPILOT_GITHUB_TOKEN" not in env
        assert env["SAFE"] == "yes"

    def test_home_points_at_the_writable_tmpfs(self, docker: DockerSandbox) -> None:
        """The image's root filesystem is read-only, so tools need somewhere to put a cache."""
        argv = docker._docker_argv(("pytest",), CONTAINER_WORKSPACE, "c1", None)
        assert _env_pairs(argv)["HOME"] == "/tmp"
        assert any(token.startswith("/tmp:") for token in argv)


# ------------------------------------------------------ workspace mapping / escape


class TestWorkspaceMapping:
    def test_the_checkout_root_maps_to_the_workspace_root(self, docker: DockerSandbox) -> None:
        assert docker._container_path(docker.workspace_root) == CONTAINER_WORKSPACE

    def test_a_subdirectory_maps_to_the_same_relative_path(
        self, docker: DockerSandbox, checkout: Path
    ) -> None:
        assert (
            docker._container_path(checkout / "src" / "app.py")
            == f"{CONTAINER_WORKSPACE}/src/app.py"
        )

    def test_a_path_outside_the_checkout_is_refused(
        self, docker: DockerSandbox, tmp_path: Path
    ) -> None:
        """The control that stops a RepoPilot bug from mounting the user's home directory.

        A refusal rather than a clamp: silently rewriting the path to the workspace root would hide
        a real defect behind a passing test.
        """
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        with pytest.raises(SandboxError, match="outside this run's workspace"):
            docker._container_path(outside)

    def test_a_traversal_out_of_the_checkout_is_refused(
        self, docker: DockerSandbox, checkout: Path
    ) -> None:
        with pytest.raises(SandboxError, match="outside this run's workspace"):
            docker._container_path(checkout / ".." / ".." / "other" / "run")

    def test_a_sibling_run_is_unreachable(self, checkout: Path) -> None:
        """One run must not be able to see another run's work on the same machine."""
        sibling = checkout.parent / "run_999"
        sibling.mkdir()
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary="docker")

        with pytest.raises(SandboxError):
            sandbox._container_path(sibling / "main.py")

    def test_the_working_directory_reaches_the_container(self, docker: DockerSandbox) -> None:
        argv = docker._docker_argv(("pytest",), f"{CONTAINER_WORKSPACE}/pkg", "c1", None)
        assert _value_of(argv, "--workdir") == f"{CONTAINER_WORKSPACE}/pkg"

    def test_the_command_runs_inside_the_image(self, docker: DockerSandbox) -> None:
        argv = docker._docker_argv(
            ("python", "-m", "pytest", "-q"), CONTAINER_WORKSPACE, "c1", None
        )
        assert argv[-4:] == ["python", "-m", "pytest", "-q"]
        assert "python:3.12-slim" in argv


# ------------------------------------------------------------ the command lifecycle


class TestDockerCommandLifecycle:
    async def test_a_command_that_works_reports_success(self, docker: DockerSandbox) -> None:
        result = await docker.run(("python", "-m", "pytest"), cwd=docker.workspace_root)

        assert result.ran
        assert result.ok
        assert result.exit_code == 0
        assert result.error == ""
        assert "ran the command" in result.stdout

    async def test_a_failing_command_is_data_not_an_exception(self, docker: DockerSandbox) -> None:
        """A red suite must arrive as a non-zero exit the verdict can read, not a traceback."""
        result = await docker.run(("pytest", MARK_EXIT + "3"), cwd=docker.workspace_root)

        assert result.exit_code == 3
        assert not result.ok
        assert result.ran
        assert result.error == ""

    async def test_a_missing_docker_binary_names_the_remedy(
        self, checkout: Path, without_docker_on_path: None
    ) -> None:
        sandbox = DockerSandbox(
            workspace_root=checkout, docker_binary="definitely-not-docker", default_timeout=10.0
        )
        result = await sandbox.run(("pytest",), cwd=checkout)

        assert not result.ran
        assert "not on PATH" in result.error
        # The message has to tell the user what to do, not just that it broke.
        assert "REPOPILOT_EXECUTION_BACKEND=local" in result.error

    async def test_an_unreachable_daemon_is_reported_as_unavailability(
        self, checkout: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead daemon is not a failing test suite, and must not be reported as one."""
        monkeypatch.setenv("FAKE_DOCKER_INFO_RC", "1")
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=10.0,
        )
        result = await sandbox.run(("pytest",), cwd=checkout)

        assert not result.ran
        assert "daemon is not responding" in result.error

    async def test_a_missing_image_is_explained(
        self, docker: DockerSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without this the run would say "the tests failed" when no test ever ran."""
        monkeypatch.setenv("FAKE_DOCKER_ERRMSG", "Unable to find image 'nope:latest' locally")
        result = await docker.run(("pytest", MARK_RUNTIME_ERROR + "125"), cwd=docker.workspace_root)

        assert not result.ran
        assert "not available locally" in result.error
        assert "REPOPILOT_SANDBOX_IMAGE" in result.error

    async def test_an_empty_command_is_refused_without_starting_anything(
        self, docker: DockerSandbox, fake_docker: FakeDocker
    ) -> None:
        result = await docker.run((), cwd=docker.workspace_root)

        assert result.error == "no command to run"
        assert fake_docker.run_calls() == []


# ------------------------------------------------------------------ image preflight


class TestTheImageIsReadyBeforeTheSuiteClockStarts:
    """A cold image must not be charged to the test suite's timeout.

    The bug this covers: ``docker run`` pulls on first use, so a slow pull used to consume the
    suite budget and then report a *test* timeout - blaming the suite for an image download, which
    is both wrong and unactionable.
    """

    async def test_a_present_image_is_not_downloaded_again(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary=str(fake_docker.binary))

        result = await sandbox.run(("pytest", "-q"), cwd=checkout)

        assert result.ok
        assert fake_docker.inspect_calls(), "the image should have been checked"
        assert fake_docker.pull_calls() == [], "an image already on the host must not be re-pulled"

    async def test_a_missing_image_is_pulled_before_the_command(
        self, checkout: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_DOCKER_INSPECT_RC", "1")
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary=str(fake_docker.binary))

        result = await sandbox.run(("pytest", "-q"), cwd=checkout)

        assert result.ok
        assert len(fake_docker.pull_calls()) == 1
        # The pull has to precede the run, or it would happen inside the suite's clock again.
        assert fake_docker.entries()[-1]["subcommand"] == "run"

    async def test_a_failed_pull_is_unavailability_not_a_failing_suite(
        self, checkout: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_DOCKER_INSPECT_RC", "1")
        monkeypatch.setenv("FAKE_DOCKER_PULL_RC", "1")
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary=str(fake_docker.binary))

        result = await sandbox.run(("pytest", "-q"), cwd=checkout)

        assert not result.ran
        assert "could not be pulled" in result.error
        assert "REPOPILOT_SANDBOX_IMAGE" in result.error
        assert fake_docker.run_calls() == [], "no container should be started without an image"

    async def test_the_suite_timeout_is_not_spent_downloading_the_image(
        self, checkout: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pull outlives the suite's own budget, and the suite still gets it in full.

        Both halves are asserted on purpose. Without the wall-clock check this test would pass
        vacuously against the old code, which never pulled at all; the elapsed time proves the
        download really happened, and the reported duration proves it was not charged to the suite.
        """
        monkeypatch.setenv("FAKE_DOCKER_INSPECT_RC", "1")
        # Longer than default_timeout, shorter than startup_timeout: the command must survive it.
        monkeypatch.setenv("FAKE_DOCKER_PULL_SLEEP", "1.5")
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=1.0,
            startup_timeout=120.0,
        )

        began = time.monotonic()
        result = await sandbox.run(("pytest", "-q"), cwd=checkout)
        elapsed = time.monotonic() - began

        assert fake_docker.pull_calls(), "the pull has to have happened for this to mean anything"
        assert elapsed >= 1.5, f"the fake pull did not run ({elapsed:.2f}s)"
        assert not result.timed_out, "the download was charged to the suite's timeout"
        assert result.duration_seconds < 1.0, (
            f"the pull was charged to the suite: reported {result.duration_seconds:.2f}s of work"
        )
        assert result.ok, result.error

    async def test_a_slow_pull_stops_at_the_startup_timeout(
        self, checkout: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_DOCKER_INSPECT_RC", "1")
        monkeypatch.setenv("FAKE_DOCKER_PULL_SLEEP", "5")
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            startup_timeout=0.75,
        )

        result = await sandbox.run(("pytest", "-q"), cwd=checkout)

        assert not result.ran
        assert "could not be made ready" in result.error
        assert fake_docker.run_calls() == []

    async def test_the_image_is_checked_once_per_run_not_once_per_command(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary=str(fake_docker.binary))

        await sandbox.run(("pytest", "-q"), cwd=checkout)
        await sandbox.run(("pytest", "-q"), cwd=checkout)
        await sandbox.run(("pytest", "-q"), cwd=checkout)

        # A suite may run several times in a run, and re-asking the daemon each time is pure
        # latency on the hot path.
        assert len(fake_docker.inspect_calls()) == 1


# ------------------------------------------------------------------ timeouts / cleanup


class TestDockerTimeoutsAndCleanup:
    async def test_a_hung_command_is_killed_and_reported_as_a_timeout(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=1.0,
        )
        result = await sandbox.run(("pytest", MARK_SLEEP + "30"), cwd=checkout)

        assert result.timed_out
        assert not result.ran
        assert not result.ok
        assert "still running after 1s" in result.error

    async def test_a_timed_out_container_is_removed(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        """Killing the client is not enough: the container keeps running unless it is removed."""
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=1.0,
        )
        await sandbox.run(("pytest", MARK_SLEEP + "30"), cwd=checkout)

        removals = fake_docker.remove_calls()
        assert len(removals) == 1
        assert removals[0][:2] == ["rm", "--force"]

        names = fake_docker.container_names()
        assert names and names[0] in removals[0]

    async def test_a_cancelled_run_removes_its_container(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=60.0,
        )
        result = await sandbox.run(
            ("pytest", MARK_SLEEP + "30"), cwd=checkout, is_cancelled=lambda: True
        )

        assert result.cancelled
        assert not result.ran
        assert len(fake_docker.remove_calls()) == 1

    async def test_a_command_that_finishes_normally_removes_nothing_extra(
        self, docker: DockerSandbox, fake_docker: FakeDocker
    ) -> None:
        """``--rm`` handles the normal case, so an extra removal would be noise."""
        await docker.run(("pytest",), cwd=docker.workspace_root)

        assert fake_docker.remove_calls() == []

    async def test_each_command_gets_its_own_container(
        self, docker: DockerSandbox, fake_docker: FakeDocker
    ) -> None:
        """One container per command is what makes cleanup infallible: there is nothing to leak."""
        await docker.run(("pytest",), cwd=docker.workspace_root)
        await docker.run(("pytest",), cwd=docker.workspace_root)

        names = fake_docker.container_names()
        assert len(names) == 2
        assert names[0] != names[1]

    async def test_a_per_command_timeout_overrides_the_default(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(
            workspace_root=checkout,
            docker_binary=str(fake_docker.binary),
            default_timeout=600.0,
        )
        result = await sandbox.run(("pytest", MARK_SLEEP + "30"), cwd=checkout, timeout_seconds=1.0)

        assert result.timed_out


# ------------------------------------------------------------- integration with a run


class TestSandboxInsideARun:
    """The backend has to work through the same service object the steps use."""

    async def test_a_step_can_execute_through_either_backend(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        """The same call works on both, which is the property the engine depends on."""
        local = build_execution_environment(Settings(_env_file=None))
        # The fake binary arrives through the setting, which is the same path production takes for
        # a real one: the engine never learns that it is being pointed somewhere unusual.
        container = build_execution_environment(
            Settings(
                _env_file=None,
                execution_backend="docker",
                sandbox_docker_binary=str(fake_docker.binary),
            ),
            scope=checkout,
        )

        local_result = await local.run(
            ("python", "-c", "print('from the host')"), cwd=checkout, timeout_seconds=60.0
        )
        assert local_result.ok
        assert "from the host" in local_result.stdout

        container_result = await container.run(("pytest", "-q"), cwd=checkout, timeout_seconds=60.0)
        assert container_result.ok
        assert local.name == "local"
        assert container.name == "docker"

    async def test_the_reported_environment_travels_to_the_ui_shape(
        self, checkout: Path, fake_docker: FakeDocker
    ) -> None:
        sandbox = DockerSandbox(workspace_root=checkout, docker_binary=str(fake_docker.binary))
        described = sandbox.describe()

        # The keys the terminal renders. Adding a backend that cannot fill these is a bug, and
        # missing one is a silent blank in the UI.
        assert set(described) == {
            "name",
            "isolation",
            "image",
            "network",
            "filesystem",
            "resources",
        }
        assert described["network"] == "none"
        assert CONTAINER_WORKSPACE in str(described["filesystem"])


@requires_docker()
class TestDockerIsActuallyAvailableHere:
    """Runs only where a container runtime exists. Skipped elsewhere rather than failing."""

    async def test_a_container_can_actually_be_started(self, checkout: Path) -> None:
        (checkout / "hello.txt").write_text("inside the container", encoding="utf-8")
        sandbox = DockerSandbox(workspace_root=checkout, default_timeout=180.0, image="alpine:3.20")

        result = await sandbox.run(("cat", "/workspace/hello.txt"), cwd=checkout)

        assert result.ok, f"stderr: {result.stderr[:500]}"
        assert "inside the container" in result.stdout


class _NoOpStep(Step):
    """A pipeline of one step that does nothing, so only the engine's own wiring is under test."""

    name = "noop"

    async def execute(self, ctx: StepContext) -> StepOutcome:
        await ctx.emit(EventType.STEP_COMPLETED, "nothing to do")
        return StepOutcome.OK


class TestTheEngineAnnouncesItsEnvironment:
    """A run has to say where its commands execute, rather than leaving it to be assumed."""

    async def test_a_local_run_says_it_ran_on_the_host(self, settings: Settings) -> None:
        engine = _engine(settings, steps=(_NoOpStep(),))
        try:
            run = await engine.start(repository="acme/widgets", task="do nothing")
            await engine.wait_for(run.id, seconds=30.0)
            events = await engine.store.get_events(run.id)
        finally:
            await engine.shutdown()

        sandbox_events = [e for e in events if e.type is EventType.SANDBOX_READY]
        assert len(sandbox_events) == 1, "the environment must be announced once per run"

        event = sandbox_events[0]
        assert event.data["name"] == "local"
        # The wording matters as much as the data: a host process must not read like a sandbox.
        assert "host process" in event.message
        assert "no filesystem or network isolation" in event.message

    async def test_a_container_run_says_what_it_mounted(
        self, settings: Settings, fake_docker: FakeDocker
    ) -> None:
        engine = _engine(
            settings,
            steps=(_NoOpStep(),),
            execution_backend="docker",
            docker_binary=str(fake_docker.binary),
        )
        try:
            run = await engine.start(repository="acme/widgets", task="do nothing")
            await engine.wait_for(run.id, seconds=30.0)
            events = await engine.store.get_events(run.id)
        finally:
            await engine.shutdown()

        event = next(e for e in events if e.type is EventType.SANDBOX_READY)
        assert event.data["name"] == "docker"
        assert event.data["image"] == "python:3.12-slim"
        assert event.data["network"] == "none"
        assert CONTAINER_WORKSPACE in str(event.data["filesystem"])
        assert "container" in event.message

    async def test_the_environment_is_announced_before_anything_executes(
        self, settings: Settings
    ) -> None:
        """A reader must be able to learn the boundary before the first command runs, not after."""
        engine = _engine(settings, steps=(_NoOpStep(),))
        try:
            run = await engine.start(repository="acme/widgets", task="do nothing")
            await engine.wait_for(run.id, seconds=30.0)
            events = await engine.store.get_events(run.id)
        finally:
            await engine.shutdown()

        sandbox_index = next(i for i, e in enumerate(events) if e.type is EventType.SANDBOX_READY)
        started_index = next(i for i, e in enumerate(events) if e.type is EventType.RUN_STARTED)
        assert sandbox_index > started_index
        assert sandbox_index < len(events) - 1
