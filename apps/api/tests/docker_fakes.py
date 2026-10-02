"""Shared helpers for testing the docker execution backend without Docker installed.

The container backend's security policy lives in the argv it builds, and the argument for testing
it does not need a container runtime: a fake ``docker`` that records what it was asked to do is a
*stronger* assertion than a real container, because it can be asked "what would you have mounted?"
and can be made to fail in ways a working daemon would not.

A fake CLI shim, rather than monkeypatching :mod:`subprocess`, keeps the code under test on its real
path all the way down to process spawn: argv construction, the spawn, stream draining, the poll
loop, the kill, and the cleanup call are all exercised for real.

Scripts are written per test into ``tmp_path`` so nothing is shared between tests, and each fake is
named uniquely so a leftover process cannot be mistaken for this one's.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

#: Behaviour markers the fake understands, embedded in the command it is asked to run so a test can
#: choose the outcome without the test needing to know the image argument's position.
MARK_SLEEP = "FAKE_SLEEP:"
MARK_EXIT = "FAKE_EXIT:"
MARK_RUNTIME_ERROR = "FAKE_RUNTIME_ERROR:"

_FAKE_SOURCE = f"""\
import json
import os
import sys
import time

MARK_SLEEP = {MARK_SLEEP!r}
MARK_EXIT = {MARK_EXIT!r}
MARK_RUNTIME_ERROR = {MARK_RUNTIME_ERROR!r}

argv = sys.argv[1:]


def record(subcommand, args):
    with open(os.environ["FAKE_DOCKER_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"subcommand": subcommand, "argv": args}}) + "\\n")


subcommand = argv[0] if argv else ""
record(subcommand, argv)

if subcommand == "info":
    code = int(os.environ.get("FAKE_DOCKER_INFO_RC", "0"))
    if code:
        sys.stderr.write(
            os.environ.get("FAKE_DOCKER_INFO_ERR", "Cannot connect to the Docker daemon") + "\\n"
        )
        sys.exit(code)
    print("27.0.0-fake")
    sys.exit(0)

# rm/kill are cleanup; they succeed and that is the point.
if subcommand in ("rm", "kill"):
    sys.exit(0)

# The image preflight. Inspect is a local question and succeeds by default; forcing it to fail is
# how a test reaches the pull branch.
if subcommand == "image":
    code = int(os.environ.get("FAKE_DOCKER_INSPECT_RC", "0"))
    if code:
        sys.stderr.write("Error: No such image\\n")
        sys.exit(code)
    sys.exit(0)

if subcommand == "pull":
    pause = float(os.environ.get("FAKE_DOCKER_PULL_SLEEP", "0"))
    if pause:
        time.sleep(pause)
    code = int(os.environ.get("FAKE_DOCKER_PULL_RC", "0"))
    if code:
        sys.stderr.write(
            os.environ.get("FAKE_DOCKER_PULL_ERR", "manifest unknown: manifest unknown")
            + "\\n"
        )
        sys.exit(code)
    sys.stdout.write("pulled\\n")
    sys.exit(0)

if subcommand != "run":
    sys.exit(0)

# The behaviour marker is the last marker anywhere in argv, so a test can put it in the command
# without caring where the image name sits.
markers = [token for token in argv if token.startswith("FAKE_")]
behaviour = markers[-1] if markers else ""

if behaviour.startswith(MARK_SLEEP):
    time.sleep(float(behaviour.split(":", 1)[1]))
    sys.exit(0)

if behaviour.startswith(MARK_RUNTIME_ERROR):
    sys.stderr.write(os.environ.get("FAKE_DOCKER_ERRMSG", "no such image"))
    sys.exit(int(behaviour.split(":", 1)[1]))

if behaviour.startswith(MARK_EXIT):
    sys.stderr.write("the suite failed")
    sys.exit(int(behaviour.split(":", 1)[1]))

sys.stdout.write("ran the command\\n")
sys.exit(0)
"""


class FakeDocker:
    """A stand-in for the docker CLI, plus the log it writes."""

    def __init__(self, root: Path, name: str = "docker") -> None:
        self.root = root
        self.log = root / "docker.log"
        self.script = root / "_fake_docker.py"
        self.script.write_text(_FAKE_SOURCE, encoding="utf-8")

        if os.name == "nt":
            # Windows cannot exec a .py directly under Popen(shell=False), so a .cmd shim stands in
            # for the binary. Confirmed to work under shell=False; a shell string would not.
            self.binary = root / f"{name}.cmd"
            self.binary.write_bytes(
                f'@echo off\r\n"{sys.executable}" "{self.script}" %*\r\n'.encode("ascii")
            )
        else:
            self.binary = root / name
            self.binary.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{self.script}" "$@"\n', encoding="utf-8"
            )
            self.binary.chmod(0o755)

    def entries(self) -> list[dict]:
        """Every call the fake received, oldest first."""
        if not self.log.exists():
            return []
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def run_calls(self) -> list[list[str]]:
        return [e["argv"] for e in self.entries() if e["subcommand"] == "run"]

    def remove_calls(self) -> list[list[str]]:
        return [e["argv"] for e in self.entries() if e["subcommand"] == "rm"]

    def container_names(self) -> list[str]:
        """The ``--name`` values the fake was asked to run containers under."""
        names: list[str] = []
        for call in self.run_calls():
            if "--name" in call:
                names.append(call[call.index("--name") + 1])
        return names

    def pull_calls(self) -> list[list[str]]:
        return [e["argv"] for e in self.entries() if e["subcommand"] == "pull"]

    def inspect_calls(self) -> list[list[str]]:
        return [e["argv"] for e in self.entries() if e["subcommand"] == "image"]


def requires_docker() -> pytest.mark:
    """Skip a test that needs a real container runtime rather than failing where there is none."""
    return pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker is not installed, so the live container path cannot be exercised here",
    )
