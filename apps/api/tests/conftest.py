"""Shared pytest fixtures.

Every test gets an isolated settings object and a client bound to it. `_env_file=None` keeps a
developer's local `.env` from leaking into test runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repopilot.config import Settings
from repopilot.main import create_app
from tests.docker_fakes import FakeDocker


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        # Pinned as well as the workspace: `repositories_dir` defaults to ./workspaces relative to
        # the working directory, so without this a test that resolves a local repository would
        # read the real one.
        repositories_dir=tmp_path / "workspaces",
        cors_origins=["http://testserver"],
        github_token=None,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def fake_docker(tmp_path: Path) -> Iterator[FakeDocker]:
    """A stand-in for the docker CLI that records the argv it was asked to run.

    Lives here rather than in the test module because several modules need it, and a fixture only
    counts if pytest can see it: an imported fixture function works, but a fixture defined in a
    helper module does not. The log path travels by environment variable because there is nowhere
    to put it in an argv the code under test controls.
    """
    fake = FakeDocker(tmp_path)
    previous = os.environ.get("FAKE_DOCKER_LOG")
    os.environ["FAKE_DOCKER_LOG"] = str(fake.log)
    try:
        yield fake
    finally:
        if previous is None:
            os.environ.pop("FAKE_DOCKER_LOG", None)
        else:
            os.environ["FAKE_DOCKER_LOG"] = previous


@pytest.fixture
def without_docker_on_path() -> Iterator[None]:
    """Make ``docker`` look absent, so the missing-binary path is testable anywhere."""
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = ""
    try:
        yield
    finally:
        os.environ["PATH"] = original
