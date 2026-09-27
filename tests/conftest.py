"""Shared pytest fixtures.

Every test gets an isolated settings object and a client bound to it. `_env_file=None` keeps a
developer's local `.env` from leaking into test runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repopilot.config import Settings
from repopilot.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_dir=tmp_path / "workspace",
        cors_origins=["http://testserver"],
        github_token=None,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
