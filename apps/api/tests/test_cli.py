"""Tests for the ``repopilot serve`` command.

The behaviour that matters is that the directories holding other people's code are excluded from the
reloader. A run clones a repository full of ``.py`` files into the workspace, and a local run copies
one in from the repositories directory, so either would restart the server out from under itself.
These assert the arguments uvicorn is handed, not uvicorn's own behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.cli import _serve, build_parser
from repopilot.config import Settings


@pytest.fixture
def captured_run(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_run(target: str, **kwargs: object) -> None:
        calls["target"] = target
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


def serve(captured_run: dict, **overrides: object) -> Settings:
    """Run ``serve`` against a settings object built only from ``overrides``."""
    configured = Settings(_env_file=None, **overrides)
    args = build_parser().parse_args(["serve", *(str(o) for o in overrides.pop("argv", []))])
    _serve(args, configured)
    return configured


class TestParser:
    def test_reload_is_on_by_default(self) -> None:
        assert build_parser().parse_args(["serve"]).reload is True

    def test_reload_can_be_turned_off(self) -> None:
        assert build_parser().parse_args(["serve", "--no-reload"]).reload is False

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestCodeDirectoryExclusions:
    def test_the_workspace_is_excluded_from_the_reloader(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "repopilot-data"

        serve(captured_run, workspace_dir=workspace, repositories_dir=tmp_path / "workspaces")

        assert str(workspace.resolve()) in captured_run["reload_excludes"]

    def test_the_repositories_directory_is_excluded_from_the_reloader(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        # Same failure as above, reached the other way round: a user dropping a project into
        # workspaces/ while a run is going would otherwise restart the server mid-run.
        repositories = tmp_path / "workspaces"

        serve(captured_run, workspace_dir=tmp_path / "ws", repositories_dir=repositories)

        assert str(repositories.resolve()) in captured_run["reload_excludes"]

    def test_the_workspace_is_created_before_serving(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        """uvicorn only prunes a reload exclusion that already exists as a directory."""
        workspace = tmp_path / "not-created-yet"

        serve(captured_run, workspace_dir=workspace, repositories_dir=tmp_path / "repos")

        assert workspace.is_dir()

    def test_the_repositories_directory_is_created_before_serving(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        repositories = tmp_path / "not-created-yet"

        serve(captured_run, workspace_dir=tmp_path / "ws", repositories_dir=repositories)

        assert repositories.is_dir()

    def test_a_relocated_workspace_is_followed(self, captured_run: dict, tmp_path: Path) -> None:
        """The exclusions are derived from the settings, not hardcoded to the defaults."""
        elsewhere = tmp_path / "somewhere-else"

        serve(captured_run, workspace_dir=elsewhere, repositories_dir=tmp_path / "repos")

        excludes = captured_run["reload_excludes"]
        assert str(elsewhere.resolve()) in excludes
        assert not any(exclude.endswith(".repopilot-data") for exclude in excludes)

    def test_both_directories_are_excluded(self, captured_run: dict, tmp_path: Path) -> None:
        serve(
            captured_run,
            workspace_dir=tmp_path / "ws",
            repositories_dir=tmp_path / "repos",
        )

        assert len(captured_run["reload_excludes"]) == 2

    def test_the_reloader_still_watches_the_project(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        serve(
            captured_run,
            workspace_dir=tmp_path / "ws",
            repositories_dir=tmp_path / "repos",
        )

        assert captured_run["reload"] is True
        assert captured_run["reload_dirs"] == [str(Path.cwd())]


class TestBindAddress:
    def test_settings_supply_the_defaults(self, captured_run: dict) -> None:
        serve(captured_run, host="0.0.0.0", port=9001, log_level="debug")

        assert captured_run["host"] == "0.0.0.0"
        assert captured_run["port"] == 9001
        assert captured_run["log_level"] == "debug"

    def test_flags_beat_settings(self, captured_run: dict, tmp_path: Path) -> None:
        configured = Settings(_env_file=None, workspace_dir=tmp_path, port=9001)
        args = build_parser().parse_args(["serve", "--port", "8123", "--no-reload"])

        _serve(args, configured)

        assert captured_run["port"] == 8123
        assert captured_run["reload"] is False
