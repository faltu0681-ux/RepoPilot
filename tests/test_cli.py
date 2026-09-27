"""Tests for the ``repopilot serve`` command.

The behaviour that matters is that the run workspace is excluded from the reloader, because a run
clones a repository full of ``.py`` files into it and would otherwise restart the server out from
under itself. These assert the arguments uvicorn is handed, not uvicorn's own behaviour.
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


class TestWorkspaceExclusion:
    def test_the_workspace_is_excluded_from_the_reloader(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "repopilot-data"

        serve(captured_run, workspace_dir=workspace)

        assert captured_run["reload_excludes"] == [str(workspace.resolve())]

    def test_the_workspace_is_created_before_serving(
        self, captured_run: dict, tmp_path: Path
    ) -> None:
        """uvicorn only prunes a reload exclusion that already exists as a directory."""
        workspace = tmp_path / "not-created-yet"

        serve(captured_run, workspace_dir=workspace)

        assert workspace.is_dir()

    def test_a_relocated_workspace_is_followed(self, captured_run: dict, tmp_path: Path) -> None:
        """The exclusion is derived from the setting, not hardcoded to the default path."""
        elsewhere = tmp_path / "somewhere-else"

        serve(captured_run, workspace_dir=elsewhere)

        assert captured_run["reload_excludes"] == [str(elsewhere.resolve())]

    def test_the_reloader_still_watches_the_project(self, captured_run: dict) -> None:
        serve(captured_run, workspace_dir=Path(".repopilot-data"))

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
