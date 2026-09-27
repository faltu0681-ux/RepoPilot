"""Tests for environment-based configuration."""

from __future__ import annotations

from pathlib import Path

from repopilot.config import Settings


def test_defaults_when_nothing_is_set() -> None:
    settings = Settings(_env_file=None)

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.github_token is None
    assert settings.has_github_token() is False


def test_reads_prefixed_env_var(monkeypatch) -> None:
    monkeypatch.setenv("REPOPILOT_PORT", "9123")
    monkeypatch.setenv("REPOPILOT_HOST", "0.0.0.0")

    settings = Settings(_env_file=None)

    assert settings.port == 9123
    assert settings.host == "0.0.0.0"


def test_cors_origins_accept_comma_separated_string(monkeypatch) -> None:
    monkeypatch.setenv("REPOPILOT_CORS_ORIGINS", "http://a.test, http://b.test")

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_accept_json_array(monkeypatch) -> None:
    monkeypatch.setenv("REPOPILOT_CORS_ORIGINS", '["http://a.test"]')

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://a.test"]


def test_blank_cors_origins_becomes_empty_list(monkeypatch) -> None:
    monkeypatch.setenv("REPOPILOT_CORS_ORIGINS", "")

    settings = Settings(_env_file=None)

    assert settings.cors_origins == []


def test_github_token_is_a_secret() -> None:
    settings = Settings(_env_file=None, github_token="ghp_supersecret")

    assert settings.has_github_token() is True
    assert "ghp_supersecret" not in repr(settings)
    assert "ghp_supersecret" not in settings.model_dump_json()


def test_blank_github_token_counts_as_absent() -> None:
    settings = Settings(_env_file=None, github_token="   ")

    assert settings.has_github_token() is False


def test_invalid_log_level_is_rejected() -> None:
    import pydantic

    try:
        Settings(_env_file=None, log_level="chatty")
    except pydantic.ValidationError as exc:
        assert "invalid log level" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValidationError for an invalid log level")


def test_ensure_workspace_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "workspace"
    settings = Settings(_env_file=None, workspace_dir=target)

    assert not target.exists()
    assert settings.ensure_workspace() == target
    assert target.is_dir()
