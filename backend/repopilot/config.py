"""Environment-based configuration.

Every setting is read from the process environment or a local `.env` file, using the
`REPOPILOT_` prefix. Secrets are held as `SecretStr` so they cannot leak into logs or
`repr()` output. `.env` is gitignored and is never read from a deployed environment.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _split_csv(value: Any) -> Any:
    """Accept `a,b,c` as well as a JSON array for list-valued env vars."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(stripped)
    return [item.strip() for item in stripped.split(",") if item.strip()]


class Settings(BaseSettings):
    """Runtime configuration for the RepoPilot API."""

    model_config = SettingsConfigDict(
        env_prefix="REPOPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API ---
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"

    # --- Frontend ---
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS)
    )

    # --- GitHub ---
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"

    # --- Workspace ---
    workspace_dir: Path = Path(".repopilot-data")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().lower()
        if level not in {"critical", "error", "warning", "info", "debug", "trace"}:
            raise ValueError(f"invalid log level: {value!r}")
        return level

    def ensure_workspace(self) -> Path:
        """Create the workspace directory if needed and return it."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir

    def has_github_token(self) -> bool:
        return bool(self.github_token and self.github_token.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
