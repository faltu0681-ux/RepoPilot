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

    # --- LLM ---
    # Phase 2 reasoning backend. Any OpenAI-compatible endpoint works (OpenAI, Ollama,
    # OpenRouter, LM Studio) as does Anthropic via ``llm_dialect``.
    #
    # Both ``llm_model`` and ``llm_base_url`` default to empty, which means *unconfigured*.
    # That is deliberate: with no default endpoint, an unconfigured install never builds a
    # client and never makes a network call, and a run that needs reasoning fails with an
    # actionable message instead of a confusing provider error.
    llm_api_key: SecretStr | None = None
    llm_base_url: str = ""
    llm_model: str = ""
    llm_dialect: str = "openai"
    llm_timeout: float = Field(default=120.0, gt=0.0, le=1800.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=32768)

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

    @field_validator("llm_dialect")
    @classmethod
    def _validate_llm_dialect(cls, value: str) -> str:
        dialect = value.strip().lower()
        if dialect not in {"openai", "anthropic"}:
            raise ValueError(
                f"unsupported llm_dialect: {value!r} (expected 'openai' or 'anthropic')"
            )
        return dialect

    def ensure_workspace(self) -> Path:
        """Create the workspace directory if needed and return it."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir

    def has_github_token(self) -> bool:
        return bool(self.github_token and self.github_token.get_secret_value().strip())

    def has_llm_api_key(self) -> bool:
        """True when a reasoning backend is fully configured.

        An API key is not required, because local runtimes such as Ollama legitimately run
        without one. A model and a base URL are, because without them there is nothing to call.
        """
        return bool(self.llm_model.strip() and self.llm_base_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
