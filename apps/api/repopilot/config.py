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
    # Where each run's isolated checkout goes: <workspace_dir>/<owner>__<repo>/<run_id>
    workspace_dir: Path = Path(".repopilot-data")
    # Where the user drops repositories they want worked on locally, referenced as
    # ``local://<folder>``. Deliberately outside ``workspace_dir``: this directory is an input the
    # user curates, and everything under the run workspace is a disposable copy of it. Keeping
    # them apart means a run can clean up its own copies without ever touching a source folder.
    repositories_dir: Path = Path("workspaces")

    # --- Verification (Phase 2C) ---
    # A test suite belongs to another project and may well be slower than RepoPilot's own
    # requests, so this is its own ceiling rather than a share of llm_timeout. Ten minutes suits a
    # suite that boots a database; the cap exists so a hung suite ends the run instead of
    # holding a worker open forever.
    test_timeout_seconds: float = Field(default=600.0, gt=0.0, le=7200.0)

    # --- Execution environment (V1) ---
    # Where a run's untrusted commands execute: "local" is a child process on this host, "docker"
    # is a throwaway container with the run's checkout mounted. See docs/architecture.md.
    #
    # The default is "local" on purpose. Docker is the better boundary, but making it the default
    # would mean a fresh clone cannot run its first test without Docker installed, and the failure
    # would arrive halfway through a run. Opting in is one environment variable, and the run
    # reports which backend it used either way.
    execution_backend: str = "local"

    # The image is per-stack: one image cannot have every language runtime, and an image missing the
    # suite's runner produces an honest UNVERIFIED run rather than a misleading failure. Pin a tag,
    # not "latest", so a verdict does not change because a tag moved.
    sandbox_image: str = "python:3.12-slim"

    #: "none" by default and it should stay that way. Anything else lets the code under test make
    #: outbound requests, which is a different and larger decision than running its tests.
    sandbox_network: str = "none"

    #: Resource caps, so a runaway suite or a fork bomb ends the run instead of the machine.
    sandbox_memory: str = "2g"
    sandbox_cpus: float = Field(default=2.0, gt=0.0, le=64.0)
    sandbox_pids_limit: int = Field(default=512, ge=1, le=100_000)

    #: A container that has to be pulled before it can run needs its own ceiling, separate from the
    #: suite's: the first command of a run may legitimately take much longer than the tests do.
    sandbox_startup_timeout_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)

    #: Host variables forwarded into the container, when set. Deliberately excludes PATH and
    #: language-tool paths such as JAVA_HOME, which name host locations that do not exist inside a
    #: container; the image supplies its own toolchain.
    sandbox_env_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["LANG", "LC_ALL", "TZ", "CI", "TERM"]
    )

    #: Empty means the image's own user. A non-root uid is better; it is not the default because it
    #: silently breaks images that expect to write to root-owned paths.
    sandbox_user: str = ""

    #: Overridable so the test suite can point at a fake docker CLI and assert on the exact argv.
    sandbox_docker_binary: str = "docker"

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

    @field_validator("execution_backend", mode="before")
    @classmethod
    def _validate_execution_backend(cls, value: str) -> str:
        """Reject an unknown backend at startup rather than at the first command.

        A typo here would otherwise surface as a run that cannot execute anything, which reads like
        a broken repository instead of a configuration mistake.
        """
        backend = value.strip().lower()
        if backend not in {"local", "docker"}:
            raise ValueError(
                f"unsupported execution_backend: {value!r} (expected 'local' or 'docker')"
            )
        return backend

    def ensure_workspace(self) -> Path:
        """Create the workspace directory if needed and return it."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir

    def ensure_repositories(self) -> Path:
        """Create the local repositories directory if needed and return it.

        Created rather than required, so a fresh install has somewhere obvious to put a folder
        and the UI can list what is already there.
        """
        self.repositories_dir.mkdir(parents=True, exist_ok=True)
        return self.repositories_dir

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
