"""A thin, honest GitHub client.

Just enough to answer "what repository is this?" and to push a branch later. It uses ``httpx``
and raises :class:`GitHubError` rather than leaking transport or HTTP-status details upward.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from repopilot.core.errors import RepopilotError
from repopilot.domain.repository import RepositoryRef

logger = logging.getLogger("repopilot.github")

DEFAULT_API_URL = "https://api.github.com"
DEFAULT_TIMEOUT = 20.0


class GitHubError(RepopilotError):
    """A GitHub request failed."""

    status_code = 502


class RepositoryNotFound(GitHubError):
    """The repository does not exist, or the token cannot see it."""

    status_code = 404


class RepositoryInfo(BaseModel):
    """The subset of GitHub's repository payload RepoPilot actually uses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    full_name: str
    description: str | None = None
    default_branch: str = "main"
    private: bool = False
    archived: bool = False
    fork: bool = False
    language: str | None = None
    stargazers_count: int = 0
    open_issues_count: int = 0
    size_kb: int = Field(default=0, alias="size")
    topics: list[str] = Field(default_factory=list)
    license: str | None = None
    pushed_at: datetime | None = None
    clone_url: str
    html_url: str

    @property
    def is_actionable(self) -> bool:
        """Archived repositories are not worth running against."""
        return not self.archived and bool(self.default_branch)

    @field_validator("license", mode="before")
    @classmethod
    def _flatten_license(cls, value: Any) -> Any:
        """GitHub sends ``{"spdx_id": ..., "name": ...}``; keep only the identifier."""
        if isinstance(value, dict):
            return value.get("spdx_id") or value.get("name")
        return value


class GitHubClient:
    """Minimal read-side GitHub API client."""

    def __init__(
        self,
        *,
        token: str | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._token = token.strip() if token else None
        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers=self._headers(),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repopilot",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @property
    def authenticated(self) -> bool:
        return bool(self._token)

    async def get_repository(self, ref: RepositoryRef) -> RepositoryInfo:
        """Fetch metadata for one repository."""
        path = f"/repos/{ref.owner}/{ref.name}"
        try:
            response = await self._client.get(path)
        except httpx.HTTPError as exc:
            raise GitHubError(f"could not reach GitHub: {exc}") from exc

        if response.status_code == 404:
            raise RepositoryNotFound(
                f"repository {ref.slug} not found"
                + ("" if self.authenticated else " (unauthenticated; private repos will 404)")
            )
        if response.status_code in (401, 403):
            raise GitHubError(
                f"GitHub rejected the request ({response.status_code}); "
                "check REPOPILOT_GITHUB_TOKEN"
            )
        if response.status_code >= 400:
            raise GitHubError(f"GitHub returned {response.status_code} for {ref.slug}")

        payload: dict[str, Any] = response.json()
        info = RepositoryInfo.model_validate(payload)
        logger.info("resolved repository %s (%s)", info.full_name, info.clone_url)
        return info

    async def aclose(self) -> None:
        await self._client.aclose()
