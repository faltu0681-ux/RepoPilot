"""Tests for the GitHub client.

Network access is replaced with an ``httpx`` mock transport, so these exercise the real request
building, the real response parsing, and the real error mapping.
"""

from __future__ import annotations

import httpx
import pytest

from repopilot.domain.repository import RepositoryRef
from repopilot.github.client import GitHubClient, GitHubError, RepositoryNotFound

REPO_PAYLOAD = {
    "full_name": "pallets/flask",
    "description": "The Python micro web framework",
    "default_branch": "main",
    "private": False,
    "archived": False,
    "fork": False,
    "language": "Python",
    "stargazers_count": 69_000,
    "open_issues_count": 12,
    "size": 4_096,
    "topics": ["web", "framework"],
    "license": {"spdx_id": "BSD-3-Clause"},
    "pushed_at": "2026-09-01T10:00:00Z",
    "clone_url": "https://github.com/pallets/flask.git",
    "html_url": "https://github.com/pallets/flask",
    # Fields RepoPilot does not use must not break parsing.
    "node_id": "R_kgDOD",
    "watchers_count": 1,
}


def make_client(handler, **kwargs) -> GitHubClient:
    transport = httpx.MockTransport(handler)
    return GitHubClient(transport=transport, **kwargs)


class TestGetRepository:
    async def test_returns_parsed_metadata(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=REPO_PAYLOAD))

        info = await client.get_repository(RepositoryRef.parse("pallets/flask"))
        await client.aclose()

        assert info.full_name == "pallets/flask"
        assert info.default_branch == "main"
        assert info.language == "Python"
        assert info.stargazers_count == 69_000
        assert info.topics == ["web", "framework"]
        assert info.size_kb == 4_096
        assert info.is_actionable is True

    async def test_license_is_flattened_to_an_spdx_id(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=REPO_PAYLOAD))

        info = await client.get_repository(RepositoryRef.parse("pallets/flask"))
        await client.aclose()

        assert info.license == "BSD-3-Clause"

    async def test_requests_the_right_path(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json=REPO_PAYLOAD)

        client = make_client(handler)
        await client.get_repository(RepositoryRef.parse("psf/requests"))
        await client.aclose()

        assert seen == ["/repos/psf/requests"]

    async def test_sends_no_authorization_header_without_a_token(self) -> None:
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return httpx.Response(200, json=REPO_PAYLOAD)

        client = make_client(handler)
        assert client.authenticated is False
        await client.get_repository(RepositoryRef.parse("pallets/flask"))
        await client.aclose()

        assert "authorization" not in seen[0]

    async def test_sends_a_bearer_token_when_configured(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization", ""))
            return httpx.Response(200, json=REPO_PAYLOAD)

        client = make_client(handler, token="ghp_secret")
        assert client.authenticated is True
        await client.get_repository(RepositoryRef.parse("pallets/flask"))
        await client.aclose()

        assert seen == ["Bearer ghp_secret"]

    async def test_missing_fields_fall_back_to_defaults(self) -> None:
        minimal = {"full_name": "a/b", "clone_url": "u", "html_url": "h"}
        client = make_client(lambda request: httpx.Response(200, json=minimal))

        info = await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

        assert info.default_branch == "main"
        assert info.stargazers_count == 0
        assert info.topics == []
        assert info.pushed_at is None

    async def test_archived_repositories_are_not_actionable(self) -> None:
        payload = {**REPO_PAYLOAD, "archived": True}
        client = make_client(lambda request: httpx.Response(200, json=payload))

        info = await client.get_repository(RepositoryRef.parse("pallets/flask"))
        await client.aclose()

        assert info.is_actionable is False


class TestErrors:
    async def test_404_raises_not_found_with_a_hint(self) -> None:
        client = make_client(lambda request: httpx.Response(404, json={}))

        with pytest.raises(RepositoryNotFound) as excinfo:
            await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

        assert "a/b" in str(excinfo.value)
        assert "unauthenticated" in str(excinfo.value)

    async def test_401_mentions_the_token(self) -> None:
        client = make_client(lambda request: httpx.Response(401, json={}))

        with pytest.raises(GitHubError) as excinfo:
            await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

        assert "REPOPILOT_GITHUB_TOKEN" in str(excinfo.value)

    async def test_403_is_mapped(self) -> None:
        client = make_client(lambda request: httpx.Response(403, json={}))

        with pytest.raises(GitHubError):
            await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

    async def test_server_error_is_mapped(self) -> None:
        client = make_client(lambda request: httpx.Response(500, json={}))

        with pytest.raises(GitHubError) as excinfo:
            await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

        assert "500" in str(excinfo.value)

    async def test_transport_failure_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host", request=request)

        client = make_client(handler)

        with pytest.raises(GitHubError) as excinfo:
            await client.get_repository(RepositoryRef.parse("a/b"))
        await client.aclose()

        assert "could not reach GitHub" in str(excinfo.value)
