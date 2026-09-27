"""GitHub integration.

Only what Phase 1 needs: resolving a repository. Branch pushing and pull-request creation land
here later without changing anything above this package.
"""

from repopilot.github.client import (
    GitHubClient,
    GitHubError,
    RepositoryInfo,
    RepositoryNotFound,
)

__all__ = ["GitHubClient", "GitHubError", "RepositoryInfo", "RepositoryNotFound"]
