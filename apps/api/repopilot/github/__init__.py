"""GitHub integration.

Resolving a repository (Phase 1), and opening the pull request for the change a run made (2D).
Both sides live here so that auth, timeouts, and error mapping are written once: a step that
hand-rolled HTTP for the write would drift from the read path in exactly the places that matter.
"""

from repopilot.github.client import (
    GitHubClient,
    GitHubError,
    NotAuthenticated,
    PullRequestInfo,
    RepositoryInfo,
    RepositoryNotFound,
)

__all__ = [
    "GitHubClient",
    "GitHubError",
    "NotAuthenticated",
    "PullRequestInfo",
    "RepositoryInfo",
    "RepositoryNotFound",
]
