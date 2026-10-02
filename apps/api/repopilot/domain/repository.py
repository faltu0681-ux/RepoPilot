"""A normalised reference to something RepoPilot can work on.

Accepts every form a human is likely to paste and resolves to one canonical value. Two sources
exist:

* ``github`` -- ``owner/name``, an HTTPS URL, or an SSH remote. The default, and unchanged.
* ``local``  -- ``local://<name>``, a folder the user placed in the configured repositories
  directory. See :mod:`repopilot.execution.workspace` for how that name becomes a path.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SHORTHAND = re.compile(r"^(?P<slug>[\w.-]+/[\w.-]+?)(?:\.git)?/?$")
_SSH = re.compile(r"^git@(?P<host>[\w.-]+):(?P<slug>[\w.-]+/[\w.-]+?)(?:\.git)?$")
_LOCAL = re.compile(r"^local://(?P<name>[\w.-]+)/?$")

#: The owner a local folder is filed under. Chosen so a local run's directory reads as
#: ``local__<name>`` on disk and a local run is never mistaken for a GitHub one.
LOCAL_OWNER = "local"


class RepositorySource(StrEnum):
    """Where the repository comes from."""

    GITHUB = "github"
    LOCAL = "local"


class RepositoryRef(BaseModel):
    """An ``owner/name`` pair, plus the URLs needed to clone and link it.

    A local folder is carried in the same shape rather than as a separate type, because
    everything downstream of resolution -- the profile, the plan, the writes, the tests -- works
    from a checkout on disk and does not care where the bytes came from. Keeping one type means
    one `Run.repository` for the store, the events, and the API to agree about.
    """

    model_config = ConfigDict(frozen=True)

    owner: str
    name: str
    source: RepositorySource = RepositorySource.GITHUB

    @field_validator("owner", "name")
    @classmethod
    def _validate_segment(cls, value: str) -> str:
        segment = value.strip()
        if not _SEGMENT.match(segment):
            raise ValueError(f"invalid repository path segment: {value!r}")
        return segment

    @classmethod
    def parse(cls, value: str | RepositoryRef) -> RepositoryRef:
        """Build a ref from ``owner/name``, an HTTPS URL, a git remote, or ``local://name``."""
        if isinstance(value, RepositoryRef):
            return value

        raw = value.strip()
        if not raw:
            raise ValueError("repository reference cannot be empty")

        local = _LOCAL.match(raw)
        if local:
            name = local.group("name")
            # ``.`` and ``..`` match the character class but are the two names that mean "not this
            # directory". Refused here as well as at resolution: a name that can only ever be a
            # traversal is not a repository.
            if name in {".", ".."}:
                raise ValueError(
                    f"invalid local repository name {name!r}; use the folder's own name"
                )
            return cls(owner=LOCAL_OWNER, name=name, source=RepositorySource.LOCAL)

        for pattern in (_SSH, _SHORTHAND):
            match = pattern.match(raw)
            if match:
                return cls._from_slug(match.group("slug"))

        if "://" in raw:
            from urllib.parse import urlparse

            parts = [p for p in urlparse(raw).path.split("/") if p]
            if len(parts) >= 2:
                return cls._from_slug(f"{parts[0]}/{parts[1]}")

        raise ValueError(
            f"could not parse repository reference {value!r}; "
            "expected 'owner/name', 'https://github.com/owner/name', a git remote, "
            "or 'local://<folder>' for a repository in the workspaces directory"
        )

    @classmethod
    def _from_slug(cls, slug: str) -> RepositoryRef:
        owner, _, name = slug.strip("/").partition("/")
        return cls(owner=owner, name=name.removesuffix(".git"))

    @property
    def is_local(self) -> bool:
        return self.source is RepositorySource.LOCAL

    @property
    def slug(self) -> str:
        """The canonical, round-trippable identifier, as shown to the user."""
        if self.is_local:
            return f"local://{self.name}"
        return f"{self.owner}/{self.name}"

    @property
    def workspace_key(self) -> str:
        """A filesystem-safe directory name for this repository.

        ``slug`` cannot be used directly: a local ref contains ``://`` and a Windows directory
        name may not contain a colon. GitHub keeps its historical ``owner__name`` shape so that
        existing workspace directories stay valid.
        """
        return f"{self.owner}__{self.name}"

    @property
    def https_url(self) -> str:
        """The browsable URL, or the workspace path for a local folder."""
        return f"https://github.com/{self.owner}/{self.name}" if not self.is_local else ""

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git" if not self.is_local else ""

    def __str__(self) -> str:
        return self.slug
