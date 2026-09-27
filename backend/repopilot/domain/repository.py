"""A normalised reference to a GitHub repository.

Accepts every form a human is likely to paste and resolves to one canonical value.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SHORTHAND = re.compile(r"^(?P<slug>[\w.-]+/[\w.-]+?)(?:\.git)?/?$")
_SSH = re.compile(r"^git@(?P<host>[\w.-]+):(?P<slug>[\w.-]+/[\w.-]+?)(?:\.git)?$")


class RepositoryRef(BaseModel):
    """An ``owner/name`` pair, plus the URLs needed to clone and link it."""

    model_config = ConfigDict(frozen=True)

    owner: str
    name: str

    @field_validator("owner", "name")
    @classmethod
    def _validate_segment(cls, value: str) -> str:
        segment = value.strip()
        if not _SEGMENT.match(segment):
            raise ValueError(f"invalid repository path segment: {value!r}")
        return segment

    @classmethod
    def parse(cls, value: str | RepositoryRef) -> RepositoryRef:
        """Build a ref from ``owner/name``, an HTTPS URL, or an SSH remote."""
        if isinstance(value, RepositoryRef):
            return value

        raw = value.strip()
        if not raw:
            raise ValueError("repository reference cannot be empty")

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
            "expected 'owner/name', 'https://github.com/owner/name', or a git remote"
        )

    @classmethod
    def _from_slug(cls, slug: str) -> RepositoryRef:
        owner, _, name = slug.strip("/").partition("/")
        return cls(owner=owner, name=name.removesuffix(".git"))

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def https_url(self) -> str:
        return f"https://github.com/{self.slug}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.slug}.git"

    def __str__(self) -> str:
        return self.slug
