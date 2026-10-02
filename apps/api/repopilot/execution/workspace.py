"""Working on a repository that already sits on this machine.

A user can drop folders into the configured repositories directory (``workspaces/`` by default)
and ask RepoPilot to work on them with ``local://<name>``. This module is the whole of that
feature: name to path, and path to an isolated per-run copy.

Two rules drive the design.

**The user's folder is read-only.** Every run copies its source into its own checkout under the
run workspace and works there, so a plan can rewrite files, a repair can rewrite them again, and
the tests can create artefacts, without the original changing by a single byte. The copy is also
what keeps one run from seeing another's edits, which is the reason per-run isolation exists at
all.

**A name is not a path.** ``local://../../etc`` is a name that fails to parse, and anything that
does parse is a single path segment resolved and then re-checked for containment. The application
source in ``apps/`` is therefore unreachable from a repository reference by construction, not by
convention.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from repopilot.core.errors import RepopilotError
from repopilot.domain.repository import RepositoryRef, RepositorySource

logger = logging.getLogger("repopilot.workspace")

#: Copied out of a local folder rather than into a run. A checkout's own ``.git`` *is* copied,
#: because the run needs a repository it can branch from and there is no reason to reconstruct a
#: history the user already has. These are the artefacts a build leaves behind, which would
#: otherwise be carried around and then profiled as if they were source.
LOCAL_COPY_IGNORE = shutil.ignore_patterns(
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "target",
    ".next",
    ".DS_Store",
    "*.pyc",
)


class WorkspaceError(RepopilotError):
    """A local repository reference did not resolve to a usable folder."""

    status_code = 400


def resolve_local_repository(ref: RepositoryRef, repositories_dir: Path) -> Path:
    """The folder a ``local://`` reference names, or refuse to guess.

    Refuses rather than searching, because a run that silently analysed the wrong folder would
    still produce a plausible-looking plan.
    """
    if ref.source is not RepositorySource.LOCAL:
        raise WorkspaceError(f"{ref.slug} is not a local repository reference")

    root = repositories_dir.expanduser().resolve()
    candidate = (root / ref.name).resolve()

    if not candidate.is_relative_to(root):
        raise WorkspaceError(
            f"local repository '{ref.name}' resolves outside {root}, so it was refused"
        )
    if not candidate.exists():
        raise WorkspaceError(
            f"no repository folder named '{ref.name}' in {root}. Create the folder and drop the "
            "repository in, or use an 'owner/name' reference to clone one from GitHub."
        )
    if not candidate.is_dir():
        raise WorkspaceError(f"{candidate} is a file, not a repository folder")
    return candidate


def list_local_repositories(repositories_dir: Path) -> list[dict[str, str]]:
    """Every folder available to work on, for discovery in the UI.

    Hidden folders and anything starting with a dot are skipped, so a stray editor directory or a
    partially-written copy is not offered as a repository.
    """
    root = repositories_dir.expanduser()
    if not root.is_dir():
        return []

    found: list[dict[str, str]] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        found.append(
            {
                "name": entry.name,
                "reference": f"local://{entry.name}",
                "path": str(entry.resolve()),
            }
        )
    return found


def copy_repository(source: Path, target: Path) -> int:
    """Copy a local repository into this run's checkout. Returns the file count.

    Blocking; the caller runs it in a worker thread. ``dirs_exist_ok`` is false because the target
    is a fresh per-run directory and an existing one means the clone step did not clear it, which
    is a bug worth surfacing rather than merging over.
    """
    logger.debug("copying local repository %s -> %s", source, target)
    shutil.copytree(
        source,
        target,
        ignore=LOCAL_COPY_IGNORE,
        symlinks=False,
        dirs_exist_ok=False,
    )
    return sum(1 for path in target.rglob("*") if path.is_file())


__all__ = [
    "LOCAL_COPY_IGNORE",
    "WorkspaceError",
    "copy_repository",
    "list_local_repositories",
    "resolve_local_repository",
]
