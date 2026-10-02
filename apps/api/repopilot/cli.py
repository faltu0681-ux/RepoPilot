"""Command line entry point.

Only one command exists today: ``serve``. It exists because the obvious way to run the API in
development is wrong for this project, and the reason is worth stating.

Two directories under the project hold other people's code: the run workspace
(``./.repopilot-data``), where a run clones a repository, and the repositories directory
(``./workspaces``), where a user drops folders to be worked on. Repositories are mostly ``.py``
files. ``uvicorn --reload`` watches the project and restarts on any ``*.py`` change, so a run
would restart the server out from under itself partway through its own clone: the in-flight run
is cancelled, the event store is thrown away, and the ``plan`` step never runs. It presents as a
mysterious mid-pipeline failure rather than as a reload, which is what makes it expensive to
diagnose.

Passing ``--reload-exclude`` by hand works, but it is a flag every developer has to remember,
and forgetting it silently breaks runs. Deriving the exclusions from the settings that decide
where those directories live makes it correct by default instead of correct-if-you-are-careful.

    repopilot serve                  # reload on, both code directories excluded
    repopilot serve --no-reload      # plain single process
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from repopilot.config import Settings, get_settings

APP_TARGET = "repopilot.main:app"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repopilot", description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Run the API server.")
    serve.add_argument("--host", default=None, help="Bind address. Defaults to the setting.")
    serve.add_argument("--port", type=int, default=None, help="Bind port. Defaults to the setting.")
    serve.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart on source changes. On by default, with the workspace excluded.",
    )
    serve.add_argument(
        "--log-level", default=None, help="uvicorn log level. Defaults to the setting."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        return _serve(args, get_settings())

    return 1  # pragma: no cover - argparse enforces the choices


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    # Created up front, because uvicorn only treats a reload exclusion as a directory to prune
    # from the watch if it already exists when the watcher starts.
    workspace = settings.ensure_workspace().resolve()
    repositories = settings.ensure_repositories().resolve()

    uvicorn.run(
        APP_TARGET,
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=args.log_level or settings.log_level,
        reload=args.reload,
        # Not globs: an existing directory is pruned from the watch outright, so cloned
        # repositories and the user's own source folders cost nothing and cannot trigger a
        # restart. The repositories directory is here for the same reason as the workspace: a
        # user dropping a project into it would otherwise restart the server mid-run.
        reload_excludes=[str(workspace), str(repositories)],
        reload_dirs=[str(Path.cwd())],
    )
    return 0


__all__ = ["APP_TARGET", "build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
