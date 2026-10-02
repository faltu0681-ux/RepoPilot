"""Working on repositories that already sit on this machine.

Covers the `local://` reference form, name-to-path resolution and its refusals, and the two steps
that have to behave differently for a local folder. The property that matters most is asserted
repeatedly: the user's folder comes out of a run byte-for-byte unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from repopilot.config import Settings
from repopilot.domain.models import EventLevel, EventType
from repopilot.domain.repository import RepositoryRef, RepositorySource
from repopilot.domain.store import RunStore
from repopilot.execution.steps import (
    CloneRepositoryStep,
    InspectRepositoryStep,
    ResolveRepositoryStep,
    RunServices,
    StepContext,
    StepOutcome,
)
from repopilot.execution.workspace import (
    WorkspaceError,
    copy_repository,
    list_local_repositories,
    resolve_local_repository,
)
from repopilot.github.client import GitHubClient

GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not GIT, reason="git is not installed")


def make_project(root: Path, *, git: bool = True) -> Path:
    """A small repository-shaped folder, optionally a real git checkout."""
    project = root / "project-a"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "widgets"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (project / "src" / "core.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    if git:
        run_git(project, "init", "--initial-branch=main")
        run_git(project, "config", "user.email", "t@example.com")
        run_git(project, "config", "user.name", "Tester")
        run_git(project, "add", "--all")
        run_git(project, "commit", "-m", "initial")
    return project


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------------------------------- references


class TestLocalReferences:
    def test_a_local_name_parses(self) -> None:
        ref = RepositoryRef.parse("local://project-a")
        assert ref.source is RepositorySource.LOCAL
        assert ref.is_local
        assert ref.name == "project-a"
        assert ref.owner == "local"
        assert ref.slug == "local://project-a"

    def test_a_local_ref_round_trips_through_its_slug(self) -> None:
        ref = RepositoryRef.parse("local://project-a")
        assert RepositoryRef.parse(str(ref)) == ref
        assert RepositoryRef.parse(ref) is ref

    def test_a_local_name_may_not_be_a_path(self) -> None:
        # Each of these is a name that tries to be a path. None may resolve to one, because the
        # name is the only thing a user controls and it ends up on the filesystem.
        for value in ("local://../secrets", "local://a/b", "local://", "local://.", "local://.."):
            with pytest.raises(ValueError):
                RepositoryRef.parse(value)

    def test_a_github_ref_is_unchanged(self) -> None:
        ref = RepositoryRef.parse("pallets/flask")
        assert ref.source is RepositorySource.GITHUB
        assert not ref.is_local
        assert ref.slug == "pallets/flask"
        assert ref.workspace_key == "pallets__flask"
        assert ref.clone_url == "https://github.com/pallets/flask.git"

    def test_a_github_owner_named_local_is_still_github(self) -> None:
        # The scheme, not the owner, is what decides. Getting this wrong would silently point a
        # run at a folder instead of cloning a repository.
        ref = RepositoryRef.parse("local/widgets")
        assert ref.source is RepositorySource.GITHUB
        assert ref.slug == "local/widgets"

    def test_a_local_ref_has_no_github_urls(self) -> None:
        ref = RepositoryRef.parse("local://project-a")
        assert ref.clone_url == ""
        assert ref.https_url == ""

    def test_a_local_workspace_key_is_usable_as_a_directory_name(self) -> None:
        # The slug contains "://", which is not a legal character in a Windows directory name, so
        # the directory must be derived from workspace_key instead.
        key = RepositoryRef.parse("local://project-a").workspace_key
        assert key == "local__project-a"
        assert not any(character in key for character in ':/\\*?"<>|')


# ------------------------------------------------------------------------ resolution


class TestResolveLocalRepository:
    def test_it_finds_a_folder_in_the_repositories_directory(self, tmp_path: Path) -> None:
        repositories = tmp_path / "workspaces"
        project = make_project(repositories)
        ref = RepositoryRef.parse("local://project-a")

        assert resolve_local_repository(ref, repositories) == project.resolve()

    def test_it_refuses_a_name_that_is_not_there(self, tmp_path: Path) -> None:
        repositories = tmp_path / "workspaces"
        repositories.mkdir()
        with pytest.raises(WorkspaceError, match="no repository folder named"):
            resolve_local_repository(RepositoryRef.parse("local://absent"), repositories)

    def test_it_refuses_a_traversal_that_resolved_outside(self, tmp_path: Path) -> None:
        # A name cannot contain a slash, so this is constructed rather than parsed: it proves the
        # containment check is doing the work rather than the regex alone.
        repositories = tmp_path / "workspaces"
        repositories.mkdir()
        outside = tmp_path / "secret"
        outside.mkdir()
        ref = RepositoryRef(owner="local", name="..")

        with pytest.raises(WorkspaceError):
            resolve_local_repository(ref, repositories)

    def test_it_refuses_a_file(self, tmp_path: Path) -> None:
        repositories = tmp_path / "workspaces"
        repositories.mkdir()
        (repositories / "project-a").write_text("not a directory", encoding="utf-8")
        with pytest.raises(WorkspaceError, match="is a file"):
            resolve_local_repository(RepositoryRef.parse("local://project-a"), repositories)

    def test_it_refuses_a_github_ref(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceError, match="not a local repository"):
            resolve_local_repository(RepositoryRef.parse("pallets/flask"), tmp_path)

    def test_listing_finds_folders_and_skips_the_rest(self, tmp_path: Path) -> None:
        repositories = tmp_path / "workspaces"
        make_project(repositories)
        (repositories / "project-b").mkdir()
        (repositories / "notes.txt").write_text("x", encoding="utf-8")
        (repositories / ".hidden").mkdir()

        found = list_local_repositories(repositories)
        assert [entry["name"] for entry in found] == ["project-a", "project-b"]
        assert found[0]["reference"] == "local://project-a"

    def test_listing_a_directory_that_does_not_exist_is_empty(self, tmp_path: Path) -> None:
        assert list_local_repositories(tmp_path / "absent") == []


# ----------------------------------------------------------------------------- copy


class TestCopyRepository:
    def test_it_copies_the_tree(self, tmp_path: Path) -> None:
        source = make_project(tmp_path / "src", git=False)
        target = tmp_path / "run"

        assert copy_repository(source, target) == 2
        assert (target / "src" / "core.py").read_text(encoding="utf-8") == (
            "def run():\n    return 1\n"
        )

    def test_it_leaves_build_artefacts_behind(self, tmp_path: Path) -> None:
        # Copying node_modules or a .venv would be slow and would then be profiled as source.
        source = make_project(tmp_path / "src", git=False)
        (source / "node_modules" / "left-pad").mkdir(parents=True)
        (source / "node_modules" / "left-pad" / "index.js").write_text("//", encoding="utf-8")
        (source / "__pycache__").mkdir()
        (source / "__pycache__" / "core.pyc").write_bytes(b"\x00")
        target = tmp_path / "run"

        copy_repository(source, target)
        assert not (target / "node_modules").exists()
        assert not (target / "__pycache__").exists()

    def test_it_keeps_the_git_directory(self, tmp_path: Path) -> None:
        # The run needs something to branch from, and the user already has a history.
        source = make_project(tmp_path / "src", git=False)
        run_git(source, "init", "--initial-branch=main")
        run_git(source, "config", "user.email", "t@example.com")
        run_git(source, "config", "user.name", "Tester")
        run_git(source, "add", "--all")
        run_git(source, "commit", "-m", "initial")
        target = tmp_path / "run"

        copy_repository(source, target)
        assert (target / ".git").is_dir()

    def test_it_refuses_to_merge_into_an_existing_directory(self, tmp_path: Path) -> None:
        source = make_project(tmp_path / "src", git=False)
        target = tmp_path / "run"
        target.mkdir()
        (target / "stale.txt").write_text("old", encoding="utf-8")

        with pytest.raises(FileExistsError):
            copy_repository(source, target)


# --------------------------------------------------------------------------- steps


class Recorder:
    """Collects emitted events so a test can assert on what the run reported."""

    def __init__(self) -> None:
        self.events: list[tuple[EventType, str, dict]] = []

    async def __call__(self, event_type, message, **kwargs) -> None:
        self.events.append((event_type, message, kwargs.get("data") or {}))

    def of(self, event_type: EventType) -> list[tuple[str, dict]]:
        return [(message, data) for kind, message, data in self.events if kind is event_type]

    def messages(self) -> list[str]:
        return [message for _, message, _ in self.events]

    def kinds(self) -> list[EventType]:
        return [kind for kind, _, _ in self.events]


async def local_context(
    settings: Settings, *, repository: str = "local://project-a", task: str = "change run()"
) -> tuple[StepContext, Recorder]:
    store = RunStore()
    run = await store.create_run(repository=repository, task=task)
    recorder = Recorder()
    ctx = StepContext(
        run=run,
        store=store,
        services=RunServices(
            github=GitHubClient(),
            workspace=settings.workspace_dir.resolve(),
            repositories_dir=settings.repositories_dir.resolve(),
        ),
        emit=recorder,
    )
    return ctx, recorder


@needs_git
class TestResolveLocalRepositoryStep:
    async def test_it_reads_the_folders_own_git_state(self, settings: Settings) -> None:
        project = make_project(settings.repositories_dir)
        ctx, recorder = await local_context(settings)

        assert await ResolveRepositoryStep().execute(ctx) is StepOutcome.OK

        _, data = recorder.of(EventType.STEP_COMPLETED)[0]
        assert data["reference"] == "local://project-a"
        assert data["path"] == str(project.resolve())
        assert data["branch"] == "main"
        assert data["head"]
        assert ctx.state["local_repository"]["branch"] == "main"

    async def test_it_does_not_call_github(self, settings: Settings) -> None:
        # The client is unauthenticated and would raise on a real call, so a green step here is
        # the proof that no network request was made for a local folder.
        make_project(settings.repositories_dir)
        ctx, recorder = await local_context(settings)

        await ResolveRepositoryStep().execute(ctx)

        assert not any("GitHub" in message for message in recorder.messages())

    async def test_it_says_so_when_the_folder_is_not_a_checkout(self, settings: Settings) -> None:
        make_project(settings.repositories_dir, git=False)
        ctx, recorder = await local_context(settings)

        await ResolveRepositoryStep().execute(ctx)

        assert any(
            "not a git checkout" in message and level is EventLevel.WARN
            for _, message, level in [
                (kind, message, data.get("level")) for kind, message, data in recorder.events
            ]
        ) or any("not a git checkout" in message for message in recorder.messages())

    async def test_it_refuses_a_folder_that_is_not_there(self, settings: Settings) -> None:
        settings.repositories_dir.mkdir(parents=True, exist_ok=True)
        ctx, _ = await local_context(settings)

        with pytest.raises(WorkspaceError, match="no repository folder named"):
            await ResolveRepositoryStep().execute(ctx)


@needs_git
class TestCloneRepositoryStep:
    async def test_it_copies_the_folder_into_the_run_checkout(self, settings: Settings) -> None:
        project = make_project(settings.repositories_dir)
        ctx, recorder = await local_context(settings)

        assert await ResolveRepositoryStep().execute(ctx) is StepOutcome.OK
        assert await CloneRepositoryStep().execute(ctx) is StepOutcome.OK

        root = ctx.local_path()
        assert root.is_dir()
        assert (root / "pyproject.toml").is_file()
        assert (root / "src" / "core.py").is_file()
        _, data = recorder.of(EventType.STEP_COMPLETED)[-1]
        assert data["source"] == str(project.resolve())
        assert data["files"] >= 2

    async def test_the_users_folder_is_never_modified(self, settings: Settings) -> None:
        # The property the whole feature rests on: the plan step rewrites files, the diagnosis
        # step rewrites them again, and the tests create artefacts. All of it has to land on the
        # copy.
        project = make_project(settings.repositories_dir)
        before = snapshot(project)

        ctx, _ = await local_context(settings)
        await ResolveRepositoryStep().execute(ctx)
        await CloneRepositoryStep().execute(ctx)

        # Diverge the copies the way a run would, then confirm the source still matches.
        (ctx.local_path() / "src" / "core.py").write_text("BROKEN\n", encoding="utf-8")
        (ctx.local_path() / "new-file.py").write_text("added\n", encoding="utf-8")

        assert snapshot(project) == before

    async def test_the_copy_is_a_git_repository(self, settings: Settings) -> None:
        make_project(settings.repositories_dir)
        ctx, _ = await local_context(settings)

        await ResolveRepositoryStep().execute(ctx)
        await CloneRepositoryStep().execute(ctx)

        assert (ctx.local_path() / ".git").is_dir()
        assert run_git(ctx.local_path(), "rev-parse", "HEAD")

    async def test_a_plain_folder_gets_a_repository(self, settings: Settings) -> None:
        # Not under version control, but the plan step creates a branch, which needs something to
        # branch from.
        make_project(settings.repositories_dir, git=False)
        ctx, recorder = await local_context(settings)

        await ResolveRepositoryStep().execute(ctx)
        await CloneRepositoryStep().execute(ctx)

        _, data = recorder.of(EventType.STEP_COMPLETED)[-1]
        assert data["initialised_repository"] is True
        assert (ctx.local_path() / ".git").is_dir()
        assert run_git(ctx.local_path(), "rev-parse", "HEAD")

    async def test_two_runs_of_one_folder_do_not_share_it(self, settings: Settings) -> None:
        make_project(settings.repositories_dir)
        first, _ = await local_context(settings)
        second, _ = await local_context(settings)

        for ctx in (first, second):
            await ResolveRepositoryStep().execute(ctx)
            await CloneRepositoryStep().execute(ctx)

        assert first.local_path() != second.local_path()
        assert first.repo_dir() == second.repo_dir()
        (first.local_path() / "src" / "core.py").write_text("MINE\n", encoding="utf-8")
        assert (second.local_path() / "src" / "core.py").read_text(encoding="utf-8") != "MINE\n"

    async def test_the_workspace_key_has_no_scheme_in_it(self, settings: Settings) -> None:
        make_project(settings.repositories_dir)
        ctx, _ = await local_context(settings)

        await CloneRepositoryStep().execute(ctx)

        assert ctx.repo_dir().name == "local__project-a"

    async def test_a_retry_replaces_the_previous_copy(self, settings: Settings) -> None:
        make_project(settings.repositories_dir)
        ctx, _ = await local_context(settings)
        await CloneRepositoryStep().execute(ctx)
        (ctx.local_path() / "stale.txt").write_text("old", encoding="utf-8")

        assert await CloneRepositoryStep().execute(ctx) is StepOutcome.OK

        assert not (ctx.local_path() / "stale.txt").exists()


@needs_git
class TestInspectAfterALocalCopy:
    async def test_the_analysis_pipeline_reads_a_copied_folder(self, settings: Settings) -> None:
        # Everything downstream works from ctx.local_path(), so a local repository has to flow
        # into the existing analysis steps with no changes of their own.
        make_project(settings.repositories_dir)
        ctx, recorder = await local_context(settings)

        assert await ResolveRepositoryStep().execute(ctx) is StepOutcome.OK
        assert await CloneRepositoryStep().execute(ctx) is StepOutcome.OK
        assert await InspectRepositoryStep().execute(ctx) is StepOutcome.OK

        _, data = recorder.of(EventType.STEP_COMPLETED)[-1]
        assert data["file_count"] >= 2
        assert "Python" in data["languages"]


# ---------------------------------------------------------------------------- api


@needs_git
class TestRepositoriesEndpoint:
    def test_it_lists_local_repositories(self, settings: Settings, client) -> None:
        make_project(settings.repositories_dir)
        (settings.repositories_dir / "project-b").mkdir()

        response = client.get("/api/runs/repositories")

        assert response.status_code == 200
        body = response.json()
        assert body["repositories_dir"] == str(settings.repositories_dir)
        assert [entry["reference"] for entry in body["repositories"]] == [
            "local://project-a",
            "local://project-b",
        ]

    def test_it_is_not_shadowed_by_the_run_id_route(self, settings: Settings, client) -> None:
        # /runs/{run_id} would happily accept the literal string "repositories" if this route were
        # declared after it, so the ordering is load-bearing.
        assert client.get("/api/runs/repositories").status_code == 200

    async def test_a_run_can_be_created_for_a_local_folder(self, settings: Settings) -> None:
        make_project(settings.repositories_dir)
        store = RunStore()
        run = await store.create_run(repository="local://project-a", task="do a thing")

        assert run.repository.is_local
        assert run.repository.slug == "local://project-a"
        assert run.repository.workspace_key == "local__project-a"
