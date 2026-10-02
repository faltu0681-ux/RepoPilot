"""Run lifecycle endpoints.

Creating a run hands it to the engine and returns immediately. Everything the engine does is
observable through the run's event stream, which the WebSocket endpoint streams live.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from repopilot.config import Settings
from repopilot.domain.models import ExecutionEvent, Run, RunStatus
from repopilot.domain.repository import RepositoryRef
from repopilot.domain.store import RunStore
from repopilot.execution.engine import RunEngine
from repopilot.execution.workspace import list_local_repositories

router = APIRouter(prefix="/runs", tags=["runs"])


def get_store(request: Request) -> RunStore:
    return request.app.state.store


def get_engine(request: Request) -> RunEngine:
    return request.app.state.engine


def get_settings_from_app(request: Request) -> Settings:
    """The settings this app was built with.

    Not :func:`repopilot.config.get_settings`: that is the process-wide singleton, and a test or
    an embedding application that built the app with its own settings would otherwise find this
    route reporting directories the app is not using.
    """
    return request.app.state.settings


StoreDep = Annotated[RunStore, Depends(get_store)]
EngineDep = Annotated[RunEngine, Depends(get_engine)]
SettingsDep = Annotated[Settings, Depends(get_settings_from_app)]


class CreateRunRequest(BaseModel):
    """What the user asks RepoPilot to do."""

    repository: str = Field(
        description=(
            "What to work on: 'owner/name' or a GitHub URL to clone one, or "
            "'local://<folder>' for a repository already placed in the workspaces directory."
        ),
        examples=["pallets/flask", "local://my-project"],
    )
    task: str = Field(
        description="What the agent should do to the repository.",
        examples=["Add a /health endpoint that reports the git commit sha"],
    )
    branch: str | None = Field(
        default=None,
        description="Branch to work on. Defaults to a per-run 'repopilot/<id>' branch.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("repository")
    @classmethod
    def _valid_repository(cls, value: str) -> str:
        try:
            RepositoryRef.parse(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value.strip()

    @field_validator("task")
    @classmethod
    def _valid_task(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task cannot be empty")
        return value.strip()


class RunListResponse(BaseModel):
    runs: list[Run]
    total: int


class EventListResponse(BaseModel):
    run_id: str
    events: list[ExecutionEvent]
    last_sequence: int


class LocalRepositoriesResponse(BaseModel):
    """The folders available to work on without cloning anything."""

    repositories_dir: str
    repositories: list[dict[str, str]]


@router.post("", response_model=Run, status_code=status.HTTP_201_CREATED)
async def create_run(payload: CreateRunRequest = Body(...), engine: EngineDep = None) -> Run:
    """Create a run and start executing it."""
    return await engine.start(
        repository=payload.repository,
        task=payload.task,
        branch=payload.branch,
        metadata=payload.metadata,
    )


@router.get("", response_model=RunListResponse)
async def list_runs(
    store: StoreDep,
    run_status: Annotated[RunStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> RunListResponse:
    """List runs, newest first."""
    runs = await store.list_runs(status=run_status, limit=limit)
    return RunListResponse(runs=runs, total=await store.count_runs(status=run_status))


@router.get("/repositories", response_model=LocalRepositoriesResponse)
async def list_repositories(settings: SettingsDep) -> LocalRepositoriesResponse:
    """List the local repositories a run can be pointed at with ``local://<name>``.

    Read-only discovery: the point is for the UI to offer what is actually on disk rather than for
    anyone to add or remove folders through the API.
    """
    return LocalRepositoriesResponse(
        repositories_dir=str(settings.repositories_dir),
        repositories=list_local_repositories(settings.repositories_dir),
    )


@router.get("/{run_id}", response_model=Run)
async def get_run(run_id: str, store: StoreDep) -> Run:
    """Fetch one run's current state."""
    return await store.require_run(run_id)


@router.get("/{run_id}/events", response_model=EventListResponse)
async def list_events(
    run_id: str,
    store: StoreDep,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> EventListResponse:
    """Fetch a run's event history, oldest first.

    ``after_sequence`` is the same cursor the WebSocket uses, so a client can fall back to
    polling without changing how it reads the stream.
    """
    events = await store.get_events(run_id, after_sequence=after_sequence, limit=limit)
    return EventListResponse(
        run_id=run_id,
        events=events,
        last_sequence=events[-1].sequence if events else after_sequence,
    )


@router.delete("/{run_id}", response_model=Run)
async def cancel_run(run_id: str, engine: EngineDep) -> Run:
    """Cancel an in-flight run."""
    return await engine.cancel(run_id)
