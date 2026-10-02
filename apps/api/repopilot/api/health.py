"""Liveness and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from repopilot import __version__
from repopilot.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    github_authenticated: bool


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings | None = None) -> HealthResponse:
    """Report that the process is up and whether GitHub credentials are configured."""
    resolved = settings or get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        github_authenticated=resolved.has_github_token(),
    )


@router.get("/version", response_model=dict[str, Any])
async def version() -> dict[str, Any]:
    """Return the running application version."""
    return {"name": "repopilot", "version": __version__}
