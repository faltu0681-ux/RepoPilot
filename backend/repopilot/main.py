"""FastAPI application factory and ASGI entrypoint.

Run with:

    uvicorn repopilot.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from repopilot import __version__
from repopilot.api import health_router
from repopilot.config import Settings, get_settings

logger = logging.getLogger("repopilot")

API_PREFIX = "/api"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=getattr(logging, resolved.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
        resolved.ensure_workspace()
        logger.info(
            "repopilot %s starting (github_token=%s, workspace=%s)",
            __version__,
            "configured" if resolved.has_github_token() else "absent",
            resolved.workspace_dir,
        )
        try:
            yield
        finally:
            logger.info("repopilot shutting down")

    app = FastAPI(
        title="RepoPilot",
        version=__version__,
        description="An autonomous coding agent that works on your GitHub repositories.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=API_PREFIX)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {"name": "repopilot", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
