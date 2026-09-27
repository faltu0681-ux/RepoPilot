"""Domain layer: pure types and state. Imports nothing from api or execution."""

from repopilot.domain.models import (
    ALLOWED_TRANSITIONS,
    TERMINAL_RUN_STATUSES,
    EventLevel,
    EventType,
    ExecutionEvent,
    Run,
    RunStatus,
)
from repopilot.domain.repository import RepositoryRef
from repopilot.domain.store import RunStore

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_RUN_STATUSES",
    "ExecutionEvent",
    "EventLevel",
    "EventType",
    "RepositoryRef",
    "Run",
    "RunStatus",
    "RunStore",
]
