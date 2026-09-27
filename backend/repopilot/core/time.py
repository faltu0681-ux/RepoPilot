"""Time helpers.

Everything is timezone-aware UTC. Naive datetimes are a persistent source of ordering bugs in
event streams, so `utcnow()` is the only way to get "now" in this codebase.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)
