"""Identifier generation.

IDs are prefixed and sortable-by-creation so they are readable in logs and in the terminal UI
without opening a lookup table.
"""

from __future__ import annotations

import uuid

_ALPHABET_LENGTH = 12


def new_id(prefix: str) -> str:
    """Return a short, unique, prefixed identifier such as ``run_9f3a1c7be204``."""
    token = uuid.uuid4().hex[:_ALPHABET_LENGTH]
    return f"{prefix}_{token}"
