"""Typed application errors.

The API layer maps these onto HTTP status codes in one place, so the domain and execution
layers never need to know about HTTP.
"""

from __future__ import annotations


class RepopilotError(Exception):
    """Base class for every error RepoPilot raises deliberately."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(RepopilotError):
    """A requested resource does not exist."""

    status_code = 404


class ConflictError(RepopilotError):
    """The request conflicts with the current state of the resource."""

    status_code = 409


class InvalidTransitionError(ConflictError):
    """A run was asked to move to a status it cannot reach from its current one."""

    def __init__(self, run_id: str, current: str, requested: str) -> None:
        super().__init__(f"run {run_id} cannot move from '{current}' to '{requested}'")
        self.run_id = run_id
        self.current = current
        self.requested = requested
