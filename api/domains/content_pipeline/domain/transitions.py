"""Typed outcomes for content-pipeline persistence transitions.

These contracts belong to the domain boundary. Persistence adapters implement
them; application code may depend on them without importing infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CreateJobOutcome(StrEnum):
    """Outcomes for the insert-only job creation boundary."""

    CREATED = "created"
    COLLISION = "collision"


class SaveJobOutcome(StrEnum):
    """Generation-scoped worker save outcomes."""

    APPLIED = "applied"
    CANCEL_REQUESTED = "cancel_requested"
    STALE_GENERATION = "stale_generation"
    ALREADY_TERMINAL = "already_terminal"


class CancelJobOutcome(StrEnum):
    """Owner-scoped cancel transition outcomes."""

    REQUESTED = "requested"
    ALREADY_TERMINAL = "already_terminal"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class CancelJobResult:
    outcome: CancelJobOutcome
    status: str | None = None
    cancel_requested: bool = False


class RetryJobOutcome(StrEnum):
    """Owner-scoped retry transition outcomes."""

    RETRIED = "retried"
    NOT_FOUND = "not_found"
    INVALID_STATE = "invalid_state"
    NOT_RETRYABLE = "not_retryable"
    MAX_RETRIES = "max_retries"
    NO_SOURCE = "no_source"


@dataclass(frozen=True, slots=True)
class RetryJobResult:
    outcome: RetryJobOutcome
    job: dict[str, Any] | None = None


class RetryCompensationOutcome(StrEnum):
    """CAS compensation outcomes for a failed retry scheduling attempt."""

    APPLIED = "applied"
    CANCEL_REQUESTED = "cancel_requested"
    STALE_GENERATION = "stale_generation"
    WORKER_STARTED = "worker_started"
    ALREADY_TERMINAL = "already_terminal"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class RetryCompensationResult:
    outcome: RetryCompensationOutcome
    status: str | None = None
    retry_count: int | None = None
    cancel_requested: bool = False


__all__ = [
    "CancelJobOutcome",
    "CancelJobResult",
    "CreateJobOutcome",
    "RetryCompensationOutcome",
    "RetryCompensationResult",
    "RetryJobOutcome",
    "RetryJobResult",
    "SaveJobOutcome",
]
