"""Ports used by the unified content pipeline application layer."""

from __future__ import annotations

from typing import Any, Protocol

from ..domain.jobs import PipelineJob


class JobRepository(Protocol):
    """Persistence contract for long-running content pipeline jobs."""

    async def save(self, job: PipelineJob) -> bool:
        ...

    async def load(self, job_id: str) -> dict[str, Any] | None:
        ...

    async def load_for_user(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        ...

    async def list_recent(
        self,
        limit: int = 20,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        ...
