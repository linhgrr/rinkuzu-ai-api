"""Mongo-backed job repository adapter for content pipeline jobs."""

from __future__ import annotations

from typing import Any

from ....repositories.pipeline_repo import PipelineRepository
from ..application.ports import JobRepository
from ..domain.jobs import PipelineJob


class MongoJobRepository(JobRepository):
    """Adapter over the existing PipelineRepository."""

    def __init__(self, repository: PipelineRepository):
        self._repository = repository

    @classmethod
    def from_db(cls, db: Any) -> "MongoJobRepository":
        return cls(PipelineRepository(db))

    async def save(self, job: PipelineJob) -> bool:
        return await self._repository.save(job)

    async def load(self, job_id: str) -> dict[str, Any] | None:
        return await self._repository.load(job_id)

    async def load_for_user(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._repository.load_for_user(job_id, user_id)

    async def list_recent(
        self,
        limit: int = 20,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._repository.list_recent(limit=limit, user_id=user_id, status=status)
