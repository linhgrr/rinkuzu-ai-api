"""
repositories/pipeline_repo.py — MongoDB persistence for pipeline jobs.
"""

from typing import Any, ClassVar, cast

from loguru import logger

from api.core.content_pipeline.domain.jobs import PipelineJob
from api.core.content_pipeline.infrastructure.serializers import pipeline_job_to_document

from .base import MongoRepository


class PipelineRepository(MongoRepository):
    """Data access layer for pipeline jobs in MongoDB."""

    COLLECTION = "al_pipeline_jobs"
    SUMMARY_PROJECTION: ClassVar[dict[str, int]] = {
        "_id": 0,
        "job_id": 1,
        "filename": 1,
        "subject_id": 1,
        "status": 1,
        "page_batch_size": 1,
        "batch_count": 1,
        "failed_batch_count": 1,
        "partial_success": 1,
        "concepts_extracted": 1,
        "concepts_after_merge": 1,
        "relations_verified": 1,
        "completed_at": 1,
    }

    async def ensure_indexes(self) -> None:
        """Create required indexes."""
        await self._db[self.COLLECTION].create_index("job_id", unique=True)
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("status", 1), ("completed_at", -1)]
        )

    async def save(self, job: PipelineJob) -> bool:
        """Persist a pipeline job state to MongoDB."""

        async def _save() -> bool:
            doc = pipeline_job_to_document(job)
            await self._db[self.COLLECTION].update_one(
                {"job_id": job.job_id},
                {"$set": doc},
                upsert=True,
            )
            logger.info("[PipelineRepo] saved job_id={}", job.job_id)
            return True

        save_default: bool = False
        return await self._run_or_default("save", save_default, _save)

    async def load(self, job_id: str) -> dict[str, Any] | None:
        """Load a pipeline job result from MongoDB."""

        async def _load() -> dict[str, Any] | None:
            return cast(
                "dict[str, Any] | None",
                await self._db[self.COLLECTION].find_one({"job_id": job_id}, {"_id": 0}),
            )

        return await self._run_or_default("load", None, _load)

    async def load_for_user(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        """Load a pipeline job only if it belongs to user_id."""

        async def _load_for_user() -> dict[str, Any] | None:
            return cast(
                "dict[str, Any] | None",
                await self._db[self.COLLECTION].find_one(
                    {"job_id": job_id, "user_id": user_id}, {"_id": 0}
                ),
            )

        return await self._run_or_default("load_for_user", None, _load_for_user)

    async def load_many_for_user(
        self,
        job_ids: list[str],
        user_id: str,
        projection: dict[str, int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Load multiple pipeline jobs for a user in one round-trip."""
        if not job_ids:
            return {}

        async def _load_many_for_user() -> dict[str, dict[str, Any]]:
            cursor = self._db[self.COLLECTION].find(
                {"job_id": {"$in": job_ids}, "user_id": user_id},
                projection or {"_id": 0},
            )
            rows = await cursor.to_list(length=len(job_ids))
            return {row["job_id"]: row for row in rows if row.get("job_id")}

        return await self._run_or_default("load_many_for_user", {}, _load_many_for_user)

    async def list_recent(
        self,
        limit: int = 20,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent pipeline jobs."""

        async def _list_recent() -> list[dict[str, Any]]:
            query = {}
            if user_id:
                query["user_id"] = user_id
            if status:
                query["status"] = status
            cursor = (
                self._db[self.COLLECTION]
                .find(
                    query,
                    self.SUMMARY_PROJECTION,
                )
                .sort("completed_at", -1)
                .limit(limit)
            )
            return cast("list[dict[str, Any]]", await cursor.to_list(length=limit))

        return await self._run_or_default("list_recent", [], _list_recent)

    async def delete(self, job_id: str, *, delete_sessions: bool = True) -> dict[str, Any]:
        """Delete a pipeline job and optionally its linked sessions."""

        async def _delete() -> dict[str, Any]:
            job_result = await self._db[self.COLLECTION].delete_one({"job_id": job_id})
            deleted_sessions = 0

            if delete_sessions:
                progress_result = await self._db["al_subject_progress"].delete_many(
                    {"job_id": job_id}
                )
                deleted_sessions = int(progress_result.deleted_count)

            return {
                "deleted_job": int(job_result.deleted_count),
                "deleted_sessions": int(deleted_sessions),
            }

        return await self._run_or_default(
            "delete",
            {"deleted_job": 0, "deleted_sessions": 0},
            _delete,
        )

    async def delete_for_user(
        self,
        job_id: str,
        user_id: str,
        *,
        delete_sessions: bool = True,
    ) -> dict[str, Any]:
        """Delete a pipeline job only if owned by user_id."""

        async def _delete_for_user() -> dict[str, Any]:
            job_result = await self._db[self.COLLECTION].delete_one(
                {"job_id": job_id, "user_id": user_id}
            )
            deleted_sessions = 0

            if delete_sessions and job_result.deleted_count:
                progress_result = await self._db["al_subject_progress"].delete_many(
                    {"job_id": job_id, "user_id": user_id}
                )
                deleted_sessions = int(progress_result.deleted_count)

            return {
                "deleted_job": int(job_result.deleted_count),
                "deleted_sessions": int(deleted_sessions),
            }

        return await self._run_or_default(
            "delete_for_user",
            {"deleted_job": 0, "deleted_sessions": 0},
            _delete_for_user,
        )
