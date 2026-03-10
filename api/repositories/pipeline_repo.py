"""
repositories/pipeline_repo.py — MongoDB persistence for pipeline jobs.
"""

import time
from typing import Optional, Dict, Any, List

from loguru import logger


class PipelineRepository:
    """Data access layer for pipeline jobs in MongoDB."""

    COLLECTION = "al_pipeline_jobs"

    def __init__(self, db):
        self._db = db

    async def ensure_indexes(self) -> None:
        """Create required indexes."""
        await self._db[self.COLLECTION].create_index("job_id", unique=True)

    async def save(self, job) -> bool:
        """Persist a completed PipelineJob's result to MongoDB."""
        try:
            doc = {
                "job_id": job.job_id,
                "filename": job.filename,
                "subject_id": job.subject_id,
                "status": job.status.value,
                "total_chunks": job.total_chunks,
                "concepts_extracted": job.concepts_extracted,
                "concepts_after_merge": job.concepts_after_merge,
                "relations_verified": job.relations_verified,
                "graph_stats": job.graph_stats if isinstance(job.graph_stats, dict) else {},
                "result": job.result,
                "created_at": job.created_at,
                "completed_at": job.completed_at or time.time(),
            }
            await self._db[self.COLLECTION].update_one(
                {"job_id": job.job_id},
                {"$set": doc},
                upsert=True,
            )
            logger.info(f"[PipelineRepo] ✓ Job {job.job_id} saved")
            return True
        except Exception as e:
            logger.error(f"[PipelineRepo] save error: {e}")
            return False

    async def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load a pipeline job result from MongoDB."""
        try:
            return await self._db[self.COLLECTION].find_one(
                {"job_id": job_id}, {"_id": 0}
            )
        except Exception as e:
            logger.error(f"[PipelineRepo] load error: {e}")
            return None

    async def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent pipeline jobs."""
        try:
            cursor = self._db[self.COLLECTION].find(
                {},
                {
                    "_id": 0,
                    "job_id": 1,
                    "filename": 1,
                    "subject_id": 1,
                    "status": 1,
                    "concepts_after_merge": 1,
                    "relations_verified": 1,
                    "completed_at": 1,
                },
            ).sort("completed_at", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"[PipelineRepo] list_recent error: {e}")
            return []

    async def delete(self, job_id: str, delete_sessions: bool = True) -> Dict[str, Any]:
        """Delete a pipeline job and optionally its linked sessions."""
        try:
            job_result = await self._db[self.COLLECTION].delete_one({"job_id": job_id})
            deleted_sessions = 0

            if delete_sessions:
                session_result = await self._db["al_sessions"].delete_many({"job_id": job_id})
                deleted_sessions = session_result.deleted_count

            return {
                "deleted_job": int(job_result.deleted_count),
                "deleted_sessions": int(deleted_sessions),
            }
        except Exception as e:
            logger.error(f"[PipelineRepo] delete error: {e}")
            return {"deleted_job": 0, "deleted_sessions": 0, "error": str(e)}
