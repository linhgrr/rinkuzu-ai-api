"""
repositories/subject_progress_repo.py — MongoDB persistence for subject-level learning history.
"""

from typing import Optional, Dict, Any, List

from loguru import logger


class SubjectProgressRepository:
    """Data access layer for subject-level progress keyed by (user_id, job_id)."""

    COLLECTION = "al_subject_progress"

    def __init__(self, db):
        self._db = db

    async def ensure_indexes(self) -> None:
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("job_id", 1)],
            unique=True,
        )
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("last_session_id", 1)],
        )
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("updated_at", -1)],
        )

    async def save_snapshot(self, job_id: str, user_id: str, doc: dict) -> bool:
        try:
            await self._db[self.COLLECTION].update_one(
                {"job_id": job_id, "user_id": user_id},
                {"$set": doc},
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] save_snapshot error: {e}")
            return False

    async def load_for_user(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            return await self._db[self.COLLECTION].find_one(
                {"job_id": job_id, "user_id": user_id},
                {"_id": 0},
            )
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] load_for_user error: {e}")
            return None

    async def load_by_session_for_user(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self._db[self.COLLECTION].find_one(
                {"last_session_id": session_id, "user_id": user_id},
                {"_id": 0},
            )
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] load_by_session_for_user error: {e}")
            return None

    async def load_many_for_user(self, job_ids: List[str], user_id: str) -> Dict[str, Dict[str, Any]]:
        if not job_ids:
            return {}

        try:
            cursor = self._db[self.COLLECTION].find(
                {"job_id": {"$in": job_ids}, "user_id": user_id},
                {"_id": 0},
            )
            rows = await cursor.to_list(length=len(job_ids))
            return {row["job_id"]: row for row in rows if row.get("job_id")}
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] load_many_for_user error: {e}")
            return {}

    async def list_recent(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query = {}
            if user_id:
                query["user_id"] = user_id
            cursor = self._db[self.COLLECTION].find(
                query,
                {"_id": 0},
            ).sort("updated_at", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] list_recent error: {e}")
            return []

    async def delete_for_user(self, job_id: str, user_id: str) -> int:
        try:
            result = await self._db[self.COLLECTION].delete_one(
                {"job_id": job_id, "user_id": user_id},
            )
            return int(result.deleted_count)
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] delete_for_user error: {e}")
            return 0

    async def delete(self, job_id: str) -> int:
        try:
            result = await self._db[self.COLLECTION].delete_many({"job_id": job_id})
            return int(result.deleted_count)
        except Exception as e:
            logger.error(f"[SubjectProgressRepo] delete error: {e}")
            return 0
