"""
repositories/subject_progress_repo.py — MongoDB persistence for subject-level learning history.
"""

from typing import Optional, Dict, Any, List

from .base import MongoRepository


class SubjectProgressRepository(MongoRepository):
    """Data access layer for subject-level progress keyed by (user_id, job_id)."""

    COLLECTION = "al_subject_progress"

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
        async def _save_snapshot() -> bool:
            await self._db[self.COLLECTION].update_one(
                {"job_id": job_id, "user_id": user_id},
                {"$set": doc},
                upsert=True,
            )
            return True

        return await self._run_or_default("save_snapshot", False, _save_snapshot)

    async def load_for_user(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        async def _load_for_user() -> Optional[Dict[str, Any]]:
            return await self._db[self.COLLECTION].find_one(
                {"job_id": job_id, "user_id": user_id},
                {"_id": 0},
            )

        return await self._run_or_default("load_for_user", None, _load_for_user)

    async def load_by_session_for_user(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        async def _load_by_session_for_user() -> Optional[Dict[str, Any]]:
            return await self._db[self.COLLECTION].find_one(
                {"last_session_id": session_id, "user_id": user_id},
                {"_id": 0},
            )

        return await self._run_or_default(
            "load_by_session_for_user",
            None,
            _load_by_session_for_user,
        )

    async def load_many_for_user(self, job_ids: List[str], user_id: str) -> Dict[str, Dict[str, Any]]:
        if not job_ids:
            return {}

        async def _load_many_for_user() -> Dict[str, Dict[str, Any]]:
            cursor = self._db[self.COLLECTION].find(
                {"job_id": {"$in": job_ids}, "user_id": user_id},
                {"_id": 0},
            )
            rows = await cursor.to_list(length=len(job_ids))
            return {row["job_id"]: row for row in rows if row.get("job_id")}

        return await self._run_or_default("load_many_for_user", {}, _load_many_for_user)

    async def list_recent(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        async def _list_recent() -> List[Dict[str, Any]]:
            query = {}
            if user_id:
                query["user_id"] = user_id
            cursor = self._db[self.COLLECTION].find(
                query,
                {"_id": 0},
            ).sort("updated_at", -1).limit(limit)
            return await cursor.to_list(length=limit)

        return await self._run_or_default("list_recent", [], _list_recent)

    async def delete_for_user(self, job_id: str, user_id: str) -> int:
        async def _delete_for_user() -> int:
            result = await self._db[self.COLLECTION].delete_one(
                {"job_id": job_id, "user_id": user_id},
            )
            return int(result.deleted_count)

        return await self._run_or_default("delete_for_user", 0, _delete_for_user)

    async def delete(self, job_id: str) -> int:
        async def _delete() -> int:
            result = await self._db[self.COLLECTION].delete_many({"job_id": job_id})
            return int(result.deleted_count)

        return await self._run_or_default("delete", 0, _delete)
