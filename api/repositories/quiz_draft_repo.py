"""MongoDB persistence for FastAPI-owned quiz draft processing jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from .base import MongoRepository


class QuizDraftRepository(MongoRepository):
    """Data access layer for quiz drafts generated from uploaded PDFs."""

    COLLECTION = "quiz_drafts"

    async def ensure_indexes(self) -> None:
        await self._db[self.COLLECTION].create_index("draft_id", unique=True)
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("status", 1), ("created_at", -1)]
        )
        await self._db[self.COLLECTION].create_index("expires_at", expireAfterSeconds=0)

    async def create(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        async def _create() -> dict[str, Any]:
            await self._db[self.COLLECTION].insert_one(doc)
            return cast(
                "dict[str, Any]",
                await self._db[self.COLLECTION].find_one(
                    {"draft_id": doc["draft_id"]}, {"_id": 0}
                ),
            )

        return await self._run_or_default("create", None, _create)

    async def load_for_user(self, draft_id: str, user_id: str) -> dict[str, Any] | None:
        async def _load_for_user() -> dict[str, Any] | None:
            return cast(
                "dict[str, Any] | None",
                await self._db[self.COLLECTION].find_one(
                    {"draft_id": draft_id, "user_id": user_id}, {"_id": 0}
                ),
            )

        return await self._run_or_default("load_for_user", None, _load_for_user)

    async def update_for_user(
        self,
        draft_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        async def _update_for_user() -> dict[str, Any] | None:
            update_doc = {**updates, "updated_at": datetime.now(UTC)}
            await self._db[self.COLLECTION].update_one(
                {"draft_id": draft_id, "user_id": user_id},
                {"$set": update_doc},
            )
            return cast(
                "dict[str, Any] | None",
                await self._db[self.COLLECTION].find_one(
                    {"draft_id": draft_id, "user_id": user_id}, {"_id": 0}
                ),
            )

        return await self._run_or_default("update_for_user", None, _update_for_user)

    async def list_recent_for_user(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        async def _list_recent_for_user() -> list[dict[str, Any]]:
            cursor = (
                self._db[self.COLLECTION]
                .find(
                    {
                        "user_id": user_id,
                        "status": {"$nin": ["expired", "submitted", "cancelled"]},
                    },
                    {"_id": 0},
                )
                .sort("created_at", -1)
                .limit(limit)
            )
            return cast("list[dict[str, Any]]", await cursor.to_list(length=limit))

        return await self._run_or_default("list_recent_for_user", [], _list_recent_for_user)

    async def delete_for_user(self, draft_id: str, user_id: str) -> dict[str, Any] | None:
        async def _delete_for_user() -> dict[str, Any] | None:
            existing = cast(
                "dict[str, Any] | None",
                await self._db[self.COLLECTION].find_one(
                    {"draft_id": draft_id, "user_id": user_id}, {"_id": 0}
                ),
            )
            if not existing:
                return None
            await self._db[self.COLLECTION].delete_one({"draft_id": draft_id, "user_id": user_id})
            return existing

        return await self._run_or_default("delete_for_user", None, _delete_for_user)
