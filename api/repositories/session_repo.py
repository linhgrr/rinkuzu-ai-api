"""
repositories/session_repo.py — MongoDB persistence for adaptive learning sessions.
"""

import time
from typing import Optional, Dict, Any, List

import numpy as np
from loguru import logger


class SessionRepository:
    """Data access layer for adaptive learning sessions in MongoDB."""

    COLLECTION = "al_sessions"

    def __init__(self, db):
        self._db = db

    async def ensure_indexes(self) -> None:
        """Create required indexes."""
        await self._db[self.COLLECTION].create_index("session_id", unique=True)
        await self._db[self.COLLECTION].create_index("job_id")
        await self._db[self.COLLECTION].create_index(
            [("user_id", 1), ("updated_at", -1)]
        )

    async def save(self, session) -> bool:
        """Persist a SessionState snapshot to MongoDB."""
        try:
            env = session.env
            bloom_mastery = env.get_mastery_matrix()
            concept_mastery = env.get_concept_mastery()

            history = [
                {
                    "exercise_id": ex.exercise_id,
                    "concept_idx": ex.concept_idx,
                    "concept_name": ex.concept_name,
                    "bloom_level": ex.bloom_level,
                    "question": ex.question,
                    "options": ex.options,
                    "correct_option": ex.correct_option,
                    "explanation": ex.explanation,
                    "user_answer": ex.user_answer,
                    "is_correct": ex.is_correct,
                    "timestamp": ex.timestamp,
                }
                for ex in session.exercise_history
            ]

            env_stats = env.get_session_stats()

            doc = {
                "session_id": session.session_id,
                "user_id": getattr(session, "user_id", None),
                "job_id": getattr(session, "job_id", None),
                "status": session.status,
                "total_correct": session.total_correct,
                "total_answered": session.total_answered,
                "accuracy": session.total_correct / max(session.total_answered, 1),
                "step": env_stats.get("step", 0),
                "max_steps": env_stats.get("max_steps", 50),
                "avg_mastery": float(np.mean(concept_mastery)),
                "concept_mastery": concept_mastery.tolist(),
                "bloom_mastery": bloom_mastery.tolist(),
                "concept_names": session.concept_names,
                "exercise_history": history,
                "created_at": session.created_at,
                "updated_at": time.time(),
            }

            await self._db[self.COLLECTION].update_one(
                {"session_id": session.session_id},
                {"$set": doc},
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"[SessionRepo] save error: {e}")
            return False

    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a raw session document from MongoDB."""
        try:
            return await self._db[self.COLLECTION].find_one(
                {"session_id": session_id}, {"_id": 0}
            )
        except Exception as e:
            logger.error(f"[SessionRepo] load error: {e}")
            return None

    async def load_for_user(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Load a session only if it belongs to user_id."""
        try:
            return await self._db[self.COLLECTION].find_one(
                {"session_id": session_id, "user_id": user_id}, {"_id": 0}
            )
        except Exception as e:
            logger.error(f"[SessionRepo] load_for_user error: {e}")
            return None

    async def list_recent(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent sessions (summary only)."""
        try:
            query = {}
            if user_id:
                query["user_id"] = user_id
            cursor = self._db[self.COLLECTION].find(
                query,
                {
                    "_id": 0,
                    "session_id": 1,
                    "status": 1,
                    "total_correct": 1,
                    "total_answered": 1,
                    "accuracy": 1,
                    "avg_mastery": 1,
                    "step": 1,
                    "max_steps": 1,
                    "updated_at": 1,
                    "created_at": 1,
                },
            ).sort("updated_at", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"[SessionRepo] list_recent error: {e}")
            return []

    async def find_latest_for_job(
        self,
        job_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent session for a given pipeline job_id."""
        try:
            query: Dict[str, Any] = {"job_id": job_id}
            if user_id:
                query["user_id"] = user_id
            return await self._db[self.COLLECTION].find_one(
                query,
                {"_id": 0},
                sort=[("updated_at", -1)],
            )
        except Exception as e:
            logger.error(f"[SessionRepo] find_latest_for_job error: {e}")
            return None
