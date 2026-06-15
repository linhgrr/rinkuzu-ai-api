"""
exercise_service.py — Exercise lifecycle business logic.

Handles: concept selection via D3QN, theory generation, exercise generation,
answer submission, and prefetch caching.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from typing import Any, TypeVar, cast
import uuid

from loguru import logger
import numpy as np
from pydantic import TypeAdapter

from api.config import settings
from api.exceptions import ExerciseGenerationError

from .agent import decode_action, select_action
from .answer_eval import evaluate_answer, normalize_text, serialize_answer_for_history
from .exercise_gen import evaluate_short_answer, generate_exercise, generate_theory
from .exercise_types import ExerciseType
from .exercise_types.payloads import ExercisePayload
from .history_formatter import format_exercise_history
from .session import ExerciseRecord

_PAYLOAD_ADAPTER: TypeAdapter[ExercisePayload] = TypeAdapter(ExercisePayload)

BLOOM_LABELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}

# Threshold below which max_steps is considered unset (likely a default placeholder)
_MAX_STEPS_UNSET_THRESHOLD = 50
T = TypeVar("T")


class ExerciseService:
    """Business logic for exercise generation, submission, and prefetching."""

    def __init__(self, session_manager: Any = None) -> None:
        self._session_manager = session_manager
        max_workers = max(1, int(settings.llm_max_workers))
        max_concurrency = settings.llm_max_concurrency or max_workers
        self._request_llm_timeout_sec = max(0.0, float(settings.llm_request_timeout_sec))
        self._prefetch_llm_timeout_sec = (
            self._request_llm_timeout_sec
            if settings.llm_prefetch_timeout_sec is None
            else max(0.0, float(settings.llm_prefetch_timeout_sec))
        )
        self._llm_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm")
        self._llm_semaphore = asyncio.Semaphore(max_concurrency)
        self._exercise_inflight: dict[
            tuple[str, int, int, int, str], asyncio.Task[dict[str, Any] | None]
        ] = {}
        self._theory_inflight: dict[tuple[str, str], asyncio.Task[dict[str, Any] | None]] = {}
        self._scheduled_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _build_id_to_concept_map(session: Any) -> dict[int, str]:
        return {v: k for k, v in session.concept_map.items()}

    @staticmethod
    def _serialize_exercise_for_prompt(exercise: Any) -> dict[str, Any]:
        history_json = format_exercise_history([exercise])
        return cast("dict[str, Any]", json.loads(history_json)[0])

    @staticmethod
    def _round_mastery(value: float) -> float:
        return round(float(value), 2)

    def _build_recommendation_reason(
        self,
        session: Any,
        *,
        concept_idx: int,
        concept_name: str,
        bloom_level: int,
    ) -> dict[str, Any]:
        id_to_concept = self._build_id_to_concept_map(session)
        concept_mastery = session.env.get_concept_mastery()
        current_mastery = (
            float(concept_mastery[concept_idx]) if len(concept_mastery) > concept_idx else 0.0
        )

        satisfied_prereqs = []
        for prereq_idx in session.prereq_graph.get(concept_idx, []):
            prereq_concept_id = id_to_concept.get(prereq_idx, str(prereq_idx))
            prereq_mastery = (
                float(concept_mastery[prereq_idx]) if len(concept_mastery) > prereq_idx else 0.0
            )
            if prereq_mastery >= settings.adaptive_mastery_threshold:
                satisfied_prereqs.append(
                    {
                        "name": session.concept_names.get(prereq_concept_id, prereq_concept_id),
                        "mastery": self._round_mastery(prereq_mastery),
                    }
                )

        satisfied_prereqs.sort(key=lambda item: item["mastery"], reverse=True)

        return {
            "concept_name": concept_name,
            "bloom_level": bloom_level,
            "bloom_label": BLOOM_LABELS.get(bloom_level, "Unknown"),
            "satisfied_prereqs": satisfied_prereqs[:3],
            "current_mastery": self._round_mastery(current_mastery),
            "next_milestone": self._round_mastery(settings.adaptive_mastery_threshold),
        }

    def _get_recent_same_concept_exercises(
        self, session: Any, concept_idx: int
    ) -> list[dict[str, Any]]:
        limit = max(0, int(settings.adaptive_exercise_recent_same_concept_limit))
        if limit == 0:
            return []

        same_concept_history = [
            ex for ex in reversed(session.exercise_history) if ex.concept_idx == concept_idx
        ]
        return [self._serialize_exercise_for_prompt(ex) for ex in same_concept_history[:limit]]

    @staticmethod
    def _recent_examples_fingerprint(recent_same_concept_exercises: list[dict[str, Any]]) -> str:
        if not recent_same_concept_exercises:
            return "none"
        serialized = json.dumps(
            recent_same_concept_exercises,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    async def _run_llm_call(self, func: Any, *args: Any, timeout_sec: float | None = None) -> T:
        loop = asyncio.get_running_loop()
        timeout = self._request_llm_timeout_sec if timeout_sec is None else timeout_sec
        async with self._llm_semaphore:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._llm_executor, func, *args),
                timeout=timeout,
            )
            return cast("T", result)

    async def _generate_exercise_dedup(
        self,
        session: Any,
        concept_idx: int,
        bloom_level: int,
        concept_name: str,
        concept_def: str,
        mastery: float | None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any] | None:
        mastery_bucket = int(max(0.0, min(1.0, mastery or 0.0)) * 10)
        recent_same_concept_exercises = self._get_recent_same_concept_exercises(
            session, concept_idx
        )
        history_fingerprint = self._recent_examples_fingerprint(recent_same_concept_exercises)
        key = (session.session_id, concept_idx, bloom_level, mastery_bucket, history_fingerprint)
        existing = self._exercise_inflight.get(key)
        if existing is not None:
            return await existing

        async def _run() -> dict[str, Any] | None:
            return await self._run_llm_call(
                generate_exercise,
                concept_name,
                concept_def,
                bloom_level,
                mastery,
                recent_same_concept_exercises,
                timeout_sec=timeout_sec,
            )

        task = asyncio.create_task(_run())
        self._exercise_inflight[key] = task
        try:
            return await task
        finally:
            self._exercise_inflight.pop(key, None)

    async def _generate_theory_dedup(
        self,
        session: Any,
        concept_id: str,
        concept_name: str,
        concept_def: str,
        timeout_sec: float | None = None,
    ) -> dict[str, Any] | None:
        key = (session.session_id, concept_id)
        existing = self._theory_inflight.get(key)
        if existing is not None:
            return await existing

        async def _run() -> dict[str, Any] | None:
            return await self._run_llm_call(
                generate_theory,
                concept_name,
                concept_def,
                timeout_sec=timeout_sec,
            )

        task = asyncio.create_task(_run())
        self._theory_inflight[key] = task
        try:
            return await task
        finally:
            self._theory_inflight.pop(key, None)

    async def get_next_concept(self, session: Any) -> dict[str, Any] | None:
        """Use D3QN to select the next concept+bloom level."""
        if session.status != "active":
            return None

        async with session._lock:
            if session.env.max_steps <= _MAX_STEPS_UNSET_THRESHOLD:
                session.env.max_steps = 9999
            env = session.env
            if env.max_steps <= _MAX_STEPS_UNSET_THRESHOLD:
                env.max_steps = 9999
            env_stats = env.get_session_stats()
            current_step = env_stats.get("step", 0)

            if current_step == 0:
                concept_idx = 0
                bloom_level = 1
                action_id = concept_idx * 6 + (bloom_level - 1)
                logger.info("[Exercise] 🎯 STEP 0 — Forcing warm-up: concept_idx=0, bloom=1")
            else:
                masks = env.action_masks()
                action_id = select_action(
                    session.q_net,
                    session.current_obs,
                    masks,
                    session.device,
                    n_concepts=env.n_concepts,
                )
                concept_idx, bloom_level = decode_action(action_id)
                logger.info("[Exercise] 🤖 D3QN selected action_id={}", action_id)

            id_to_concept = self._build_id_to_concept_map(session)
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)

            logger.info(
                "[Exercise] Concept: [{}] {} | Bloom: {} | Step: {}/{}",
                concept_idx,
                concept_name,
                bloom_level,
                current_step + 1,
                env_stats.get("max_steps", "?"),
            )

            session._pending_concept_idx = concept_idx
            session._pending_bloom_level = bloom_level
            session._pending_action = action_id
            session.current_exercise = None

            return {
                "concept_name": concept_name,
                "concept_idx": concept_idx,
                "bloom_level": bloom_level,
                "bloom_label": BLOOM_LABELS.get(bloom_level, "Unknown"),
                "step": env_stats["step"],
                "max_steps": env_stats["max_steps"],
            }

    async def get_theory(self, session: Any) -> dict[str, Any] | None:
        """Generate theory for the pending concept."""
        if not hasattr(session, "_pending_concept_idx"):
            return None

        concept_idx = session._pending_concept_idx
        id_to_concept = self._build_id_to_concept_map(session)
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))

        # Check pre-generated theory cache
        if session.concept_theories.get(concept_id):
            return cast("dict[str, Any]", session.concept_theories[concept_id])

        concept_name = session.concept_names.get(concept_id, concept_id)
        concept_def = session.concept_definitions.get(concept_id, "")

        logger.info("[Exercise] Generating theory for {}...", concept_name)
        theory_data = await self._generate_theory_dedup(
            session=session,
            concept_id=concept_id,
            concept_name=concept_name,
            concept_def=concept_def,
        )

        if not theory_data:
            logger.warning("[Exercise] Theory generation returned empty for {}", concept_name)
            return None

        session.concept_theories[concept_id] = theory_data
        return theory_data

    async def generate_exercise(self, session: Any, background_tasks: Any = None) -> Any:
        """Generate exercise from prefetch cache or LLM."""
        if not hasattr(session, "_pending_concept_idx"):
            return None

        concept_idx = session._pending_concept_idx
        bloom_level = session._pending_bloom_level

        id_to_concept = self._build_id_to_concept_map(session)
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))
        concept_name = session.concept_names.get(concept_id, concept_id)

        # Check prefetch cache
        exercise_data = None
        for branch in ("eager", "correct", "incorrect"):
            cached = session._prefetch_cache.get(branch)
            if (
                cached
                and cached["concept_idx"] == concept_idx
                and cached["bloom_level"] == bloom_level
            ):
                exercise_data = cached["exercise_data"]
                session._prefetch_cache.clear()
                logger.info(
                    "[Exercise] ⚡ Cache HIT ({}) for {} (Bloom {})",
                    branch,
                    concept_name,
                    bloom_level,
                )
                break

        # Cache miss → generate via LLM
        if exercise_data is None:
            session._prefetch_cache.clear()
            concept_def = session.concept_definitions.get(concept_id, "")
            concept_mastery = session.env.get_concept_mastery()
            mastery = (
                float(concept_mastery[concept_idx]) if len(concept_mastery) > concept_idx else None
            )
            logger.info("[Exercise] Generating for {} (Bloom {})...", concept_name, bloom_level)
            try:
                exercise_data = await self._generate_exercise_dedup(
                    session=session,
                    concept_idx=concept_idx,
                    bloom_level=bloom_level,
                    concept_name=concept_name,
                    concept_def=concept_def,
                    mastery=mastery,
                )
            except Exception:
                logger.exception("[Exercise] ✗ Generation failed")
                return None

            if not exercise_data:
                logger.error("[Exercise] ✗ Generation returned None")
                return None

        payload = _PAYLOAD_ADAPTER.validate_python(exercise_data["payload"])
        exercise = ExerciseRecord(
            exercise_id=str(uuid.uuid4())[:8],
            concept_idx=concept_idx,
            concept_name=concept_name,
            bloom_level=bloom_level,
            question=exercise_data["question"],
            payload=payload,
            explanation="",
            explanation_correct=exercise_data.get("explanation_correct", ""),
            explanation_incorrect=exercise_data.get("explanation_incorrect", ""),
            theory=None,
        )
        session.current_exercise = exercise
        session._current_recommendation_reason = self._build_recommendation_reason(
            session,
            concept_idx=concept_idx,
            concept_name=concept_name,
            bloom_level=bloom_level,
        )

        # Fire background prefetch
        try:
            if background_tasks:
                background_tasks.add_task(self._prefetch_next_exercises, session)
            else:
                prefetch_task = asyncio.create_task(self._prefetch_next_exercises(session))
                self._scheduled_tasks.add(prefetch_task)
                prefetch_task.add_done_callback(self._scheduled_tasks.discard)
        except Exception:
            logger.exception("[Prefetch] Failed to schedule")

        return exercise

    @staticmethod
    def _normalize_text(value: str) -> str:
        return normalize_text(value)

    @classmethod
    def _serialize_answer_for_history(cls, exercise: Any, answer: dict[str, Any]) -> Any:
        return serialize_answer_for_history(exercise, answer)

    def _evaluate_answer(self, exercise: Any, answer: dict[str, Any]) -> tuple[bool, str]:
        return evaluate_answer(
            exercise,
            answer,
            short_answer_grader=evaluate_short_answer,
        )

    async def submit_answer(
        self, session: Any, answer: dict[str, Any], _background_tasks: Any = None
    ) -> dict[str, Any] | None:
        """Process user's answer, update environment, return result."""

        async with session._lock:
            if not session.current_exercise:
                return None

            exercise = session.current_exercise
            if exercise.payload.exercise_type == ExerciseType.SHORT_ANSWER:
                is_correct, answer_summary = cast(
                    "tuple[bool, str]",
                    await self._run_llm_call(
                        self._evaluate_answer,
                        exercise,
                        answer,
                        timeout_sec=self._request_llm_timeout_sec,
                    ),
                )
            else:
                is_correct, answer_summary = self._evaluate_answer(exercise, answer)
            verdict = "✓ ĐÚNG" if is_correct else "✗ SAI"

            logger.info(
                "[Exercise] 📝 {} | Type: {} | Answer: {} → {}",
                exercise.concept_name,
                exercise.payload.exercise_type,
                answer_summary,
                verdict,
            )

            # Use pre-generated explanations
            explanation = (
                exercise.explanation_correct if is_correct else exercise.explanation_incorrect
            )

            exercise.explanation = explanation
            from .exercise_types.registry import get_handler

            correct_option = (
                get_handler(exercise.payload.exercise_type)
                .to_response_dict(exercise)
                .get("correct_option", "")
            )
            exercise.user_answer = self._serialize_answer_for_history(exercise, answer)
            exercise.is_correct = is_correct
            session.exercise_history.append(exercise)

            if is_correct:
                session.total_correct += 1
            session.total_answered += 1

            # Step environment
            action_id = getattr(session, "_pending_action", 0)
            obs, _reward, terminated, _truncated, info = session.env.step(
                action_id, human_correct=is_correct
            )
            session.current_obs = obs
            session.current_exercise = None

            if terminated:
                session.status = "completed"

            concept_mastery = session.env.get_concept_mastery()
            mastery_val = float(concept_mastery[exercise.concept_idx])
            avg_mastery = float(np.mean(concept_mastery))

        logger.info(
            "[Exercise] Mastery: {:.3f} | Avg: {:.3f} | Step: {} | Status: {}",
            mastery_val,
            avg_mastery,
            info["step"],
            session.status,
        )

        if self._session_manager:
            subject_saved = False
            try:
                subject_saved = await self._session_manager.persist_subject_progress(session)
            except Exception:
                logger.exception("[Exercise] Subject progress save error")

            if not subject_saved:
                self._session_manager.remove_session(session.session_id)
                raise ExerciseGenerationError("Failed to persist subject progress")

        return {
            "is_correct": is_correct,
            "explanation": explanation,
            "correct_option": correct_option,
            "concept_name": exercise.concept_name,
            "bloom_level": exercise.bloom_level,
            "mastery_after": mastery_val,
            "avg_mastery": avg_mastery,
            "step": info["step"],
            "session_completed": session.status == "completed",
            "stats": {
                "total_correct": session.total_correct,
                "total_answered": session.total_answered,
                "accuracy": session.total_correct / max(session.total_answered, 1),
            },
        }

    async def eager_generate_first_exercise(self, session: Any) -> None:
        """Background: pre-generate the first exercise when a session starts."""
        try:
            env_stats = session.env.get_session_stats()
            current_step = env_stats.get("step", 0)

            if current_step == 0:
                concept_idx = 0
                bloom_level = 1
            else:
                masks = session.env.action_masks()
                action_id = select_action(
                    session.q_net,
                    session.current_obs,
                    masks,
                    session.device,
                    n_concepts=session.env.n_concepts,
                )
                concept_idx, bloom_level = decode_action(action_id)

            id_to_concept = self._build_id_to_concept_map(session)
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)
            concept_def = session.concept_definitions.get(concept_id, "")
            concept_mastery = session.env.get_concept_mastery()
            mastery = (
                float(concept_mastery[concept_idx]) if len(concept_mastery) > concept_idx else None
            )

            logger.info("[Eager] Pre-generating: {} (Bloom {})...", concept_name, bloom_level)

            exercise_data = await self._generate_exercise_dedup(
                session=session,
                concept_idx=concept_idx,
                bloom_level=bloom_level,
                concept_name=concept_name,
                concept_def=concept_def,
                mastery=mastery,
                timeout_sec=self._prefetch_llm_timeout_sec,
            )

            if exercise_data:
                session._prefetch_cache["eager"] = {
                    "concept_idx": concept_idx,
                    "bloom_level": bloom_level,
                    "exercise_data": exercise_data,
                }
                logger.info("[Eager] ✓ Cached: {} (Bloom {})", concept_name, bloom_level)
        except Exception:
            logger.exception("[Eager] ✗ Failed")

    async def _prefetch_next_exercises(self, session: Any) -> None:
        """Background: simulate correct/incorrect paths and pre-generate exercises."""
        if not hasattr(session, "_pending_action"):
            return

        action_id = session._pending_action
        id_to_concept = self._build_id_to_concept_map(session)

        async def _prefetch_branch(*, is_correct: bool, branch: str) -> Any:
            try:
                env_snap = session.env.create_snapshot()
                obs, _, terminated, _, _ = env_snap.step(action_id, human_correct=is_correct)
                if terminated:
                    logger.debug("[Prefetch] {}: session would terminate — skipping", branch)
                    return

                masks = env_snap.action_masks()
                next_action = select_action(
                    session.q_net,
                    obs,
                    masks,
                    session.device,
                    n_concepts=env_snap.n_concepts,
                )
                next_concept_idx, next_bloom = decode_action(next_action)

                next_concept_id = id_to_concept.get(next_concept_idx, str(next_concept_idx))
                next_concept_name = session.concept_names.get(next_concept_id, next_concept_id)
                next_concept_def = session.concept_definitions.get(next_concept_id, "")

                logger.debug(
                    "[Prefetch] {}: predicted → {} (Bloom {})",
                    branch,
                    next_concept_name,
                    next_bloom,
                )
                concept_mastery = session.env.get_concept_mastery()
                mastery = (
                    float(concept_mastery[next_concept_idx])
                    if len(concept_mastery) > next_concept_idx
                    else None
                )

                ex_data = await self._generate_exercise_dedup(
                    session=session,
                    concept_idx=next_concept_idx,
                    bloom_level=next_bloom,
                    concept_name=next_concept_name,
                    concept_def=next_concept_def,
                    mastery=mastery,
                    timeout_sec=self._prefetch_llm_timeout_sec,
                )

                if ex_data:
                    session._prefetch_cache[branch] = {
                        "concept_idx": next_concept_idx,
                        "bloom_level": next_bloom,
                        "exercise_data": ex_data,
                    }
                    logger.info(
                        "[Prefetch] ✓ {} cached: {} (Bloom {})",
                        branch,
                        next_concept_name,
                        next_bloom,
                    )

            except Exception:
                logger.exception("[Prefetch] ✗ {} failed", branch)

        await asyncio.gather(
            _prefetch_branch(is_correct=True, branch="correct"),
            _prefetch_branch(is_correct=False, branch="incorrect"),
        )

    def close(self) -> None:
        """Gracefully release thread-pool resources on app shutdown."""
        self._llm_executor.shutdown(wait=False, cancel_futures=True)
