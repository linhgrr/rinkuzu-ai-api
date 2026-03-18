"""
services/exercise_service.py — Exercise lifecycle business logic.

Handles: concept selection via D3QN, theory generation, exercise generation,
answer submission, and prefetch caching.
"""

import uuid
import asyncio
import os
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from ..core.agent import select_action, decode_action
from ..core.exercise_gen import generate_exercise, generate_theory


BLOOM_LABELS = {
    1: "Remember", 2: "Understand", 3: "Apply",
    4: "Analyze", 5: "Evaluate", 6: "Create",
}


class ExerciseService:
    """Business logic for exercise generation, submission, and prefetching."""

    def __init__(self, session_repo=None):
        self._session_repo = session_repo
        max_workers = int(os.getenv("ADAPTIVE_LLM_MAX_WORKERS", "8"))
        max_concurrency = int(os.getenv("ADAPTIVE_LLM_MAX_CONCURRENCY", str(max_workers)))
        self._llm_timeout_sec = float(os.getenv("ADAPTIVE_LLM_TIMEOUT_SEC", "120"))
        self._llm_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm")
        self._llm_semaphore = asyncio.Semaphore(max_concurrency)
        self._exercise_inflight: Dict[Tuple[str, int, int], asyncio.Future] = {}
        self._theory_inflight: Dict[Tuple[str, str], asyncio.Future] = {}

    @staticmethod
    def _build_id_to_concept_map(session) -> Dict[int, str]:
        return {v: k for k, v in session.concept_map.items()}

    @staticmethod
    def _schedule_background_task(coro, label: str) -> None:
        task = asyncio.create_task(coro)

        def _done_callback(done_task: asyncio.Task):
            try:
                done_task.result()
            except Exception as exc:
                logger.warning(f"[Background] {label} failed: {exc}")

        task.add_done_callback(_done_callback)

    async def _run_llm_call(self, func, *args):
        loop = asyncio.get_running_loop()
        async with self._llm_semaphore:
            return await asyncio.wait_for(
                loop.run_in_executor(self._llm_executor, func, *args),
                timeout=self._llm_timeout_sec,
            )

    async def _generate_exercise_dedup(
        self,
        session,
        concept_idx: int,
        bloom_level: int,
        concept_name: str,
        concept_def: str,
    ) -> Optional[Dict[str, Any]]:
        key = (session.session_id, concept_idx, bloom_level)
        existing = self._exercise_inflight.get(key)
        if existing is not None:
            return await existing

        async def _run() -> Optional[Dict[str, Any]]:
            return await self._run_llm_call(
                generate_exercise,
                concept_name,
                concept_def,
                bloom_level,
            )

        task = asyncio.create_task(_run())
        self._exercise_inflight[key] = task
        try:
            return await task
        finally:
            self._exercise_inflight.pop(key, None)

    async def _generate_theory_dedup(
        self,
        session,
        concept_id: str,
        concept_name: str,
        concept_def: str,
    ) -> Optional[Dict[str, Any]]:
        key = (session.session_id, concept_id)
        existing = self._theory_inflight.get(key)
        if existing is not None:
            return await existing

        async def _run() -> Optional[Dict[str, Any]]:
            return await self._run_llm_call(
                generate_theory,
                concept_name,
                concept_def,
            )

        task = asyncio.create_task(_run())
        self._theory_inflight[key] = task
        try:
            return await task
        finally:
            self._theory_inflight.pop(key, None)

    async def get_next_concept(self, session) -> Optional[Dict[str, Any]]:
        """Use D3QN to select the next concept+bloom level."""
        if session.status != "active":
            return None

        async with session._lock:
            if session.env.max_steps <= 50:
                session.env.max_steps = 9999
            env = session.env
            if env.max_steps <= 50:
                env.max_steps = 9999
            env_stats = env.get_session_stats()
            current_step = env_stats.get("step", 0)

            if current_step == 0:
                concept_idx = 0
                bloom_level = 1
                action_id = concept_idx * 6 + (bloom_level - 1)
                logger.info(f"[Exercise] 🎯 STEP 0 — Forcing warm-up: concept_idx=0, bloom=1")
            else:
                masks = env.action_masks()
                action_id = select_action(
                    session.q_net, session.current_obs, masks, session.device,
                    n_concepts=env.n_concepts,
                )
                concept_idx, bloom_level = decode_action(action_id)
                logger.info(f"[Exercise] 🤖 D3QN selected action_id={action_id}")

            id_to_concept = self._build_id_to_concept_map(session)
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)

            logger.info(f"[Exercise] Concept: [{concept_idx}] {concept_name} | Bloom: {bloom_level} | Step: {current_step + 1}/{env_stats.get('max_steps', '?')}")

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

    async def get_theory(self, session) -> Optional[Dict[str, Any]]:
        """Generate theory for the pending concept."""
        if not hasattr(session, '_pending_concept_idx'):
            return None

        concept_idx = session._pending_concept_idx
        id_to_concept = self._build_id_to_concept_map(session)
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))

        # Check pre-generated theory cache
        if concept_id in session.concept_theories and session.concept_theories[concept_id]:
            return session.concept_theories[concept_id]

        concept_name = session.concept_names.get(concept_id, concept_id)
        concept_def = session.concept_definitions.get(concept_id, "")

        logger.info(f"[Exercise] Generating theory for {concept_name}...")
        theory_data = await self._generate_theory_dedup(
            session=session,
            concept_id=concept_id,
            concept_name=concept_name,
            concept_def=concept_def,
        )

        if not theory_data:
            logger.warning(f"[Exercise] Theory generation returned empty for {concept_name}")
            return None

        session.concept_theories[concept_id] = theory_data
        return theory_data

    async def generate_exercise(self, session):
        """Generate exercise from prefetch cache or LLM."""
        from ..core.session import ExerciseRecord

        if not hasattr(session, '_pending_concept_idx'):
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
            if (cached
                    and cached["concept_idx"] == concept_idx
                    and cached["bloom_level"] == bloom_level):
                exercise_data = cached["exercise_data"]
                session._prefetch_cache.clear()
                logger.info(f"[Exercise] ⚡ Cache HIT ({branch}) for {concept_name} (Bloom {bloom_level})")
                break

        # Cache miss → generate via LLM
        if exercise_data is None:
            session._prefetch_cache.clear()
            concept_def = session.concept_definitions.get(concept_id, "")
            logger.info(f"[Exercise] Generating for {concept_name} (Bloom {bloom_level})...")
            try:
                exercise_data = await self._generate_exercise_dedup(
                    session=session,
                    concept_idx=concept_idx,
                    bloom_level=bloom_level,
                    concept_name=concept_name,
                    concept_def=concept_def,
                )
            except Exception as e:
                logger.error(f"[Exercise] ✗ Generation failed: {e}")
                return None

            if not exercise_data:
                logger.error("[Exercise] ✗ Generation returned None")
                return None

        exercise = ExerciseRecord(
            exercise_id=str(uuid.uuid4())[:8],
            concept_idx=concept_idx,
            concept_name=concept_name,
            bloom_level=bloom_level,
            question=exercise_data["question"],
            options=exercise_data.get("options", {}),
            correct_option=exercise_data.get("correct_option", "A"),
            explanation="",
            explanation_correct=exercise_data.get("explanation_correct", ""),
            explanation_incorrect=exercise_data.get("explanation_incorrect", ""),
            theory=None,
        )
        session.current_exercise = exercise

        # Fire background prefetch
        try:
            self._schedule_background_task(
                self._prefetch_next_exercises(session),
                f"prefetch session={session.session_id}",
            )
        except Exception as e:
            logger.warning(f"[Prefetch] Failed to schedule: {e}")

        return exercise

    async def submit_answer(self, session, answer: str) -> Optional[Dict[str, Any]]:
        """Process user's answer, update environment, return result."""
        if not session.current_exercise:
            return None

        exercise = session.current_exercise
        is_correct = answer.strip().upper() == exercise.correct_option.strip().upper()
        verdict = "✓ ĐÚNG" if is_correct else "✗ SAI"

        logger.info(f"[Exercise] 📝 {exercise.concept_name} | Answer: {answer} → {verdict}")

        # Use pre-generated explanations
        explanation = exercise.explanation_correct if is_correct else exercise.explanation_incorrect

        exercise.explanation = explanation
        exercise.user_answer = answer
        exercise.is_correct = is_correct
        session.exercise_history.append(exercise)

        if is_correct:
            session.total_correct += 1
        session.total_answered += 1

        # Step environment
        action_id = getattr(session, "_pending_action", 0)
        obs, reward, terminated, truncated, info = session.env.step(action_id, human_correct=is_correct)
        session.current_obs = obs
        session.current_exercise = None

        if terminated:
            session.status = "completed"

        # Get updated mastery
        concept_mastery = session.env.get_concept_mastery()
        mastery_val = float(concept_mastery[exercise.concept_idx])
        avg_mastery = float(np.mean(concept_mastery))

        logger.info(f"[Exercise] Mastery: {mastery_val:.3f} | Avg: {avg_mastery:.3f} | Step: {info['step']} | Status: {session.status}")

        # Persist to MongoDB (fire-and-forget)
        if self._session_repo:
            try:
                self._schedule_background_task(
                    self._session_repo.save(session),
                    f"save session={session.session_id}",
                )
            except Exception as e:
                logger.warning(f"[Exercise] MongoDB save schedule error: {e}")

        return {
            "is_correct": is_correct,
            "explanation": explanation,
            "correct_option": exercise.correct_option,
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

    async def eager_generate_first_exercise(self, session) -> None:
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
                    session.q_net, session.current_obs, masks, session.device,
                    n_concepts=session.env.n_concepts,
                )
                concept_idx, bloom_level = decode_action(action_id)

            id_to_concept = self._build_id_to_concept_map(session)
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)
            concept_def = session.concept_definitions.get(concept_id, "")

            logger.info(f"[Eager] Pre-generating: {concept_name} (Bloom {bloom_level})...")

            exercise_data = await self._generate_exercise_dedup(
                session=session,
                concept_idx=concept_idx,
                bloom_level=bloom_level,
                concept_name=concept_name,
                concept_def=concept_def,
            )

            if exercise_data:
                session._prefetch_cache["eager"] = {
                    "concept_idx": concept_idx,
                    "bloom_level": bloom_level,
                    "exercise_data": exercise_data,
                }
                logger.info(f"[Eager] ✓ Cached: {concept_name} (Bloom {bloom_level})")
        except Exception as e:
            logger.error(f"[Eager] ✗ Failed: {e}")

    async def _prefetch_next_exercises(self, session) -> None:
        """Background: simulate correct/incorrect paths and pre-generate exercises."""
        if not hasattr(session, '_pending_action'):
            return

        action_id = session._pending_action
        id_to_concept = self._build_id_to_concept_map(session)

        async def _prefetch_branch(is_correct: bool, branch: str):
            try:
                env_snap = session.env.create_snapshot()
                obs, _, terminated, _, _ = env_snap.step(action_id, human_correct=is_correct)
                if terminated:
                    logger.debug(f"[Prefetch] {branch}: session would terminate — skipping")
                    return

                masks = env_snap.action_masks()
                next_action = select_action(
                    session.q_net, obs, masks, session.device,
                    n_concepts=env_snap.n_concepts,
                )
                next_concept_idx, next_bloom = decode_action(next_action)

                next_concept_id = id_to_concept.get(next_concept_idx, str(next_concept_idx))
                next_concept_name = session.concept_names.get(next_concept_id, next_concept_id)
                next_concept_def = session.concept_definitions.get(next_concept_id, "")

                logger.debug(f"[Prefetch] {branch}: predicted → {next_concept_name} (Bloom {next_bloom})")

                ex_data = await self._generate_exercise_dedup(
                    session=session,
                    concept_idx=next_concept_idx,
                    bloom_level=next_bloom,
                    concept_name=next_concept_name,
                    concept_def=next_concept_def,
                )

                if ex_data:
                    session._prefetch_cache[branch] = {
                        "concept_idx": next_concept_idx,
                        "bloom_level": next_bloom,
                        "exercise_data": ex_data,
                    }
                    logger.info(f"[Prefetch] ✓ {branch} cached: {next_concept_name} (Bloom {next_bloom})")

            except Exception as e:
                logger.error(f"[Prefetch] ✗ {branch} failed: {e}")

        await asyncio.gather(
            _prefetch_branch(True, "correct"),
            _prefetch_branch(False, "incorrect"),
        )

    def close(self) -> None:
        """Gracefully release thread-pool resources on app shutdown."""
        self._llm_executor.shutdown(wait=False, cancel_futures=True)
