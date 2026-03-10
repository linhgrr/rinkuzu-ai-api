"""
services/exercise_service.py — Exercise lifecycle business logic.

Handles: concept selection via D3QN, theory generation, exercise generation,
answer submission, and prefetch caching.
"""

import uuid
import asyncio
from typing import Optional, Dict, Any
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
        self._llm_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

    async def get_next_concept(self, session) -> Optional[Dict[str, Any]]:
        """Use D3QN to select the next concept+bloom level."""
        if session.status != "active":
            return None

        async with session._lock:
            env = session.env
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

            id_to_concept = {v: k for k, v in session.concept_map.items()}
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
        id_to_concept = {v: k for k, v in session.concept_map.items()}
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))

        # Check pre-generated theory cache
        if concept_id in session.concept_theories and session.concept_theories[concept_id]:
            return session.concept_theories[concept_id]

        concept_name = session.concept_names.get(concept_id, concept_id)
        concept_def = session.concept_definitions.get(concept_id, "")

        logger.info(f"[Exercise] Generating theory for {concept_name}...")
        loop = asyncio.get_event_loop()
        theory_data = await loop.run_in_executor(
            self._llm_executor,
            generate_theory, concept_name, concept_def,
        )

        session.concept_theories[concept_id] = theory_data
        return theory_data

    async def generate_exercise(self, session):
        """Generate exercise from prefetch cache or LLM."""
        from ..core.session import ExerciseRecord

        if not hasattr(session, '_pending_concept_idx'):
            return None

        concept_idx = session._pending_concept_idx
        bloom_level = session._pending_bloom_level

        id_to_concept = {v: k for k, v in session.concept_map.items()}
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
            loop = asyncio.get_event_loop()
            try:
                exercise_data = await loop.run_in_executor(
                    self._llm_executor,
                    generate_exercise, concept_name, concept_def, bloom_level,
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
            asyncio.create_task(self._prefetch_next_exercises(session))
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
                asyncio.create_task(self._session_repo.save(session))
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

            id_to_concept = {v: k for k, v in session.concept_map.items()}
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)
            concept_def = session.concept_definitions.get(concept_id, "")

            logger.info(f"[Eager] Pre-generating: {concept_name} (Bloom {bloom_level})...")

            loop = asyncio.get_event_loop()
            exercise_data = await loop.run_in_executor(
                self._llm_executor,
                generate_exercise, concept_name, concept_def, bloom_level,
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
        id_to_concept = {v: k for k, v in session.concept_map.items()}

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

                loop = asyncio.get_event_loop()
                ex_data = await loop.run_in_executor(
                    self._llm_executor,
                    generate_exercise, next_concept_name, next_concept_def, next_bloom,
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
