"""
subject_progress_snapshot.py — Shared helpers for serializing subject progress state.
"""

from typing import Any

from api.config import get_settings

from .progress_metrics import summarize_mastery_progress

_MASTERY_THRESHOLD = float(get_settings().adaptive_mastery_threshold)


def _serialize_exercise(ex: Any) -> dict[str, Any]:
    return {
        "exercise_id": ex.exercise_id,
        "concept_idx": ex.concept_idx,
        "concept_name": ex.concept_name,
        "bloom_level": ex.bloom_level,
        "question": ex.question,
        "payload": ex.payload.model_dump(mode="json"),
        "explanation": ex.explanation,
        "explanation_correct": ex.explanation_correct,
        "explanation_incorrect": ex.explanation_incorrect,
        "theory": ex.theory,
        "user_answer": ex.user_answer,
        "is_correct": ex.is_correct,
        "timestamp": ex.timestamp,
    }


def build_subject_progress_snapshot(session: Any) -> dict[str, Any]:
    env_stats = session.env.get_session_stats()
    concept_mastery = session.env.get_concept_mastery()
    bloom_mastery = session.env.get_mastery_matrix()
    unlocked_mask = session.env.get_prereq_ok_mask(threshold=_MASTERY_THRESHOLD)
    progress_metrics = summarize_mastery_progress(
        concept_mastery=concept_mastery,
        unlocked_mask=unlocked_mask,
        threshold=_MASTERY_THRESHOLD,
    )

    history = [_serialize_exercise(ex) for ex in session.exercise_history]

    return {
        "job_id": session.job_id,
        "user_id": session.user_id,
        "last_session_id": session.session_id,
        "status": session.status,
        "total_correct": session.total_correct,
        "total_answered": session.total_answered,
        "accuracy": session.total_correct / max(session.total_answered, 1),
        "step": env_stats.get("step", 0),
        "max_steps": env_stats.get("max_steps", 9999),
        "avg_mastery": progress_metrics["avg_mastery"],
        "unlocked_concepts": progress_metrics["unlocked_concepts"],
        "locked_concepts": progress_metrics["locked_concepts"],
        "mastered_concepts": progress_metrics["mastered_concepts"],
        "progress_percent": progress_metrics["progress_percent"],
        "concept_indices": session.concept_map,
        "concept_mastery": concept_mastery.tolist(),
        "bloom_mastery": bloom_mastery.tolist(),
        "concept_names": session.concept_names,
        "exercise_history": history,
        "current_exercise": (
            _serialize_exercise(current_exercise)
            if (current_exercise := getattr(session, "current_exercise", None))
            else None
        ),
        "pending_concept_idx": getattr(session, "_pending_concept_idx", None),
        "pending_bloom_level": getattr(session, "_pending_bloom_level", None),
        "pending_action": getattr(session, "_pending_action", None),
        "recommendation_reason": getattr(session, "_current_recommendation_reason", None),
        "submission_receipts": dict(getattr(session, "submission_receipts", {})),
        "version": getattr(session, "version", 0),
        "created_at": session.created_at,
        "updated_at": session.accessed_at,
    }
