"""
subject_progress_snapshot.py — Shared helpers for serializing subject progress state.
"""

from typing import Any

import numpy as np


def build_subject_progress_snapshot(session) -> dict[str, Any]:
    env_stats = session.env.get_session_stats()
    concept_mastery = session.env.get_concept_mastery()
    bloom_mastery = session.env.get_mastery_matrix()

    history = [
        {
            "exercise_id": ex.exercise_id,
            "concept_idx": ex.concept_idx,
            "concept_name": ex.concept_name,
            "bloom_level": ex.bloom_level,
            "question": ex.question,
            "sentence": ex.sentence,
            "exercise_type": ex.exercise_type.value
            if hasattr(ex.exercise_type, "value")
            else ex.exercise_type,
            "options": ex.options,
            "statement": ex.statement,
            "hint": ex.hint,
            "items": ex.items,
            "pairs": ex.pairs,
            "right_items": ex.right_items,
            "rubric": ex.rubric,
            "correct_option": ex.correct_option,
            "correct_answer": ex.correct_answer,
            "explanation": ex.explanation,
            "explanation_correct": ex.explanation_correct,
            "explanation_incorrect": ex.explanation_incorrect,
            "theory": ex.theory,
            "user_answer": ex.user_answer,
            "is_correct": ex.is_correct,
            "timestamp": ex.timestamp,
        }
        for ex in session.exercise_history
    ]

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
        "avg_mastery": float(np.mean(concept_mastery)),
        "concept_indices": session.concept_map,
        "concept_mastery": concept_mastery.tolist(),
        "bloom_mastery": bloom_mastery.tolist(),
        "concept_names": session.concept_names,
        "exercise_history": history,
        "created_at": session.created_at,
        "updated_at": session.accessed_at,
    }
