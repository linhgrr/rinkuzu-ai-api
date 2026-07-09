from types import SimpleNamespace

import numpy as np

from api.domains.learning.exercise_types.payloads import OrderingPayload
from api.domains.learning.subject_progress_snapshot import build_subject_progress_snapshot


def _session_with_one_ordering_exercise():
    rec = SimpleNamespace(
        exercise_id="ex1",
        concept_idx=0,
        concept_name="C",
        bloom_level=4,
        question="Sắp xếp",
        payload=OrderingPayload(correct_order=["a", "b", "c"]),
        explanation="",
        explanation_correct="ok",
        explanation_incorrect="no",
        theory=None,
        user_answer=None,
        is_correct=None,
        timestamp=1.0,
    )
    return SimpleNamespace(
        env=SimpleNamespace(
            get_session_stats=lambda: {"step": 1, "max_steps": 10},
            get_concept_mastery=lambda: np.array([0.5]),
            get_mastery_matrix=lambda: np.zeros((1, 6)),
            get_prereq_ok_mask=lambda threshold=None: np.array([True]),
        ),
        job_id="job1",
        user_id="u1",
        session_id="s1",
        status="active",
        total_correct=0,
        total_answered=1,
        concept_map={"C": 0},
        concept_names={"C": "C"},
        exercise_history=[rec],
        created_at=0.0,
        accessed_at=1.0,
    )


def test_snapshot_entry_carries_nested_canonical_payload():
    snap = build_subject_progress_snapshot(_session_with_one_ordering_exercise())
    entry = snap["exercise_history"][0]
    assert entry["payload"]["exercise_type"] == "ordering"
    assert entry["payload"]["correct_order"] == ["a", "b", "c"]
    for flat in ("items", "statement", "options", "right_items"):
        assert flat not in entry
