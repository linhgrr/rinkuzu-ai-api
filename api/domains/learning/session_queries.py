"""Read-only session queries: status, knowledge graph, mastery matrix, concept detail.

These are pure presentation helpers over a live SessionState — no lifecycle or
mutable manager state — so they live apart from SessionManager and can be tested
against a plain session object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bloom import BLOOM_LABEL_SEQUENCE
from .progress_metrics import summarize_mastery_progress

if TYPE_CHECKING:
    from .session import SessionState

_BLOOM_LEVELS = 6


def resolve_concept_status(
    mastery: float, *, visited: bool, prereq_ok: bool, threshold: float
) -> str:
    """Status priority: prerequisite lock > mastered > in-progress > available."""
    if not prereq_ok:
        return "locked"
    if mastery >= threshold:
        return "mastered"
    if visited:
        return "in_progress"
    return "available"


def build_session_status(session: SessionState, *, threshold: float) -> dict[str, Any]:
    env_stats = session.env.get_session_stats()
    concept_mastery = session.env.get_concept_mastery()
    unlocked_mask = session.env.get_prereq_ok_mask(threshold=threshold)
    progress = summarize_mastery_progress(
        concept_mastery=concept_mastery,
        unlocked_mask=unlocked_mask,
        threshold=threshold,
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "step": env_stats["step"],
        "max_steps": env_stats["max_steps"],
        "concepts_visited": env_stats["concepts_visited"],
        "total_concepts": progress["total_concepts"],
        "unlocked_concepts": progress["unlocked_concepts"],
        "locked_concepts": progress["locked_concepts"],
        "mastered_concepts": progress["mastered_concepts"],
        "avg_mastery": progress["avg_mastery"],
        "progress_percent": progress["progress_percent"],
        "coverage": env_stats["coverage"],
        "total_correct": session.total_correct,
        "total_answered": session.total_answered,
        "accuracy": session.total_correct / max(session.total_answered, 1),
        "exercises": [
            {
                "exercise_id": ex.exercise_id,
                "concept_name": ex.concept_name,
                "bloom_level": ex.bloom_level,
                "is_correct": ex.is_correct,
            }
            for ex in session.exercise_history
        ],
    }


def build_knowledge_graph(session: SessionState, *, threshold: float) -> dict[str, Any]:
    concept_mastery = session.env.get_concept_mastery()
    id_to_concept = session.id_to_concept
    prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=threshold)

    nodes = []
    for idx in range(len(session.concept_map)):
        cid = id_to_concept.get(idx, str(idx))
        mastery = float(concept_mastery[idx])
        visited = session.env.is_concept_visited(idx)
        nodes.append(
            {
                "id": cid,
                "index": idx,
                "name": session.concept_names.get(cid, cid),
                "mastery": mastery,
                "status": resolve_concept_status(
                    mastery,
                    visited=visited,
                    prereq_ok=bool(prereq_ok_mask[idx]),
                    threshold=threshold,
                ),
                "visited": visited,
            }
        )

    edges = []
    for tgt_idx, src_list in session.prereq_graph.items():
        tgt_id = id_to_concept.get(tgt_idx, str(tgt_idx))
        for src_idx in src_list:
            src_id = id_to_concept.get(src_idx, str(src_idx))
            edges.append({"source": src_id, "target": tgt_id})

    return {"nodes": nodes, "edges": edges}


def build_mastery_matrix(session: SessionState) -> dict[str, Any]:
    bloom_mastery = session.env.get_mastery_matrix()
    id_to_concept = session.id_to_concept

    matrix = [
        {
            "concept_id": id_to_concept.get(idx, str(idx)),
            "concept_name": session.concept_names.get(id_to_concept.get(idx, str(idx)), str(idx)),
            "bloom_levels": [float(bloom_mastery[idx, b]) for b in range(_BLOOM_LEVELS)],
        }
        for idx in range(len(session.concept_map))
    ]
    return {"matrix": matrix, "bloom_labels": list(BLOOM_LABEL_SEQUENCE)}


def build_concept_detail(
    session: SessionState, concept_id: str, *, threshold: float
) -> dict[str, Any] | None:
    if concept_id not in session.concept_map:
        return None

    idx = session.concept_map[concept_id]
    concept_mastery = session.env.get_concept_mastery()
    bloom_mastery = session.env.get_mastery_matrix()
    prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=threshold)
    id_to_concept = session.id_to_concept

    prereqs = [
        {
            "id": id_to_concept.get(p, str(p)),
            "name": session.concept_names.get(id_to_concept.get(p, str(p)), str(p)),
            "mastery": float(concept_mastery[p]),
        }
        for p in session.prereq_graph.get(idx, [])
    ]

    dependents = []
    for tgt_idx, src_list in session.prereq_graph.items():
        if idx in src_list:
            tgt_id = id_to_concept.get(tgt_idx, str(tgt_idx))
            dependents.append(
                {
                    "id": tgt_id,
                    "name": session.concept_names.get(tgt_id, tgt_id),
                    "mastery": float(concept_mastery[tgt_idx]),
                }
            )

    return {
        "id": concept_id,
        "name": session.concept_names.get(concept_id, concept_id),
        "definition": session.concept_definitions.get(concept_id, ""),
        "mastery": float(concept_mastery[idx]),
        "status": resolve_concept_status(
            float(concept_mastery[idx]),
            visited=session.env.is_concept_visited(idx),
            prereq_ok=bool(prereq_ok_mask[idx]),
            threshold=threshold,
        ),
        "bloom_mastery": [float(bloom_mastery[idx, b]) for b in range(_BLOOM_LEVELS)],
        "prerequisites": prereqs,
        "dependents": dependents,
        "visited": session.env.is_concept_visited(idx),
        "visit_count": session.env.get_visit_count(idx),
    }
