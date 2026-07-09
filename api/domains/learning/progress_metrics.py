"""Progress metrics that exclude prerequisite-locked concepts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_BLOOM_APPLY_IDX = 2
_BLOOM_MATRIX_DIMENSIONS = 2


def build_prereq_graph_from_edges(
    prereq_edges: Sequence[Mapping[str, Any]],
    concept_map: Mapping[str, int],
) -> dict[int, list[int]]:
    graph: dict[int, list[int]] = {}
    for edge in prereq_edges:
        src = str(edge.get("source", "")).strip()
        tgt = str(edge.get("target", "")).strip()
        if src in concept_map and tgt in concept_map:
            graph.setdefault(int(concept_map[tgt]), []).append(int(concept_map[src]))
    return graph


def compute_unlocked_mask(
    *,
    concept_count: int,
    bloom_mastery: Sequence[Sequence[float]] | np.ndarray,
    prereq_graph: Mapping[int, Sequence[int]],
    threshold: float,
) -> np.ndarray:
    """Return concepts whose transitive prerequisites are mastered enough."""
    if concept_count <= 0:
        return np.zeros(0, dtype=bool)

    bloom = np.asarray(bloom_mastery, dtype=np.float32)
    if (
        bloom.shape[0] < concept_count
        or bloom.ndim != _BLOOM_MATRIX_DIMENSIONS
        or bloom.shape[1] <= _BLOOM_APPLY_IDX
    ):
        bloom = np.zeros((concept_count, _BLOOM_APPLY_IDX + 1), dtype=np.float32)

    ancestors: dict[int, set[int]] = {}

    def visit(node: int, seen: set[int]) -> set[int]:
        if node in ancestors:
            return ancestors[node]
        parents: set[int] = set()
        for parent in prereq_graph.get(node, []):
            if parent in seen or parent < 0 or parent >= concept_count:
                continue
            parents.add(parent)
            parents.update(visit(parent, seen | {parent}))
        ancestors[node] = parents
        return parents

    mask = np.ones(concept_count, dtype=bool)
    for idx in range(concept_count):
        for prereq_idx in visit(idx, {idx}):
            if float(bloom[prereq_idx, _BLOOM_APPLY_IDX]) < threshold:
                mask[idx] = False
                break
    return mask


def summarize_mastery_progress(
    *,
    concept_mastery: Sequence[float] | np.ndarray,
    unlocked_mask: Sequence[bool] | np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    mastery = np.asarray(concept_mastery, dtype=np.float32)
    unlocked = np.asarray(unlocked_mask, dtype=bool)
    concept_count = int(mastery.shape[0])

    if unlocked.shape[0] != concept_count:
        unlocked = np.ones(concept_count, dtype=bool)

    unlocked_count = int(np.count_nonzero(unlocked))
    locked_count = max(concept_count - unlocked_count, 0)

    if unlocked_count <= 0:
        return {
            "total_concepts": concept_count,
            "unlocked_concepts": 0,
            "locked_concepts": locked_count,
            "mastered_concepts": 0,
            "avg_mastery": 0.0,
            "progress_percent": 0,
        }

    unlocked_mastery = mastery[unlocked]
    mastered_count = int(np.count_nonzero(unlocked_mastery >= threshold))
    return {
        "total_concepts": concept_count,
        "unlocked_concepts": unlocked_count,
        "locked_concepts": locked_count,
        "mastered_concepts": mastered_count,
        "avg_mastery": float(np.mean(unlocked_mastery)),
        "progress_percent": max(0, min(100, round((mastered_count / unlocked_count) * 100))),
    }
