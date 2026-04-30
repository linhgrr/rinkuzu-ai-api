"""Embedding-based concept deduplication (fixed & schema-compliant)."""

from collections import Counter
import warnings

from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from api.config import settings
from api.core.content_pipeline.infrastructure.llm.schemas import Concept, Relation

_EMB_MATRIX_NDIM = 2


def deduplicate_by_embedding(
    concepts: list[Concept],
    similarity_threshold: float | None = None,
) -> list[Concept]:
    """
    Deduplicate concepts using embedding similarity with connected components.

    .. deprecated::
        This function is deprecated. Use LLM-based verification with 'same_concept'
        direction instead for more accurate concept merging. The LLM can better
        understand semantic equivalence and context than pure embedding similarity.

    - Build similarity graph where sim(i, j) >= threshold.
    - Find connected components (union-find).
    - Merge each component into a canonical concept.
    - Remap relations across ALL concepts to the canonical ids.

    Uses name_embedding for similarity matching.

    Args:
        concepts: List of concepts to deduplicate
        similarity_threshold: Threshold for cosine similarity (0 < threshold <= 1)

    Returns:
        Deduplicated list of concepts with remapped relations
    """
    # Issue deprecation warning
    warnings.warn(
        "deduplicate_by_embedding is deprecated. Use LLM-based verification with "
        "'same_concept' direction for more accurate concept merging.",
        DeprecationWarning,
        stacklevel=2
    )

    if not concepts:
        return []

    threshold = similarity_threshold or settings.similarity_threshold

    # Validate threshold
    if not (0 < threshold <= 1):
        logger.warning(f"Invalid threshold {threshold}, using 0.9")
        threshold = 0.9

    # Note: combined_embedding is deprecated, using name_embedding for deduplication
    with_emb_idx = [i for i, c in enumerate(concepts) if c.name_embedding is not None]
    without_emb_idx = [i for i, c in enumerate(concepts) if c.name_embedding is None]

    if not with_emb_idx:
        logger.info("No concepts with embeddings to deduplicate")
        return concepts

    with_emb_idx = _filter_consistent_dim(concepts, with_emb_idx)
    if not with_emb_idx:
        return concepts

    emb_mat = np.array([concepts[i].name_embedding for i in with_emb_idx], dtype=float)

    if len(emb_mat.shape) != _EMB_MATRIX_NDIM:
        logger.error("Embeddings must be 2D arrays")
        return concepts

    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb_mat = emb_mat / norms

    sim = cosine_similarity(emb_mat)

    id_map, merged_concepts = _build_components(concepts, with_emb_idx, without_emb_idx, sim, threshold)

    final = []
    for concept in merged_concepts:
        remapped = concept.copy(deep=True)
        remapped.relations = _remap_and_dedup_relations(
            remapped.relations, id_map, self_id=remapped.concept_id)
        final.append(remapped)

    reduction = len(concepts) - len(final)
    logger.info(
        f"Deduplicated {len(concepts)} concepts into {len(final)} (reduction={reduction})")
    return final


def _filter_consistent_dim(concepts: list[Concept], with_emb_idx: list[int]) -> list[int]:
    """Filter embedding indices to keep only the most common dimension."""
    emb_dims = [len(concepts[i].name_embedding) for i in with_emb_idx]
    if len(set(emb_dims)) <= 1:
        return with_emb_idx
    logger.warning(f"Embeddings have inconsistent dimensions: {set(emb_dims)}")
    most_common_dim = Counter(emb_dims).most_common(1)[0][0]
    logger.info(f"Filtering to keep only embeddings with dimension {most_common_dim}")
    filtered = [with_emb_idx[i] for i, dim in enumerate(emb_dims) if dim == most_common_dim]
    if not filtered:
        logger.warning("No valid embeddings after filtering")
    return filtered


def _union_find(n: int, sim: np.ndarray, threshold: float) -> list[int]:
    """Run union-find on similarity matrix, return parent array."""
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    return parent


def _build_components(
    concepts: list[Concept],
    with_emb_idx: list[int],
    without_emb_idx: list[int],
    sim: np.ndarray,
    threshold: float,
) -> tuple[dict[str, str], list[Concept]]:
    """Build union-find components and merge them."""
    n = len(with_emb_idx)
    parent = _union_find(n, sim, threshold)

    # Group by root — need path-compressed find here too
    def find_root(parent_arr: list[int], x: int) -> int:
        while parent_arr[x] != x:
            x = parent_arr[x]
        return x

    comp: dict[int, list[int]] = {}
    for k in range(n):
        comp.setdefault(find_root(parent, k), []).append(with_emb_idx[k])

    id_map, merged_concepts = _merge_component_groups(concepts, comp)

    for idx in without_emb_idx:
        concept = concepts[idx]
        id_map[concept.concept_id] = concept.concept_id
        merged_concepts.append(concept)

    return id_map, merged_concepts


def _merge_component_groups(
    concepts: list[Concept],
    comp: dict[int, list[int]],
) -> tuple[dict[str, str], list[Concept]]:
    """Merge grouped concept indices into canonical concepts."""
    id_map: dict[str, str] = {}
    merged_concepts: list[Concept] = []

    for member_idx_list in comp.values():
        if len(member_idx_list) == 1:
            concept = concepts[member_idx_list[0]]
            id_map[concept.concept_id] = concept.concept_id
            merged_concepts.append(concept)
        else:
            group = [concepts[idx] for idx in member_idx_list]
            merged, group_id_map = _merge_component(group)
            merged_concepts.append(merged)
            id_map.update(group_id_map)

    return id_map, merged_concepts


def _merge_component(group: list[Concept]) -> tuple[Concept, dict[str, str]]:
    """
    Merge a connected component (>=2 concepts) into a canonical concept.

    Selection criteria:
    1. Longest definition
    2. First in group (stable fallback)

    Returns:
        merged_concept, id_map (old_id -> canonical_id)
    """

    def _selection_key(c: Concept) -> int:
        return len(c.definition or "")

    canonical = max(group, key=_selection_key)

    concept_ids = [c.concept_id for c in group]
    subject_ids = {c.subject_id for c in group}
    if len(subject_ids) > 1:
        logger.debug(
            f"Mixed subject_id in merged group {concept_ids}; keeping '{canonical.subject_id}'")

    examples = _collect_unique(
        (ex for c in group for ex in (c.examples or []) if ex),
    )

    f_seen: set[str] = set()
    formulas = []
    for c in group:
        for f in c.formulas or []:
            key = getattr(f, "latex", None)
            if key and key not in f_seen:
                f_seen.add(key)
                formulas.append(f.model_dump())

    all_relations = [rel.model_dump() for c in group for rel in (c.relations or [])]

    avg_name_embedding = _average_embeddings(
        [np.asarray(c.name_embedding, dtype=float) for c in group if c.name_embedding is not None],
        label="name_embedding",
    )
    avg_definition_embedding = _average_embeddings(
        [np.asarray(c.definition_embedding, dtype=float) for c in group if c.definition_embedding is not None],
        label="definition_embedding",
    )

    merged = Concept(
        concept_id=canonical.concept_id,
        subject_id=canonical.subject_id,
        name=canonical.name,
        definition=canonical.definition,
        examples=examples,
        formulas=formulas,
        relations=all_relations,
        name_embedding=avg_name_embedding,
        definition_embedding=avg_definition_embedding,
    )

    id_map = {c.concept_id: canonical.concept_id for c in group}
    return merged, id_map


def _collect_unique(items) -> list:
    """Return unique items from an iterable, preserving first-seen order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _average_embeddings(emb_list: list[np.ndarray], label: str) -> list[float] | None:
    """Average a list of embedding arrays, handling inconsistent shapes."""
    if not emb_list:
        return None
    emb_shapes = [e.shape for e in emb_list]
    if len(set(emb_shapes)) > 1:
        logger.warning(f"Inconsistent {label} shapes in merge group: {set(emb_shapes)}, using first valid")
        return emb_list[0].tolist()
    return np.mean(emb_list, axis=0).tolist()


def _remap_and_dedup_relations(
    relations: list[Relation],
    id_map: dict[str, str],
    self_id: str,
) -> list[Relation]:
    """
    Remap target_id via id_map, drop self-loops, dedup by (type, target_id).
    Merge evidences; keep max confidence.

    Args:
        relations: List of relations to process
        id_map: Mapping from old concept_id to canonical concept_id
        self_id: Current concept's id (to filter self-loops)

    Returns:
        Cleaned and deduplicated list of relations
    """
    if not relations:
        return []

    bucket: dict[tuple[str, str], Relation] = {}

    for rel in relations:
        if rel is None or not getattr(rel, "target_id", None):
            continue

        tgt = id_map.get(rel.target_id, rel.target_id)

        if not tgt or tgt == self_id:
            continue

        key = (rel.type, tgt)

        if key not in bucket:
            new_rel = Relation(
                type=rel.type,
                target_id=tgt,
                confidence=rel.confidence,
                evidence=rel.evidence if hasattr(rel, "evidence") else None,
            )
            bucket[key] = new_rel

        cur = bucket[key]
        if rel.confidence is not None:
            cur.confidence = max(cur.confidence or 0.0, rel.confidence)

        if hasattr(rel, "evidence") and rel.evidence:
            if cur.evidence:
                if rel.evidence not in cur.evidence:
                    cur.evidence = cur.evidence + "\n" + rel.evidence
            else:
                cur.evidence = rel.evidence

    out = list(bucket.values())
    out.sort(key=lambda r: (r.type, r.target_id))
    return out
