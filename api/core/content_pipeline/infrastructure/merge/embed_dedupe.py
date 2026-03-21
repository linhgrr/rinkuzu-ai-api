"""Embedding-based concept deduplication (fixed & schema-compliant)."""

import warnings
from typing import Dict, List, Set, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

from ..llm.schemas import Concept, Relation


def deduplicate_by_embedding(
    concepts: List[Concept],
    similarity_threshold: float = None,
) -> List[Concept]:
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

    threshold = similarity_threshold or getattr(
        settings, "similarity_threshold", 0.9)

    # Validate threshold
    if not (0 < threshold <= 1):
        logger.warning(f"Invalid threshold {threshold}, using 0.9")
        threshold = 0.9

    # Note: combined_embedding is deprecated, using name_embedding for deduplication
    with_emb_idx = [i for i, c in enumerate(
        concepts) if c.name_embedding is not None]
    without_emb_idx = [i for i, c in enumerate(
        concepts) if c.name_embedding is None]

    if not with_emb_idx:
        logger.info("No concepts with embeddings to deduplicate")
        return concepts

    # Validate embedding dimensions before building matrix
    emb_dims = [len(concepts[i].name_embedding) for i in with_emb_idx]
    if len(set(emb_dims)) > 1:
        logger.warning(
            f"Embeddings have inconsistent dimensions: {set(emb_dims)}")
        from collections import Counter
        most_common_dim = Counter(emb_dims).most_common(1)[0][0]
        logger.info(
            f"Filtering to keep only embeddings with dimension {most_common_dim}")
        filtered_idx = [with_emb_idx[i]
                        for i, dim in enumerate(emb_dims) if dim == most_common_dim]
        with_emb_idx = filtered_idx
        if not with_emb_idx:
            logger.warning("No valid embeddings after filtering")
            return concepts

    # build embedding matrix
    emb_mat = np.array(
        [concepts[i].name_embedding for i in with_emb_idx], dtype=float)

    # validate embedding matrix shape
    if len(emb_mat.shape) != 2:
        logger.error("Embeddings must be 2D arrays")
        return concepts

    # Normalize embeddings
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb_mat = emb_mat / norms

    # Compute cosine similarity
    sim = cosine_similarity(emb_mat)

    # Union-Find for connected components
    parent = list(range(len(with_emb_idx)))

    def find(x: int) -> int:
        """Find root with path compression."""
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(a: int, b: int) -> None:
        """Union two components."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build similarity graph
    n = len(with_emb_idx)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    # Group by connected components
    comp: Dict[int, List[int]] = {}
    for k in range(n):
        rk = find(k)
        comp.setdefault(rk, []).append(with_emb_idx[k])

    # Merge components
    id_map: Dict[str, str] = {}  # old_id -> canonical_id
    merged_concepts: List[Concept] = []

    for root, member_idx_list in comp.items():
        if len(member_idx_list) == 1:
            # Single concept, no merge needed
            c = concepts[member_idx_list[0]]
            id_map[c.concept_id] = c.concept_id
            merged_concepts.append(c)
            continue

        # Merge multiple concepts
        group = [concepts[idx] for idx in member_idx_list]
        merged, group_id_map = _merge_component(group)
        merged_concepts.append(merged)
        id_map.update(group_id_map)

    # Add concepts without embeddings
    for idx in without_emb_idx:
        c = concepts[idx]
        id_map[c.concept_id] = c.concept_id
        merged_concepts.append(c)

    # Remap relations across all concepts
    final = []
    for c in merged_concepts:
        c = c.copy(deep=True)
        c.relations = _remap_and_dedup_relations(
            c.relations, id_map, self_id=c.concept_id)
        final.append(c)

    reduction = len(concepts) - len(final)
    logger.info(
        f"Deduplicated {len(concepts)} concepts into {len(final)} (reduction={reduction})")
    return final


def _merge_component(group: List[Concept]) -> Tuple[Concept, Dict[str, str]]:
    """
    Merge a connected component (>=2 concepts) into a canonical concept.

    Selection criteria:
    1. Longest definition
    2. First in group (stable fallback)

    Returns:
        merged_concept, id_map (old_id -> canonical_id)
    """

    def _selection_key(c: Concept) -> int:
        """Generate sort key for canonical selection."""
        def_len = len(c.definition or "")
        return def_len

    canonical = max(group, key=_selection_key)

    concept_ids = [c.concept_id for c in group]
    subject_ids = {c.subject_id for c in group}
    if len(subject_ids) > 1:
        logger.debug(
            f"Mixed subject_id in merged group {concept_ids}; keeping '{canonical.subject_id}'")

    ex_seen: Set[str] = set()
    examples: List[str] = []
    for c in group:
        for ex in c.examples or []:
            if ex and ex not in ex_seen:
                ex_seen.add(ex)
                examples.append(ex)

    f_seen: Set[str] = set()
    formulas = []
    for c in group:
        for f in c.formulas or []:
            key = getattr(f, "latex", None)
            if key and key not in f_seen:
                f_seen.add(key)
                formulas.append(f.model_dump())

    all_relations = []
    for c in group:
        all_relations.extend([rel.model_dump() for rel in (c.relations or [])])

    name_emb_list = [np.asarray(c.name_embedding, dtype=float)
                     for c in group if c.name_embedding is not None]
    avg_name_embedding = None
    if name_emb_list:
        emb_shapes = [e.shape for e in name_emb_list]
        if len(set(emb_shapes)) > 1:
            logger.warning(
                f"Inconsistent name_embedding shapes in merge group: {set(emb_shapes)}, using first valid")
            avg_name_embedding = name_emb_list[0].tolist()
        else:
            avg_name_embedding = np.mean(name_emb_list, axis=0).tolist()

    def_emb_list = [np.asarray(c.definition_embedding, dtype=float)
                    for c in group if c.definition_embedding is not None]
    avg_definition_embedding = None
    if def_emb_list:
        emb_shapes = [e.shape for e in def_emb_list]
        if len(set(emb_shapes)) > 1:
            logger.warning(
                f"Inconsistent definition_embedding shapes in merge group: {set(emb_shapes)}, using first valid")
            avg_definition_embedding = def_emb_list[0].tolist()
        else:
            avg_definition_embedding = np.mean(def_emb_list, axis=0).tolist()

    merged = Concept(
        concept_id=canonical.concept_id,
        subject_id=canonical.subject_id,
        name=canonical.name,
        definition=canonical.definition,
        examples=examples,
        formulas=formulas,
        relations=all_relations,  # will be cleaned after global mapping
        name_embedding=avg_name_embedding,
        definition_embedding=avg_definition_embedding,
    )

    id_map = {c.concept_id: canonical.concept_id for c in group}
    return merged, id_map


def _remap_and_dedup_relations(
    relations: List[Relation],
    id_map: Dict[str, str],
    self_id: str,
) -> List[Relation]:
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

    bucket: Dict[Tuple[str, str], Relation] = {}

    for rel in relations:
        if rel is None or not getattr(rel, "target_id", None):
            continue

        # Remap target to canonical id
        tgt = id_map.get(rel.target_id, rel.target_id)

        if not tgt or tgt == self_id:
            # Drop invalid or self-loop
            continue

        key = (rel.type, tgt)

        if key not in bucket:
            # Create new relation entry
            new_rel = Relation(
                type=rel.type,
                target_id=tgt,
                confidence=rel.confidence,
                evidence=rel.evidence if hasattr(rel, "evidence") else None,
            )
            bucket[key] = new_rel

        # Merge scores (keep max confidence)
        cur = bucket[key]
        if rel.confidence is not None:
            cur.confidence = max(cur.confidence or 0.0, rel.confidence)

        # Merge evidence text (concatenate if multiple)
        if hasattr(rel, "evidence") and rel.evidence:
            if cur.evidence:
                # Append if not duplicate
                if rel.evidence not in cur.evidence:
                    cur.evidence = cur.evidence + "\n" + rel.evidence
            else:
                cur.evidence = rel.evidence

    # Return relations in stable order: by type then target_id
    out = list(bucket.values())
    out.sort(key=lambda r: (r.type, r.target_id))
    return out
