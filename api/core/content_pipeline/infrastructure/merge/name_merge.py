"""Exact name-based concept merging."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from loguru import logger

from ..llm.postprocess import normalize_concept_name
from ..llm.schemas import Concept, Relation
from ..utils import clean_text


def merge_by_name(concepts: List[Concept]) -> List[Concept]:
    """
    Merge concepts by exact normalized name matching.

    Args:
        concepts: List of concepts to merge

    Returns:
        List of merged concepts
    """
    if not concepts:
        return []

    name_to_concepts: Dict[str, List[Concept]] = defaultdict(list)

    for concept in concepts:
        norm_name = normalize_concept_name(concept.name)
        if norm_name:
            name_to_concepts[norm_name].append(concept)

    merged = []
    processed = set()

    for concept in concepts:
        concept_id = concept.concept_id

        if concept_id in processed:
            continue

        norm_name = normalize_concept_name(concept.name)
        if not norm_name:
            merged.append(concept)
            processed.add(concept_id)
            continue

        similar_concepts = name_to_concepts[norm_name]

        if len(similar_concepts) == 1:
            merged.append(concept)
            processed.add(concept_id)
        else:
            merged_concept, id_map = _merge_concepts(similar_concepts)
            merged.append(merged_concept)

            # Mark all as processed
            for c in similar_concepts:
                processed.add(c.concept_id)

    # Remap relations across all merged concepts
    final = []
    id_map_global = {}

    # Build global id mapping
    for concept in concepts:
        norm_name = normalize_concept_name(concept.name)
        similar = name_to_concepts[norm_name]
        if len(similar) > 1:
            canonical_id = _select_canonical(similar).concept_id
            for c in similar:
                id_map_global[c.concept_id] = canonical_id
        else:
            id_map_global[concept.concept_id] = concept.concept_id

    # Remap relations
    for concept in merged:
        concept = concept.copy(deep=True)
        concept.relations = _remap_relations(
            concept.relations,
            id_map_global,
            self_id=concept.concept_id
        )
        final.append(concept)

    reduction = len(concepts) - len(final)
    logger.info(
        f"Merged {len(concepts)} concepts into {len(final)} by name (reduction={reduction})"
    )

    return final


def _select_canonical(concepts: List[Concept]) -> Concept:
    """
    Select canonical concept from a group.

    Selection criteria:
    1. Longest definition
    2. First in list (stable fallback)
    """
    def _key(c: Concept) -> int:
        def_len = len(c.definition or "")
        return -def_len  

    return min(concepts, key=_key)


def _merge_concepts(concepts: List[Concept]) -> Tuple[Concept, Dict[str, str]]:
    """
    Merge multiple concepts into one.

    Returns:
        (merged_concept, id_map)
    """
    if not concepts:
        raise ValueError("Cannot merge empty concept list")

    if len(concepts) == 1:
        return concepts[0], {concepts[0].concept_id: concepts[0].concept_id}

    # Select canonical concept
    canonical = _select_canonical(concepts)

    # Check for mixed subject_ids
    subject_ids = {c.subject_id for c in concepts}
    if len(subject_ids) > 1:
        concept_ids = [c.concept_id for c in concepts]
        logger.debug(
            f"Mixed subject_id in merged group {concept_ids}; keeping '{canonical.subject_id}'"
        )

    # Merge examples (preserve order, dedup)
    ex_seen: Set[str] = set()
    examples: List[str] = []
    for c in concepts:
        for ex in c.examples or []:
            ex_clean = clean_text(ex)
            if ex_clean and ex_clean not in ex_seen:
                ex_seen.add(ex_clean)
                examples.append(ex_clean)

    # Merge formulas (dedup by latex)
    f_seen: Set[str] = set()
    formulas: List[dict] = []
    for c in concepts:
        for f in c.formulas or []:
            key = getattr(f, "latex", None)
            if key and key not in f_seen:
                f_seen.add(key)
                formulas.append(f.model_dump())

    # Collect all relations (will be cleaned and remapped later)
    all_relations: List[dict] = []
    for c in concepts:
        all_relations.extend([rel.model_dump() for rel in (c.relations or [])])

    # Average embeddings (with dimension validation)
    # Average name_embedding
    name_emb_list = [
        np.asarray(c.name_embedding, dtype=float)
        for c in concepts if c.name_embedding is not None
    ]
    avg_name_embedding = None
    if name_emb_list:
        emb_shapes = [e.shape for e in name_emb_list]
        if len(set(emb_shapes)) > 1:
            logger.warning(
                f"Inconsistent name_embedding shapes in merge group: {set(emb_shapes)}, using canonical"
            )
            if canonical.name_embedding:
                avg_name_embedding = canonical.name_embedding
            else:
                avg_name_embedding = name_emb_list[0].tolist()
        else:
            avg_name_embedding = np.mean(name_emb_list, axis=0).tolist()

    # Average definition_embedding
    def_emb_list = [
        np.asarray(c.definition_embedding, dtype=float)
        for c in concepts if c.definition_embedding is not None
    ]
    avg_definition_embedding = None
    if def_emb_list:
        emb_shapes = [e.shape for e in def_emb_list]
        if len(set(emb_shapes)) > 1:
            logger.warning(
                f"Inconsistent definition_embedding shapes in merge group: {set(emb_shapes)}, using canonical"
            )
            if canonical.definition_embedding:
                avg_definition_embedding = canonical.definition_embedding
            else:
                avg_definition_embedding = def_emb_list[0].tolist()
        else:
            avg_definition_embedding = np.mean(def_emb_list, axis=0).tolist()

    # Create merged concept
    merged = Concept(
        concept_id=canonical.concept_id,
        subject_id=canonical.subject_id,
        name=canonical.name,
        definition=canonical.definition,
        examples=examples,
        formulas=formulas,
        relations=all_relations,  # Will be cleaned after global mapping
        name_embedding=avg_name_embedding,
        definition_embedding=avg_definition_embedding,
    )

    # Build id map (all -> canonical)
    id_map = {c.concept_id: canonical.concept_id for c in concepts}

    return merged, id_map


def _remap_relations(
    relations: List[Relation],
    id_map: Dict[str, str],
    self_id: str,
) -> List[Relation]:
    """
    Remap target_id via id_map, drop self-loops, dedup by (type, target_id).
    Merge evidences; keep max confidence.
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

        # Merge evidence text (concatenate if multiple, separated by newline)
        if hasattr(rel, "evidence") and rel.evidence:
            rel_evidence = clean_text(rel.evidence)
            if rel_evidence:
                if cur.evidence:
                    if rel_evidence not in cur.evidence:
                        cur.evidence = cur.evidence + "\n" + rel_evidence
                else:
                    cur.evidence = rel_evidence

    out = list(bucket.values())
    out.sort(key=lambda r: (r.type, r.target_id))
    return out
