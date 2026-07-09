"""Exact name-based concept merging."""

from collections import defaultdict

from loguru import logger

from api.domains.content_pipeline.infrastructure.llm.postprocess import normalize_concept_name
from api.domains.content_pipeline.infrastructure.llm.schemas import Concept, Relation
from api.domains.content_pipeline.infrastructure.utils import clean_text


def merge_by_name(concepts: list[Concept]) -> list[Concept]:
    """
    Merge concepts by exact normalized name matching.

    Args:
        concepts: List of concepts to merge

    Returns:
        List of merged concepts
    """
    if not concepts:
        return []

    name_to_concepts: dict[str, list[Concept]] = defaultdict(list)

    for concept in concepts:
        norm_name = normalize_concept_name(concept.name)
        if norm_name:
            name_to_concepts[norm_name].append(concept)

    merged = _group_and_merge(concepts, name_to_concepts)
    id_map_global = _build_global_id_map(concepts, name_to_concepts)

    final = []
    for concept in merged:
        remapped = concept.copy(deep=True)
        remapped.relations = _remap_relations(
            remapped.relations,
            id_map_global,
            self_id=remapped.concept_id,
        )
        final.append(remapped)

    reduction = len(concepts) - len(final)
    logger.info(
        "Merged {} concepts into {} by name (reduction={})",
        len(concepts),
        len(final),
        reduction,
    )

    return final


def _group_and_merge(
    concepts: list[Concept],
    name_to_concepts: dict[str, list[Concept]],
) -> list[Concept]:
    """Group concepts by normalized name and merge each group."""
    merged = []
    processed: set[str] = set()

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
            merged_concept, _ = _merge_concepts(similar_concepts)
            merged.append(merged_concept)
            for c in similar_concepts:
                processed.add(c.concept_id)

    return merged


def _build_global_id_map(
    concepts: list[Concept],
    name_to_concepts: dict[str, list[Concept]],
) -> dict[str, str]:
    """Build a mapping from old concept IDs to canonical concept IDs."""
    id_map: dict[str, str] = {}
    for concept in concepts:
        norm_name = normalize_concept_name(concept.name)
        similar = name_to_concepts[norm_name]
        if len(similar) > 1:
            canonical_id = _select_canonical(similar).concept_id
            for c in similar:
                id_map[c.concept_id] = canonical_id
        else:
            id_map[concept.concept_id] = concept.concept_id
    return id_map


def _select_canonical(concepts: list[Concept]) -> Concept:
    """
    Select canonical concept from a group.

    Selection criteria:
    1. Longest definition
    2. First in list (stable fallback)
    """

    def _key(c: Concept) -> int:
        return -len(c.definition or "")

    return min(concepts, key=_key)


def _merge_concepts(concepts: list[Concept]) -> tuple[Concept, dict[str, str]]:
    """
    Merge multiple concepts into one.

    Returns:
        (merged_concept, id_map)
    """
    if not concepts:
        raise ValueError("Cannot merge empty concept list")

    if len(concepts) == 1:
        return concepts[0], {concepts[0].concept_id: concepts[0].concept_id}

    canonical = _select_canonical(concepts)

    subject_ids = {c.subject_id for c in concepts}
    if len(subject_ids) > 1:
        concept_ids = [c.concept_id for c in concepts]
        logger.debug(
            "Mixed subject_id in merged group {}; keeping '{}'",
            concept_ids,
            canonical.subject_id,
        )

    ex_seen: set[str] = set()
    examples: list[str] = []
    for c in concepts:
        for ex in c.examples or []:
            ex_clean = clean_text(ex)
            if ex_clean and ex_clean not in ex_seen:
                ex_seen.add(ex_clean)
                examples.append(ex_clean)

    f_seen: set[str] = set()
    formulas: list[dict] = []
    for c in concepts:
        for f in c.formulas or []:
            key = getattr(f, "latex", None)
            if key and key not in f_seen:
                f_seen.add(key)
                formulas.append(f.model_dump())

    all_relations: list[dict] = [rel.model_dump() for c in concepts for rel in (c.relations or [])]

    merged = Concept(
        concept_id=canonical.concept_id,
        subject_id=canonical.subject_id,
        name=canonical.name,
        definition=canonical.definition,
        examples=examples,
        formulas=formulas,
        relations=all_relations,
    )

    id_map = {c.concept_id: canonical.concept_id for c in concepts}

    return merged, id_map


def _remap_relations(
    relations: list[Relation],
    id_map: dict[str, str],
    self_id: str,
) -> list[Relation]:
    """
    Remap target_id via id_map, drop self-loops, dedup by (type, target_id).
    Merge evidences; keep max confidence.
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
