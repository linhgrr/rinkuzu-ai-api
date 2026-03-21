"""Post-processing for extracted concepts."""

from typing import List, Optional, Set
import re
import unicodedata

from llm.schemas import Concept, Relation
from utils import clean_text
from loguru import logger


def postprocess_concepts(concepts: List[Concept]) -> List[Concept]:
    """
    Post-process extracted concepts.

    Args:
        concepts: List of raw extracted concepts

    Returns:
        Cleaned and normalized concepts
    """
    if not concepts:
        logger.info("No concepts to post-process")
        return []

    # Normalize concept IDs first so relation.target_id can be matched reliably.
    for concept in concepts:
        concept.concept_id = normalize_concept_id(
            getattr(concept, "concept_id", "") or ""
        )

    concept_ids: Set[str] = {
        c.concept_id for c in concepts if getattr(c, "concept_id", None)}

    processed: List[Concept] = []
    for concept in concepts:
        concept.name = clean_text(getattr(concept, "name", "")) or ""
        concept.definition = clean_text(
            getattr(concept, "definition", "")) or ""

        ex_seen = set()
        cleaned_examples: List[str] = []
        for ex in getattr(concept, "examples", []) or []:
            ex_clean = clean_text(ex)
            if ex_clean and ex_clean not in ex_seen:
                ex_seen.add(ex_clean)
                cleaned_examples.append(ex_clean)
        concept.examples = cleaned_examples

        cleaned_relations: List[Relation] = []
        for rel in getattr(concept, "relations", []) or []:
            pr = _postprocess_relation(rel, concept_ids)
            if pr and _is_valid_relation(pr, concept_ids):
                cleaned_relations.append(pr)
        concept.relations = cleaned_relations

        processed.append(concept)

    logger.info(f"Post-processed {len(processed)} concepts")
    return processed


def _postprocess_relation(relation: Optional[Relation], concept_ids: Set[str]) -> Optional[Relation]:
    """Post-process a single relation.
    
    NOTE: We do NOT drop relations whose target_id is not in concept_ids,
    because the target may be a concept from a different extraction batch.
    The graph builder will handle missing targets gracefully.
    """
    if relation is None:
        return None

    relation.target_id = normalize_concept_id(
        getattr(relation, "target_id", "") or ""
    )
    if not relation.target_id:
        return None

    if relation.target_id not in concept_ids:
        logger.debug(
            f"Target concept ID '{relation.target_id}' not in current batch "
            f"(may be cross-batch reference, keeping)")

    # Clean evidence text if exists
    if hasattr(relation, "evidence") and relation.evidence:
        relation.evidence = clean_text(relation.evidence)

    return relation


def _is_valid_relation(relation: Optional[Relation], concept_ids: Set[str]) -> bool:
    """Check if a relation is valid.

    Args:
        relation: Relation to check
        concept_ids: Set of valid concept IDs
    Returns:
        True if valid
    """
    if relation is None:
        return False

    if not relation.target_id:
        return False

    # NOTE: We intentionally do NOT check target_id in concept_ids.
    # Cross-batch relations reference concepts from other batches.
    # The graph builder + content_pipeline will filter invalid targets
    # after ALL batches are merged.

    return True


def normalize_concept_name(name: str) -> str:
    """
    Normalize concept name for matching.

    Steps:
    - Lowercase
    - Remove Vietnamese accents
    - Remove special characters
    - Normalize whitespace
    """
    if not name:
        return ""

    normalized = name.lower().strip()

    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn")

    normalized = normalized.replace("đ", "d")

    normalized = re.sub(r"[^\w\s]", " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def normalize_concept_id(concept_id: str) -> str:
    """
    Normalize concept ID while preserving underscore separators.

    Steps:
    - Trim surrounding spaces / backticks
    - Normalize unicode form
    - Convert whitespace to underscore
    - Keep only letters/digits/underscore/hyphen
    - Collapse repeated underscores
    - Lowercase
    """
    if not concept_id:
        return ""

    cid = str(concept_id).strip().strip("`")
    cid = unicodedata.normalize("NFKC", cid)
    cid = re.sub(r"\s+", "_", cid)
    cid = re.sub(r"[^\w-]", "_", cid, flags=re.UNICODE)
    cid = re.sub(r"_+", "_", cid).strip("_")
    return cid.lower()
