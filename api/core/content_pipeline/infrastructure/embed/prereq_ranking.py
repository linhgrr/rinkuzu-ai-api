"""Prerequisite ranking using embeddings."""

from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from api.config import settings
from api.core.content_pipeline.infrastructure.llm.schemas import Concept
from api.core.content_pipeline.infrastructure.utils import timeit

_MIN_CONCEPTS_FOR_RANKING = 2


@timeit
def rank_prerequisites(
    concepts: list[Concept],
    prs_threshold: float | None = None,
):
    """
    Rank and add prerequisite edges using embeddings.

    Uses Context Similarity Ranking (CSR) approach:
    - CSR(A→B) = cos_sim(name_embedding(B), definition_embedding(A))
    - PRS(A,B) = max(CSR(A→B), CSR(B→A))
    Every pair with PRS >= threshold is considered a prerequisite relation, parse to Prerequisite agent to determine direction.

    Args:
        concepts: List of concepts with embeddings
        graph: Knowledge graph
        prs_threshold: Minimum PRS score (default from settings)

    Returns:
        List of potential prerequisite edges as tuples:
        (concept_id_1, concept_id_2)
    """
    threshold = prs_threshold or settings.prs_threshold

    concepts_with_emb = [
        c for c in concepts if c.name_embedding is not None and c.definition_embedding is not None
    ]

    if len(concepts_with_emb) < _MIN_CONCEPTS_FOR_RANKING:
        logger.warning("Not enough concepts with embeddings for prerequisite ranking")
        return []

    prereq_pairs = _compute_prerequisite_scores(
        concepts_with_emb,
        threshold,
    )

    added_count = 0
    potential_edges = []
    for concept_id_1, concept_id_2, prs_score in prereq_pairs:
        logger.info(
            "Prerequisite candidate: {} -> {} | PRS: {:.4f}",
            concept_id_1,
            concept_id_2,
            prs_score,
        )
        potential_edges.append((concept_id_1, concept_id_2))
        added_count += 1

    logger.info("Total prerequisite candidates found: {}", added_count)
    return potential_edges


def _compute_prerequisite_scores(
    concepts: list[Concept],
    threshold: float,
) -> list[tuple[str, str, float]]:
    """
    Compute PRS scores for concept pairs.

    Returns:
        List of (concept_id_1, concept_id_2, prs_score)
    """
    # Extract embeddings
    concept_ids = [c.concept_id for c in concepts]
    name_embeds = np.array([c.name_embedding for c in concepts])
    definition_embeds = np.array([c.definition_embedding for c in concepts])

    # Build existing relations map for quick lookup
    # Map concept_id -> set of target_ids that already have relations
    existing_relations = {}
    for concept in concepts:
        relations_set = set()
        if concept.relations:
            for rel in concept.relations:
                relations_set.add(rel.target_id)
        existing_relations[concept.concept_id] = relations_set

    # Compute CSR scores
    # CSR(A→B) = cos_sim(name_embedding(B), definition_embedding(A))
    csr_ab = cosine_similarity(definition_embeds, name_embeds)  # [i, j] = CSR(i→j)
    csr_ba = csr_ab.T  # [i, j] = CSR(j→i)

    prs_matrix = np.maximum(csr_ab, csr_ba)

    # select pairs above threshold
    pairs = []
    n = len(concepts)
    skipped_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            prs = prs_matrix[i, j]

            if prs < threshold:
                continue

            concept_id_i = concept_ids[i]
            concept_id_j = concept_ids[j]

            # Check if relation already exists (in either direction)
            has_existing_relation = concept_id_j in existing_relations.get(
                concept_id_i, set()
            ) or concept_id_i in existing_relations.get(concept_id_j, set())

            if has_existing_relation:
                logger.debug(
                    "Skipped pair ({}, {}) - relation already exists",
                    concept_id_i,
                    concept_id_j,
                )
                skipped_count += 1
                continue

            pairs.append((concept_id_i, concept_id_j, prs))

    if skipped_count > 0:
        logger.info("Skipped {} pairs that already have relations", skipped_count)

    return pairs
