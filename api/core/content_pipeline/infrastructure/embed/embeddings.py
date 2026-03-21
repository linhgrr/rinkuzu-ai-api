"""Embedding computation for concepts."""

from typing import List

from loguru import logger

from ..llm.schemas import Concept
from .embedding_client import EmbeddingClient


def compute_embedding_for_concepts(concepts: List[Concept],
                                   client: EmbeddingClient) -> None:
    """
    Compute embeddings for concepts IN-PLACE.

    This function modifies the input concepts by adding embeddings:
    - name_embedding: embedding of concept name only (used for CSR prerequisite ranking)
    - definition_embedding: embedding of definition only (used for CSR prerequisite ranking)

    Args:
        concepts: List of concepts to compute embeddings for (modified in-place)
        client: Embedding client for computing embeddings

    Returns:
        None (concepts are modified in-place)
    """

    if not concepts:
        logger.warning("No concepts provided for embedding computation.")
        return

    for concept in concepts:
        if not concept.name:
            logger.warning(
                f"Concept with ID {concept.concept_id} has no name. Skipping embedding computation.")
            continue
        if not concept.definition:
            logger.warning(
                f"Concept with ID {concept.concept_id} has no definition. Skipping embedding computation.")
            continue

        try:
            logger.info(
                f"Computing embeddings for concept ID {concept.concept_id}.")

            # Compute name embedding (for CSR prerequisite ranking)
            name_embedding = client.embed_query(concept.name)
            concept.name_embedding = name_embedding

            # Compute definition embedding (for CSR prerequisite ranking)
            definition_embedding = client.embed_query(concept.definition)
            concept.definition_embedding = definition_embedding

        except Exception as e:
            logger.warning(
                f"Failed to compute embeddings for concept ID {concept.concept_id}: {e}")
