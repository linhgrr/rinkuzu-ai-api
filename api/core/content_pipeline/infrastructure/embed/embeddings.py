"""Embedding computation for concepts."""

from loguru import logger

from api.core.content_pipeline.infrastructure.llm.schemas import Concept

from .embedding_client import EmbeddingClient


def compute_embedding_for_concepts(concepts: list[Concept], client: EmbeddingClient) -> None:
    """
    Compute embeddings for concepts IN-PLACE.

    This function modifies the input concepts by adding embeddings:
    - name_embedding: embedding of concept name (used for concept merging via cosine dedup)
    - definition_embedding: embedding of definition (averaged across duplicates during merge)

    Note: prerequisite ranking is now performed by MLPPrerequisiteRanker, which
    computes its own XLM-RoBERTa encodings of concept names and does not consume
    these vietnamese-sbert embeddings.

    Args:
        concepts: List of concepts to compute embeddings for (modified in-place)
        client: Embedding client for computing embeddings

    Returns:
        None (concepts are modified in-place)
    """

    if not concepts:
        logger.warning("No concepts provided for embedding computation.")
        return

    concepts_with_names: list[Concept] = []
    name_texts: list[str] = []
    concepts_with_definitions: list[Concept] = []
    definition_texts: list[str] = []

    for concept in concepts:
        if not concept.name:
            logger.warning(
                "Concept with ID {} has no name. Skipping embedding computation.",
                concept.concept_id,
            )
            continue
        concepts_with_names.append(concept)
        name_texts.append(concept.name)
        if concept.definition:
            concepts_with_definitions.append(concept)
            definition_texts.append(concept.definition)
        else:
            logger.warning(
                "Concept with ID {} has no definition. Skipping definition embedding.",
                concept.concept_id,
            )

    if not concepts_with_names:
        return

    try:
        logger.info("Computing name embeddings for {} concepts.", len(concepts_with_names))
        name_embeddings = client.embed_documents(name_texts)
        for concept, embedding in zip(concepts_with_names, name_embeddings, strict=False):
            concept.name_embedding = embedding
    except Exception:
        logger.exception("Failed to compute batched name embeddings")

    if not concepts_with_definitions:
        return

    try:
        logger.info(
            "Computing definition embeddings for {} concepts.",
            len(concepts_with_definitions),
        )
        definition_embeddings = client.embed_documents(definition_texts)
        for concept, embedding in zip(
            concepts_with_definitions,
            definition_embeddings,
            strict=False,
        ):
            concept.definition_embedding = embedding
    except Exception:
        logger.exception("Failed to compute batched definition embeddings")
