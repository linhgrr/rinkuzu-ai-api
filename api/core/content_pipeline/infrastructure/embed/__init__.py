"""Embedding utilities for concepts and texts."""

from typing import List

from loguru import logger

from .....config import settings

from .embedding_client import EmbeddingClient
from .embeddings import compute_embedding_for_concepts


def compute_embeddings_batch(
    texts: List[str],
    batch_size: int = 50,
    truncate_long_texts: bool = True,
    max_length: int = 256,
) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding
        truncate_long_texts: Whether to truncate texts exceeding max_length
        max_length: Maximum number of tokens per text (default: 256)

    Returns:
        List of embedding vectors
    """
    client = EmbeddingClient(
        model_name=settings.embedding_model,
        batch_size=batch_size,
    )

    # Truncate long texts to avoid index out of range errors
    if truncate_long_texts:
        processed_texts = []
        for text in texts:
            if not text:
                processed_texts.append("")
                continue
            # Simple word-based truncation (approximate)
            words = text.split()
            if len(words) > max_length:
                truncated = " ".join(words[:max_length])
                logger.debug(f"Truncated text from {len(words)} to {max_length} words")
                processed_texts.append(truncated)
            else:
                processed_texts.append(text)
        texts = processed_texts

    return client.embed_documents(texts)


__all__ = [
    "EmbeddingClient",
    "compute_embedding_for_concepts",
    "compute_embeddings_batch",
]
