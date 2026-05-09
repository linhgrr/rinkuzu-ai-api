"""Embedding utilities for concepts and texts."""

from __future__ import annotations

from loguru import logger

from api.config import settings
from api.core.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from api.core.content_pipeline.infrastructure.embed.embeddings import compute_embedding_for_concepts

__all__ = [
    "EmbeddingClient",
    "compute_embedding_for_concepts",
    "compute_embeddings_batch",
]


def compute_embeddings_batch(
    texts: list[str],
    batch_size: int = 50,
    *,
    truncate_long_texts: bool = True,
    max_length: int = 256,
) -> list[list[float]]:
    client = EmbeddingClient(
        model_name=settings.embedding_model,
        batch_size=batch_size,
    )

    if truncate_long_texts:
        processed_texts = []
        for text in texts:
            if not text:
                processed_texts.append("")
                continue
            words = text.split()
            if len(words) > max_length:
                truncated = " ".join(words[:max_length])
                logger.debug("Truncated text from {} to {} words", len(words), max_length)
                processed_texts.append(truncated)
            else:
                processed_texts.append(text)
        texts = processed_texts

    return client.embed_documents(texts)
