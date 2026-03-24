"""Embedding utilities for concepts and texts."""

from __future__ import annotations

from typing import List

from loguru import logger

from .....config import settings

__all__ = [
    "EmbeddingClient",
    "compute_embedding_for_concepts",
    "compute_embeddings_batch",
]


def __getattr__(name: str):
    if name == "EmbeddingClient":
        from .embedding_client import EmbeddingClient

        return EmbeddingClient
    if name == "compute_embedding_for_concepts":
        from .embeddings import compute_embedding_for_concepts

        return compute_embedding_for_concepts
    raise AttributeError(name)


def compute_embeddings_batch(
    texts: List[str],
    batch_size: int = 50,
    truncate_long_texts: bool = True,
    max_length: int = 256,
) -> List[List[float]]:
    from .embedding_client import EmbeddingClient

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
                logger.debug(f"Truncated text from {len(words)} to {max_length} words")
                processed_texts.append(truncated)
            else:
                processed_texts.append(text)
        texts = processed_texts

    return client.embed_documents(texts)
