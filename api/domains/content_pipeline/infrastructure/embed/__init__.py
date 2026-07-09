"""Embedding utilities for concepts and texts."""

from __future__ import annotations

from loguru import logger

from api.config import settings
from api.domains.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from api.domains.content_pipeline.infrastructure.embed.embeddings import (
    compute_embedding_for_concepts,
)

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
        tokenizer = getattr(getattr(client, "_model_handle", None), "model", None)
        tokenizer = getattr(tokenizer, "tokenizer", None)

        if tokenizer is not None:
            processed_texts = []
            for text in texts:
                if not text:
                    processed_texts.append("")
                    continue

                token_ids = tokenizer.encode(
                    text,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                truncated_text = tokenizer.decode(token_ids)
                if truncated_text != text:
                    logger.debug(
                        "Truncated text from {} to {} tokens",
                        len(token_ids),
                        max_length,
                    )
                processed_texts.append(truncated_text)
            texts = processed_texts

    return client.embed_documents(texts)
