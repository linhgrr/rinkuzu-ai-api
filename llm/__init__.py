"""Backward-compatible shim for legacy root imports."""

from api.core.content_pipeline.infrastructure.llm import get_embeddings, get_llm

__all__ = ["get_llm", "get_embeddings"]
