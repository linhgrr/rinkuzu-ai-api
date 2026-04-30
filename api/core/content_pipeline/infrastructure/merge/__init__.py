"""Concept merging and deduplication utilities."""

from .embed_dedupe import deduplicate_by_embedding
from .name_merge import merge_by_name

__all__ = [
    "deduplicate_by_embedding",
    "merge_by_name",
]
