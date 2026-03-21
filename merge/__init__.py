"""Concept merging and deduplication utilities."""

from .embed_dedupe import deduplicate_by_embedding  # deprecated: use LLM verification with same_concept
from .name_merge import merge_by_name

__all__ = [
    "deduplicate_by_embedding",  # deprecated
    "merge_by_name",
]
