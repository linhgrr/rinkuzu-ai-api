"""Concept merging and deduplication utilities."""

__all__ = [
    "deduplicate_by_embedding",
    "merge_by_name",
]


def __getattr__(name: str):
    if name == "deduplicate_by_embedding":
        from .embed_dedupe import deduplicate_by_embedding

        return deduplicate_by_embedding
    if name == "merge_by_name":
        from .name_merge import merge_by_name

        return merge_by_name
    raise AttributeError(name)
