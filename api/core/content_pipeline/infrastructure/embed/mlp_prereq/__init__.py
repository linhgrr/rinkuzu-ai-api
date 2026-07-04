"""MLP-based prerequisite ranking module.

Replaces the legacy cosine-similarity PRS with a supervised classifier
trained on ViMath using BAAI/bge-m3 name+definition embeddings.
"""

from .ranker import MLPPrerequisiteRanker

__all__ = ["MLPPrerequisiteRanker"]
