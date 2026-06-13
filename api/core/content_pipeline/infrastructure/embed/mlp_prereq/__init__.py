"""MLP-based prerequisite ranking module.

Replaces the legacy cosine-similarity PRS with a supervised classifier
trained on LectureBank using BAAI/bge-m3 embeddings.
"""

from .ranker import MLPPrerequisiteRanker

__all__ = ["MLPPrerequisiteRanker"]
