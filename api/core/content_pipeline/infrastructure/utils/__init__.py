"""Utility functions."""

from .mime import get_file_type, guess_mime_type
from .text import clean_text
from .timeit import timeit

__all__ = [
    "clean_text",
    "get_file_type",
    "guess_mime_type",
    "timeit",
]
