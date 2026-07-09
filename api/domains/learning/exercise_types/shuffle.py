"""
shuffle.py — Deterministic display-order shuffle seeded by exercise_id.

The DB stores canonical order only; the shuffled order shown to the learner is
re-derived on every serve from the exercise_id, so it is stable across
generate/tutor/refetch without persisting any extra state.
"""

from __future__ import annotations

import random


def deterministic_shuffle(items: list[str], seed: str) -> list[str]:
    out = list(items)
    random.Random(seed).shuffle(out)  # noqa: S311
    return out
