"""Bloom taxonomy labels shared across the learning domain."""

from __future__ import annotations

# Canonical Bloom level → English label (1-indexed, matching the RL action space).
BLOOM_LABELS: dict[int, str] = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}

# Ordered label list (Bloom 1..6) for payloads that expose the sequence directly.
BLOOM_LABEL_SEQUENCE: list[str] = [BLOOM_LABELS[level] for level in range(1, 7)]
