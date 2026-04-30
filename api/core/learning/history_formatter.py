from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _normalize_history_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item

    payload: dict[str, Any] = {
        "question": getattr(item, "question", ""),
        "exercise_type": getattr(
            getattr(item, "exercise_type", None),
            "value",
            getattr(item, "exercise_type", None),
        ),
        "bloom_level": getattr(item, "bloom_level", None),
    }
    optional_fields = (
        "statement",
        "sentence",
        "hint",
        "options",
        "items",
        "pairs",
        "right_items",
        "rubric",
        "correct_option",
        "correct_answer",
    )
    for field in optional_fields:
        value = getattr(item, field, None)
        if value not in (None, "", [], {}):
            payload[field] = value
    return payload


def format_exercise_history(history: Sequence[Any]) -> str:
    normalized = [_normalize_history_item(item) for item in history]
    return json.dumps(normalized, ensure_ascii=False, indent=2)
