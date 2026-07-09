from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _normalize_history_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item

    from api.domains.learning.exercise_types.registry import get_handler

    payload = getattr(item, "payload", None)
    base: dict[str, Any] = {
        "question": getattr(item, "question", ""),
        "bloom_level": getattr(item, "bloom_level", None),
    }
    if payload is None:
        return base

    content = get_handler(payload.exercise_type).to_response_dict(item)
    base["exercise_type"] = payload.exercise_type.value
    for key in (
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
    ):
        value = content.get(key)
        if value not in (None, "", [], {}):
            base[key] = value
    return base


def format_exercise_history(history: Sequence[Any]) -> str:
    normalized = [_normalize_history_item(item) for item in history]
    return json.dumps(normalized, ensure_ascii=False, indent=2)
