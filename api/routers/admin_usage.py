"""
Admin usage router — aggregate LLM token/cost stats for the admin dashboard.

Guarded by ``get_current_user`` (the Next.js proxy only forwards this with a
valid internal service token, and only exposes it to admin callers).
"""

from __future__ import annotations

from datetime import timedelta
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_current_user
from api.schemas.common import ok
from api.shared.persistence.common import utc_now
from api.shared.persistence.documents import LlmUsageDocument

router = APIRouter(prefix="/admin/llm-usage", tags=["admin-usage"])

_WINDOW_RE = re.compile(r"^(\d+)([dh])$")


def _parse_window_days(window: str) -> int:
    match = _WINDOW_RE.match(window.strip().lower())
    if not match:
        return 30
    amount = int(match.group(1))
    return amount if match.group(2) == "d" else max(1, amount // 24)


async def _group_sum(match_stage: dict[str, Any], group_key: str) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": match_stage},
        {
            "$group": {
                "_id": f"${group_key}",
                "total_tokens": {"$sum": "$total_tokens"},
                "input_tokens": {"$sum": "$input_tokens"},
                "output_tokens": {"$sum": "$output_tokens"},
                "cost_usd": {"$sum": "$cost_usd"},
                "calls": {"$sum": 1},
            }
        },
        {"$sort": {"cost_usd": -1}},
    ]
    rows = await LlmUsageDocument.aggregate(pipeline).to_list()
    return [{"key": r["_id"], **{k: v for k, v in r.items() if k != "_id"}} for r in rows]


@router.get("/summary")
async def get_llm_usage_summary(
    _user_id: Annotated[str, Depends(get_current_user)],
    window: str = Query(default="30d"),
) -> Any:
    """Aggregate LLM usage/cost over a time window, broken down by action + model."""
    days = _parse_window_days(window)
    since = utc_now() - timedelta(days=days)
    match_stage = {"created_at": {"$gte": since}}

    totals_rows = await LlmUsageDocument.aggregate(
        [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "input_tokens": {"$sum": "$input_tokens"},
                    "output_tokens": {"$sum": "$output_tokens"},
                    "cost_usd": {"$sum": "$cost_usd"},
                    "calls": {"$sum": 1},
                }
            },
        ]
    ).to_list()
    totals = totals_rows[0] if totals_rows else {}
    totals.pop("_id", None)

    daily = await LlmUsageDocument.aggregate(
        [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "cost_usd": {"$sum": "$cost_usd"},
                    "calls": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
    ).to_list()

    return ok(
        {
            "window_days": days,
            "totals": {
                "total_tokens": totals.get("total_tokens", 0),
                "input_tokens": totals.get("input_tokens", 0),
                "output_tokens": totals.get("output_tokens", 0),
                "cost_usd": totals.get("cost_usd", 0.0),
                "calls": totals.get("calls", 0),
            },
            "by_action": await _group_sum(match_stage, "action"),
            "by_model": await _group_sum(match_stage, "model"),
            "daily": [
                {
                    "date": r["_id"],
                    "total_tokens": r["total_tokens"],
                    "cost_usd": r["cost_usd"],
                    "calls": r["calls"],
                }
                for r in daily
            ],
        }
    )
