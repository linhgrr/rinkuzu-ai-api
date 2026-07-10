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
from api.schemas.admin_usage import LlmUsageSummaryResponse
from api.schemas.common import StandardResponse, ok
from api.shared.persistence.common import utc_now
from api.shared.persistence.documents import LlmUsageDocument

router = APIRouter(prefix="/api/v1/admin/llm-usage", tags=["admin-usage"])

_WINDOW_RE = re.compile(r"^(\d+)([dh])$")


def _parse_window_days(window: str) -> int:
    match = _WINDOW_RE.match(window.strip().lower())
    if not match:
        return 30
    amount = int(match.group(1))
    return amount if match.group(2) == "d" else max(1, amount // 24)


async def _group_sum(
    match_stage: dict[str, Any], group_key: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    pipeline: list[dict[str, Any]] = [
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
    if limit is not None:
        pipeline.append({"$limit": limit})
    rows = await LlmUsageDocument.aggregate(pipeline).to_list()
    return [{"key": r["_id"], **{k: v for k, v in r.items() if k != "_id"}} for r in rows]


def _cost_share(row: dict[str, Any] | None, total_cost: float) -> float:
    if not row or total_cost <= 0:
        return 0.0
    return float(row.get("cost_usd") or 0.0) / total_cost


def _build_usage_analysis(
    *,
    totals: dict[str, Any],
    by_user: list[dict[str, Any]],
    by_action: list[dict[str, Any]],
    by_model: list[dict[str, Any]],
    active_users: int,
) -> dict[str, Any]:
    calls = int(totals.get("calls") or 0)
    total_tokens = int(totals.get("total_tokens") or 0)
    total_cost = float(totals.get("cost_usd") or 0.0)
    top_user = by_user[0] if by_user else None
    top_action = by_action[0] if by_action else None
    top_model = by_model[0] if by_model else None

    return {
        "active_users": active_users,
        "average_tokens_per_call": total_tokens / calls if calls else 0.0,
        "average_cost_usd_per_call": total_cost / calls if calls else 0.0,
        "top_user_id": top_user.get("key") if top_user else None,
        "top_user_cost_usd": float(top_user.get("cost_usd") or 0.0) if top_user else 0.0,
        "top_action": top_action.get("key") if top_action else None,
        "top_action_cost_share": _cost_share(top_action, total_cost),
        "top_model": top_model.get("key") if top_model else None,
        "top_model_cost_share": _cost_share(top_model, total_cost),
    }


@router.get("/summary", response_model=StandardResponse[LlmUsageSummaryResponse])
async def get_llm_usage_summary(
    _user_id: Annotated[str, Depends(get_current_user)],
    window: str = Query(default="30d"),
) -> dict[str, object]:
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
    by_action = await _group_sum(match_stage, "action")
    by_model = await _group_sum(match_stage, "model")
    by_user = await _group_sum(match_stage, "user_id", limit=10)
    active_user_rows = await LlmUsageDocument.aggregate(
        [
            {"$match": {**match_stage, "user_id": {"$ne": None}}},
            {"$group": {"_id": "$user_id"}},
            {"$count": "active_users"},
        ]
    ).to_list()
    active_users = int(active_user_rows[0]["active_users"]) if active_user_rows else 0

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
            "by_action": by_action,
            "by_model": by_model,
            "by_user": by_user,
            "analysis": _build_usage_analysis(
                totals=totals,
                by_user=by_user,
                by_action=by_action,
                by_model=by_model,
                active_users=active_users,
            ),
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
