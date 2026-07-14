from __future__ import annotations

import asyncio
from datetime import datetime
import json
from math import ceil
from statistics import median
import sys
from time import perf_counter
from typing import Any

from bson import BSON
from pymongo import DESCENDING, AsyncMongoClient

from api.config import get_settings

RUNS = 30
WARMUPS = 5

COMPACT_FIELDS = {
    "job_id": 1,
    "filename": 1,
    "subject_id": 1,
    "status": 1,
    "current_step": 1,
    "progress": 1,
    "total_chunks": 1,
    "page_batch_size": 1,
    "batch_count": 1,
    "failed_batch_count": 1,
    "partial_success": 1,
    "concepts_extracted": 1,
    "concepts_after_merge": 1,
    "relations_verified": 1,
    "graph_stats": 1,
    "quality_report": 1,
    "partial_graph": 1,
    "error_message": 1,
    "error_code": 1,
    "user_message": 1,
    "retryable": 1,
    "retry_count": 1,
    "eta_seconds": 1,
    "created_at": 1,
    "updated_at": 1,
    "heartbeat_at": 1,
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_result(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2) + "\n"
    )


async def _measure(fetch) -> dict[str, float | int]:
    for _ in range(WARMUPS):
        await fetch()

    durations: list[float] = []
    payload = None
    for _ in range(RUNS):
        started = perf_counter()
        payload = await fetch()
        durations.append((perf_counter() - started) * 1000)

    return {
        "bson_bytes": len(BSON.encode(payload or {})),
        "median_ms": round(median(durations), 2),
        "p95_ms": round(_percentile(durations, 0.95), 2),
        "runs": RUNS,
    }


async def main() -> None:
    uri = get_settings().mongodb_uri
    if not uri:
        _write_result({"status": "skipped", "reason": "MONGODB_URI is not configured"})
        return

    client = AsyncMongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    collection = client["adaptive_learning"]["al_pipeline_jobs"]
    try:
        job = await collection.find_one(
            {"status": "completed", "result": {"$ne": None}},
            {"job_id": 1, "user_id": 1},
            sort=[("updated_at", DESCENDING)],
        )
        if not job:
            _write_result({"status": "skipped", "reason": "No completed pipeline job found"})
            return

        match = {"job_id": job["job_id"], "user_id": job.get("user_id")}

        async def fetch_full():
            return await collection.find_one(match)

        async def fetch_compact():
            return await collection.find_one(match, COMPACT_FIELDS)

        async def fetch_debug():
            cursor = await collection.aggregate(
                [
                    {"$match": match},
                    {
                        "$set": {
                            "result.concept_embedding_count": {
                                "$size": {"$ifNull": ["$result.concept_embeddings", []]}
                            }
                        }
                    },
                    {"$unset": "result.concept_embeddings"},
                    {"$limit": 1},
                ]
            )
            rows = await cursor.to_list(length=1)
            return rows[0] if rows else None

        result = {
            "status": "ok",
            "scope": "read-only completed pipeline job",
            "full": await _measure(fetch_full),
            "compact": await _measure(fetch_compact),
            "debug_without_embeddings": await _measure(fetch_debug),
        }
        _write_result(result)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
