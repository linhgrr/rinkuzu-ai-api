"""Cache restore stages for existing pipeline outputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import time
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from api.config import get_settings
from api.domains.content_pipeline.domain.errors import PipelineStageTimeoutError
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from ..ports import SaveJobFn, raise_for_save_outcome
from .execution import run_blocking_stage

LoadMongoJobFn = Callable[[str], Awaitable[dict[str, Any] | None]]
PopulateMetricsFn = Callable[[PipelineJob], None]
HashFileFn = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class S3CacheRestoreResult:
    cache_key: str | None
    restored: bool


def _is_degradable_cache_error(exc: BaseException) -> bool:
    """Return true only for a genuine miss or provider/transport outage."""
    if isinstance(exc, ClientError):
        response = exc.response or {}
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {
            "NoSuchKey",
            "NotFound",
            "404",
            "AccessDenied",
            "RequestTimeout",
            "SlowDown",
            "ServiceUnavailable",
            "InternalError",
        } or status in {403, 404, 408, 429, 500, 502, 503, 504}
    return isinstance(
        exc,
        (
            BotoCoreError,
            ConnectionError,
            OSError,
            PipelineStageTimeoutError,
            TimeoutError,
        ),
    )


def _validate_cached_result(payload: Any) -> dict[str, Any]:
    """Validate the minimum durable pipeline result contract without coercion."""
    if not isinstance(payload, dict):
        raise TypeError("S3 pipeline cache must contain a JSON object")
    expected_types: dict[str, type] = {
        "concepts_data": dict,
        "concept_map": dict,
        "prereq_edges": list,
        "graph": dict,
        "stats": dict,
    }
    for field, expected_type in expected_types.items():
        if field not in payload or not isinstance(payload[field], expected_type):
            raise ValueError(f"S3 pipeline cache field {field!r} has an invalid shape")
    if any(
        not isinstance(concept_id, str) or isinstance(index, bool) or not isinstance(index, int)
        for concept_id, index in payload["concept_map"].items()
    ):
        raise ValueError("S3 pipeline cache concept_map must map string ids to integers")
    if any(not isinstance(item, dict) for item in payload["prereq_edges"]):
        raise ValueError("S3 pipeline cache prereq_edges must contain objects")
    return payload


async def try_restore_completed_job_from_mongo(
    job: PipelineJob,
    *,
    load_job: LoadMongoJobFn,
    populate_metrics: PopulateMetricsFn,
) -> bool:
    """Restore a completed job from MongoDB when available."""
    mongo_doc: dict[str, Any] | None = await load_job(job.job_id)
    if not (
        mongo_doc
        and mongo_doc.get("status") == PipelineStatus.COMPLETED.value
        and mongo_doc.get("result")
    ):
        return False

    job.filename = mongo_doc.get("filename", job.filename)
    job.subject_id = mongo_doc.get("subject_id", job.subject_id)
    job.total_chunks = int(mongo_doc.get("total_chunks", 0) or 0)
    job.result = mongo_doc["result"]
    job.concepts_extracted = int(mongo_doc.get("concepts_extracted", 0) or 0)
    job.concepts_after_merge = int(mongo_doc.get("concepts_after_merge", 0) or 0)
    job.relations_verified = int(mongo_doc.get("relations_verified", 0) or 0)
    graph_stats = mongo_doc.get("graph_stats")
    job.graph_stats = graph_stats if isinstance(graph_stats, dict) else {}
    populate_metrics(job)
    job.status = PipelineStatus.COMPLETED
    job.current_step = "Loaded from MongoDB"
    job.progress = PipelineProgress.COMPLETE
    job.completed_at = mongo_doc.get("completed_at", time.time())
    logger.info("[Pipeline] Job {} restored from MongoDB", job.job_id)
    return True


async def try_restore_completed_job_from_s3(
    job: PipelineJob,
    *,
    file_path: str,
    s3_client: Any,
    bucket_name: str | None,
    hash_file: HashFileFn,
    save_job: SaveJobFn,
    populate_metrics: PopulateMetricsFn,
) -> S3CacheRestoreResult:
    """Restore a completed job from S3 JSON cache when available."""
    if not s3_client or not bucket_name:
        return S3CacheRestoreResult(cache_key=None, restored=False)

    file_hash = await run_blocking_stage(
        hash_file,
        file_path,
        stage_name="s3_cache_hash",
    )
    cache_key = f"cache/{file_hash}.json"

    now = time.time()
    job.status = PipelineStatus.LOADING
    job.current_step = "Kiểm tra cache trên S3..."
    job.progress = PipelineProgress.CACHE_RESTORE
    job.updated_at = now
    job.heartbeat_at = now
    raise_for_save_outcome(
        job,
        await save_job(job),
        operation="persisting S3 cache lookup state",
    )

    settings = get_settings()
    cache_timeout_sec = max(1.0, float(settings.content_pipeline_cache_restore_timeout_sec))
    logger.info(
        "[Pipeline] Checking S3 cache job_id={} key={} timeout_sec={}",
        job.job_id,
        cache_key,
        cache_timeout_sec,
    )
    try:
        response: dict[str, Any] = await run_blocking_stage(
            s3_client.get_object,
            Bucket=bucket_name,
            Key=cache_key,
            stage_name="s3_cache_restore",
            timeout_sec=cache_timeout_sec,
        )
        cache_bytes = await run_blocking_stage(
            response["Body"].read,
            stage_name="s3_cache_body_read",
            timeout_sec=cache_timeout_sec,
        )
        cache_content = cache_bytes.decode("utf-8")
        cached_result = _validate_cached_result(json.loads(cache_content))
    except Exception as exc:
        if not _is_degradable_cache_error(exc):
            raise
        logger.warning(
            "[Pipeline] S3 cache unavailable/miss job_id={} key={} error_type={} error={}",
            job.job_id,
            cache_key,
            type(exc).__name__,
            str(exc)[:200],
        )
        return S3CacheRestoreResult(cache_key=cache_key, restored=False)

    job.result = cached_result
    populate_metrics(job)
    # Stay non-terminal until chunk rebuild succeeds — never expose a false
    # COMPLETED usable state when rebuild cannot finish.
    job.status = PipelineStatus.LOADING
    job.current_step = "Loaded from S3 cache"
    job.progress = PipelineProgress.CHUNKS_PERSISTING
    job.completed_at = None
    logger.info("[Pipeline] Job {} loaded from S3 cache {}", job.job_id, cache_key)
    raise_for_save_outcome(
        job,
        await save_job(job),
        operation="persisting S3-cached pipeline result",
    )
    return S3CacheRestoreResult(cache_key=cache_key, restored=True)
