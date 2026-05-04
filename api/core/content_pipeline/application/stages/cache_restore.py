"""Cache restore stages for existing pipeline outputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
import time

from loguru import logger

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .execution import run_blocking_stage

LoadMongoJobFn = Callable[[str], Awaitable[dict | None]]
SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]
PopulateMetricsFn = Callable[[PipelineJob], None]
HashFileFn = Callable[[str], str]


async def try_restore_completed_job_from_mongo(
    job: PipelineJob,
    *,
    load_job: LoadMongoJobFn,
    populate_metrics: PopulateMetricsFn,
) -> bool:
    """Restore a completed job from MongoDB when available."""
    mongo_doc = await load_job(job.job_id)
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
    job.graph_stats = (
        mongo_doc.get("graph_stats") if isinstance(mongo_doc.get("graph_stats"), dict) else {}
    )
    populate_metrics(job)
    job.status = PipelineStatus.COMPLETED
    job.current_step = "Loaded from MongoDB"
    job.progress = 1.0
    job.completed_at = mongo_doc.get("completed_at", time.time())
    logger.info(f"[Pipeline] Job {job.job_id} restored from MongoDB")
    return True


async def try_restore_completed_job_from_s3(
    job: PipelineJob,
    *,
    file_path: str,
    s3_client,
    bucket_name: str | None,
    hash_file: HashFileFn,
    save_job: SaveJobFn,
    populate_metrics: PopulateMetricsFn,
) -> str | None:
    """Restore a completed job from S3 JSON cache when available."""
    cache_key = f"cache/{hash_file(file_path)}.json"
    if not s3_client or not bucket_name:
        return cache_key

    job.status = PipelineStatus.LOADING
    job.current_step = "Kiểm tra cache trên S3..."
    job.progress = 0.02
    saved = False
    try:
        response = await run_blocking_stage(
            s3_client.get_object,
            Bucket=bucket_name,
            Key=cache_key,
            stage_name="s3_cache_restore",
        )
        cache_content = response["Body"].read().decode("utf-8")
        job.result = json.loads(cache_content)
        populate_metrics(job)
        job.status = PipelineStatus.COMPLETED
        job.current_step = "Loaded from S3 cache"
        job.progress = 1.0
        job.completed_at = time.time()
        logger.info(f"[Pipeline] Job {job.job_id} loaded from S3 cache {cache_key}")
        saved = await save_job(job)
    except Exception:
        logger.debug(f"[Pipeline] Cache miss: {cache_key}")
        return cache_key
    if not saved:
        raise RuntimeError("Failed to persist S3-cached pipeline result to MongoDB")
    return cache_key
