"""
routers/pipeline.py — Content pipeline endpoints.
"""

from pathlib import Path
from typing import Annotated
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from api.config import get_settings
from api.main import limiter
from api.core.content_pipeline import PipelineStatus
from api.core.shared import mongo_store
from api.core.shared.url_fetch import UnsafeURLError, stream_download
from api.dependencies import (
    get_content_pipeline_availability,
    get_content_pipeline_service,
    get_current_user,
    get_session_manager,
    get_session_service,
)
from api.exceptions import PipelineNotCompletedError, PipelineNotFoundError, ServiceUnavailableError

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_MAX_FILENAME_LENGTH = 200


@router.get("/status")
async def pipeline_status(availability: Annotated[dict, Depends(get_content_pipeline_availability)]):
    """Check if content pipeline runtime modules are available."""
    available = availability["available"]
    return {
        "available": available,
        "service_initialized": availability["service_initialized"],
        "message": "Content pipeline ready" if available else "Content pipeline unavailable",
    }


class ProcessDocumentRequest(BaseModel):
    file_url: str
    filename: str
    subject_id: str | None = None
    prs_threshold: float = 0.75
    min_confidence: float = 0.6
    apply_reduction: bool = True


@router.post("/process")
@limiter.limit(get_settings().rate_limit_pipeline)
async def process_document(
    http_request: Request,
    request: ProcessDocumentRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    availability: Annotated[dict, Depends(get_content_pipeline_availability)],
    pipeline_service=Depends(get_content_pipeline_service),
):
    """Run the full content processing pipeline from an S3 uploaded file."""
    if not availability["available"]:
        logger.error(
            "[PipelineRouter] Content pipeline unavailable",
            user_id=user_id,
            error=availability["error"],
            src=availability["src"],
            service_initialized=availability["service_initialized"],
        )
        raise HTTPException(
            status_code=503,
            detail="Content pipeline is unavailable.",
        )
    if not mongo_store.is_available():
        raise ServiceUnavailableError("MongoDB persistence")

    # Sanitize filename — strip directory components, enforce .pdf, reject NUL / path seps
    raw_name = Path(request.filename or "").name
    if not raw_name or not raw_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if len(raw_name) > _MAX_FILENAME_LENGTH or any(c in raw_name for c in ("\x00", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_id = uuid.uuid4().hex[:8]
    save_path = (UPLOAD_DIR / f"{file_id}_{raw_name}").resolve()
    if not save_path.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Safe download — validates scheme, blocks private IPs, enforces size cap
    settings = get_settings()
    try:
        await stream_download(
            request.file_url,
            save_path,
            max_bytes=settings.download_max_bytes,
            allowlist=settings.download_host_allowlist or None,
        )
    except UnsafeURLError as exc:
        logger.warning("[PipelineRouter] Rejected unsafe URL: {}", exc)
        raise HTTPException(status_code=400, detail="URL not allowed.") from None
    except (httpx.HTTPError, OSError):
        logger.exception("[PipelineRouter] Failed to download file from {}", request.file_url)
        raise HTTPException(status_code=502, detail="Failed to download file.") from None

    try:
        job = await pipeline_service.start_job(
            file_path=str(save_path),
            subject_id=request.subject_id,
            prs_threshold=request.prs_threshold,
            min_confidence=request.min_confidence,
            apply_reduction=request.apply_reduction,
            user_id=user_id,
            content_processor_available=availability["available"],
            content_processor_src=availability["src"] or "",
        )
    except (RuntimeError, ValueError, OSError):
        logger.exception("[PipelineRouter] Failed to initialize pipeline job for {}", request.filename)
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            logger.warning("[PipelineRouter] Failed to cleanup upload {}", save_path)
        raise HTTPException(status_code=500, detail="Failed to initialize pipeline job.") from None

    return {
        "job_id": job.job_id,
        "filename": request.filename,
        "file_size": save_path.stat().st_size,
        "subject_id": job.subject_id,
        "status": job.status.value,
        "message": "Processing started. Poll /api/pipeline/jobs/{job_id} for progress.",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user_id: Annotated[str, Depends(get_current_user)]):
    """Get pipeline job status and progress."""
    job_doc = await mongo_store.load_pipeline_job_for_user(job_id, user_id)
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    result = job_doc.get("result") or {}
    response = {
        "job_id": job_doc.get("job_id", job_id),
        "filename": job_doc.get("filename", ""),
        "subject_id": job_doc.get("subject_id", ""),
        "status": job_doc.get("status", "unknown"),
        "current_step": job_doc.get("current_step", "Loaded from MongoDB"),
        "progress": job_doc.get("progress", 1.0 if job_doc.get("status") == "completed" else 0.0),
        "total_chunks": job_doc.get("total_chunks", 0),
        "concepts_extracted": job_doc.get("concepts_extracted", 0),
        "concepts_after_merge": job_doc.get("concepts_after_merge", 0),
        "relations_verified": job_doc.get("relations_verified", 0),
        "graph_stats": job_doc.get("graph_stats", {}),
        "error_message": job_doc.get("error_message"),
        "error_code": job_doc.get("error_code"),
        "user_message": job_doc.get("user_message"),
        "retryable": bool(job_doc.get("retryable", False)),
        "is_terminal": job_doc.get("status") in {
            PipelineStatus.COMPLETED.value,
            PipelineStatus.FAILED.value,
            PipelineStatus.CANCELLED.value,
        },
        "partial_graph": job_doc.get("partial_graph"),
    }
    if job_doc.get("status") == PipelineStatus.COMPLETED.value and result:
        response["result"] = {
            "graph": result.get("graph", {"nodes": [], "edges": []}),
            "stats": result.get("stats", {}),
            "n_concepts": len(result.get("concept_map", {})),
        }
    return response


@router.post("/jobs/{job_id}/create-session")
async def create_session_from_pipeline(
    job_id: str,
    background_tasks: BackgroundTasks,
    max_steps: int = 9999,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    """Create a learning session from a completed pipeline job."""
    job_doc = await mongo_store.load_pipeline_job_for_user(job_id, user_id)
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    if job_doc.get("status") != "completed":
        raise PipelineNotCompletedError(job_id, job_doc.get("status", "unknown"))

    result = job_doc.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Job found but has no result data.")

    for required_key in ("concepts_data", "concept_map", "prereq_edges"):
        if required_key not in result:
            logger.error(
                "[PipelineRouter] job_id={} missing_result_key={}",
                job_id,
                required_key,
            )
            raise HTTPException(status_code=500, detail="Pipeline result is incomplete.")

    concept_ids = set(result["concept_map"].keys())
    invalid_edges = [
        edge for edge in result["prereq_edges"]
        if edge.get("source") not in concept_ids or edge.get("target") not in concept_ids
    ]
    if invalid_edges:
        logger.warning(
            "[PipelineRouter] job_id={} invalid_edge_count={}",
            job_id,
            len(invalid_edges),
        )
        raise HTTPException(
            status_code=409,
            detail="Pipeline graph validation failed. Please reprocess.",
        )

    session = await manager.create_session_from_pipeline(
        concepts_data=result["concepts_data"],
        concept_map=result["concept_map"],
        prereq_edges=result["prereq_edges"],
        max_steps=max_steps,
        precomputed_embeddings=result.get("concept_embeddings"),
        job_id=job_id,
        user_id=user_id,
    )

    # Fire eager prefetch
    try:
        background_tasks.add_task(exercise_svc.eager_generate_first_exercise, session)
    except TypeError as exc:
        logger.warning("[PipelineRouter] Failed to schedule eager prefetch: {}", exc)

    return {
        "session_id": session.session_id,
        "n_concepts": len(result["concept_map"]),
        "source": "new_session",
        "job_id": job_id,
        "status": "active",
    }
