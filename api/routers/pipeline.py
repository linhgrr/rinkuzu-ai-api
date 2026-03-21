"""
routers/pipeline.py — Content pipeline endpoints.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from loguru import logger
import aiofiles

from ..core.content_pipeline import (
    PipelineStatus,
)
from ..core import mongo_store
from ..dependencies import (
    get_content_pipeline_availability,
    get_content_pipeline_service,
    get_current_user,
    get_session_manager,
    get_session_service,
)
from ..exceptions import (
    ServiceUnavailableError,
    PipelineNotFoundError,
    PipelineNotCompletedError,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.get("/status")
async def pipeline_status(availability: dict = Depends(get_content_pipeline_availability)):
    """Check if content pipeline runtime modules are available."""
    import sys
    return {
        "available": availability["available"],
        "error": availability["error"],
        "service_initialized": availability["service_initialized"],
        "message": (
            "Content pipeline ready"
            if availability["available"]
            else f"Import error: {availability['error']}"
        ),
        "content_processor_src": availability["src"],
        "sys_path": sys.path,
    }


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    subject_id: Optional[str] = Form(None),
    prs_threshold: float = Form(0.75),
    min_confidence: float = Form(0.6),
    apply_reduction: bool = Form(True),
    user_id: str = Depends(get_current_user),
    availability: dict = Depends(get_content_pipeline_availability),
    pipeline_service=Depends(get_content_pipeline_service),
):
    """Upload a PDF and run the full content processing pipeline."""
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
            detail=(
                f"ContentPipeline unavailable: {availability['error']}"
                if availability["error"]
                else "ContentPipeline unavailable: unknown startup error"
            ),
        )
    if not mongo_store.is_available():
        raise ServiceUnavailableError("MongoDB persistence")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        job = await pipeline_service.start_job(
            file_path=str(save_path),
            subject_id=subject_id,
            prs_threshold=prs_threshold,
            min_confidence=min_confidence,
            apply_reduction=apply_reduction,
            user_id=user_id,
            content_processor_available=availability["available"],
            content_processor_src=availability["src"] or "",
        )
    except Exception as exc:
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            logger.warning(f"[PipelineRouter] Failed to cleanup upload {save_path}")
        raise HTTPException(status_code=503, detail=f"Failed to initialize pipeline job: {exc}")

    return {
        "job_id": job.job_id,
        "filename": file.filename,
        "file_size": os.path.getsize(save_path),
        "subject_id": job.subject_id,
        "status": job.status.value,
        "message": "Processing started. Poll /api/pipeline/jobs/{job_id} for progress.",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(get_current_user)):
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
            raise HTTPException(status_code=500, detail=f"Result missing key: {required_key}")

    concept_ids = set(result["concept_map"].keys())
    invalid_edges = [
        edge for edge in result["prereq_edges"]
        if edge.get("source") not in concept_ids or edge.get("target") not in concept_ids
    ]
    if invalid_edges:
        raise HTTPException(
            status_code=409,
            detail=f"Graph has {len(invalid_edges)} invalid edges. Please reprocess.",
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
    except Exception as exc:
        logger.warning(f"[PipelineRouter] Failed to schedule eager prefetch: {exc}")

    return {
        "session_id": session.session_id,
        "n_concepts": len(result["concept_map"]),
        "source": "new_session",
        "job_id": job_id,
        "status": "active",
    }
