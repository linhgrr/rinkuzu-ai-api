"""
routers/pipeline.py — Content pipeline endpoints.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from ..core.content_pipeline import (
    process_pdf,
    get_job,
    PipelineStatus,
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
)
from ..core import mongo_store
from ..dependencies import get_session_manager, get_current_user
from ..exceptions import (
    ServiceUnavailableError,
    PipelineNotFoundError,
    PipelineNotCompletedError,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.get("/status")
async def pipeline_status():
    """Check if content processor is available."""
    import sys
    return {
        "available": CONTENT_PROCESSOR_AVAILABLE,
        "message": (
            "Content processor ready"
            if CONTENT_PROCESSOR_AVAILABLE
            else f"Import error: {CONTENT_PROCESSOR_ERROR}"
        ),
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
):
    """Upload a PDF and run the full content processing pipeline."""
    if not CONTENT_PROCESSOR_AVAILABLE:
        raise ServiceUnavailableError("Content processor")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    job = await process_pdf(
        file_path=str(save_path),
        subject_id=subject_id,
        prs_threshold=prs_threshold,
        min_confidence=min_confidence,
        apply_reduction=apply_reduction,
        user_id=user_id,
    )

    return {
        "job_id": job.job_id,
        "filename": file.filename,
        "file_size": os.path.getsize(save_path),
        "subject_id": job.subject_id,
        "status": job.status.value,
        "message": "Processing started. Poll /api/pipeline/jobs/{job_id} for progress.",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get pipeline job status and progress."""
    job = get_job(job_id)
    if job:
        response = {
            "job_id": job.job_id,
            "filename": job.filename,
            "subject_id": job.subject_id,
            "status": job.status.value,
            "current_step": job.current_step,
            "progress": job.progress,
            "total_chunks": job.total_chunks,
            "concepts_extracted": job.concepts_extracted,
            "concepts_after_merge": job.concepts_after_merge,
            "relations_verified": job.relations_verified,
            "graph_stats": job.graph_stats,
            "error_message": job.error_message,
            "partial_graph": job.partial_graph,
        }
        if job.status == PipelineStatus.COMPLETED and job.result:
            response["result"] = {
                "graph": job.result["graph"],
                "stats": job.result["stats"],
                "n_concepts": len(job.result["concept_map"]),
            }
        return response

    # Fallback to MongoDB
    job_doc = await mongo_store.load_pipeline_job(job_id)
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    result = job_doc.get("result") or {}
    response = {
        "job_id": job_doc.get("job_id", job_id),
        "filename": job_doc.get("filename", ""),
        "subject_id": job_doc.get("subject_id", ""),
        "status": job_doc.get("status", "unknown"),
        "current_step": "Loaded from MongoDB",
        "progress": 1.0 if job_doc.get("status") == "completed" else 0.0,
        "total_chunks": job_doc.get("total_chunks", 0),
        "concepts_extracted": job_doc.get("concepts_extracted", 0),
        "concepts_after_merge": job_doc.get("concepts_after_merge", 0),
        "relations_verified": job_doc.get("relations_verified", 0),
        "graph_stats": job_doc.get("graph_stats", {}),
        "error_message": None,
        "partial_graph": None,
    }
    if job_doc.get("status") == "completed" and result:
        response["result"] = {
            "graph": result.get("graph", {"nodes": [], "edges": []}),
            "stats": result.get("stats", {}),
            "n_concepts": len(result.get("concept_map", {})),
        }
    return response


@router.post("/jobs/{job_id}/create-session")
async def create_session_from_pipeline(
    job_id: str,
    max_steps: int = 50,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    """Create a learning session from a completed pipeline job."""
    job_doc = await mongo_store.load_pipeline_job(job_id)
    if not job_doc:
        mem_job = get_job(job_id)
        if mem_job and mem_job.status != PipelineStatus.COMPLETED:
            raise PipelineNotCompletedError(job_id, mem_job.status.value)
        if mem_job and mem_job.status == PipelineStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail="Job completed in memory but not persisted to MongoDB yet. Please retry shortly.",
            )
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
        import asyncio
        exercise_service = getattr(manager, '_exercise_service', None)
        if exercise_service:
            asyncio.create_task(exercise_service.eager_generate_first_exercise(session))
    except Exception:
        pass

    return {
        "session_id": session.session_id,
        "n_concepts": len(result["concept_map"]),
        "source": "mongodb",
        "job_id": job_id,
        "status": "active",
    }
