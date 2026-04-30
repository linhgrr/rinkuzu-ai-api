"""
history.py — Endpoints for querying persisted subject progress and pipeline jobs from MongoDB.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from api.config import get_settings
from api.core.shared import mongo_store
from api.dependencies import get_current_user
from api.exceptions import PipelineNotFoundError
from api.schemas import (
    SubjectHistoryDetailResponse,
    SubjectHistoryListResponse,
    SubjectProgressListResponse,
)

router = APIRouter(prefix="/api/history", tags=["history"])
_MASTERED_THRESHOLD = float(get_settings().adaptive_mastery_threshold)


def _to_progress_percent(mastered_concept: int, all_concept: int) -> int:
    if all_concept <= 0:
        return 0
    return max(0, min(100, round((mastered_concept / all_concept) * 100)))


def _count_mastered_concepts(
    concept_mastery: list[float],
    threshold: float = _MASTERED_THRESHOLD,
) -> int:
    return int(sum(1 for value in concept_mastery if value >= threshold))


def _build_subject_progress_detail(
    job_doc: dict[str, Any],
    progress_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    result = job_doc.get("result") or {}
    concept_map = result.get("concept_map") or {}
    concept_names = (progress_doc or {}).get("concept_names") or {
        str(cid): (cdata or {}).get("name", str(cid))
        for cid, cdata in (result.get("concepts_data") or {}).items()
    }
    concept_count = len(concept_map)

    if progress_doc:
        return {
            "job_id": job_doc["job_id"],
            "filename": job_doc.get("filename", ""),
            "subject_id": job_doc.get("subject_id", ""),
            "status": progress_doc.get("status", "active"),
            "total_correct": progress_doc.get("total_correct", 0),
            "total_answered": progress_doc.get("total_answered", 0),
            "accuracy": progress_doc.get("accuracy", 0.0),
            "step": progress_doc.get("step", 0),
            "max_steps": progress_doc.get("max_steps", 9999),
            "avg_mastery": progress_doc.get("avg_mastery", 0.0),
            "concept_names": concept_names,
            "concept_mastery": progress_doc.get("concept_mastery", []),
            "bloom_mastery": progress_doc.get("bloom_mastery", []),
            "exercise_history": progress_doc.get("exercise_history", []),
            "created_at": progress_doc.get("created_at", job_doc.get("completed_at", 0)),
            "updated_at": progress_doc.get("updated_at", job_doc.get("completed_at", 0)),
            "last_session_id": progress_doc.get("last_session_id"),
        }

    return {
        "job_id": job_doc["job_id"],
        "filename": job_doc.get("filename", ""),
        "subject_id": job_doc.get("subject_id", ""),
        "status": "not_started",
        "total_correct": 0,
        "total_answered": 0,
        "accuracy": 0.0,
        "step": 0,
        "max_steps": 9999,
        "avg_mastery": 0.0,
        "concept_names": concept_names,
        "concept_mastery": [0.0] * concept_count,
        "bloom_mastery": [[0.0] * 6 for _ in range(concept_count)],
        "exercise_history": [],
        "created_at": job_doc.get("completed_at", 0),
        "updated_at": job_doc.get("completed_at", 0),
        "last_session_id": None,
    }


def _build_subject_progress_summary(
    progress_doc: dict[str, Any],
    job_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "job_id": progress_doc.get("job_id") or "",
        "filename": (job_doc or {}).get("filename", ""),
        "subject_id": (job_doc or {}).get("subject_id", ""),
        "status": progress_doc.get("status", "active"),
        "total_correct": progress_doc.get("total_correct", 0),
        "total_answered": progress_doc.get("total_answered", 0),
        "accuracy": progress_doc.get("accuracy", 0.0),
        "avg_mastery": progress_doc.get("avg_mastery", 0.0),
        "step": progress_doc.get("step", 0),
        "max_steps": progress_doc.get("max_steps", 9999),
        "created_at": progress_doc.get("created_at", 0),
        "updated_at": progress_doc.get("updated_at", 0),
        "last_session_id": progress_doc.get("last_session_id"),
    }


@router.get("/subjects", response_model=SubjectHistoryListResponse)
async def list_subjects(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    user_id: str = Depends(get_current_user),
):
    """List all completed pipeline jobs (= subjects) enriched with mastery stats."""
    subjects = await mongo_store.list_pipeline_jobs(
        limit=limit,
        user_id=user_id,
        status="completed",
    )

    if not subjects:
        return {"subjects": [], "count": 0}

    job_ids = [s["job_id"] for s in subjects]
    progress_map = await mongo_store.load_subject_progress_map(job_ids, user_id)

    for subj in subjects:
        jid = subj["job_id"]
        progress_doc = progress_map.get(jid, {})
        concept_mastery = progress_doc.get("concept_mastery") or []
        all_c = (
            len(concept_mastery)
            or subj.get("concepts_after_merge")
            or subj.get("concepts_extracted")
            or 0
        )
        mastered_c = _count_mastered_concepts(concept_mastery)

        subj["all_concept"] = all_c
        subj["mastered_concept"] = mastered_c
        subj["progress_percent"] = _to_progress_percent(mastered_c, all_c)

    return {"subjects": subjects, "count": len(subjects)}


@router.get("/subjects/progress", response_model=SubjectProgressListResponse)
async def list_subject_progress(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    user_id: str = Depends(get_current_user),
):
    """List recent subject-level progress records."""
    progress_docs = await mongo_store.list_subject_progress(limit=limit, user_id=user_id)
    job_ids = [doc.get("job_id") for doc in progress_docs if doc.get("job_id")]
    job_map = await mongo_store.load_pipeline_job_map_for_user(
        job_ids,
        user_id,
        projection={"_id": 0, "job_id": 1, "filename": 1, "subject_id": 1},
    )
    items = []
    for progress_doc in progress_docs:
        job_id = progress_doc.get("job_id")
        if not job_id:
            continue
        items.append(_build_subject_progress_summary(progress_doc, job_map.get(job_id)))
    return {"subjects": items, "count": len(items)}


@router.get("/subjects/{job_id}", response_model=SubjectHistoryDetailResponse)
async def get_subject_history(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
):
    """Get subject-level learning history for one pipeline job."""
    job_doc = await mongo_store.load_pipeline_job_for_user(job_id, user_id)
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    progress_doc = await mongo_store.load_subject_progress_for_user(job_id, user_id)
    return _build_subject_progress_detail(job_doc, progress_doc)


@router.get("/pipeline-jobs")
async def list_pipeline_jobs(
    limit: Annotated[int, Query(ge=1, le=500)] = 20,
    user_id: str = Depends(get_current_user),
):
    """List recent pipeline jobs."""
    jobs = await mongo_store.list_pipeline_jobs(limit=limit, user_id=user_id)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/pipeline-jobs/{job_id}")
async def get_pipeline_job(job_id: str, user_id: Annotated[str, Depends(get_current_user)]):
    """Get full pipeline job result."""
    doc = await mongo_store.load_pipeline_job_for_user(job_id, user_id)
    if not doc:
        raise PipelineNotFoundError(job_id)
    return doc


@router.delete("/subjects/{job_id}")
async def delete_subject(
    job_id: str,
    delete_sessions: bool = True,
    user_id: str = Depends(get_current_user),
):
    """Delete a subject (pipeline job) and optionally its sessions."""
    result = await mongo_store.delete_pipeline_job_for_user(
        job_id=job_id,
        user_id=user_id,
        delete_sessions=delete_sessions,
    )
    if result.get("deleted_job", 0) == 0:
        raise PipelineNotFoundError(job_id)

    return {
        "job_id": job_id,
        "deleted_job": result.get("deleted_job", 0),
        "deleted_sessions": result.get("deleted_sessions", 0),
        "status": "deleted",
    }
