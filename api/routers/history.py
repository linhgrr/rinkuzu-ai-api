"""
history.py — Endpoints for querying persisted subject progress and pipeline jobs from MongoDB.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from api.config import get_settings
from api.core.learning.progress_metrics import (
    build_prereq_graph_from_edges,
    compute_unlocked_mask,
    summarize_mastery_progress,
)
from api.core.shared.persistence import (
    delete_pipeline_job_for_user,
    list_recent_pipeline_jobs,
    list_recent_subject_progress,
    load_many_pipeline_jobs_for_user,
    load_many_subject_progress_for_user,
    load_pipeline_job_for_user,
    load_subject_progress_for_user,
)
from api.dependencies import get_current_user
from api.exceptions import PipelineNotFoundError
from api.rate_limit import is_admin_request, limiter
from api.schemas import (
    DeleteSubjectResponse,
    PipelineJobHistoryListResponse,
    SubjectHistoryDetailResponse,
    SubjectHistoryListResponse,
    SubjectProgressListResponse,
)
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID

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


def _coerce_concept_map(
    progress_doc: dict[str, Any] | None,
    job_doc: dict[str, Any] | None,
    concept_count: int,
) -> dict[str, int]:
    result = (job_doc or {}).get("result") or {}
    raw_map = result.get("concept_map") or (progress_doc or {}).get("concept_indices") or {}
    if isinstance(raw_map, dict) and raw_map:
        return {str(concept_id): int(index) for concept_id, index in raw_map.items()}
    return {str(idx): idx for idx in range(concept_count)}


def _build_unlocked_progress_metrics(
    progress_doc: dict[str, Any] | None,
    job_doc: dict[str, Any] | None,
) -> dict[str, int | float]:
    result = (job_doc or {}).get("result") or {}
    concept_mastery = list((progress_doc or {}).get("concept_mastery") or [])
    concept_count = (
        len(concept_mastery)
        or len(result.get("concept_map") or {})
        or int((job_doc or {}).get("concepts_after_merge") or 0)
        or int((job_doc or {}).get("concepts_extracted") or 0)
    )
    if not concept_mastery:
        concept_mastery = [0.0] * concept_count

    bloom_mastery = (progress_doc or {}).get("bloom_mastery") or [
        [0.0] * 6 for _ in range(concept_count)
    ]
    concept_map = _coerce_concept_map(progress_doc, job_doc, concept_count)
    prereq_graph = build_prereq_graph_from_edges(result.get("prereq_edges") or [], concept_map)
    unlocked_mask = compute_unlocked_mask(
        concept_count=concept_count,
        bloom_mastery=bloom_mastery,
        prereq_graph=prereq_graph,
        threshold=_MASTERED_THRESHOLD,
    )
    return summarize_mastery_progress(
        concept_mastery=concept_mastery,
        unlocked_mask=unlocked_mask,
        threshold=_MASTERED_THRESHOLD,
    )


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
    progress_metrics = _build_unlocked_progress_metrics(progress_doc, job_doc)

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
            "avg_mastery": progress_metrics["avg_mastery"],
            "unlocked_concepts": progress_metrics["unlocked_concepts"],
            "locked_concepts": progress_metrics["locked_concepts"],
            "mastered_concepts": progress_metrics["mastered_concepts"],
            "progress_percent": progress_metrics["progress_percent"],
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
        "avg_mastery": progress_metrics["avg_mastery"],
        "unlocked_concepts": progress_metrics["unlocked_concepts"],
        "locked_concepts": progress_metrics["locked_concepts"],
        "mastered_concepts": progress_metrics["mastered_concepts"],
        "progress_percent": progress_metrics["progress_percent"],
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
    progress_metrics = _build_unlocked_progress_metrics(progress_doc, job_doc)
    return {
        "job_id": progress_doc.get("job_id") or "",
        "filename": (job_doc or {}).get("filename", ""),
        "subject_id": (job_doc or {}).get("subject_id", ""),
        "status": progress_doc.get("status", "active"),
        "total_correct": progress_doc.get("total_correct", 0),
        "total_answered": progress_doc.get("total_answered", 0),
        "accuracy": progress_doc.get("accuracy", 0.0),
        "avg_mastery": progress_metrics["avg_mastery"],
        "unlocked_concepts": progress_metrics["unlocked_concepts"],
        "locked_concepts": progress_metrics["locked_concepts"],
        "mastered_concepts": progress_metrics["mastered_concepts"],
        "progress_percent": progress_metrics["progress_percent"],
        "step": progress_doc.get("step", 0),
        "max_steps": progress_doc.get("max_steps", 9999),
        "created_at": progress_doc.get("created_at", 0),
        "updated_at": progress_doc.get("updated_at", 0),
        "last_session_id": progress_doc.get("last_session_id"),
    }


@router.get("/subjects", response_model=StandardResponse[SubjectHistoryListResponse])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def list_subjects(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    user_id: str = Depends(get_current_user),
) -> Any:
    del request
    """List all completed pipeline jobs (= subjects) enriched with mastery stats."""
    subjects = await list_recent_pipeline_jobs(
        limit=limit,
        user_id=user_id,
        status="completed",
    )

    if not subjects:
        return ok({"subjects": [], "count": 0})

    job_ids = [s["job_id"] for s in subjects]
    progress_map = await load_many_subject_progress_for_user(job_ids, user_id)
    full_job_map = await load_many_pipeline_jobs_for_user(job_ids, user_id)

    for subj in subjects:
        jid = subj["job_id"]
        progress_doc = progress_map.get(jid, {})
        progress_metrics = _build_unlocked_progress_metrics(
            progress_doc, full_job_map.get(jid, subj)
        )

        subj["all_concept"] = progress_metrics["total_concepts"]
        subj["unlocked_concept"] = progress_metrics["unlocked_concepts"]
        subj["locked_concept"] = progress_metrics["locked_concepts"]
        subj["mastered_concept"] = progress_metrics["mastered_concepts"]
        subj["progress_percent"] = progress_metrics["progress_percent"]

    return ok({"subjects": subjects, "count": len(subjects)})


@router.get("/subjects/progress", response_model=StandardResponse[SubjectProgressListResponse])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def list_subject_progress(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    user_id: str = Depends(get_current_user),
) -> Any:
    del request
    """List recent subject-level progress records."""
    progress_docs = await list_recent_subject_progress(limit=limit, user_id=user_id)
    job_ids = [job_id for doc in progress_docs if isinstance((job_id := doc.get("job_id")), str)]
    job_map = await load_many_pipeline_jobs_for_user(job_ids, user_id)
    items = []
    for progress_doc in progress_docs:
        job_id = progress_doc.get("job_id")
        if not job_id:
            continue
        items.append(_build_subject_progress_summary(progress_doc, job_map.get(job_id)))
    return ok({"subjects": items, "count": len(items)})


@router.get("/subjects/{job_id}", response_model=StandardResponse[SubjectHistoryDetailResponse])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def get_subject_history(
    request: Request,
    job_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    del request
    """Get subject-level learning history for one pipeline job."""
    job_doc = await load_pipeline_job_for_user(job_id, user_id)
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    progress_doc = await load_subject_progress_for_user(job_id, user_id)
    return ok(_build_subject_progress_detail(job_doc, progress_doc))


@router.get("/pipeline-jobs", response_model=StandardResponse[PipelineJobHistoryListResponse])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def list_pipeline_jobs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 20,
    user_id: str = Depends(get_current_user),
) -> Any:
    del request
    """List recent pipeline jobs."""
    jobs = await list_recent_pipeline_jobs(limit=limit, user_id=user_id)
    return ok({"jobs": jobs, "count": len(jobs)})


@router.get("/pipeline-jobs/{job_id}", response_model=StandardResponse[dict])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def get_pipeline_job(
    request: Request,
    job_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    del request
    """Get full pipeline job result."""
    doc = await load_pipeline_job_for_user(job_id, user_id)
    if not doc:
        raise PipelineNotFoundError(job_id)
    return ok(doc)


@router.delete("/subjects/{job_id}", response_model=StandardResponse[DeleteSubjectResponse])
@limiter.limit(get_settings().rate_limit_history, exempt_when=is_admin_request)
async def delete_subject(
    request: Request,
    job_id: PathID,
    user_id: str = Depends(get_current_user),
    *,
    delete_sessions: bool = True,
) -> Any:
    del request
    """Delete a subject (pipeline job) and optionally its sessions."""
    result = await delete_pipeline_job_for_user(
        job_id=job_id,
        user_id=user_id,
        delete_sessions=delete_sessions,
    )
    if result.get("deleted_job", 0) == 0:
        raise PipelineNotFoundError(job_id)

    return ok(
        {
            "job_id": job_id,
            "deleted_job": result.get("deleted_job", 0),
            "deleted_sessions": result.get("deleted_sessions", 0),
            "status": "deleted",
        }
    )
