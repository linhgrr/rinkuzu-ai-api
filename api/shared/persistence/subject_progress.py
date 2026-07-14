from __future__ import annotations

from typing import TYPE_CHECKING, Any

from beanie.odm.enums import SortDirection
from loguru import logger

from .common import epoch_to_utc, normalize_for_bson, utc_to_epoch

if TYPE_CHECKING:
    from pymongo.asynchronous.client_session import AsyncClientSession

from .documents import (
    BloomMasteryEntry,
    ConceptMasteryEntry,
    ExerciseEntry,
    SubjectProgressDocument,
    SubjectProgressSummaryProjection,
)


def _resolve_concept_indices(snapshot: dict[str, Any]) -> dict[str, int]:
    raw = snapshot.get("concept_indices")
    if isinstance(raw, dict) and raw:
        return {str(key): int(value) for key, value in raw.items()}

    concept_names = snapshot.get("concept_names") or {}
    if isinstance(concept_names, dict) and concept_names:
        return {str(concept_id): idx for idx, concept_id in enumerate(concept_names.keys())}

    concept_mastery = snapshot.get("concept_mastery") or []
    return {str(idx): idx for idx in range(len(concept_mastery))}


def _snapshot_to_document_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    concept_indices = _resolve_concept_indices(snapshot)
    inverse = {index: concept_id for concept_id, index in concept_indices.items()}

    concept_mastery_values = snapshot.get("concept_mastery") or []
    concept_mastery = {
        inverse[idx]: ConceptMasteryEntry(concept_idx=idx, mastery=float(value))
        for idx, value in enumerate(concept_mastery_values)
        if idx in inverse
    }

    bloom_mastery_rows = snapshot.get("bloom_mastery") or []
    bloom_mastery = {
        inverse[idx]: BloomMasteryEntry(concept_idx=idx, levels=[float(item) for item in row])
        for idx, row in enumerate(bloom_mastery_rows)
        if idx in inverse
    }

    exercise_history = [
        _exercise_entry_from_payload(entry) for entry in snapshot.get("exercise_history") or []
    ]

    current_exercise = None
    if snapshot.get("current_exercise"):
        current_exercise = _exercise_entry_from_payload(snapshot["current_exercise"])

    return {
        "job_id": str(snapshot.get("job_id") or ""),
        "user_id": str(snapshot.get("user_id") or ""),
        "last_session_id": snapshot.get("last_session_id"),
        "status": str(snapshot.get("status") or "active"),
        "total_correct": int(snapshot.get("total_correct", 0) or 0),
        "total_answered": int(snapshot.get("total_answered", 0) or 0),
        "accuracy": float(snapshot.get("accuracy", 0.0) or 0.0),
        "step": int(snapshot.get("step", 0) or 0),
        "max_steps": int(snapshot.get("max_steps", 9999) or 9999),
        "avg_mastery": float(snapshot.get("avg_mastery", 0.0) or 0.0),
        "unlocked_concepts": int(snapshot.get("unlocked_concepts", 0) or 0),
        "locked_concepts": int(snapshot.get("locked_concepts", 0) or 0),
        "mastered_concepts": int(snapshot.get("mastered_concepts", 0) or 0),
        "progress_percent": int(snapshot.get("progress_percent", 0) or 0),
        "concept_names": {
            str(key): str(value) for key, value in (snapshot.get("concept_names") or {}).items()
        },
        "concept_indices": concept_indices,
        "concept_mastery": concept_mastery,
        "bloom_mastery": bloom_mastery,
        "exercise_history": exercise_history,
        "current_exercise": current_exercise,
        "pending_concept_idx": snapshot.get("pending_concept_idx"),
        "pending_bloom_level": snapshot.get("pending_bloom_level"),
        "pending_action": snapshot.get("pending_action"),
        "recommendation_reason": normalize_for_bson(snapshot.get("recommendation_reason")),
        "submission_receipts": normalize_for_bson(snapshot.get("submission_receipts") or {}),
        "version": int(snapshot.get("version", 0) or 0),
        "created_at": epoch_to_utc(snapshot.get("created_at")),
        "updated_at": epoch_to_utc(snapshot.get("updated_at")),
    }


def _exercise_entry_from_payload(entry: dict[str, Any]) -> ExerciseEntry:
    payload = dict(entry)
    payload["timestamp"] = epoch_to_utc(payload.get("timestamp"))
    payload["payload"] = normalize_for_bson(payload.get("payload") or {})
    payload["theory"] = normalize_for_bson(payload.get("theory"))
    return ExerciseEntry(**payload)


def _exercise_entry_to_payload(entry: ExerciseEntry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    payload = entry.model_dump()
    payload["timestamp"] = utc_to_epoch(entry.timestamp)
    return payload


def _document_to_legacy_payload(doc: SubjectProgressDocument) -> dict[str, Any]:
    ordered_concepts = sorted(
        doc.concept_indices.items(),
        key=lambda item: item[1],
    )
    concept_mastery = [
        float(
            doc.concept_mastery.get(
                concept_id, ConceptMasteryEntry(concept_idx=idx, mastery=0.0)
            ).mastery
        )
        for concept_id, idx in ordered_concepts
    ]
    bloom_mastery = [
        list(
            doc.bloom_mastery.get(
                concept_id, BloomMasteryEntry(concept_idx=idx, levels=[0.0] * 6)
            ).levels
        )
        for concept_id, idx in ordered_concepts
    ]
    exercise_history = [
        payload
        for entry in doc.exercise_history
        if (payload := _exercise_entry_to_payload(entry)) is not None
    ]
    return {
        "job_id": doc.job_id,
        "user_id": doc.user_id,
        "last_session_id": doc.last_session_id,
        "status": doc.status,
        "total_correct": doc.total_correct,
        "total_answered": doc.total_answered,
        "accuracy": doc.accuracy,
        "step": doc.step,
        "max_steps": doc.max_steps,
        "avg_mastery": doc.avg_mastery,
        "unlocked_concepts": doc.unlocked_concepts,
        "locked_concepts": doc.locked_concepts,
        "mastered_concepts": doc.mastered_concepts,
        "progress_percent": doc.progress_percent,
        "concept_names": dict(doc.concept_names),
        "concept_indices": dict(doc.concept_indices),
        "concept_mastery": concept_mastery,
        "bloom_mastery": bloom_mastery,
        "exercise_history": exercise_history,
        "current_exercise": _exercise_entry_to_payload(doc.current_exercise),
        "pending_concept_idx": doc.pending_concept_idx,
        "pending_bloom_level": doc.pending_bloom_level,
        "pending_action": doc.pending_action,
        "recommendation_reason": doc.recommendation_reason,
        "submission_receipts": dict(doc.submission_receipts),
        "version": doc.version,
        "created_at": utc_to_epoch(doc.created_at),
        "updated_at": utc_to_epoch(doc.updated_at),
    }


async def save_subject_progress_snapshot(
    job_id: str, user_id: str, snapshot: dict[str, Any]
) -> bool:
    try:
        payload = _snapshot_to_document_payload(snapshot)
        existing = await SubjectProgressDocument.find_one(
            SubjectProgressDocument.job_id == job_id,
            SubjectProgressDocument.user_id == user_id,
        )
        if existing is None:
            await SubjectProgressDocument(**payload).insert()
        else:
            original_created_at = existing.created_at
            payload["version"] = existing.version + 1
            for key, value in payload.items():
                setattr(existing, key, value)
            existing.created_at = original_created_at
            await existing.replace()
    except Exception:
        logger.exception(
            "[SubjectProgressStore] save_snapshot failed job_id={} user_id={}", job_id, user_id
        )
        return False
    return True


async def load_subject_progress_for_user(job_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        doc = await SubjectProgressDocument.find_one(
            SubjectProgressDocument.job_id == job_id,
            SubjectProgressDocument.user_id == user_id,
        )
    except Exception:
        logger.exception(
            "[SubjectProgressStore] load_for_user failed job_id={} user_id={}", job_id, user_id
        )
        return None
    return None if doc is None else _document_to_legacy_payload(doc)


async def load_subject_progress_by_session_for_user(
    session_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    try:
        doc = await SubjectProgressDocument.find_one(
            SubjectProgressDocument.last_session_id == session_id,
            SubjectProgressDocument.user_id == user_id,
        )
    except Exception:
        logger.exception(
            "[SubjectProgressStore] load_by_session_for_user failed session_id={} user_id={}",
            session_id,
            user_id,
        )
        return None
    return None if doc is None else _document_to_legacy_payload(doc)


async def load_many_subject_progress_for_user(
    job_ids: list[str],
    user_id: str,
) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}
    try:
        docs = await SubjectProgressDocument.find(
            {"job_id": {"$in": job_ids}, "user_id": user_id}
        ).to_list()
    except Exception:
        logger.exception("[SubjectProgressStore] load_many_for_user failed user_id={}", user_id)
        return {}
    return {doc.job_id: _document_to_legacy_payload(doc) for doc in docs}


async def list_recent_subject_progress(
    *,
    limit: int = 50,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    filters: list[Any] = []
    if user_id is not None:
        filters.append(SubjectProgressDocument.user_id == user_id)
    try:
        rows = await (
            SubjectProgressDocument.find(
                *filters,
                projection_model=SubjectProgressSummaryProjection,
            )
            .sort(("updated_at", SortDirection.DESCENDING))
            .limit(limit)
            .to_list()
        )
    except Exception:
        logger.exception("[SubjectProgressStore] list_recent failed")
        return []
    return [
        {
            "job_id": row.job_id,
            "last_session_id": row.last_session_id,
            "status": row.status,
            "total_correct": row.total_correct,
            "total_answered": row.total_answered,
            "accuracy": row.accuracy,
            "avg_mastery": row.avg_mastery,
            "unlocked_concepts": row.unlocked_concepts,
            "locked_concepts": row.locked_concepts,
            "mastered_concepts": row.mastered_concepts,
            "progress_percent": row.progress_percent,
            "step": row.step,
            "max_steps": row.max_steps,
            "created_at": utc_to_epoch(row.created_at),
            "updated_at": utc_to_epoch(row.updated_at),
        }
        for row in rows
    ]


async def delete_subject_progress_for_user(
    job_id: str,
    user_id: str,
    *,
    session: AsyncClientSession | None = None,
) -> int:
    result = await SubjectProgressDocument.find(
        SubjectProgressDocument.job_id == job_id,
        SubjectProgressDocument.user_id == user_id,
        session=session,
    ).delete(session=session)
    return 0 if result is None else int(result.deleted_count)
