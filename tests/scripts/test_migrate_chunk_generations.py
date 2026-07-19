from __future__ import annotations

from bson import ObjectId
import pytest
from scripts.migrate_chunk_generations import (
    DESIRED_INDEXES,
    LEGACY_INDEXES,
    build_migration_plan,
    verify_plan_is_clean,
)


def _legacy_indexes():
    return {
        "_id_": {"key": [("_id", 1)]},
        "job_id_1_chunk_index_1": {
            "key": list(LEGACY_INDEXES["job_id_1_chunk_index_1"]),
            "unique": True,
        },
        "subject_id_1_job_id_1": {
            "key": list(LEGACY_INDEXES["subject_id_1_job_id_1"]),
        },
    }


def _desired_indexes():
    indexes = {"_id_": {"key": [("_id", 1)]}}
    for name, (keys, unique) in DESIRED_INDEXES.items():
        indexes[name] = {"key": list(keys), "unique": unique}
    return indexes


def test_plan_backfills_owned_artifacts_and_purges_only_orphans():
    owned_id = ObjectId()
    orphan_id = ObjectId()
    plan = build_migration_plan(
        legacy_chunk_documents=[
            {"_id": owned_id, "job_id": "job-1"},
            {"_id": orphan_id, "job_id": "missing"},
        ],
        job_generations={"job-1": 2},
        chroma_records=[
            ("owned-vector", {"job_id": "job-1", "chunk_index": 0}),
            ("orphan-vector", {"job_id": "missing", "chunk_index": 0}),
        ],
        indexes=_legacy_indexes(),
    )

    assert [(item.document_id, item.generation) for item in plan.mongo_backfills] == [(owned_id, 2)]
    assert plan.mongo_orphan_ids == (orphan_id,)
    assert plan.chroma_backfills[0].metadata["generation"] == 2
    assert plan.chroma_orphan_ids == ("orphan-vector",)
    assert plan.drop_indexes == tuple(sorted(LEGACY_INDEXES))
    assert plan.create_indexes == tuple(sorted(DESIRED_INDEXES))


def test_plan_is_deterministic_for_reordered_sources():
    documents = [
        {"_id": ObjectId("000000000000000000000002"), "job_id": "job-2"},
        {"_id": ObjectId("000000000000000000000001"), "job_id": "job-1"},
    ]
    records = [
        ("vector-2", {"job_id": "job-2"}),
        ("vector-1", {"job_id": "job-1"}),
    ]
    first = build_migration_plan(
        legacy_chunk_documents=documents,
        job_generations={"job-1": 0, "job-2": 1},
        chroma_records=records,
        indexes=_legacy_indexes(),
    )
    second = build_migration_plan(
        legacy_chunk_documents=list(reversed(documents)),
        job_generations={"job-2": 1, "job-1": 0},
        chroma_records=list(reversed(records)),
        indexes=dict(reversed(list(_legacy_indexes().items()))),
    )
    assert first.source_fingerprint == second.source_fingerprint
    assert first.plan_sha == second.plan_sha


def test_clean_plan_is_idempotent_and_verifiable():
    plan = build_migration_plan(
        legacy_chunk_documents=[],
        job_generations={},
        chroma_records=[],
        indexes=_desired_indexes(),
    )
    verify_plan_is_clean(plan)
    assert plan.drop_indexes == ()
    assert plan.create_indexes == ()


def test_verify_rejects_pending_destructive_or_schema_operations():
    plan = build_migration_plan(
        legacy_chunk_documents=[],
        job_generations={},
        chroma_records=[],
        indexes=_legacy_indexes(),
    )
    with pytest.raises(RuntimeError, match="postcondition failed"):
        verify_plan_is_clean(plan)


def test_noncanonical_desired_index_fails_closed():
    indexes = _desired_indexes()
    indexes["job_id_1_generation_1_chunk_index_1"]["unique"] = False
    with pytest.raises(RuntimeError, match="noncanonical"):
        build_migration_plan(
            legacy_chunk_documents=[],
            job_generations={},
            chroma_records=[],
            indexes=indexes,
        )
