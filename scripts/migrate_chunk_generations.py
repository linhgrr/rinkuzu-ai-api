"""Migrate document-chunk persistence to generation-scoped storage.

Dry-run (default):
    .venv/bin/python scripts/migrate_chunk_generations.py

Apply an inspected plan:
    .venv/bin/python scripts/migrate_chunk_generations.py \
      --apply --approve-plan=<plan_sha>

Verify postconditions:
    .venv/bin/python scripts/migrate_chunk_generations.py --verify

The migration owns only derived Mongo/Chroma chunk artifacts. It never mutates
pipeline jobs, user records, learning progress, quizzes, or attempts.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import chromadb
from dotenv import dotenv_values
from pymongo import ASCENDING, AsyncMongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_NAME = "adaptive_learning"
CHUNK_COLLECTION = "al_document_chunks"
JOB_COLLECTION = "al_pipeline_jobs"
CHROMA_COLLECTION = "document_chunks"
MIGRATION_VERSION = "chunk-generation-v1"

LEGACY_INDEXES: dict[str, tuple[tuple[str, int], ...]] = {
    "job_id_1_chunk_index_1": (("job_id", ASCENDING), ("chunk_index", ASCENDING)),
    "subject_id_1_job_id_1": (("subject_id", ASCENDING), ("job_id", ASCENDING)),
}
DESIRED_INDEXES: dict[str, tuple[tuple[tuple[str, int], ...], bool]] = {
    "job_id_1_generation_1_chunk_index_1": (
        (("job_id", ASCENDING), ("generation", ASCENDING), ("chunk_index", ASCENDING)),
        True,
    ),
    "subject_id_1_job_id_1_generation_1": (
        (("subject_id", ASCENDING), ("job_id", ASCENDING), ("generation", ASCENDING)),
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class MongoBackfill:
    document_id: Any
    job_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class ChromaBackfill:
    vector_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    mongo_backfills: tuple[MongoBackfill, ...]
    mongo_orphan_ids: tuple[Any, ...]
    chroma_backfills: tuple[ChromaBackfill, ...]
    chroma_orphan_ids: tuple[str, ...]
    drop_indexes: tuple[str, ...]
    create_indexes: tuple[str, ...]
    source_fingerprint: str
    plan_sha: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "migration": MIGRATION_VERSION,
            "mongoBackfillCount": len(self.mongo_backfills),
            "mongoOrphanDeleteCount": len(self.mongo_orphan_ids),
            "chromaBackfillCount": len(self.chroma_backfills),
            "chromaOrphanDeleteCount": len(self.chroma_orphan_ids),
            "dropIndexes": list(self.drop_indexes),
            "createIndexes": list(self.create_indexes),
            "sourceFingerprint": self.source_fingerprint,
            "planSha": self.plan_sha,
        }


def _canonical_index_spec(info: dict[str, Any]) -> tuple[tuple[tuple[str, int], ...], bool]:
    keys = tuple((str(field), int(direction)) for field, direction in info.get("key", []))
    return keys, bool(info.get("unique", False))


def _validate_index_state(indexes: dict[str, dict[str, Any]]) -> None:
    for name, (expected_keys, expected_unique) in DESIRED_INDEXES.items():
        existing = indexes.get(name)
        if existing is None:
            continue
        if _canonical_index_spec(existing) != (expected_keys, expected_unique):
            raise RuntimeError(f"Desired index {name!r} exists with a noncanonical definition")

    for name, expected_keys in LEGACY_INDEXES.items():
        existing = indexes.get(name)
        if existing is None:
            continue
        actual_keys, _unique = _canonical_index_spec(existing)
        if actual_keys != expected_keys:
            raise RuntimeError(f"Legacy index {name!r} exists with an unexpected definition")


def _valid_generation(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def build_migration_plan(
    *,
    legacy_chunk_documents: list[dict[str, Any]],
    job_generations: dict[str, int],
    chroma_records: list[tuple[str, dict[str, Any] | None]],
    indexes: dict[str, dict[str, Any]],
) -> MigrationPlan:
    """Build a deterministic plan from Mongo, Chroma, and index snapshots."""
    _validate_index_state(indexes)

    mongo_backfills: list[MongoBackfill] = []
    mongo_orphans: list[Any] = []
    for document in legacy_chunk_documents:
        job_id = document.get("job_id")
        if not isinstance(job_id, str) or job_id not in job_generations:
            mongo_orphans.append(document["_id"])
            continue
        mongo_backfills.append(
            MongoBackfill(
                document_id=document["_id"],
                job_id=job_id,
                generation=job_generations[job_id],
            )
        )

    chroma_backfills: list[ChromaBackfill] = []
    chroma_orphans: list[str] = []
    for vector_id, source_metadata in chroma_records:
        metadata = dict(source_metadata or {})
        job_id = metadata.get("job_id")
        if not isinstance(job_id, str) or job_id not in job_generations:
            chroma_orphans.append(vector_id)
            continue
        generation = job_generations[job_id]
        if metadata.get("generation") == generation:
            continue
        metadata["generation"] = generation
        chroma_backfills.append(ChromaBackfill(vector_id=vector_id, metadata=metadata))

    drop_indexes = tuple(sorted(name for name in LEGACY_INDEXES if name in indexes))
    create_indexes = tuple(sorted(name for name in DESIRED_INDEXES if name not in indexes))

    source_payload = {
        "mongoBackfills": sorted(
            (str(item.document_id), item.job_id, item.generation) for item in mongo_backfills
        ),
        "mongoOrphans": sorted(str(value) for value in mongo_orphans),
        "chromaBackfills": sorted(
            (item.vector_id, str(item.metadata.get("job_id")), item.metadata["generation"])
            for item in chroma_backfills
        ),
        "chromaOrphans": sorted(chroma_orphans),
        "indexes": {
            name: _canonical_index_spec(info)
            for name, info in sorted(indexes.items())
            if name != "_id_"
        },
    }
    source_fingerprint = sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_payload = {
        "migration": MIGRATION_VERSION,
        "sourceFingerprint": source_fingerprint,
        "dropIndexes": drop_indexes,
        "createIndexes": create_indexes,
        "mongoBackfillCount": len(mongo_backfills),
        "mongoOrphanCount": len(mongo_orphans),
        "chromaBackfillCount": len(chroma_backfills),
        "chromaOrphanCount": len(chroma_orphans),
    }
    plan_sha = sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MigrationPlan(
        mongo_backfills=tuple(sorted(mongo_backfills, key=lambda item: str(item.document_id))),
        mongo_orphan_ids=tuple(sorted(mongo_orphans, key=str)),
        chroma_backfills=tuple(sorted(chroma_backfills, key=lambda item: item.vector_id)),
        chroma_orphan_ids=tuple(sorted(chroma_orphans)),
        drop_indexes=drop_indexes,
        create_indexes=create_indexes,
        source_fingerprint=source_fingerprint,
        plan_sha=plan_sha,
    )


async def _load_job_generations(db: Any, job_ids: set[str]) -> dict[str, int]:
    if not job_ids:
        return {}
    cursor = db[JOB_COLLECTION].find(
        {"job_id": {"$in": sorted(job_ids)}},
        {"_id": 0, "job_id": 1, "retry_count": 1},
    )
    documents = await cursor.to_list()
    generations: dict[str, int] = {}
    for document in documents:
        job_id = document.get("job_id")
        generation = document.get("retry_count", 0)
        if isinstance(job_id, str) and _valid_generation(generation):
            generations[job_id] = generation
    return generations


def _read_chroma_records(collection: Any | None) -> list[tuple[str, dict[str, Any] | None]]:
    if collection is None or collection.count() == 0:
        return []
    payload = collection.get(include=["metadatas"])
    ids = payload.get("ids") or []
    metadatas = payload.get("metadatas") or []
    return [
        (str(vector_id), metadata if isinstance(metadata, dict) else None)
        for vector_id, metadata in zip(ids, metadatas, strict=True)
    ]


async def inspect_migration(db: Any, chroma_collection: Any | None) -> MigrationPlan:
    chunk_collection = db[CHUNK_COLLECTION]
    legacy_documents = await chunk_collection.find(
        {"generation": {"$exists": False}},
        {"_id": 1, "job_id": 1},
    ).to_list()
    chroma_records = _read_chroma_records(chroma_collection)
    job_ids = {
        job_id
        for job_id in [
            *(document.get("job_id") for document in legacy_documents),
            *((metadata or {}).get("job_id") for _vector_id, metadata in chroma_records),
        ]
        if isinstance(job_id, str)
    }
    job_generations = await _load_job_generations(db, job_ids)
    indexes = await chunk_collection.index_information()
    return build_migration_plan(
        legacy_chunk_documents=legacy_documents,
        job_generations=job_generations,
        chroma_records=chroma_records,
        indexes=indexes,
    )


async def apply_migration(db: Any, chroma_collection: Any | None, plan: MigrationPlan) -> None:
    chunk_collection = db[CHUNK_COLLECTION]
    for item in plan.mongo_backfills:
        result = await chunk_collection.update_one(
            {"_id": item.document_id, "job_id": item.job_id, "generation": {"$exists": False}},
            {"$set": {"generation": item.generation}},
        )
        if result.modified_count != 1:
            raise RuntimeError(f"Mongo chunk backfill CAS missed for {item.document_id}")

    if plan.mongo_orphan_ids:
        result = await chunk_collection.delete_many(
            {"_id": {"$in": list(plan.mongo_orphan_ids)}, "generation": {"$exists": False}}
        )
        if result.deleted_count != len(plan.mongo_orphan_ids):
            raise RuntimeError("Mongo orphan delete postcondition failed")

    if chroma_collection is not None:
        if plan.chroma_backfills:
            chroma_collection.update(
                ids=[item.vector_id for item in plan.chroma_backfills],
                metadatas=[item.metadata for item in plan.chroma_backfills],
            )
        if plan.chroma_orphan_ids:
            chroma_collection.delete(ids=list(plan.chroma_orphan_ids))

    # Build replacement indexes before dropping legacy indexes. Backfilled data
    # satisfies both contracts during the short migration window.
    for name in plan.create_indexes:
        keys, unique = DESIRED_INDEXES[name]
        await chunk_collection.create_index(list(keys), name=name, unique=unique)
    for name in plan.drop_indexes:
        await chunk_collection.drop_index(name)


def verify_plan_is_clean(plan: MigrationPlan) -> None:
    remaining = {
        "mongoBackfills": len(plan.mongo_backfills),
        "mongoOrphans": len(plan.mongo_orphan_ids),
        "chromaBackfills": len(plan.chroma_backfills),
        "chromaOrphans": len(plan.chroma_orphan_ids),
        "dropIndexes": list(plan.drop_indexes),
        "createIndexes": list(plan.create_indexes),
    }
    if any(value for value in remaining.values()):
        raise RuntimeError(f"Chunk generation migration postcondition failed: {remaining}")


def _get_chroma_collection(path: Path) -> Any | None:
    client = chromadb.PersistentClient(path=str(path))
    names = {
        item.name if hasattr(item, "name") else str(item) for item in client.list_collections()
    }
    return client.get_collection(CHROMA_COLLECTION) if CHROMA_COLLECTION in names else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--approve-plan")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=PROJECT_ROOT / "api" / "core" / "chroma_db",
    )
    args = parser.parse_args()
    if args.apply and args.verify:
        parser.error("--apply and --verify are mutually exclusive")
    if args.apply and not args.approve_plan:
        parser.error("--apply requires --approve-plan=<plan_sha>")
    if not args.apply and args.approve_plan:
        parser.error("--approve-plan is only valid with --apply")
    return args


async def main() -> None:
    args = _parse_args()
    uri = dotenv_values(PROJECT_ROOT / ".env").get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is required")

    client = AsyncMongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        db = client[DATABASE_NAME]
        await db.command("ping")
        chroma_collection = _get_chroma_collection(args.chroma_path.resolve())
        plan = await inspect_migration(db, chroma_collection)

        if args.verify:
            verify_plan_is_clean(plan)
            output = {"mode": "verify", "database": db.name, **plan.public_summary()}
        elif args.apply:
            if args.approve_plan != plan.plan_sha:
                raise RuntimeError(
                    f"Approved plan SHA mismatch: expected {plan.plan_sha}, got {args.approve_plan}"
                )
            # Re-inspect immediately before the first write to close stale-plan use.
            replan = await inspect_migration(db, chroma_collection)
            if replan.plan_sha != plan.plan_sha:
                raise RuntimeError("Migration source changed between plan and apply")
            await apply_migration(db, chroma_collection, replan)
            verification = await inspect_migration(db, chroma_collection)
            verify_plan_is_clean(verification)
            output = {
                "mode": "apply",
                "database": db.name,
                "appliedPlanSha": plan.plan_sha,
                **verification.public_summary(),
            }
        else:
            output = {"mode": "dry-run", "database": db.name, **plan.public_summary()}
        sys.stdout.write(f"{json.dumps(output, indent=2, sort_keys=True)}\n")
    finally:
        close_result = client.close()
        if inspect.isawaitable(close_result):
            await close_result


if __name__ == "__main__":
    asyncio.run(main())
