"""
migrate_exercise_payload.py — One-time, idempotent migration of al_subject_progress
exercise_history entries from the legacy flat shape to the nested `payload` shape.

Run dry first:   .venv/bin/python scripts/migrate_exercise_payload.py
Apply:           .venv/bin/python scripts/migrate_exercise_payload.py --force
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from loguru import logger
from pymongo import AsyncMongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ENVELOPE_KEYS = (
    "exercise_id",
    "concept_idx",
    "concept_name",
    "bloom_level",
    "question",
    "explanation",
    "explanation_correct",
    "explanation_incorrect",
    "theory",
    "user_answer",
    "is_correct",
    "timestamp",
)


def _get_settings() -> Any:
    return import_module("api.config").get_settings()


def flat_to_payload(entry: dict[str, Any]) -> dict[str, Any]:
    et = entry.get("exercise_type", "mcq")
    if et == "mcq":
        return {
            "exercise_type": "mcq",
            "options": entry.get("options") or {},
            "correct_option": entry.get("correct_option", ""),
        }
    if et == "true_false":
        return {
            "exercise_type": "true_false",
            "statement": entry.get("statement") or "",
            "correct_answer": bool(entry.get("correct_answer")),
        }
    if et == "fill_blank":
        ca = entry.get("correct_answer")
        answers = ca if isinstance(ca, list) else [a for a in [entry.get("correct_option")] if a]
        return {
            "exercise_type": "fill_blank",
            "sentence": entry.get("sentence") or "",
            "hint": entry.get("hint") or "",
            "blank_answers": answers,
        }
    if et == "multi_correct":
        ca = entry.get("correct_answer")
        correct = ca if isinstance(ca, list) else []
        return {
            "exercise_type": "multi_correct",
            "options": entry.get("options") or {},
            "correct_options": sorted(correct),
        }
    if et == "ordering":
        ca = entry.get("correct_answer")
        order = ca if isinstance(ca, list) else (entry.get("items") or [])
        return {"exercise_type": "ordering", "correct_order": order}
    if et == "matching":
        ca = entry.get("correct_answer")
        if isinstance(ca, dict):
            pairs = [{"left": left, "right": right} for left, right in ca.items()]
        else:
            pairs = entry.get("pairs") or []
        return {"exercise_type": "matching", "pairs": pairs}
    if et == "short_answer":
        sample = entry.get("correct_answer") or entry.get("correct_option") or ""
        return {
            "exercise_type": "short_answer",
            "rubric": entry.get("rubric") or [],
            "sample_answer": sample if isinstance(sample, str) else "",
        }
    raise ValueError(f"Unknown exercise_type in legacy entry: {et!r}")


def needs_migration(entry: dict[str, Any]) -> bool:
    return "payload" not in entry


def migrate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not needs_migration(entry):
        return entry
    out = {key: entry[key] for key in _ENVELOPE_KEYS if key in entry}
    out["payload"] = flat_to_payload(entry)
    return out


def migrate_document(doc: dict[str, Any]) -> tuple[dict[str, Any], int]:
    history = doc.get("exercise_history") or []
    changed = 0
    new_history = []
    for entry in history:
        if needs_migration(entry):
            changed += 1
            new_history.append(migrate_entry(entry))
        else:
            new_history.append(entry)
    return new_history, changed


async def run(*, force: bool) -> None:
    settings = _get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncMongoClient(settings.mongodb_uri)
    try:
        db = client["adaptive_learning"]
        collection = db["al_subject_progress"]
        total_docs = 0
        total_entries = 0
        async for doc in collection.find({}):
            new_history, changed = migrate_document(doc)
            if changed == 0:
                continue
            total_docs += 1
            total_entries += changed
            if force:
                await collection.update_one(
                    {"_id": doc["_id"]}, {"$set": {"exercise_history": new_history}}
                )
        verb = "migrated" if force else "would migrate"
        logger.info("{} {} entries across {} documents", verb, total_entries, total_docs)
        if not force:
            logger.info("dry-run only — re-run with --force to apply")
    finally:
        await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Apply the migration (default: dry-run)."
    )
    args = parser.parse_args()
    await run(force=args.force)


if __name__ == "__main__":
    asyncio.run(main())
