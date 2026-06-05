---
name: persistence
description: "Skill for the persistence area of rinkuzu-ai-api after removal of the legacy OpenAI file cache."
---

# Persistence

Persistence is centered on Beanie document models and helper modules under `api/core/shared/persistence/`.

## When to Use

- Working with MongoDB / Beanie persistence in `api/`
- Updating pipeline job, quiz draft, subject progress, or document chunk storage
- Verifying which collections are part of the active persistence surface

## Key Files

| File | Purpose |
|------|---------|
| `api/core/shared/persistence/documents.py` | Active Beanie document models |
| `api/core/shared/persistence/pipeline_jobs.py` | Pipeline job CRUD helpers |
| `api/core/shared/persistence/quiz_drafts.py` | Quiz draft CRUD helpers |
| `api/core/shared/persistence/subject_progress.py` | Subject progress snapshot persistence |
| `api/core/shared/persistence/document_chunks.py` | Chunk persistence for extracted document text |
| `api/core/shared/mongo_store.py` | Mongo/Beanie bootstrap using the active document model set |

## Notes

- The legacy provider-specific file-cache document and helper module have been removed.
- New persistence work should extend the active Beanie documents instead of reintroducing provider-specific cache collections.
