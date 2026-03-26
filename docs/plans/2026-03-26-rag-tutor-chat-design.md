# RAG for Tutor Chat — Design Document

**Date:** 2026-03-26
**Status:** Implemented (`feat/rag-tutor-chat` branch)
**Branch:** `feat/rag-tutor-chat`

---

## 1. Goal

Add Retrieval-Augmented Generation (RAG) to the tutor chat (Rin-chan) so that when a student asks a question about a quiz exercise, the system first retrieves relevant document chunks from the original PDF and injects them into the LLM prompt.

**Before:** Rin-chan relied only on parametric knowledge (LLM's training data).
**After:** Rin-chan retrieves the top-3 most relevant text chunks from the pipeline document and uses them as ground-truth context.

---

## 2. Architecture

```
[Pipeline runs]
  load_document_chunks() → chunks (in-memory)
  persist_document_chunks()  ← NEW
    ├── MongoDB: upsert into "al_document_chunks"
    └── ChromaDB: add to "document_chunks" (vector store)

[Tutor Chat — at runtime]
  User question
    ├── Embed query (keepitreal/vietnamese-sbert)
    ├── ChromaDB similarity search (k=3, filter: job_id)
    ├── Build rag_context string
    └── Inject into build_tutor_prompt() → LLM → answer
```

---

## 3. Components

### 3.1 `ChunkChromaStore` (new)
**File:** `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py`

- ChromaDB collection `"document_chunks"`
- Reuses existing `EmbeddingClient` (`keepitreal/vietnamese-sbert`)
- Methods:
  - `add_chunks(chunks, job_id, subject_id)` — persist chunks to ChromaDB
  - `aretrieve(query, job_id, k=3)` — async retrieval filtered by `job_id`

### 3.2 Pipeline Stage `persist_document_chunks` (new)
**File:** `api/core/content_pipeline/application/stages/chunk_persistence.py`

- Runs **after** `load_document_chunks`, **before** `concept_extraction`
- Writes to both MongoDB and ChromaDB
- Graceful degradation: logs warning but continues pipeline if storage fails
- MongoDB schema:

```python
{
    "job_id": str,          # pipeline job ID
    "subject_id": str,      # subject/topic
    "chunk_index": int,     # sequential index in document
    "text": str,           # raw chunk text
    "start_page": int,
    "end_page": int,
    "created_at": float,
}
# Index: {job_id: 1, chunk_index: 1} — unique
```

### 3.3 `PipelineRunner` Update
**File:** `api/core/content_pipeline/application/pipeline_runner.py`

- New constructor args: `chunk_chroma_store`, `document_chunks_col`
- Calls `persist_document_chunks()` after `load_document_chunks()`

### 3.4 App Startup Update
**File:** `api/main.py`

- Init `ChunkChromaStore` with `EmbeddingClient` in lifespan
- Init MongoDB `al_document_chunks` collection reference
- Pass both to `PipelineRunner` constructor

### 3.5 Tutor Chat — RAG Integration
**Files:**
- `api/core/quiz/tutor_chat.py`
- `api/routers/session.py`

- `build_tutor_prompt()` gains `rag_context: str = ""` parameter
- When non-empty, injects:
  ```
  NGỮ CẢNH TỪ TÀI LIỆU (dùng để trả lời chính xác):
  [Đoạn 1] (trang X)
  <text>

  [Đoạn 2] (trang Y)
  <text>
  ...
  ```
- `create_tutor_chat_stream()` and `generate_tutor_chat_response()` pass `rag_context`
- `_build_rag_context()` helper in session router:
  - Uses `session.job_id` to filter to correct course
  - Returns empty string if store unavailable or no results
  - All errors caught and logged — chat continues without RAG

### 3.6 `get_chunk_chroma_store` Dependency
**File:** `api/dependencies.py`

- Returns `app.state.chunk_chroma_store` (may be `None`)

---

## 4. Data Flow

1. **Pipeline execution:** User uploads PDF → chunks extracted → `persist_document_chunks` writes to MongoDB + ChromaDB → concepts extracted → knowledge graph built → job saved to MongoDB

2. **Session creation:** `POST /api/pipeline/jobs/{id}/create-session` → `SessionState.job_id` set to this pipeline's job_id

3. **Tutor chat:** `POST /api/session/{id}/chat` →
   - `_build_rag_context(session, chunk_store, user_question)` → ChromaDB query
   - Top-3 chunks formatted as context block
   - `build_tutor_prompt(..., rag_context=rag_context)` → full prompt
   - LLM streams/generates answer with document context

---

## 5. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage | MongoDB + ChromaDB | MongoDB = durability; ChromaDB = fast vector search |
| Embedding model | `keepitreal/vietnamese-sbert` (existing) | No new dependency; consistent with concept embeddings |
| Retrieval k | 3 | Small enough to avoid prompt overflow; enough for context |
| Chunk ID format | `{job_id}_chunk_{i}` | Unique, sortable, filterable by prefix |
| Graceful degradation | Yes | Pipeline and chat continue even if RAG storage fails |
| Filter by | `job_id` | Ensures chat only uses chunks from the course the student is studying |

---

## 6. Error Handling

| Scenario | Behavior |
|---|---|
| ChromaDB unavailable at startup | `chunk_chroma_store = None` in app state |
| Chunk retrieval fails | Returns `""`, logs warning, chat proceeds without RAG |
| No chunks found for `job_id` | Returns `""`, chat proceeds without RAG |
| MongoDB write fails in pipeline | Logs warning, continues pipeline (chunks are nice-to-have) |
| Session has no `job_id` | `_build_rag_context` returns `""` immediately |

---

## 7. Commits

```
feat(rag): add ChunkChromaStore for document chunk retrieval
feat(rag): add persist_document_chunks pipeline stage
feat(rag): wire persist_document_chunks into pipeline runner
feat(rag): init ChunkChromaStore and MongoDB collection at app startup
fix(rag): remove unused persist_directory param from ChunkChromaStore init
feat(rag): integrate RAG retrieval into tutor chat
fix(rag): restore _append_tutor_chat_turn body accidentally removed
fix(rag): remove duplicate _append_tutor_chat_turn body in session.py
```

---

## 8. Testing Checklist

- [ ] Unit: `ChunkChromaStore.add_chunks()` + `aretrieve()`
- [ ] Unit: `persist_document_chunks` stage with mocked stores
- [ ] Integration: Run full pipeline on a small PDF → verify chunks in MongoDB + ChromaDB
- [ ] Integration: Tutor chat with a session → verify RAG retrieval is called
- [ ] Integration: Tutor chat when `job_id` is `None` → should return empty context gracefully
