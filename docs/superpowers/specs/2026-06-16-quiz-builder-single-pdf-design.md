# Quiz Builder — single PDF + single LLM call + dọn 2 hệ draft

Date: 2026-06-16

## Mục tiêu

1. Chỉ cho upload **1 file PDF** duy nhất, **tối đa 5MB**.
2. BE FastAPI dùng OCR service (đã có) lấy **full text**.
3. Gửi **full text trong 1 lần** (single LLM call) cùng các trường cần thiết tới LLM.
4. UI: trang home/dashboard có dropzone PDF; drop xong → chuyển sang trang upload với PDF đã sẵn, user chỉ điền thông tin thêm.
5. Dọn dead code / techdebt, chuẩn hoá quyền sở hữu draft về **FastAPI**.

## Quyết định nền tảng (đã chốt)

- **Canonical draft = FastAPI** (`/api/quiz/drafts` + `QuizDraftService`). FE gọi thẳng FastAPI.
- **Mang File qua trang:** giữ `File` trong zustand store (không persist), upload khi submit.
- **Single LLM call:** gửi full text 1 lần, có guard cắt + cảnh báo khi vượt ngưỡng ký tự.
- **Published Quiz** vẫn tạo ở Next.js (Mongoose `Quiz`, duyệt admin, slug). Hai DB tách biệt.

## Kiến trúc đích

```
Dashboard (drop 1 PDF ≤5MB) → giữ File trong zustand (pending-upload)
   → /create (PDF đã sẵn, điền title/category/description/prompt)
   → uploadPdfViaPresignedUrl → s3Key
   → FastAPI POST /api/quiz/drafts {title, s3_key, file_name, file_size, category_id, description, prompt}
        background: S3 → OCR full text → 1 LLM call (full text) → questions
   → FE poll GET /api/quiz/drafts/{id} → completed
   → /draft/{id}/edit (load + PATCH thẳng FastAPI)
   → Submit: Next tạo published Quiz + FastAPI mark draft submitted
```

## Backend (FastAPI)

- `extraction.py`: bỏ vòng lặp `build_text_batches`; gọi `invoke_structured_completion` **một lần** với `document_text.text` (full). Guard: vượt `quiz_extract_max_chars` (config mới) → cắt + log cảnh báo.
- `MAX_PDF_BYTES = 5 * 1024 * 1024`; sửa message "50MB" → "5MB".
- Giữ OCR cache + `QuizDraftService` background; chỉ đổi bước extract.

## Frontend

- `QuizDropZone` ở dashboard: 1 file, accept pdf, maxSize 5MB → lưu `File` vào `usePendingQuizUploadStore` (zustand, không persist) → `router.push('/create')`.
- `/create`: đọc File từ store; bỏ multi-file + `mergePdfFiles` + "Total size"; `maxFiles:1`, `maxSize:5MB`. Submit gọi thẳng FastAPI.
- `/create/mobile`: đồng bộ 5MB + gọi FastAPI.
- `usePdfProcessor` / `DraftSyncProvider` / `useDraftStore`: trỏ poll & list sang FastAPI; bỏ field `chunks` lỗi thời.
- `/draft/[id]/edit`: load + PATCH từ FastAPI.

## Dead code cần xoá

- `lib/quiz/extractApi.ts` (trỏ `/api/quiz/extract` đã bị xoá khỏi FastAPI).
- `app/api/draft/[id]/process-chunk/route.ts` (không còn caller).
- `app/api/draft/create|list|[id]` route (thay bằng gọi thẳng FastAPI).
- `app/api/draft/[id]/submit/route.ts`: giữ logic tạo `Quiz`, bỏ đọc `DraftQuiz` → chuyển thành tạo Quiz từ body + gọi FastAPI mark submitted.
- `models/DraftQuiz.ts` sau khi hết route Next dùng.
- Field `chunks` + comment US-507 trong store.

## Testing

- BE: assert LLM gọi đúng 1 lần với full text; test guard cắt; test `MAX_PDF_BYTES` 5MB.
- FE: build + typecheck; e2e thủ công drop → /create → submit → poll → edit → publish.
