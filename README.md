---
title: Rinkuzu AI API
emoji: 🚀
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# Rinkuzu AI API

FastAPI backend cho adaptive learning, quiz tutor, quiz extraction, và content pipeline.

## Runtime

- Python `3.11`
- FastAPI + Uvicorn
- MongoDB
- OpenAI-compatible LLM endpoint
- Optional: S3-compatible object storage cho OCR, quiz extract, và pipeline assets

## Core Layout

`api/core` hiện được chia theo domain:

- `content_pipeline/`: xử lý ingest, OCR, extraction, graph build, pipeline orchestration
- `learning/`: session state, RL selection, exercise generation, mastery flow
- `quiz/`: tutor chat và ask-AI flow cho quiz
- `shared/`: helper dùng chung như LLM client và Mongo persistence

Chi tiết hơn xem [ARCHITECTURE.md](/home/linh/Downloads/datn_1/new/data/rinkuzu-ai-api/ARCHITECTURE.md).

## Local Setup

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate linhdz
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload
```

## Test

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate linhdz
pytest
```

## Key Endpoints

- `GET /api/health`
- `GET /api/info`
- `POST /api/session/start`
- `POST /api/pipeline/process`
- `POST /api/quiz/extract`
- `POST /api/quiz/ask-ai`

## Canonical Imports

Khi viết code mới, dùng path canonical mới thay vì path cũ:

- `api.core.learning.session`
- `api.core.learning.exercise_service`
- `api.core.quiz.tutor_chat`
- `api.core.quiz.quiz_tutor`
- `api.core.shared.llm`
- `api.core.shared.mongo_store`

## Production Notes

- Không commit secret thật vào `.env`
- Set `INTERNAL_SERVICE_TOKEN` ở mọi môi trường không-local
- Giới hạn `CORS_ORIGINS` theo domain thật thay vì `["*"]`
- Pin dependency chặt hơn trước khi deploy production nếu cần reproducible build nghiêm ngặt
