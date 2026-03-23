# Architecture

## Overview

Backend được tổ chức theo 4 khối chính dưới `api/core`:

- `content_pipeline`
- `learning`
- `quiz`
- `shared`

Mục tiêu của layout này là tách nghiệp vụ theo domain trước, rồi mới quan tâm tới technical detail bên trong từng domain.

## Domain Map

### `api/core/content_pipeline`

Phụ trách:

- document loading
- OCR / extraction
- concept merge
- relation verification
- graph building / optimization
- pipeline job orchestration

Nên chứa các flow liên quan trực tiếp tới xử lý tài liệu và knowledge graph.

### `api/core/learning`

Phụ trách:

- session state
- adaptive learning environment
- RL action selection
- exercise generation / submission
- subject progress snapshot

Canonical modules:

- `api.core.learning.session`
- `api.core.learning.exercise_service`
- `api.core.learning.exercise_gen`
- `api.core.learning.agent`

### `api/core/quiz`

Phụ trách:

- tutor chat
- quiz ask-AI flow

Canonical modules:

- `api.core.quiz.tutor_chat`
- `api.core.quiz.quiz_tutor`

### `api/core/shared`

Phụ trách:

- shared LLM helpers
- shared Mongo persistence adapter

Canonical modules:

- `api.core.shared.llm`
- `api.core.shared.mongo_store`

## Dependency Direction

Nguyên tắc hiện tại:

- `routers/` gọi vào domain module phù hợp
- `learning/` và `quiz/` có thể dùng `shared/`
- `content_pipeline/` có thể dùng `shared/`
- `shared/` không phụ thuộc ngược lại vào `learning/`, `quiz/`, hay `content_pipeline`

Nói ngắn gọn:

- `shared` là tầng thấp
- `learning`, `quiz`, `content_pipeline` là các bounded context
- `routers` và `main` là entrypoint layer

## Current Canonical Entry Points

- App startup: `api.main`
- Learning session manager: `api.core.learning.session`
- Exercise flow: `api.core.learning.exercise_service`
- Quiz tutor flow: `api.core.quiz.quiz_tutor`
- Tutor chat flow: `api.core.quiz.tutor_chat`
- Shared LLM helpers: `api.core.shared.llm`
- Shared Mongo adapter: `api.core.shared.mongo_store`

## Refactor Rule

Nếu thêm code mới:

- chọn domain trước: `content_pipeline`, `learning`, `quiz`, hay `shared`
- không thêm lại module mới trực tiếp vào `api/core` root
- nếu logic là dùng chung thật sự giữa nhiều domain, đặt vào `shared`
- nếu logic chỉ phục vụ một flow nghiệp vụ, giữ nó trong domain tương ứng
