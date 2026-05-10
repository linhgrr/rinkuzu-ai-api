from datetime import UTC, datetime

import pytest

from api.core.quiz.draft_service import (
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)


def test_quiz_draft_s3_key_must_belong_to_user():
    with pytest.raises(QuizDraftValidationError):
        QuizDraftService._normalize_and_validate_s3_key(
            "uploads/quiz_extract/user-2/file.pdf",
            "user-1",
        )


def test_public_draft_uses_safe_defaults():
    now = datetime.now(UTC)

    draft = public_draft(
        {
            "draft_id": "draft-1",
            "title": "Quiz",
            "status": "queued",
            "pdf": {"s3_key": "uploads/quiz_extract/user-1/file.pdf"},
            "created_at": now,
            "updated_at": now,
            "expires_at": now,
        }
    )

    assert draft["draft_id"] == "draft-1"
    assert draft["questions"] == []
    assert draft["progress"] == {"processed": 0, "total": 1, "percent": 0}
