import asyncio
from types import SimpleNamespace

import pytest

from api.domains.learning.router import _append_tutor_chat_turn, _get_tutor_chat_history


@pytest.mark.anyio
async def test_tutor_chat_history_is_scoped_to_current_exercise():
    session = SimpleNamespace(
        _lock=asyncio.Lock(),
        tutor_chat_history=[],
        tutor_chat_exercise_id=None,
    )

    await _append_tutor_chat_turn(
        session,
        exercise_id="exercise-1",
        user_question="Giải thích giúp mình",
        assistant_response="Đây là lời giải từng bước.",
    )

    history = await _get_tutor_chat_history(session, "exercise-1")
    assert history == [
        {"role": "user", "content": "Giải thích giúp mình"},
        {"role": "assistant", "content": "Đây là lời giải từng bước."},
    ]

    next_history = await _get_tutor_chat_history(session, "exercise-2")
    assert next_history == []
