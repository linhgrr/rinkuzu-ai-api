from unittest.mock import AsyncMock

import pytest

from api.domains.assistant import legacy_router, router
from api.domains.assistant.service import AskRinImageUnsupportedError
from api.exceptions import AppError


@pytest.mark.parametrize(
    "raise_error",
    [router._raise_ask_rin_input_error, legacy_router._raise_legacy_tutor_error],
)
def test_image_capability_error_is_actionable_without_exposing_model(
    raise_error,
) -> None:
    with pytest.raises(AppError) as captured:
        raise_error(AskRinImageUnsupportedError())

    error = captured.value
    assert error.code == "ask_rin_image_unsupported"
    assert error.status_code == 422
    assert "another question" in str(error.detail).lower()
    assert "model" not in str(error.detail).lower()
    assert "deepseek" not in str(error.detail).lower()


@pytest.mark.asyncio
async def test_legacy_chat_refunds_turn_before_mapping_image_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingService:
        async def create_stream(self, _context):
            raise AskRinImageUnsupportedError

    refund = AsyncMock()
    monkeypatch.setattr(router, "refund_turn", refund)

    with pytest.raises(AppError) as captured:
        await router._create_stream_or_refund(
            service=FailingService(),
            context=object(),
            user_id="user-1",
            client_request_id="request-1",
        )

    assert captured.value.code == "ask_rin_image_unsupported"
    refund.assert_awaited_once_with(user_id="user-1", client_request_id="request-1")
