"""Mongo persistence for Ask Rin-chan conversations and request idempotency."""

from __future__ import annotations

from datetime import timedelta
from typing import cast
import uuid

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.exceptions import AppError
from api.shared.persistence.common import utc_now

from .documents import (
    AskRinConversationDocument,
    AskRinMessageDocument,
    AskRinRequestDocument,
)

_LEASE_SECONDS = 240
_MODEL_HISTORY_MESSAGES = 12


async def _get_or_create_conversation(user_id: str, context_id: str) -> dict:
    now = utc_now()
    collection = AskRinConversationDocument.get_pymongo_collection()
    result = await collection.find_one_and_update(
        {"user_id": user_id, "exercise_context_id": context_id},
        {
            "$setOnInsert": {
                "conversation_id": str(uuid.uuid4()),
                "user_id": user_id,
                "exercise_context_id": context_id,
                "summary": "",
                "in_flight_request_id": None,
                "lease_expires_at": None,
                "created_at": now,
            },
            "$set": {"updated_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if result is None:
        raise RuntimeError("Failed to create Ask Rin-chan conversation")
    return cast("dict", result)


async def begin_turn(
    *, user_id: str, context_id: str, client_request_id: str, message: str
) -> tuple[dict, str | None]:
    existing = await AskRinRequestDocument.find_one(
        AskRinRequestDocument.user_id == user_id,
        AskRinRequestDocument.client_request_id == client_request_id,
    )
    if existing and existing.exercise_context_id != context_id:
        raise AppError(
            code="idempotency_conflict",
            message="This request id belongs to another exercise",
            detail="Create a new request id for this message",
            status_code=409,
        )
    if existing and existing.status in {"completed", "interrupted"}:
        assistant = await AskRinMessageDocument.find_one(
            AskRinMessageDocument.user_id == user_id,
            AskRinMessageDocument.client_request_id == client_request_id,
            AskRinMessageDocument.role == "assistant",
        )
        return {"conversation_id": existing.conversation_id}, assistant.content if assistant else ""
    now = utc_now()
    if existing and existing.status == "in_progress":
        if existing.lease_expires_at and existing.lease_expires_at <= now:
            await refund_turn(user_id=user_id, client_request_id=client_request_id)
            existing = await AskRinRequestDocument.find_one(
                AskRinRequestDocument.user_id == user_id,
                AskRinRequestDocument.client_request_id == client_request_id,
            )
        else:
            raise AppError(
                code="ask_rin_request_in_progress",
                message="This message is already being answered",
                detail="Wait for the current response before retrying",
                status_code=409,
            )

    conversation = await _get_or_create_conversation(user_id, context_id)
    conversation_id = str(conversation["conversation_id"])
    lease_expires_at = now + timedelta(seconds=_LEASE_SECONDS)
    acquired = await AskRinConversationDocument.get_pymongo_collection().find_one_and_update(
        {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "$or": [
                {"in_flight_request_id": None},
                {"lease_expires_at": {"$lte": now}},
                {"in_flight_request_id": client_request_id},
            ],
        },
        {
            "$set": {
                "in_flight_request_id": client_request_id,
                "lease_expires_at": lease_expires_at,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not acquired:
        raise AppError(
            code="ask_rin_conversation_busy",
            message="Rin-chan is already answering this exercise",
            detail="Wait for the current response before sending another message",
            status_code=409,
        )

    try:
        if existing and existing.status == "refunded":
            existing.status = "in_progress"
            existing.lease_expires_at = lease_expires_at
            existing.updated_at = now
            await existing.save()
        else:
            await AskRinRequestDocument(
                client_request_id=client_request_id,
                conversation_id=conversation_id,
                user_id=user_id,
                exercise_context_id=context_id,
                status="in_progress",
                lease_expires_at=lease_expires_at,
            ).insert()
    except DuplicateKeyError as exc:
        await _release_conversation(conversation_id, client_request_id)
        raise AppError(
            code="ask_rin_request_in_progress",
            message="This message is already being answered",
            detail="Wait for the current response before retrying",
            status_code=409,
        ) from exc
    except Exception:
        await _release_conversation(conversation_id, client_request_id)
        raise

    try:
        await AskRinMessageDocument(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_id=user_id,
            client_request_id=client_request_id,
            role="user",
            content=message,
        ).insert()
    except Exception:
        await refund_turn(user_id=user_id, client_request_id=client_request_id)
        raise

    return {"conversation_id": conversation_id}, None


async def load_model_history(conversation_id: str, user_id: str) -> list[dict[str, str]]:
    rows = (
        await AskRinMessageDocument.find(
            AskRinMessageDocument.conversation_id == conversation_id,
            AskRinMessageDocument.user_id == user_id,
        )
        .sort("-created_at")
        .limit(_MODEL_HISTORY_MESSAGES + 1)
        .to_list()
    )
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows[:-1]]


async def finish_turn(
    *, user_id: str, client_request_id: str, content: str, interrupted: bool
) -> str:
    request = await AskRinRequestDocument.find_one(
        AskRinRequestDocument.user_id == user_id,
        AskRinRequestDocument.client_request_id == client_request_id,
    )
    if not request:
        raise RuntimeError("Ask Rin-chan request record is missing")
    if request.status in {"completed", "interrupted"} and request.assistant_message_id:
        return request.assistant_message_id

    message_id = str(uuid.uuid4())
    try:
        await AskRinMessageDocument(
            message_id=message_id,
            conversation_id=request.conversation_id,
            user_id=user_id,
            client_request_id=client_request_id,
            role="assistant",
            content=content,
            status="interrupted" if interrupted else "complete",
        ).insert()
    except DuplicateKeyError:
        existing = await AskRinMessageDocument.find_one(
            AskRinMessageDocument.user_id == user_id,
            AskRinMessageDocument.client_request_id == client_request_id,
            AskRinMessageDocument.role == "assistant",
        )
        if existing:
            message_id = existing.message_id

    request.status = "interrupted" if interrupted else "completed"
    request.assistant_message_id = message_id
    request.lease_expires_at = None
    request.updated_at = utc_now()
    await request.save()
    await _release_conversation(request.conversation_id, client_request_id)
    return message_id


async def refund_turn(*, user_id: str, client_request_id: str) -> None:
    request = await AskRinRequestDocument.find_one(
        AskRinRequestDocument.user_id == user_id,
        AskRinRequestDocument.client_request_id == client_request_id,
    )
    if not request or request.status != "in_progress":
        return
    request.status = "refunded"
    request.lease_expires_at = None
    request.updated_at = utc_now()
    await request.save()
    await AskRinMessageDocument.find(
        AskRinMessageDocument.user_id == user_id,
        AskRinMessageDocument.client_request_id == client_request_id,
    ).delete()
    await _release_conversation(request.conversation_id, client_request_id)


async def _release_conversation(conversation_id: str, client_request_id: str) -> None:
    await AskRinConversationDocument.find_one(
        AskRinConversationDocument.conversation_id == conversation_id,
        AskRinConversationDocument.in_flight_request_id == client_request_id,
    ).update({"$set": {"in_flight_request_id": None, "lease_expires_at": None}})


async def get_conversation(user_id: str, context_id: str) -> dict | None:
    conversation = await AskRinConversationDocument.find_one(
        AskRinConversationDocument.user_id == user_id,
        AskRinConversationDocument.exercise_context_id == context_id,
    )
    if not conversation:
        return None
    messages = (
        await AskRinMessageDocument.find(
            AskRinMessageDocument.user_id == user_id,
            AskRinMessageDocument.conversation_id == conversation.conversation_id,
        )
        .sort("created_at")
        .to_list()
    )
    return {"conversation": conversation, "messages": messages}


async def delete_conversation(user_id: str, context_id: str) -> bool:
    now = utc_now()
    deletion_lock = f"delete:{uuid.uuid4()}"
    collection = AskRinConversationDocument.get_pymongo_collection()
    conversation = await collection.find_one_and_update(
        {
            "user_id": user_id,
            "exercise_context_id": context_id,
            "$or": [
                {"in_flight_request_id": None},
                {"lease_expires_at": {"$lte": now}},
            ],
        },
        {
            "$set": {
                "in_flight_request_id": deletion_lock,
                "lease_expires_at": now + timedelta(seconds=_LEASE_SECONDS),
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not conversation:
        existing = await collection.find_one(
            {"user_id": user_id, "exercise_context_id": context_id},
            {"_id": 1},
        )
        if not existing:
            return False
        raise AppError(
            code="ask_rin_conversation_busy",
            message="Rin-chan is still answering this exercise",
            detail="Wait for the current response before clearing the conversation",
            status_code=409,
        )

    conversation_id = str(conversation["conversation_id"])
    try:
        await AskRinMessageDocument.find(
            AskRinMessageDocument.user_id == user_id,
            AskRinMessageDocument.conversation_id == conversation_id,
        ).delete()
        await AskRinRequestDocument.find(
            AskRinRequestDocument.user_id == user_id,
            AskRinRequestDocument.conversation_id == conversation_id,
        ).delete()
        deleted = await collection.delete_one(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "in_flight_request_id": deletion_lock,
            }
        )
    except Exception:
        await _release_conversation(conversation_id, deletion_lock)
        raise
    else:
        return deleted.deleted_count == 1
