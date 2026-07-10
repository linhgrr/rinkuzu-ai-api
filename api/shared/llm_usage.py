"""llm_usage.py — Per-user LLM token/cost accounting.

Captures the ``usage`` field that LiteLLM returns on completions, attributes it
to the request's user (via context vars set by ``get_current_user``), computes a
USD cost from configurable per-model pricing, and persists one record per call.

All persistence is best-effort (fail-soft): a failure here must never break the
LLM request that triggered it.
"""

from __future__ import annotations

import contextvars

from loguru import logger

from api.config import get_settings
from api.shared import mongo_store
from api.shared.persistence.documents import LlmUsageDocument

# Set by get_current_user for request-bound calls. Background work that still
# belongs to a session should set this explicitly from the persisted owner.
current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id", default=None
)


class LlmAction:
    """Feature labels attributed to each LLM call in usage accounting.

    Passed explicitly down to record_llm_usage so the attribution survives the
    thread-pool executor that a ContextVar would not cross.
    """

    ADAPTIVE_EXERCISE = "adaptive_exercise"
    ADAPTIVE_THEORY = "adaptive_theory"
    ADAPTIVE_SHORT_ANSWER_EVAL = "adaptive_short_answer_eval"
    ADAPTIVE_TUTOR_CHAT = "adaptive_tutor_chat"
    TUTOR_CHAT_SUMMARY = "tutor_chat_summary"
    QUIZ_EXTRACTION = "quiz_extraction"
    QUIZ_TUTOR = "quiz_tutor"
    PIPELINE_CONCEPT_EXTRACTION = "pipeline_concept_extraction"
    PIPELINE_RELATION_VERIFICATION = "pipeline_relation_verification"
    PIPELINE_CYCLE_REMOVAL = "pipeline_cycle_removal"


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a call. Model matched by substring: "pro" → Pro, else Flash.

    # ponytail: substring match model→tier; switch to an explicit model→price
    # table if more models are added.
    """
    settings = get_settings()
    if "pro" in (model or "").lower():
        in_price = settings.llm_price_pro_input_per_m
        out_price = settings.llm_price_pro_output_per_m
    else:
        in_price = settings.llm_price_flash_input_per_m
        out_price = settings.llm_price_flash_output_per_m
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def extract_usage(response: object) -> dict[str, int] | None:
    """Read prompt/completion/total tokens from a LiteLLM response (obj or dict)."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None

    def _get(key: str) -> int:
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    prompt = _get("prompt_tokens")
    completion = _get("completion_tokens")
    total = _get("total_tokens") or (prompt + completion)
    if not (prompt or completion or total):
        return None
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
    }


async def record_llm_usage(
    *,
    model: str,
    provider: str | None,
    usage: dict[str, int] | None,
    action: str | None = None,
) -> None:
    """Persist one usage record. Best-effort — never raises.

    ``action`` is the feature that issued the call (e.g. "adaptive_exercise");
    the caller passes it explicitly so it survives thread-pool hops that a
    ContextVar would not.
    """
    if not usage:
        return
    if not mongo_store.is_available():
        return
    try:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        await LlmUsageDocument(
            user_id=current_user_id.get(),
            action=action,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cost_usd=compute_cost_usd(model, input_tokens, output_tokens),
        ).insert()
    except Exception as exc:
        logger.warning("[llm_usage] failed to record usage: {}", exc)
