from __future__ import annotations

from api.schemas.common import BaseStandardModel


class LlmUsageTotalsResponse(BaseStandardModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class LlmUsageBreakdownRow(BaseStandardModel):
    key: str | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class LlmUsageDailyRow(BaseStandardModel):
    date: str
    total_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class LlmUsageAnalysisResponse(BaseStandardModel):
    active_users: int = 0
    average_tokens_per_call: float = 0.0
    average_cost_usd_per_call: float = 0.0
    top_user_id: str | None = None
    top_user_cost_usd: float = 0.0
    top_action: str | None = None
    top_action_cost_share: float = 0.0
    top_model: str | None = None
    top_model_cost_share: float = 0.0


class LlmUsageSummaryResponse(BaseStandardModel):
    window_days: int
    totals: LlmUsageTotalsResponse
    by_action: list[LlmUsageBreakdownRow]
    by_model: list[LlmUsageBreakdownRow]
    by_user: list[LlmUsageBreakdownRow]
    analysis: LlmUsageAnalysisResponse
    daily: list[LlmUsageDailyRow]
