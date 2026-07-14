from api.routers.admin_usage import _build_usage_analysis


def test_build_usage_analysis_derives_user_and_cost_insights():
    analysis = _build_usage_analysis(
        totals={"calls": 4, "total_tokens": 1000, "cost_usd": 2.0},
        by_user=[{"key": "user-1", "cost_usd": 1.25}],
        by_action=[{"key": "adaptive_exercise", "cost_usd": 1.5}],
        by_model=[{"key": "deepseek-v4-flash", "cost_usd": 2.0}],
        active_users=3,
    )

    assert analysis["active_users"] == 3
    assert analysis["average_tokens_per_call"] == 250
    assert analysis["average_cost_usd_per_call"] == 0.5
    assert analysis["top_user_id"] == "user-1"
    assert analysis["top_action_cost_share"] == 0.75
    assert analysis["top_model_cost_share"] == 1.0


def test_build_usage_analysis_handles_empty_totals():
    analysis = _build_usage_analysis(
        totals={},
        by_user=[],
        by_action=[],
        by_model=[],
        active_users=0,
    )

    assert analysis["average_tokens_per_call"] == 0.0
    assert analysis["average_cost_usd_per_call"] == 0.0
    assert analysis["top_user_id"] is None
