from api.config import Settings


def test_pipeline_resilience_settings_have_safe_defaults():
    s = Settings()
    assert s.content_pipeline_reaper_interval_sec == 60
    assert s.content_pipeline_job_stalled_after_sec == 900
    assert s.content_pipeline_recovery_max_age_sec == 3600
    assert s.content_pipeline_dedup_window_sec == 30
    assert s.content_pipeline_max_retry_count == 3
    # stalled threshold MUST be stricter than the "delayed" UX threshold
    assert s.content_pipeline_job_stalled_after_sec > s.content_pipeline_job_delayed_after_sec
