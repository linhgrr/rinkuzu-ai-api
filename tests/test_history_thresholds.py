from api.config import get_settings
from api.routers import history


def test_count_mastered_concepts_uses_backend_mastery_threshold():
    threshold = get_settings().adaptive_mastery_threshold

    assert history._count_mastered_concepts([0.74, 0.75, 0.95]) == 2
    assert threshold == 0.75


def test_progress_percent_counts_mastered_concepts_consistently():
    mastered = history._count_mastered_concepts([0.75, 0.2, 0.8, 0.7499])

    assert mastered == 2
    assert history._to_progress_percent(mastered, 4) == 50
