import pytest

from api.config import get_settings
from api.domains.learning import history_router as history


def test_count_mastered_concepts_uses_backend_mastery_threshold():
    threshold = get_settings().adaptive_mastery_threshold

    assert history._count_mastered_concepts([0.74, 0.75, 0.95]) == 2
    assert threshold == 0.75


def test_progress_percent_counts_mastered_concepts_consistently():
    mastered = history._count_mastered_concepts([0.75, 0.2, 0.8, 0.7499])

    assert mastered == 2
    assert history._to_progress_percent(mastered, 4) == 50


def test_unlocked_progress_metrics_exclude_locked_concepts():
    metrics = history._build_unlocked_progress_metrics(
        {
            "concept_indices": {"root": 0, "dependent": 1},
            "concept_mastery": [0.8, 0.95],
            "bloom_mastery": [
                [0.8, 0.8, 0.7, 0.8, 0.8, 0.8],
                [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            ],
        },
        {
            "result": {
                "concept_map": {"root": 0, "dependent": 1},
                "prereq_edges": [{"source": "root", "target": "dependent"}],
            }
        },
    )

    assert metrics["total_concepts"] == 2
    assert metrics["unlocked_concepts"] == 1
    assert metrics["locked_concepts"] == 1
    assert metrics["mastered_concepts"] == 1
    assert metrics["progress_percent"] == 100
    assert metrics["avg_mastery"] == pytest.approx(0.8)
