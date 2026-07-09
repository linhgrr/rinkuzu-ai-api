"""Best-effort ETA for in-flight content-pipeline jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.domains.content_pipeline.domain.jobs import PipelineJob


def estimate_eta_seconds(job: PipelineJob, *, secs_per_page: float) -> float | None:
    """Remaining-seconds estimate from progress and page count.

    Returns 0.0 when terminal, None when we lack enough signal (no pages yet).
    """
    if job.status.is_terminal:
        return 0.0
    if job.total_pages <= 0:
        return None
    total_budget = max(1.0, secs_per_page * job.total_pages)
    remaining_fraction = max(0.0, 1.0 - min(1.0, job.progress))
    return round(total_budget * remaining_fraction, 1)
