"""Run evaluation, projections, evidence, metrics, reports, and studies."""

from .integrity import audit_run_event_integrity
from .metrics import aggregate_trials, score, score_details
from .projections import build_execution_trace, build_result_projection

__all__ = [
    "aggregate_trials",
    "audit_run_event_integrity",
    "build_execution_trace",
    "build_result_projection",
    "score",
    "score_details",
]
