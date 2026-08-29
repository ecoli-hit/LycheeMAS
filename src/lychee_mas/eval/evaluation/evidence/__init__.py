"""Framework-neutral evidence normalization for benchmark runs."""

from .normalizer import (
    EVIDENCE_COVERAGE_FILENAME,
    EVIDENCE_EVENTS_FILENAME,
    iter_normalized_events,
    normalize_run_evidence,
)

__all__ = [
    "EVIDENCE_COVERAGE_FILENAME",
    "EVIDENCE_EVENTS_FILENAME",
    "iter_normalized_events",
    "normalize_run_evidence",
]
