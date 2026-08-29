"""Terminal Run-state transitions and event emission.

The coordinator owns the live benchmark workflow. This module owns the single
place where a Run segment becomes stopped, paused, failed, or complete.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import exception_record

from .run_state import atomic_write_json, graceful_stop_request

STOP_AFTER_ACTIVE_TRIALS = "stop_after_active_trials"


def record_cancelled_run(
    *,
    run_status: dict[str, Any],
    run_status_path: str | Path,
    event_writer,
    operation_id: str,
    previous_elapsed_s: float,
    segment_started_at_unix_s: float,
    vllm_metrics_summary: dict[str, Any] | None,
    out_dir: str | Path,
) -> None:
    """Persist a process-signal cancellation before propagating it."""

    finished_at = round(time.time(), 6)
    run_status.update(
        status="stopped",
        accepting_new_trials=False,
        active_cases=[],
        stop_reason="process_signal",
        finished_at_unix_s=finished_at,
        cumulative_run_elapsed_s=round(
            previous_elapsed_s + max(0.0, finished_at - segment_started_at_unix_s),
            6,
        ),
        updated_at_unix_s=finished_at,
    )
    atomic_write_json(run_status_path, run_status)
    event_writer.log_event(
        "run.stopped",
        operation_id=operation_id,
        run_status="stopped",
        stop_reason="process_signal",
        interrupted_active_trials=True,
        vllm_metrics_summary=vllm_metrics_summary,
        out_dir=out_dir,
    )


def record_failed_run(
    *,
    run_status: dict[str, Any],
    run_status_path: str | Path,
    event_writer,
    operation_id: str,
    error: BaseException,
    vllm_metrics_summary: dict[str, Any] | None,
    out_dir: str | Path,
) -> None:
    """Persist one fatal Run failure before propagating it."""

    finished_at = round(time.time(), 6)
    error_record = exception_record(error)
    run_status.update(
        status="failed",
        active_cases=[],
        finished_at_unix_s=finished_at,
        updated_at_unix_s=finished_at,
        last_run_error=error_record,
    )
    atomic_write_json(run_status_path, run_status)
    event_writer.log_event(
        "run.failed",
        operation_id=operation_id,
        run_status="failed",
        vllm_metrics_summary=vllm_metrics_summary,
        out_dir=out_dir,
        **error_record,
    )


def finalize_run_segment(
    *,
    run_status: dict[str, Any],
    run_status_path: str | Path,
    run_control_path: str | Path,
    event_writer,
    operation_id: str,
    trial_records: list[dict[str, Any]],
    completed_trials: dict[str, set[int]],
    expected_cases: int,
    trials_per_case: int,
    segment_truncated: bool,
    graceful_stop_requested: bool,
    previous_elapsed_s: float,
    segment_started_at_unix_s: float,
    vllm_metrics_summary: dict[str, Any] | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Settle one normally returned Trial pool into one terminal Run state."""

    trial_records.sort(
        key=lambda record: (
            int(record.get("dataset_index", 10**18) or 10**18),
            int(record.get("trial_index", 0) or 0),
        )
    )
    skipped_total = sum(len(indices) for indices in completed_trials.values())
    completed_total = int(run_status["successful_trials"]) + int(run_status["failed_trials"])
    run_status["missing_trials"] = max(
        0, expected_cases * trials_per_case - completed_total
    )
    stopped_gracefully = bool(
        graceful_stop_requested or graceful_stop_request(run_control_path)
    )
    segment_paused = (
        not stopped_gracefully
        and segment_truncated
        and run_status["missing_trials"] > 0
    )
    stop_reason: str | None
    if stopped_gracefully:
        status = "stopped"
        stop_reason = STOP_AFTER_ACTIVE_TRIALS
    elif segment_paused:
        status = "paused"
        stop_reason = "segment_case_limit_reached"
    else:
        status = (
            "complete"
            if run_status["failed_trials"] == 0 and run_status["missing_trials"] == 0
            else "complete_with_errors"
        )
        previous_stop_reason = run_status.get("stop_reason")
        stop_reason = (
            str(previous_stop_reason) if previous_stop_reason is not None else None
        )

    finished_at = round(time.time(), 6)
    run_status.update(
        status=status,
        active_cases=[],
        accepting_new_trials=False,
        stop_reason=stop_reason,
        finished_at_unix_s=finished_at,
        cumulative_run_elapsed_s=round(
            previous_elapsed_s + max(0.0, finished_at - segment_started_at_unix_s),
            6,
        ),
        updated_at_unix_s=finished_at,
    )
    atomic_write_json(run_status_path, run_status)
    event_writer.log_event(
        (
            "run.stopped"
            if stopped_gracefully
            else "run.paused"
            if segment_paused
            else "run.completed"
        ),
        operation_id=operation_id,
        num_trials=len(trial_records),
        trials_per_case=trials_per_case,
        skipped_trials=skipped_total,
        failed_trials=run_status["failed_trials"],
        run_status=status,
        vllm_metrics_summary=vllm_metrics_summary,
        out_dir=out_dir,
    )
    return {
        "status": status,
        "skipped_trials": skipped_total,
        "stopped_gracefully": stopped_gracefully,
        "segment_paused": segment_paused,
    }
