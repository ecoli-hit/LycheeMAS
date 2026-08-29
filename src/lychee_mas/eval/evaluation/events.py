"""Append benchmark Evaluation Events without duplicating unchanged facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import (
    RunEventWriter,
    event_payload,
    exception_record,
    read_evaluation_records,
    run_events_path,
)

_EVALUATION_FIELDS = (
    "case_id",
    "dataset_index",
    "trial_index",
    "task",
    "benchmark_id",
    "scorer_kind",
    "gold",
    "score",
    "score_details",
    "is_correct",
    "correct",
    "contamination_audit",
    "error_type",
    "error_message",
    "traceback",
)
_ENVELOPE_FIELDS = {
    "event_id",
    "worker_id",
    "case_id",
    "dataset_index",
    "trial_index",
    "attempt",
    "operation_id",
    "parent_event_id",
    "correlation_id",
}


def evaluation_record(
    trial: Mapping[str, Any],
    *,
    trial_event_id: str,
) -> dict[str, Any]:
    """Build one Evaluation record without copying the Trial prediction."""

    status = str(trial.get("evaluation_status") or "completed")
    return {
        **{key: trial.get(key) for key in _EVALUATION_FIELDS if key in trial},
        "evaluation_status": status,
        "parent_event_id": trial_event_id,
        "trial_event_id": trial_event_id,
    }


def _event_type(record: Mapping[str, Any]) -> str:
    status = str(record.get("evaluation_status") or "completed")
    return "evaluation.failed" if status in {"error", "failed"} else "evaluation.completed"


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in _ENVELOPE_FIELDS and key not in {"evaluation_status", "final_output"}
    }


def write_evaluation_events(
    run_dir: str | Path,
    trials: Sequence[Mapping[str, Any]],
) -> int:
    """Append changed Evaluation facts and return the number of Events written."""

    run_path = Path(run_dir)
    writer = RunEventWriter(run_events_path(run_path), append=True)
    latest = {
        (str(event.get("case_id") or ""), int(event.get("trial_index") or 0)): event
        for event in read_evaluation_records(run_path)
    }
    written = 0
    for trial in trials:
        trial_event_id = str(trial.get("trial_event_id") or "")
        if not trial_event_id:
            raise RuntimeError(
                f"completed Trial {trial.get('case_id')!r}/{trial.get('trial_index')} "
                "has no terminal event ID"
            )
        record = evaluation_record(trial, trial_event_id=trial_event_id)
        event_type = _event_type(record)
        desired_payload = _payload(record)
        key = (str(record.get("case_id") or ""), int(record.get("trial_index") or 0))
        previous = latest.get(key)
        if (
            previous is not None
            and previous.get("event_type") == event_type
            and previous.get("parent_event_id") == trial_event_id
            and event_payload(previous) == desired_payload
        ):
            continue
        event_id = writer.record_evaluation(record)
        latest[key] = {
            **record,
            "event_id": event_id,
            "event_type": event_type,
            "parent_event_id": trial_event_id,
            "payload": desired_payload,
        }
        written += 1
    return written


def failed_evaluations(
    trials: Sequence[Mapping[str, Any]],
    exc: BaseException,
) -> list[dict[str, Any]]:
    """Attach one normalized scorer failure to every affected Trial."""

    error = exception_record(exc)
    return [{**dict(trial), "evaluation_status": "failed", **error} for trial in trials]
