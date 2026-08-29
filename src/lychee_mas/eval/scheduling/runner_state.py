"""Read-only projection of one benchmark runner's persisted Trial state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def completed_distinct_cases(run_dir: str | Path) -> int:
    """Return the number of distinct completed Cases observed in one Run."""

    status = _read_object(Path(run_dir) / "run_status.json")
    try:
        return max(0, int(status.get("completed_distinct_cases") or 0))
    except (TypeError, ValueError):
        return 0


def observed_trial_state(allocation: dict[str, Any]) -> dict[str, Any]:
    """Project runner occupancy and demand without making admission decisions."""

    run_dir = str(allocation.get("run_dir") or "").strip()
    if not run_dir:
        return _empty_state()
    run_path = Path(run_dir)
    segment = _read_object(run_path / "run_segment.json")
    launching_demand = int(
        segment.get("scheduler_trial_concurrency", segment.get("scheduler_case_concurrency"))
        or 0
    )
    launching_total = max(0, int(segment.get("scheduler_trial_admission_total") or 0))
    segment_authority = str(segment.get("concurrency_authority") or "").strip().lower()
    status = _read_object(run_path / "run_status.json")
    if not status:
        return {
            **_empty_state(),
            "requested_trial_slots": launching_demand or None,
            "admission_total_issued": launching_total,
            "trial_admission_protocol": "scheduler_trial_admission_total" in segment,
            "runner_status": "launching" if launching_demand else None,
            "concurrency_authority": segment_authority or None,
        }

    active_trials = list(status.get("active_cases") or [])
    active_count = len(active_trials)
    protocol_version = int(status.get("control_protocol_version") or 0)
    segment_started = max(0, int(status.get("segment_started_trials") or 0))
    admission_total = max(
        segment_started,
        int(status.get("scheduler_trial_admission_total") or 0),
    )
    common = {
        "in_flight_trial_slots": active_count,
        "active_trials": active_trials,
        "segment_started_trials": segment_started,
        "runner_admission_total_observed": admission_total,
    }
    if int(segment.get("segment_index") or 0) > int(status.get("segment_index") or 0):
        return {
            **common,
            "requested_trial_slots": max(active_count, launching_demand) or None,
            "admission_total_issued": max(admission_total, launching_total),
            "trial_admission_protocol": (
                protocol_version >= 2 or "scheduler_trial_admission_total" in segment
            ),
            "runner_status": "launching",
            "concurrency_authority": segment_authority or None,
        }

    run_status = str(status.get("status") or "").strip().lower()
    authority = str(
        status.get("concurrency_authority") or segment_authority or "runner"
    ).strip().lower()
    terminal = {
        "draining", "stopped", "paused", "complete", "completed",
        "complete_with_errors", "failed", "failed_fast",
    }
    if run_status in terminal:
        return {
            **common,
            "requested_trial_slots": active_count,
            "admission_total_issued": admission_total,
            "trial_admission_protocol": protocol_version >= 2,
            "runner_status": run_status,
            "concurrency_authority": authority,
        }

    policy = dict(status.get("concurrency_policy") or {})
    current_value = status.get(
        "current_trial_concurrency",
        status.get("current_case_concurrency"),
    )
    if current_value is None and not policy and not active_count:
        return {
            **common,
            "requested_trial_slots": None,
            "admission_total_issued": admission_total,
            "trial_admission_protocol": protocol_version >= 2,
            "runner_status": run_status or None,
            "concurrency_authority": authority,
        }

    current = max(0, int(current_value or 0))
    desired = max(current, int(policy.get("maximum") or policy.get("initial") or current))
    requested: int | None
    if authority == "scheduler":
        requested = None
    else:
        if str(policy.get("mode") or "fixed") == "auto":
            adaptive = status.get(
                "adaptive_trial_concurrency_target",
                status.get("adaptive_case_concurrency_target"),
            )
            if adaptive is None:
                adaptive = _latest_adaptive_target(status.get("concurrency_history") or [])
            if adaptive is not None:
                desired = max(current, int(adaptive))
        requested = max(active_count, desired)
    return {
        **common,
        "requested_trial_slots": requested,
        "admission_total_issued": admission_total,
        "trial_admission_protocol": protocol_version >= 2,
        "runner_status": run_status or None,
        "concurrency_authority": authority,
    }


def remaining_work(allocation: dict[str, Any]) -> int | None:
    """Return remaining Trial work under dataset, Case target, and segment bounds."""

    run_dir = str(allocation.get("run_dir") or "").strip()
    if not run_dir:
        return None
    status = _read_object(Path(run_dir) / "run_status.json")
    if not status:
        return None
    try:
        expected = int(status.get("expected_trials") or 0)
        completed = int(status.get("successful_trials") or 0) + int(
            status.get("failed_trials") or 0
        )
        completed_cases = int(status.get("completed_distinct_cases") or 0)
    except (TypeError, ValueError):
        return None
    remaining = max(0, expected - completed) if expected > 0 else None
    for key in ("case_completion_target", "segment_case_target"):
        target = allocation.get(key)
        if target is None:
            continue
        bounded = max(0, int(target) - completed_cases)
        remaining = bounded if remaining is None else min(remaining, bounded)
    return remaining


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _empty_state() -> dict[str, Any]:
    return {
        "requested_trial_slots": None,
        "in_flight_trial_slots": 0,
        "active_trials": [],
        "segment_started_trials": 0,
        "admission_total_issued": 0,
        "runner_admission_total_observed": 0,
        "trial_admission_protocol": False,
        "runner_status": None,
        "concurrency_authority": None,
    }


def _latest_adaptive_target(history: list[Any]) -> Any:
    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        value = event.get("adaptive_target")
        if value is None and event.get("reason") in {
            "initial", "healthy_window", "overload_signal",
        }:
            value = event.get("target")
        if value is not None:
            return value
    return None


__all__ = ["completed_distinct_cases", "observed_trial_state", "remaining_work"]
