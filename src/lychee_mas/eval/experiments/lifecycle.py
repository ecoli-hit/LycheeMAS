"""Pure ExperimentInstance lifecycle projections plus run-status loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_run_status(
    job_state: dict[str, Any],
    *,
    fallback_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read the run-owned status document referenced by one Job state."""

    run_dir = str(job_state.get("run_dir") or fallback_run_dir or "").strip()
    if not run_dir:
        return {}
    try:
        value = json.loads((Path(run_dir) / "run_status.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def finished_instance_projection(
    instance: dict[str, Any],
    job_state: dict[str, Any],
    run_status: dict[str, Any],
    *,
    case_completion_target: int | None,
    finished_at_utc: str,
) -> dict[str, Any]:
    """Project terminal Job and Run facts into one ExperimentInstance update."""

    job_status = str(job_state.get("status") or "failed")
    run_state = str(run_status.get("status") or "")
    lifecycle_consistent = True
    lifecycle_conflict = None
    launcher_failure_after_run = False
    post_run_failure = None
    if run_state == "paused":
        status = "paused"
    elif run_state == "stopped":
        status = "stopped"
    elif run_state == "failed":
        status = "failed"
        if job_status == "completed":
            lifecycle_consistent = False
            lifecycle_conflict = "Run failed although the launcher Job returned completed"
    elif run_state in {"complete", "complete_with_errors"}:
        status = "completed"
        if job_status != "completed":
            launcher_failure_after_run = True
            post_run_failure = {
                "job_status": job_status,
                "failure_reason": job_state.get("failure_reason"),
                "return_code": job_state.get("return_code"),
            }
    elif job_status == "completed":
        status = "completed"
    elif job_status == "stopped":
        status = "stopped"
    else:
        status = "failed"

    queue = dict(instance.get("queue") or {})
    queue.pop("resume_from_status", None)
    completed_cases = max(0, int(run_status.get("completed_distinct_cases") or 0))
    target_reached = (
        case_completion_target is not None and completed_cases >= case_completion_target
    )
    if status == "completed" or target_reached:
        queue["enabled"] = False
    elif status in {"paused", "stopped"} and queue.get("enabled"):
        queue["resume_from_status"] = status
        status = "queued"
    elif status in {"failed", "stopped"}:
        queue["enabled"] = False

    return {
        "status": status,
        "queue": queue,
        "finished_at_utc": finished_at_utc,
        "paused_at_utc": finished_at_utc if status == "paused" else None,
        "job_status": job_status,
        "run_status": run_state or None,
        "lifecycle_consistent": lifecycle_consistent,
        "lifecycle_conflict": lifecycle_conflict,
        "launcher_failure_after_run": launcher_failure_after_run,
        "post_run_failure": post_run_failure,
        "return_code": job_state.get("return_code"),
    }
