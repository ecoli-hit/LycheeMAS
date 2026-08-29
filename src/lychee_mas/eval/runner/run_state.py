"""Durable control and status-file helpers for benchmark runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STOP_AFTER_ACTIVE_TRIALS = "stop_after_active_trials"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(
                    {"status": "error", "error_type": "InvalidJSONL", "raw_line": line}
                )
    return records


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one object without making recovery depend on a pristine file."""

    target = Path(path)
    if not target.is_file():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def run_status_elapsed_seconds(status: dict[str, Any]) -> float:
    """Return elapsed wall time across completed and current resume segments."""

    segment_started = status.get("current_segment_started_at_unix_s")
    accumulated = status.get("accumulated_run_elapsed_before_segment_s")
    segment_finished = status.get("finished_at_unix_s") or status.get("updated_at_unix_s")
    if accumulated is not None and segment_started is not None and segment_finished is not None:
        return max(0.0, float(accumulated)) + max(
            0.0,
            float(segment_finished) - float(segment_started),
        )
    cumulative = status.get("cumulative_run_elapsed_s")
    if cumulative is not None:
        return max(0.0, float(cumulative))
    started = status.get("started_at_unix_s")
    if started is not None and segment_finished is not None:
        return max(0.0, float(segment_finished) - float(started))
    return 0.0


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Replace one JSON file atomically."""

    target = Path(path)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def graceful_stop_request(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    return value if value.get("action") == STOP_AFTER_ACTIVE_TRIALS else {}


def scheduler_trial_concurrency(
    path: str | Path,
    fallback: int | None = None,
) -> int | None:
    value = read_json(path)
    raw = value.get(
        "scheduler_trial_concurrency",
        value.get("scheduler_case_concurrency", fallback),
    )
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def scheduler_trial_admission_total(
    path: str | Path,
    fallback: int | None = None,
) -> int | None:
    raw = read_json(path).get("scheduler_trial_admission_total", fallback)
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback
