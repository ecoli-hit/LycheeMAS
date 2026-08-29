"""Incremental readers for append-only run event files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LINE_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}


def _jsonl_line_count(path: Path) -> int:
    """Count records once per file signature so event paging has a real total."""

    if not path.is_file():
        return 0
    stat = path.stat()
    key = str(path.resolve())
    cached = _LINE_COUNT_CACHE.get(key)
    signature = (stat.st_size, stat.st_mtime_ns)
    if cached and cached[:2] == signature:
        return cached[2]
    count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if stat.st_size and last_byte != b"\n":
        count += 1
    _LINE_COUNT_CACHE[key] = (*signature, count)
    return count


def read_jsonl_events(
    path: Path,
    *,
    start_line: int = 0,
    limit: int = 500,
    include_total: bool = True,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    requested_start = max(0, int(start_line))
    total_lines = _jsonl_line_count(path) if include_total else None
    effective_start = (
        min(requested_start, total_lines)
        if total_lines is not None
        else requested_start
    )
    next_line = effective_start
    if not path.is_file():
        return {
            "events": events,
            "start_line": effective_start,
            "next_line": next_line,
            "total_lines": 0 if include_total else None,
            "has_more": False,
            "exists": False,
        }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index < effective_start:
                continue
            next_line = index + 1
            try:
                value = json.loads(line)
            except ValueError:
                value = {"span_type": "invalid_jsonl", "raw": line.rstrip("\n")}
            events.append(value if isinstance(value, dict) else {"value": value})
            if len(events) >= max(1, int(limit)):
                break
    return {
        "events": events,
        "start_line": effective_start,
        "next_line": next_line,
        "total_lines": total_lines,
        "has_more": total_lines is not None and next_line < total_lines,
        "exists": True,
    }


def read_run_events(
    run_dir: Path,
    *,
    start_line: int = 0,
    limit: int = 500,
    include_total: bool = True,
) -> dict[str, Any]:
    """Page one logical EventLog across all of its physical JSONL segments."""

    from lychee_mas.runtime.events.store import run_event_paths

    paths = run_event_paths(run_dir)
    requested_start = max(0, int(start_line))
    counts = [_jsonl_line_count(path) for path in paths]
    total_lines = sum(counts) if include_total else None
    effective_start = (
        min(requested_start, total_lines)
        if total_lines is not None
        else requested_start
    )
    events: list[dict[str, Any]] = []
    global_index = 0
    next_line = effective_start
    page_limit = max(1, int(limit))
    for path, count in zip(paths, counts):
        segment_end = global_index + count
        if segment_end <= effective_start:
            global_index = segment_end
            continue
        local_start = max(0, effective_start - global_index)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for local_index, line in enumerate(handle):
                if local_index < local_start:
                    continue
                next_line = global_index + local_index + 1
                try:
                    value = json.loads(line)
                except ValueError:
                    value = {
                        "event_type": "invalid_jsonl",
                        "source_artifact": path.name,
                        "raw": line.rstrip("\n"),
                    }
                events.append(value if isinstance(value, dict) else {"value": value})
                if len(events) >= page_limit:
                    return {
                        "events": events,
                        "start_line": effective_start,
                        "next_line": next_line,
                        "total_lines": total_lines,
                        "has_more": total_lines is not None and next_line < total_lines,
                        "exists": True,
                        "segment_count": len(paths),
                        "segments": [item.name for item in paths],
                    }
        global_index = segment_end
    return {
        "events": events,
        "start_line": effective_start,
        "next_line": next_line,
        "total_lines": total_lines,
        "has_more": total_lines is not None and next_line < total_lines,
        "exists": bool(paths),
        "segment_count": len(paths),
        "segments": [item.name for item in paths],
    }


def read_execution_trace(
    run_dir: Path,
    *,
    case_id: str | None = None,
    trial_index: int | None = None,
    filters: tuple[str, ...] = (),
    start_node: int = 0,
    limit: int = 500,
) -> dict[str, Any]:
    """Page the lightweight flat index of the derived Execution Trace.

    The canonical EventLog remains the lossless source of truth. Execution Trace is a
    navigation projection, so returning recursive ``children`` trees and complete raw
    ``events`` for every node would duplicate the journal and defeat pagination.
    """

    from ...evaluation.projections import build_execution_trace

    trace = build_execution_trace(
        run_dir,
        case_id=case_id,
        trial_index=trial_index,
        filters=filters,
    )
    raw_nodes = trace.pop("nodes")
    nodes: list[dict[str, Any]] = raw_nodes if isinstance(raw_nodes, list) else []
    raw_roots = trace.pop("roots", [])
    roots: list[dict[str, Any]] = raw_roots if isinstance(raw_roots, list) else []
    total = len(nodes)
    start = min(max(0, int(start_node)), total)
    stop = min(total, start + max(1, int(limit)))

    def lightweight_node(node: dict[str, Any]) -> dict[str, Any]:
        raw_events = node.get("events")
        events: list[dict[str, Any]] = raw_events if isinstance(raw_events, list) else []
        raw_children = node.get("children")
        children: list[dict[str, Any]] = (
            raw_children if isinstance(raw_children, list) else []
        )
        return {
            key: value
            for key, value in node.items()
            if key not in {"events", "children"}
        } | {
            "event_refs": [
                {
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "seq": event.get("seq"),
                }
                for event in events
            ],
            "child_count": len(children),
            "child_node_ids": [child.get("node_id") for child in children],
        }

    return {
        **trace,
        "root_node_ids": [root.get("node_id") for root in roots],
        "nodes": [lightweight_node(node) for node in nodes[start:stop]],
        "start_node": start,
        "next_node": stop,
        "total_nodes": total,
        "has_more": stop < total,
    }


def read_result_projection(
    run_dir: Path,
    *,
    start_trial: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Page the one-row-per-Trial result projection."""

    from ...evaluation.projections import build_result_projection

    trials = build_result_projection(run_dir)
    total = len(trials)
    start = min(max(0, int(start_trial)), total)
    stop = min(total, start + max(1, int(limit)))
    return {
        "trials": trials[start:stop],
        "start_trial": start,
        "next_trial": stop,
        "total_trials": total,
        "has_more": stop < total,
    }


def read_metric_trial_groups(
    path: Path,
    *,
    start_trial: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Group MetricObservations by the Trial they describe."""

    groups: dict[tuple[str, int, int], dict[str, Any]] = {}
    observation_count = 0
    if path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(value, dict):
                    continue
                case_id = str(value.get("case_id") or "unknown-case")
                dataset_index = int(value.get("dataset_index") or 0)
                trial_index = int(value.get("trial_index") or 0)
                key = (case_id, dataset_index, trial_index)
                group = groups.setdefault(
                    key,
                    {
                        "trial_key": f"{case_id}:{trial_index}",
                        "case_id": case_id,
                        "dataset_index": dataset_index,
                        "trial_index": trial_index,
                        "observations": [],
                    },
                )
                group["observations"].append(value)
                observation_count += 1

    trials = list(groups.values())
    total_trials = len(trials)
    start = min(max(0, int(start_trial)), total_trials)
    stop = min(total_trials, start + max(1, int(limit)))
    return {
        "trials": trials[start:stop],
        "start_trial": start,
        "next_trial": stop,
        "total_trials": total_trials,
        "total_observations": observation_count,
        "has_more": stop < total_trials,
        "exists": path.is_file(),
    }
