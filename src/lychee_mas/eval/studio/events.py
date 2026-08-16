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
