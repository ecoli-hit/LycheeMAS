#!/usr/bin/env python3
"""Backfill offline evidence and metrics for completed benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from lychee_mas.runtime.events.store import run_events_path  # noqa: E402

COMPLETED_STATUSES = {"complete", "completed", "complete_with_errors", "stopped"}


def _status(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return str(value.get("status") or "unknown") if isinstance(value, dict) else "unknown"


def discover_runs(root: Path, *, include_existing: bool) -> list[Path]:
    runs = []
    for status_path in root.rglob("run_status.json"):
        run_dir = status_path.parent
        if _status(status_path) not in COMPLETED_STATUSES:
            continue
        if not run_events_path(run_dir).is_file():
            continue
        if not include_existing and (run_dir / "metric_evaluation.json").is_file():
            continue
        runs.append(run_dir)
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evidence and MetricObservations from existing completed runs."
    )
    parser.add_argument("--runs-root", default="runs/benchmarks")
    parser.add_argument("--evaluation-profile", default="core")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.runs_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"runs root not found: {root}")
    runs = discover_runs(root, include_existing=args.include_existing)
    if args.limit is not None:
        runs = runs[: max(0, args.limit)]
    print(f"[backfill] root={root} selected={len(runs)} profile={args.evaluation_profile}")
    if args.dry_run:
        for run_dir in runs:
            print(f"[would-analyze] {run_dir}")
        return

    failures: list[tuple[Path, str]] = []
    for index, run_dir in enumerate(runs, start=1):
        try:
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    str(Path(ROOT) / "scripts/analyze_benchmark_run.py"),
                    str(run_dir),
                    "--evaluation-profile",
                    args.evaluation_profile,
                ],
                check=True,
            )
            summary = json.loads((run_dir / "metric_evaluation.json").read_text(encoding="utf-8"))[
                "summary"
            ]
            print(
                f"[{index}/{len(runs)}] analyzed observations="
                f"{summary['observation_count']} errors={summary['evaluator_error_count']} "
                f"run={run_dir}"
            )
        except Exception as exc:
            failures.append((run_dir, f"{type(exc).__name__}: {exc}"))
            print(f"[{index}/{len(runs)}] failed run={run_dir}: {failures[-1][1]}")

    print(f"[backfill] completed={len(runs) - len(failures)} failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
