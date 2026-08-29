#!/usr/bin/env python3
"""Build strict case/run/study tables from scored benchmark runs."""

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

from lychee_mas.eval.evaluation.studies import StudyRun, aggregate_study  # noqa: E402


def _study_run(value: str) -> StudyRun:
    system_id, separator, run_dir = value.partition("=")
    if not separator or not system_id.strip() or not run_dir.strip():
        raise argparse.ArgumentTypeError("--run must use SYSTEM_ID=/path/to/run")
    return StudyRun(system_id.strip(), Path(run_dir.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate scored runs with explicit systems and strict paired completeness."
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=_study_run,
        required=True,
        help="Repeatable SYSTEM_ID=/path/to/run binding.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    report = aggregate_study(
        args.study_id,
        args.runs,
        args.output_dir,
        bootstrap_samples=max(0, args.bootstrap_samples),
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[study] {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
