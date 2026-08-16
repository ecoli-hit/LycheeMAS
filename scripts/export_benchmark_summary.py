#!/usr/bin/env python3
"""Export model-by-benchmark summary metrics to an Excel workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

from lychee_mas.eval.benchmark_excel import write_benchmark_workbook


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate task-level LycheeMAS benchmark metrics into one model-by-benchmark "
            "Excel workbook."
        )
    )
    parser.add_argument(
        "runs_root",
        help="Run tree containing task config.yaml/metrics.json files.",
    )
    parser.add_argument(
        "--output",
        help="Output .xlsx path (default: <runs_root>/benchmark_summary.xlsx).",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "latest"),
        default="error",
        help="How to handle more than one run for the same model and runnable task.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output) if args.output else runs_root / "benchmark_summary.xlsx"
    workbook = write_benchmark_workbook(
        runs_root,
        output,
        duplicate_policy=args.duplicate_policy,
    )
    print(f"[benchmark-summary] workbook={workbook}")


if __name__ == "__main__":
    main()
