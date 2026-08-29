"""Incrementally score completed benchmark Trials while inference is still running."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.eval.benchmarks import get_benchmark  # noqa: E402
from lychee_mas.runtime.events.store import iter_run_events  # noqa: E402


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed_trial_count(run_dir: Path) -> int:
    latest: dict[tuple[str, int], str] = {}
    try:
        events = iter_run_events(
            run_dir,
            event_types=("trial.completed", "trial.failed"),
        )
        for event in events:
            case_id = str(event.get("case_id") or "")
            if case_id:
                latest[(case_id, int(event.get("trial_index") or 0))] = str(
                    event.get("event_type")
                )
    except (FileNotFoundError, OSError, ValueError):
        return 0
    return sum(event_type == "trial.completed" for event_type in latest.values())


def _analyze(run_dir: Path, profile: str) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/analyze_benchmark_run.py"),
        str(run_dir),
        "--incremental",
        "--evaluation-profile",
        profile,
    ]
    print(f"[live-evaluation] scoring completed Trials: {run_dir}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _evaluation_mode(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return "incremental"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        task = str(config.get("task") or config.get("resolved", {}).get("task") or "")
        return get_benchmark(task).evaluation_mode if task else "incremental"
    except (OSError, TypeError, ValueError, KeyError):
        return "incremental"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--max-wait-s", type=float, default=30.0)
    parser.add_argument("--evaluation-profile", default="core")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    interval = max(0.5, float(args.interval_s))
    max_wait = max(interval, float(args.max_wait_s))
    last_observed = 0
    last_analyzed = 0
    last_analysis_at = 0.0
    evaluation_mode = _evaluation_mode(run_dir)
    print(f"[live-evaluation] mode={evaluation_mode} run={run_dir}", flush=True)

    while True:
        alive = _process_alive(args.runner_pid)
        observed = _completed_trial_count(run_dir)
        now = time.monotonic()
        should_analyze = observed > last_analyzed and (
            not alive
            or (
                evaluation_mode == "incremental"
                and (observed > last_observed or now - last_analysis_at >= max_wait)
            )
        )
        if should_analyze:
            if _analyze(run_dir, args.evaluation_profile) == 0:
                last_analyzed = observed
            last_analysis_at = time.monotonic()
        last_observed = observed
        if not alive:
            break
        time.sleep(interval)

    print(
        f"[live-evaluation] stopped completed={last_observed} evaluated={last_analyzed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
