#!/usr/bin/env python3
"""Re-evaluate terminal ExperimentInstances without repeating model inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lychee_mas.eval.experiments.finalization import ExperimentFinalizer
from lychee_mas.eval.experiments.lifecycle import read_run_status
from lychee_mas.eval.experiments.registry import (
    ExperimentRegistry,
    normalize_instance_execution,
)
from lychee_mas.eval.runner.run_state import atomic_write_json

_ACTIVE_STATUSES = {"queued", "starting", "running", "draining"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _archive_job_state(launch_dir: Path, state: dict[str, Any]) -> Path:
    history_dir = launch_dir / "job_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = history_dir / f"before_reanalysis_{stamp}.json"
    atomic_write_json(path, state)
    return path


def _repaired_job_state(
    state: dict[str, Any],
    *,
    archived_state: Path,
    finished_at_utc: str,
) -> dict[str, Any]:
    return {
        **state,
        "status": "completed",
        "return_code": 0,
        "failure_reason": None,
        "updated_at_utc": finished_at_utc,
        "finished_at_utc": finished_at_utc,
        "reanalysis": {
            "status": "completed",
            "completed_at_utc": finished_at_utc,
            "previous_job_status": state.get("status"),
            "previous_return_code": state.get("return_code"),
            "archived_job_state": str(archived_state),
        },
    }


def reanalyze_instance(repo_root: Path, instance_id: str) -> bool:
    registry = ExperimentRegistry(repo_root)
    instance = registry.get_instance(instance_id)
    launch_dir = Path(str(instance.get("launch_dir") or "")).expanduser().resolve()
    run_dir = Path(str(instance.get("run_dir") or "")).expanduser().resolve()
    analyze_script = launch_dir / "analyze.sh"
    job_path = launch_dir / "job.json"
    if not analyze_script.is_file() or not job_path.is_file() or not run_dir.is_dir():
        raise FileNotFoundError(
            f"incomplete launch/run artifacts for ExperimentInstance {instance_id!r}"
        )

    job_state = _read_json(job_path)
    run_status = read_run_status(job_state, fallback_run_dir=run_dir)
    if str(job_state.get("status") or "") in _ACTIVE_STATUSES or str(
        run_status.get("status") or ""
    ) in _ACTIVE_STATUSES:
        raise ValueError(f"cannot reanalyze active ExperimentInstance {instance_id!r}")

    log_path = launch_dir / "reanalysis.log"
    print(f"[reanalyze] start instance={instance_id} run_dir={run_dir}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] reanalysis start instance={instance_id}\n")
        result = subprocess.run(
            ["bash", str(analyze_script)],
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{_utc_now()}] reanalysis exit_code={result.returncode}\n")

    finished_at_utc = _utc_now()
    finalizer = ExperimentFinalizer(registry)
    final_job_state = job_state
    if result.returncode == 0:
        archived_state = _archive_job_state(launch_dir, job_state)
        final_job_state = _repaired_job_state(
            job_state,
            archived_state=archived_state,
            finished_at_utc=finished_at_utc,
        )
        atomic_write_json(job_path, final_job_state)
        (launch_dir / "exit_code").write_text("0\n", encoding="utf-8")
    finalizer.finalize(
        instance_id,
        final_job_state,
        case_completion_target=normalize_instance_execution(instance.get("execution"))[
            "case_completion_target"
        ],
        finished_at_utc=finished_at_utc,
    )
    print(
        f"[reanalyze] {'completed' if result.returncode == 0 else 'failed'} "
        f"instance={instance_id} log={log_path}",
        flush=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run terminal benchmark evaluation and reconcile ExperimentInstance "
            "state without repeating inference."
        )
    )
    parser.add_argument("instance_ids", nargs="+")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    failures: list[str] = []
    for instance_id in args.instance_ids:
        try:
            if not reanalyze_instance(repo_root, instance_id):
                failures.append(instance_id)
        except Exception as exc:
            failures.append(instance_id)
            print(
                f"[reanalyze] failed instance={instance_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if failures:
        raise SystemExit("reanalysis failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
