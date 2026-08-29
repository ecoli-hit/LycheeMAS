"""Idempotent terminal reconciliation for Run and ExperimentInstance state."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import run_events_path

from ..evaluation.evaluators import materialize_run_evaluation
from ..evaluation.projections import build_result_projection
from ..runner.run_state import atomic_write_json
from .lifecycle import finished_instance_projection, read_run_status

_RUN_TERMINAL_STATUSES = {
    "complete",
    "complete_with_errors",
    "failed",
    "paused",
    "stopped",
}


class ExperimentFinalizer:
    """Reconcile terminal Job, Run, evaluation, and ExperimentInstance facts.

    Finalization deliberately remains idempotent. The receipt is written before and
    after evaluation so a Studio restart can repeat a partially completed operation
    without guessing which artifact was last updated.
    """

    def __init__(
        self,
        registry,
        *,
        evaluator: Callable[..., dict[str, Any]] = materialize_run_evaluation,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator

    def finalize(
        self,
        instance_id: str,
        job_state: dict[str, Any],
        *,
        case_completion_target: int | None,
        finished_at_utc: str,
    ) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        run_dir = self._run_dir(job_state, instance)
        run_status = read_run_status(job_state, fallback_run_dir=run_dir)
        receipt_path = (
            run_dir / "run_finalization.json"
            if run_dir is not None and run_dir.is_dir()
            else None
        )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "experiment_instance_id": instance_id,
            "status": "finalizing",
            "finished_at_utc": finished_at_utc,
            "job_status": job_state.get("status"),
            "run_status": run_status.get("status"),
            "evaluation": {"status": "not_available"},
        }
        if receipt_path is not None:
            atomic_write_json(receipt_path, receipt)

        evaluation = self._materialize_evaluation(run_dir, run_status)
        projection = finished_instance_projection(
            instance,
            job_state,
            run_status,
            case_completion_target=case_completion_target,
            finished_at_utc=finished_at_utc,
        )
        projection.update(
            evaluation_status=evaluation["status"],
            evaluation_error=evaluation.get("error"),
            finalized_at_utc=finished_at_utc,
        )
        updated = self.registry.update_instance(instance_id, **projection)

        receipt.update(
            status="finalized",
            experiment_instance_status=updated.get("status"),
            evaluation=evaluation,
        )
        if receipt_path is not None:
            atomic_write_json(receipt_path, receipt)
        return updated

    def needs_reconciliation(self, instance: dict[str, Any]) -> bool:
        """Return whether a terminal launch lacks a completed finalization receipt."""

        run_dir = self._run_dir({}, instance)
        if run_dir is None or not run_dir.is_dir():
            return False
        receipt = self._read_receipt(run_dir / "run_finalization.json")
        return receipt.get("status") != "finalized"

    def project_terminal_lifecycle(
        self,
        instance_id: str,
        job_state: dict[str, Any],
        *,
        case_completion_target: int | None,
        finished_at_utc: str,
        queue: dict[str, Any] | None = None,
        evaluation_status: str | None = None,
    ) -> dict[str, Any]:
        """Synchronize terminal Job and Run facts without running offline evaluation."""

        instance = self.registry.get_instance(instance_id)
        if queue is not None:
            instance = {**instance, "queue": dict(queue)}
        run_status = read_run_status(
            job_state,
            fallback_run_dir=instance.get("run_dir"),
        )
        projection = finished_instance_projection(
            instance,
            job_state,
            run_status,
            case_completion_target=case_completion_target,
            finished_at_utc=finished_at_utc,
        )
        if evaluation_status is not None:
            projection.update(
                evaluation_status=evaluation_status,
                finalized_at_utc=None,
            )
        return self.registry.update_instance(instance_id, **projection)

    def _materialize_evaluation(
        self,
        run_dir: Path | None,
        run_status: dict[str, Any],
    ) -> dict[str, Any]:
        if run_dir is None or not run_dir.is_dir():
            return {"status": "not_available", "reason": "run directory is unavailable"}
        status = str(run_status.get("status") or "")
        if status not in _RUN_TERMINAL_STATUSES:
            return {
                "status": "not_available",
                "reason": f"run status {status or 'unknown'} is not terminal",
            }
        try:
            result = self.evaluator(run_dir, write=True)
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        integrity = self._evaluation_integrity(run_dir)
        if integrity["status"] != "completed":
            return integrity
        return {
            "status": "completed",
            "artifacts": {
                "metrics": str(run_dir / "metrics.json"),
                "evidence": str(run_dir / "evidence.jsonl"),
                "metric_observations": str(run_dir / "metric_observations.jsonl"),
            },
            "summary": (result.get("metric_evaluation") or {}).get("summary"),
        }

    @staticmethod
    def _evaluation_integrity(run_dir: Path) -> dict[str, Any]:
        """Require one successful benchmark evaluation for every terminal Trial."""

        if not run_events_path(run_dir).is_file():
            return {"status": "completed"}
        rows = build_result_projection(run_dir)
        terminal = [row for row in rows if row.get("trial_status") in {"completed", "failed"}]
        if not terminal:
            return {"status": "completed"}
        status_counts: dict[str, int] = {}
        for row in terminal:
            status = str(row.get("evaluation_status") or "not_scored")
            status_counts[status] = status_counts.get(status, 0) + 1
        unresolved = len(terminal) - status_counts.get("completed", 0)
        if unresolved == 0:
            return {"status": "completed"}
        detail = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        return {
            "status": "failed",
            "error": (
                "Benchmark evaluation is incomplete for "
                f"{unresolved}/{len(terminal)} terminal Trials ({detail})."
            ),
            "trial_evaluation_status_counts": status_counts,
        }

    @staticmethod
    def _run_dir(
        job_state: dict[str, Any], instance: dict[str, Any]
    ) -> Path | None:
        value = str(job_state.get("run_dir") or instance.get("run_dir") or "").strip()
        return Path(value).expanduser().resolve() if value else None

    @staticmethod
    def _read_receipt(path: Path) -> dict[str, Any]:
        from ..runner.run_state import read_json

        return read_json(path)


__all__ = ["ExperimentFinalizer"]
