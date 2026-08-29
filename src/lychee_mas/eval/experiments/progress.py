"""Read model for ExperimentInstance cards and persisted runner progress."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class ExperimentProgressProjector:
    """Enrich registry Instances without admitting or launching Trial work."""

    def __init__(self, registry, jobs, *, finish: Callable, case_target: Callable) -> None:
        self.registry = registry
        self.jobs = jobs
        self.finish = finish
        self.case_target = case_target

    def instances(self, *, tail: int = 12, compact: bool = False) -> list[dict[str, Any]]:
        rows = [self.observe(item, tail=tail) for item in self.registry.instances()]
        return [self.compact(item) for item in rows] if compact else rows

    def instance_progress(self, instance_id: str, *, tail: int = 40) -> dict[str, Any]:
        return self.observe(self.registry.get_instance(instance_id), tail=tail)

    def observe(self, instance: dict[str, Any], *, tail: int) -> dict[str, Any]:
        launch_dir = str(instance.get("launch_dir") or "").strip()
        if not launch_dir:
            return {**instance, "progress": None}
        try:
            progress = self.jobs.progress(Path(launch_dir), tail=tail)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            return {
                **instance,
                "progress": {
                    "state": {"status": str(instance.get("status") or "unknown")},
                    "observation_status": "unavailable",
                    "percent": 0,
                    "message": f"progress unavailable: {type(exc).__name__}: {exc}",
                    "lines": [],
                },
            }

        job_status = str((progress.get("state") or {}).get("status") or "")
        if instance.get("status") == "running" and job_status not in {"starting", "running"}:
            instance = self.finish(instance["id"], dict(progress["state"]))
        target = self.case_target(instance)
        completed_cases = progress.get("completed_distinct_cases")
        target_percent = (
            min(100, int(int(completed_cases or 0) * 100 / target))
            if target is not None and target > 0 and completed_cases is not None
            else None
        )
        projected = {
            **progress,
            "dataset_percent": progress.get("percent"),
            "percent": target_percent if target_percent is not None else progress.get("percent"),
            "case_completion_target": target,
            "remaining_to_case_target": (
                max(0, target - int(completed_cases or 0))
                if target is not None and completed_cases is not None
                else None
            ),
        }
        return {**instance, "progress": projected}

    @staticmethod
    def compact(instance: dict[str, Any]) -> dict[str, Any]:
        """Return the card-list projection without repeated Spec and log payloads."""

        result = {
            key: value
            for key, value in instance.items()
            if key not in {"experiment_spec", "registry_path"}
        }
        progress = dict(result.get("progress") or {})
        if progress:
            state = dict(progress.get("state") or {})
            progress["state"] = {
                key: value
                for key, value in state.items()
                if key
                in {
                    "status", "return_code", "run_dir", "created_at_utc",
                    "started_at_utc", "finished_at_utc",
                }
            }
            progress["lines"] = []
            result["progress"] = progress
        return result


__all__ = ["ExperimentProgressProjector"]
