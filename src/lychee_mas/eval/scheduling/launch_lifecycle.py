"""Experiment launch, resume, segmentation, and allocation lifecycle.

The queue manager owns scheduling state. This mixin keeps the launch lifecycle
in one independently testable module while preserving the manager's public API.
Host requirements are intentionally small: ``registry``, ``compiler``,
``jobs``, ``assemble_project``, ``_lock``, pending maps, the run supervisor,
the finalizer, and ``_max_running_trials``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..experiments.finalization import ExperimentFinalizer
from ..experiments.lifecycle import read_run_status
from ..experiments.registry import (
    ExperimentRegistry,
    _utc_now,
    normalize_instance_execution,
    normalize_launcher,
)
from .run_supervisor import RunSupervisor
from .runner_state import completed_distinct_cases

RESUMABLE_STATUSES = {"paused", "stopped", "failed"}


class ExperimentLaunchLifecycle:
    """Reusable launch-lifecycle behavior for ``ExperimentQueueManager``."""

    registry: ExperimentRegistry
    compiler: Any
    jobs: Any
    assemble_project: Callable[..., dict[str, Any]]
    _lock: Any
    _pending_plans: dict[str, Any]
    _pending_allocations: dict[str, dict[str, Any]]
    _active_allocations: dict[str, dict[str, Any]]
    _run_supervisor: RunSupervisor
    _finalizer: ExperimentFinalizer
    _wake_event: threading.Event
    _max_running_trials: int

    def _compile(self, instance: dict[str, Any]):
        self.registry._require_current(instance)
        spec = self.registry.get_spec(str(instance["experiment_spec_id"]))
        project = self.assemble_project(
            spec,
            benchmark_instance_id=str(instance["benchmark_instance_id"]),
            team_instance_id=str(instance["team_instance_id"]),
            experiment_instance=instance,
        )
        if instance.get("run_dir"):
            project.setdefault("environment", {})["run_dir"] = instance["run_dir"]
        return self.compiler.compile(project, launch_id=instance["id"])

    def _plan_for_instance(self, instance: dict[str, Any]):
        resume_from = str((instance.get("queue") or {}).get("resume_from_status") or "")
        if instance.get("status") in RESUMABLE_STATUSES or resume_from in RESUMABLE_STATUSES:
            launch_dir = str(instance.get("launch_dir") or "").strip()
            if not launch_dir:
                raise ValueError("resumable ExperimentInstance has no previous launch_dir")
            return self.jobs.existing_plan(Path(launch_dir))
        return self._compile(instance)

    @staticmethod
    def _case_completion_target(instance: dict[str, Any]) -> int | None:
        return normalize_instance_execution(instance.get("execution"))["case_completion_target"]

    @staticmethod
    def _next_segment_case_limit(instance: dict[str, Any]) -> int | None:
        return normalize_instance_execution(instance.get("execution"))["next_segment_case_limit"]

    @staticmethod
    def _completed_distinct_cases(run_dir: str | Path) -> int:
        return completed_distinct_cases(run_dir)

    def _remaining_to_case_target(self, instance: dict[str, Any], plan) -> int | None:
        target = self._case_completion_target(instance)
        if target is None:
            return None
        completed = self._completed_distinct_cases(getattr(plan, "run_dir", ""))
        return max(0, target - completed)

    def _settle_reached_case_target(self, instance: dict[str, Any], plan) -> bool:
        if self._remaining_to_case_target(instance, plan) != 0:
            return False
        queue = dict(instance.get("queue") or {})
        queue["enabled"] = False
        queue.pop("resume_from_status", None)
        run_dir = Path(str(getattr(plan, "run_dir", "") or ""))
        status_payload: dict[str, Any] = {}
        try:
            status_payload = json.loads(
                (run_dir / "run_status.json").read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            pass
        expected = max(0, int(status_payload.get("expected_trials") or 0))
        completed = max(0, int(status_payload.get("completed_distinct_cases") or 0))
        settled_status = "completed" if expected > 0 and completed >= expected else "paused"
        self.registry.update_instance(
            str(instance["id"]),
            status=settled_status,
            queue=queue,
            paused_at_utc=_utc_now() if settled_status == "paused" else None,
            finished_at_utc=_utc_now(),
            error=None,
        )
        with self._lock:
            self._pending_plans.pop(str(instance["id"]), None)
            self._pending_allocations.pop(str(instance["id"]), None)
        return True

    def _configure_segment(self, instance: dict[str, Any], plan, allocation: dict[str, Any]):
        configure = getattr(self.jobs, "configure_run_segment", None)
        if configure is None:
            return None
        segment_case_limit = self._next_segment_case_limit(instance)
        remaining_to_target = self._remaining_to_case_target(instance, plan)
        if remaining_to_target == 0:
            raise ValueError(
                f"ExperimentInstance {instance['id']!r} already reached its cumulative Case target"
            )
        if remaining_to_target is not None:
            segment_case_limit = (
                remaining_to_target
                if segment_case_limit is None
                else min(segment_case_limit, remaining_to_target)
            )
        return configure(
            plan,
            segment_case_limit=segment_case_limit,
            scheduler_trial_concurrency=max(1, int(allocation.get("max_trial_slots") or 1)),
            scheduler_trial_admission_total=max(
                0, int(allocation.get("admission_total_issued") or 0)
            ),
        )

    def _start_instance(self, instance: dict[str, Any], *, launch_origin: str, plan=None):
        instance_id = instance["id"]
        plan = plan or self._compile(instance)
        queue = dict(instance.get("queue") or {})
        queue["enabled"] = False
        queue.pop("resume_from_status", None)
        self.registry.update_instance(
            instance_id,
            status="running",
            queue=queue,
            started_at_utc=_utc_now(),
            finished_at_utc=None,
            launch_id=plan.launch_id,
            launch_dir=str(plan.launch_dir),
            run_dir=str(plan.run_dir),
            launch_origin=launch_origin,
            error=None,
            drain_requested=False,
            drain_requested_at_utc=None,
        )
        mode = normalize_launcher(instance.get("launcher"))["type"]
        try:
            state = self.jobs.launch(plan, mode=mode)
        except Exception as exc:
            self.registry.update_instance(
                instance_id,
                status="failed",
                finished_at_utc=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return plan, state

    def _resume_instance(self, instance: dict[str, Any], plan, *, launch_origin: str):
        state = self.jobs.resume(
            plan, mode=normalize_launcher(instance.get("launcher"))["type"]
        )
        queue = dict(instance.get("queue") or {})
        queue.pop("resume_from_status", None)
        queue["enabled"] = False
        self.registry.update_instance(
            instance["id"],
            status="running",
            queue=queue,
            started_at_utc=_utc_now(),
            finished_at_utc=None,
            finalized_at_utc=None,
            job_status=state.get("status"),
            run_status="starting",
            evaluation_status="pending",
            evaluation_error=None,
            return_code=None,
            error=None,
            lifecycle_consistent=True,
            lifecycle_conflict=None,
            launcher_failure_after_run=False,
            post_run_failure=None,
            launch_origin=launch_origin,
            resume_count=int(state.get("resume_count") or 0),
            resumed_at_utc=state.get("resumed_at_utc"),
            drain_requested=False,
            drain_requested_at_utc=None,
            paused_at_utc=None,
        )
        return state

    def _start_monitor(self, instance_id: str, launch_dir: Path) -> None:
        self._run_supervisor.start(instance_id, launch_dir)

    def _finish(self, instance_id: str, state: dict[str, Any]) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        updated = self._finalizer.finalize(
            instance_id,
            state,
            case_completion_target=self._case_completion_target(instance),
            finished_at_utc=_utc_now(),
        )
        self._wake_event.set()
        return updated

    def _track_running_instance(self, instance: dict[str, Any], launch_dir: Path) -> None:
        instance_id = str(instance["id"])
        with self._lock:
            if instance_id in self._active_allocations:
                return
        job_state = self.jobs.status(launch_dir)
        run_dir_text = str(instance.get("run_dir") or job_state.get("run_dir") or "").strip()
        run_dir = Path(run_dir_text) if run_dir_text else None
        current_run_status = read_run_status(job_state, fallback_run_dir=run_dir)
        active_checker = getattr(self.jobs, "is_active_state", None)
        job_is_active = (
            bool(active_checker(job_state))
            if callable(active_checker)
            else str(job_state.get("status") or "") in {"starting", "running"}
        )
        if (
            job_is_active
            and str(current_run_status.get("status") or "")
            in {"starting", "running", "draining"}
            and run_dir is not None
        ):
            archive_receipt = getattr(self.jobs, "archive_finalization_receipt", None)
            if callable(archive_receipt):
                archive_receipt(
                    run_dir,
                    resume_count=max(1, int(job_state.get("resume_count") or 0)),
                )
            instance = self.registry.update_instance(
                instance_id,
                status="running",
                job_status=str(job_state.get("status") or "running"),
                run_status=str(current_run_status.get("status") or "running"),
                evaluation_status="pending",
                evaluation_error=None,
                finalized_at_utc=None,
                finished_at_utc=None,
                lifecycle_consistent=True,
                lifecycle_conflict=None,
                launcher_failure_after_run=False,
                post_run_failure=None,
            )
        try:
            plan = self._compile(instance)
            allocation = self._allocation_for_plan(
                instance,
                plan,
                launch_origin=str(instance.get("launch_origin") or "recovered"),
            )
        except Exception as compile_error:
            try:
                allocation = self._allocation_from_launch_snapshot(
                    instance, launch_dir, compile_error=compile_error
                )
            except (OSError, TypeError, ValueError) as snapshot_error:
                allocation = {
                    "instance_id": instance_id,
                    "launch_origin": str(instance.get("launch_origin") or "recovered"),
                    "launch_dir": str(launch_dir),
                    "run_dir": str(instance.get("run_dir") or ""),
                    "max_trial_slots": 1,
                    "current_trial_slots": 1,
                    "deployment_capacities": {},
                    "recovery_warning": (
                        f"compile={type(compile_error).__name__}: {compile_error}; "
                        f"snapshot={type(snapshot_error).__name__}: {snapshot_error}"
                    ),
                }
        with self._lock:
            self._active_allocations.setdefault(instance_id, allocation)

    def _allocation_from_launch_snapshot(
        self,
        instance: dict[str, Any],
        launch_dir: Path,
        *,
        compile_error: Exception,
    ) -> dict[str, Any]:
        project_path = launch_dir / "project.json"
        project = json.loads(project_path.read_text(encoding="utf-8"))
        if not isinstance(project, dict):
            raise ValueError(f"invalid launch project snapshot: {project_path}")
        run_dir = str(
            instance.get("run_dir")
            or (project.get("environment") or {}).get("run_dir")
            or ""
        )
        plan = SimpleNamespace(
            project=project,
            launch_dir=launch_dir,
            run_dir=Path(run_dir) if run_dir else "",
        )
        allocation = self._allocation_for_plan(
            instance,
            plan,
            launch_origin=str(instance.get("launch_origin") or "recovered_snapshot"),
        )
        allocation["recovered_from_launch_snapshot"] = True
        allocation["compile_recovery_reason"] = (
            f"{type(compile_error).__name__}: {compile_error}"
        )
        return allocation

    def _allocation_for_plan(
        self,
        instance: dict[str, Any],
        plan,
        *,
        launch_origin: str,
    ) -> dict[str, Any]:
        instance_id = str(instance["id"])
        project = dict(getattr(plan, "project", {}) or {})
        runtime = dict(project.get("runtime") or {})
        policy = dict(runtime.get("concurrency_policy") or {})
        concurrency_mode = str(policy.get("mode") or "fixed")
        case_limit = max(
            1,
            int(
                policy.get("maximum")
                or runtime.get("trial_concurrency")
                or runtime.get("case_concurrency")
                or 1
            ),
        )
        benchmark = dict(project.get("benchmark") or {})
        raw_cases = benchmark.get("n")
        trials_per_case = max(1, int(runtime.get("trials_per_case") or 1))
        if isinstance(raw_cases, int) or (
            isinstance(raw_cases, str) and raw_cases.isdigit()
        ):
            total_work = max(1, int(raw_cases) * trials_per_case)
            max_trial_slots = min(case_limit, total_work)
        else:
            max_trial_slots = case_limit
        deployment_capacities: dict[str, int] = {}
        deployment_admission_policies: dict[str, dict[str, Any]] = {}
        for deployment in project.get("deployment_instances") or []:
            deployment_id = str(deployment.get("id") or "").strip()
            if not deployment_id:
                continue
            limits = dict(deployment.get("request_limits") or {})
            try:
                capacity = int(limits.get("max_concurrency") or 0)
            except (TypeError, ValueError):
                capacity = 0
            deployment_capacities[deployment_id] = (
                capacity if capacity > 0 else max_trial_slots
            )
            if str(deployment.get("kind") or "") == "vllm":
                deployment_admission_policies[deployment_id] = dict(
                    deployment.get("admission_control") or {}
                )
        target_remaining = self._remaining_to_case_target(instance, plan)
        segment_remaining = self._next_segment_case_limit(instance)
        execution_remaining = segment_remaining
        if target_remaining is not None:
            execution_remaining = (
                target_remaining
                if execution_remaining is None
                else min(execution_remaining, target_remaining)
            )
        if execution_remaining is not None:
            max_trial_slots = min(max_trial_slots, max(1, execution_remaining))
        if concurrency_mode == "fixed" and max_trial_slots > self._max_running_trials:
            raise ValueError(
                f"fixed Trial concurrency {max_trial_slots} exceeds Scheduler running "
                f"Trial hard limit {self._max_running_trials}"
            )
        completed_cases = self._completed_distinct_cases(getattr(plan, "run_dir", ""))
        return {
            "instance_id": instance_id,
            "launch_origin": launch_origin,
            "launch_dir": str(getattr(plan, "launch_dir", "")),
            "run_dir": str(getattr(plan, "run_dir", "")),
            "max_trial_slots": max_trial_slots,
            "trial_concurrency_mode": concurrency_mode,
            "current_trial_slots": 0,
            "in_flight_trial_slots": 0,
            "admission_total_issued": 0,
            "runner_admission_total_observed": 0,
            "segment_started_trials": 0,
            "active_trials": [],
            "trial_admission_protocol": True,
            "segment_case_limit": segment_remaining,
            "segment_case_target": (
                completed_cases + segment_remaining
                if segment_remaining is not None
                else None
            ),
            "case_completion_target": self._case_completion_target(instance),
            "deployment_capacities": deployment_capacities,
            "deployment_admission_policies": deployment_admission_policies,
        }
