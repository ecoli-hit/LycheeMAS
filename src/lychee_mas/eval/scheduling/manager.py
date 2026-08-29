"""Experiment queue lifecycle and shared Trial scheduling."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..deployments.pressure import DeploymentHealthMonitor
from ..experiments.finalization import ExperimentFinalizer
from ..experiments.progress import ExperimentProgressProjector
from ..experiments.registry import (
    ExperimentRegistry,
    _utc_now,
    normalize_instance_execution,
    normalize_launcher,
)
from .admission import rebalance_trial_admissions
from .launch_lifecycle import RESUMABLE_STATUSES, ExperimentLaunchLifecycle
from .pressure import summarize_api_pressure, summarize_vllm_pressure
from .projections import resource_usage as project_resource_usage
from .projections import running_trial_pool as project_running_trial_pool
from .projections import trial_queue_snapshot as project_trial_queue_snapshot
from .projections import trial_sequences as project_trial_sequences
from .run_supervisor import RunSupervisor
from .runner_state import (
    observed_trial_state,
    remaining_work,
)

_RESUMABLE_STATUSES = RESUMABLE_STATUSES
_DEFAULT_MAX_PARALLEL_INSTANCES = 16
_MAX_PARALLEL_INSTANCES = 64
_DEFAULT_MAX_RUNNING_TRIALS = 64
_MAX_RUNNING_TRIALS = 4096
_MAX_NEW_TRIALS_PER_TICK = 4
_TRIAL_ADMISSION_SAMPLE_INTERVAL_S = 2.0
_SCHEDULER_POLL_INTERVAL_S = 2.0


class ExperimentQueueManager(ExperimentLaunchLifecycle):
    """Run queued ExperimentInstances against shared deployment capacity pools."""

    def __init__(
        self,
        registry: ExperimentRegistry,
        compiler,
        jobs,
        assemble_project,
        deployment_registry=None,
        deployment_health_monitor=None,
    ) -> None:
        self.registry = registry
        self.compiler = compiler
        self.jobs = jobs
        self.assemble_project = assemble_project
        self.deployment_registry = deployment_registry
        self._deployment_health_monitor = (
            deployment_health_monitor or DeploymentHealthMonitor(deployment_registry)
        )
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_after_current = False
        self._max_parallel_instances = _DEFAULT_MAX_PARALLEL_INSTANCES
        self._max_running_trials = _DEFAULT_MAX_RUNNING_TRIALS
        self._active_allocations: dict[str, dict[str, Any]] = {}
        self._pending_plans: dict[str, Any] = {}
        self._pending_allocations: dict[str, dict[str, Any]] = {}
        self._deployment_health: dict[str, dict[str, Any]] = {}
        self._wake_event = threading.Event()
        self._last_trial_admission_monotonic = 0.0
        self._progress_projector = ExperimentProgressProjector(
            registry,
            jobs,
            finish=self._finish,
            case_target=self._case_completion_target,
        )
        self._finalizer = ExperimentFinalizer(registry)
        self._run_supervisor = RunSupervisor(
            jobs,
            on_finished=self._finish,
            on_released=self._release_allocation,
        )

    def start(
        self,
        *,
        max_parallel_instances: int = _DEFAULT_MAX_PARALLEL_INSTANCES,
        max_running_trials: int = _DEFAULT_MAX_RUNNING_TRIALS,
    ) -> dict[str, Any]:
        try:
            parallel_limit = int(max_parallel_instances)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_parallel_instances must be an integer") from exc
        if not 1 <= parallel_limit <= _MAX_PARALLEL_INSTANCES:
            raise ValueError(
                f"max_parallel_instances must be between 1 and {_MAX_PARALLEL_INSTANCES}"
            )
        try:
            running_trial_limit = int(max_running_trials)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_running_trials must be an integer") from exc
        if not 1 <= running_trial_limit <= _MAX_RUNNING_TRIALS:
            raise ValueError(
                f"max_running_trials must be between 1 and {_MAX_RUNNING_TRIALS}"
            )
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._stop_after_current = False
                self._max_parallel_instances = parallel_limit
                self._max_running_trials = running_trial_limit
                self._wake_event.set()
                return self.status()
            queued_ids = [
                item["id"]
                for item in self.registry.instances()
                if item.get("status") == "queued" or bool((item.get("queue") or {}).get("enabled"))
            ]
            recoverable_running_ids = [
                item["id"]
                for item in self.registry.instances()
                if item.get("status") == "running" and item.get("launch_dir")
            ]
            if not queued_ids and not recoverable_running_ids:
                raise ValueError(
                    "没有 queued ExperimentInstance，也没有可恢复的 running ExperimentInstance"
                )
            self._stop_after_current = False
            self._max_parallel_instances = parallel_limit
            self._max_running_trials = running_trial_limit
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="eval-studio-experiment-queue", daemon=True
            )
            self._thread.start()
        return {
            **self.status(),
            "accepted_instance_ids": queued_ids,
            "recoverable_running_instance_ids": recoverable_running_ids,
        }

    def stop_after_current(self) -> dict[str, Any]:
        """Stop admitting queued runners while continuing to schedule active Trials."""

        self._stop_after_current = True
        self._wake_event.set()
        return self.status()

    def enqueue(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        if instance.get("status") not in {"ready", "running", *_RESUMABLE_STATUSES}:
            raise ValueError(
                "only a ready, draining, paused, stopped, or failed ExperimentInstance "
                "can be queued"
            )
        if instance.get("status") == "running":
            queued = self.registry.enqueue(instance_id)
            self._wake_event.set()
            return queued
        plan = self._plan_for_instance(instance)
        if self._remaining_to_case_target(instance, plan) == 0:
            raise ValueError(
                f"ExperimentInstance {instance_id!r} already reached its cumulative Case target"
            )
        mode = normalize_launcher(instance.get("launcher"))["type"]
        self.jobs.validate(plan, mode=mode)
        allocation = self._allocation_for_plan(instance, plan, launch_origin="queue")
        with self._lock:
            queued = self.registry.enqueue(instance_id)
            self._pending_plans[instance_id] = plan
            self._pending_allocations[instance_id] = allocation
            self._wake_event.set()
            return queued

    def dequeue(self, instance_id: str) -> dict[str, Any]:
        with self._lock:
            instance = self.registry.get_instance(instance_id)
            if instance_id in self._active_allocations and not bool(
                (instance.get("queue") or {}).get("enabled")
            ):
                raise ValueError("the active ExperimentInstance cannot be removed from the queue")
            dequeued = self.registry.dequeue(instance_id)
            self._pending_plans.pop(instance_id, None)
            self._pending_allocations.pop(instance_id, None)
            return dequeued

    def plan_instance(self, instance_id: str):
        return self._compile(self.registry.get_instance(instance_id))

    def launch_now(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        if instance.get("status") != "ready":
            raise ValueError("only a ready ExperimentInstance can be started")
        plan = self._compile(instance)
        allocation = self._allocation_for_plan(instance, plan, launch_origin="immediate")
        with self._lock:
            admitted, reason = self._can_admit_locked(allocation)
            if not admitted:
                raise ValueError(
                    f"ExperimentInstance cannot start now: {reason}; add it to the queue"
                )
            self._active_allocations[instance_id] = allocation
        self._rebalance_active_allocations()
        try:
            self._configure_segment(instance, plan, allocation)
            plan, state = self._start_instance(instance, launch_origin="immediate", plan=plan)
        except Exception:
            self._release_allocation(instance_id)
            raise
        self._start_monitor(instance_id, plan.launch_dir)
        self._ensure_scheduler_thread()
        return {
            "plan": plan.as_dict(),
            "job": state,
            "instance": self.registry.get_instance(instance_id),
        }

    def stop_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        if instance.get("status") != "running":
            raise ValueError("only a running ExperimentInstance can be stopped")
        launch_dir = str(instance.get("launch_dir") or "").strip()
        if not launch_dir:
            raise ValueError("running ExperimentInstance has no launch_dir")
        state = self.jobs.stop(Path(launch_dir))
        queue = dict(instance.get("queue") or {})
        queue["enabled"] = False
        queue.pop("resume_from_status", None)
        finished_at_utc = _utc_now()
        updated = self._finalizer.project_terminal_lifecycle(
            instance_id,
            state,
            case_completion_target=self._case_completion_target(instance),
            finished_at_utc=finished_at_utc,
            queue=queue,
            evaluation_status="pending",
        )
        self._release_allocation(instance_id)
        return {"instance": updated, "job": state}

    def resume_instance(self, instance_id: str) -> dict[str, Any]:
        """Resume the same Run, preserving successful Trials and frozen launch config."""

        instance = self.registry.get_instance(instance_id)
        if instance.get("status") not in _RESUMABLE_STATUSES:
            raise ValueError("only a paused, stopped, or failed ExperimentInstance can be resumed")
        launch_dir_text = str(instance.get("launch_dir") or "").strip()
        if not launch_dir_text:
            raise ValueError("ExperimentInstance has no previous launch_dir")
        plan = self.jobs.existing_plan(Path(launch_dir_text))
        if plan.launch_id != instance_id:
            raise ValueError(f"launch snapshot belongs to {plan.launch_id!r}, not {instance_id!r}")
        allocation = self._allocation_for_plan(instance, plan, launch_origin="resume")
        with self._lock:
            admitted, reason = self._can_admit_locked(allocation)
            if not admitted:
                raise ValueError(
                    f"ExperimentInstance cannot resume now: {reason}; stop another Run first"
                )
            self._active_allocations[instance_id] = allocation
        self._rebalance_active_allocations()
        try:
            self._configure_segment(instance, plan, allocation)
            state = self.jobs.resume(
                plan,
                mode=normalize_launcher(instance.get("launcher"))["type"],
            )
            updated = self.registry.update_instance(
                instance_id,
                status="running",
                started_at_utc=_utc_now(),
                finished_at_utc=None,
                job_status=state.get("status"),
                return_code=None,
                error=None,
                launch_origin="resume",
                resume_count=int(state.get("resume_count") or 0),
                resumed_at_utc=state.get("resumed_at_utc"),
                drain_requested=False,
                drain_requested_at_utc=None,
                paused_at_utc=None,
            )
        except Exception:
            self._release_allocation(instance_id)
            raise
        self._start_monitor(instance_id, plan.launch_dir)
        self._ensure_scheduler_thread()
        return {"plan": plan.as_dict(), "job": state, "instance": updated}

    def _ensure_scheduler_thread(self) -> None:
        """Keep immediate and resumed Runs on the shared Trial admission loop."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._wake_event.set()
                return
            self._stop_after_current = False
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="eval-studio-experiment-queue",
                daemon=True,
            )
            self._thread.start()

    def drain_instance(self, instance_id: str) -> dict[str, Any]:
        """Finish active Trials, then stop this ExperimentInstance."""

        instance = self.registry.get_instance(instance_id)
        if instance.get("status") != "running":
            raise ValueError("only a running ExperimentInstance can be drained")
        launch_dir = str(instance.get("launch_dir") or "").strip()
        if not launch_dir:
            raise ValueError("running ExperimentInstance has no launch_dir")
        state = self.jobs.request_drain(Path(launch_dir))
        updated = self.registry.update_instance(
            instance_id,
            drain_requested=True,
            drain_requested_at_utc=state.get("drain_requested_at_utc"),
        )
        return {"instance": updated, "job": state}

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        launch_dir = str(instance.get("launch_dir") or "").strip()
        if launch_dir and (Path(launch_dir) / "job.json").is_file():
            state = self.jobs.status(Path(launch_dir))
            if self.jobs.is_active_state(state):
                raise ValueError(
                    "an ExperimentInstance with an active launch cannot be deleted; stop it first"
                )
        return self.registry.delete_instance(instance_id)

    def reconcile(self) -> dict[str, Any]:
        """Reattach persisted instances to live launches after a Studio restart."""

        recovered: list[str] = []
        finished: list[str] = []
        for instance in self.registry.instances():
            # Only an instance persisted as running can own a live launch after a
            # Studio restart. Historical terminal instances are analyzed on
            # demand; touching every launch here made startup proportional to the
            # complete Run archive and could trigger hundreds of evaluations.
            if instance.get("status") != "running":
                continue
            launch_dir_text = str(instance.get("launch_dir") or "").strip()
            if not launch_dir_text:
                continue
            launch_dir = Path(launch_dir_text)
            try:
                state = self.jobs.status(launch_dir)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                continue
            if self.jobs.is_active_state(state):
                if instance.get("status") != "running":
                    self.registry.update_instance(instance["id"], status="running")
                self._track_running_instance(instance, launch_dir)
                self._start_monitor(instance["id"], launch_dir)
                recovered.append(instance["id"])
            elif self._finalizer.needs_reconciliation(instance):
                self._finish(instance["id"], state)
                finished.append(instance["id"])
        if recovered:
            self._ensure_scheduler_thread()
        orphans = self.orphaned_launches(tail=0)
        return {
            "recovered_instance_ids": recovered,
            "finished_instance_ids": finished,
            "orphaned_launch_ids": [item["launch_id"] for item in orphans],
        }

    def orphaned_launches(self, *, tail: int = 12) -> list[dict[str, Any]]:
        """Expose active launch processes that no ExperimentInstance currently owns."""

        registered_launch_ids = {
            str(item.get("launch_id") or item["id"]) for item in self.registry.instances()
        }
        rows: list[dict[str, Any]] = []
        for state in self.jobs.launches(active_only=True):
            launch_id = str(state.get("launch_id") or "")
            if launch_id in registered_launch_ids or not self.jobs.is_active_state(state):
                continue
            launch_dir = Path(str(state["launch_dir"]))
            snapshot = self.jobs.experiment_instance_snapshot(launch_dir)
            try:
                progress = self.jobs.progress(launch_dir, tail=tail)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                progress = None
            rows.append(
                {
                    "launch_id": launch_id,
                    "status": "orphaned",
                    "recoverable": bool(snapshot),
                    "experiment_instance_snapshot": snapshot or None,
                    "job": state,
                    "launch_dir": str(launch_dir),
                    "run_dir": state.get("run_dir"),
                    "progress": progress,
                }
            )
        return rows

    def stop_orphaned_launch(self, launch_id: str) -> dict[str, Any]:
        orphan = self._get_orphaned_launch(launch_id)
        return self.jobs.stop(Path(orphan["launch_dir"]))

    def recover_orphaned_launch(self, launch_id: str) -> dict[str, Any]:
        with self._lock:
            orphan = self._get_orphaned_launch(launch_id)
            snapshot = dict(orphan.get("experiment_instance_snapshot") or {})
            if not snapshot:
                raise ValueError("orphaned launch has no recoverable ExperimentInstance snapshot")
            if str(snapshot.get("id") or "") != launch_id:
                raise ValueError("orphaned launch snapshot ID does not match launch_id")
            created = self.registry.create_instance(
                instance_id=launch_id,
                spec_id=str(snapshot.get("experiment_spec_id") or ""),
                benchmark_instance_id=str(snapshot.get("benchmark_instance_id") or ""),
                team_instance_id=str(snapshot.get("team_instance_id") or ""),
                run_dir=str(orphan.get("run_dir") or snapshot.get("run_dir") or ""),
                priority=int((snapshot.get("queue") or {}).get("priority", 100)),
                launcher=normalize_launcher(snapshot.get("launcher")),
                execution=normalize_instance_execution(snapshot.get("execution")),
            )
            state = dict(orphan["job"])
            updated = self.registry.update_instance(
                created["id"],
                status="running",
                launch_id=launch_id,
                launch_dir=str(orphan["launch_dir"]),
                run_dir=str(orphan.get("run_dir") or created.get("run_dir") or ""),
                started_at_utc=state.get("created_at_utc"),
                recovered_at_utc=_utc_now(),
                launch_origin="recovered_orphan",
                error=None,
            )
        self._track_running_instance(updated, Path(str(orphan["launch_dir"])))
        self._start_monitor(updated["id"], Path(str(orphan["launch_dir"])))
        return {"instance": updated, "job": state}

    def _get_orphaned_launch(self, launch_id: str) -> dict[str, Any]:
        for item in self.orphaned_launches(tail=12):
            if item.get("launch_id") == launch_id:
                return item
        raise ValueError(f"active orphaned launch {launch_id!r} was not found")

    def status(self) -> dict[str, Any]:
        # A read-only status request must not admit work. Only the Scheduler
        # loop advances Trial dispatch so polling frequency cannot change runs.
        self._refresh_active_demands(rebalance=False)
        instances = self.registry.instances()
        running_instance_ids = [item["id"] for item in instances if item.get("status") == "running"]
        orphaned_launch_ids = [item["launch_id"] for item in self.orphaned_launches(tail=0)]
        with self._lock:
            active_ids = sorted(self._active_allocations)
            resource_usage = self._resource_usage_locked()
            running_trial_pool = self._running_trial_pool_locked()
            trial_queues = self._trial_queue_snapshot_locked(instances)
            trial_sequences = self._trial_sequences_locked(trial_queues)
            max_parallel_instances = self._max_parallel_instances
            max_running_trials = self._max_running_trials
            pressure = self._pressure_snapshot_locked(resource_usage)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "mode": "priority_trial_admission",
            "stop_after_current": self._stop_after_current,
            "max_parallel_instances": max_parallel_instances,
            "max_running_trials": max_running_trials,
            "active_instance_id": active_ids[0]
            if active_ids
            else (running_instance_ids[0] if running_instance_ids else None),
            "active_instance_ids": active_ids or running_instance_ids,
            "running_instance_ids": running_instance_ids,
            "queued_instance_ids": [
                item["id"]
                for item in instances
                if (
                    item.get("configuration_status") == "current"
                    and (
                        item.get("status") == "queued"
                        or bool((item.get("queue") or {}).get("enabled"))
                    )
                )
            ],
            "invalid_queued_instance_ids": [
                item["id"]
                for item in instances
                if (
                    item.get("configuration_status") != "current"
                    and (
                        item.get("status") == "queued"
                        or bool((item.get("queue") or {}).get("enabled"))
                    )
                )
            ],
            "refill_order": [item["instance_id"] for item in trial_queues],
            "experiment_scheduling": trial_queues,
            **trial_sequences,
            "running_trial_pool": running_trial_pool,
            "resource_usage": resource_usage,
            "pressure": pressure,
            "orphaned_launch_ids": orphaned_launch_ids,
        }

    def _pressure_snapshot_locked(
        self, resource_usage: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Return truthful live pressure for resources currently used by the pool."""

        deployments: dict[str, dict[str, Any]] = {}
        if self.deployment_registry is not None:
            try:
                deployments = {
                    str(item["id"]): item
                    for item in self.deployment_registry.instances(probe=False)
                }
            except (OSError, TypeError, ValueError):
                deployments = {}

        vllm_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        api_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for deployment_id, usage in resource_usage.items():
            deployment = deployments.get(deployment_id, {})
            kind = str(deployment.get("kind") or "")
            if kind == "vllm":
                vllm_rows.append((deployment_id, usage, deployment))
            elif kind == "api":
                api_rows.append((deployment_id, usage, deployment))
        pressure = {
            "vllm": summarize_vllm_pressure(vllm_rows),
            "api": summarize_api_pressure(api_rows),
            "gpu": self._deployment_health_monitor.gpu_pressure(),
        }
        pressure["sampled_at_unix_s"] = time.time()
        return pressure

    def instances(
        self,
        *,
        tail: int = 12,
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        """Return persisted instances enriched with recoverable launch progress."""

        return self._progress_projector.instances(tail=tail, compact=compact)

    @staticmethod
    def _compact_instance(instance: dict[str, Any]) -> dict[str, Any]:
        return ExperimentProgressProjector.compact(instance)

    def instance_progress(self, instance_id: str, *, tail: int = 40) -> dict[str, Any]:
        return self._progress_projector.instance_progress(instance_id, tail=tail)

    def _observe_instance(self, instance: dict[str, Any], *, tail: int) -> dict[str, Any]:
        return self._progress_projector.observe(instance, tail=tail)

    def _run(self) -> None:
        try:
            self._recover_interrupted_instances()
            while True:
                self._refresh_active_demands()
                with self._lock:
                    has_active = bool(self._active_allocations)
                if self._stop_after_current and not has_active:
                    return
                queued = (
                    []
                    if self._stop_after_current
                    else [
                        item
                        for item in self.registry.instances()
                        if item.get("status") == "queued"
                        and item.get("configuration_status") == "current"
                    ]
                )
                launched = False
                planned: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
                for instance in queued:
                    if self._stop_after_current:
                        break
                    try:
                        with self._lock:
                            plan = self._pending_plans.get(instance["id"])
                            allocation = self._pending_allocations.get(instance["id"])
                        if plan is None or allocation is None:
                            plan = self._plan_for_instance(instance)
                            if self._settle_reached_case_target(instance, plan):
                                launched = True
                                continue
                            allocation = self._allocation_for_plan(
                                instance, plan, launch_origin="queue"
                            )
                            with self._lock:
                                self._pending_plans[instance["id"]] = plan
                                self._pending_allocations[instance["id"]] = allocation
                        planned.append((instance, plan, allocation))
                    except Exception as exc:
                        with self._lock:
                            self._pending_plans.pop(instance["id"], None)
                            self._pending_allocations.pop(instance["id"], None)
                        self.registry.update_instance(
                            instance["id"],
                            status="failed",
                            finished_at_utc=_utc_now(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        launched = True
                        continue

                # Admit concrete Trial starts across active Runs and queued
                # Runs in persistent priority order. A queued Run is launched
                # only after receiving its first start permit.
                self._rebalance_active_allocations()
                for instance, plan, allocation in planned:
                    if self._stop_after_current:
                        break
                    with self._lock:
                        if len(self._active_allocations) >= self._max_parallel_instances:
                            continue
                        if int(allocation.get("current_trial_slots") or 0) < 1:
                            continue
                        self._active_allocations[instance["id"]] = allocation
                        self._pending_plans.pop(instance["id"], None)
                        self._pending_allocations.pop(instance["id"], None)
                    self._rebalance_active_allocations()
                    try:
                        self._configure_segment(instance, plan, allocation)
                        resume_from = str(
                            (instance.get("queue") or {}).get("resume_from_status") or ""
                        )
                        if resume_from in _RESUMABLE_STATUSES:
                            _state = self._resume_instance(
                                instance,
                                plan,
                                launch_origin="queue_resume",
                            )
                        else:
                            plan, _state = self._start_instance(
                                instance, launch_origin="queue", plan=plan
                            )
                    except Exception as exc:
                        self._release_allocation(instance["id"])
                        queue = dict(instance.get("queue") or {})
                        queue["enabled"] = False
                        queue.pop("resume_from_status", None)
                        self.registry.update_instance(
                            instance["id"],
                            status="failed",
                            queue=queue,
                            finished_at_utc=_utc_now(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        launched = True
                        continue
                    self._start_monitor(instance["id"], plan.launch_dir)
                    launched = True

                remaining_queued = any(
                    item.get("status") == "queued" or bool((item.get("queue") or {}).get("enabled"))
                    for item in self.registry.instances()
                )
                with self._lock:
                    has_active = bool(self._active_allocations)
                if not remaining_queued and not has_active:
                    return
                self._wake_event.wait(0.05 if launched else _SCHEDULER_POLL_INTERVAL_S)
                self._wake_event.clear()
        finally:
            self._wake_event.clear()

    def _recover_interrupted_instances(self) -> None:
        for item in self.registry.instances():
            if item.get("status") != "running":
                continue
            launch_dir = item.get("launch_dir")
            if launch_dir:
                try:
                    state = self.jobs.status(Path(launch_dir))
                except (FileNotFoundError, ValueError):
                    state = {}
                if state.get("status") in {"starting", "running"}:
                    self._track_running_instance(item, Path(launch_dir))
                    self._start_monitor(item["id"], Path(launch_dir))
                    continue
                if state:
                    self._finish(item["id"], state)
                    continue
            self.registry.update_instance(
                item["id"],
                status="failed",
                finished_at_utc=_utc_now(),
                error="launch state could not be recovered",
            )

    def _refresh_active_demands(self, *, rebalance: bool = True) -> None:
        with self._lock:
            allocations = [dict(item) for item in self._active_allocations.values()]
        for allocation in allocations:
            remaining = remaining_work(allocation)
            observed = observed_trial_state(allocation)
            instance_id = str(allocation["instance_id"])
            with self._lock:
                stored = self._active_allocations.get(instance_id)
                if stored is not None:
                    if remaining is not None:
                        stored["remaining_work"] = max(0, remaining)
                    stored["concurrency_authority"] = observed["concurrency_authority"]
                    if observed["concurrency_authority"] == "scheduler":
                        stored.pop("observed_trial_slot_demand", None)
                    elif observed["requested_trial_slots"] is not None:
                        stored["observed_trial_slot_demand"] = max(
                            0, int(observed["requested_trial_slots"])
                        )
                    stored["in_flight_trial_slots"] = max(0, int(observed["in_flight_trial_slots"]))
                    stored["active_trials"] = list(observed.get("active_trials") or [])
                    stored["segment_started_trials"] = max(
                        0,
                        int(observed.get("segment_started_trials") or 0),
                    )
                    stored["runner_admission_total_observed"] = max(
                        int(observed.get("runner_admission_total_observed") or 0),
                        int(observed.get("segment_started_trials") or 0),
                    )
                    stored["admission_total_issued"] = max(
                        int(stored.get("admission_total_issued") or 0),
                        int(observed.get("admission_total_issued") or 0),
                        int(observed.get("segment_started_trials") or 0),
                    )
                    stored["trial_admission_protocol"] = bool(
                        observed.get("trial_admission_protocol")
                    )
                    stored["runner_status"] = observed["runner_status"]
        self._update_deployment_health()
        if rebalance:
            self._rebalance_active_allocations()

    def _can_admit_locked(self, allocation: dict[str, Any]) -> tuple[bool, str]:
        if len(self._active_allocations) >= self._max_parallel_instances:
            return False, "parallel ExperimentInstance limit reached"
        global_used = sum(
            max(0, int(item.get("current_trial_slots") or 0))
            for item in self._active_allocations.values()
            if item is not allocation
        )
        if global_used >= self._max_running_trials:
            return (
                False,
                f"Scheduler running Trial hard limit reached: "
                f"{global_used}/{self._max_running_trials}",
            )
        # Fixed concurrency is guaranteed only after an ExperimentInstance has
        # entered the running pool. Initial admission still obeys pressure.
        for deployment_id in allocation.get("deployment_capacities", {}):
            blocked_reason = self._deployment_admission_block_reason_locked(deployment_id)
            if blocked_reason:
                return False, f"DeploymentInstance {deployment_id} {blocked_reason}"
        return True, "admitted"

    def _allocation_pressure_block_reason_locked(
        self,
        allocation: dict[str, Any],
    ) -> str | None:
        for deployment_id in allocation.get("deployment_capacities", {}):
            blocked_reason = self._deployment_admission_block_reason_locked(deployment_id)
            if blocked_reason:
                return f"{deployment_id}: {blocked_reason}"
        return None

    def _running_pool_pressure_block_reason_locked(
        self,
        allocations: list[dict[str, Any]],
        active_ids: set[str],
    ) -> str | None:
        """Return the first overloaded resource currently used by the running pool."""

        for allocation in allocations:
            if str(allocation["instance_id"]) not in active_ids:
                continue
            blocked_reason = self._allocation_pressure_block_reason_locked(allocation)
            if blocked_reason:
                return blocked_reason
        return None

    def _deployment_admission_block_reason_locked(self, deployment_id: str) -> str | None:
        health = self._deployment_health.get(deployment_id) or {}
        if health.get("vllm_admission_blocked"):
            return "is waiting for vLLM pressure relief"
        if health.get("gpu_admission_blocked"):
            return "is waiting for GPU pressure relief"
        return None

    def _rebalance_active_allocations(self) -> None:
        """Apply one pure admission-policy tick, then persist Runner permits."""

        with self._lock:
            decision = rebalance_trial_admissions(
                active_allocations=self._active_allocations,
                pending_allocations=self._pending_allocations,
                instances={item["id"]: item for item in self.registry.instances()},
                scheduler_accepting_pending=bool(
                    self._thread and self._thread.is_alive() and not self._stop_after_current
                ),
                max_parallel_instances=self._max_parallel_instances,
                max_running_trials=self._max_running_trials,
                now_monotonic=time.monotonic(),
                last_admission_monotonic=self._last_trial_admission_monotonic,
                sample_interval_s=_TRIAL_ADMISSION_SAMPLE_INTERVAL_S,
                max_new_trials_per_tick=_MAX_NEW_TRIALS_PER_TICK,
                allocation_pressure_block_reason=
                    self._allocation_pressure_block_reason_locked,
                running_pool_pressure_block_reason=
                    self._running_pool_pressure_block_reason_locked,
            )
            if decision.admitted_at_monotonic is not None:
                self._last_trial_admission_monotonic = decision.admitted_at_monotonic

        for (
            _instance_id,
            launch_dir,
            current,
            supports_admission,
            admission_total,
        ) in decision.updates:
            if not launch_dir:
                continue
            setter = getattr(
                self.jobs,
                "set_trial_admission_total"
                if supports_admission
                else "set_trial_concurrency_limit",
                None,
            )
            if setter is None:
                continue
            try:
                setter(
                    Path(launch_dir),
                    admission_total if supports_admission else current,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError):
                # The segment request carries the initial allocation until the
                # launcher has materialized job.json and run_status.json.
                pass

    def _resource_usage_locked(self) -> dict[str, dict[str, Any]]:
        return project_resource_usage(self._active_allocations, self._deployment_health)

    def _running_trial_pool_locked(self) -> dict[str, int]:
        return project_running_trial_pool(
            self._active_allocations,
            self._pending_allocations,
            hard_limit=self._max_running_trials,
        )

    def _trial_queue_snapshot_locked(
        self,
        instances: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return project_trial_queue_snapshot(
            self._active_allocations,
            self._pending_allocations,
            instances,
        )

    @staticmethod
    def _trial_sequences_locked(
        queues: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        return project_trial_sequences(queues)

    def _effective_deployment_capacity_locked(
        self,
        deployment_id: str,
        hard_capacity: int,
    ) -> int:
        health = self._deployment_health.get(deployment_id) or {}
        return max(
            1,
            min(
                int(hard_capacity),
                int(health.get("effective_capacity") or hard_capacity),
            ),
        )

    def _update_deployment_health(self) -> None:
        """Refresh deployment feedback without embedding provider logic here."""

        with self._lock:
            allocations = [
                dict(item)
                for item in {
                    **self._pending_allocations,
                    **self._active_allocations,
                }.values()
            ]
            current_health = {
                key: dict(value) for key, value in self._deployment_health.items()
            }
        next_health = self._deployment_health_monitor.update(
            allocations,
            current_health,
        )
        with self._lock:
            self._deployment_health = next_health

    def _release_allocation(self, instance_id: str) -> None:
        with self._lock:
            removed = self._active_allocations.pop(instance_id, None)
        if removed is not None:
            self._rebalance_active_allocations()
            self._wake_event.set()
