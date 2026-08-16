"""Persistent ExperimentSpec and queueable ExperimentInstance registries."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from .lifecycle import bind_instance_to_spec, lifecycle_error, utc_now

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STATUSES = {"ready", "queued", "running", "completed", "failed", "stopped"}
_LAUNCHER_TYPES = {"subprocess", "new_tmux_session", "existing_tmux_session"}
_DEFAULT_MAX_PARALLEL_INSTANCES = 8
_MAX_PARALLEL_INSTANCES = 64
_SCHEDULER_POLL_INTERVAL_S = 2.0


def _utc_now() -> str:
    return utc_now()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_launcher(value: dict[str, Any] | None) -> dict[str, Any]:
    launcher = dict(value or {})
    launcher_type = str(launcher.get("type") or "subprocess")
    if launcher_type not in _LAUNCHER_TYPES:
        raise ValueError(
            "ExperimentInstance launcher.type must be subprocess, new_tmux_session, "
            "or existing_tmux_session"
        )
    result = {"type": launcher_type}
    if launcher_type in {"new_tmux_session", "existing_tmux_session"}:
        session = str(launcher.get("tmux_session") or "lychee-eval")
        window = str(launcher.get("tmux_window") or "benchmark")
        if not _SAFE_ID.fullmatch(session) or not _SAFE_ID.fullmatch(window):
            raise ValueError(
                "tmux session and window may contain only letters, numbers, '.', '_' and '-'"
            )
        result.update(tmux_session=session, tmux_window=window)
    return result


class ExperimentRegistry:
    """Persist reusable recipes and concrete queue entries under configs/."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/experiments"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        rows = []
        for item in self._all(self.spec_root):
            try:
                normalized = self.normalize_spec(item)
            except (TypeError, ValueError):
                continue
            rows.append({**normalized, "registry_path": item["registry_path"]})
        return rows

    def instances(self) -> list[dict[str, Any]]:
        specs = {item["id"]: item for item in self.specs()}
        return sorted(
            [
                self._with_lifecycle_status(
                    item,
                    specs.get(str(item.get("experiment_spec_id") or "")),
                )
                for item in self._all(self.instance_root)
                if int(item.get("schema_version") or 0) == 3
                and item.get("experiment_spec_id")
                and item.get("benchmark_instance_id")
                and item.get("team_instance_id")
            ],
            key=lambda item: (
                int(item.get("queue", {}).get("priority", 100)),
                str(item.get("created_at_utc") or ""),
                item["id"],
            ),
        )

    @staticmethod
    def _all(root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []
        rows = []
        for path in sorted(root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict) or not _SAFE_ID.fullmatch(str(value.get("id") or "")):
                continue
            rows.append({**value, "registry_path": str(path)})
        return rows

    def default_spec(self) -> dict[str, Any]:
        from .environment import readme_environment

        environment = readme_environment(self.repo_root)
        return {
            "schema_version": 3,
            "id": "eval-experiment",
            "benchmark": {
                "benchmark_spec_id": "gsm8k",
                "runnable_task": "gsm8k",
                "cases": 10,
                "start_index": 0,
            },
            "team_spec_id": "reason",
            "runtime": {
                "method": "none",
                "samples": 1,
                "max_rounds": 4,
                "max_turns": 20,
                "max_model_calls_per_case": None,
                "max_case_retries": 0,
                "on_case_error": "continue",
                "max_new_tokens": 16384,
                "max_input_tokens": None,
                "min_output_reserve_tokens": 2048,
                "min_thinking_reserve_tokens": 0,
                "max_thinking_budget_tokens": None,
                "min_final_reserve_tokens": 1024,
                "safety_margin_tokens": 256,
                "case_concurrency": 1,
                "concurrency_policy": {
                    "mode": "fixed",
                    "initial": 1,
                    "minimum": 1,
                    "maximum": 1,
                    "increase_step": 1,
                    "decrease_factor": 0.5,
                    "control_window_cases": 1,
                },
                "seed": 0,
                "do_sample": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": None,
                "min_p": None,
                "presence_penalty": None,
                "repetition_penalty": 1.0,
                "code_executor": "docker",
                "code_timeout": 60,
                "work_root": "runs/lychee_tool_workspaces",
                "web_headless": True,
                "save_screenshots": False,
                "docker_image": None,
                "trace_model_calls": True,
                "trace_detail_level": "compact",
            },
            "observability": {
                "collect_vllm_metrics": True,
                "vllm_metrics_interval_s": 5.0,
            },
            "network": {
                "mode": "benchmark_default",
                "proxy_url": "",
                "no_proxy": "127.0.0.1,localhost,::1",
                "targets": {
                    "downloads": False,
                    "web_surfer": False,
                    "code_executor": False,
                    "model_backend": False,
                },
                "docker_bridge_host": "172.17.0.1",
                "container_proxy_port": 17897,
                "probe_url": "https://www.google.com/generate_204",
            },
            "environment": {
                "repo_root": str(self.repo_root),
                "python": environment["python"],
                "conda_sh": environment["conda_sh"],
                "conda_env": environment["conda_env"],
                "raw_root": str(self.repo_root / "data/benchmarks/raw"),
                "prepared_root": str(self.repo_root / "data/benchmarks/prepared"),
                "models_root": str(self.repo_root / "models"),
                "runs_root": str(self.repo_root / "runs/benchmarks"),
            },
        }

    def normalize_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("ExperimentSpec must be a JSON object")
        spec = self.default_spec()
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(spec.get(key), dict):
                spec[key] = {**spec[key], **item}
            else:
                spec[key] = item
        if int(spec.get("schema_version") or 0) != 3:
            raise ValueError("ExperimentSpec requires schema_version 3")
        spec_id = str(spec.get("id") or "")
        self._validate_id(spec_id)
        benchmark = dict(spec.get("benchmark") or {})
        benchmark_spec_id = str(benchmark.get("benchmark_spec_id") or "")
        runnable_task = str(benchmark.get("runnable_task") or "")
        self._validate_id(benchmark_spec_id)
        if not runnable_task:
            raise ValueError("ExperimentSpec requires benchmark.runnable_task")
        cases = benchmark.get("cases", 10)
        if isinstance(cases, str) and cases.strip().lower() == "all":
            cases = "all"
        else:
            try:
                cases = int(cases)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ExperimentSpec benchmark.cases must be positive or 'all'"
                ) from exc
        if cases != "all" and cases < 1:
            raise ValueError("ExperimentSpec benchmark.cases must be positive or 'all'")
        start_index = int(benchmark.get("start_index") or 0)
        if start_index < 0:
            raise ValueError("ExperimentSpec benchmark.start_index must be non-negative")
        team_spec_id = str(spec.get("team_spec_id") or "")
        self._validate_id(team_spec_id)
        normalized_benchmark = {
            "benchmark_spec_id": benchmark_spec_id,
            "runnable_task": runnable_task,
            "cases": cases if cases == "all" else int(cases),
            "start_index": start_index,
        }
        if benchmark.get("scoring_profile"):
            normalized_benchmark["scoring_profile"] = str(benchmark["scoring_profile"])
        normalized_environment = dict(spec.get("environment") or {})
        normalized_environment.pop("run_dir", None)
        runtime = dict(spec.get("runtime") or {})
        try:
            runtime["samples"] = int(runtime.get("samples", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("ExperimentSpec runtime.samples must be positive") from exc
        if runtime["samples"] < 1:
            raise ValueError("ExperimentSpec runtime.samples must be positive")
        try:
            runtime["max_case_retries"] = int(runtime.get("max_case_retries", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ExperimentSpec runtime.max_case_retries must be non-negative"
            ) from exc
        if runtime["max_case_retries"] < 0:
            raise ValueError(
                "ExperimentSpec runtime.max_case_retries must be non-negative"
            )
        runtime["do_sample"] = bool(runtime.get("do_sample", False))
        for key in ("temperature", "top_p", "repetition_penalty"):
            try:
                runtime[key] = float(runtime[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"ExperimentSpec runtime.{key} must be numeric") from exc
        if runtime["temperature"] < 0:
            raise ValueError("ExperimentSpec runtime.temperature must be non-negative")
        if not 0.0 < runtime["top_p"] <= 1.0:
            raise ValueError(
                "ExperimentSpec runtime.top_p must be greater than 0 and at most 1"
            )
        if runtime["repetition_penalty"] <= 0:
            raise ValueError("ExperimentSpec runtime.repetition_penalty must be positive")
        if runtime.get("top_k") in (None, ""):
            runtime["top_k"] = None
        else:
            runtime["top_k"] = int(runtime["top_k"])
            if runtime["top_k"] < 1:
                raise ValueError("ExperimentSpec runtime.top_k must be positive or null")
        for key in ("min_p", "presence_penalty"):
            if runtime.get(key) in (None, ""):
                runtime[key] = None
            else:
                runtime[key] = float(runtime[key])
        if runtime["min_p"] is not None and not 0.0 <= runtime["min_p"] <= 1.0:
            raise ValueError("ExperimentSpec runtime.min_p must be between 0 and 1")
        if runtime["presence_penalty"] is not None and not (
            -2.0 <= runtime["presence_penalty"] <= 2.0
        ):
            raise ValueError(
                "ExperimentSpec runtime.presence_penalty must be between -2 and 2"
            )
        runtime["on_case_error"] = str(runtime.get("on_case_error") or "continue")
        if runtime["on_case_error"] not in {"continue", "fail-fast"}:
            raise ValueError(
                "ExperimentSpec runtime.on_case_error must be continue or fail-fast"
            )
        from ...runtime.concurrency import normalize_concurrency_policy

        runtime["case_concurrency"] = int(runtime.get("case_concurrency", 1))
        runtime["concurrency_policy"] = normalize_concurrency_policy(
            runtime.get("concurrency_policy"),
            configured_concurrency=runtime["case_concurrency"],
        )
        for key, default in (("max_rounds", 4), ("max_turns", 20)):
            try:
                runtime[key] = int(runtime.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ExperimentSpec runtime.{key} must be positive") from exc
            if runtime[key] < 1:
                raise ValueError(f"ExperimentSpec runtime.{key} must be positive")
        raw_model_call_limit = runtime.get("max_model_calls_per_case")
        if isinstance(raw_model_call_limit, str) and raw_model_call_limit.strip().lower() in {
            "",
            "none",
            "null",
            "unlimited",
            "unbounded",
        }:
            raw_model_call_limit = None
        if raw_model_call_limit is None:
            runtime["max_model_calls_per_case"] = None
        else:
            try:
                runtime["max_model_calls_per_case"] = int(raw_model_call_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ExperimentSpec runtime.max_model_calls_per_case must be positive or null"
                ) from exc
            if runtime["max_model_calls_per_case"] < 1:
                raise ValueError(
                    "ExperimentSpec runtime.max_model_calls_per_case must be positive or null"
                )
        observability = dict(spec.get("observability") or {})
        observability["collect_vllm_metrics"] = bool(
            observability.get("collect_vllm_metrics", True)
        )
        try:
            observability["vllm_metrics_interval_s"] = float(
                observability.get("vllm_metrics_interval_s", 5.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ExperimentSpec observability.vllm_metrics_interval_s must be positive"
            ) from exc
        if observability["vllm_metrics_interval_s"] <= 0:
            raise ValueError(
                "ExperimentSpec observability.vllm_metrics_interval_s must be positive"
            )
        return {
            "schema_version": 3,
            "id": spec_id,
            "benchmark": normalized_benchmark,
            "team_spec_id": team_spec_id,
            "runtime": runtime,
            "observability": observability,
            "network": dict(spec.get("network") or {}),
            "environment": normalized_environment,
            "notes": str(spec.get("notes") or ""),
        }

    def save_spec(self, spec_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        self._validate_id(spec_id)
        if spec.get("id") and str(spec["id"]) != spec_id:
            raise ValueError("ExperimentSpec id does not match URL")
        value = self.normalize_spec({**spec, "id": spec_id})
        value["updated_at_utc"] = _utc_now()
        path = self.spec_root / f"{spec_id}.json"
        _write_json(path, value)
        return {**value, "registry_path": str(path)}

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        raw = self._read(self.spec_root, spec_id)
        normalized = self.normalize_spec(raw)
        return {**normalized, "registry_path": raw["registry_path"]}

    def delete_spec(self, spec_id: str) -> dict[str, Any]:
        if any(item.get("experiment_spec_id") == spec_id for item in self.instances()):
            raise ValueError("ExperimentSpec is referenced by an ExperimentInstance")
        return self._delete(self.spec_root, spec_id)

    def create_instance(
        self,
        *,
        instance_id: str,
        spec_id: str,
        benchmark_instance_id: str,
        team_instance_id: str,
        run_dir: str | None = None,
        priority: int = 100,
        launcher: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_id(instance_id)
        if (self.instance_root / f"{instance_id}.json").exists():
            raise ValueError(f"ExperimentInstance {instance_id!r} already exists")
        self._validate_id(spec_id)
        spec = self.get_spec(spec_id)
        self._validate_id(benchmark_instance_id)
        self._validate_id(team_instance_id)
        normalized_launcher = normalize_launcher(launcher)
        value = bind_instance_to_spec(
            {
            "schema_version": 3,
            "id": instance_id,
            "benchmark_instance_id": benchmark_instance_id,
            "team_instance_id": team_instance_id,
            "launcher": normalized_launcher,
            "status": "ready",
            "queue": {"priority": int(priority)},
            "updated_at_utc": _utc_now(),
            "launch_id": None,
            "launch_dir": None,
            "run_dir": str(run_dir or "") or None,
            "error": None,
            },
            spec=spec,
            prefix="experiment",
            creation_source="studio",
        )
        return self._save_instance(value)

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        value = self._read(self.instance_root, instance_id)
        try:
            spec = self.get_spec(str(value.get("experiment_spec_id") or ""))
        except (FileNotFoundError, ValueError):
            spec = None
        return self._with_lifecycle_status(value, spec)

    def update_instance(self, instance_id: str, **changes: Any) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        value.pop("registry_path", None)
        if "status" in changes and changes["status"] not in _STATUSES:
            raise ValueError(f"invalid ExperimentInstance status {changes['status']!r}")
        value.update(changes, updated_at_utc=_utc_now())
        return self._save_instance(value)

    def next_queued(self) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.instances()
                if item.get("status") == "queued"
                and item.get("configuration_status") == "current"
            ),
            None,
        )

    def enqueue(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        self._require_current(value)
        if value.get("status") != "ready":
            raise ValueError("only a ready ExperimentInstance can be added to the queue")
        return self.update_instance(instance_id, status="queued", queued_at_utc=_utc_now())

    def dequeue(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        if value.get("status") != "queued":
            raise ValueError("only a queued ExperimentInstance can be removed from the queue")
        return self.update_instance(instance_id, status="ready", queued_at_utc=None)

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        if value.get("status") in {"queued", "running"}:
            raise ValueError("a queued or running ExperimentInstance cannot be deleted")
        return self._delete(self.instance_root, instance_id)

    def _save_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(value.get("id") or "")
        self._validate_id(instance_id)
        path = self.instance_root / f"{instance_id}.json"
        derived = {
            "registry_path",
            "experiment_spec",
            "configuration_status",
            "validation_error",
        }
        clean = {key: item for key, item in value.items() if key not in derived}
        _write_json(path, clean)
        return {**clean, "registry_path": str(path)}

    @staticmethod
    def _with_lifecycle_status(
        value: dict[str, Any], spec: dict[str, Any] | None
    ) -> dict[str, Any]:
        error = lifecycle_error(value, spec=spec, prefix="experiment")
        return {
            **value,
            "experiment_spec": spec,
            "configuration_status": "invalid" if error else "current",
            "validation_error": error,
        }

    @staticmethod
    def _require_current(value: dict[str, Any]) -> None:
        if value.get("configuration_status") == "invalid":
            raise ValueError(
                f"ExperimentInstance {value.get('id')!r} is invalid: "
                f"{value.get('validation_error')}"
            )

    def _read(self, root: Path, value_id: str) -> dict[str, Any]:
        self._validate_id(value_id)
        path = root / f"{value_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid registry document {path}")
        return {**value, "registry_path": str(path)}

    def _delete(self, root: Path, value_id: str) -> dict[str, Any]:
        value = self._read(root, value_id)
        path = Path(value["registry_path"])
        path.unlink()
        return {"id": value_id, "deleted": True, "path": str(path)}

    @staticmethod
    def _validate_id(value_id: str) -> None:
        if not _SAFE_ID.fullmatch(str(value_id or "")):
            raise ValueError("registry id may contain only letters, numbers, '.', '_' and '-'")


class ExperimentQueueManager:
    """Run queued ExperimentInstances against shared deployment capacity pools."""

    def __init__(self, registry: ExperimentRegistry, compiler, jobs, assemble_project) -> None:
        self.registry = registry
        self.compiler = compiler
        self.jobs = jobs
        self.assemble_project = assemble_project
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_after_current = False
        self._max_parallel_instances = _DEFAULT_MAX_PARALLEL_INSTANCES
        self._active_allocations: dict[str, dict[str, Any]] = {}
        self._pending_plans: dict[str, Any] = {}
        self._pending_allocations: dict[str, dict[str, Any]] = {}
        self._monitored_instance_ids: set[str] = set()
        self._wake_event = threading.Event()

    def start(
        self, *, max_parallel_instances: int = _DEFAULT_MAX_PARALLEL_INSTANCES
    ) -> dict[str, Any]:
        try:
            parallel_limit = int(max_parallel_instances)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_parallel_instances must be an integer") from exc
        if not 1 <= parallel_limit <= _MAX_PARALLEL_INSTANCES:
            raise ValueError(
                f"max_parallel_instances must be between 1 and {_MAX_PARALLEL_INSTANCES}"
            )
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            queued_ids = [
                item["id"]
                for item in self.registry.instances()
                if item.get("status") == "queued"
            ]
            if not queued_ids:
                raise ValueError("没有 queued ExperimentInstance，调度器无需启动")
            self._stop_after_current = False
            self._max_parallel_instances = parallel_limit
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="eval-studio-experiment-queue", daemon=True
            )
            self._thread.start()
        return {**self.status(), "accepted_instance_ids": queued_ids}

    def stop_after_current(self) -> dict[str, Any]:
        self._stop_after_current = True
        self._wake_event.set()
        return self.status()

    def enqueue(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        if instance.get("status") != "ready":
            raise ValueError("only a ready ExperimentInstance can be queued")
        plan = self._compile(instance)
        mode = normalize_launcher(instance.get("launcher"))["type"]
        self.jobs.validate(plan, mode=mode)
        allocation = self._allocation_for_plan(instance_id, plan, launch_origin="queue")
        with self._lock:
            queued = self.registry.enqueue(instance_id)
            self._pending_plans[instance_id] = plan
            self._pending_allocations[instance_id] = allocation
            self._wake_event.set()
            return queued

    def dequeue(self, instance_id: str) -> dict[str, Any]:
        with self._lock:
            if instance_id in self._active_allocations:
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
        allocation = self._allocation_for_plan(instance_id, plan, launch_origin="immediate")
        with self._lock:
            admitted, reason = self._can_admit_locked(allocation)
            if not admitted:
                raise ValueError(
                    f"ExperimentInstance cannot start now: {reason}; add it to the queue"
                )
            self._active_allocations[instance_id] = allocation
        try:
            plan, state = self._start_instance(
                instance, launch_origin="immediate", plan=plan
            )
        except Exception:
            self._release_allocation(instance_id)
            raise
        self._start_monitor(instance_id, plan.launch_dir)
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
        updated = self.registry.update_instance(
            instance_id,
            status="stopped",
            finished_at_utc=_utc_now(),
            job_status=state.get("status"),
            return_code=state.get("return_code"),
        )
        self._release_allocation(instance_id)
        return {"instance": updated, "job": state}

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self.registry.get_instance(instance_id)
        launch_dir = str(instance.get("launch_dir") or "").strip()
        if launch_dir and (Path(launch_dir) / "job.json").is_file():
            state = self.jobs.status(Path(launch_dir))
            if self.jobs.is_active_state(state):
                raise ValueError(
                    "an ExperimentInstance with an active launch cannot be deleted; "
                    "stop it first"
                )
        return self.registry.delete_instance(instance_id)

    def reconcile(self) -> dict[str, Any]:
        """Reattach persisted instances to live launches after a Studio restart."""

        recovered: list[str] = []
        finished: list[str] = []
        for instance in self.registry.instances():
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
            elif instance.get("status") == "running":
                self._finish(instance["id"], state)
                finished.append(instance["id"])
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
        for state in self.jobs.launches():
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
        self._refresh_active_demands()
        instances = self.registry.instances()
        running_instance_ids = [
            item["id"] for item in instances if item.get("status") == "running"
        ]
        orphaned_launch_ids = [item["launch_id"] for item in self.orphaned_launches(tail=0)]
        with self._lock:
            active_ids = sorted(self._active_allocations)
            resource_usage = self._resource_usage_locked()
            max_parallel_instances = self._max_parallel_instances
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "mode": "deployment_capacity_parallel",
            "stop_after_current": self._stop_after_current,
            "max_parallel_instances": max_parallel_instances,
            "active_instance_id": active_ids[0]
            if active_ids
            else (running_instance_ids[0] if running_instance_ids else None),
            "active_instance_ids": active_ids or running_instance_ids,
            "running_instance_ids": running_instance_ids,
            "queued_instance_ids": [
                item["id"] for item in instances if item.get("status") == "queued"
            ],
            "resource_usage": resource_usage,
            "orphaned_launch_ids": orphaned_launch_ids,
        }

    def instances(self, *, tail: int = 12) -> list[dict[str, Any]]:
        """Return persisted instances enriched with recoverable launch progress."""

        return [self._observe_instance(item, tail=tail) for item in self.registry.instances()]

    def instance_progress(self, instance_id: str, *, tail: int = 40) -> dict[str, Any]:
        return self._observe_instance(self.registry.get_instance(instance_id), tail=tail)

    def _observe_instance(self, instance: dict[str, Any], *, tail: int) -> dict[str, Any]:
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
            instance = self._finish(instance["id"], dict(progress["state"]))
        return {**instance, "progress": progress}

    def _run(self) -> None:
        try:
            self._recover_interrupted_instances()
            while True:
                if self._stop_after_current:
                    return
                self._refresh_active_demands()
                queued = [
                    item
                    for item in self.registry.instances()
                    if item.get("status") == "queued"
                    and item.get("configuration_status") == "current"
                ]
                launched = False
                for instance in queued:
                    if self._stop_after_current:
                        return
                    try:
                        with self._lock:
                            plan = self._pending_plans.get(instance["id"])
                            allocation = self._pending_allocations.get(instance["id"])
                        if plan is None or allocation is None:
                            plan = self._compile(instance)
                            allocation = self._allocation_for_plan(
                                instance["id"], plan, launch_origin="queue"
                            )
                            with self._lock:
                                self._pending_plans[instance["id"]] = plan
                                self._pending_allocations[instance["id"]] = allocation
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
                    with self._lock:
                        admitted, _reason = self._can_admit_locked(allocation)
                        if not admitted:
                            continue
                        self._active_allocations[instance["id"]] = allocation
                        self._pending_plans.pop(instance["id"], None)
                        self._pending_allocations.pop(instance["id"], None)
                    try:
                        plan, _state = self._start_instance(
                            instance, launch_origin="queue", plan=plan
                        )
                    except Exception:
                        self._release_allocation(instance["id"])
                        launched = True
                        continue
                    self._start_monitor(instance["id"], plan.launch_dir)
                    launched = True

                remaining_queued = any(
                    item.get("status") == "queued" for item in self.registry.instances()
                )
                with self._lock:
                    queued_active = any(
                        item.get("launch_origin") == "queue"
                        for item in self._active_allocations.values()
                    )
                if not remaining_queued and not queued_active:
                    return
                self._wake_event.wait(
                    0.05 if launched else _SCHEDULER_POLL_INTERVAL_S
                )
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

    def _start_instance(
        self, instance: dict[str, Any], *, launch_origin: str, plan=None
    ):
        instance_id = instance["id"]
        plan = plan or self._compile(instance)
        self.registry.update_instance(
            instance_id,
            status="running",
            started_at_utc=_utc_now(),
            finished_at_utc=None,
            launch_id=plan.launch_id,
            launch_dir=str(plan.launch_dir),
            run_dir=str(plan.run_dir),
            launch_origin=launch_origin,
            error=None,
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

    def _start_monitor(self, instance_id: str, launch_dir: Path) -> None:
        with self._lock:
            if instance_id in self._monitored_instance_ids:
                return
            self._monitored_instance_ids.add(instance_id)

        def monitor() -> None:
            try:
                self._monitor(instance_id, launch_dir)
            finally:
                with self._lock:
                    self._monitored_instance_ids.discard(instance_id)
                self._release_allocation(instance_id)

        threading.Thread(
            target=monitor,
            name=f"eval-studio-monitor-{instance_id}",
            daemon=True,
        ).start()

    def _monitor(self, instance_id: str, launch_dir: Path) -> None:
        while True:
            state = self.jobs.status(launch_dir)
            status = str(state.get("status") or "")
            if status not in {"starting", "running"}:
                self._finish(instance_id, state)
                return
            time.sleep(2.0)

    def _finish(self, instance_id: str, state: dict[str, Any]) -> dict[str, Any]:
        job_status = str(state.get("status") or "failed")
        if job_status == "completed":
            status = "completed"
        elif job_status == "stopped":
            status = "stopped"
        else:
            status = "failed"
        return self.registry.update_instance(
            instance_id,
            status=status,
            finished_at_utc=_utc_now(),
            job_status=job_status,
            return_code=state.get("return_code"),
        )

    def _track_running_instance(self, instance: dict[str, Any], launch_dir: Path) -> None:
        instance_id = str(instance["id"])
        with self._lock:
            if instance_id in self._active_allocations:
                return
        try:
            plan = self._compile(instance)
            allocation = self._allocation_for_plan(
                instance_id,
                plan,
                launch_origin=str(instance.get("launch_origin") or "recovered"),
            )
        except Exception:
            allocation = {
                "instance_id": instance_id,
                "launch_origin": str(instance.get("launch_origin") or "recovered"),
                "launch_dir": str(launch_dir),
                "run_dir": str(instance.get("run_dir") or ""),
                "max_case_slots": 1,
                "current_case_slots": 1,
                "deployment_capacities": {},
            }
        with self._lock:
            self._active_allocations.setdefault(instance_id, allocation)

    @staticmethod
    def _allocation_for_plan(
        instance_id: str, plan, *, launch_origin: str
    ) -> dict[str, Any]:
        project = dict(getattr(plan, "project", {}) or {})
        runtime = dict(project.get("runtime") or {})
        policy = dict(runtime.get("concurrency_policy") or {})
        case_limit = max(
            1,
            int(policy.get("maximum") or runtime.get("case_concurrency") or 1),
        )
        benchmark = dict(project.get("benchmark") or {})
        raw_cases = benchmark.get("n")
        samples = max(1, int(runtime.get("samples") or 1))
        if isinstance(raw_cases, int) or (
            isinstance(raw_cases, str) and raw_cases.isdigit()
        ):
            total_work = max(1, int(raw_cases) * samples)
            max_case_slots = min(case_limit, total_work)
        else:
            max_case_slots = case_limit
        deployment_capacities: dict[str, int] = {}
        for deployment in project.get("deployment_instances") or []:
            deployment_id = str(deployment.get("id") or "").strip()
            if not deployment_id:
                continue
            limits = dict(deployment.get("request_limits") or {})
            try:
                capacity = int(limits.get("max_concurrency") or 0)
            except (TypeError, ValueError):
                capacity = 0
            # An undeclared or unbounded client limit is not safe to share across
            # processes, so one experiment conservatively owns the deployment.
            deployment_capacities[deployment_id] = (
                capacity if capacity > 0 else max_case_slots
            )
        return {
            "instance_id": instance_id,
            "launch_origin": launch_origin,
            "launch_dir": str(getattr(plan, "launch_dir", "")),
            "run_dir": str(getattr(plan, "run_dir", "")),
            "max_case_slots": max_case_slots,
            "current_case_slots": max_case_slots,
            "deployment_capacities": deployment_capacities,
        }

    def _refresh_active_demands(self) -> None:
        with self._lock:
            allocations = [dict(item) for item in self._active_allocations.values()]
        changed = False
        for allocation in allocations:
            remaining = self._remaining_work(allocation)
            if remaining is None:
                continue
            current = min(int(allocation["max_case_slots"]), max(0, remaining))
            instance_id = str(allocation["instance_id"])
            with self._lock:
                stored = self._active_allocations.get(instance_id)
                if stored is not None and stored.get("current_case_slots") != current:
                    stored["current_case_slots"] = current
                    changed = True
        if changed:
            self._wake_event.set()

    @staticmethod
    def _remaining_work(allocation: dict[str, Any]) -> int | None:
        run_dir = str(allocation.get("run_dir") or "").strip()
        if not run_dir:
            return None
        path = Path(run_dir) / "run_status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            expected = int(status.get("expected_predictions") or 0)
            completed = sum(
                int(status.get(key) or 0)
                for key in (
                    "successful_predictions",
                    "error_predictions",
                    "skipped_predictions",
                )
            )
        except (OSError, TypeError, ValueError):
            return None
        return max(0, expected - completed) if expected > 0 else None

    def _can_admit_locked(self, allocation: dict[str, Any]) -> tuple[bool, str]:
        if len(self._active_allocations) >= self._max_parallel_instances:
            return False, "parallel ExperimentInstance limit reached"
        usage = self._resource_usage_locked()
        slots = int(allocation["current_case_slots"])
        for deployment_id, capacity in allocation["deployment_capacities"].items():
            requested = min(slots, int(capacity))
            used = int((usage.get(deployment_id) or {}).get("used") or 0)
            if used + requested > int(capacity):
                return (
                    False,
                    f"DeploymentInstance {deployment_id} capacity "
                    f"{used}/{capacity} cannot admit {requested} more slots",
                )
        return True, "admitted"

    def _resource_usage_locked(self) -> dict[str, dict[str, Any]]:
        usage: dict[str, dict[str, Any]] = {}
        for allocation in self._active_allocations.values():
            slots = int(allocation.get("current_case_slots") or 0)
            for deployment_id, raw_capacity in allocation.get(
                "deployment_capacities", {}
            ).items():
                capacity = int(raw_capacity)
                row = usage.setdefault(
                    deployment_id,
                    {
                        "used": 0,
                        "capacity": capacity,
                        "available": capacity,
                        "instance_ids": [],
                    },
                )
                row["capacity"] = min(int(row["capacity"]), capacity)
                row["used"] += min(slots, capacity)
                row["instance_ids"].append(allocation["instance_id"])
        for row in usage.values():
            row["available"] = max(0, int(row["capacity"]) - int(row["used"]))
        return usage

    def _release_allocation(self, instance_id: str) -> None:
        with self._lock:
            removed = self._active_allocations.pop(instance_id, None)
        if removed is not None:
            self._wake_event.set()
