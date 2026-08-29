"""Experiment assembly use cases shared by every Eval interface."""

from __future__ import annotations

import uuid
from typing import Any

from ...runtime.coordination.compiler import (
    compile_coordination,
    coordination_control_node_id,
)
from ..benchmarks.assets import resolve_scoring_profile
from ..experiments.compiler import ExecutionPlanCompiler
from ..experiments.registry import normalize_launcher
from ..scheduling.manager import ExperimentQueueManager
from .ports import (
    APIAccessRegistryPort,
    BenchmarkRegistryPort,
    DeploymentRegistryPort,
    ExperimentRepositoryPort,
    ModelRegistryPort,
    PricingRegistryPort,
    TeamInstanceRepositoryPort,
    TeamSpecRepositoryPort,
)


def _snapshot_document(value: Any) -> Any:
    """Remove repository metadata and secrets from a frozen run snapshot."""

    if isinstance(value, list):
        return [_snapshot_document(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if key == "registry_path":
            continue
        if lowered in {"api_key", "access_token", "secret", "authorization", "password"}:
            result[key] = "<redacted>"
        else:
            result[key] = _snapshot_document(item)
    return result


class ExperimentApplicationService:
    """Validate ExperimentSpecs and assemble immutable executable projects."""

    def __init__(
        self,
        *,
        experiments: ExperimentRepositoryPort,
        benchmarks: BenchmarkRegistryPort,
        teams: TeamSpecRepositoryPort,
        team_instances: TeamInstanceRepositoryPort,
        deployments: DeploymentRegistryPort,
        pricing: PricingRegistryPort,
        models: ModelRegistryPort,
        api_access: APIAccessRegistryPort,
    ) -> None:
        self.experiments = experiments
        self.benchmarks = benchmarks
        self.teams = teams
        self.team_instances = team_instances
        self.deployments = deployments
        self.pricing = pricing
        self.models = models
        self.api_access = api_access

    def validate_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        """Validate Spec-to-Spec references without resolving runtime Instances."""

        spec = self.experiments.normalize_spec(value)
        benchmark = spec["benchmark"]
        benchmark_spec = self.benchmarks.get_spec(benchmark["benchmark_spec_id"])
        if benchmark["runnable_task"] not in benchmark_spec["runnable_tasks"]:
            raise ValueError(
                f"task {benchmark['runnable_task']!r} is not registered by "
                f"Benchmark {benchmark_spec['id']!r}"
            )

        resolve_scoring_profile(
            benchmark_spec,
            benchmark["runnable_task"],
            benchmark.get("scoring_profile"),
        )
        self.teams.get(spec["team_spec_id"])
        return spec

    def assemble(
        self,
        experiment_spec: dict[str, Any],
        *,
        benchmark_instance_id: str,
        team_instance_id: str,
        experiment_instance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one Spec recipe plus concrete resource bindings for execution."""

        experiment_instance = experiment_instance or {}
        instance_run_dir = str(
            experiment_instance.get("run_dir")
            or (experiment_spec.get("environment") or {}).get("run_dir")
            or ""
        ).strip()
        spec = self.validate_spec(experiment_spec)
        if instance_run_dir:
            spec.setdefault("environment", {})["run_dir"] = instance_run_dir

        benchmark_spec, benchmark_instance, benchmark = self._resolve_benchmark(
            spec,
            benchmark_instance_id,
        )
        team_instance, team, bindings, referenced_deployments = self._resolve_team(
            spec,
            team_instance_id,
        )
        pricing_specs, pricing_instances = self._resolve_pricing(referenced_deployments)
        deployment_specs = self._resolve_deployment_specs(referenced_deployments)
        resource_specs, resource_instances = self._resolve_resources(referenced_deployments)
        launcher = normalize_launcher(experiment_instance.get("launcher"))
        experiment_instance_snapshot = {
            "schema_version": 3,
            "id": str(experiment_instance.get("id") or f"{spec['id']}-adhoc"),
            "experiment_spec_id": spec["id"],
            "benchmark_instance_id": benchmark_instance["id"],
            "team_instance_id": team_instance["id"],
            "launcher": launcher,
            "status": str(experiment_instance.get("status") or "planned"),
            "queue": dict(experiment_instance.get("queue") or {}),
            "execution": dict(experiment_instance.get("execution") or {}),
            "run_dir": instance_run_dir or experiment_instance.get("run_dir"),
        }
        snapshot = {
            "schema_version": 1,
            "experiment_spec": {
                **spec,
                "environment": {
                    key: item
                    for key, item in spec["environment"].items()
                    if key != "run_dir"
                },
            },
            "experiment_instance": experiment_instance_snapshot,
            "benchmark_spec": benchmark_spec,
            "benchmark_instance": benchmark_instance,
            "team_spec": team,
            "team_instance": team_instance,
            "deployment_specs": deployment_specs,
            "deployment_instances": referenced_deployments,
            "pricing_specs": pricing_specs,
            "pricing_instances": pricing_instances,
            "resource_specs": resource_specs,
            "resource_instances": resource_instances,
        }
        return {
            "schema_version": 4,
            "name": spec["id"],
            "environment": dict(spec["environment"]),
            "benchmark": benchmark,
            "team": team,
            "team_instance": {
                "id": team_instance["id"],
                "team_spec_id": team_instance["team_spec_id"],
                "runtime_framework": team_instance["runtime_framework"],
                "framework_options": dict(team_instance.get("framework_options") or {}),
            },
            "deployment_bindings": bindings,
            "deployment_instances": referenced_deployments,
            "runtime": {
                **dict(spec["runtime"]),
                "framework": team_instance["runtime_framework"],
                "framework_options": dict(team_instance.get("framework_options") or {}),
            },
            "observability": dict(spec["observability"]),
            "evaluation": dict(spec["evaluation"]),
            "network": dict(spec["network"]),
            "execution": {
                "mode": launcher["type"],
                **{key: item for key, item in launcher.items() if key != "type"},
                **dict(experiment_instance.get("execution") or {}),
            },
            "configuration_snapshot": _snapshot_document(snapshot),
        }

    def _resolve_benchmark(
        self,
        spec: dict[str, Any],
        benchmark_instance_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        config = dict(spec["benchmark"])
        benchmark_spec = self.benchmarks.get_spec(config["benchmark_spec_id"])
        benchmark_instance = self.benchmarks.get_instance(benchmark_instance_id)
        if benchmark_instance["benchmark_spec_id"] != benchmark_spec["id"]:
            raise ValueError(
                f"BenchmarkInstance {benchmark_instance['id']!r} is not an instance of "
                f"BenchmarkSpec {benchmark_spec['id']!r}"
            )
        if not benchmark_instance.get("available"):
            raise ValueError(
                f"BenchmarkInstance {benchmark_instance['id']!r} is unavailable"
            )
        task = config["runnable_task"]
        scoring = resolve_scoring_profile(
            benchmark_spec,
            task,
            config.get("scoring_profile"),
        )
        benchmark = {
            "benchmark_spec_id": benchmark_spec["id"],
            "benchmark_instance_id": benchmark_instance["id"],
            "prepare_target": benchmark_spec["prepare_target"],
            "task": task,
            "n": config["cases"],
            "start_index": config["start_index"],
            "case_selection": config.get("case_selection", "head"),
            "case_strata_field": config.get("case_strata_field"),
            "source": "registered_instance",
            "prepare_before_run": False,
            "prepared_path": benchmark_instance["prepared_path"],
            "capabilities": benchmark_spec.get("capabilities") or {},
            "scoring": scoring,
        }
        return benchmark_spec, benchmark_instance, benchmark

    def _resolve_team(
        self,
        spec: dict[str, Any],
        team_instance_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        deployment_instances = self.deployments.instances(probe=True)
        team_instance = self.team_instances.get(team_instance_id, deployment_instances)
        if team_instance["team_spec_id"] != spec["team_spec_id"]:
            raise ValueError(
                f"TeamInstance {team_instance_id!r} is not an instance of "
                f"TeamSpec {spec['team_spec_id']!r}"
            )
        if not team_instance.get("available"):
            missing = team_instance.get("missing_deployment_instance_ids") or []
            unavailable = team_instance.get("unavailable_deployment_instance_ids") or []
            details = [
                *(f"missing={item}" for item in missing),
                *(f"unavailable={item}" for item in unavailable),
            ]
            raise ValueError(
                f"TeamInstance {team_instance_id!r} is unavailable"
                + (f" ({', '.join(details)})" if details else "")
            )
        team = team_instance["team_spec"]
        model_bindings = [
            item
            for item in team_instance["resource_bindings"]
            if item["requirement"] == "model_inference"
            and item["resource_instance_type"] == "DeploymentInstance"
        ]
        if not model_bindings:
            raise ValueError(f"TeamInstance {team_instance_id!r} has no model binding")
        coordination = compile_coordination(team)
        node_bindings = [
            {
                "role_id": item["node_id"],
                "deployment_instance_id": item["resource_instance_id"],
                **(
                    {"generation_overrides": item["generation_overrides"]}
                    if item.get("generation_overrides")
                    else {}
                ),
            }
            for item in model_bindings
        ]
        first_binding = node_bindings[0]
        default_id = str(
            first_binding.get("deployment_instance_id")
            or first_binding.get("resource_instance_id")
        )
        control_node_id = coordination_control_node_id(coordination)
        control_binding = next(
            (item for item in model_bindings if item["node_id"] == control_node_id),
            None,
        )
        bindings = {
            "default_deployment_instance_id": default_id,
            "control_deployment_instance_id": (
                control_binding["resource_instance_id"]
                if control_binding is not None
                else default_id
            ),
            # The runtime contract still calls these role bindings, but every
            # model-backed Node is represented here, including Nodes that only
            # provide coordination operations.
            "role_bindings": node_bindings,
            **(
                {"control_generation_overrides": control_binding["generation_overrides"]}
                if control_binding and control_binding.get("generation_overrides")
                else {}
            ),
        }
        referenced_ids = {item["resource_instance_id"] for item in model_bindings}
        by_id = {item["id"]: item for item in deployment_instances}
        return (
            team_instance,
            team,
            bindings,
            [by_id[item] for item in sorted(referenced_ids)],
        )

    def _resolve_pricing(
        self,
        deployments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        instances: list[dict[str, Any]] = []
        specs: list[dict[str, Any]] = []
        seen_specs: set[str] = set()
        for deployment in deployments:
            for field in (
                "actual_pricing_instance_id",
                "api_equivalent_pricing_instance_id",
            ):
                pricing_id = str(deployment.get(field) or "")
                if not pricing_id or any(item["id"] == pricing_id for item in instances):
                    continue
                pricing = self.pricing.get_instance(pricing_id)
                instances.append(pricing)
                pricing_spec_id = str(pricing["pricing_spec_id"])
                if pricing_spec_id not in seen_specs:
                    seen_specs.add(pricing_spec_id)
                    specs.append(self.pricing.get_spec(pricing_spec_id))
        return specs, instances

    def _resolve_deployment_specs(
        self,
        deployments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {item["id"]: item for item in self.deployments.specs()}
        return [
            by_id[item["deployment_spec_id"]]
            for item in deployments
            if item.get("deployment_spec_id") in by_id
        ]

    def _resolve_resources(
        self,
        deployments: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        specs: dict[str, list[dict[str, Any]]] = {"models": [], "apis": []}
        instances: dict[str, list[dict[str, Any]]] = {"models": [], "apis": []}
        seen_instances: set[tuple[str, str]] = set()
        seen_specs: set[tuple[str, str]] = set()
        for deployment in deployments:
            model_spec_id = str(deployment.get("model_spec_id") or "")
            model_key = ("model", model_spec_id)
            if model_spec_id and model_key not in seen_specs:
                seen_specs.add(model_key)
                self._append_snapshot(
                    specs["models"],
                    model_spec_id,
                    lambda: self.models.get_spec(model_spec_id),
                )
            source = dict(deployment.get("source_instance") or {})
            source_type = str(source.get("type") or "")
            source_id = str(source.get("id") or "")
            key = (source_type, source_id)
            if not source_id or key in seen_instances:
                continue
            seen_instances.add(key)
            try:
                if source_type == "model":
                    instance = self.models.get_instance(source_id)
                    instances["models"].append(instance)
                    spec_id = str(instance["model_spec_id"])
                    spec_key = ("model", spec_id)
                    if spec_key not in seen_specs:
                        seen_specs.add(spec_key)
                        specs["models"].append(self.models.get_spec(spec_id))
                elif source_type == "api":
                    instance = self.api_access.get(source_id)
                    instances["apis"].append(instance)
                    spec_id = str(instance["api_spec_id"])
                    spec_key = ("api", spec_id)
                    if spec_key not in seen_specs:
                        seen_specs.add(spec_key)
                        specs["apis"].append(self.api_access.get_spec(spec_id))
            except (FileNotFoundError, ValueError) as exc:
                bucket = "models" if source_type == "model" else "apis"
                instances[bucket].append(
                    {
                        "id": source_id,
                        "source_type": source_type,
                        "snapshot_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return specs, instances

    @staticmethod
    def _append_snapshot(
        target: list[dict[str, Any]],
        item_id: str,
        loader,
    ) -> None:
        try:
            target.append(loader())
        except (FileNotFoundError, ValueError) as exc:
            target.append(
                {
                    "id": item_id,
                    "snapshot_error": f"{type(exc).__name__}: {exc}",
                }
            )


class ExperimentControlApplicationService:
    """Own Experiment CRUD, planning, queue commands, and progress queries."""

    def __init__(
        self,
        *,
        assembler: ExperimentApplicationService,
        experiments: ExperimentRepositoryPort,
        compiler: ExecutionPlanCompiler,
        queue: ExperimentQueueManager,
    ) -> None:
        self.assembler = assembler
        self.experiments = experiments
        self.compiler = compiler
        self.queue = queue

    def specs(self) -> list[dict[str, Any]]:
        return self.experiments.specs()

    def save_spec(self, spec_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self.assembler.validate_spec({**payload, "id": spec_id})
        return self.experiments.save_spec(spec_id, normalized)

    def delete_spec(self, spec_id: str) -> dict[str, Any]:
        return self.experiments.delete_spec(spec_id)

    def dashboard(self, *, include_specs: bool = False) -> dict[str, Any]:
        return {
            **({"specs": self.specs()} if include_specs else {}),
            "instances": self.queue.instances(tail=0, compact=True),
            "queue": self.queue.status(),
            "orphans": self.queue.orphaned_launches(tail=0),
        }

    def instances(self, *, tail: int = 0, compact: bool = True) -> list[dict[str, Any]]:
        return self.queue.instances(tail=max(0, min(int(tail), 200)), compact=compact)

    def create_instance(self, payload: dict[str, Any]) -> dict[str, Any]:
        spec_id = str(payload.get("experiment_spec_id") or "").strip()
        spec = self.assembler.validate_spec(self.experiments.get_spec(spec_id))
        benchmark_instance_id = str(payload.get("benchmark_instance_id") or "").strip()
        team_instance_id = str(payload.get("team_instance_id") or "").strip()
        instance_id = str(payload.get("id") or f"{spec_id}-{uuid.uuid4().hex[:8]}")
        requested_run_dir = str(payload.get("run_dir") or "").strip()
        launcher = normalize_launcher(payload.get("launcher"))
        execution = {
            "next_segment_case_limit": payload.get("next_segment_case_limit"),
            "case_completion_target": payload.get("case_completion_target"),
        }
        project = self.assembler.assemble(
            spec,
            benchmark_instance_id=benchmark_instance_id,
            team_instance_id=team_instance_id,
            experiment_instance={
                "id": instance_id,
                "launcher": launcher,
                "status": "ready",
                "queue": {"priority": int(payload.get("priority", 100))},
                "execution": execution,
                "run_dir": requested_run_dir or None,
            },
        )
        plan = self.compiler.compile(project, launch_id=instance_id)
        return self.experiments.create_instance(
            instance_id=instance_id,
            spec_id=spec_id,
            benchmark_instance_id=benchmark_instance_id,
            team_instance_id=team_instance_id,
            run_dir=str(plan.run_dir),
            priority=int(payload.get("priority", 100)),
            launcher=launcher,
            execution=execution,
        )

    def update_instance(self, instance_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "priority",
            "next_segment_case_limit",
            "case_completion_target",
        }
        return self.experiments.update_controls(
            instance_id,
            **{key: payload[key] for key in allowed if key in payload},
        )

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        return self.queue.delete_instance(instance_id)

    def plan(self, instance_id: str) -> dict[str, Any]:
        return self.queue.plan_instance(instance_id).as_dict()

    def launch(self, instance_id: str) -> dict[str, Any]:
        return self.queue.launch_now(instance_id)

    def enqueue(self, instance_id: str) -> dict[str, Any]:
        return self.queue.enqueue(instance_id)

    def dequeue(self, instance_id: str) -> dict[str, Any]:
        return self.queue.dequeue(instance_id)

    def stop(self, instance_id: str) -> dict[str, Any]:
        return self.queue.stop_instance(instance_id)

    def resume(self, instance_id: str) -> dict[str, Any]:
        return self.queue.resume_instance(instance_id)

    def drain(self, instance_id: str) -> dict[str, Any]:
        return self.queue.drain_instance(instance_id)

    def progress(self, instance_id: str, *, tail: int) -> dict[str, Any]:
        return self.queue.instance_progress(instance_id, tail=tail)

    def orphans(self) -> list[dict[str, Any]]:
        return self.queue.orphaned_launches()

    def stop_orphan(self, launch_id: str) -> dict[str, Any]:
        return self.queue.stop_orphaned_launch(launch_id)

    def recover_orphan(self, launch_id: str) -> dict[str, Any]:
        return self.queue.recover_orphaned_launch(launch_id)

    def queue_status(self) -> dict[str, Any]:
        return self.queue.status()

    def start_queue(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        value = payload or {}
        return self.queue.start(
            max_parallel_instances=int(value.get("max_parallel_instances", 16)),
            max_running_trials=int(value.get("max_running_trials", 64)),
        )

    def stop_queue_after_current(self) -> dict[str, Any]:
        return self.queue.stop_after_current()
