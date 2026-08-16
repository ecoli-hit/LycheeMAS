"""FastAPI application factory for LycheeMAS Eval Studio."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .apis import APIRegistry
from .benchmarks import BenchmarkRegistry, resolve_scoring_profile
from .catalog import StudioCatalog, repository_root
from .deployments import DeploymentRegistry
from .environment import discover_environments, inspect_environment, installation_profiles
from .events import read_jsonl_events
from .execution import ExecutionPlanCompiler
from .experiments import ExperimentQueueManager, ExperimentRegistry, normalize_launcher
from .jobs import JobManager
from .models import ModelRegistry
from .pricing import PricingRegistry
from .team_instances import TeamInstanceRegistry, team_instance_availability
from .teams import TeamSpecRegistry


def _snapshot_document(value: Any) -> Any:
    """Remove registry-local metadata and accidental secret values from run snapshots."""

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


def create_app(repo_root: str | os.PathLike | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    root = Path(repo_root or repository_root()).resolve()
    catalog = StudioCatalog(root)
    compiler = ExecutionPlanCompiler(root)
    jobs = JobManager(compiler)
    deployment_registry = DeploymentRegistry(root)
    pricing_registry = PricingRegistry(root)
    team_registry = TeamSpecRegistry(root)
    team_instance_registry = TeamInstanceRegistry(root)
    benchmark_registry = BenchmarkRegistry(root)
    model_registry = ModelRegistry(root)
    api_registry = APIRegistry(root)
    experiment_registry = ExperimentRegistry(root)
    launches_root = root / "runs/eval_studio/launches"
    launches_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="LycheeMAS Eval Studio", version="0.1")

    def assemble_project(
        experiment_spec: dict[str, Any],
        *,
        benchmark_instance_id: str,
        team_instance_id: str,
        experiment_instance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve a Spec-only recipe and concrete Instance bindings for execution."""

        instance_run_dir = str(
            (experiment_instance or {}).get("run_dir")
            or (experiment_spec.get("environment") or {}).get("run_dir")
            or ""
        ).strip()
        spec = experiment_registry.normalize_spec(experiment_spec)
        if instance_run_dir:
            spec.setdefault("environment", {})["run_dir"] = instance_run_dir
        benchmark_config = dict(spec["benchmark"])
        benchmark_spec = benchmark_registry.get_spec(benchmark_config["benchmark_spec_id"])
        benchmark_instance = benchmark_registry.get_instance(benchmark_instance_id)
        if benchmark_instance["benchmark_spec_id"] != benchmark_spec["id"]:
            raise ValueError(
                f"BenchmarkInstance {benchmark_instance_id!r} is not an instance of "
                f"BenchmarkSpec {benchmark_spec['id']!r}"
            )
        if not benchmark_instance.get("available"):
            raise ValueError(f"BenchmarkInstance {benchmark_instance_id!r} is unavailable")
        task = benchmark_config["runnable_task"]
        scoring = resolve_scoring_profile(
            benchmark_spec,
            task,
            benchmark_config.get("scoring_profile"),
        )

        deployment_instances = deployment_registry.instances(probe=True)
        team_instance = team_instance_registry.get(team_instance_id, deployment_instances)
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
        slots = {item["id"]: item for item in team.get("inference_slots") or []}
        inference_bindings = team_instance["inference_bindings"]
        participant_bindings = [
            {
                "role_id": slots[item["slot_id"]]["participant_id"],
                "deployment_instance_id": item["deployment_instance_id"],
                **(
                    {"generation_overrides": item["generation_overrides"]}
                    if item.get("generation_overrides")
                    else {}
                ),
            }
            for item in inference_bindings
            if slots[item["slot_id"]]["kind"] == "participant"
        ]
        controller_binding = next(
            (item for item in inference_bindings if slots[item["slot_id"]]["kind"] == "controller"),
            None,
        )
        first_binding = participant_bindings[0] if participant_bindings else inference_bindings[0]
        default_id = first_binding["deployment_instance_id"]
        control_id = (
            controller_binding["deployment_instance_id"]
            if controller_binding is not None
            else default_id
        )
        bindings = {
            "default_deployment_instance_id": default_id,
            "control_deployment_instance_id": control_id,
            "role_bindings": participant_bindings,
            **(
                {"control_generation_overrides": controller_binding["generation_overrides"]}
                if controller_binding and controller_binding.get("generation_overrides")
                else {}
            ),
        }
        referenced_ids = {item["deployment_instance_id"] for item in inference_bindings}
        by_id = {item["id"]: item for item in deployment_instances}
        referenced_deployments = [by_id[item] for item in sorted(referenced_ids)]
        referenced_pricing_instances = []
        referenced_pricing_specs = []
        seen_pricing_specs: set[str] = set()
        for deployment in referenced_deployments:
            for field in (
                "actual_pricing_instance_id",
                "api_equivalent_pricing_instance_id",
            ):
                pricing_id = str(deployment.get(field) or "")
                if not pricing_id or any(
                    item["id"] == pricing_id for item in referenced_pricing_instances
                ):
                    continue
                pricing = pricing_registry.get_instance(pricing_id)
                referenced_pricing_instances.append(pricing)
                pricing_spec_id = str(pricing["pricing_spec_id"])
                if pricing_spec_id not in seen_pricing_specs:
                    seen_pricing_specs.add(pricing_spec_id)
                    referenced_pricing_specs.append(pricing_registry.get_spec(pricing_spec_id))
        deployment_specs_by_id = {item["id"]: item for item in deployment_registry.specs()}
        referenced_deployment_specs = [
            deployment_specs_by_id[item["deployment_spec_id"]]
            for item in referenced_deployments
            if item.get("deployment_spec_id") in deployment_specs_by_id
        ]
        resource_specs: dict[str, list[dict[str, Any]]] = {"models": [], "apis": []}
        resource_instances: dict[str, list[dict[str, Any]]] = {"models": [], "apis": []}
        seen_resources: set[tuple[str, str]] = set()
        seen_resource_specs: set[tuple[str, str]] = set()
        for deployment in referenced_deployments:
            source = dict(deployment.get("source_instance") or {})
            source_type = str(source.get("type") or "")
            source_id = str(source.get("id") or "")
            key = (source_type, source_id)
            if not source_id or key in seen_resources:
                continue
            seen_resources.add(key)
            try:
                if source_type == "model":
                    instance = model_registry.get_instance(source_id)
                    resource_instances["models"].append(instance)
                    spec_key = ("model", str(instance["model_spec_id"]))
                    if spec_key not in seen_resource_specs:
                        seen_resource_specs.add(spec_key)
                        resource_specs["models"].append(
                            model_registry.get_spec(instance["model_spec_id"])
                        )
                elif source_type == "api":
                    instance = api_registry.get(source_id)
                    resource_instances["apis"].append(instance)
                    spec_key = ("api", str(instance["api_spec_id"]))
                    if spec_key not in seen_resource_specs:
                        seen_resource_specs.add(spec_key)
                        resource_specs["apis"].append(
                            api_registry.get_spec(instance["api_spec_id"])
                        )
            except (FileNotFoundError, ValueError) as exc:
                bucket = "models" if source_type == "model" else "apis"
                resource_instances[bucket].append(
                    {
                        "id": source_id,
                        "source_type": source_type,
                        "snapshot_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        benchmark = {
            "benchmark_spec_id": benchmark_spec["id"],
            "benchmark_instance_id": benchmark_instance["id"],
            "prepare_target": benchmark_spec["prepare_target"],
            "task": task,
            "n": benchmark_config["cases"],
            "start_index": benchmark_config["start_index"],
            "source": "registered_instance",
            "prepare_before_run": False,
            "prepared_path": benchmark_instance["prepared_path"],
            "capabilities": benchmark_spec.get("capabilities") or {},
            "scoring": scoring,
        }
        launcher = normalize_launcher((experiment_instance or {}).get("launcher"))
        experiment_instance_snapshot = {
            "schema_version": 3,
            "id": str((experiment_instance or {}).get("id") or f"{spec['id']}-adhoc"),
            "experiment_spec_id": spec["id"],
            "benchmark_instance_id": benchmark_instance["id"],
            "team_instance_id": team_instance["id"],
            "launcher": launcher,
            "status": str((experiment_instance or {}).get("status") or "planned"),
            "queue": dict((experiment_instance or {}).get("queue") or {}),
            "run_dir": instance_run_dir or (experiment_instance or {}).get("run_dir"),
        }
        snapshot = {
            "schema_version": 1,
            "experiment_spec": {
                **spec,
                "environment": {
                    key: item for key, item in spec["environment"].items() if key != "run_dir"
                },
            },
            "experiment_instance": experiment_instance_snapshot,
            "benchmark_spec": benchmark_spec,
            "benchmark_instance": benchmark_instance,
            "team_spec": team,
            "team_instance": team_instance,
            "deployment_specs": referenced_deployment_specs,
            "deployment_instances": referenced_deployments,
            "pricing_specs": referenced_pricing_specs,
            "pricing_instances": referenced_pricing_instances,
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
            },
            "deployment_bindings": bindings,
            "deployment_instances": referenced_deployments,
            "runtime": dict(spec["runtime"]),
            "network": dict(spec["network"]),
            "execution": {
                "mode": launcher["type"],
                **{key: item for key, item in launcher.items() if key != "type"},
            },
            "configuration_snapshot": _snapshot_document(snapshot),
        }

    experiment_queue = ExperimentQueueManager(experiment_registry, compiler, jobs, assemble_project)
    startup_reconciliation = experiment_queue.reconcile()

    def registered_runs_roots() -> list[str]:
        """Return default plus custom roots declared by saved ExperimentSpecs."""

        values = [str(catalog.runs_root)]
        for spec in experiment_registry.specs():
            environment = spec.get("environment") or {}
            for key in ("runs_root",):
                value = str(environment.get(key) or "").strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = root / path
                resolved = str(path.resolve())
                if resolved not in values:
                    values.append(resolved)
        for instance in experiment_registry.instances():
            value = str(instance.get("run_dir") or "").strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            resolved = str(path.resolve())
            if resolved not in values:
                values.append(resolved)
        return values

    def registered_deployment_spec(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate that a DeploymentSpec references a registered resource Spec."""

        value = dict(payload)
        value.pop("api_key", None)
        source_spec = dict(value.get("source_spec") or {})
        if source_spec.get("type") == "model":
            model_registry.get_spec(str(source_spec.get("id") or ""))
        elif source_spec.get("type") == "api":
            api_registry.get_spec(str(source_spec.get("id") or ""))
        else:
            raise ValueError("DeploymentSpec source_spec.type must be model or api")
        return value

    def deployment_source(
        spec: dict[str, Any], source_instance_id: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Resolve one resource Instance compatible with a DeploymentSpec source Spec."""

        source_spec = spec["source_spec"]
        if source_spec["type"] == "model":
            model = model_registry.get_instance(source_instance_id)
            if model["model_spec_id"] != source_spec["id"]:
                raise ValueError(
                    f"ModelInstance {source_instance_id!r} is not an instance of "
                    f"ModelSpec {source_spec['id']!r}"
                )
            if not model.get("available"):
                raise ValueError(f"ModelInstance {source_instance_id!r} is not ready")
            model_spec = model["model_spec"]
            capabilities = {item: True for item in model_spec.get("capabilities") or []}
            capabilities.update(dict(spec.get("capabilities") or {}))
            fields = {
                "model_id": model_spec["name"],
                "model_path": model["path"],
                "capabilities": capabilities,
            }
            if spec["kind"] == "vllm":
                fields.update(auth_mode="none", trust_env=False)
            return {"type": "model", "id": source_instance_id}, fields

        api_instance = api_registry.get(source_instance_id)
        if api_instance["api_spec_id"] != source_spec["id"]:
            raise ValueError(
                f"APIInstance {source_instance_id!r} is not an instance of "
                f"APISpec {source_spec['id']!r}"
            )
        fields = api_registry.deployment_fields(source_instance_id)
        if fields["kind"] != spec["kind"]:
            raise ValueError(
                f"APIInstance {source_instance_id!r} provides {fields['kind']!r}, "
                f"not {spec['kind']!r}"
            )
        capabilities = dict(fields.get("capabilities") or {})
        capabilities.update(dict(spec.get("capabilities") or {}))
        fields["capabilities"] = capabilities
        return {"type": "api", "id": source_instance_id}, fields

    def validated_experiment_spec(value: dict[str, Any]) -> dict[str, Any]:
        """Validate all Spec-to-Spec references without resolving any Instance."""

        spec = experiment_registry.normalize_spec(value)
        benchmark = spec["benchmark"]
        benchmark_spec = benchmark_registry.get_spec(benchmark["benchmark_spec_id"])
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
        team_registry.get(spec["team_spec_id"])
        return spec

    @app.get("/api/health")
    def health():
        return {"status": "ok", "repo_root": str(root)}

    @app.get("/api/bootstrap")
    def bootstrap():
        deployment_instances = deployment_registry.instances()
        benchmark_rows = catalog.benchmarks()
        benchmark_instances = benchmark_registry.instances()
        return {
            "paths": catalog.paths(),
            "teams": {"specs": team_registry.all()},
            "team_instances": {"instances": team_instance_registry.all(deployment_instances)},
            "benchmarks": benchmark_rows,
            "benchmark_registry": {
                "specs": benchmark_registry.specs(),
                "instances": benchmark_instances,
            },
            "model_registry": {
                "specs": model_registry.specs(),
                "instances": model_registry.instances(),
            },
            "api_registry": {
                "specs": api_registry.specs(),
                "instances": api_registry.instances(),
            },
            "deployments": {
                "specs": deployment_registry.specs(),
                "instances": deployment_instances,
            },
            "pricing_registry": {
                "specs": pricing_registry.specs(),
                "instances": pricing_registry.instances(),
            },
            "runs": catalog.runs(limit=40, roots=registered_runs_roots()),
            "tmux_sessions": jobs.tmux_sessions(),
            "environments": discover_environments(root),
            "installation_profiles": installation_profiles(),
            "experiments": {
                "specs": experiment_registry.specs(),
                "instances": experiment_queue.instances(),
                "queue": experiment_queue.status(),
                "orphans": experiment_queue.orphaned_launches(),
                "startup_reconciliation": startup_reconciliation,
            },
            "default_experiment_spec": experiment_registry.default_spec(),
        }

    @app.get("/api/environments")
    def environments():
        return discover_environments(root)

    @app.post("/api/environments/check")
    def check_environment(payload: dict[str, Any]):
        try:
            return inspect_environment(
                root,
                payload.get("environment") or payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/deployments")
    def deployments(probe: bool = False):
        return {
            "specs": deployment_registry.specs(),
            "instances": deployment_registry.instances(probe=probe),
        }

    @app.get("/api/pricing")
    def pricing():
        return {
            "specs": pricing_registry.specs(),
            "instances": pricing_registry.instances(),
        }

    @app.put("/api/pricing-specs/{spec_id}")
    def save_pricing_spec(spec_id: str, payload: dict[str, Any]):
        try:
            return pricing_registry.save_spec(spec_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/pricing-specs/{spec_id}")
    def delete_pricing_spec(spec_id: str):
        try:
            if any(
                spec_id
                in {
                    item.get("actual_pricing_spec_id"),
                    item.get("api_equivalent_pricing_spec_id"),
                }
                for item in deployment_registry.specs()
            ):
                raise ValueError("PricingSpec is referenced by a DeploymentSpec")
            return pricing_registry.delete_spec(spec_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/pricing-specs/{spec_id}/instances")
    def instantiate_pricing_instance(spec_id: str, payload: dict[str, Any]):
        try:
            instance_id = str(payload.get("id") or "")
            if not instance_id:
                raise ValueError("PricingInstance id is required")
            return pricing_registry.instantiate_instance(
                spec_id,
                instance_id,
                payload,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/pricing-instances/{instance_id}")
    def delete_pricing_instance(instance_id: str):
        try:
            if any(
                instance_id
                in {
                    item.get("actual_pricing_instance_id"),
                    item.get("api_equivalent_pricing_instance_id"),
                }
                for item in deployment_registry.instances()
            ):
                raise ValueError("PricingInstance is referenced by a DeploymentInstance")
            return pricing_registry.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/deployments/{deployment_id}")
    def save_deployment(deployment_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != deployment_id:
            raise HTTPException(status_code=422, detail="deployment id does not match URL")
        try:
            return deployment_registry.save(
                registered_deployment_spec({**payload, "id": deployment_id})
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/deployments/{deployment_id}")
    def delete_deployment(deployment_id: str):
        try:
            if any(
                item.get("deployment_spec_id") == deployment_id
                for item in deployment_registry.instances()
            ):
                raise ValueError("DeploymentSpec is referenced by a DeploymentInstance")
            return deployment_registry.delete(deployment_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/deployments/{deployment_id}/deploy")
    def deploy_deployment(deployment_id: str, payload: dict[str, Any]):
        try:
            spec = next(
                (item for item in deployment_registry.specs() if item["id"] == deployment_id),
                None,
            )
            if spec is None:
                raise ValueError(f"unknown DeploymentSpec {deployment_id!r}")
            source_instance_id = str(payload.get("source_instance_id") or "").strip()
            if not source_instance_id:
                raise ValueError("DeploymentInstance requires source_instance_id")
            source_instance, source_fields = deployment_source(spec, source_instance_id)
            instance_id = str(payload.get("instance_id") or f"instance-{deployment_id}")
            actual_pricing_instance_id = str(
                payload.get("actual_pricing_instance_id") or ""
            ).strip()
            if not actual_pricing_instance_id:
                raise ValueError("DeploymentInstance requires actual_pricing_instance_id")
            equivalent_pricing_instance_id = str(
                payload.get("api_equivalent_pricing_instance_id") or ""
            ).strip()
            return deployment_registry.deploy(
                spec,
                instance_id=instance_id,
                source_instance=source_instance,
                source_fields=source_fields,
                actual_pricing_instance_id=actual_pricing_instance_id,
                api_equivalent_pricing_instance_id=(equivalent_pricing_instance_id or None),
                api_key=str(payload.get("api_key") or ""),
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/deployment-instances/{instance_id}")
    def delete_deployment_instance(instance_id: str):
        try:
            if any(
                binding.get("deployment_instance_id") == instance_id
                for team in team_instance_registry.all()
                for binding in team.get("inference_bindings") or []
            ):
                raise ValueError("DeploymentInstance is referenced by a TeamInstance")
            return deployment_registry.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/resources/catalog")
    def resource_catalog(
        raw_root: str | None = None,
        prepared_root: str | None = None,
        models_root: str | None = None,
    ):
        try:
            dynamic = StudioCatalog(
                root,
                raw_root=Path(raw_root or catalog.raw_root).expanduser().resolve(),
                prepared_root=Path(prepared_root or catalog.prepared_root).expanduser().resolve(),
                models_root=Path(models_root or catalog.models_root).expanduser().resolve(),
                runs_root=catalog.runs_root,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "paths": dynamic.paths(),
            "benchmarks": dynamic.benchmarks(),
            "benchmark_specs": benchmark_registry.specs(),
            "benchmark_instances": benchmark_registry.instances(),
            "model_specs": model_registry.specs(),
            "model_instances": model_registry.instances(),
            "api_registry": {
                "specs": api_registry.specs(),
                "instances": api_registry.instances(),
            },
        }

    @app.get("/api/api-specs")
    def api_specs():
        return api_registry.specs()

    @app.get("/api/api-instances")
    def api_instances():
        return api_registry.instances()

    @app.post("/api/api-instances/{instance_id}/instantiate")
    def instantiate_api_instance(instance_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != instance_id:
            raise HTTPException(status_code=422, detail="APIInstance id does not match URL")
        if any(item["id"] == instance_id for item in api_registry.instances()):
            raise HTTPException(
                status_code=422, detail=f"APIInstance {instance_id!r} already exists"
            )
        try:
            return api_registry.save_instance({**payload, "id": instance_id})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/api-instances/{instance_id}")
    def delete_api_instance(instance_id: str):
        if any(
            (item.get("source_instance") or {}).get("type") == "api"
            and (item.get("source_instance") or {}).get("id") == instance_id
            for item in deployment_registry.instances()
        ):
            raise HTTPException(
                status_code=422,
                detail="APIInstance is referenced by a DeploymentInstance",
            )
        try:
            return api_registry.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/model-specs")
    def model_specs():
        return model_registry.specs()

    @app.get("/api/model-instances")
    def model_instances():
        return model_registry.instances()

    @app.post("/api/model-instances/scan")
    def scan_model_instances(payload: dict[str, Any]):
        try:
            return model_registry.scan_root(payload.get("models_root") or catalog.models_root)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/model-instances/{instance_id}")
    def delete_model_instance(instance_id: str):
        if any(
            (item.get("source_instance") or {}).get("type") == "model"
            and (item.get("source_instance") or {}).get("id") == instance_id
            for item in deployment_registry.instances()
        ):
            raise HTTPException(
                status_code=422,
                detail="ModelInstance is referenced by a DeploymentInstance",
            )
        try:
            return model_registry.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/resource-registries")
    def resource_registries():
        return {
            "models": {"specs": model_registry.specs(), "instances": model_registry.instances()},
            "apis": {
                "specs": api_registry.specs(),
                "instances": api_registry.instances(),
            },
        }

    @app.get("/api/benchmarks")
    def benchmarks():
        return catalog.benchmarks()

    @app.get("/api/benchmark-specs")
    def benchmark_specs():
        return benchmark_registry.specs()

    @app.get("/api/benchmark-instances")
    def benchmark_instances():
        return benchmark_registry.instances()

    @app.post("/api/benchmark-instances/scan")
    def scan_benchmark_instances(payload: dict[str, Any]):
        try:
            return benchmark_registry.scan_roots(
                raw_root=payload.get("raw_root") or catalog.raw_root,
                prepared_root=payload.get("prepared_root") or catalog.prepared_root,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/benchmark-instances/{instance_id}/instantiate")
    def instantiate_benchmark_instance(instance_id: str, payload: dict[str, Any]):
        from ..benchmarks import BENCHMARKS
        from ..benchmarks.manifest import benchmark_key_for_target

        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", instance_id):
                raise ValueError("invalid BenchmarkInstance id")
            if any(item["id"] == instance_id for item in benchmark_registry.instances()):
                raise ValueError(f"BenchmarkInstance {instance_id!r} already exists")
            spec = benchmark_registry.get_spec(str(payload.get("benchmark_spec_id") or ""))
            mode = str(payload.get("mode") or "download")
            if mode not in {"local", "download"}:
                raise ValueError("benchmark acquisition mode must be local or download")
            raw_root = _allowed_path(payload.get("raw_root") or catalog.raw_root, root)
            prepared_root = _allowed_path(
                payload.get("prepared_root") or catalog.prepared_root, root
            )
            benchmark_key = benchmark_key_for_target(spec["prepare_target"])
            managed_raw_path = raw_root / benchmark_key
            managed_prepared_path = prepared_root / benchmark_key
            job = None
            provenance: list[dict[str, Any]] = []
            if mode == "local":
                source_value = str(payload.get("source_path") or "").strip()
                if not source_value:
                    raise ValueError("local benchmark acquisition requires source_path")
                source_path = Path(source_value).expanduser().resolve()
                stage = str(payload.get("stage") or "prepared")
                if stage not in {"raw", "prepared"}:
                    raise ValueError("local benchmark stage must be raw or prepared")
                provenance = [
                    {
                        "selection": "local_path",
                        "provider": "local",
                        "source_id": instance_id,
                        "stage": stage,
                        "path": str(source_path),
                    }
                ]
                raw_path = source_path if stage == "raw" else Path()
                prepared_path = source_path if stage == "prepared" else managed_prepared_path
                prepared_managed = stage == "raw"
                status = "ready" if stage == "prepared" else "preparing"
                if stage == "raw":
                    job = launch_dataset_prepare(
                        {
                            **payload,
                            "source": "auto",
                            "conversion_only": True,
                            "raw_overrides": [
                                {
                                    "benchmark_key": benchmark_key,
                                    "provider": "local",
                                    "source_id": instance_id,
                                    "path": str(source_path),
                                }
                            ],
                        },
                        spec["prepare_target"],
                    )
            else:
                source = str(payload.get("source") or "auto")
                benchmark = BENCHMARKS.for_prepare_target(spec["prepare_target"])
                selected_providers = benchmark.source_backend_order(source)
                registered_candidates = [
                    {"provider": provider, "source_id": source_id}
                    for provider in selected_providers
                    for source_id in benchmark.provider_ids(provider)
                ]
                raw_path = managed_raw_path
                prepared_path = managed_prepared_path
                prepared_managed = True
                status = "preparing"
                job = launch_dataset_prepare(
                    {**payload, "source": source},
                    spec["prepare_target"],
                )
                provenance = [
                    {
                        "selection": "requested",
                        "provider": source,
                        "registered_candidates": registered_candidates,
                    }
                ]

            manifest_root = prepared_path if prepared_path.is_dir() else prepared_path.parent
            saved = benchmark_registry.save_instance(
                {
                    "id": instance_id,
                    "benchmark_spec_id": spec["id"],
                    "acquisition": {
                        "mode": mode,
                        "stage": str(payload.get("stage") or "prepared"),
                        "source": str(payload.get("source") or "local"),
                        "source_path": str(payload.get("source_path") or ""),
                    },
                    "raw_path": str(raw_path) if str(raw_path) != "." else "",
                    "prepared_path": str(prepared_path),
                    "manifest_path": str(manifest_root / "manifest.json"),
                    "supported_tasks": spec["runnable_tasks"],
                    "status": status,
                    "provenance": provenance,
                    "managed": prepared_managed,
                    "job_id": str((job or {}).get("launch_id") or ""),
                }
            )
            return {"instance": saved, "job": job}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/benchmark-instances/{instance_id}/check")
    def check_benchmark_instance(instance_id: str):
        try:
            return benchmark_registry.check_instance(instance_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/benchmark-instances/{instance_id}")
    def delete_benchmark_instance(instance_id: str):
        if any(
            item.get("benchmark_instance_id") == instance_id
            for item in experiment_registry.instances()
        ):
            raise HTTPException(
                status_code=422,
                detail="BenchmarkInstance is referenced by an ExperimentInstance",
            )
        try:
            return benchmark_registry.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/teams")
    def teams():
        return {"specs": team_registry.all()}

    @app.get("/api/team-specs")
    def team_specs():
        return {"specs": team_registry.all()}

    @app.put("/api/team-specs/{team_id}")
    def save_team_spec(team_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != team_id:
            raise HTTPException(status_code=422, detail="team id does not match URL")
        try:
            return team_registry.save({**payload, "id": team_id})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/team-specs/{team_id}")
    def delete_team_spec(team_id: str):
        try:
            if any(
                item.get("team_spec_id") == team_id for item in team_instance_registry.all()
            ) or any(item.get("team_spec_id") == team_id for item in experiment_registry.specs()):
                raise ValueError("TeamSpec is referenced by a TeamInstance or ExperimentSpec")
            return team_registry.delete(team_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/team-instances")
    def team_instances(probe: bool = False):
        deployment_instances = deployment_registry.instances(probe=probe)
        return {"instances": team_instance_registry.all(deployment_instances)}

    @app.put("/api/team-instances/{instance_id}")
    def save_team_instance(instance_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != instance_id:
            raise HTTPException(status_code=422, detail="team instance id does not match URL")
        try:
            saved = team_instance_registry.save({**payload, "id": instance_id})
            return team_instance_availability(
                saved,
                saved["team_spec"],
                deployment_registry.instances(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/team-instances/{instance_id}")
    def delete_team_instance(instance_id: str):
        try:
            if any(
                item.get("team_instance_id") == instance_id
                for item in experiment_registry.instances()
            ):
                raise ValueError("TeamInstance is referenced by an ExperimentInstance")
            return team_instance_registry.delete(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs")
    def runs(limit: int = Query(default=100, ge=1, le=1000)):
        return catalog.runs(limit=limit, roots=registered_runs_roots())

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return read_jsonl_events(run_dir / "spans.jsonl", start_line=start_line, limit=limit)

    @app.get("/api/runs/{run_id}/group-chat")
    def run_group_chat(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return read_jsonl_events(run_dir / "group_chat.jsonl", start_line=start_line, limit=limit)

    @app.get("/api/runs/{run_id}/evidence")
    def run_evidence(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return read_jsonl_events(run_dir / "evidence.jsonl", start_line=start_line, limit=limit)

    @app.get("/api/runs/{run_id}/evidence-coverage")
    def run_evidence_coverage(run_id: str):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        coverage_path = run_dir / "evidence_coverage.json"
        if coverage_path.is_file():
            try:
                value = json.loads(coverage_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            if isinstance(value, dict):
                return value
        from ..evidence import normalize_run_evidence

        return normalize_run_evidence(run_dir, write=False)

    @app.get("/api/runs/{run_id}/events/stream")
    async def stream_events(run_id: str, start_line: int = Query(default=0, ge=0)):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        spans_path = run_dir / "spans.jsonl"

        async def generate():
            cursor = start_line
            while True:
                batch = read_jsonl_events(
                    spans_path, start_line=cursor, limit=200, include_total=False
                )
                for event in batch["events"]:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                cursor = batch["next_line"]
                yield f"event: cursor\ndata: {cursor}\n\n"
                await asyncio.sleep(0.75)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/group-chat/stream")
    async def stream_group_chat(run_id: str, start_line: int = Query(default=0, ge=0)):
        try:
            run_dir = catalog.run_dir(run_id, roots=registered_runs_roots())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        group_chat_path = run_dir / "group_chat.jsonl"

        async def generate():
            cursor = start_line
            while True:
                batch = read_jsonl_events(
                    group_chat_path, start_line=cursor, limit=200, include_total=False
                )
                for event in batch["events"]:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                cursor = batch["next_line"]
                yield f"event: cursor\ndata: {cursor}\n\n"
                await asyncio.sleep(0.75)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/tmux/sessions")
    def tmux_sessions():
        return jobs.tmux_sessions()

    def launch_dataset_prepare(payload: dict[str, Any], target: str) -> dict[str, Any]:
        from ..benchmarks import BENCHMARKS, PREPARERS

        if target not in PREPARERS:
            raise ValueError(f"unknown prepare target {target!r}")
        raw_root = _allowed_path(payload.get("raw_root") or catalog.raw_root, root)
        prepared_root = _allowed_path(payload.get("prepared_root") or catalog.prepared_root, root)
        python = _python_executable(payload.get("python"), default=sys.executable)
        source = str(payload.get("source") or "auto")
        if source not in {"auto", "modelscope", "huggingface", "github"}:
            raise ValueError("benchmark source must be auto, modelscope, huggingface or github")
        BENCHMARKS.for_prepare_target(target).source_backend_order(source)
        command = [
            python,
            "scripts/prepare_benchmarks.py",
            "--raw-root",
            str(raw_root),
            "--prepared-root",
            str(prepared_root),
            "--source",
            source,
            "--tasks",
            target,
            "--docker-images",
            str(payload.get("docker_images") or "auto"),
        ]
        if payload.get("conversion_only"):
            command.append("--conversion-only")
        if payload.get("force"):
            command.append("--force")
        job_id = f"prepare-{target}-{uuid.uuid4().hex[:8]}"
        environment = _child_environment(payload)
        raw_overrides = payload.get("raw_overrides") or []
        if not isinstance(raw_overrides, list):
            raise ValueError("raw_overrides must be a list")
        environment["LYCHEE_BENCHMARK_RAW_OVERRIDES"] = json.dumps(raw_overrides)
        return jobs.launch_utility(
            job_id=job_id,
            command=command,
            cwd=root,
            job_root=launches_root,
            env=environment,
        )

    @app.post("/api/datasets/prepare")
    def prepare_dataset(payload: dict[str, Any]):
        target = str(payload.get("target") or "")
        try:
            return launch_dataset_prepare(payload, target)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/model-instances/{instance_id}/instantiate")
    def instantiate_model_instance(instance_id: str, payload: dict[str, Any]):
        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", instance_id):
                raise ValueError("invalid ModelInstance id")
            if any(item["id"] == instance_id for item in model_registry.instances()):
                raise ValueError(f"ModelInstance {instance_id!r} already exists")
            spec = model_registry.get_spec(str(payload.get("model_spec_id") or ""))
            mode = str(payload.get("mode") or "download")
            if mode not in {"local", "download"}:
                raise ValueError("model acquisition mode must be local or download")
            job = None
            if mode == "local":
                source_value = str(payload.get("source_path") or "").strip()
                if not source_value:
                    raise ValueError("local model acquisition requires source_path")
                target_dir = Path(source_value).expanduser().resolve()
                saved = model_registry.save_instance(
                    {
                        "id": instance_id,
                        "model_spec_id": spec["id"],
                        "acquisition": {
                            "mode": "local",
                            "source": "local",
                            "source_path": str(target_dir),
                            "revision": "",
                        },
                        "path": str(target_dir),
                        "managed": False,
                        "status": "unknown",
                    }
                )
                if not saved["available"]:
                    model_registry.delete_instance(instance_id)
                    raise ValueError(
                        "local model path is not a complete Transformers model "
                        "(config, weights and tokenizer are required)"
                    )
            else:
                source = str(payload.get("source") or "auto")
                if source not in {"auto", "huggingface", "modelscope"}:
                    raise ValueError("invalid registered model source")
                if not model_registry.sources(spec["id"], source):
                    raise ValueError(f"ModelSpec {spec['id']!r} has no {source!r} download source")
                models_root = _allowed_path(payload.get("models_root") or catalog.models_root, root)
                target_dir = models_root / instance_id
                python = _python_executable(payload.get("python"), default=sys.executable)
                command = [
                    python,
                    "scripts/download_model.py",
                    "--target-dir",
                    str(target_dir),
                    "--model-spec-id",
                    spec["id"],
                    "--source",
                    source,
                    "--models-root",
                    str(models_root),
                ]
                if payload.get("revision"):
                    command.extend(["--revision", str(payload["revision"])])
                job_id = f"model-{instance_id}-{uuid.uuid4().hex[:8]}".replace("_", "-")
                job = jobs.launch_utility(
                    job_id=job_id,
                    command=command,
                    cwd=root,
                    job_root=launches_root,
                    env=_child_environment(payload),
                )
                saved = model_registry.save_instance(
                    {
                        "id": instance_id,
                        "model_spec_id": spec["id"],
                        "acquisition": {
                            "mode": "download",
                            "source": source,
                            "source_path": "",
                            "revision": str(payload.get("revision") or ""),
                        },
                        "path": str(target_dir),
                        "managed": True,
                        "status": "preparing",
                        "job_id": job["launch_id"],
                    }
                )
            return {"instance": saved, "job": job}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/resources/import")
    def reject_unregistered_resource_import(_payload: dict[str, Any]):
        raise HTTPException(
            status_code=422,
            detail=(
                "Resources must be created by instantiating a registered "
                "BenchmarkSpec, ModelSpec or APISpec"
            ),
        )

    @app.get("/api/launches/{launch_id}")
    def launch_status(launch_id: str):
        try:
            launch_dir = _launch_dir(launch_id, launches_root)
            return jobs.status(launch_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/launches/{launch_id}/log")
    def launch_log(launch_id: str, tail: int = Query(default=200, ge=1, le=5000)):
        try:
            launch_dir = _launch_dir(launch_id, launches_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = launch_dir / "launch.log"
        if not path.is_file():
            return {"lines": [], "path": str(path)}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-tail:], "path": str(path)}

    @app.get("/api/launches/{launch_id}/progress")
    def launch_progress(launch_id: str):
        try:
            return jobs.progress(_launch_dir(launch_id, launches_root))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/launches/{launch_id}/stop")
    def stop_launch(launch_id: str):
        try:
            launch_dir = _launch_dir(launch_id, launches_root)
            return jobs.stop(launch_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/experiment-specs")
    def list_experiment_specs():
        return experiment_registry.specs()

    @app.put("/api/experiment-specs/{spec_id}")
    def save_experiment_spec(spec_id: str, spec: dict[str, Any]):
        try:
            normalized = validated_experiment_spec({**spec, "id": spec_id})
            return experiment_registry.save_spec(spec_id, normalized)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/experiment-specs/{spec_id}")
    def delete_experiment_spec(spec_id: str):
        try:
            return experiment_registry.delete_spec(spec_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/experiment-instances")
    def list_experiment_instances():
        return experiment_queue.instances()

    @app.post("/api/experiment-instances")
    def create_experiment_instance(payload: dict[str, Any]):
        try:
            spec_id = str(payload.get("experiment_spec_id") or "").strip()
            spec = validated_experiment_spec(experiment_registry.get_spec(spec_id))
            benchmark_instance_id = str(payload.get("benchmark_instance_id") or "").strip()
            team_instance_id = str(payload.get("team_instance_id") or "").strip()
            instance_id = str(payload.get("id") or f"{spec_id}-{uuid.uuid4().hex[:8]}")
            requested_run_dir = str(payload.get("run_dir") or "").strip()
            launcher = normalize_launcher(payload.get("launcher"))
            if requested_run_dir:
                spec.setdefault("environment", {})["run_dir"] = requested_run_dir
            project = assemble_project(
                spec,
                benchmark_instance_id=benchmark_instance_id,
                team_instance_id=team_instance_id,
                experiment_instance={
                    "id": instance_id,
                    "launcher": launcher,
                    "status": "ready",
                    "queue": {"priority": int(payload.get("priority", 100))},
                    "run_dir": requested_run_dir or None,
                },
            )
            plan = compiler.compile(project, launch_id=instance_id)
            return experiment_registry.create_instance(
                instance_id=instance_id,
                spec_id=spec_id,
                benchmark_instance_id=benchmark_instance_id,
                team_instance_id=team_instance_id,
                run_dir=str(plan.run_dir),
                priority=int(payload.get("priority", 100)),
                launcher=launcher,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/experiment-instances/{instance_id}")
    def delete_experiment_instance(instance_id: str):
        try:
            return experiment_queue.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-instances/{instance_id}/plan")
    def plan_experiment_instance(instance_id: str):
        try:
            return experiment_queue.plan_instance(instance_id).as_dict()
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-instances/{instance_id}/launch")
    def launch_experiment_instance(instance_id: str):
        try:
            return experiment_queue.launch_now(instance_id)
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            subprocess.CalledProcessError,
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-instances/{instance_id}/enqueue")
    def enqueue_experiment_instance(instance_id: str):
        try:
            return experiment_queue.enqueue(instance_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-instances/{instance_id}/dequeue")
    def dequeue_experiment_instance(instance_id: str):
        try:
            return experiment_queue.dequeue(instance_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-instances/{instance_id}/stop")
    def stop_experiment_instance(instance_id: str):
        try:
            return experiment_queue.stop_instance(instance_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/experiment-instances/{instance_id}/progress")
    def experiment_instance_progress(instance_id: str, tail: int = Query(40, ge=0, le=200)):
        try:
            return experiment_queue.instance_progress(instance_id, tail=tail)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/experiment-orphans")
    def list_orphaned_experiment_launches():
        return experiment_queue.orphaned_launches()

    @app.post("/api/experiment-orphans/{launch_id}/stop")
    def stop_orphaned_experiment_launch(launch_id: str):
        try:
            return experiment_queue.stop_orphaned_launch(launch_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-orphans/{launch_id}/recover")
    def recover_orphaned_experiment_launch(launch_id: str):
        try:
            return experiment_queue.recover_orphaned_launch(launch_id)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/experiment-queue")
    def experiment_queue_status():
        return experiment_queue.status()

    @app.post("/api/experiment-queue/start")
    def start_experiment_queue(payload: dict[str, Any] | None = None):
        try:
            return experiment_queue.start(
                max_parallel_instances=int((payload or {}).get("max_parallel_instances", 8))
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiment-queue/stop-after-current")
    def stop_experiment_queue():
        return experiment_queue.stop_after_current()

    frontend_dist = root / "apps/eval_studio/frontend/dist"
    if frontend_dist.is_dir():
        assets = frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{page:path}", include_in_schema=False)
        def frontend(page: str):
            candidate = (frontend_dist / page).resolve()
            if page and candidate.is_file() and frontend_dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app


def _safe_name(value: str) -> str:
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("invalid name")
    return value


def _launch_dir(launch_id: str, launches_root: Path) -> Path:
    safe = _safe_name(launch_id)
    path = (launches_root / safe).resolve()
    if launches_root.resolve() not in path.parents:
        raise ValueError("invalid launch path")
    return path


def _allowed_path(value: Any, repo_root: Path) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside repository root {repo_root}") from exc
    return path


def _python_executable(value: Any, *, default: str) -> str:
    path = Path(str(value or default)).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python executable is missing or not executable: {path}")
    return str(path)


def _child_environment(payload: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    proxy = payload.get("proxy") if isinstance(payload.get("proxy"), dict) else {}
    if proxy.get("enabled"):
        mappings = {
            "http_proxy": ("HTTP_PROXY", "http_proxy"),
            "https_proxy": ("HTTPS_PROXY", "https_proxy"),
            "all_proxy": ("ALL_PROXY", "all_proxy"),
            "no_proxy": ("NO_PROXY", "no_proxy"),
        }
        for field, names in mappings.items():
            value = str(proxy.get(field) or "").strip()
            if not value:
                continue
            if field != "no_proxy":
                parsed = urlparse(value)
                if (
                    parsed.scheme not in {"http", "https", "socks5", "socks5h"}
                    or not parsed.hostname
                ):
                    raise ValueError(f"invalid {field} URL")
            for name in names:
                env[name] = value
    mirrors = payload.get("mirrors") if isinstance(payload.get("mirrors"), dict) else {}
    if mirrors.get("enabled"):
        index_url = str(mirrors.get("pip_index_url") or "").strip()
        extra_index_url = str(mirrors.get("pip_extra_index_url") or "").strip()
        for value, field in (
            (index_url, "pip_index_url"),
            (extra_index_url, "pip_extra_index_url"),
        ):
            if value:
                parsed = urlparse(value)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError(f"invalid {field} URL")
        if index_url:
            env.update(
                PIP_INDEX_URL=index_url,
                UV_INDEX_URL=index_url,
                UV_DEFAULT_INDEX=index_url,
            )
        if extra_index_url:
            env["PIP_EXTRA_INDEX_URL"] = extra_index_url
    return env
