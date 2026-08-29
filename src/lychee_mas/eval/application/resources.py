"""Model, API-access, and Benchmark resource use cases plus utility jobs."""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from ..benchmarks import BENCHMARKS, PREPARERS
from ..benchmarks.manifest import benchmark_key_for_target
from ..environment.service import build_child_environment
from .ports import (
    APIAccessRegistryPort,
    BenchmarkRegistryPort,
    DeploymentRegistryPort,
    ExperimentRepositoryPort,
    JobManagerPort,
    ModelRegistryPort,
)
from .read_models.catalog import EvalCatalog


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("invalid name")
    return value


class ResourceApplicationService:
    """Coordinate registered assets without inventing a fourth resource domain."""

    def __init__(
        self,
        *,
        repo_root: Path,
        catalog: EvalCatalog,
        jobs: JobManagerPort,
        benchmarks: BenchmarkRegistryPort,
        models: ModelRegistryPort,
        api_access: APIAccessRegistryPort,
        deployments: DeploymentRegistryPort,
        experiments: ExperimentRepositoryPort,
    ) -> None:
        self.root = repo_root.resolve()
        self.catalog = catalog
        self.jobs = jobs
        self.benchmarks = benchmarks
        self.models = models
        self.api_access = api_access
        self.deployments = deployments
        self.experiments = experiments
        self.launches_root = self.root / "runs/eval_studio/launches"
        self.launches_root.mkdir(parents=True, exist_ok=True)

    def catalog_payload(
        self,
        *,
        raw_root: str | None,
        prepared_root: str | None,
        models_root: str | None,
    ) -> dict[str, Any]:
        dynamic = EvalCatalog(
            self.root,
            raw_root=Path(raw_root or self.catalog.raw_root).expanduser().resolve(),
            prepared_root=Path(
                prepared_root or self.catalog.prepared_root
            ).expanduser().resolve(),
            models_root=Path(models_root or self.catalog.models_root).expanduser().resolve(),
            runs_root=self.catalog.runs_root,
        )
        return {
            "paths": dynamic.paths(),
            "benchmarks": dynamic.benchmarks(),
            "benchmark_specs": self.benchmarks.specs(),
            "benchmark_instances": self.benchmarks.instances(),
            "model_specs": self.models.specs(),
            "model_instances": self.models.instances(),
            "api_registry": {
                "specs": self.api_access.specs(),
                "instances": self.api_access.instances(),
            },
        }

    def registries(self) -> dict[str, Any]:
        return {
            "models": {"specs": self.models.specs(), "instances": self.models.instances()},
            "apis": {
                "specs": self.api_access.specs(),
                "instances": self.api_access.instances(),
            },
        }

    def api_specs(self) -> list[dict[str, Any]]:
        return self.api_access.specs()

    def api_instances(self) -> list[dict[str, Any]]:
        return self.api_access.instances()

    def instantiate_api(self, instance_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if any(item["id"] == instance_id for item in self.api_access.instances()):
            raise ValueError(f"APIInstance {instance_id!r} already exists")
        return self.api_access.save_instance({**payload, "id": instance_id})

    def delete_api(self, instance_id: str) -> dict[str, Any]:
        if any(
            (item.get("source_instance") or {}).get("type") == "api"
            and (item.get("source_instance") or {}).get("id") == instance_id
            for item in self.deployments.instances()
        ):
            raise ValueError("APIInstance is referenced by a DeploymentInstance")
        return self.api_access.delete_instance(instance_id)

    def model_specs(self) -> list[dict[str, Any]]:
        return self.models.specs()

    def model_instances(self) -> list[dict[str, Any]]:
        return self.models.instances()

    def scan_models(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.models.scan_root(payload.get("models_root") or self.catalog.models_root)

    def delete_model(self, instance_id: str) -> dict[str, Any]:
        if any(
            (item.get("source_instance") or {}).get("type") == "model"
            and (item.get("source_instance") or {}).get("id") == instance_id
            for item in self.deployments.instances()
        ):
            raise ValueError("ModelInstance is referenced by a DeploymentInstance")
        return self.models.delete_instance(instance_id)

    def instantiate_model(
        self,
        instance_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _safe_name(instance_id)
        if any(item["id"] == instance_id for item in self.models.instances()):
            raise ValueError(f"ModelInstance {instance_id!r} already exists")
        spec = self.models.get_spec(str(payload.get("model_spec_id") or ""))
        mode = str(payload.get("mode") or "download")
        if mode not in {"local", "download"}:
            raise ValueError("model acquisition mode must be local or download")
        if mode == "local":
            source_value = str(payload.get("source_path") or "").strip()
            if not source_value:
                raise ValueError("local model acquisition requires source_path")
            target_dir = Path(source_value).expanduser().resolve()
            saved = self.models.save_instance(
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
                self.models.delete_instance(instance_id)
                raise ValueError(
                    "local model path is not a complete Transformers model "
                    "(config, weights and tokenizer are required)"
                )
            return {"instance": saved, "job": None}

        source = str(payload.get("source") or "auto")
        if source not in {"auto", "huggingface", "modelscope"}:
            raise ValueError("invalid registered model source")
        if not self.models.sources(spec["id"], source):
            raise ValueError(f"ModelSpec {spec['id']!r} has no {source!r} download source")
        models_root = self._allowed_path(
            payload.get("models_root") or self.catalog.models_root
        )
        target_dir = models_root / instance_id
        python = self._python_executable(payload.get("python"))
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
        job = self._launch_utility(
            job_id=f"model-{instance_id}-{uuid.uuid4().hex[:8]}".replace("_", "-"),
            command=command,
            payload=payload,
        )
        saved = self.models.save_instance(
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

    def benchmark_catalog(self) -> list[dict[str, Any]]:
        return self.catalog.benchmarks()

    def benchmark_specs(self) -> list[dict[str, Any]]:
        return self.benchmarks.specs()

    def benchmark_instances(self) -> list[dict[str, Any]]:
        return self.benchmarks.instances()

    def scan_benchmarks(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.benchmarks.scan_roots(
            raw_root=payload.get("raw_root") or self.catalog.raw_root,
            prepared_root=payload.get("prepared_root") or self.catalog.prepared_root,
        )

    def instantiate_benchmark(
        self,
        instance_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _safe_name(instance_id)
        if any(item["id"] == instance_id for item in self.benchmarks.instances()):
            raise ValueError(f"BenchmarkInstance {instance_id!r} already exists")
        spec = self.benchmarks.get_spec(str(payload.get("benchmark_spec_id") or ""))
        mode = str(payload.get("mode") or "download")
        if mode not in {"local", "download"}:
            raise ValueError("benchmark acquisition mode must be local or download")
        raw_root = self._allowed_path(payload.get("raw_root") or self.catalog.raw_root)
        prepared_root = self._allowed_path(
            payload.get("prepared_root") or self.catalog.prepared_root
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
                job = self.prepare_benchmark(
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
                    target=spec["prepare_target"],
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
            job = self.prepare_benchmark(
                {**payload, "source": source},
                target=spec["prepare_target"],
            )
            provenance = [
                {
                    "selection": "requested",
                    "provider": source,
                    "registered_candidates": registered_candidates,
                }
            ]

        manifest_root = prepared_path if prepared_path.is_dir() else prepared_path.parent
        saved = self.benchmarks.save_instance(
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

    def check_benchmark(self, instance_id: str) -> dict[str, Any]:
        return self.benchmarks.check_instance(instance_id)

    def delete_benchmark(self, instance_id: str) -> dict[str, Any]:
        if any(
            item.get("benchmark_instance_id") == instance_id
            for item in self.experiments.instances()
        ):
            raise ValueError("BenchmarkInstance is referenced by an ExperimentInstance")
        return self.benchmarks.delete_instance(instance_id)

    def prepare_benchmark(
        self,
        payload: dict[str, Any],
        *,
        target: str,
    ) -> dict[str, Any]:
        if target not in PREPARERS:
            raise ValueError(f"unknown prepare target {target!r}")
        raw_root = self._allowed_path(payload.get("raw_root") or self.catalog.raw_root)
        prepared_root = self._allowed_path(
            payload.get("prepared_root") or self.catalog.prepared_root
        )
        source = str(payload.get("source") or "auto")
        if source not in {"auto", "modelscope", "huggingface", "github"}:
            raise ValueError(
                "benchmark source must be auto, modelscope, huggingface or github"
            )
        BENCHMARKS.for_prepare_target(target).source_backend_order(source)
        command = [
            self._python_executable(payload.get("python")),
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
        environment = build_child_environment(payload)
        raw_overrides = payload.get("raw_overrides") or []
        if not isinstance(raw_overrides, list):
            raise ValueError("raw_overrides must be a list")
        environment["LYCHEE_BENCHMARK_RAW_OVERRIDES"] = json.dumps(raw_overrides)
        return self._launch_utility(
            job_id=f"prepare-{target}-{uuid.uuid4().hex[:8]}",
            command=command,
            payload=payload,
            environment=environment,
        )

    def tmux_sessions(self) -> list[dict[str, Any]]:
        return self.jobs.tmux_sessions()

    def launch_status(self, launch_id: str) -> dict[str, Any]:
        return self.jobs.status(self._launch_dir(launch_id))

    def launch_log(self, launch_id: str, *, tail: int) -> dict[str, Any]:
        path = self._launch_dir(launch_id) / "launch.log"
        if not path.is_file():
            return {"lines": [], "path": str(path)}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-tail:], "path": str(path)}

    def launch_progress(self, launch_id: str) -> dict[str, Any]:
        return self.jobs.progress(self._launch_dir(launch_id))

    def stop_launch(self, launch_id: str) -> dict[str, Any]:
        return self.jobs.stop(self._launch_dir(launch_id))

    def _launch_utility(
        self,
        *,
        job_id: str,
        command: list[str],
        payload: dict[str, Any],
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self.jobs.launch_utility(
            job_id=job_id,
            command=command,
            cwd=self.root,
            job_root=self.launches_root,
            env=environment or build_child_environment(payload),
        )

    def _launch_dir(self, launch_id: str) -> Path:
        path = (self.launches_root / _safe_name(launch_id)).resolve()
        if self.launches_root.resolve() not in path.parents:
            raise ValueError("invalid launch path")
        return path

    def _allowed_path(self, value: Any) -> Path:
        path = Path(value).expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path must stay inside repository root {self.root}") from exc
        return path

    @staticmethod
    def _python_executable(value: Any) -> str:
        path = Path(str(value or sys.executable)).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Python executable is missing or not executable: {path}")
        return str(path)
