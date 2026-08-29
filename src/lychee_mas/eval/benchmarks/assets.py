"""File-backed BenchmarkSpec and BenchmarkInstance registries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..contracts.lifecycle import (
    bind_instance_to_spec,
    lifecycle_error,
    normalize_instance_lifecycle,
    utc_now,
)
from ..infrastructure.json_store import atomic_write_json as _write_json

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_LOADER_PROBE_MARKER = "__LYCHEE_BENCHMARK_LOADER_PROBE__="
_LOADER_PROBE = rf"""
import json
import os

marker = {_LOADER_PROBE_MARKER!r}
tasks = json.loads(os.environ["LYCHEE_BENCHMARK_LOADER_PROBE_TASKS"])
required_fields = {{"task", "kind", "question", "gold", "context"}}
results = []

try:
    from lychee_mas.eval.benchmarks import load
except Exception as exc:
    results.append({{"task": "<registry>", "status": "failed", "detail": repr(exc)}})
else:
    for task in tasks:
        try:
            rows = load(task, n=1)
            if not rows:
                raise ValueError("loader returned no rows")
            row = rows[0]
            if not isinstance(row, dict):
                raise TypeError(f"loader returned {{type(row).__name__}}, expected dict")
            missing = sorted(required_fields - set(row))
            if missing:
                raise ValueError("loader row is missing fields: " + ", ".join(missing))
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {{}}
            case_id = (
                metadata.get("id")
                or metadata.get("task_id")
                or row.get("task")
                or "available"
            )
            results.append({{
                "task": task,
                "status": "ready",
                "kind": str(row.get("kind") or ""),
                "case_id": str(case_id),
                "sample_count": len(rows),
            }})
        except Exception as exc:
            results.append({{"task": task, "status": "failed", "detail": repr(exc)}})

status = "ready" if results and all(item["status"] == "ready" for item in results) else "failed"
print(marker + json.dumps({{"status": status, "tasks": results}}, ensure_ascii=False))
"""


def _utc_now() -> str:
    return utc_now()


def normalize_benchmark_spec(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("BenchmarkSpec must be a JSON object")
    spec_id = str(value.get("id") or "").strip()
    if not _SAFE_ID.fullmatch(spec_id):
        raise ValueError("BenchmarkSpec id may contain only letters, numbers, '.', '_' and '-'")
    try:
        from ..benchmarks import BENCHMARKS

        implementation = BENCHMARKS.get(spec_id)
    except KeyError as exc:
        raise ValueError(
            f"BenchmarkSpec {spec_id!r} has no registered Benchmark implementation"
        ) from exc
    allowed = {"schema_version", "id", "name", "category"}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(
            "BenchmarkSpec stores identity metadata only; move implementation fields to "
            f"Benchmark: {', '.join(unexpected)}"
        )
    return {
        "schema_version": 4,
        "id": spec_id,
        "name": str(value.get("name") or implementation.name),
        "category": str(value.get("category") or implementation.category),
    }


def benchmark_spec_view(spec: dict[str, Any]) -> dict[str, Any]:
    """Combine a minimal persisted Spec with its read-only Benchmark contract."""

    from ..benchmarks import BENCHMARKS

    implementation = BENCHMARKS.get(str(spec["id"]))
    descriptor = implementation.descriptor()
    derived = {
        key: value
        for key, value in descriptor.items()
        if key not in {"schema_version", "id", "name", "category"}
    }
    return {**spec, **derived, "implementation": descriptor}


def resolve_scoring_profile(
    benchmark_spec: dict[str, Any], task: str, profile_id: str | None = None
) -> dict[str, Any]:
    """Resolve one allowed scoring profile from the merged Benchmark view."""

    contract = dict((benchmark_spec.get("task_contracts") or {}).get(task) or {})
    if not contract:
        raise ValueError(f"Benchmark {benchmark_spec.get('id')!r} does not register task {task!r}")
    scoring = dict(contract.get("scoring") or {})
    selected = str(profile_id or scoring.get("default_profile") or "")
    profile = dict((scoring.get("profiles") or {}).get(selected) or {})
    if not profile:
        raise ValueError(f"Benchmark task {task!r} does not register scoring profile {selected!r}")
    return {
        "profile_id": selected,
        "scorer_id": profile["scorer_id"],
        "parameters": dict(profile.get("parameters") or {}),
    }


def normalize_benchmark_instance(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("BenchmarkInstance must be a JSON object")
    instance_id = str(value.get("id") or "").strip()
    spec_id = str(value.get("benchmark_spec_id") or "").strip()
    if not _SAFE_ID.fullmatch(instance_id) or not _SAFE_ID.fullmatch(spec_id):
        raise ValueError("BenchmarkInstance requires valid id and benchmark_spec_id")
    prepared_path = str(value.get("prepared_path") or "").strip()
    if not prepared_path:
        raise ValueError("BenchmarkInstance requires prepared_path")
    acquisition = dict(value.get("acquisition") or {})
    mode = str(acquisition.get("mode") or "local")
    if mode not in {"local", "download"}:
        raise ValueError("BenchmarkInstance acquisition mode must be local or download")
    lifecycle = normalize_instance_lifecycle(value, prefix="benchmark")
    return {
        "schema_version": 3,
        "id": instance_id,
        **lifecycle,
        "acquisition": {
            "mode": mode,
            "stage": str(acquisition.get("stage") or "prepared"),
            "source": str(acquisition.get("source") or "local"),
            "source_path": str(acquisition.get("source_path") or ""),
        },
        "raw_path": str(value.get("raw_path") or ""),
        "prepared_path": prepared_path,
        "manifest_path": str(value.get("manifest_path") or ""),
        "supported_tasks": [str(item) for item in value.get("supported_tasks") or []],
        "status": str(value.get("status") or "unknown"),
        "integrity": dict(value.get("integrity") or {}),
        "loader_check": dict(value.get("loader_check") or {}),
        "provenance": list(value.get("provenance") or []),
        "managed": bool(value.get("managed", True)),
        "job_id": str(value.get("job_id") or ""),
        "updated_at_utc": value.get("updated_at_utc"),
    }


class BenchmarkRegistry:
    """Store benchmark contracts separately from prepared data instances."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/benchmarks"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                **benchmark_spec_view(spec),
                "registry_path": str(self.spec_root / f"{spec['id']}.json"),
            }
            for spec in self._spec_documents()
        ]

    def _spec_documents(self) -> list[dict[str, Any]]:
        if not self.spec_root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.spec_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                rows.append(normalize_benchmark_spec(raw))
            except (OSError, TypeError, ValueError):
                continue
        return rows

    def _spec_document(self, spec_id: str) -> dict[str, Any]:
        match = next((item for item in self._spec_documents() if item["id"] == spec_id), None)
        if match is None:
            raise ValueError(f"unknown registered BenchmarkSpec {spec_id!r}")
        return match

    def instances(self) -> list[dict[str, Any]]:
        spec_documents = {item["id"]: item for item in self._spec_documents()}
        spec_views = {item["id"]: item for item in self.specs()}
        rows: list[dict[str, Any]] = []
        if not self.instance_root.is_dir():
            return rows
        for registry_path in sorted(self.instance_root.glob("*.json")):
            try:
                raw = json.loads(registry_path.read_text(encoding="utf-8"))
                row = self._inspect_instance(
                    self._materialize_instance(normalize_benchmark_instance(raw))
                )
            except (OSError, TypeError, ValueError):
                continue
            row["registry_path"] = str(registry_path)
            rows.append(row)
        for row in rows:
            row["benchmark_spec"] = spec_views.get(row["benchmark_spec_id"])
            lifecycle_failure = lifecycle_error(
                row,
                spec=spec_documents.get(row["benchmark_spec_id"]),
                prefix="benchmark",
            )
            row["configuration_status"] = "invalid" if lifecycle_failure else "current"
            row["validation_error"] = lifecycle_failure
            loader_status = str((row.get("loader_check") or {}).get("status") or "unchecked")
            row["loader_status"] = loader_status
            if lifecycle_failure:
                row["observed_status"] = "invalid"
            elif row["status"] != "ready":
                row["observed_status"] = row["status"]
            elif loader_status == "failed":
                row["observed_status"] = "loader_failed"
            else:
                row["observed_status"] = "ready"
            row["available"] = (
                not lifecycle_failure and row["status"] == "ready" and loader_status != "failed"
            )
        return rows

    def save_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_benchmark_spec(value)
        path = self.spec_root / f"{normalized['id']}.json"
        _write_json(path, normalized)
        return {**benchmark_spec_view(normalized), "registry_path": str(path)}

    def save_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        spec_id = str(value.get("benchmark_spec_id") or "")
        spec_document = self._spec_document(spec_id)
        spec = benchmark_spec_view(spec_document)
        payload = bind_instance_to_spec(
            value,
            spec=spec_document,
            prefix="benchmark",
            creation_source=str(value.get("creation_source") or "studio"),
            created_at_utc=value.get("created_at_utc"),
        )
        normalized = self._portable_instance(normalize_benchmark_instance(payload))
        supported_tasks = normalized["supported_tasks"] or spec["runnable_tasks"]
        unknown_tasks = sorted(set(supported_tasks) - set(spec["runnable_tasks"]))
        if unknown_tasks:
            raise ValueError(
                "BenchmarkInstance contains tasks outside its Benchmark implementation: "
                + ", ".join(unknown_tasks)
            )
        normalized["supported_tasks"] = supported_tasks
        normalized["updated_at_utc"] = _utc_now()
        path = self.instance_root / f"{normalized['id']}.json"
        _write_json(path, normalized)
        return next(item for item in self.instances() if item["id"] == normalized["id"])

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        document = self._spec_document(spec_id)
        return {
            **benchmark_spec_view(document),
            "registry_path": str(self.spec_root / f"{spec_id}.json"),
        }

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        match = next((item for item in self.instances() if item["id"] == instance_id), None)
        if match is None:
            raise ValueError(f"unknown BenchmarkInstance {instance_id!r}")
        if match.get("configuration_status") == "invalid":
            raise ValueError(
                f"BenchmarkInstance {instance_id!r} is invalid: {match.get('validation_error')}"
            )
        return match

    def check_instance(self, instance_id: str) -> dict[str, Any]:
        """Load one sample from every runnable task owned by an instance.

        File and manifest inspection happens while listing instances. This probe is
        deliberately separate: it validates the prepared data through the same
        public loader contract that an experiment will use.
        """

        instance = self.get_instance(instance_id)
        if instance["status"] != "ready":
            raise ValueError(
                f"BenchmarkInstance {instance_id!r} prepared data is {instance['status']!r}"
            )
        spec = instance.get("benchmark_spec") or self.get_spec(instance["benchmark_spec_id"])
        tasks = list(instance.get("supported_tasks") or spec["runnable_tasks"])
        if not tasks:
            raise ValueError(f"BenchmarkInstance {instance_id!r} has no runnable tasks")

        from ..benchmarks.manifest import benchmark_key_for_target

        benchmark_key = benchmark_key_for_target(spec["prepare_target"])
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.repo_root / "src"), current_pythonpath) if part
        )
        env["LYCHEE_BENCHMARK_PREPARED_OVERRIDES"] = json.dumps(
            {benchmark_key: instance["prepared_path"]}, ensure_ascii=False
        )
        env["LYCHEE_BENCHMARK_LOADER_PROBE_TASKS"] = json.dumps(tasks, ensure_ascii=False)
        checked_at = _utc_now()
        try:
            process = subprocess.run(
                [sys.executable, "-c", _LOADER_PROBE],
                cwd=self.repo_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )
            payload = self._loader_probe_payload(process.stdout)
            if process.returncode or payload is None:
                payload = {
                    "status": "failed",
                    "tasks": [
                        {
                            "task": "<probe>",
                            "status": "failed",
                            "detail": (
                                f"loader probe exited with code {process.returncode}"
                                if process.returncode
                                else "loader probe did not return a result"
                            ),
                        }
                    ],
                }
            loader_check = {
                "schema_version": 1,
                "status": str(payload.get("status") or "failed"),
                "checked_at_utc": checked_at,
                "python": sys.executable,
                "benchmark_key": benchmark_key,
                "tasks": list(payload.get("tasks") or []),
                "return_code": process.returncode,
                "stderr": process.stderr[-4000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            loader_check = {
                "schema_version": 1,
                "status": "failed",
                "checked_at_utc": checked_at,
                "python": sys.executable,
                "benchmark_key": benchmark_key,
                "tasks": [{"task": "<probe>", "status": "failed", "detail": repr(exc)}],
                "return_code": None,
                "stderr": "",
            }

        path = self.instance_root / f"{instance_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot update BenchmarkInstance {instance_id!r}: {exc}") from exc
        raw["loader_check"] = loader_check
        raw["updated_at_utc"] = checked_at
        _write_json(path, normalize_benchmark_instance(raw))
        return self.get_instance(instance_id)

    @staticmethod
    def _loader_probe_payload(stdout: str) -> dict[str, Any] | None:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(_LOADER_PROBE_MARKER):
                continue
            try:
                value = json.loads(line[len(_LOADER_PROBE_MARKER) :])
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None
        return None

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(instance_id):
            raise ValueError("invalid BenchmarkInstance id")
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": instance_id, "deleted": True, "path": str(path)}

    def scan_roots(
        self,
        *,
        raw_root: str | os.PathLike,
        prepared_root: str | os.PathLike,
    ) -> dict[str, Any]:
        """Register assets that can be tied unambiguously to a BenchmarkSpec.

        Scanning is intentionally non-owning: it records the discovered paths but
        never copies, rewrites, or deletes data. Repeating the same scan is
        idempotent because existing instances are matched by their resolved paths.
        """

        from ..benchmarks.manifest import benchmark_key_for_target

        raw_base = Path(raw_root).expanduser().resolve()
        prepared_base = Path(prepared_root).expanduser().resolve()
        if not raw_base.is_dir() and not prepared_base.is_dir():
            raise ValueError("neither Benchmark raw root nor prepared root exists")

        existing = self.instances()
        findings: list[dict[str, Any]] = []
        created: list[dict[str, Any]] = []
        for spec in self.specs():
            key = benchmark_key_for_target(spec["prepare_target"])
            raw_sources = self._scan_registered_raw_sources(raw_base / key, key)
            prepared = self._scan_prepared_asset(prepared_base / key, key)
            if not raw_sources and not prepared:
                continue
            prepared_data = prepared or {}

            prepared_path = prepared_base / key
            raw_path = Path(raw_sources[0]["raw_path"]) if raw_sources else None
            match = next(
                (
                    item
                    for item in existing
                    if item["benchmark_spec_id"] == spec["id"]
                    and (
                        self._same_path(item.get("prepared_path"), prepared_path)
                        or (
                            prepared is None
                            and raw_path is not None
                            and self._same_path(item.get("raw_path"), raw_path)
                        )
                    )
                ),
                None,
            )
            if match is not None:
                source = (
                    str(raw_sources[0]["provider"])
                    if raw_sources
                    else str(prepared_data.get("source_mode") or "registered_scan")
                )
                integrity = dict(prepared_data.get("integrity") or {})
                provenance = raw_sources or list(prepared_data.get("raw_sources") or [])
                raw_value = str(raw_path) if raw_path else ""
                prepared_value = str(prepared_path)
                manifest_value = str(prepared["manifest_path"]) if prepared else ""
                status = str(prepared["status"]) if prepared else "raw_only"
                data_changed = any(
                    (
                        match.get("raw_path") != raw_value,
                        match.get("prepared_path") != prepared_value,
                        match.get("manifest_path") != manifest_value,
                        match.get("integrity") != integrity,
                        match.get("provenance") != provenance,
                    )
                )
                saved = self.save_instance(
                    {
                        "id": match["id"],
                        "benchmark_spec_id": spec["id"],
                        "created_at_utc": match.get("created_at_utc"),
                        "creation_source": str(match.get("creation_source") or "scan"),
                        "acquisition": {
                            "mode": "download",
                            "stage": "prepared" if prepared else "raw",
                            "source": source,
                            "source_path": prepared_value if prepared else raw_value,
                        },
                        "raw_path": raw_value,
                        "prepared_path": prepared_value,
                        "manifest_path": manifest_value,
                        "supported_tasks": spec["runnable_tasks"],
                        "status": status,
                        "integrity": integrity,
                        "loader_check": {} if data_changed else match.get("loader_check") or {},
                        "provenance": provenance,
                        "managed": False,
                        "job_id": str(match.get("job_id") or ""),
                    }
                )
                findings.append(
                    {
                        "benchmark_spec_id": spec["id"],
                        "benchmark_key": key,
                        "action": "existing",
                        "instance_id": saved["id"],
                        "status": saved["status"],
                        "refreshed": True,
                        "loader_check_reset": data_changed,
                        "raw_sources": raw_sources,
                        "prepared_path": str(prepared_path) if prepared else "",
                    }
                )
                continue

            identity_path = prepared_path if prepared else raw_path
            assert identity_path is not None
            instance_id = self._scanned_id(spec["id"], identity_path)
            source = (
                str(raw_sources[0]["provider"])
                if raw_sources
                else str(prepared_data.get("source_mode") or "registered_scan")
            )
            integrity = dict(prepared_data.get("integrity") or {})
            saved = self.save_instance(
                {
                    "id": instance_id,
                    "benchmark_spec_id": spec["id"],
                    "acquisition": {
                        "mode": "download",
                        "stage": "prepared" if prepared else "raw",
                        "source": source,
                        "source_path": str(identity_path),
                    },
                    "raw_path": str(raw_path) if raw_path else "",
                    "prepared_path": str(prepared_path),
                    "manifest_path": str(prepared["manifest_path"]) if prepared else "",
                    "supported_tasks": spec["runnable_tasks"],
                    "status": str(prepared["status"]) if prepared else "raw_only",
                    "integrity": integrity,
                    "provenance": raw_sources
                    or list(prepared_data.get("raw_sources") or []),
                    "managed": False,
                    "creation_source": "scan",
                }
            )
            existing.append(saved)
            created.append(saved)
            findings.append(
                {
                    "benchmark_spec_id": spec["id"],
                    "benchmark_key": key,
                    "action": "created",
                    "instance_id": saved["id"],
                    "status": saved["status"],
                    "raw_sources": raw_sources,
                    "prepared_path": str(prepared_path) if prepared else "",
                }
            )

        return {
            "roots": {"raw_root": str(raw_base), "prepared_root": str(prepared_base)},
            "summary": {
                "matched_assets": len(findings),
                "created_instances": len(created),
                "existing_instances": sum(item["action"] == "existing" for item in findings),
                "raw_only": sum(item["status"] == "raw_only" for item in findings),
            },
            "findings": findings,
            "instances": created,
        }

    @staticmethod
    def _same_path(value: Any, candidate: Path) -> bool:
        if not value:
            return False
        return Path(str(value)).expanduser().resolve() == candidate.resolve()

    @staticmethod
    def _scanned_id(spec_id: str, path: Path) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        prefix = spec_id[:48].rstrip("-._")
        return f"{prefix}-scan-{digest}"

    @staticmethod
    def _payload_files(path: Path) -> list[Path]:
        if not path.exists():
            return []
        candidates = path.rglob("*") if path.is_dir() else [path]
        return [
            item
            for item in candidates
            if item.is_file()
            and item.name not in {".lychee_source.json", "manifest.json"}
            and ".cache" not in item.parts
            and not item.name.endswith((".lock", ".part", ".incomplete"))
        ]

    @staticmethod
    def _registered_source_pairs(benchmark_key: str) -> set[tuple[str, str]]:
        from ..benchmarks import BENCHMARKS

        benchmark = BENCHMARKS.get(benchmark_key)
        pairs = {
            (provider, source_id)
            for provider in ("modelscope", "huggingface", "github")
            for source_id in benchmark.provider_ids(provider)
        }
        for item in [*benchmark.fallback_specs(), *benchmark.other_defaults()]:
            provider = str(item.get("provider") or "")
            if provider.startswith("huggingface"):
                provider = "huggingface"
            elif provider.startswith("modelscope"):
                provider = "modelscope"
            source_id = str(item.get("repo_id") or item.get("id") or "")
            if provider and source_id:
                pairs.add((provider, source_id))
        return pairs

    @classmethod
    def _scan_registered_raw_sources(
        cls, benchmark_root: Path, benchmark_key: str
    ) -> list[dict[str, Any]]:
        if not benchmark_root.is_dir():
            return []
        try:
            registered = cls._registered_source_pairs(benchmark_key)
        except KeyError:
            return []
        registered_ids = {source_id for _, source_id in registered}
        rows = []
        for provider_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
            for source_dir in sorted(path for path in provider_dir.iterdir() if path.is_dir()):
                if not cls._payload_files(source_dir):
                    continue
                metadata_path = source_dir / ".lychee_source.json"
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError):
                    continue
                provider = str(metadata.get("provider") or "")
                source_id = str(metadata.get("source_id") or "")
                if not provider or not source_id:
                    continue
                if (provider, source_id) not in registered and source_id not in registered_ids:
                    continue
                rows.append(
                    {
                        "provider": provider,
                        "source_id": source_id,
                        "revision": metadata.get("revision"),
                        "recorded_at_utc": metadata.get("recorded_at_utc"),
                        "raw_path": str(source_dir),
                        "exists": True,
                        "scan_match": "registered_source",
                    }
                )
        return rows

    @classmethod
    def _scan_prepared_asset(
        cls, benchmark_root: Path, benchmark_key: str
    ) -> dict[str, Any] | None:
        manifest_path = benchmark_root / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if manifest.get("benchmark_key") != benchmark_key:
            return None
        records = [dict(item) for item in manifest.get("prepared_files") or []]
        declared = Path(str(manifest.get("prepared_location") or benchmark_root))
        suffix = Path()
        parts = list(declared.parts)
        if benchmark_key in parts:
            suffix = Path(*parts[parts.index(benchmark_key) + 1 :])
        payload_root = benchmark_root / suffix
        missing = []
        changed = []
        for record in records:
            candidate = payload_root / str(record.get("path") or "")
            if not candidate.is_file():
                missing.append(str(record.get("path") or ""))
            elif record.get("size_bytes") is not None and candidate.stat().st_size != int(
                record["size_bytes"]
            ):
                changed.append(str(record.get("path") or ""))
        declared_ready = manifest.get("integrity", {}).get("status") == "ready"
        status = (
            "ready" if declared_ready and records and not missing and not changed else "invalid"
        )
        return {
            "manifest_path": str(manifest_path),
            "payload_root": str(payload_root),
            "status": status,
            "source_mode": manifest.get("source_mode"),
            "raw_sources": list(manifest.get("raw_sources") or []),
            "integrity": {
                "validation": "prepared_manifest_scan",
                "file_count": len(records),
                "size_bytes": sum(int(item.get("size_bytes") or 0) for item in records),
                "record_count": sum(
                    int(item["record_count"])
                    for item in records
                    if item.get("record_count") is not None
                ),
                "missing_files": missing,
                "changed_files": changed,
            },
        }

    def _portable_path(self, value: str) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)

    def _runtime_path(self, value: str) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else self.repo_root / path)

    def _portable_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        portable = dict(value)
        portable["raw_path"] = self._portable_path(portable.get("raw_path", ""))
        portable["prepared_path"] = self._portable_path(portable["prepared_path"])
        portable["manifest_path"] = self._portable_path(portable.get("manifest_path", ""))
        provenance = []
        for source in portable.get("provenance") or []:
            item = dict(source)
            for key in ("raw_path", "path"):
                if item.get(key):
                    item[key] = self._portable_path(str(item[key]))
            provenance.append(item)
        portable["provenance"] = provenance
        return portable

    def _materialize_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        materialized = dict(value)
        materialized["raw_path"] = self._runtime_path(materialized.get("raw_path", ""))
        materialized["portable_prepared_path"] = materialized["prepared_path"]
        materialized["prepared_path"] = self._runtime_path(materialized["prepared_path"])
        materialized["manifest_path"] = self._runtime_path(materialized.get("manifest_path", ""))
        provenance = []
        for source in materialized.get("provenance") or []:
            item = dict(source)
            for key in ("raw_path", "path"):
                if item.get(key):
                    item[key] = self._runtime_path(str(item[key]))
            provenance.append(item)
        materialized["provenance"] = provenance
        return materialized

    @staticmethod
    def _inspect_instance(value: dict[str, Any]) -> dict[str, Any]:
        from ..benchmarks.manifest import verify_prepared_manifest

        inspected = dict(value)
        prepared_path = Path(inspected["prepared_path"])
        manifest_path = Path(str(inspected.get("manifest_path") or ""))
        files = (
            [item for item in prepared_path.rglob("*") if item.is_file()]
            if prepared_path.is_dir()
            else [prepared_path]
            if prepared_path.is_file()
            else []
        )
        payload_files = [
            item
            for item in files
            if item.name not in {"manifest.json", ".lychee_source.json"}
            and not item.name.endswith((".lock", ".part", ".incomplete"))
        ]
        if manifest_path.is_file():
            try:
                verified = verify_prepared_manifest(manifest_path, verify_hashes=False)
                inspected["status"] = str(verified["status"])
                if inspected["status"] == "invalid" and not inspected.get("managed", True):
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    relocated = BenchmarkRegistry._scan_prepared_asset(
                        prepared_path,
                        str(manifest.get("benchmark_key") or ""),
                    )
                    if relocated:
                        inspected["status"] = str(relocated["status"])
                        inspected["integrity"] = dict(relocated["integrity"])
            except (OSError, KeyError, TypeError, ValueError):
                inspected["status"] = "invalid"
        elif payload_files:
            inspected["status"] = "ready"
            inspected["integrity"] = {
                **dict(inspected.get("integrity") or {}),
                "validation": "nonempty_local_prepared_path",
                "file_count": len(payload_files),
                "size_bytes": sum(item.stat().st_size for item in payload_files),
            }
        elif (
            inspected.get("status") == "raw_only"
            and inspected.get("raw_path")
            and BenchmarkRegistry._payload_files(Path(str(inspected["raw_path"])))
        ):
            inspected["status"] = "raw_only"
        elif inspected.get("status") != "preparing":
            inspected["status"] = "missing"
        return inspected

    def resolve_task(
        self, task: str, *, instance_id: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        spec = next((item for item in self.specs() if task in item["runnable_tasks"]), None)
        if spec is None:
            raise ValueError(f"no BenchmarkSpec owns runnable task {task!r}")
        instances = [item for item in self.instances() if item["benchmark_spec_id"] == spec["id"]]
        instance = next((item for item in instances if item["id"] == instance_id), None)
        if instance_id and instance is None:
            raise ValueError(
                f"BenchmarkInstance {instance_id!r} does not belong to BenchmarkSpec {spec['id']!r}"
            )
        if instance is None:
            instance = next((item for item in instances if item.get("available")), None)
        return spec, instance
