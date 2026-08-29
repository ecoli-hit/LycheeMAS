"""File-backed ModelSpec and ModelInstance registries."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ..contracts.lifecycle import (
    bind_instance_to_spec,
    lifecycle_error,
    normalize_instance_lifecycle,
    utc_now,
)
from ..infrastructure.json_store import (
    atomic_write_json as _write_json,
)
from ..infrastructure.json_store import (
    load_normalized_documents,
)
from ..infrastructure.json_store import (
    validate_identifier as _validate_id,
)


def _utc_now() -> str:
    return utc_now()


def normalize_model_spec(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ModelSpec must be a JSON object")
    spec_id = _validate_id(str(value.get("id") or ""), "ModelSpec id")
    name = str(value.get("name") or "").strip()
    if not name:
        raise ValueError("ModelSpec requires name")
    sources = {
        str(provider): [dict(item) for item in items if isinstance(item, dict)]
        for provider, items in dict(value.get("sources") or {}).items()
    }
    raw_capabilities = value.get("capabilities") or {}
    if isinstance(raw_capabilities, list):
        capabilities = {str(item): True for item in raw_capabilities}
    elif isinstance(raw_capabilities, dict):
        capabilities = dict(raw_capabilities)
    else:
        raise ValueError("ModelSpec capabilities must be an object or a string list")
    model_info = value.get("model_info") or {}
    if not isinstance(model_info, dict):
        raise ValueError("ModelSpec model_info must be an object")
    provider_model_ids = value.get("provider_model_ids") or {}
    if not isinstance(provider_model_ids, dict) or any(
        not str(provider).strip() or not str(model_id).strip()
        for provider, model_id in provider_model_ids.items()
    ):
        raise ValueError("ModelSpec provider_model_ids must map providers to model IDs")
    thinking_protocol = value.get("thinking_protocol")
    if thinking_protocol is not None:
        thinking_protocol = str(thinking_protocol).strip() or None
    supported_backends = [str(item) for item in value.get("supported_backends") or []]
    if any(item not in {"hf", "vllm", "api"} for item in supported_backends):
        raise ValueError("ModelSpec supported_backends may contain hf, vllm, or api")
    return {
        "schema_version": 2,
        "id": spec_id,
        "name": name,
        "organization": str(value.get("organization") or "Other"),
        "family": str(value.get("family") or ""),
        "parameter_size": str(value.get("parameter_size") or ""),
        "license": str(value.get("license") or ""),
        "gated": bool(value.get("gated", False)),
        "capabilities": capabilities,
        "model_info": dict(model_info),
        "thinking_protocol": thinking_protocol,
        "supported_backends": list(dict.fromkeys(supported_backends)),
        "provider_model_ids": {
            str(provider): str(model_id)
            for provider, model_id in provider_model_ids.items()
        },
        "sources": sources,
        "notes": str(value.get("notes") or ""),
    }


def normalize_model_instance(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ModelInstance must be a JSON object")
    instance_id = _validate_id(str(value.get("id") or ""), "ModelInstance id")
    _validate_id(str(value.get("model_spec_id") or ""), "ModelSpec id")
    path = str(value.get("path") or "").strip()
    if not path:
        raise ValueError("ModelInstance requires path")
    acquisition = dict(value.get("acquisition") or {})
    mode = str(acquisition.get("mode") or "local")
    if mode not in {"local", "download"}:
        raise ValueError("ModelInstance acquisition mode must be local or download")
    lifecycle = normalize_instance_lifecycle(value, prefix="model")
    return {
        "schema_version": 1,
        "id": instance_id,
        **lifecycle,
        "acquisition": {
            "mode": mode,
            "source": str(acquisition.get("source") or "local"),
            "source_path": str(acquisition.get("source_path") or ""),
            "revision": str(acquisition.get("revision") or ""),
        },
        "path": path,
        "managed": bool(value.get("managed", mode == "download")),
        "status": str(value.get("status") or "unknown"),
        "integrity": dict(value.get("integrity") or {}),
        "job_id": str(value.get("job_id") or ""),
        "updated_at_utc": value.get("updated_at_utc"),
    }


class ModelRegistry:
    """Keep supported model products separate from concrete local model paths."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/models"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        return self._all(self.spec_root, normalize_model_spec)

    def instances(self) -> list[dict[str, Any]]:
        specs = {item["id"]: item for item in self.specs()}
        rows = []
        for row in self._all(self.instance_root, normalize_model_instance):
            inspected = self._inspect_instance(self._materialize_instance(row))
            inspected["model_spec"] = specs.get(inspected["model_spec_id"])
            error = lifecycle_error(
                inspected,
                spec=inspected["model_spec"],
                prefix="model",
            )
            inspected["configuration_status"] = "invalid" if error else "current"
            inspected["validation_error"] = error
            inspected["observed_status"] = "invalid" if error else inspected["status"]
            inspected["available"] = (
                not error and inspected["status"] == "ready"
            )
            rows.append(inspected)
        return rows

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        match = next((item for item in self.specs() if item["id"] == spec_id), None)
        if match is None:
            raise ValueError(f"unknown registered ModelSpec {spec_id!r}")
        return match

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        match = next((item for item in self.instances() if item["id"] == instance_id), None)
        if match is None:
            raise ValueError(f"unknown ModelInstance {instance_id!r}")
        if match.get("configuration_status") == "invalid":
            raise ValueError(
                f"ModelInstance {instance_id!r} is invalid: {match.get('validation_error')}"
            )
        return match

    def sources(self, spec_id: str, provider: str = "auto") -> list[dict[str, Any]]:
        spec = self.get_spec(spec_id)
        sources = dict(spec.get("sources") or {})
        if provider == "auto":
            values = [
                {"provider": source_provider, **item}
                for source_provider, items in sources.items()
                for item in items
            ]
        else:
            values = [{"provider": provider, **item} for item in sources.get(provider, [])]
        return sorted(values, key=lambda item: (int(item.get("priority", 100)), item["provider"]))

    def save_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        spec = self.get_spec(str(value.get("model_spec_id") or ""))
        payload = bind_instance_to_spec(
            value,
            spec=spec,
            prefix="model",
            creation_source=str(value.get("creation_source") or "studio"),
            created_at_utc=value.get("created_at_utc"),
        )
        normalized = self._portable_instance(normalize_model_instance(payload))
        normalized["updated_at_utc"] = _utc_now()
        path = self.instance_root / f"{normalized['id']}.json"
        _write_json(path, normalized)
        return self.get_instance(normalized["id"])

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        _validate_id(instance_id, "ModelInstance id")
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": instance_id, "deleted": True, "path": str(path)}

    def scan_root(self, models_root: str | os.PathLike) -> dict[str, Any]:
        """Register complete model snapshots that uniquely match a ModelSpec."""

        root = Path(models_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Model root does not exist or is not a directory: {root}")
        specs = self.specs()
        existing = self.instances()
        findings: list[dict[str, Any]] = []
        created: list[dict[str, Any]] = []
        unmatched: list[dict[str, Any]] = []
        for candidate in self._model_directories(root):
            inspected = self._inspect_instance(
                {
                    "path": str(candidate),
                    "status": "unknown",
                    "integrity": {},
                }
            )
            if inspected["status"] != "ready":
                continue
            match = self._match_model_spec(candidate, specs)
            if match.get("status") != "matched":
                unmatched.append(
                    {
                        "path": str(candidate),
                        "reason": match.get("status"),
                        "candidate_spec_ids": match.get("candidate_spec_ids") or [],
                    }
                )
                continue
            spec = match["spec"]
            current = next(
                (
                    item
                    for item in existing
                    if item["model_spec_id"] == spec["id"]
                    and Path(item["path"]).expanduser().resolve() == candidate.resolve()
                ),
                None,
            )
            if current is not None:
                findings.append(
                    {
                        "model_spec_id": spec["id"],
                        "action": "existing",
                        "instance_id": current["id"],
                        "path": str(candidate),
                        "match_basis": match["match_basis"],
                    }
                )
                continue
            instance_id = self._scanned_id(spec["id"], candidate)
            manifest = match.get("manifest") or {}
            provider = str(manifest.get("provider") or match.get("provider") or "registered_scan")
            source_id = str(manifest.get("source_id") or match.get("source_id") or "")
            saved = self.save_instance(
                {
                    "id": instance_id,
                    "model_spec_id": spec["id"],
                    "acquisition": {
                        "mode": "download" if source_id else "local",
                        "source": provider,
                        "source_path": source_id or str(candidate),
                        "revision": str(manifest.get("revision") or ""),
                    },
                    "path": str(candidate),
                    "managed": False,
                    "status": "ready",
                    "integrity": inspected["integrity"],
                    "creation_source": "scan",
                }
            )
            existing.append(saved)
            created.append(saved)
            findings.append(
                {
                    "model_spec_id": spec["id"],
                    "action": "created",
                    "instance_id": saved["id"],
                    "path": str(candidate),
                    "match_basis": match["match_basis"],
                }
            )
        return {
            "root": str(root),
            "summary": {
                "matched_assets": len(findings),
                "created_instances": len(created),
                "existing_instances": sum(item["action"] == "existing" for item in findings),
                "unmatched_complete_models": len(unmatched),
            },
            "findings": findings,
            "unmatched": unmatched,
            "instances": created,
        }

    @staticmethod
    def _scanned_id(spec_id: str, path: Path) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        prefix = spec_id[:48].rstrip("-._")
        return f"{prefix}-scan-{digest}"

    @staticmethod
    def _model_directories(root: Path) -> list[Path]:
        candidates = {root} if (root / "config.json").is_file() else set()
        for config in root.rglob("config.json"):
            relative = config.relative_to(root)
            ignored = {".cache", ".downloads", "snapshots", "blobs"}
            if any(part in ignored for part in relative.parts):
                continue
            candidates.add(config.parent)
        return sorted(candidates)

    @staticmethod
    def _normalized_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @classmethod
    def _match_model_spec(
        cls, candidate: Path, specs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        manifest_path = candidate / "model_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            manifest = {}
        by_id = {item["id"]: item for item in specs}
        manifest_spec = by_id.get(str(manifest.get("model_spec_id") or ""))
        if manifest_spec is not None:
            source_id = str(manifest.get("source_id") or "")
            provider = str(manifest.get("provider") or "")
            registered_ids = {
                (source_provider, str(source.get("id") or ""))
                for source_provider, sources in (manifest_spec.get("sources") or {}).items()
                for source in sources
            }
            if registered_ids and (provider, source_id) in registered_ids:
                return {
                    "status": "matched",
                    "spec": manifest_spec,
                    "manifest": manifest,
                    "match_basis": "model_manifest",
                    "provider": provider,
                    "source_id": source_id,
                }

        names = {cls._normalized_name(candidate.name)}
        try:
            config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            config = {}
        if config.get("_name_or_path"):
            names.add(cls._normalized_name(str(config["_name_or_path"]).split("/")[-1]))

        matches = []
        for spec in specs:
            registered_names = {cls._normalized_name(spec["name"])}
            sources = [
                (provider, str(source.get("id") or ""))
                for provider, values in (spec.get("sources") or {}).items()
                for source in values
            ]
            if not sources:
                continue
            registered_names.update(
                cls._normalized_name(source_id.split("/")[-1])
                for _, source_id in sources
                if source_id
            )
            if names & registered_names:
                matches.append((spec, sources))
        if len(matches) != 1:
            return {
                "status": "ambiguous" if matches else "unregistered",
                "candidate_spec_ids": [item[0]["id"] for item in matches],
            }
        spec, sources = matches[0]
        provider, source_id = sources[0] if sources else ("", "")
        return {
            "status": "matched",
            "spec": spec,
            "manifest": manifest,
            "match_basis": "registered_name",
            "provider": provider,
            "source_id": source_id,
        }

    @staticmethod
    def _all(root: Path, normalizer) -> list[dict[str, Any]]:
        return load_normalized_documents(root, normalizer)

    def _portable_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)

    def _runtime_path(self, value: str) -> str:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else self.repo_root / path)

    def _portable_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        portable = dict(value)
        portable["path"] = self._portable_path(portable["path"])
        return portable

    def _materialize_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        materialized = dict(value)
        materialized["portable_path"] = materialized["path"]
        materialized["path"] = self._runtime_path(materialized["path"])
        return materialized

    @staticmethod
    def _inspect_instance(value: dict[str, Any]) -> dict[str, Any]:
        inspected = dict(value)
        root = Path(inspected["path"])
        config = root / "config.json"
        weights = [
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in root.rglob(pattern)
            if ".cache" not in path.parts
        ] if root.is_dir() else []
        tokenizer = [
            root / name
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
            if (root / name).is_file()
        ]
        incomplete = (
            list(root.rglob("*.incomplete"))
            + list(root.rglob("*.part"))
            + list(root.rglob("*.lock"))
            if root.is_dir()
            else []
        )
        ready = config.is_file() and bool(weights) and bool(tokenizer) and not incomplete
        if ready:
            inspected["status"] = "ready"
        elif inspected.get("status") != "preparing":
            inspected["status"] = "incomplete" if root.exists() else "missing"
        inspected["integrity"] = {
            **dict(inspected.get("integrity") or {}),
            "config_present": config.is_file(),
            "weight_files": len(weights),
            "tokenizer_files": len(tokenizer),
            "incomplete_files": len(incomplete),
            "size_bytes": sum(path.stat().st_size for path in weights),
        }
        return inspected
