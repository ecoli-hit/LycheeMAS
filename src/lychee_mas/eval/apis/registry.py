"""File-backed APISpec and APIInstance registries."""

from __future__ import annotations

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

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SESSION_KEY_PREFIX = "LYCHEE_STUDIO_RESOURCE_KEY_"


def _utc_now() -> str:
    return utc_now()


def normalize_api_spec(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("APISpec must be a JSON object")
    obsolete = {
        "model_id",
        "model_id_editable",
        "model_info",
        "capabilities",
        "thinking_protocol",
    } & set(value)
    if obsolete:
        raise ValueError(
            "APISpec cannot define model semantics; move fields to ModelSpec: "
            + ", ".join(sorted(obsolete))
        )
    spec_id = _validate_id(str(value.get("id") or ""), "APISpec id")
    deployment_kind = str(value.get("deployment_kind") or "api")
    if deployment_kind not in {"api", "vllm"}:
        raise ValueError("APISpec deployment_kind must be api or vllm")
    provider_key = str(value.get("provider_key") or "").strip()
    if not provider_key:
        raise ValueError("APISpec requires provider_key")
    allowed_model_spec_ids = [
        _validate_id(str(item), "ModelSpec id")
        for item in value.get("allowed_model_spec_ids") or []
    ]
    if not allowed_model_spec_ids:
        raise ValueError("APISpec requires allowed_model_spec_ids")
    default_base_url = str(value.get("default_base_url") or "").strip().rstrip("/")
    if not default_base_url:
        raise ValueError("APISpec requires default_base_url")
    auth_modes = [str(item) for item in value.get("auth_modes") or ["env"]]
    if not auth_modes or any(item not in {"env", "none"} for item in auth_modes):
        raise ValueError("APISpec auth_modes must contain env or none")
    protocol_capabilities = value.get("protocol_capabilities") or {}
    if not isinstance(protocol_capabilities, dict):
        raise ValueError("APISpec protocol_capabilities must be an object")
    return {
        "schema_version": 2,
        "id": spec_id,
        "name": str(value.get("name") or spec_id),
        "provider": str(value.get("provider") or "Other"),
        "provider_key": provider_key,
        "organization": str(value.get("organization") or "Other"),
        "protocol": str(value.get("protocol") or "openai_compatible"),
        "deployment_kind": deployment_kind,
        "allowed_model_spec_ids": list(dict.fromkeys(allowed_model_spec_ids)),
        "default_base_url": default_base_url,
        "base_url_editable": bool(value.get("base_url_editable", False)),
        "auth_modes": auth_modes,
        "default_api_key_env": str(value.get("default_api_key_env") or ""),
        "trust_env": bool(value.get("trust_env", True)),
        "shared": bool(value.get("shared", False)),
        "request_limits": dict(value.get("request_limits") or {}),
        "protocol_capabilities": dict(protocol_capabilities),
        "notes": str(value.get("notes") or ""),
    }


def normalize_api_instance(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("APIInstance must be a JSON object")
    obsolete = {"model_id", "model_info", "capabilities", "thinking_protocol"} & set(value)
    if obsolete:
        raise ValueError(
            "APIInstance cannot copy model semantics: " + ", ".join(sorted(obsolete))
        )
    instance_id = _validate_id(str(value.get("id") or ""), "APIInstance id")
    _validate_id(str(value.get("api_spec_id") or ""), "APISpec id")
    base_url = str(value.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("APIInstance requires base_url")
    auth_mode = str(value.get("auth_mode") or "env")
    if auth_mode not in {"env", "none"}:
        raise ValueError("APIInstance auth_mode must be env or none")
    api_key_env = str(value.get("api_key_env") or "")
    if auth_mode == "env" and not _ENV_NAME.fullmatch(api_key_env):
        raise ValueError("APIInstance api_key_env must be an environment variable name")
    lifecycle = normalize_instance_lifecycle(value, prefix="api")
    return {
        "schema_version": 2,
        "id": instance_id,
        **lifecycle,
        "base_url": base_url,
        "auth_mode": auth_mode,
        **({"api_key_env": api_key_env} if auth_mode == "env" else {}),
        "trust_env": bool(value.get("trust_env", True)),
        "shared": bool(value.get("shared", False)),
        "request_limits": dict(value.get("request_limits") or {}),
        "observed_protocol_capabilities": dict(
            value.get("observed_protocol_capabilities") or {}
        ),
        "credential_source": str(value.get("credential_source") or "environment"),
        "notes": str(value.get("notes") or ""),
        "updated_at_utc": value.get("updated_at_utc"),
    }


class APIRegistry:
    """Keep supported API products separate from concrete endpoint credentials."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/apis"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        return self._all(self.spec_root, normalize_api_spec)

    def instances(self) -> list[dict[str, Any]]:
        specs = {item["id"]: item for item in self.specs()}
        rows = self._all(self.instance_root, normalize_api_instance)
        for row in rows:
            spec = specs.get(row["api_spec_id"])
            error = lifecycle_error(row, spec=spec, prefix="api")
            credential_present = row["auth_mode"] == "none" or bool(
                os.environ.get(str(row.get("api_key_env") or ""))
            )
            configuration_status = (
                "invalid"
                if error
                else "configured"
                if credential_present
                else "auth_required"
            )
            row.update(
                api_spec=spec,
                credential_present=credential_present,
                configuration_status=configuration_status,
                configured=not error and credential_present,
                observed_status="invalid" if error else "unknown",
                validation_error=error,
            )
        return rows

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        match = next((item for item in self.specs() if item["id"] == spec_id), None)
        if match is None:
            raise ValueError(f"unknown registered APISpec {spec_id!r}")
        return match

    def get(self, instance_id: str) -> dict[str, Any]:
        match = next((item for item in self.instances() if item["id"] == instance_id), None)
        if match is None:
            raise ValueError(f"unknown registered APIInstance {instance_id!r}")
        if match.get("configuration_status") == "invalid":
            raise ValueError(
                f"APIInstance {instance_id!r} is invalid: {match.get('validation_error')}"
            )
        return match

    def save_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = dict(value)
        manual_key = str(payload.pop("api_key", "") or "")
        spec = self.get_spec(str(payload.get("api_spec_id") or ""))
        base_url = str(payload.get("base_url") or spec["default_base_url"])
        if not spec["base_url_editable"] and base_url.rstrip("/") != spec["default_base_url"]:
            raise ValueError("APISpec does not allow base_url overrides")
        auth_mode = str(payload.get("auth_mode") or spec["auth_modes"][0])
        if auth_mode not in spec["auth_modes"]:
            raise ValueError(f"APISpec does not support auth_mode {auth_mode!r}")
        payload.update(
            base_url=base_url,
            auth_mode=auth_mode,
            trust_env=bool(payload.get("trust_env", spec["trust_env"])),
            shared=bool(payload.get("shared", spec["shared"])),
            request_limits=dict(payload.get("request_limits") or spec["request_limits"]),
            observed_protocol_capabilities=dict(
                payload.get("observed_protocol_capabilities") or {}
            ),
        )
        if auth_mode == "env":
            payload["api_key_env"] = str(
                payload.get("api_key_env") or spec["default_api_key_env"]
            )
            if manual_key and not payload["api_key_env"]:
                resource_key = (
                    str(payload.get("id") or "")
                    .upper()
                    .replace("-", "_")
                    .replace(".", "_")
                )
                payload["api_key_env"] = f"{_SESSION_KEY_PREFIX}{resource_key}"
        payload = bind_instance_to_spec(
            payload,
            spec=spec,
            prefix="api",
            creation_source=str(value.get("creation_source") or "studio"),
            created_at_utc=value.get("created_at_utc"),
        )
        normalized = normalize_api_instance(payload)
        if manual_key:
            key_name = normalized["api_key_env"]
            os.environ[key_name] = manual_key
            normalized["credential_source"] = "manual_session"
        else:
            normalized["credential_source"] = str(
                value.get("credential_source") or "environment"
            )
        normalized["updated_at_utc"] = _utc_now()
        path = self.instance_root / f"{normalized['id']}.json"
        _write_json(path, normalized)
        return self.get(normalized["id"])

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        _validate_id(instance_id, "APIInstance id")
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        key_name = str(value.get("api_key_env") or "")
        if value.get("credential_source") == "manual_session" and key_name.startswith(
            _SESSION_KEY_PREFIX
        ):
            os.environ.pop(key_name, None)
        path.unlink()
        return {"id": instance_id, "deleted": True, "path": str(path)}

    def deployment_fields(
        self,
        instance_id: str,
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        instance = self.get(instance_id)
        if not instance.get("configured"):
            raise ValueError(
                f"APIInstance {instance_id!r} is not configured; configure its credential in "
                "Resource Center before instantiating a DeploymentSpec"
            )
        spec = instance.get("api_spec") or {}
        model_spec_id = str(model_spec.get("id") or "")
        if model_spec_id not in set(spec.get("allowed_model_spec_ids") or []):
            raise ValueError(
                f"APISpec {spec.get('id')!r} does not allow ModelSpec {model_spec_id!r}"
            )
        provider_key = str(spec.get("provider_key") or "")
        model_id = str(
            (model_spec.get("provider_model_ids") or {}).get(provider_key) or ""
        ).strip()
        if not model_id:
            raise ValueError(
                f"ModelSpec {model_spec_id!r} has no provider model ID for {provider_key!r}"
            )
        capabilities = dict(model_spec.get("capabilities") or {})
        capabilities.update(dict(spec.get("protocol_capabilities") or {}))
        capabilities.update(dict(instance.get("observed_protocol_capabilities") or {}))
        return {
            "kind": str(spec.get("deployment_kind") or "api"),
            "model_id": model_id,
            "base_url": instance["base_url"],
            "auth_mode": instance["auth_mode"],
            "api_key_env": instance.get("api_key_env"),
            "trust_env": bool(instance.get("trust_env", True)),
            "managed": False,
            "shared": bool(instance.get("shared", False)),
            "request_limits": dict(instance.get("request_limits") or {}),
            "model_info": dict(model_spec.get("model_info") or {}),
            "capabilities": capabilities,
            "thinking_protocol": model_spec.get("thinking_protocol"),
        }

    @staticmethod
    def _all(root: Path, normalizer) -> list[dict[str, Any]]:
        return load_normalized_documents(root, normalizer)
