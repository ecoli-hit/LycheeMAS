"""File-backed TeamInstance registry and availability checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...runtime.adapters.frameworks.bindings import build_framework_binding_report
from ...runtime.coordination.compiler import compile_coordination
from ..contracts.lifecycle import (
    bind_instance_to_spec,
    lifecycle_error,
    normalize_instance_lifecycle,
    utc_now,
)
from ..contracts.specs import normalize_invocation_overrides
from ..infrastructure.json_store import atomic_write_json
from .contracts import member_nodes, model_resource_requirements
from .registry import TeamSpecRegistry

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_AVAILABLE_DEPLOYMENT_STATUSES = {"running", "ready_on_run"}
RUNTIME_FRAMEWORKS = {"autogen", "langgraph", "crewai"}


def _utc_now() -> str:
    return utc_now()


def normalize_team_instance_document(
    value: dict[str, Any], team_spec: dict[str, Any]
) -> dict[str, Any]:
    """Validate concrete bindings against one canonical TeamSpec."""

    if not isinstance(value, dict):
        raise ValueError("TeamInstance must be a JSON object")
    if int(value.get("schema_version") or 0) != 5:
        raise ValueError("TeamInstance requires schema_version 5")
    instance_id = str(value.get("id") or "").strip()
    if not _SAFE_ID.fullmatch(instance_id):
        raise ValueError("TeamInstance id may contain only letters, numbers, '.', '_' and '-'")
    team_spec_id = str(value.get("team_spec_id") or "").strip()
    if team_spec_id != str(team_spec["id"]):
        raise ValueError(
            f"TeamInstance references TeamSpec {team_spec_id!r}, expected {team_spec['id']!r}"
        )
    runtime_framework = str(value.get("runtime_framework") or "").strip().lower()
    if runtime_framework not in RUNTIME_FRAMEWORKS:
        raise ValueError(
            "TeamInstance runtime_framework must be one of "
            + ", ".join(sorted(RUNTIME_FRAMEWORKS))
        )
    framework_options = value.get("framework_options") or {}
    if not isinstance(framework_options, dict):
        raise ValueError("TeamInstance framework_options must be an object")

    requirements = {
        (str(item["node_id"]), str(item["requirement"])): item
        for item in model_resource_requirements(team_spec)
    }
    raw_bindings = value.get("resource_bindings") or []
    if not isinstance(raw_bindings, list):
        raise ValueError("TeamInstance resource_bindings must be a list")
    normalized_bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_bindings:
        if not isinstance(raw, dict):
            raise ValueError("each resource binding must be an object")
        node_id = str(raw.get("node_id") or "").strip()
        requirement = str(raw.get("requirement") or "").strip()
        resource_type = str(raw.get("resource_instance_type") or "").strip()
        resource_id = str(raw.get("resource_instance_id") or "").strip()
        key = (node_id, requirement)
        if key not in requirements:
            raise ValueError(
                f"TeamInstance binding references unknown Node requirement "
                f"{node_id!r}/{requirement!r}"
            )
        if key in seen:
            raise ValueError(
                f"duplicate TeamInstance binding for Node requirement {node_id!r}/{requirement!r}"
            )
        if requirement == "model_inference" and resource_type != "DeploymentInstance":
            raise ValueError(
                f"Node {node_id!r} model_inference requires a DeploymentInstance"
            )
        if not _SAFE_ID.fullmatch(resource_id):
            raise ValueError(
                f"Node requirement {node_id!r}/{requirement!r} requires a resource Instance id"
            )
        seen.add(key)
        overrides = normalize_invocation_overrides(
            raw.get("generation_overrides"), label=f"Node {node_id} model inference"
        )
        normalized_bindings.append(
            {
                "node_id": node_id,
                "requirement": requirement,
                "resource_instance_type": resource_type,
                "resource_instance_id": resource_id,
                **({"generation_overrides": overrides} if overrides else {}),
            }
        )
    missing = sorted(set(requirements) - seen)
    if missing:
        raise ValueError(
            "TeamInstance requires one resource binding for every external Node requirement; "
            + "missing: "
            + ", ".join(f"{node_id}/{requirement}" for node_id, requirement in missing)
        )
    lifecycle = normalize_instance_lifecycle(value, prefix="team")
    framework_binding_report = build_framework_binding_report(
        runtime_framework, compile_coordination(team_spec), team_spec
    )
    if not framework_binding_report["supported"]:
        raise ValueError(str(framework_binding_report["reason"]))
    supported_execution_kinds = set(
        framework_binding_report["adapter_capabilities"].get("execution_kinds") or []
    )
    unsupported_execution_kinds = sorted(
        {
            str(node.get("kind") or "")
            for node in member_nodes(team_spec)
        }
        - supported_execution_kinds
    )
    if unsupported_execution_kinds:
        raise ValueError(
            f"{runtime_framework} cannot execute member Node kinds: "
            + ", ".join(unsupported_execution_kinds)
        )
    return {
        "schema_version": 5,
        "id": instance_id,
        **lifecycle,
        "runtime_framework": runtime_framework,
        "framework_options": dict(framework_options),
        "resource_bindings": normalized_bindings,
        "framework_binding_report": framework_binding_report,
    }


def team_instance_availability(
    value: dict[str, Any],
    team_spec: dict[str, Any],
    deployment_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve bindings and verify deployment health and declared capabilities."""

    normalized = normalize_team_instance_document(value, team_spec)
    by_id = {str(item.get("id")): item for item in deployment_instances}
    requirements = {
        (str(item["node_id"]), str(item["requirement"])): item
        for item in model_resource_requirements(team_spec)
    }
    referenced = {
        str(item["resource_instance_id"])
        for item in normalized.get("resource_bindings") or []
        if item["resource_instance_type"] == "DeploymentInstance"
    }
    missing = sorted(instance_id for instance_id in referenced if instance_id not in by_id)
    unavailable = sorted(
        instance_id
        for instance_id in referenced
        if instance_id in by_id
        and str(by_id[instance_id].get("observed_status") or by_id[instance_id].get("status"))
        not in _AVAILABLE_DEPLOYMENT_STATUSES
    )
    capability_errors: list[str] = []
    for binding in normalized.get("resource_bindings") or []:
        if binding["resource_instance_type"] != "DeploymentInstance":
            continue
        deployment = by_id.get(str(binding["resource_instance_id"]))
        if deployment is None:
            continue
        capabilities = dict(deployment.get("capabilities") or {})
        requirement = requirements[(binding["node_id"], binding["requirement"])]
        for capability in requirement.get("required_capabilities") or []:
            if capabilities.get(capability) is not True:
                capability_errors.append(
                    f"{binding['node_id']} requires {capability} from "
                    f"{binding['resource_instance_id']}"
                )
    available = not missing and not unavailable and not capability_errors
    return {
        **normalized,
        "team_spec": team_spec,
        "observed_status": "ready" if available else "unavailable",
        "available": available,
        "missing_deployment_instance_ids": missing,
        "unavailable_deployment_instance_ids": unavailable,
        "missing_resource_instance_ids": missing,
        "unavailable_resource_instance_ids": unavailable,
        "capability_errors": capability_errors,
    }


class TeamInstanceRegistry:
    """Store TeamSpec references and concrete Node resource bindings."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.root = self.repo_root / "configs/eval_studio/teams/instances"
        self.team_specs = TeamSpecRegistry(self.repo_root)

    def all(self, deployment_instances: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        specs = {item["id"]: item for item in self.team_specs.all()}
        rows = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                spec = specs.get(str(raw.get("team_spec_id") or ""))
                if spec is None:
                    rows.append(
                        {
                            **raw,
                            "team_spec": None,
                            "registry_path": str(path),
                            "observed_status": "invalid",
                            "available": False,
                            "configuration_status": "invalid",
                            "validation_error": "referenced TeamSpec is missing",
                        }
                    )
                    continue
                normalized = normalize_team_instance_document(raw, spec)
            except (OSError, TypeError, ValueError):
                continue
            error = lifecycle_error(normalized, spec=spec, prefix="team")
            row = (
                team_instance_availability(normalized, spec, deployment_instances)
                if deployment_instances is not None
                else {**normalized, "team_spec": spec}
            )
            rows.append(
                {
                    **row,
                    "registry_path": str(path),
                    "configuration_status": "invalid" if error else "current",
                    "validation_error": error,
                    **(
                        {"observed_status": "invalid", "available": False}
                        if error
                        else {}
                    ),
                }
            )
        return rows

    def save(self, value: dict[str, Any]) -> dict[str, Any]:
        team_spec = self.team_specs.get(str(value.get("team_spec_id") or ""))
        payload = bind_instance_to_spec(
            value,
            spec=team_spec,
            prefix="team",
            creation_source=str(value.get("creation_source") or "studio"),
            created_at_utc=value.get("created_at_utc"),
        )
        normalized = normalize_team_instance_document(payload, team_spec)
        path = self.root / f"{normalized['id']}.json"
        if path.exists():
            raise ValueError(
                f"TeamInstance {normalized['id']!r} already exists; use a new ID"
            )
        atomic_write_json(path, normalized)
        return {**normalized, "team_spec": team_spec, "registry_path": str(path)}

    def get(
        self,
        instance_id: str,
        deployment_instances: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(instance_id):
            raise ValueError("invalid TeamInstance id")
        path = self.root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        team_spec = self.team_specs.get(str(raw.get("team_spec_id") or ""))
        normalized = normalize_team_instance_document(raw, team_spec)
        error = lifecycle_error(normalized, spec=team_spec, prefix="team")
        if error:
            raise ValueError(f"TeamInstance {instance_id!r} is invalid: {error}")
        value = (
            team_instance_availability(normalized, team_spec, deployment_instances)
            if deployment_instances is not None
            else {**normalized, "team_spec": team_spec}
        )
        return {**value, "registry_path": str(path)}

    def delete(self, instance_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(instance_id):
            raise ValueError("invalid TeamInstance id")
        path = self.root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": instance_id, "deleted": True, "path": str(path)}
