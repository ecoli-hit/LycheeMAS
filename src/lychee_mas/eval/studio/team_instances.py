"""File-backed TeamInstance registry and availability checks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .lifecycle import (
    bind_instance_to_spec,
    lifecycle_error,
    normalize_instance_lifecycle,
    utc_now,
)
from .specs import normalize_invocation_overrides
from .teams import TeamSpecRegistry

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_AVAILABLE_DEPLOYMENT_STATUSES = {"running", "ready_on_run"}


def _utc_now() -> str:
    return utc_now()


def normalize_team_instance_document(
    value: dict[str, Any], team_spec: dict[str, Any]
) -> dict[str, Any]:
    """Validate concrete bindings against one canonical TeamSpec."""

    if not isinstance(value, dict):
        raise ValueError("TeamInstance must be a JSON object")
    instance_id = str(value.get("id") or "").strip()
    if not _SAFE_ID.fullmatch(instance_id):
        raise ValueError("TeamInstance id may contain only letters, numbers, '.', '_' and '-'")
    team_spec_id = str(value.get("team_spec_id") or "").strip()
    if team_spec_id != str(team_spec["id"]):
        raise ValueError(
            f"TeamInstance references TeamSpec {team_spec_id!r}, expected {team_spec['id']!r}"
        )

    slots = {str(item["id"]): item for item in team_spec.get("inference_slots") or []}
    raw_bindings = value.get("inference_bindings") or []
    if not isinstance(raw_bindings, list):
        raise ValueError("TeamInstance inference_bindings must be a list")
    normalized_bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_bindings:
        if not isinstance(raw, dict):
            raise ValueError("each inference binding must be an object")
        slot_id = str(raw.get("slot_id") or "").strip()
        deployment_instance_id = str(raw.get("deployment_instance_id") or "").strip()
        if slot_id not in slots:
            raise ValueError(f"TeamInstance binding references unknown inference slot {slot_id!r}")
        if slot_id in seen:
            raise ValueError(f"duplicate TeamInstance binding for inference slot {slot_id!r}")
        if not _SAFE_ID.fullmatch(deployment_instance_id):
            raise ValueError(f"inference slot {slot_id!r} requires a DeploymentInstance id")
        seen.add(slot_id)
        overrides = normalize_invocation_overrides(
            raw.get("generation_overrides"), label=f"inference slot {slot_id}"
        )
        normalized_bindings.append(
            {
                "slot_id": slot_id,
                "deployment_instance_id": deployment_instance_id,
                **({"generation_overrides": overrides} if overrides else {}),
            }
        )
    missing = sorted(set(slots) - seen)
    if missing:
        raise ValueError(
            "TeamInstance requires one DeploymentInstance binding for every inference slot; "
            f"missing: {', '.join(missing)}"
        )
    lifecycle = normalize_instance_lifecycle(value, prefix="team")
    return {
        "schema_version": 3,
        "id": instance_id,
        **lifecycle,
        "inference_bindings": normalized_bindings,
    }


def team_instance_availability(
    value: dict[str, Any],
    team_spec: dict[str, Any],
    deployment_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve bindings and verify deployment health and declared capabilities."""

    normalized = normalize_team_instance_document(value, team_spec)
    by_id = {str(item.get("id")): item for item in deployment_instances}
    slots = {str(item["id"]): item for item in team_spec.get("inference_slots") or []}
    referenced = {
        str(item["deployment_instance_id"])
        for item in normalized.get("inference_bindings") or []
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
    for binding in normalized.get("inference_bindings") or []:
        deployment = by_id.get(str(binding["deployment_instance_id"]))
        if deployment is None:
            continue
        capabilities = dict(deployment.get("capabilities") or {})
        for capability in slots[binding["slot_id"]].get("required_capabilities") or []:
            if capabilities.get(capability) is not True:
                capability_errors.append(
                    f"{binding['slot_id']} requires {capability} from "
                    f"{binding['deployment_instance_id']}"
                )
    available = not missing and not unavailable and not capability_errors
    return {
        **normalized,
        "team_spec": team_spec,
        "observed_status": "ready" if available else "unavailable",
        "available": available,
        "missing_deployment_instance_ids": missing,
        "unavailable_deployment_instance_ids": unavailable,
        "capability_errors": capability_errors,
    }


class TeamInstanceRegistry:
    """Store TeamSpec references and concrete inference-slot bindings."""

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
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{normalized['id']}.json"
        if path.exists():
            raise ValueError(
                f"TeamInstance {normalized['id']!r} already exists; use a new ID"
            )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
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
