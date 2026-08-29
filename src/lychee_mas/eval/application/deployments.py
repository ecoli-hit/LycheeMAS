"""Deployment use cases shared by HTTP, CLI, and future TUI transports."""

from __future__ import annotations

from typing import Any

from .ports import (
    APIAccessRegistryPort,
    DeploymentPressureProvider,
    DeploymentRegistryPort,
    ModelRegistryPort,
    TeamInstanceRepositoryPort,
)


class DeploymentApplicationService:
    """Coordinate model, API-access, deployment, and team registries."""

    def __init__(
        self,
        *,
        deployments: DeploymentRegistryPort,
        models: ModelRegistryPort,
        api_access: APIAccessRegistryPort,
        team_instances: TeamInstanceRepositoryPort,
        health_monitor: DeploymentPressureProvider | None = None,
    ) -> None:
        self.deployments = deployments
        self.models = models
        self.api_access = api_access
        self.team_instances = team_instances
        self.health_monitor = health_monitor

    def catalog(self, *, probe: bool = False) -> dict[str, Any]:
        return {
            "specs": self.deployments.specs(),
            "instances": self.deployments.instances(probe=probe),
            "pressure": self.health_monitor.snapshot() if self.health_monitor else None,
        }

    def validate_spec_references(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate Spec-to-Spec references without resolving resource Instances."""

        value = dict(payload)
        value.pop("api_key", None)
        source_spec = dict(value.get("source_spec") or {})
        source_type = str(source_spec.get("type") or "")
        source_id = str(source_spec.get("id") or "")
        model_spec_id = str(value.get("model_spec_id") or "")
        if source_type == "model":
            self.models.get_spec(source_id)
            if model_spec_id and model_spec_id != source_id:
                raise ValueError(
                    "local DeploymentSpec model_spec_id must match source_spec.id"
                )
            value["model_spec_id"] = source_id
        elif source_type == "api":
            api_spec = self.api_access.get_spec(source_id)
            self.models.get_spec(model_spec_id)
            if model_spec_id not in set(api_spec.get("allowed_model_spec_ids") or []):
                raise ValueError(
                    f"APISpec {api_spec['id']!r} does not allow "
                    f"ModelSpec {model_spec_id!r}"
                )
        else:
            raise ValueError("DeploymentSpec source_spec.type must be model or api")
        return value

    def save_spec(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.deployments.save(
            self.validate_spec_references({**payload, "id": deployment_id})
        )

    def delete_spec(self, deployment_id: str) -> dict[str, Any]:
        if any(
            item.get("deployment_spec_id") == deployment_id
            for item in self.deployments.instances()
        ):
            raise ValueError("DeploymentSpec is referenced by a DeploymentInstance")
        return self.deployments.delete(deployment_id)

    def resolve_source(
        self,
        spec: dict[str, Any],
        source_instance_id: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Resolve one resource Instance and freeze its effective model capabilities."""

        source_spec = dict(spec["source_spec"])
        model_spec = self.models.get_spec(str(spec["model_spec_id"]))
        if source_spec["type"] == "model":
            model = self.models.get_instance(source_instance_id)
            if model["model_spec_id"] != source_spec["id"]:
                raise ValueError(
                    f"ModelInstance {source_instance_id!r} is not an instance of "
                    f"ModelSpec {source_spec['id']!r}"
                )
            if not model.get("available"):
                raise ValueError(f"ModelInstance {source_instance_id!r} is not ready")
            fields = {
                "model_id": model_spec["name"],
                "model_path": model["path"],
                "model_info": dict(model_spec.get("model_info") or {}),
                "capabilities": dict(model_spec.get("capabilities") or {}),
                "thinking_protocol": model_spec.get("thinking_protocol"),
            }
            if spec["kind"] == "vllm":
                fields.update(auth_mode="none", trust_env=False)
            return {"type": "model", "id": source_instance_id}, fields

        api_instance = self.api_access.get(source_instance_id)
        if api_instance["api_spec_id"] != source_spec["id"]:
            raise ValueError(
                f"APIInstance {source_instance_id!r} is not an instance of "
                f"APISpec {source_spec['id']!r}"
            )
        fields = self.api_access.deployment_fields(
            source_instance_id,
            model_spec=model_spec,
        )
        if fields["kind"] != spec["kind"]:
            raise ValueError(
                f"APIInstance {source_instance_id!r} provides {fields['kind']!r}, "
                f"not {spec['kind']!r}"
            )
        return {"type": "api", "id": source_instance_id}, fields

    def instantiate(
        self,
        deployment_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        spec = next(
            (item for item in self.deployments.specs() if item["id"] == deployment_id),
            None,
        )
        if spec is None:
            raise FileNotFoundError(f"unknown DeploymentSpec {deployment_id!r}")
        source_instance_id = str(payload.get("source_instance_id") or "").strip()
        if not source_instance_id:
            raise ValueError("DeploymentInstance requires source_instance_id")
        actual_pricing_instance_id = str(
            payload.get("actual_pricing_instance_id") or ""
        ).strip()
        if not actual_pricing_instance_id:
            raise ValueError("DeploymentInstance requires actual_pricing_instance_id")
        source_instance, source_fields = self.resolve_source(spec, source_instance_id)
        return self.deployments.deploy(
            spec,
            instance_id=str(
                payload.get("instance_id") or f"instance-{deployment_id}"
            ).strip(),
            source_instance=source_instance,
            source_fields=source_fields,
            actual_pricing_instance_id=actual_pricing_instance_id,
            api_equivalent_pricing_instance_id=(
                str(payload["api_equivalent_pricing_instance_id"])
                if payload.get("api_equivalent_pricing_instance_id")
                else None
            ),
            api_key=str(payload.get("api_key") or ""),
        )

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        if any(
            binding.get("resource_instance_type") == "DeploymentInstance"
            and binding.get("resource_instance_id") == instance_id
            for team in self.team_instances.all()
            for binding in team.get("resource_bindings") or []
        ):
            raise ValueError("DeploymentInstance is referenced by a TeamInstance")
        return self.deployments.delete_instance(instance_id)
