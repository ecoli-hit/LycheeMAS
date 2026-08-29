"""TeamSpec and TeamInstance use cases shared by all transports."""

from __future__ import annotations

from typing import Any

from ...runtime.adapters.frameworks.bindings import build_framework_binding_report
from ...runtime.coordination.compiler import compile_coordination
from ..teams.contracts import normalize_team_spec_document
from ..teams.instances import team_instance_availability
from .ports import (
    DeploymentRegistryPort,
    ExperimentRepositoryPort,
    TeamInstanceRepositoryPort,
    TeamSpecRepositoryPort,
)


class TeamApplicationService:
    def __init__(
        self,
        *,
        teams: TeamSpecRepositoryPort,
        team_instances: TeamInstanceRepositoryPort,
        deployments: DeploymentRegistryPort,
        experiments: ExperimentRepositoryPort,
    ) -> None:
        self.teams = teams
        self.team_instances = team_instances
        self.deployments = deployments
        self.experiments = experiments

    def specs(self) -> dict[str, Any]:
        return {"specs": self.teams.all()}

    def save_spec(self, team_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.teams.save({**payload, "id": team_id})

    def delete_spec(self, team_id: str) -> dict[str, Any]:
        if any(item.get("team_spec_id") == team_id for item in self.team_instances.all()) or any(
            item.get("team_spec_id") == team_id for item in self.experiments.specs()
        ):
            raise ValueError("TeamSpec is referenced by a TeamInstance or ExperimentSpec")
        return self.teams.delete(team_id)

    def binding_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        framework = str(payload.get("runtime_framework") or "").strip().lower()
        team_document = dict(payload.get("team_spec") or {})
        team_document.pop("registry_path", None)
        team_document.pop("updated_at_utc", None)
        team_spec = normalize_team_spec_document(team_document)
        return build_framework_binding_report(
            framework,
            compile_coordination(team_spec),
            team_spec,
        )

    def instances(self, *, probe: bool = False) -> dict[str, Any]:
        deployments = self.deployments.instances(probe=probe)
        return {"instances": self.team_instances.all(deployments)}

    def save_instance(self, instance_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        saved = self.team_instances.save({**payload, "id": instance_id})
        return team_instance_availability(
            saved,
            saved["team_spec"],
            self.deployments.instances(),
        )

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        if any(
            item.get("team_instance_id") == instance_id
            for item in self.experiments.instances()
        ):
            raise ValueError("TeamInstance is referenced by an ExperimentInstance")
        return self.team_instances.delete(instance_id)
