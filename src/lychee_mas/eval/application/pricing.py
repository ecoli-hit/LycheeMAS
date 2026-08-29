"""Pricing use cases shared by every Eval interface."""

from __future__ import annotations

from typing import Any

from .ports import DeploymentRegistryPort, PricingRegistryPort


class PricingApplicationService:
    def __init__(
        self,
        *,
        pricing: PricingRegistryPort,
        deployments: DeploymentRegistryPort,
    ) -> None:
        self.pricing = pricing
        self.deployments = deployments

    def catalog(self) -> dict[str, Any]:
        return {"specs": self.pricing.specs(), "instances": self.pricing.instances()}

    def save_spec(self, spec_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.pricing.save_spec(spec_id, payload)

    def delete_spec(self, spec_id: str) -> dict[str, Any]:
        if any(
            spec_id
            in {
                item.get("actual_pricing_spec_id"),
                item.get("api_equivalent_pricing_spec_id"),
            }
            for item in self.deployments.specs()
        ):
            raise ValueError("PricingSpec is referenced by a DeploymentSpec")
        return self.pricing.delete_spec(spec_id)

    def instantiate(
        self,
        spec_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        instance_id = str(payload.get("id") or "").strip()
        if not instance_id:
            raise ValueError("PricingInstance id is required")
        return self.pricing.instantiate_instance(spec_id, instance_id, payload)

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        if any(
            instance_id
            in {
                item.get("actual_pricing_instance_id"),
                item.get("api_equivalent_pricing_instance_id"),
            }
            for item in self.deployments.instances()
        ):
            raise ValueError("PricingInstance is referenced by a DeploymentInstance")
        return self.pricing.delete_instance(instance_id)
