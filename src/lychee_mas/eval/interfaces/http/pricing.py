"""HTTP transport for PricingSpec and PricingInstance use cases."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...application.pricing import PricingApplicationService


def build_pricing_router(service: PricingApplicationService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/pricing")
    def pricing():
        return service.catalog()

    @router.put("/pricing-specs/{spec_id}")
    def save_pricing_spec(spec_id: str, payload: dict[str, Any]):
        try:
            return service.save_spec(spec_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/pricing-specs/{spec_id}")
    def delete_pricing_spec(spec_id: str):
        try:
            return service.delete_spec(spec_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/pricing-specs/{spec_id}/instances")
    def instantiate_pricing_instance(spec_id: str, payload: dict[str, Any]):
        try:
            return service.instantiate(spec_id, payload)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/pricing-instances/{instance_id}")
    def delete_pricing_instance(instance_id: str):
        try:
            return service.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
