"""HTTP transport for DeploymentSpec and DeploymentInstance use cases."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...application.deployments import DeploymentApplicationService


def build_deployments_router(service: DeploymentApplicationService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/deployments")
    def deployments(probe: bool = False):
        return service.catalog(probe=probe)

    @router.put("/deployments/{deployment_id}")
    def save_deployment(deployment_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != deployment_id:
            raise HTTPException(status_code=422, detail="deployment id does not match URL")
        try:
            return service.save_spec(deployment_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/deployments/{deployment_id}")
    def delete_deployment(deployment_id: str):
        try:
            return service.delete_spec(deployment_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/deployments/{deployment_id}/deploy")
    def deploy_deployment(deployment_id: str, payload: dict[str, Any]):
        try:
            return service.instantiate(deployment_id, payload)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/deployment-instances/{instance_id}")
    def delete_deployment_instance(instance_id: str):
        try:
            return service.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
