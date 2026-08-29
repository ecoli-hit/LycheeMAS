"""FastAPI transport for Studio bootstrap and environment operations."""

from __future__ import annotations


def build_system_router(service):
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.get("/api/health")
    def health():
        return service.health()

    @router.get("/api/bootstrap")
    def bootstrap():
        return service.bootstrap()

    @router.get("/api/workspaces/{workspace_name}")
    def workspace_payload(workspace_name: str):
        try:
            return service.workspace(workspace_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/environments")
    def environments():
        return service.environments()

    @router.post("/api/environments/check")
    def check_environment(payload: dict):
        try:
            return service.check_environment(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router


__all__ = ["build_system_router"]
