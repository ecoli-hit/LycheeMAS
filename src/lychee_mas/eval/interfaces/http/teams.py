"""HTTP transport for TeamSpec and TeamInstance use cases."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ...application.teams import TeamApplicationService

_LOGGER = logging.getLogger(__name__)
_REPORTED_BINDING_ERRORS: set[str] = set()


def build_teams_router(service: TeamApplicationService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/teams")
    @router.get("/team-specs")
    def team_specs():
        return service.specs()

    @router.put("/team-specs/{team_id}")
    def save_team_spec(team_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != team_id:
            raise HTTPException(status_code=422, detail="team id does not match URL")
        try:
            return service.save_spec(team_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/team-specs/{team_id}")
    def delete_team_spec(team_id: str):
        try:
            return service.delete_spec(team_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/team-binding-report")
    def team_binding_report(payload: dict[str, Any]):
        try:
            return service.binding_report(payload)
        except ValueError as exc:
            detail = str(exc)
            if detail not in _REPORTED_BINDING_ERRORS:
                _REPORTED_BINDING_ERRORS.add(detail)
                _LOGGER.warning(
                    "Rejected Team binding report: framework=%s team_id=%s detail=%s",
                    payload.get("runtime_framework"),
                    (payload.get("team_spec") or {}).get("id"),
                    detail,
                )
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/team-instances")
    def team_instances(probe: bool = False):
        return service.instances(probe=probe)

    @router.put("/team-instances/{instance_id}")
    def save_team_instance(instance_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != instance_id:
            raise HTTPException(status_code=422, detail="team instance id does not match URL")
        try:
            return service.save_instance(instance_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/team-instances/{instance_id}")
    def delete_team_instance(instance_id: str):
        try:
            return service.delete_instance(instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
