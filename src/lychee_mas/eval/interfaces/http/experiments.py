"""HTTP transport for ExperimentSpec, ExperimentInstance, and queue commands."""

from __future__ import annotations

import subprocess
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from ...application.experiments import ExperimentControlApplicationService


def _translate(
    operation: Callable[[], Any],
    *,
    missing_status: int = 422,
) -> Any:
    try:
        return operation()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=missing_status, detail=str(exc)) from exc
    except (
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def build_experiments_router(service: ExperimentControlApplicationService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/experiment-specs")
    def list_experiment_specs():
        return service.specs()

    @router.get("/experiment-dashboard")
    def experiment_dashboard(include_specs: bool = False):
        return service.dashboard(include_specs=include_specs)

    @router.put("/experiment-specs/{spec_id}")
    def save_experiment_spec(spec_id: str, payload: dict[str, Any]):
        return _translate(lambda: service.save_spec(spec_id, payload))

    @router.delete("/experiment-specs/{spec_id}")
    def delete_experiment_spec(spec_id: str):
        return _translate(lambda: service.delete_spec(spec_id), missing_status=404)

    @router.get("/experiment-instances")
    def list_experiment_instances(tail: int = 0, compact: bool = True):
        return service.instances(tail=tail, compact=compact)

    @router.post("/experiment-instances")
    def create_experiment_instance(payload: dict[str, Any]):
        return _translate(lambda: service.create_instance(payload))

    @router.delete("/experiment-instances/{instance_id}")
    def delete_experiment_instance(instance_id: str):
        return _translate(lambda: service.delete_instance(instance_id), missing_status=404)

    @router.patch("/experiment-instances/{instance_id}")
    def update_experiment_instance(instance_id: str, payload: dict[str, Any]):
        return _translate(
            lambda: service.update_instance(instance_id, payload),
            missing_status=404,
        )

    @router.post("/experiment-instances/{instance_id}/plan")
    def plan_experiment_instance(instance_id: str):
        return _translate(lambda: service.plan(instance_id))

    @router.post("/experiment-instances/{instance_id}/launch")
    def launch_experiment_instance(instance_id: str):
        return _translate(lambda: service.launch(instance_id))

    @router.post("/experiment-instances/{instance_id}/enqueue")
    def enqueue_experiment_instance(instance_id: str):
        return _translate(lambda: service.enqueue(instance_id))

    @router.post("/experiment-instances/{instance_id}/dequeue")
    def dequeue_experiment_instance(instance_id: str):
        return _translate(lambda: service.dequeue(instance_id))

    @router.post("/experiment-instances/{instance_id}/stop")
    def stop_experiment_instance(instance_id: str):
        return _translate(lambda: service.stop(instance_id))

    @router.post("/experiment-instances/{instance_id}/resume")
    def resume_experiment_instance(instance_id: str):
        return _translate(lambda: service.resume(instance_id))

    @router.post("/experiment-instances/{instance_id}/drain")
    def drain_experiment_instance(instance_id: str):
        return _translate(lambda: service.drain(instance_id))

    @router.get("/experiment-instances/{instance_id}/progress")
    def experiment_instance_progress(
        instance_id: str,
        tail: int = Query(40, ge=0, le=200),
    ):
        return _translate(lambda: service.progress(instance_id, tail=tail), missing_status=404)

    @router.get("/experiment-orphans")
    def list_orphaned_experiment_launches():
        return service.orphans()

    @router.post("/experiment-orphans/{launch_id}/stop")
    def stop_orphaned_experiment_launch(launch_id: str):
        return _translate(lambda: service.stop_orphan(launch_id))

    @router.post("/experiment-orphans/{launch_id}/recover")
    def recover_orphaned_experiment_launch(launch_id: str):
        return _translate(lambda: service.recover_orphan(launch_id))

    @router.get("/experiment-queue")
    def experiment_queue_status():
        return service.queue_status()

    @router.post("/experiment-queue/start")
    def start_experiment_queue(payload: dict[str, Any] | None = None):
        return _translate(lambda: service.start_queue(payload))

    @router.post("/experiment-queue/stop-after-current")
    def stop_experiment_queue():
        return service.stop_queue_after_current()

    return router
