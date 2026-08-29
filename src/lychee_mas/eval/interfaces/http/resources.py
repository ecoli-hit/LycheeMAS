"""HTTP transport for registered Model, API-access, and Benchmark assets."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from ...application.resources import ResourceApplicationService


def _translate(operation: Callable[[], Any], *, missing_status: int = 422) -> Any:
    try:
        return operation()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=missing_status, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def build_resources_router(service: ResourceApplicationService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/resources/catalog")
    def resource_catalog(
        raw_root: str | None = None,
        prepared_root: str | None = None,
        models_root: str | None = None,
    ):
        return _translate(
            lambda: service.catalog_payload(
                raw_root=raw_root,
                prepared_root=prepared_root,
                models_root=models_root,
            )
        )

    @router.get("/resource-registries")
    def resource_registries():
        return service.registries()

    @router.get("/api-specs")
    def api_specs():
        return service.api_specs()

    @router.get("/api-instances")
    def api_instances():
        return service.api_instances()

    @router.post("/api-instances/{instance_id}/instantiate")
    def instantiate_api_instance(instance_id: str, payload: dict[str, Any]):
        if payload.get("id") and str(payload["id"]) != instance_id:
            raise HTTPException(status_code=422, detail="APIInstance id does not match URL")
        return _translate(lambda: service.instantiate_api(instance_id, payload))

    @router.delete("/api-instances/{instance_id}")
    def delete_api_instance(instance_id: str):
        return _translate(lambda: service.delete_api(instance_id), missing_status=404)

    @router.get("/model-specs")
    def model_specs():
        return service.model_specs()

    @router.get("/model-instances")
    def model_instances():
        return service.model_instances()

    @router.post("/model-instances/scan")
    def scan_model_instances(payload: dict[str, Any]):
        return _translate(lambda: service.scan_models(payload))

    @router.post("/model-instances/{instance_id}/instantiate")
    def instantiate_model_instance(instance_id: str, payload: dict[str, Any]):
        return _translate(lambda: service.instantiate_model(instance_id, payload))

    @router.delete("/model-instances/{instance_id}")
    def delete_model_instance(instance_id: str):
        return _translate(lambda: service.delete_model(instance_id), missing_status=404)

    @router.get("/benchmarks")
    def benchmarks():
        return service.benchmark_catalog()

    @router.get("/benchmark-specs")
    def benchmark_specs():
        return service.benchmark_specs()

    @router.get("/benchmark-instances")
    def benchmark_instances():
        return service.benchmark_instances()

    @router.post("/benchmark-instances/scan")
    def scan_benchmark_instances(payload: dict[str, Any]):
        return _translate(lambda: service.scan_benchmarks(payload))

    @router.post("/benchmark-instances/{instance_id}/instantiate")
    def instantiate_benchmark_instance(instance_id: str, payload: dict[str, Any]):
        return _translate(lambda: service.instantiate_benchmark(instance_id, payload))

    @router.post("/benchmark-instances/{instance_id}/check")
    def check_benchmark_instance(instance_id: str):
        return _translate(lambda: service.check_benchmark(instance_id))

    @router.delete("/benchmark-instances/{instance_id}")
    def delete_benchmark_instance(instance_id: str):
        return _translate(lambda: service.delete_benchmark(instance_id), missing_status=404)

    @router.post("/datasets/prepare")
    def prepare_dataset(payload: dict[str, Any]):
        return _translate(
            lambda: service.prepare_benchmark(
                payload,
                target=str(payload.get("target") or ""),
            )
        )

    @router.post("/resources/import")
    def reject_unregistered_resource_import(_payload: dict[str, Any]):
        raise HTTPException(
            status_code=422,
            detail=(
                "Resources must be created by instantiating a registered "
                "BenchmarkSpec, ModelSpec or APISpec"
            ),
        )

    @router.get("/tmux/sessions")
    def tmux_sessions():
        return service.tmux_sessions()

    @router.get("/launches/{launch_id}")
    def launch_status(launch_id: str):
        return _translate(lambda: service.launch_status(launch_id), missing_status=404)

    @router.get("/launches/{launch_id}/log")
    def launch_log(launch_id: str, tail: int = Query(default=200, ge=1, le=5000)):
        return _translate(
            lambda: service.launch_log(launch_id, tail=tail),
            missing_status=404,
        )

    @router.get("/launches/{launch_id}/progress")
    def launch_progress(launch_id: str):
        return _translate(lambda: service.launch_progress(launch_id), missing_status=404)

    @router.post("/launches/{launch_id}/stop")
    def stop_launch(launch_id: str):
        return _translate(lambda: service.stop_launch(launch_id), missing_status=404)

    return router
