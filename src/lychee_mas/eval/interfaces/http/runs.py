"""HTTP transport for run browsing, replay, and offline evaluation."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...application.runs import RunApplicationService


def build_runs_router(
    service: RunApplicationService,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/runs")
    def runs(limit: int = Query(default=100, ge=1, le=1000)):
        return service.list_runs(limit=limit)

    @router.get("/run-page")
    def run_page(
        query: str = Query(default="", max_length=512),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=250),
    ):
        return service.list_runs_page(query=query, offset=offset, limit=limit)

    @router.get("/metric-contracts")
    def metric_contracts():
        return service.metric_contracts()

    @router.get("/evaluation-profiles")
    def evaluation_profile_contracts():
        return service.evaluation_profile_contracts()

    @router.get("/runs/{run_id}/run-events")
    def run_events(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            return service.events(run_id, start_line=start_line, limit=limit)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/execution-trace")
    def run_execution_trace(
        run_id: str,
        case_id: str | None = None,
        trial_index: int | None = Query(default=None, ge=0),
        filters: str = "",
        start_node: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            return service.execution_trace(
                run_id,
                case_id=case_id,
                trial_index=trial_index,
                filters=tuple(item.strip() for item in filters.split(",") if item.strip()),
                start_node=start_node,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported or invalid RunEvent log: {exc}",
            ) from exc

    @router.get("/runs/{run_id}/results")
    def run_results(
        run_id: str,
        start_trial: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.results(run_id, start_trial=start_trial, limit=limit)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported or invalid RunEvent log: {exc}",
            ) from exc

    @router.get("/runs/{run_id}/evidence")
    def run_evidence(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            return service.evidence(run_id, start_line=start_line, limit=limit)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/metric-observations")
    def run_metric_observations(
        run_id: str,
        start_line: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        try:
            return service.metric_observations(
                run_id,
                start_line=start_line,
                limit=limit,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/metric-trials")
    def run_metric_trials(
        run_id: str,
        start_trial: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.metric_trials(run_id, start_trial=start_trial, limit=limit)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/evidence-coverage")
    def run_evidence_coverage(run_id: str):
        try:
            return service.evidence_coverage(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/metric-applicability")
    def run_metric_applicability(run_id: str):
        try:
            return service.metric_applicability(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/metric-evaluation")
    def run_metric_evaluation(
        run_id: str,
        profile_id: str = Query(default="core"),
    ):
        try:
            return service.metric_evaluation(run_id, profile_id=profile_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/analyze")
    def analyze_existing_run(
        run_id: str,
        profile_id: str = Query(default="core"),
    ):
        try:
            return service.analyze(run_id, profile_id=profile_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/run-events/stream")
    async def stream_events(run_id: str, start_line: int = Query(default=0, ge=0)):
        try:
            service.run_dir(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def generate():
            cursor = start_line
            while True:
                batch = service.events(
                    run_id,
                    start_line=cursor,
                    limit=200,
                    include_total=False,
                )
                for event in batch["events"]:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                cursor = batch["next_line"]
                yield f"event: cursor\ndata: {cursor}\n\n"
                await asyncio.sleep(0.75)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router
