"""Composition root for the LycheeMAS Eval HTTP server and Web client."""

from __future__ import annotations

import os
from pathlib import Path

from lychee_mas.eval.apis.registry import APIRegistry
from lychee_mas.eval.application.deployments import DeploymentApplicationService
from lychee_mas.eval.application.experiments import (
    ExperimentApplicationService,
    ExperimentControlApplicationService,
)
from lychee_mas.eval.application.pricing import PricingApplicationService
from lychee_mas.eval.application.read_models import WorkspaceCatalog
from lychee_mas.eval.application.read_models.catalog import EvalCatalog, repository_root
from lychee_mas.eval.application.resources import ResourceApplicationService
from lychee_mas.eval.application.runs import RunApplicationService
from lychee_mas.eval.application.system import SystemApplicationService
from lychee_mas.eval.application.teams import TeamApplicationService
from lychee_mas.eval.benchmarks.assets import BenchmarkRegistry
from lychee_mas.eval.deployments.pressure import DeploymentHealthMonitor
from lychee_mas.eval.deployments.registry import DeploymentRegistry
from lychee_mas.eval.environment.service import (
    discover_environments,
    installation_profiles,
)
from lychee_mas.eval.evaluation.evaluators import EvaluationProfileRegistry
from lychee_mas.eval.evaluation.metrics_registry import MetricRegistry
from lychee_mas.eval.experiments.compiler import ExecutionPlanCompiler
from lychee_mas.eval.experiments.registry import ExperimentRegistry
from lychee_mas.eval.interfaces.http.deployments import build_deployments_router
from lychee_mas.eval.interfaces.http.experiments import build_experiments_router
from lychee_mas.eval.interfaces.http.pricing import build_pricing_router
from lychee_mas.eval.interfaces.http.resources import build_resources_router
from lychee_mas.eval.interfaces.http.runs import build_runs_router
from lychee_mas.eval.interfaces.http.system import build_system_router
from lychee_mas.eval.interfaces.http.teams import build_teams_router
from lychee_mas.eval.models.registry import ModelRegistry
from lychee_mas.eval.pricing.registry import PricingRegistry
from lychee_mas.eval.scheduling.jobs import JobManager
from lychee_mas.eval.scheduling.manager import ExperimentQueueManager
from lychee_mas.eval.teams.instances import TeamInstanceRegistry
from lychee_mas.eval.teams.registry import TeamSpecRegistry


def create_app(repo_root: str | os.PathLike | None = None):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    root = Path(repo_root or repository_root()).resolve()
    catalog = EvalCatalog(root)
    compiler = ExecutionPlanCompiler(root)
    jobs = JobManager(compiler)
    deployment_registry = DeploymentRegistry(root)
    pricing_registry = PricingRegistry(root)
    team_registry = TeamSpecRegistry(root)
    team_instance_registry = TeamInstanceRegistry(root)
    benchmark_registry = BenchmarkRegistry(root)
    model_registry = ModelRegistry(root)
    api_registry = APIRegistry(root)
    experiment_registry = ExperimentRegistry(root)
    metric_registry = MetricRegistry()
    evaluation_profile_registry = EvaluationProfileRegistry()
    deployment_health_monitor = DeploymentHealthMonitor(deployment_registry)
    deployment_service = DeploymentApplicationService(
        deployments=deployment_registry,
        models=model_registry,
        api_access=api_registry,
        team_instances=team_instance_registry,
        health_monitor=deployment_health_monitor,
    )
    experiment_service = ExperimentApplicationService(
        experiments=experiment_registry,
        benchmarks=benchmark_registry,
        teams=team_registry,
        team_instances=team_instance_registry,
        deployments=deployment_registry,
        pricing=pricing_registry,
        models=model_registry,
        api_access=api_registry,
    )
    run_service = RunApplicationService(
        catalog=catalog,
        experiments=experiment_registry,
        metrics=metric_registry,
        evaluation_profiles=evaluation_profile_registry,
    )
    pricing_service = PricingApplicationService(
        pricing=pricing_registry,
        deployments=deployment_registry,
    )
    team_service = TeamApplicationService(
        teams=team_registry,
        team_instances=team_instance_registry,
        deployments=deployment_registry,
        experiments=experiment_registry,
    )
    resource_service = ResourceApplicationService(
        repo_root=root,
        catalog=catalog,
        jobs=jobs,
        benchmarks=benchmark_registry,
        models=model_registry,
        api_access=api_registry,
        deployments=deployment_registry,
        experiments=experiment_registry,
    )
    app = FastAPI(title="LycheeMAS Eval Studio", version="0.1")
    experiment_queue = ExperimentQueueManager(
        experiment_registry,
        compiler,
        jobs,
        experiment_service.assemble,
        deployment_registry=deployment_registry,
        deployment_health_monitor=deployment_health_monitor,
    )
    experiment_control_service = ExperimentControlApplicationService(
        assembler=experiment_service,
        experiments=experiment_registry,
        compiler=compiler,
        queue=experiment_queue,
    )
    app.include_router(build_experiments_router(experiment_control_service))
    startup_reconciliation = experiment_queue.reconcile()
    workspace_catalog = WorkspaceCatalog(
        catalog=catalog,
        jobs=jobs,
        benchmark_registry=benchmark_registry,
        model_registry=model_registry,
        api_registry=api_registry,
        deployment_registry=deployment_registry,
        pricing_registry=pricing_registry,
        team_registry=team_registry,
        team_instance_registry=team_instance_registry,
        experiment_registry=experiment_registry,
        experiment_queue=experiment_queue,
        discover_environments=lambda: discover_environments(root),
        installation_profiles=installation_profiles,
        startup_reconciliation=startup_reconciliation,
    )
    system_service = SystemApplicationService(
        repo_root=root,
        workspace_catalog=workspace_catalog,
    )
    app.include_router(build_system_router(system_service))
    app.include_router(build_deployments_router(deployment_service))
    app.include_router(build_pricing_router(pricing_service))
    app.include_router(build_resources_router(resource_service))
    app.include_router(build_runs_router(run_service))
    app.include_router(build_teams_router(team_service))

    frontend_dist = root / "apps/eval/web/dist"
    if frontend_dist.is_dir():
        assets = frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{page:path}", include_in_schema=False)
        def frontend(page: str):
            candidate = (frontend_dist / page).resolve()
            if page and candidate.is_file() and frontend_dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app
