"""Page-scoped read models for Eval Studio.

The browser should not need every registry just to render the environment page.
This module keeps page projections outside the HTTP transport layer and makes
their dependency and payload boundaries explicit.
"""

from __future__ import annotations

from typing import Any, Callable


class WorkspaceCatalog:
    """Build read-only payloads for one Studio workspace at a time."""

    def __init__(
        self,
        *,
        catalog,
        jobs,
        benchmark_registry,
        model_registry,
        api_registry,
        deployment_registry,
        pricing_registry,
        team_registry,
        team_instance_registry,
        experiment_registry,
        experiment_queue,
        discover_environments: Callable[[], dict[str, Any]],
        installation_profiles: Callable[[], list[dict[str, Any]]],
        startup_reconciliation: dict[str, Any],
    ) -> None:
        self.catalog = catalog
        self.jobs = jobs
        self.benchmark_registry = benchmark_registry
        self.model_registry = model_registry
        self.api_registry = api_registry
        self.deployment_registry = deployment_registry
        self.pricing_registry = pricing_registry
        self.team_registry = team_registry
        self.team_instance_registry = team_instance_registry
        self.experiment_registry = experiment_registry
        self.experiment_queue = experiment_queue
        self.discover_environments = discover_environments
        self.installation_profiles = installation_profiles
        self.startup_reconciliation = startup_reconciliation

    def core(self) -> dict[str, Any]:
        """Return only data needed before a management workspace opens."""

        return {
            "paths": self.catalog.paths(),
            "environments": self.discover_environments(),
            "installation_profiles": self.installation_profiles(),
            "tmux_sessions": self.jobs.tmux_sessions(),
            "runs": [],
            # Stable empty shapes keep rendering predictable until a page is
            # hydrated from its own read model.
            "teams": {"specs": []},
            "team_instances": {"instances": []},
            "benchmarks": [],
            "benchmark_registry": {"specs": [], "instances": []},
            "model_registry": {"specs": [], "instances": []},
            "api_registry": {"specs": [], "instances": []},
            "deployments": {"specs": [], "instances": []},
            "pricing_registry": {"specs": [], "instances": []},
            "metric_registry": {},
            "evaluation_profiles": {},
            "experiments": {"specs": [], "instances": [], "queue": {}, "orphans": []},
            "default_experiment_spec": {},
        }

    def workspace(self, name: str) -> dict[str, Any]:
        builders = {
            "environment": lambda: {},
            "resources": self.resources,
            "cost": self.cost,
            "deployment": self.deployment,
            "team": self.team,
            "experiment": self.experiment,
            "runs": lambda: {},
        }
        try:
            return builders[name]()
        except KeyError as exc:
            raise ValueError(f"unknown Studio workspace {name!r}") from exc

    def resources(self) -> dict[str, Any]:
        return {
            "benchmarks": self.catalog.benchmarks(include_asset_stats=False),
            "benchmark_registry": {
                "specs": self.benchmark_registry.specs(),
                "instances": self.benchmark_registry.instances(),
            },
            "model_registry": {
                "specs": self.model_registry.specs(),
                "instances": self.model_registry.instances(),
            },
            "api_registry": {
                "specs": self.api_registry.specs(),
                "instances": self.api_registry.instances(),
            },
        }

    def cost(self) -> dict[str, Any]:
        return {
            "pricing_registry": {
                "specs": self.pricing_registry.specs(),
                "instances": self.pricing_registry.instances(),
            }
        }

    def deployment(self) -> dict[str, Any]:
        return {
            **self.resources_for_deployment(),
            **self.cost(),
            "deployments": {
                "specs": self.deployment_registry.specs(),
                "instances": self.deployment_registry.instances(),
            },
        }

    def resources_for_deployment(self) -> dict[str, Any]:
        return {
            "model_registry": {
                "specs": self.model_registry.specs(),
                "instances": self.model_registry.instances(),
            },
            "api_registry": {
                "specs": self.api_registry.specs(),
                "instances": self.api_registry.instances(),
            },
        }

    def _team_instances(
        self,
        deployment_instances: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in item.items()
                if key not in {"team_spec", "registry_path"}
            }
            for item in self.team_instance_registry.all(deployment_instances)
        ]

    def team(self) -> dict[str, Any]:
        deployment_instances = self.deployment_registry.instances()
        return {
            "teams": {"specs": self.team_registry.all()},
            "team_instances": {
                "instances": self._team_instances(deployment_instances),
            },
            "deployments": {
                "specs": self.deployment_registry.specs(),
                "instances": deployment_instances,
            },
        }

    def experiment(self) -> dict[str, Any]:
        deployment_instances = self.deployment_registry.instances()
        return {
            "benchmark_registry": {
                "specs": self.benchmark_registry.specs(),
                "instances": self.benchmark_registry.instances(),
            },
            "teams": {"specs": self.team_registry.all()},
            "team_instances": {
                "instances": self._team_instances(deployment_instances),
            },
            "experiments": {
                "specs": self.experiment_registry.specs(),
                "instances": self.experiment_queue.instances(tail=0, compact=True),
                "queue": self.experiment_queue.status(),
                "orphans": self.experiment_queue.orphaned_launches(),
                "startup_reconciliation": self.startup_reconciliation,
            },
            "default_experiment_spec": self.experiment_registry.default_spec(),
            "tmux_sessions": self.jobs.tmux_sessions(),
        }
