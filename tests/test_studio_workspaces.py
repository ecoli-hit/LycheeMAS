from __future__ import annotations

from apps.eval.server.app import create_app
from fastapi.testclient import TestClient
from lychee_mas.eval.application.read_models.catalog import repository_root


def test_repository_root_resolves_project_root() -> None:
    root = repository_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "lychee_mas").is_dir()


def test_bootstrap_contains_only_core_workspace_shapes(tmp_path) -> None:
    payload = TestClient(create_app(tmp_path)).get("/api/bootstrap").json()

    assert payload["environments"]["default"]
    assert payload["experiments"] == {
        "specs": [],
        "instances": [],
        "queue": {},
        "orphans": [],
    }
    assert payload["team_instances"] == {"instances": []}
    assert payload["runs"] == []


def test_workspaces_hydrate_their_own_read_models(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    resources = client.get("/api/workspaces/resources")
    experiment = client.get("/api/workspaces/experiment")

    assert resources.status_code == 200
    assert set(resources.json()) == {
        "benchmarks",
        "benchmark_registry",
        "model_registry",
        "api_registry",
    }
    assert experiment.status_code == 200
    assert set(experiment.json()) == {
        "benchmark_registry",
        "teams",
        "team_instances",
        "experiments",
        "default_experiment_spec",
        "tmux_sessions",
    }


def test_unknown_workspace_returns_not_found(tmp_path) -> None:
    response = TestClient(create_app(tmp_path)).get("/api/workspaces/not-a-workspace")

    assert response.status_code == 404
    assert "unknown Studio workspace" in response.json()["detail"]
