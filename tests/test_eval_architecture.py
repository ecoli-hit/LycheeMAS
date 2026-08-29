from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    if path.name.startswith("._"):
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def test_scheduler_policy_modules_remain_transport_and_io_free() -> None:
    scheduling = ROOT / "src/lychee_mas/eval/scheduling"

    for name in ("admission.py", "projections.py"):
        imports = _imports(scheduling / name)
        assert not any(
            item.startswith(("fastapi", "subprocess", "pathlib")) for item in imports
        ), f"{name} must remain a deterministic policy/read-model module"


def test_deployment_health_monitor_is_owned_outside_scheduler_policy() -> None:
    scheduling = ROOT / "src/lychee_mas/eval/scheduling"

    assert not (scheduling / "health.py").exists()
    assert (ROOT / "src/lychee_mas/eval/deployments/pressure.py").is_file()


def test_scheduler_manager_delegates_provider_health_math() -> None:
    source = (
        ROOT / "src/lychee_mas/eval/scheduling/manager.py"
    ).read_text(encoding="utf-8")

    assert "assess_vllm_pressure" not in source
    assert "vllm_metrics.jsonl" not in source
    assert "DeploymentHealthMonitor" in source
    assert "rebalance_trial_admissions" in source


def test_workspace_read_models_do_not_depend_on_fastapi() -> None:
    imports = _imports(ROOT / "src/lychee_mas/eval/application/read_models/workspace_catalog.py")

    assert not any(item.startswith("fastapi") for item in imports)


def test_application_services_do_not_depend_on_fastapi() -> None:
    application = ROOT / "src/lychee_mas/eval/application"

    for path in application.glob("*.py"):
        if path.name.startswith("._"):
            continue
        imports = _imports(path)
        assert not any(item.startswith("fastapi") for item in imports), (
            f"{path.name} must be reusable by HTTP, CLI, and future TUI transports"
        )


def test_application_services_type_against_registry_ports() -> None:
    application = ROOT / "src/lychee_mas/eval/application"
    concrete_imports = {
        "from ..apis.registry import APIRegistry",
        "from ..benchmarks.assets import BenchmarkRegistry",
        "from ..deployments.registry import DeploymentRegistry",
        "from ..experiments.registry import ExperimentRegistry",
        "from ..models.registry import ModelRegistry",
        "from ..pricing.registry import PricingRegistry",
        "from ..teams.instances import TeamInstanceRegistry",
        "from ..teams.registry import TeamSpecRegistry",
    }

    for path in application.glob("*.py"):
        if path.name.startswith("._"):
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(statement in source for statement in concrete_imports), (
            f"{path.name} must depend on structural ports, not file-backed registries"
        )


def test_api_factory_contains_no_business_api_routes() -> None:
    source = (ROOT / "apps/eval/server/app.py").read_text(encoding="utf-8")

    assert '@app.get("/api/' not in source
    assert '@app.post("/api/' not in source
    assert source.count("\n") < 220


def test_eval_clients_and_server_have_distinct_roots() -> None:
    assert (ROOT / "apps/eval/server/app.py").is_file()
    assert (ROOT / "apps/eval/server/main.py").is_file()
    assert (ROOT / "apps/eval/web/package.json").is_file()
    assert not (ROOT / "apps/eval_studio").exists()
    assert not (ROOT / "src/lychee_mas/eval/control_plane").exists()


def test_eval_and_runtime_roots_only_contain_package_entry_points() -> None:
    for relative in ("src/lychee_mas/eval", "src/lychee_mas/runtime"):
        root = ROOT / relative
        assert sorted(
            path.name for path in root.glob("*.py") if not path.name.startswith("._")
        ) == ["__init__.py"]

    assert not (ROOT / "src/lychee_mas/eval/studio").exists()
    assert not (ROOT / "src/lychee_mas/runtime/backends").exists()


def test_cycle_sensitive_package_initializers_remain_lightweight() -> None:
    for relative in (
        "src/lychee_mas/eval/application/__init__.py",
        "src/lychee_mas/eval/experiments/__init__.py",
        "src/lychee_mas/eval/interfaces/http/__init__.py",
        "src/lychee_mas/eval/scheduling/__init__.py",
        "src/lychee_mas/eval/teams/__init__.py",
    ):
        assert not _imports(ROOT / relative), f"{relative} must not eagerly initialize owners"


def test_http_interfaces_depend_on_application_services_not_registries() -> None:
    interface_root = ROOT / "src/lychee_mas/eval/interfaces/http"

    for path in interface_root.glob("*.py"):
        imports = _imports(path)
        assert not any(
            item.startswith(
                (
                    "lychee_mas.eval.apis",
                    "lychee_mas.eval.benchmarks.assets",
                    "lychee_mas.eval.deployments",
                    "lychee_mas.eval.experiments",
                    "lychee_mas.eval.models",
                    "lychee_mas.eval.pricing",
                    "lychee_mas.eval.scheduling",
                    "lychee_mas.eval.teams",
                )
            )
            for item in imports
        ), f"{path.name} must call application services rather than domain registries"


def test_eval_domains_do_not_depend_on_web_or_fastapi() -> None:
    eval_root = ROOT / "src/lychee_mas/eval"
    domain_names = (
        "apis",
        "benchmarks",
        "contracts",
        "deployments",
        "environment",
        "evaluation",
        "experiments",
        "infrastructure",
        "models",
        "pricing",
        "scheduling",
        "teams",
    )

    for domain_name in domain_names:
        root = eval_root / domain_name
        for path in root.rglob("*.py"):
            imports = _imports(path)
            assert not any(
                item.startswith(("fastapi", "apps.eval")) for item in imports
            ), f"{path.relative_to(eval_root)} must remain transport-independent"


def test_runtime_adapters_share_result_projection() -> None:
    adapters = ROOT / "src/lychee_mas/runtime/adapters/frameworks"

    for path in (adapters / "autogen/runtime.py", adapters / "base.py"):
        source = path.read_text(encoding="utf-8")
        assert "from lychee_mas.runtime.results.projection import project_result" in source
        assert "validate_result_contract(" not in source


def test_core_runtime_and_benchmark_contracts_do_not_depend_on_studio_or_frameworks() -> None:
    for path in (
        ROOT / "src/lychee_mas/runtime/contracts/runtime.py",
        ROOT / "src/lychee_mas/eval/benchmarks/base.py",
    ):
        imports = _imports(path)
        assert not any(
            item.startswith(
                (
                    "lychee_mas.eval.application",
                    "autogen_",
                    "langgraph",
                    "crewai",
                    "fastapi",
                )
            )
            for item in imports
        ), f"{path.name} is a framework-neutral inner contract"


def test_runner_state_projection_is_separate_from_scheduler_policy() -> None:
    scheduler = (ROOT / "src/lychee_mas/eval/scheduling/manager.py").read_text(
        encoding="utf-8"
    )
    finalization = (
        ROOT / "src/lychee_mas/eval/experiments/finalization.py"
    ).read_text(encoding="utf-8")

    assert "from .runner_state import" in scheduler
    assert "remaining = remaining_work(allocation)" in scheduler
    assert "observed = observed_trial_state(allocation)" in scheduler
    assert "def _observed_trial_slot_state(" not in scheduler
    assert "def _remaining_work(" not in scheduler
    assert "finished_instance_projection(" not in scheduler
    assert "finished_instance_projection(" in finalization
    assert "def _read_run_status(" not in scheduler
    assert "RunSupervisor(" in scheduler
    assert "def _monitor(" not in scheduler


def test_run_cli_is_a_thin_process_composition_root() -> None:
    source = (ROOT / "scripts/run_mas.py").read_text(encoding="utf-8")

    assert "from lychee_mas.eval.runner import configuration as _runner_config" in source
    assert "from lychee_mas.eval.runner.cli import build_run_parser" in source
    assert "from lychee_mas.eval.runner.coordinator import run_one" in source
    assert "from lychee_mas.eval.runner.team_inspection import print_team_structure" in source
    assert "def _read_json(" not in source
    assert "def _atomic_write_json(" not in source
    assert "def _usage_summary(" not in source
    assert "async def _execute_trial(" not in source
    assert "async def run_one(" not in source
    assert "class TeeStream" not in source
    assert "ArgumentParser(" not in source
    assert "ap.add_argument(" not in source
    assert len(source.splitlines()) < 180


def test_pure_run_service_modules_remain_transport_and_backend_free() -> None:
    runner = ROOT / "src/lychee_mas/eval/runner"

    for name in (
        "cli.py",
        "configuration.py",
        "model_calls.py",
        "run_artifacts.py",
        "run_state.py",
        "trial_identity.py",
    ):
        path = runner / name
        imports = _imports(path)
        assert not any(
            item.startswith(
                (
                    "fastapi",
                    "lychee_mas.runtime.adapters.frameworks",
                    "lychee_mas.runtime.adapters.inference",
                )
            )
            for item in imports
        ), f"{path.name} must remain reusable outside HTTP and model/framework adapters"


def test_run_coordinator_delegates_trial_execution() -> None:
    source = (ROOT / "src/lychee_mas/eval/runner/coordinator.py").read_text(
        encoding="utf-8"
    )

    assert "from .backend_assembly import BackendAssemblyRequest, assemble_backend" in source
    assert "from .trial_pool import TrialPoolRequest, execute_trial_pool" in source
    assert "async def run_one(" in source
    assert "pool_result = await execute_trial_pool(" in source
    assert "OpenAICompatibleBackend(" not in source
    assert "HFBackend(" not in source
    assert "async def worker_loop(" not in source
    assert "async def _execute_trial(" not in source


def test_team_page_delegates_graph_rendering_and_force_layout() -> None:
    page = (ROOT / "apps/eval/web/src/TeamManagement.tsx").read_text(
        encoding="utf-8"
    )
    canvas = (ROOT / "apps/eval/web/src/TeamGraphCanvas.tsx").read_text(
        encoding="utf-8"
    )

    assert "import TeamGraphCanvas from './TeamGraphCanvas'" in page
    assert "<TeamGraphCanvas" in page
    assert "from '@xyflow/react'" not in page
    assert "from 'd3-force'" not in page
    assert len(page.splitlines()) < 800
    assert "from '@xyflow/react'" in canvas
    assert "from 'd3-force'" in canvas


def test_deployment_routes_delegate_business_logic_to_application_service() -> None:
    source = (ROOT / "apps/eval/server/app.py").read_text(encoding="utf-8")

    assert "DeploymentApplicationService(" in source
    assert "def registered_deployment_spec(" not in source
    assert "def deployment_source(" not in source


def test_frontend_bootstrap_is_page_scoped() -> None:
    app = (ROOT / "apps/eval/web/src/App.tsx").read_text(encoding="utf-8")
    api = (ROOT / "apps/eval/web/src/api.ts").read_text(encoding="utf-8")

    assert "api.workspace(next)" in app
    assert "lazy(() => import(" in app
    assert "/api/workspaces/" in api


def test_team_fallback_does_not_trigger_binding_report_render_loop() -> None:
    source = (ROOT / "apps/eval/web/src/TeamManagement.tsx").read_text(
        encoding="utf-8"
    )

    assert "const fallback = useMemo(() => fallbackTeam(), [])" in source
    assert "|| specs[0] || fallback" in source
    assert "|| specs[0] || fallbackTeam()" not in source
