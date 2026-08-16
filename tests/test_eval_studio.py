from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.eval.studio.benchmarks import BenchmarkRegistry
from lychee_mas.eval.studio.environment import readme_environment
from lychee_mas.eval.studio.events import read_jsonl_events
from lychee_mas.eval.studio.execution import ExecutionPlanCompiler, default_project
from lychee_mas.eval.studio.experiments import ExperimentQueueManager, ExperimentRegistry
from lychee_mas.eval.studio.jobs import JobManager
from lychee_mas.eval.studio.pricing import PricingRegistry
from lychee_mas.eval.studio.team_spec import load_team_spec
from lychee_mas.runtime.backends.deployment_pool import DeploymentPool
from lychee_mas.runtime.backends.hf_backend import HFBackend
from lychee_mas.runtime.backends.openai_api_backend import RequestRateLimiter


@pytest.fixture(autouse=True)
def _register_test_pricing(tmp_path: Path) -> None:
    registry = PricingRegistry(tmp_path)
    registry.save_spec(
        "allocated-gpu-time",
        {
            "basis": "allocated_gpu_time",
            "supported_deployment_kinds": ["hf", "vllm"],
            "required_rates": ["gpu_hour"],
            "currency": "CNY",
            "rates": {"gpu_hour": 1.0},
            "metadata": {},
        },
    )
    registry.save_spec(
        "api-token-usage",
        {
            "basis": "token_usage",
            "supported_deployment_kinds": ["api", "vllm"],
            "required_rates": ["input", "output"],
            "optional_rates": ["cached_input", "reasoning_output"],
            "currency": "CNY",
            "rates": {"input": 1.0, "output": 2.0},
            "metadata": {"model_ids": ["*"], "source_spec_ids": ["*"]},
        },
    )
    registry.instantiate_instance("allocated-gpu-time", "test-gpu-pricing")
    registry.instantiate_instance("api-token-usage", "test-token-pricing")


def _benchmark_spec(benchmark_id: str) -> dict:
    from lychee_mas.eval.benchmarks import BENCHMARKS

    benchmark = BENCHMARKS.get(benchmark_id)
    return {
        "schema_version": 4,
        "id": benchmark.id,
        "name": benchmark.name,
        "category": benchmark.category,
    }


def _register_gsm8k_benchmark(root: Path) -> None:
    benchmarks = BenchmarkRegistry(root)
    benchmarks.save_spec(_benchmark_spec("gsm8k"))
    prepared = root / "data/benchmarks/prepared/gsm8k"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "test.parquet").write_bytes(b"prepared-data")
    benchmarks.save_instance(
        {
            "id": "instance-gsm8k",
            "benchmark_spec_id": "gsm8k",
            "prepared_path": str(prepared),
            "supported_tasks": ["gsm8k"],
            "status": "ready",
        }
    )


def _register_test_api_resource(root: Path) -> None:
    spec = root / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "organization": "Test Organization",
                "deployment_kind": "api",
                "model_id": "qwen-plus",
                "default_base_url": "https://example.invalid/v1",
                "auth_modes": ["env"],
                "default_api_key_env": "MODEL_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    from lychee_mas.eval.studio.apis import APIRegistry

    APIRegistry(root).save_instance(
        {
            "id": "test-qwen",
            "api_spec_id": "test-api",
            "model_id": "qwen-plus",
            "base_url": "https://example.invalid/v1",
            "auth_mode": "env",
            "api_key_env": "MODEL_API_KEY",
        }
    )


def _register_test_model_spec(root: Path) -> None:
    spec = root / "configs/eval_studio/models/specs/test-qwen.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-qwen",
                "name": "Test Qwen",
                "organization": "Test Organization",
                "family": "Qwen",
                "parameter_size": "tiny",
                "capabilities": ["hf", "vllm", "latent"],
                "sources": {
                    "modelscope": [{"id": "test/Test-Qwen", "priority": 10}],
                    "huggingface": [{"id": "test/Test-Qwen", "priority": 20}],
                },
            }
        ),
        encoding="utf-8",
    )


def _complete_test_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def _bind_project_to(project: dict, deployment_instance_id: str) -> None:
    project["deployment_bindings"].update(
        {
            "default_deployment_instance_id": deployment_instance_id,
            "control_deployment_instance_id": deployment_instance_id,
            "role_bindings": [
                {
                    "role_id": slot["participant_id"],
                    "deployment_instance_id": deployment_instance_id,
                }
                for slot in project["team"]["inference_slots"]
                if slot["kind"] == "participant"
            ],
        }
    )


def _single_team_spec(team_id: str = "single", participant_id: str = "Solver") -> dict:
    return {
        "schema_version": 4,
        "id": team_id,
        "participants": [{"id": participant_id, "name": participant_id, "agent_type": "assistant"}],
        "group_chat": {"type": "round_robin"},
        "termination": {"conditions": []},
        "extensions": {},
    }


def _instance_record(
    instance_id: str,
    deployment_spec_id: str,
    *,
    kind: str,
    model_id: str,
    source_type: str = "model",
    source_id: str = "test-source",
    **options,
) -> dict:
    actual_pricing_instance_id = (
        "test-token-pricing"
        if source_type == "api" or options.get("managed") is False
        else "test-gpu-pricing"
    )
    return {
        "schema_version": 2,
        "id": instance_id,
        "deployment_spec_id": deployment_spec_id,
        "source_instance": {"type": source_type, "id": source_id},
        "kind": kind,
        "model_id": model_id,
        "actual_pricing_instance_id": actual_pricing_instance_id,
        **options,
    }


def test_hf_generation_budget_is_clamped_to_remaining_context() -> None:
    backend = HFBackend.__new__(HFBackend)
    backend.context_window = 100
    backend.do_sample = False
    backend.repetition_penalty = 1.0
    backend.max_repeated_token_run = 0
    backend.tok = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 1})()

    kwargs = backend._gen_kwargs(50, prompt_len=90, context_len=90)

    assert kwargs["max_new_tokens"] == 10


def test_hf_generation_applies_explicit_sampling_overrides() -> None:
    backend = HFBackend.__new__(HFBackend)
    backend.context_window = 100
    backend.do_sample = False
    backend.temperature = 0.7
    backend.top_p = 0.8
    backend.top_k = None
    backend.min_p = None
    backend.presence_penalty = None
    backend.repetition_penalty = 1.0
    backend.max_repeated_token_run = 0
    backend.tok = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 1})()

    kwargs = backend._gen_kwargs(
        20,
        prompt_len=5,
        request_overrides={
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        },
    )

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 20
    assert kwargs["min_p"] == 0.0
    assert "repetition_penalty" not in kwargs


def test_process_rate_limiter_waits_between_request_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lychee_mas.runtime.backends import openai_api_backend

    clock = [10.0]
    monkeypatch.setattr(openai_api_backend.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        openai_api_backend.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    limiter = RequestRateLimiter(key="test", min_interval_s=2.0)
    limiter._last_start = 9.0

    with limiter.slot():
        pass

    assert clock[0] == 11.0
    assert limiter._last_start == 11.0


def test_execution_plan_generates_complete_shell_bundle(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["environment"].update(
        {
            "python": "/usr/bin/python3",
            "conda_sh": "",
            "conda_env": "",
        }
    )
    project["runtime"].update(
        {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    )
    compiler = ExecutionPlanCompiler(tmp_path)
    plan = compiler.compile(project, launch_id="unit-plan")
    compiler.materialize(plan)

    expected = {
        "prepare.sh",
        "verify_deployment_instances.sh",
        "run.sh",
        "resume.sh",
        "analyze.sh",
        "run_all.sh",
        "project.json",
        "team_spec.json",
        "deployments.json",
        "runtime_config.yaml",
        "launch_manifest.json",
    }
    assert expected <= set(plan.files)
    assert "PYTHONPATH" in plan.files["run.sh"]
    assert "scripts/run_mas.py" in plan.files["run.sh"]
    assert "scripts/analyze_benchmark_run.py" in plan.files["analyze.sh"]
    assert "--team-spec" in plan.files["run.sh"]
    assert "--deployment-config" in plan.files["run.sh"]
    assert "--trace-detail-level" in plan.files["run.sh"]
    assert "--run-dir" in plan.files["run.sh"]
    assert str(plan.run_dir) in plan.files["run.sh"]
    assert plan.run_dir.parent == (
        tmp_path / "runs/benchmarks/gsm8k/gsm8k/reason/none/eval-experiment"
    )
    assert re.fullmatch(r"\d{8}T\d{9}Z", plan.run_dir.name)
    assert '"trace_detail_level": "compact"' in plan.files["runtime_config.yaml"]
    runtime_config = json.loads(plan.files["runtime_config.yaml"])
    assert runtime_config["runtime"]["max_turns"] == 12
    assert runtime_config["runtime"]["code_timeout"] == 60
    assert runtime_config["runtime"]["work_root"] == "runs/lychee_tool_workspaces"
    assert runtime_config["runtime"]["web_headless"] is True
    assert runtime_config["runtime"]["save_screenshots"] is False
    assert runtime_config["run"]["max_model_calls_per_case"] is None
    assert runtime_config["run"]["max_case_retries"] == 0
    assert runtime_config["run"]["on_case_error"] == "continue"
    assert runtime_config["backend"]["do_sample"] is True
    assert runtime_config["backend"]["temperature"] == 1.0
    assert runtime_config["backend"]["top_p"] == 0.95
    assert runtime_config["backend"]["top_k"] == 20
    assert runtime_config["backend"]["min_p"] == 0.0
    assert runtime_config["backend"]["presence_penalty"] == 0.0
    assert runtime_config["backend"]["repetition_penalty"] == 1.0
    assert runtime_config["observability"] == {
        "collect_vllm_metrics": True,
        "vllm_metrics_interval_s": 5.0,
    }
    assert "--collect-vllm-metrics" in plan.files["run.sh"]
    assert "--vllm-metrics-interval-s \\\n  5.0" in plan.files["run.sh"]
    assert "--max-model-calls-per-case \\\n  unlimited" in plan.files["run.sh"]
    assert "--max-case-retries \\\n  0" in plan.files["run.sh"]
    assert "--on-case-error \\\n  continue" in plan.files["run.sh"]
    assert "--code-timeout \\\n  60" in plan.files["run.sh"]
    assert "--work-root \\\n  runs/lychee_tool_workspaces" in plan.files["run.sh"]
    run_script = plan.files["run.sh"]
    for option, value in (
        ("--top-k", "20"),
        ("--min-p", "0.0"),
        ("--presence-penalty", "0.0"),
        ("--repetition-penalty", "1.0"),
    ):
        assert option in run_script
        assert value in run_script
    for name in expected:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", plan.launch_dir / name], check=True)


def test_dynamic_group_chat_uses_max_turns_and_model_call_limit(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["team"]["group_chat"] = {"type": "selector"}
    project["runtime"].update(
        max_rounds=99,
        max_turns=27,
        max_model_calls_per_case=41,
        max_case_retries=2,
        on_case_error="fail-fast",
    )

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="dynamic-limits")
    runtime_config = json.loads(plan.files["runtime_config.yaml"])

    assert plan.project["runtime"]["effective_max_turns"] == 27
    assert runtime_config["runtime"]["max_turns"] == 27
    assert runtime_config["run"]["max_model_calls_per_case"] == 41
    assert runtime_config["run"]["max_case_retries"] == 2
    assert runtime_config["run"]["on_case_error"] == "fail-fast"
    assert "--max-turns \\\n  27" in plan.files["run.sh"]
    assert "--max-model-calls-per-case \\\n  41" in plan.files["run.sh"]
    assert "--max-case-retries \\\n  2" in plan.files["run.sh"]
    assert "--on-case-error \\\n  fail-fast" in plan.files["run.sh"]


def test_execution_plan_enables_nounset_after_conda_activation(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["environment"].update(
        {
            "python": "/usr/bin/python3",
            "conda_sh": "/opt/conda/etc/profile.d/conda.sh",
            "conda_env": "benchmark",
        }
    )

    script = ExecutionPlanCompiler(tmp_path).compile(project).files["run.sh"]

    assert script.index("set -eo pipefail") < script.index('conda activate "$CONDA_ENV"')
    assert script.index('conda activate "$CONDA_ENV"') < script.index("set -u")


def test_execution_plan_rejects_unknown_trace_detail_level(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["runtime"]["trace_detail_level"] = "verbose"
    with pytest.raises(ValueError, match="unsupported trace_detail_level"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_experiment_model_backend_proxy_is_applied_to_api_instances(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"] = [
        {
            "id": "instance-api",
            "spec_id": "api",
            "kind": "api",
            "model_id": "remote-model",
            "base_url": "https://provider.example/v1",
            "auth_mode": "none",
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    project["deployment_bindings"].update(
        default_deployment_instance_id="instance-api",
        control_deployment_instance_id="instance-api",
    )
    for binding in project["deployment_bindings"]["role_bindings"]:
        binding["deployment_instance_id"] = "instance-api"
    project["network"] = {
        "mode": "proxy",
        "proxy_url": "http://127.0.0.1:7897",
        "targets": {"model_backend": True},
    }

    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)

    assert normalized["deployment_instances"][0]["proxy_url"] == ("http://127.0.0.1:7897")
    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="api-proxy")
    deployment_config = json.loads(plan.files["deployments.json"])
    assert deployment_config["deployments"][0]["proxy_url"] == ("http://127.0.0.1:7897")
    assert "curl --proxy http://127.0.0.1:7897" in plan.files["verify_deployment_instances.sh"]


def test_execution_plan_exports_selected_benchmark_instance_path(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    external = tmp_path.parent / "external-gsm8k"
    project["benchmark"].update(
        {
            "prepared_path": str(external),
        }
    )
    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="external-data")
    assert "LYCHEE_BENCHMARK_PREPARED_OVERRIDES" in plan.files["run.sh"]
    assert str(external) in plan.files["run.sh"]


def test_vllm_and_api_are_rejected_for_latent() -> None:
    project = default_project("/tmp/lychee-studio-test")
    project["runtime"]["method"] = "both"
    project["deployment_instances"] = [
        {
            "id": "instance-vllm",
            "kind": "vllm",
            "model_id": "model",
            "base_url": "http://127.0.0.1:8000/v1",
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    _bind_project_to(project, "instance-vllm")
    with pytest.raises(ValueError, match="require HF DeploymentInstances"):
        ExecutionPlanCompiler("/tmp/lychee-studio-test").normalize(project)


def test_latent_requires_one_shared_hf_deployment(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["runtime"]["method"] = "latent_only"
    project["deployment_instances"].append(
        {
            "id": "instance-second-hf",
            "kind": "hf",
            "model_id": "second",
            "model_path": str(tmp_path / "models/second"),
            "status": "ready_on_run",
            "available": True,
            "actual_pricing_instance_id": "test-gpu-pricing",
        }
    )
    with pytest.raises(ValueError, match="one shared HF DeploymentInstance"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_multiple_hf_deployments_share_one_process_cuda_namespace(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["runtime"]["method"] = "none"
    project["deployment_instances"].append(
        {
            "id": "instance-second-hf",
            "kind": "hf",
            "model_id": "second",
            "model_path": str(tmp_path / "models/second"),
            "cuda_visible_devices": "5",
            "device": "cuda:0",
            "status": "ready_on_run",
            "available": True,
            "actual_pricing_instance_id": "test-gpu-pricing",
        }
    )
    project["deployment_instances"][0]["cuda_visible_devices"] = "4"
    project["deployment_bindings"]["role_bindings"][1]["deployment_instance_id"] = (
        "instance-second-hf"
    )

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="multi-hf")
    deployment_config = json.loads(plan.files["deployments.json"])
    by_id = {item["id"]: item for item in deployment_config["deployments"]}
    assert by_id["instance-local-hf"]["runtime_device"] == "cuda:0"
    assert by_id["instance-second-hf"]["runtime_device"] == "cuda:1"
    assert "CUDA_VISIBLE_DEVICES=4,5" in plan.files["run.sh"]


def test_hf_deployments_reject_overlapping_physical_gpu_allocations(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"].append(
        {
            "id": "instance-second-hf",
            "kind": "hf",
            "model_id": "second",
            "model_path": str(tmp_path / "models/second"),
            "cuda_visible_devices": "0",
            "status": "ready_on_run",
            "available": True,
            "actual_pricing_instance_id": "test-gpu-pricing",
        }
    )
    with pytest.raises(ValueError, match="cannot share physical CUDA device"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_vllm_defaults_and_environment_name_validation(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"] = [
        {
            "id": "instance-local-vllm",
            "kind": "vllm",
            "model_id": "model",
            "port": 8123,
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    _bind_project_to(project, "instance-local-vllm")
    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)
    deployment = normalized["deployment_instances"][0]
    assert deployment["base_url"] == "http://127.0.0.1:8123/v1"
    assert deployment["api_key_env"] == "VLLM_API_KEY"

    project["deployment_instances"][0]["api_key_env"] = "BAD-NAME; echo unsafe"
    with pytest.raises(ValueError, match="invalid api_key_env"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_execution_plan_preserves_vllm_instance_options_without_restarting_it(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"] = [
        {
            "id": "instance-qwen-vllm",
            "kind": "vllm",
            "model_id": "Qwen3.6-27B",
            "base_url": "http://127.0.0.1:6100/v1",
            "port": 6100,
            "max_num_seqs": 8,
            "prompt_tokens_details_mode": "auto",
            "reasoning_parser": "qwen3",
            "reasoning_config": {
                "reasoning_start_str": "<think>",
                "reasoning_end_str": "Stop reasoning now.</think>",
            },
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
            "enable_prefix_caching": True,
            "library_paths": ["/opt/conda/lib"],
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    _bind_project_to(project, "instance-qwen-vllm")

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="vllm-options")
    runtime_config = json.loads(plan.files["deployments.json"])
    deployment = runtime_config["deployments"][0]
    assert deployment["max_num_seqs"] == 8
    assert deployment["prompt_tokens_details_mode"] == "auto"
    assert deployment["reasoning_parser"] == "qwen3"
    assert deployment["reasoning_config"]["reasoning_end_str"].endswith("</think>")
    assert deployment["tool_call_parser"] == "qwen3_coder"
    assert "vllm.entrypoints.openai.api_server" not in plan.files["verify_deployment_instances.sh"]


def test_execution_allows_custom_runs_root_but_keeps_asset_roots_inside_repository(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    custom_runs = tmp_path.parent / f"{tmp_path.name}-custom-runs"
    project["environment"]["runs_root"] = str(custom_runs)
    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)
    assert normalized["environment"]["runs_root"] == str(custom_runs.resolve())

    project["environment"]["models_root"] = str(tmp_path.parent / "outside-models")
    with pytest.raises(ValueError, match="models_root must stay inside"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_execution_plan_uses_exact_custom_run_dir_for_run_resume_and_analysis(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    custom_run_dir = tmp_path.parent / f"{tmp_path.name}-exact-run" / "gaia-validation"
    project["environment"]["run_dir"] = str(custom_run_dir)

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="exact-run-dir")

    assert plan.run_dir == custom_run_dir.resolve()
    assert f"LYCHEE_BENCHMARK_RUN_DIR={custom_run_dir.resolve()}" in plan.files["run.sh"]
    assert '"$LYCHEE_BENCHMARK_RUN_DIR"' in plan.files["run.sh"]
    assert '"$LYCHEE_BENCHMARK_RUN_DIR"' in plan.files["resume.sh"]
    assert f"RUN_DIR={custom_run_dir.resolve()}" in plan.files["analyze.sh"]


def test_exact_run_dir_belongs_to_experiment_instance_not_spec(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    assert "run_dir" not in spec["environment"]
    assert "name" not in spec

    spec["id"] = "custom-run-dir"
    spec["name"] = "obsolete display name"
    spec["environment"]["run_dir"] = "runs/custom/exact"
    saved = registry.save_spec("custom-run-dir", spec)
    instance = registry.create_instance(
        instance_id="custom-run",
        spec_id="custom-run-dir",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir="runs/custom/exact",
    )

    assert "run_dir" not in saved["environment"]
    assert "name" not in saved
    assert "execution" not in saved
    assert instance["run_dir"] == "runs/custom/exact"


def test_experiment_spec_accepts_all_cases_and_validates_samples(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "all-cases"
    spec["benchmark"]["cases"] = "ALL"
    spec["runtime"]["samples"] = 3

    saved = registry.save_spec("all-cases", spec)

    assert saved["benchmark"]["cases"] == "all"
    assert saved["runtime"]["samples"] == 3
    assert saved["observability"] == {
        "collect_vllm_metrics": True,
        "vllm_metrics_interval_s": 5.0,
    }

    spec["runtime"]["samples"] = 0
    with pytest.raises(ValueError, match="runtime.samples must be positive"):
        registry.save_spec("all-cases", spec)


def test_gaia_smoke_specs_only_change_case_count() -> None:
    specs_root = (
        Path(__file__).resolve().parents[1] / "configs" / "eval_studio" / "experiments" / "specs"
    )
    full = json.loads((specs_root / "gaia-qwen36.json").read_text(encoding="utf-8"))

    def comparable(spec: dict) -> dict:
        normalized = json.loads(json.dumps(spec))
        normalized["id"] = "gaia-qwen36"
        normalized["benchmark"]["cases"] = "all"
        normalized["notes"] = full["notes"]
        normalized["updated_at_utc"] = full["updated_at_utc"]
        return normalized

    for filename, expected_cases in (
        ("gaia-qwen36-smoke.json", 5),
        ("gaia-vllm-cost-smoke.json", 1),
    ):
        smoke = json.loads((specs_root / filename).read_text(encoding="utf-8"))
        assert smoke["benchmark"]["cases"] == expected_cases
        assert comparable(smoke) == full


def test_saved_custom_runs_root_is_visible_through_runs_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    custom_root = tmp_path.parent / f"{tmp_path.name}-custom-runs"
    run_dir = custom_root / "model/team_none/gsm8k"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")
    (run_dir / "spans.jsonl").write_text('{"event":"case_start"}\n', encoding="utf-8")
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "custom-output"
    spec["environment"]["runs_root"] = str(custom_root)
    registry.save_spec("custom-output", spec)

    client = TestClient(create_app(tmp_path))
    rows = client.get("/api/runs").json()
    row = next(item for item in rows if item["run_dir"] == str(run_dir.resolve()))
    assert row["runs_root"] == str(custom_root.resolve())
    assert row["id"].startswith("external.")
    events = client.get(f"/api/runs/{row['id']}/events").json()
    assert events["events"][0]["event"] == "case_start"


def test_custom_team_spec_preserves_edges_without_deployments(tmp_path: Path) -> None:
    path = tmp_path / "team.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "id": "custom",
                "group_chat": {
                    "type": "selector",
                    "selector_func_factory": "topology_selector",
                },
                "participants": [
                    {"id": "Planner"},
                    {"id": "Solver"},
                ],
                "termination": {"conditions": []},
                "extensions": {
                    "topology": {
                        "topology_family": "linear",
                        "edges": {"Planner": ["Solver"]},
                        "speaking_order": ["Planner", "Solver"],
                    },
                    "context_visibility": {"type": "topology_filtered"},
                },
            }
        ),
        encoding="utf-8",
    )
    graph, _ = load_team_spec(path, rounds=3)
    assert graph.names == ["Planner", "Solver"]
    assert graph.edges == {"Planner": ["Solver"], "Solver": []}
    assert graph.nodes[1].meta["receives_from"] == ["Planner"]
    assert "deployment_id" not in graph.nodes[1].meta
    assert "control_deployment_id" not in graph.meta
    assert graph.meta["group_chat"] == {
        "type": "selector",
        "selector_func_factory": "topology_selector",
        "allow_repeated_speaker": False,
        "max_selector_attempts": 3,
        "model_client_streaming": False,
    }
    assert graph.meta["dynamic_topology"] is False
    assert graph.meta["context_visibility"] == "topology_filtered"


def test_deployment_pool_reuses_one_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = DeploymentPool(
        {
            "control_deployment_id": "shared",
            "deployments": [{"id": "shared", "kind": "api", "model_id": "model"}],
        }
    )
    instance = object()
    monkeypatch.setattr(pool, "_build", lambda _config: instance)
    assert pool.resolve("shared") is instance
    assert pool.resolve("shared") is instance
    assert pool.model_id("shared") == "model"


def test_team_deployment_bindings_are_applied_only_at_runtime() -> None:
    from lychee_mas.layers.construct.templates import RoleProfileTeamBuilder

    graph = RoleProfileTeamBuilder(team="reason", rounds=2).build()
    pool = DeploymentPool(
        {
            "deployments": [
                {"id": "default", "kind": "api", "model_id": "model-a"},
                {"id": "solver", "kind": "api", "model_id": "model-b"},
            ],
            "team_deployment_bindings": {
                "default_deployment_id": "default",
                "control_deployment_id": "default",
                "role_bindings": [{"role_id": "solver", "deployment_id": "solver"}],
            },
        }
    )
    assert all("deployment_id" not in node.meta for node in graph.nodes)
    pool.bind_graph(graph)
    assert [node.meta["deployment_id"] for node in graph.nodes] == [
        "default",
        "solver",
        "default",
    ]
    assert graph.meta["control_deployment_id"] == "default"


def test_deployment_pool_resolves_role_and_control_generation_budgets() -> None:
    pool = DeploymentPool(
        {
            "deployments": [{"id": "default", "kind": "api", "model_id": "model"}],
            "team_deployment_bindings": {
                "default_deployment_id": "default",
                "control_deployment_id": "default",
                "control_generation_overrides": {"max_new_tokens": 4096},
                "role_bindings": [
                    {
                        "role_id": "solver",
                        "deployment_id": "default",
                        "generation_overrides": {"max_new_tokens": 8192},
                    }
                ],
            },
        }
    )
    assert pool.max_new_tokens_for_role("solver", 1024) == 8192
    assert pool.max_new_tokens_for_role("reviewer", 1024) == 1024
    assert pool.control_max_new_tokens(1024) == 4096


def test_deployment_pool_translates_one_qwen_deployment_per_call() -> None:
    pool = DeploymentPool(
        {
            "deployments": [
                {
                    "id": "qwen",
                    "kind": "vllm",
                    "model_id": "Qwen3.6-27B",
                    "thinking_protocol": "qwen_chat_template",
                    "capabilities": {"native_thinking_budget": True},
                }
            ],
            "team_deployment_bindings": {
                "default_deployment_id": "qwen",
                "control_deployment_id": "qwen",
                "role_bindings": [
                    {
                        "role_id": "Coder",
                        "deployment_id": "qwen",
                        "generation_overrides": {
                            "thinking_mode": "enabled",
                            "preserve_thinking": False,
                            "max_thinking_budget_tokens": 2048,
                            "temperature": 0.6,
                            "top_k": 20,
                        },
                    }
                ],
                "control_generation_overrides": {
                    "thinking_mode": "disabled",
                    "do_sample": False,
                    "temperature": 0.7,
                    "presence_penalty": 1.5,
                },
            },
        }
    )

    assert pool.deployment_for_role("Coder") == "qwen"
    assert pool.control_deployment_id == "qwen"
    assert pool.invocation_overrides_for_role("Coder") == {
        "temperature": 0.6,
        "token_budget_policy": {"max_thinking_budget_tokens": 2048},
        "extra_body": {
            "top_k": 20,
            "thinking_token_budget": 2048,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": False,
            },
        },
    }
    assert pool.control_invocation_overrides() == {
        "do_sample": False,
        "temperature": 0.7,
        "extra_body": {
            "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_deployment_pool_applies_global_sampling_policy_to_vllm_calls() -> None:
    pool = DeploymentPool(
        {
            "deployments": [
                {
                    "id": "qwen",
                    "kind": "vllm",
                    "model_id": "Qwen3.6-27B",
                    "thinking_protocol": "qwen_chat_template",
                }
            ],
            "team_deployment_bindings": {
                "default_deployment_id": "qwen",
                "control_deployment_id": "qwen",
                "role_bindings": [{"role_id": "Coder", "deployment_id": "qwen"}],
            },
        },
        generation={
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    )

    expected = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    }
    assert pool.invocation_overrides_for_role("Coder") == expected
    assert pool.control_invocation_overrides() == expected


def test_deployment_pool_translates_deepseek_thinking_toggle() -> None:
    pool = DeploymentPool(
        {
            "deployments": [
                {
                    "id": "deepseek",
                    "kind": "api",
                    "model_id": "deepseek-v4-pro",
                    "base_url": "https://api.deepseek.com",
                    "thinking_protocol": "deepseek",
                }
            ],
            "team_deployment_bindings": {
                "default_deployment_id": "deepseek",
                "control_deployment_id": "deepseek",
                "role_bindings": [
                    {
                        "role_id": "Solver",
                        "deployment_id": "deepseek",
                        "generation_overrides": {"thinking_mode": "enabled"},
                    }
                ],
                "control_generation_overrides": {"thinking_mode": "disabled"},
            },
        }
    )

    assert pool.invocation_overrides_for_role("Solver") == {
        "extra_body": {"thinking": {"type": "enabled"}}
    }
    assert pool.control_invocation_overrides() == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_local_qwen_rejects_undeclared_native_thinking_budget() -> None:
    pool = DeploymentPool(
        {
            "deployments": [
                {
                    "id": "qwen",
                    "kind": "vllm",
                    "model_id": "Qwen3.6-27B",
                    "thinking_protocol": "qwen_chat_template",
                }
            ],
            "team_deployment_bindings": {
                "default_deployment_id": "qwen",
                "role_bindings": [
                    {
                        "role_id": "Coder",
                        "deployment_id": "qwen",
                        "generation_overrides": {"max_thinking_budget_tokens": 2048},
                    }
                ],
            },
        }
    )
    with pytest.raises(ValueError, match="does not declare native thinking budget"):
        pool.invocation_overrides_for_role("Coder")


def test_studio_rejects_local_qwen_budget_before_launch(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"] = [
        {
            "id": "instance-qwen",
            "kind": "vllm",
            "model_id": "Qwen3.6-27B",
            "base_url": "http://127.0.0.1:6100/v1",
            "thinking_protocol": "qwen_chat_template",
            "capabilities": {
                "native_thinking_budget": False,
                "thinking_toggle": True,
            },
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    project["team"]["participants"] = [{"id": "Solver", "name": "Solver"}]
    project["team"]["inference_slots"] = [
        {"id": "Solver", "kind": "participant", "participant_id": "Solver"}
    ]
    project["deployment_bindings"] = {
        "default_deployment_instance_id": "instance-qwen",
        "control_deployment_instance_id": "instance-qwen",
        "role_bindings": [
            {
                "role_id": "Solver",
                "deployment_instance_id": "instance-qwen",
                "generation_overrides": {
                    "thinking_mode": "enabled",
                    "max_thinking_budget_tokens": 2048,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="does not declare native thinking budget"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_studio_accepts_declared_vllm_native_thinking_budget(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["deployment_instances"] = [
        {
            "id": "instance-qwen",
            "kind": "vllm",
            "model_id": "Qwen3.6-27B",
            "base_url": "http://127.0.0.1:6100/v1",
            "thinking_protocol": "qwen_chat_template",
            "capabilities": {
                "native_thinking_budget": True,
                "thinking_toggle": True,
            },
            "status": "running",
            "available": True,
            "actual_pricing_instance_id": "test-token-pricing",
        }
    ]
    project["team"]["participants"] = [{"id": "Solver", "name": "Solver"}]
    project["team"]["inference_slots"] = [
        {"id": "Solver", "kind": "participant", "participant_id": "Solver"}
    ]
    project["deployment_bindings"] = {
        "default_deployment_instance_id": "instance-qwen",
        "control_deployment_instance_id": "instance-qwen",
        "role_bindings": [
            {
                "role_id": "Solver",
                "deployment_instance_id": "instance-qwen",
                "generation_overrides": {
                    "thinking_mode": "enabled",
                    "max_thinking_budget_tokens": 2048,
                },
            }
        ],
    }

    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)
    overrides = normalized["deployment_bindings"]["role_bindings"][0]["generation_overrides"]
    assert overrides["max_thinking_budget_tokens"] == 2048


def test_obsolete_project_schema_is_rejected(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["schema_version"] = 1
    project["team"] = {
        "id": "custom",
        "profile": None,
        "scheduler": "topology",
        "control_deployment_id": "local-hf",
        "nodes": [{"id": "Planner", "deployment_id": "local-hf"}],
        "edges": {},
    }
    with pytest.raises(ValueError, match="schema_version 1"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_group_chat_rejects_mixed_or_irrelevant_selection_fields(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["team"]["group_chat"] = {
        "type": "selector",
        "selector_func_factory": "topology_selector",
        "candidate_func_factory": "topology_candidates",
    }
    with pytest.raises(ValueError, match="one selection strategy"):
        ExecutionPlanCompiler(tmp_path).normalize(project)

    project["team"]["group_chat"] = {
        "type": "round_robin",
        "selector_func_factory": "topology_selector",
    }
    with pytest.raises(ValueError, match="does not support fields"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_incremental_event_reader(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    full_payload = "x" * 5000
    path.write_text(
        json.dumps({"span_type": "a", "provider_response_payload": full_payload})
        + '\n{"span_type":"b"}\n',
        encoding="utf-8",
    )
    first = read_jsonl_events(path, start_line=0, limit=1)
    second = read_jsonl_events(path, start_line=first["next_line"], limit=10)
    assert [event["span_type"] for event in first["events"]] == ["a"]
    assert [event["span_type"] for event in second["events"]] == ["b"]
    assert first["events"][0]["provider_response_payload"] == full_payload
    assert first["start_line"] == 0
    assert first["next_line"] == 1
    assert first["total_lines"] == 2
    assert first["has_more"] is True
    assert second["total_lines"] == 2
    assert second["has_more"] is False


def test_studio_api_health(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    client = TestClient(create_app(tmp_path))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["repo_root"] == str(tmp_path)


def test_studio_api_reads_group_chat_separately_from_spans(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    run_dir = tmp_path / "runs/benchmarks/model/team/task"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "task": "task"}), encoding="utf-8"
    )
    (run_dir / "spans.jsonl").write_text('{"span_type":"model_call_start"}\n', encoding="utf-8")
    (run_dir / "group_chat.jsonl").write_text(
        '{"event_type":"message","source":"Coder","content":"hello"}\n',
        encoding="utf-8",
    )

    client = TestClient(create_app(tmp_path))
    row = client.get("/api/runs").json()[0]
    assert row["has_group_chat"] is True
    assert row["group_chat_bytes"] > 0
    group_events = client.get(f"/api/runs/{row['id']}/group-chat").json()["events"]
    span_events = client.get(f"/api/runs/{row['id']}/events").json()["events"]
    assert group_events == [{"event_type": "message", "source": "Coder", "content": "hello"}]
    assert span_events == [{"span_type": "model_call_start"}]


def test_readme_environment_is_the_studio_default(tmp_path: Path) -> None:
    environment = readme_environment(tmp_path)
    project = default_project(tmp_path)
    assert environment["conda_env"] == "LycheeMAS"
    assert environment["python"] == str(tmp_path / ".venv/bin/python")
    assert project["environment"]["python"] == environment["python"]


def test_job_progress_reads_explicit_percent(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/progress"
    launch_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps({"launch_id": "progress", "status": "completed", "pid": None}),
        encoding="utf-8",
    )
    (launch_dir / "launch.log").write_text(
        "download 12%\n[progress] percent=68 bytes=68/100\n", encoding="utf-8"
    )
    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)
    assert progress["percent"] == 100
    assert progress["message"].startswith("[progress] percent=68")


def test_job_progress_reads_benchmark_case_fraction(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/progress-cases"
    launch_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps({"launch_id": "progress-cases", "status": "running", "pid": None}),
        encoding="utf-8",
    )
    (launch_dir / "launch.log").write_text(
        "[1/10] start case_id=a\n[3/10] score=1.00\n", encoding="utf-8"
    )
    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)
    assert progress["percent"] == 30


def test_job_progress_does_not_count_started_case_as_completed(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/progress-running-cases"
    launch_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps({"launch_id": "progress-running-cases", "status": "running", "pid": None}),
        encoding="utf-8",
    )
    (launch_dir / "launch.log").write_text(
        "[1/2] start case_id=a\n[1/2] msgs=4 ans='ok'\n[2/2] start case_id=b\n",
        encoding="utf-8",
    )

    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)

    assert progress["percent"] == 50


def test_job_progress_prefers_persisted_run_status_and_reports_active_cases(
    tmp_path: Path,
) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/persisted-progress"
    run_dir = tmp_path / "runs/benchmarks/persisted-progress"
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps(
            {
                "launch_id": "persisted-progress",
                "status": "running",
                "pid": None,
                "run_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    (launch_dir / "launch.log").write_text(
        "[model:solver] start turn=2 sender=planner messages=3\n",
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "expected_predictions": 10,
                "successful_predictions": 3,
                "error_predictions": 1,
                "skipped_predictions": 0,
                "effective_case_concurrency": 2,
                "started_at_unix_s": 100.0,
                "finished_at_unix_s": 112.5,
                "active_cases": [
                    {
                        "worker_id": 1,
                        "case_order": 5,
                        "num_cases": 10,
                        "case_id": "case-5",
                        "k_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)

    assert progress["percent"] == 40
    assert progress["completed_predictions"] == 4
    assert progress["remaining_predictions"] == 6
    assert progress["case_concurrency"] == 2
    assert progress["current_cases"][0]["case_id"] == "case-5"
    assert progress["current_activity"] == {
        "role": "solver",
        "phase": "start",
        "turn": 2,
        "sender": "planner",
    }
    assert progress["elapsed_s"] == 12.5


def test_experiment_instance_observation_recovers_terminal_job_state(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    instance_root = tmp_path / "configs/eval_studio/experiments/instances"
    instance_root.mkdir(parents=True)
    launch_dir = tmp_path / "runs/eval_studio/launches/recovered"
    instance = {
        "schema_version": 3,
        "id": "recovered",
        "experiment_spec_id": "experiment",
        "benchmark_instance_id": "benchmark",
        "team_instance_id": "team",
        "launcher": {"type": "subprocess"},
        "status": "running",
        "launch_id": "recovered",
        "launch_dir": str(launch_dir),
        "run_dir": str(tmp_path / "runs/benchmarks/recovered"),
        "queue": {"priority": 100},
    }
    (instance_root / "recovered.json").write_text(json.dumps(instance), encoding="utf-8")

    class Jobs:
        @staticmethod
        def progress(_launch_dir, *, tail):
            assert tail == 12
            return {
                "state": {"status": "completed", "return_code": 0},
                "percent": 100,
                "completed_predictions": 2,
                "expected_predictions": 2,
                "lines": ["done"],
            }

    manager = ExperimentQueueManager(registry, None, Jobs(), None)
    observed = manager.instances()[0]

    assert observed["status"] == "completed"
    assert observed["progress"]["percent"] == 100
    assert registry.get_instance("recovered")["status"] == "completed"


def test_experiment_progress_failure_does_not_replace_lifecycle_status(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    instance_root = tmp_path / "configs/eval_studio/experiments/instances"
    instance_root.mkdir(parents=True)
    launch_dir = tmp_path / "runs/eval_studio/launches/running"
    instance = {
        "schema_version": 3,
        "id": "running",
        "experiment_spec_id": "experiment",
        "benchmark_instance_id": "benchmark",
        "team_instance_id": "team",
        "launcher": {"type": "subprocess"},
        "status": "running",
        "launch_id": "running",
        "launch_dir": str(launch_dir),
        "run_dir": str(tmp_path / "runs/benchmarks/running"),
        "queue": {"priority": 100},
    }
    (instance_root / "running.json").write_text(json.dumps(instance), encoding="utf-8")

    class Jobs:
        @staticmethod
        def progress(_launch_dir, *, tail):
            raise OSError("transient observation failure")

    observed = ExperimentQueueManager(registry, None, Jobs(), None).instances()[0]

    assert observed["status"] == "running"
    assert observed["progress"]["state"]["status"] == "running"
    assert observed["progress"]["observation_status"] == "unavailable"
    assert "transient observation failure" in observed["progress"]["message"]


def test_job_state_atomic_writes_use_independent_temporary_files(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/concurrent"
    launch_dir.mkdir(parents=True)
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            barrier.wait()
            JobManager._write_state(
                launch_dir,
                {"launch_id": "concurrent", "status": "running", "revision": index},
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [threading.Thread(target=write, args=(index,)) for index in range(16)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    state = json.loads((launch_dir / "job.json").read_text(encoding="utf-8"))
    assert state["launch_id"] == "concurrent"
    assert state["status"] == "running"
    assert 0 <= state["revision"] < 16
    assert list(launch_dir.glob(".job.*.tmp")) == []


def test_mark_run_stopped_preserves_interrupted_case_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs/benchmarks/interrupted"
    run_dir.mkdir(parents=True)
    status_path = run_dir / "run_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "expected_predictions": 10,
                "successful_predictions": 2,
                "active_cases": [{"case_id": "case-3", "worker_id": 0}],
            }
        ),
        encoding="utf-8",
    )

    JobManager._mark_run_stopped({"run_dir": str(run_dir)})

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    assert status["active_cases"] == []
    assert status["interrupted_active_cases"] == [{"case_id": "case-3", "worker_id": 0}]
    assert status["successful_predictions"] == 2
    assert status["finished_at_unix_s"] >= status["updated_at_unix_s"]
    assert list(run_dir.glob(".run_status.*.tmp")) == []


def test_stopped_job_does_not_reconstruct_active_cases_from_old_log(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/stopped"
    run_dir = tmp_path / "runs/benchmarks/stopped"
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps(
            {
                "launch_id": "stopped",
                "status": "stopped",
                "run_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    (launch_dir / "launch.log").write_text(
        "  [3/10] start case_id=case-3\n",
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "stopped", "active_cases": []}),
        encoding="utf-8",
    )

    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)

    assert progress["run_status"] == "stopped"
    assert progress["current_cases"] == []


@pytest.mark.parametrize(("code", "status"), [(0, "completed"), (7, "failed")])
def test_job_status_uses_launcher_exit_marker(tmp_path: Path, code: int, status: str) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/managed"
    launch_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps(
            {
                "launch_id": "managed",
                "mode": "new_tmux_session",
                "status": "running",
                "pid": None,
                "tmux_target": "nightly:benchmark",
            }
        ),
        encoding="utf-8",
    )
    (launch_dir / "exit_code").write_text(str(code), encoding="utf-8")

    state = JobManager(ExecutionPlanCompiler(tmp_path)).status(launch_dir)

    assert state["status"] == status
    assert state["return_code"] == code


def test_job_status_does_not_overwrite_stopped_with_exit_marker(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/stopped"
    launch_dir.mkdir(parents=True)
    (launch_dir / "job.json").write_text(
        json.dumps(
            {
                "launch_id": "stopped",
                "mode": "subprocess",
                "status": "stopped",
                "pid": None,
                "tmux_target": None,
            }
        ),
        encoding="utf-8",
    )
    (launch_dir / "exit_code").write_text("143", encoding="utf-8")

    state = JobManager(ExecutionPlanCompiler(tmp_path)).status(launch_dir)

    assert state["status"] == "stopped"


def test_subprocess_launch_rejects_missing_selected_python(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["environment"]["python"] = str(tmp_path / "missing/bin/python")
    compiler = ExecutionPlanCompiler(tmp_path)
    plan = compiler.compile(project, launch_id="missing-python")

    manager = JobManager(compiler)
    with pytest.raises(FileNotFoundError, match="所选 Python 可执行文件不存在"):
        manager.validate(plan, mode="subprocess")
    with pytest.raises(FileNotFoundError, match="所选 Python 可执行文件不存在"):
        manager.launch(plan, mode="subprocess")


def test_job_process_identity_rejects_reused_pid(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        state = {
            "launch_id": "identity",
            "launch_dir": str(tmp_path / "identity"),
            **JobManager._process_identity(process.pid),
        }
        assert JobManager._process_identity_matches(state) == (True, None)
        state["pid_start_time_ticks"] += 1
        alive, reason = JobManager._process_identity_matches(state)
        assert alive is False
        assert reason == "recorded PID has been reused by another process"
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def test_agent_collab_metadata_only_cache_is_not_ready(tmp_path: Path) -> None:
    from lychee_mas.eval.benchmarks import agent_collab

    (tmp_path / ".lychee_source.json").write_text("{}", encoding="utf-8")
    assert not agent_collab._has_source(tmp_path)


def test_manifest_does_not_treat_source_marker_as_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.benchmarks.manifest import (
        verify_prepared_manifest,
        write_prepared_manifest,
    )

    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    target = prepared / "agent_collab"
    target.mkdir(parents=True)
    (target / ".lychee_source.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", str(raw))
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", str(prepared))
    manifest = write_prepared_manifest("agent_collab", target, source_mode="huggingface")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["integrity"]["status"] == "incomplete"
    assert verify_prepared_manifest(manifest)["status"] == "invalid"


def test_local_model_instance_never_copies_or_deletes_source(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.models import ModelRegistry

    _register_test_model_spec(tmp_path)
    source = tmp_path / "outside/source"
    _complete_test_model(source)
    registry = ModelRegistry(tmp_path)
    instance = registry.save_instance(
        {
            "id": "external-test-qwen",
            "model_spec_id": "test-qwen",
            "acquisition": {
                "mode": "local",
                "source": "local",
                "source_path": str(source),
            },
            "path": str(source),
            "managed": False,
            "status": "ready",
        }
    )
    assert instance["path"] == str(source)
    assert instance["available"] is True
    assert not (tmp_path / "models/ExternalModel").exists()
    registry.delete_instance(instance["id"])
    assert (source / "model.safetensors").read_bytes() == b"weights"


def test_benchmark_task_config_does_not_bind_teams() -> None:
    from lychee_mas.eval.task_config import TASK_CONFIG

    assert TASK_CONFIG
    assert all("team" not in config for config in TASK_CONFIG.values())


def test_model_spec_catalog_separates_platform_sources_without_creating_instance(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.models import ModelRegistry

    _register_test_model_spec(tmp_path)
    registry = ModelRegistry(tmp_path)
    qwen_sources = registry.sources("test-qwen")
    assert [item["provider"] for item in qwen_sources] == ["modelscope", "huggingface"]
    assert registry.sources("test-qwen", "unknown") == []
    assert registry.specs()[0]["organization"] == "Test Organization"
    assert registry.instances() == []


def test_benchmark_root_scan_registers_manifest_assets_idempotently(tmp_path: Path) -> None:
    _register_gsm8k_benchmark(tmp_path)
    registry = BenchmarkRegistry(tmp_path)
    registry.delete_instance("instance-gsm8k")
    raw_root = tmp_path / "scan/raw"
    prepared_root = tmp_path / "scan/prepared"
    raw_source = raw_root / "gsm8k/modelscope/AI-ModelScope--gsm8k"
    raw_source.mkdir(parents=True)
    (raw_source / ".lychee_source.json").write_text(
        json.dumps({"provider": "modelscope", "source_id": "AI-ModelScope/gsm8k"}),
        encoding="utf-8",
    )
    (raw_source / "dataset.parquet").write_bytes(b"raw")
    prepared = prepared_root / "gsm8k"
    payload = prepared / "main"
    payload.mkdir(parents=True)
    (payload / "test.parquet").write_bytes(b"prepared")
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_key": "gsm8k",
                "prepare_target": "gsm8k",
                "source_mode": "modelscope",
                "prepared_location": str(payload),
                "integrity": {"status": "ready"},
                "prepared_files": [
                    {
                        "path": "test.parquet",
                        "size_bytes": len(b"prepared"),
                        "record_count": 1,
                    }
                ],
                "raw_sources": [
                    {
                        "provider": "modelscope",
                        "source_id": "AI-ModelScope/gsm8k",
                        "raw_path": str(raw_source),
                        "exists": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = registry.scan_roots(raw_root=raw_root, prepared_root=prepared_root)
    assert first["summary"] == {
        "matched_assets": 1,
        "created_instances": 1,
        "existing_instances": 0,
        "raw_only": 0,
    }
    instance = first["instances"][0]
    assert instance["benchmark_spec_id"] == "gsm8k"
    assert instance["available"] is True
    assert instance["managed"] is False
    assert instance["prepared_path"] == str(prepared)

    second = registry.scan_roots(raw_root=raw_root, prepared_root=prepared_root)
    assert second["summary"]["created_instances"] == 0
    assert second["summary"]["existing_instances"] == 1
    assert len(registry.instances()) == 1

    instance_path = registry.instance_root / f"{instance['id']}.json"
    stored = json.loads(instance_path.read_text(encoding="utf-8"))
    stored["loader_check"] = {"status": "ready"}
    instance_path.write_text(json.dumps(stored), encoding="utf-8")
    (payload / "test.parquet").write_bytes(b"prepared-v2")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    manifest["prepared_files"][0]["size_bytes"] = len(b"prepared-v2")
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    refreshed = registry.scan_roots(raw_root=raw_root, prepared_root=prepared_root)
    assert refreshed["findings"][0]["loader_check_reset"] is True
    current = registry.instances()[0]
    assert current["id"] == instance["id"]
    assert current["integrity"]["size_bytes"] == len(b"prepared-v2")
    assert current["loader_status"] == "unchecked"


def test_benchmark_instance_loader_check_is_scoped_to_its_prepared_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.studio import benchmarks as benchmark_module

    _register_gsm8k_benchmark(tmp_path)
    registry = BenchmarkRegistry(tmp_path)
    captured: dict = {}

    def successful_probe(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "loader noise\n"
                '__LYCHEE_BENCHMARK_LOADER_PROBE__={"status":"ready","tasks":'
                '[{"task":"gsm8k","status":"ready","kind":"exact",'
                '"case_id":"test-0","sample_count":1}]}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark_module.subprocess, "run", successful_probe)
    checked = registry.check_instance("instance-gsm8k")

    assert captured["command"][:2] == [sys.executable, "-c"]
    overrides = json.loads(captured["env"]["LYCHEE_BENCHMARK_PREPARED_OVERRIDES"])
    assert overrides == {"gsm8k": str(tmp_path / "data/benchmarks/prepared/gsm8k")}
    assert json.loads(captured["env"]["LYCHEE_BENCHMARK_LOADER_PROBE_TASKS"]) == ["gsm8k"]
    assert checked["loader_status"] == "ready"
    assert checked["available"] is True
    persisted = json.loads(
        (tmp_path / "configs/eval_studio/benchmarks/instances/instance-gsm8k.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["loader_check"]["status"] == "ready"
    assert persisted["loader_check"]["tasks"][0]["case_id"] == "test-0"

    def failed_probe(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '__LYCHEE_BENCHMARK_LOADER_PROBE__={"status":"failed","tasks":'
                '[{"task":"gsm8k","status":"failed","detail":"bad parquet"}]}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark_module.subprocess, "run", failed_probe)
    failed = registry.check_instance("instance-gsm8k")
    assert failed["loader_status"] == "failed"
    assert failed["observed_status"] == "loader_failed"
    assert failed["available"] is False


def test_environment_probe_does_not_read_benchmark_data() -> None:
    from lychee_mas.eval.studio.environment import _PROBE

    assert "prepared_loader" not in _PROBE
    assert 'load("gsm8k"' not in _PROBE


def test_benchmark_root_scan_registers_registered_raw_only_asset(tmp_path: Path) -> None:
    _register_gsm8k_benchmark(tmp_path)
    registry = BenchmarkRegistry(tmp_path)
    registry.delete_instance("instance-gsm8k")
    raw_root = tmp_path / "scan/raw"
    raw_source = raw_root / "gsm8k/huggingface/openai--gsm8k"
    raw_source.mkdir(parents=True)
    (raw_source / ".lychee_source.json").write_text(
        json.dumps({"provider": "huggingface", "source_id": "openai/gsm8k"}),
        encoding="utf-8",
    )
    (raw_source / "dataset.parquet").write_bytes(b"raw")
    unproven = raw_root / "gsm8k/huggingface/unproven--dataset"
    unproven.mkdir(parents=True)
    (unproven / "dataset.parquet").write_bytes(b"not registered")

    result = registry.scan_roots(
        raw_root=raw_root,
        prepared_root=tmp_path / "scan/prepared",
    )
    assert result["summary"]["raw_only"] == 1
    assert result["instances"][0]["status"] == "raw_only"
    assert result["instances"][0]["available"] is False


def test_model_root_scan_uses_registered_manifest_and_is_idempotent(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.models import ModelRegistry

    _register_test_model_spec(tmp_path)
    model_root = tmp_path / "scan/models"
    snapshot = model_root / "opaque-snapshot"
    _complete_test_model(snapshot)
    (snapshot / "model_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_spec_id": "test-qwen",
                "provider": "modelscope",
                "source_id": "test/Test-Qwen",
                "revision": "main",
            }
        ),
        encoding="utf-8",
    )
    unknown = model_root / "unknown-complete-model"
    _complete_test_model(unknown)

    registry = ModelRegistry(tmp_path)
    first = registry.scan_root(model_root)
    assert first["summary"] == {
        "matched_assets": 1,
        "created_instances": 1,
        "existing_instances": 0,
        "unmatched_complete_models": 1,
    }
    instance = first["instances"][0]
    assert instance["model_spec_id"] == "test-qwen"
    assert instance["available"] is True
    assert instance["managed"] is False
    assert instance["acquisition"]["source"] == "modelscope"

    second = registry.scan_root(model_root)
    assert second["summary"]["created_instances"] == 0
    assert second["summary"]["existing_instances"] == 1
    assert len(registry.instances()) == 1


def test_studio_scan_endpoints_create_resource_instances(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    _register_gsm8k_benchmark(tmp_path)
    BenchmarkRegistry(tmp_path).delete_instance("instance-gsm8k")
    _register_test_model_spec(tmp_path)

    raw_root = tmp_path / "assets/raw"
    raw_source = raw_root / "gsm8k/modelscope/AI-ModelScope--gsm8k"
    raw_source.mkdir(parents=True)
    (raw_source / "data.parquet").write_bytes(b"raw")
    (raw_source / ".lychee_source.json").write_text(
        json.dumps({"provider": "modelscope", "source_id": "AI-ModelScope/gsm8k"}),
        encoding="utf-8",
    )
    models_root = tmp_path / "assets/models"
    model = models_root / "Test-Qwen"
    _complete_test_model(model)

    client = TestClient(create_app(tmp_path))
    benchmark_response = client.post(
        "/api/benchmark-instances/scan",
        json={"raw_root": str(raw_root), "prepared_root": str(tmp_path / "assets/prepared")},
    )
    model_response = client.post(
        "/api/model-instances/scan",
        json={"models_root": str(models_root)},
    )

    assert benchmark_response.status_code == 200
    assert benchmark_response.json()["summary"]["raw_only"] == 1
    assert model_response.status_code == 200
    assert model_response.json()["summary"]["created_instances"] == 1
    refreshed = client.get(
        "/api/resources/catalog",
        params={
            "raw_root": str(raw_root),
            "prepared_root": str(tmp_path / "assets/prepared"),
            "models_root": str(models_root),
        },
    )
    assert refreshed.status_code == 200
    assert len(refreshed.json()["benchmark_instances"]) == 1
    assert len(refreshed.json()["model_instances"]) == 1


def test_api_spec_does_not_create_an_instance_by_itself(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.apis import APIRegistry

    spec = tmp_path / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "deployment_kind": "api",
                "model_id": "test-model",
                "default_base_url": "https://example.invalid/v1",
                "auth_modes": ["none"],
            }
        ),
        encoding="utf-8",
    )
    registry = APIRegistry(tmp_path)

    assert registry.specs()[0]["id"] == "test-api"
    assert registry.instances() == []


def test_model_download_instance_uses_registered_model_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    commands: list[list[str]] = []

    def fake_launch_utility(_self, **kwargs):
        commands.append(kwargs["command"])
        return {"launch_id": kwargs["job_id"], "status": "running"}

    monkeypatch.setattr(JobManager, "launch_utility", fake_launch_utility)
    _register_test_model_spec(tmp_path)
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/model-instances/test-qwen-download/instantiate",
        json={
            "model_spec_id": "test-qwen",
            "mode": "download",
            "source": "modelscope",
            "models_root": str(tmp_path / "models"),
            "python": sys.executable,
        },
    )

    assert response.status_code == 200
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model-spec-id") + 1] == "test-qwen"
    assert commands[0][commands[0].index("--source") + 1] == "modelscope"
    instance = response.json()["instance"]
    assert instance["id"] == "test-qwen-download"
    assert instance["status"] == "preparing"


def test_catalog_exposes_benchmark_scenarios(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.catalog import StudioCatalog

    rows = StudioCatalog(tmp_path).benchmarks()
    by_key = {item["benchmark_key"]: item for item in rows}
    assert by_key["gsm8k"]["scenario"] == "math"
    assert by_key["gaia"]["scenario"] == "general_assistant"
    assert by_key["bbeh"]["download_sources"]["github"] == [
        "https://github.com/google-deepmind/bbeh.git"
    ]
    assert by_key["hle"]["download_sources"]["github"] == []
    assert by_key["hle"]["reference_sources"][0]["provider"] == "github"


def test_persistent_deployment_registry_round_trip(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    saved = registry.save(
        {
            "id": "shared-vllm",
            "kind": "vllm",
            "source_spec": {"type": "api", "id": "shared-qwen-api"},
            "managed": False,
        }
    )
    assert Path(saved["registry_path"]).is_file()
    assert registry.specs()[0]["id"] == "shared-vllm"
    assert registry.delete("shared-vllm")["deleted"] is True
    assert registry.specs() == []


def test_external_vllm_without_auth_is_probed_but_never_managed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    probes = []

    def probe(base_url, api_key_env, **options):
        probes.append((base_url, api_key_env, options))
        return {
            "status": "running",
            "health_detail": "minimal chat completion succeeded",
        }

    monkeypatch.setattr(registry, "_probe_endpoint", probe)
    instance = registry.deploy(
        {
            "id": "shared-glm",
            "kind": "vllm",
            "source_spec": {"type": "api", "id": "shared-glm-api"},
            "managed": False,
            "shared": True,
        },
        instance_id="instance-shared-glm",
        source_instance={"type": "api", "id": "shared-glm-api-instance"},
        source_fields={
            "model_id": "GLM-4.7-Flash",
            "base_url": "http://127.0.0.1:6000/v1",
            "auth_mode": "none",
            "trust_env": False,
        },
        actual_pricing_instance_id="test-token-pricing",
    )
    assert instance["status"] == "running"
    assert instance["managed"] is False
    assert instance["instance_type"] == "external_shared_server"
    assert "model_path" not in instance
    assert probes == [
        (
            "http://127.0.0.1:6000/v1",
            None,
            {
                "model_id": "GLM-4.7-Flash",
                "auth_mode": "none",
                "trust_env": False,
            },
        )
    ]


def test_team_instance_tracks_normalized_team_spec(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["team_instance"] = {"id": "reason-on-shared-glm", "team_spec_id": "obsolete"}
    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)
    assert normalized["team_instance"] == {
        "id": "reason-on-shared-glm",
        "team_spec_id": "reason",
    }


def test_team_instance_registry_resolves_canonical_team_spec(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.team_instances import TeamInstanceRegistry
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    teams = TeamSpecRegistry(tmp_path)
    team = teams.save(_single_team_spec("research-team"))
    registry = TeamInstanceRegistry(tmp_path)
    saved = registry.save(
        {
            "id": "research-team-local",
            "team_spec_id": "research-team",
            "inference_bindings": [
                {
                    "slot_id": "Solver",
                    "deployment_instance_id": "instance-local",
                    "generation_overrides": {
                        "max_new_tokens": 2048,
                        "thinking_mode": "disabled",
                        "preserve_thinking": False,
                    },
                }
            ],
        }
    )
    persisted = json.loads(Path(saved["registry_path"]).read_text(encoding="utf-8"))
    assert "team_spec" not in persisted
    assert persisted["team_spec_id"] == "research-team"
    teams.save({**team, "participants": [{**team["participants"][0], "name": "Changed"}]})
    loaded = registry.all()[0]
    assert loaded["team_spec"]["participants"][0]["name"] == "Changed"
    assert loaded["inference_bindings"][0]["generation_overrides"]["max_new_tokens"] == 2048
    assert loaded["inference_bindings"][0]["generation_overrides"]["thinking_mode"] == "disabled"
    assert loaded["inference_bindings"][0]["generation_overrides"]["preserve_thinking"] is False
    assert saved["team_spec_id"] == "research-team"
    assert registry.delete("research-team-local")["deleted"] is True


def test_persistent_team_spec_registry_round_trip(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    registry = TeamSpecRegistry(tmp_path)
    saved = registry.save(
        {
            "schema_version": 4,
            "id": "research-team",
            "participants": [
                {"id": "Planner", "name": "Planner", "agent_type": "assistant"},
                {"id": "Reviewer", "name": "Reviewer", "agent_type": "assistant"},
            ],
            "group_chat": {"type": "selector"},
            "termination": {"conditions": []},
            "extensions": {},
        }
    )
    assert Path(saved["registry_path"]).is_file()
    assert saved["extensions"] == {}
    assert registry.all()[0]["group_chat"]["type"] == "selector"
    assert registry.delete("research-team")["deleted"] is True
    assert registry.all() == []


def test_api_probe_explains_missing_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urllib.error import HTTPError

    import lychee_mas.eval.studio.deployments as module

    def reject(*_args: object, **_kwargs: object) -> None:
        raise HTTPError("https://example.test/v1/models", 401, "Unauthorized", {}, None)

    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.setattr(module, "urlopen", reject)
    health = module.DeploymentRegistry(tmp_path)._probe_endpoint(
        "https://example.test/v1", "TEST_API_KEY"
    )
    assert health["status"] == "auth_required"
    assert health["credential_present"] is False
    assert "TEST_API_KEY" in health["health_detail"]
    assert "not set" in health["health_detail"]


def test_api_probe_executes_minimal_chat_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "你好，有什么可以帮你？"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def respond(request, **_kwargs: object):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    monkeypatch.setattr(module, "urlopen", respond)
    health = module.DeploymentRegistry(tmp_path)._probe_endpoint(
        "https://example.test/v1", "TEST_API_KEY", model_id="test-model"
    )

    assert health["status"] == "running"
    assert health["probe_response_preview"].startswith("你好")
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-value"
    assert captured["body"]["messages"] == [{"role": "user", "content": "你好"}]


def test_vllm_probe_verifies_native_thinking_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "brief reasoning",
                                "content": "你好",
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def respond(request, **_kwargs: object):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(module, "urlopen", respond)
    health = module.DeploymentRegistry(tmp_path)._probe_endpoint(
        "http://127.0.0.1:6100/v1",
        None,
        model_id="Qwen3.6-27B",
        auth_mode="none",
        thinking_budget_field="thinking_token_budget",
    )

    assert health["status"] == "running"
    assert health["thinking_budget_probe"] is True
    assert health["thinking_budget_parameter"] == "thinking_token_budget"
    assert health["reasoning_token_accounting"] == "client_retokenized_required"
    assert health["probe_reasoning_content_present"] is True
    assert health["provider_reasoning_tokens_reported"] is False
    assert captured["body"]["thinking_token_budget"] == 16
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_probe_prefers_provider_reasoning_token_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "brief reasoning",
                                "content": "323",
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 12,
                        "completion_tokens_details": {"reasoning_tokens": 9},
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: Response())
    health = module.DeploymentRegistry(tmp_path)._probe_endpoint(
        "http://127.0.0.1:6100/v1",
        None,
        model_id="Qwen3.6-27B",
        auth_mode="none",
        thinking_budget_field="thinking_token_budget",
    )

    assert health["reasoning_token_accounting"] == "provider_usage"
    assert health["provider_reasoning_tokens_reported"] is True
    assert health["probe_completion_tokens"] == 12
    assert health["probe_reasoning_tokens"] == 9


def test_manual_api_key_is_session_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)

    def probe(_base_url: str, api_key_env: str, *, model_id: str, **_options):
        assert os.environ[api_key_env] == "manual-secret"
        assert model_id == "qwen-plus"
        return {
            "status": "running",
            "credential_present": True,
            "api_key_env": api_key_env,
            "health_detail": "minimal chat completion succeeded",
        }

    monkeypatch.setattr(registry, "_probe_endpoint", probe)
    instance = registry.deploy(
        {
            "id": "manual-api",
            "kind": "api",
            "source_spec": {"type": "api", "id": "manual-api-resource"},
        },
        instance_id="instance-manual-api",
        source_instance={"type": "api", "id": "manual-api-resource-instance"},
        source_fields={
            "model_id": "qwen-plus",
            "base_url": "https://example.test/v1",
            "auth_mode": "env",
            "api_key_env": "MODEL_API_KEY",
        },
        actual_pricing_instance_id="test-token-pricing",
        api_key="manual-secret",
    )

    assert instance["credential_source"] == "manual_session"
    assert instance["api_key_env"].startswith("LYCHEE_STUDIO_API_KEY_")
    instance_path = registry.instance_root / f"{instance['id']}.json"
    assert "manual-secret" not in instance_path.read_text(encoding="utf-8")
    key_name = instance["api_key_env"]
    registry.delete_instance(instance["id"])
    assert key_name not in os.environ


def test_deployment_registry_rejects_secret_in_api_key_env(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    with pytest.raises(ValueError, match="cannot bind resource Instance fields"):
        registry.save(
            {
                "id": "unsafe-api",
                "kind": "api",
                "source_spec": {"type": "api", "id": "unsafe-api-resource"},
                "api_key_env": "sk-secret-value",
            }
        )

    health = registry._probe_endpoint("https://example.test/v1", "sk-secret-value")
    assert health["status"] == "invalid"
    assert health["api_key_env"] == "<invalid>"


def test_team_spec_api_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    client = TestClient(create_app(tmp_path))
    payload = {
        "schema_version": 4,
        "id": "api-team",
        "participants": [{"id": "Solver", "name": "Solver", "agent_type": "assistant"}],
        "group_chat": {"type": "round_robin"},
        "termination": {"conditions": []},
        "extensions": {},
    }
    saved = client.put("/api/team-specs/api-team", json=payload)
    assert saved.status_code == 200
    listing = client.get("/api/team-specs").json()
    assert [item["id"] for item in listing["specs"]] == ["api-team"]
    assert client.delete("/api/team-specs/api-team").status_code == 200


def test_team_instance_api_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    client = TestClient(create_app(tmp_path))
    assert (
        client.put(
            "/api/team-specs/api-team",
            json=_single_team_spec("api-team"),
        ).status_code
        == 200
    )
    payload = {
        "schema_version": 3,
        "id": "api-team-instance",
        "team_spec_id": "api-team",
        "inference_bindings": [
            {
                "slot_id": "Solver",
                "deployment_instance_id": "instance-local-hf",
                "generation_overrides": {"max_new_tokens": 1024},
            }
        ],
    }
    saved = client.put("/api/team-instances/api-team-instance", json=payload)
    assert saved.status_code == 200
    listing = client.get("/api/team-instances").json()
    assert [item["id"] for item in listing["instances"]] == ["api-team-instance"]
    assert "team_instances" in client.get("/api/bootstrap").json()
    assert client.delete("/api/team-instances/api-team-instance").status_code == 200


def test_execution_plan_resolves_persisted_team_and_deployment_instances(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app
    from lychee_mas.eval.studio.deployments import DeploymentRegistry
    from lychee_mas.eval.studio.team_instances import TeamInstanceRegistry
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    _register_gsm8k_benchmark(tmp_path)
    deployments = DeploymentRegistry(tmp_path)
    deployments.save(
        {
            "id": "local",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "test-model"},
            "capabilities": {"text_generation": True},
        }
    )
    deployments._write_instance(
        _instance_record(
            "instance-local",
            "local",
            kind="hf",
            model_id="Qwen",
            model_path=str(tmp_path),
            python=sys.executable,
            status="ready_on_run",
            capabilities={"text_generation": True},
        )
    )
    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    TeamInstanceRegistry(tmp_path).save(
        {
            "id": "single-local",
            "team_spec_id": "single",
            "inference_bindings": [
                {"slot_id": "Solver", "deployment_instance_id": "instance-local"}
            ],
        }
    )
    experiment_registry = ExperimentRegistry(tmp_path)
    experiment = experiment_registry.default_spec()
    experiment.update(id="single-gsm8k", team_spec_id="single")
    experiment["environment"]["python"] = sys.executable
    experiment["benchmark"].update(
        benchmark_spec_id="gsm8k",
        runnable_task="gsm8k",
        scoring_profile="official",
    )
    exact_run_dir = tmp_path / "runs/benchmarks/single-local/none/gsm8k/fixed-stamp"
    experiment_registry.save_spec("single-gsm8k", experiment)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/experiment-instances",
        json={
            "id": "single-gsm8k-run",
            "experiment_spec_id": "single-gsm8k",
            "benchmark_instance_id": "instance-gsm8k",
            "team_instance_id": "single-local",
            "run_dir": str(exact_run_dir),
            "launcher": {"type": "subprocess"},
        },
    )
    assert created.status_code == 200
    assert created.json()["status"] == "ready"
    response = client.post("/api/experiment-instances/single-gsm8k-run/plan")

    assert response.status_code == 200
    plan = response.json()
    assert plan["project"]["schema_version"] == 4
    assert plan["project"]["deployment_instances"][0]["id"] == "instance-local"
    assert plan["project"]["benchmark"]["scoring"] == {
        "profile_id": "official",
        "scorer_id": "exact",
        "parameters": {},
    }
    assert plan["run_dir"] == str(exact_run_dir)
    assert str(exact_run_dir) in plan["files"]["run.sh"]
    assert str(exact_run_dir) in plan["files"]["resume.sh"]
    assert str(exact_run_dir) in plan["files"]["analyze.sh"]
    snapshot = json.loads(plan["files"]["config_snapshot/configuration_snapshot.json"])
    assert snapshot["experiment_spec"]["id"] == "single-gsm8k"
    assert snapshot["experiment_instance"]["id"] == "single-gsm8k-run"
    assert snapshot["experiment_instance"]["launcher"] == {"type": "subprocess"}
    assert snapshot["benchmark_spec"]["id"] == "gsm8k"
    assert snapshot["benchmark_instance"]["id"] == "instance-gsm8k"
    assert snapshot["team_spec"]["id"] == "single"
    assert snapshot["team_instance"]["id"] == "single-local"
    assert snapshot["deployment_specs"][0]["id"] == "local"
    assert snapshot["deployment_instances"][0]["id"] == "instance-local"
    assert [item["id"] for item in snapshot["pricing_specs"]] == ["allocated-gpu-time"]
    assert [item["id"] for item in snapshot["pricing_instances"]] == ["test-gpu-pricing"]
    assert "pricing_instance_id" not in snapshot["experiment_spec"]
    assert "pricing_instance_id" not in snapshot["experiment_instance"]
    assert 'cp -a "$LAUNCH_DIR/config_snapshot/."' in plan["files"]["run.sh"]
    assert "verify_deployment_instances.sh" in plan["files"]
    queued = client.post("/api/experiment-instances/single-gsm8k-run/enqueue")
    assert queued.status_code == 200
    assert queued.json()["status"] == "queued"
    dequeued = client.post("/api/experiment-instances/single-gsm8k-run/dequeue")
    assert dequeued.status_code == 200
    assert dequeued.json()["status"] == "ready"


def test_hf_deployment_creates_ready_on_run_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    registry = DeploymentRegistry(tmp_path)
    monkeypatch.setattr(
        registry,
        "_probe_hf",
        lambda _value: {
            "status": "ready_on_run",
            "health_detail": "minimal local HF generation succeeded",
        },
    )
    spec = registry.save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
            "python": sys.executable,
        }
    )
    instance = registry.deploy(
        spec,
        instance_id="instance-local-hf",
        source_instance={"type": "model", "id": "qwen-model-instance"},
        source_fields={"model_id": "Qwen", "model_path": str(model)},
        actual_pricing_instance_id="test-gpu-pricing",
    )
    assert instance["status"] == "ready_on_run"
    assert instance["instance_type"] == "in_process_backend"
    tracked = next(item for item in registry.instances() if item["id"] == "instance-local-hf")
    assert tracked["deployment_spec_id"] == "local-hf"
    assert registry.delete_instance("instance-local-hf")["deleted"] is True


def test_vllm_deployment_uses_selected_environment_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    python = tmp_path / "envs/swift2/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    cuda_root = python.parent.parent / "lib/python3.12/site-packages/nvidia/cu13"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "include").mkdir(parents=True)
    (cuda_root / "lib").mkdir(parents=True)
    (cuda_root / "bin/nvcc").write_text("", encoding="utf-8")
    (python.parent.parent / "lib/python3.1").symlink_to("python3.12", target_is_directory=True)
    captured: dict[str, object] = {}
    reaped = threading.Event()

    class Process:
        pid = 12345

        def wait(self) -> int:
            reaped.set()
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> Process:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        module.DeploymentRegistry,
        "_probe_endpoint",
        staticmethod(lambda *_args, **_kwargs: {"status": "unreachable"}),
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    instance = module.DeploymentRegistry(tmp_path).deploy(
        {
            "id": "local-vllm",
            "kind": "vllm",
            "source_spec": {"type": "model", "id": "qwen-model"},
            "python": str(python),
            "cuda_visible_devices": "0,1,2,3",
            "tensor_parallel_size": 2,
            "data_parallel_size": 2,
            "library_paths": ["/configured/lib"],
            "max_num_seqs": 8,
            "reasoning_parser": "qwen3",
            "reasoning_config": {
                "reasoning_start_str": "<think>",
                "reasoning_end_str": "Stop reasoning now.</think>",
            },
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
            "enable_prefix_caching": True,
            "use_flashinfer_sampler": False,
        },
        instance_id="instance-local-vllm",
        source_instance={"type": "model", "id": "qwen-model-instance"},
        source_fields={
            "model_id": "Qwen",
            "model_path": str(model),
            "auth_mode": "none",
            "trust_env": False,
        },
        actual_pricing_instance_id="test-gpu-pricing",
    )

    environment = captured["environment"]
    command = captured["command"]
    assert isinstance(environment, dict)
    assert environment["LD_LIBRARY_PATH"] == (
        f"{python.parent.parent / 'lib'}:{cuda_root / 'lib'}:/configured/lib:/existing/lib"
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["PATH"].startswith(f"{python.parent}:{cuda_root / 'bin'}:")
    assert environment["CUDA_HOME"] == str(cuda_root)
    assert environment["CUDACXX"] == str(cuda_root / "bin/nvcc")
    assert environment["CPATH"].startswith(str(cuda_root / "include"))
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert reaped.wait(timeout=1)
    assert isinstance(command, list)
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--data-parallel-size") + 1] == "2"
    assert command[command.index("--max-num-seqs") + 1] == "8"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    reasoning_config = json.loads(command[command.index("--reasoning-config") + 1])
    assert reasoning_config["reasoning_start_str"] == "<think>"
    assert reasoning_config["reasoning_end_str"] == "Stop reasoning now.</think>"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--enable-prefix-caching" in command
    assert "--enable-per-request-metrics" not in command
    assert "--enable-prompt-tokens-details" not in command
    assert instance["status"] == "starting"
    assert instance["per_request_metrics_supported"] is False
    assert instance["per_request_metrics_enabled"] is False
    assert instance["prompt_tokens_details_supported"] is False
    assert instance["prompt_tokens_details_enabled"] is False


def test_managed_vllm_requires_enough_visible_devices_for_tp_and_dp(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    with pytest.raises(ValueError, match=r"tensor_parallel_size \* data_parallel_size = 4"):
        DeploymentRegistry(tmp_path).validate(
            {
                "id": "local-vllm",
                "kind": "vllm",
                "source_spec": {"type": "model", "id": "qwen-model"},
                "python": "/env/bin/python",
                "cuda_visible_devices": "4,5",
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "max_num_seqs": 8,
            }
        )


def test_vllm_enabled_per_request_metrics_rejects_unsupported_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    python = tmp_path / "envs/vllm/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module.DeploymentRegistry,
        "_probe_endpoint",
        staticmethod(lambda *_args, **_kwargs: {"status": "unreachable"}),
    )

    with pytest.raises(ValueError, match="does not support --enable-per-request-metrics"):
        module.DeploymentRegistry(tmp_path).deploy(
            {
                "id": "local-vllm",
                "kind": "vllm",
                "source_spec": {"type": "model", "id": "qwen-model"},
                "python": str(python),
                "per_request_metrics_mode": "enabled",
            },
            instance_id="instance-local-vllm",
            source_instance={"type": "model", "id": "qwen-model-instance"},
            source_fields={
                "model_id": "Qwen",
                "model_path": str(model),
                "auth_mode": "none",
            },
            actual_pricing_instance_id="test-gpu-pricing",
        )


def test_vllm_auto_enables_per_request_metrics_when_cli_supports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    python = tmp_path / "envs/vllm/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho --enable-per-request-metrics\n", encoding="utf-8")
    python.chmod(0o755)
    captured: dict[str, object] = {}

    class Process:
        pid = 12345

    monkeypatch.setattr(module, "_vllm_supports_flag", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **_kwargs: captured.update(command=command) or Process(),
    )
    monkeypatch.setattr(
        module.DeploymentRegistry,
        "_probe_endpoint",
        staticmethod(lambda *_args, **_kwargs: {"status": "unreachable"}),
    )

    instance = module.DeploymentRegistry(tmp_path).deploy(
        {
            "id": "local-vllm",
            "kind": "vllm",
            "source_spec": {"type": "model", "id": "qwen-model"},
            "python": str(python),
            "per_request_metrics_mode": "auto",
        },
        instance_id="instance-local-vllm",
        source_instance={"type": "model", "id": "qwen-model-instance"},
        source_fields={
            "model_id": "Qwen",
            "model_path": str(model),
            "auth_mode": "none",
        },
        actual_pricing_instance_id="test-gpu-pricing",
    )

    assert "--enable-per-request-metrics" in captured["command"]
    assert "--enable-prompt-tokens-details" in captured["command"]
    assert instance["per_request_metrics_supported"] is True
    assert instance["per_request_metrics_enabled"] is True
    assert instance["prompt_tokens_details_supported"] is True
    assert instance["prompt_tokens_details_enabled"] is True


def test_vllm_enabled_prompt_token_details_rejects_unsupported_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.studio.deployments as module

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    python = tmp_path / "envs/vllm/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module.DeploymentRegistry,
        "_probe_endpoint",
        staticmethod(lambda *_args, **_kwargs: {"status": "unreachable"}),
    )

    with pytest.raises(ValueError, match="does not support --enable-prompt-tokens-details"):
        module.DeploymentRegistry(tmp_path).deploy(
            {
                "id": "local-vllm",
                "kind": "vllm",
                "source_spec": {"type": "model", "id": "qwen-model"},
                "python": str(python),
                "per_request_metrics_mode": "disabled",
                "prompt_tokens_details_mode": "enabled",
            },
            instance_id="instance-local-vllm",
            source_instance={"type": "model", "id": "qwen-model-instance"},
            source_fields={
                "model_id": "Qwen",
                "model_path": str(model),
                "auth_mode": "none",
            },
            actual_pricing_instance_id="test-gpu-pricing",
        )


def test_deployment_spec_normalizes_request_limits_and_extra_body(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    saved = DeploymentRegistry(tmp_path).save(
        {
            "id": "shared-api",
            "kind": "api",
            "source_spec": {"type": "api", "id": "shared-api-source"},
            "request_limits": {
                "max_concurrency": "3",
                "min_interval_s": "0.5",
                "scope": "host",
            },
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "library_paths": [" /opt/provider/lib "],
        }
    )

    assert saved["request_limits"] == {
        "max_concurrency": 3,
        "min_interval_s": 0.5,
        "scope": "host",
    }
    assert saved["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert saved["library_paths"] == ["/opt/provider/lib"]


def test_deployment_instances_are_loaded_only_from_instance_files(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "declared",
            "kind": "vllm",
            "source_spec": {"type": "api", "id": "declared-api"},
            "managed": False,
        }
    )
    registry._write_instance(
        _instance_record(
            "instance-declared",
            "declared",
            kind="vllm",
            model_id="Qwen",
            source_type="api",
            base_url="http://127.0.0.1:8000/v1",
            status="running",
        )
    )

    assert [item["id"] for item in registry.instances()] == ["instance-declared"]


def test_vllm_probe_updates_its_instance_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "vllm",
            "kind": "vllm",
            "source_spec": {"type": "model", "id": "qwen-model"},
        }
    )
    record = _instance_record(
        "instance-vllm",
        "vllm",
        kind="vllm",
        model_id="Qwen",
        pid=12345,
        base_url="http://127.0.0.1:8000/v1",
        status="starting",
    )
    registry._write_instance(record)
    monkeypatch.setattr(registry, "_probe_api", lambda _record: {"status": "unreachable"})
    assert registry.instances(probe=True)[0]["status"] == "unreachable"
    monkeypatch.setattr(
        registry, "_probe_api", lambda _record: {"status": "running", "health_status": 200}
    )
    assert registry.instances(probe=True)[0]["status"] == "running"
    assert registry.instances(probe=False)[0]["status"] == "running"


def test_api_probe_diagnostics_persist_between_page_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "api",
            "kind": "api",
            "source_spec": {"type": "api", "id": "qwen-api"},
        }
    )
    registry._write_instance(
        _instance_record(
            "instance-api",
            "api",
            kind="api",
            model_id="qwen-plus",
            source_type="api",
            base_url="https://example.test/v1",
            api_key_env="TEST_API_KEY",
            status="auth_required",
        )
    )
    monkeypatch.setattr(
        registry,
        "_probe_api",
        lambda _record: {
            "status": "auth_required",
            "credential_present": False,
            "health_detail": "environment variable TEST_API_KEY is not set",
        },
    )

    probed = registry.instances(probe=True)[0]
    refreshed = registry.instances(probe=False)[0]
    assert probed["health_detail"] == refreshed["health_detail"]
    assert refreshed["credential_present"] is False


def test_team_instance_becomes_unavailable_when_deployment_instance_is_deleted(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry
    from lychee_mas.eval.studio.team_instances import TeamInstanceRegistry

    deployments = DeploymentRegistry(tmp_path)
    deployments.save(
        {
            "id": "local",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
            "capabilities": {"text_generation": True},
        }
    )
    deployments._write_instance(
        _instance_record(
            "instance-local",
            "local",
            kind="hf",
            model_id="Qwen",
            model_path=str(tmp_path),
            python=sys.executable,
            status="ready_on_run",
            capabilities={"text_generation": True},
        )
    )
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    teams = TeamInstanceRegistry(tmp_path)
    teams.save(
        {
            "id": "team-local",
            "team_spec_id": "single",
            "inference_bindings": [
                {"slot_id": "Solver", "deployment_instance_id": "instance-local"}
            ],
        }
    )
    assert teams.all(deployments.instances())[0]["available"] is True
    deployments.delete_instance("instance-local")
    row = teams.all(deployments.instances())[0]
    assert row["available"] is False
    assert row["missing_deployment_instance_ids"] == ["instance-local"]


def test_environment_installation_profiles_are_composable(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.environment import (
        installation_command,
        installation_profiles,
    )

    ids = {item["id"] for item in installation_profiles()}
    assert {"basic", "eval", "runtime", "agentinit", "docs", "full"} <= ids
    command = installation_command(tmp_path, sys.executable, ["eval", "agentinit"])
    assert command[-2:] == ["-e", f"{tmp_path}[benchmark,studio,construct]"]


def test_environment_download_network_maps_proxy_and_mirrors() -> None:
    from lychee_mas.eval.studio.api import _child_environment

    environment = _child_environment(
        {
            "proxy": {
                "enabled": True,
                "https_proxy": "http://127.0.0.1:7890",
                "no_proxy": "localhost",
            },
            "mirrors": {
                "enabled": True,
                "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
                "pip_extra_index_url": "https://pypi.org/simple",
            },
        }
    )
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert environment["https_proxy"] == "http://127.0.0.1:7890"
    assert environment["PIP_INDEX_URL"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert environment["UV_DEFAULT_INDEX"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert environment["PIP_EXTRA_INDEX_URL"] == "https://pypi.org/simple"


def test_single_external_raw_override_is_used_by_native_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.benchmarks.common import raw_source_dir

    external = tmp_path / "external-raw"
    external.mkdir()
    monkeypatch.setenv(
        "LYCHEE_BENCHMARK_RAW_OVERRIDES",
        json.dumps(
            [
                {
                    "benchmark_key": "gsm8k",
                    "provider": "local",
                    "source_id": "custom-copy",
                    "path": str(external),
                }
            ]
        ),
    )
    assert raw_source_dir("gsm8k", "huggingface", "openai/gsm8k") == external


def test_local_raw_benchmark_instance_passes_its_path_to_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    launched: dict = {}

    def fake_launch_utility(_self, **kwargs):
        launched.update(kwargs)
        return {"launch_id": kwargs["job_id"], "status": "running"}

    monkeypatch.setattr(JobManager, "launch_utility", fake_launch_utility)
    raw = tmp_path.parent / f"{tmp_path.name}-external-raw"
    raw.mkdir()
    (raw / "test.jsonl").write_text('{"question": "1+1", "answer": "2"}\n')
    BenchmarkRegistry(tmp_path).save_spec(_benchmark_spec("gsm8k"))

    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/benchmark-instances/external-gsm8k-raw/instantiate",
        json={
            "benchmark_spec_id": "gsm8k",
            "mode": "local",
            "source_path": str(raw),
            "stage": "raw",
            "python": sys.executable,
        },
    )

    assert response.status_code == 200
    instance = response.json()["instance"]
    assert instance["raw_path"] == str(raw)
    assert instance["managed"] is True
    overrides = json.loads(launched["env"]["LYCHEE_BENCHMARK_RAW_OVERRIDES"])
    assert overrides == [
        {
            "benchmark_key": "gsm8k",
            "provider": "local",
            "source_id": "external-gsm8k-raw",
            "path": str(raw),
        }
    ]
    assert "--conversion-only" in launched["command"]
    assert not (tmp_path / "configs/eval_studio/benchmarks/references.json").exists()


def test_studio_api_persists_and_unregisters_deployments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    monkeypatch.setattr(
        DeploymentRegistry,
        "_probe_hf",
        lambda _self, _value: {
            "status": "ready_on_run",
            "health_detail": "minimal local HF generation succeeded",
        },
    )

    monkeypatch.setenv("MODEL_API_KEY", "test-secret")
    _register_test_api_resource(tmp_path)
    client = TestClient(create_app(tmp_path))
    deployment = {
        "id": "api-shared",
        "kind": "api",
        "source_spec": {"type": "api", "id": "test-api"},
    }
    response = client.put("/api/deployments/api-shared", json=deployment)
    assert response.status_code == 200
    assert client.get("/api/deployments").json()["specs"][0]["id"] == "api-shared"
    from lychee_mas.eval.studio.models import ModelRegistry

    _register_test_model_spec(tmp_path)
    model = tmp_path / "models/Test-Qwen"
    _complete_test_model(model)
    ModelRegistry(tmp_path).save_instance(
        {
            "id": "test-qwen-local",
            "model_spec_id": "test-qwen",
            "acquisition": {"mode": "local", "source": "local"},
            "path": str(model),
            "managed": False,
            "status": "ready",
        }
    )
    hf_spec = {
        "id": "local-hf",
        "kind": "hf",
        "source_spec": {"type": "model", "id": "test-qwen"},
        "python": sys.executable,
        "capabilities": {"text_generation": True},
    }
    assert client.put("/api/deployments/local-hf", json=hf_spec).status_code == 200
    deployed = client.post(
        "/api/deployments/local-hf/deploy",
        json={
            "instance_id": "instance-local-hf",
            "source_instance_id": "test-qwen-local",
            "actual_pricing_instance_id": "test-gpu-pricing",
        },
    )
    assert deployed.status_code == 200
    assert deployed.json()["status"] == "ready_on_run"
    tracked = next(
        item
        for item in client.get("/api/deployments").json()["instances"]
        if item["id"] == "instance-local-hf"
    )
    assert tracked["deployment_spec_id"] == "local-hf"
    assert client.delete("/api/deployment-instances/instance-local-hf").status_code == 200
    assert client.delete("/api/deployments/api-shared").status_code == 200


def test_studio_api_registers_external_benchmark_path_without_owning_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.benchmarks.common import prepared_benchmark_dir
    from lychee_mas.eval.studio.api import create_app

    external = tmp_path.parent / f"{tmp_path.name}-external-gsm8k"
    external.mkdir()
    (external / "test-00000-of-00001.parquet").write_bytes(b"external-data")
    BenchmarkRegistry(tmp_path).save_spec(_benchmark_spec("gsm8k"))
    client = TestClient(create_app(tmp_path))
    registered = client.post(
        "/api/benchmark-instances/external-gsm8k/instantiate",
        json={
            "benchmark_spec_id": "gsm8k",
            "mode": "local",
            "source_path": str(external),
            "stage": "prepared",
        },
    )
    assert registered.status_code == 200
    assert registered.json()["instance"]["id"] == "external-gsm8k"
    assert registered.json()["instance"]["available"] is True
    instance = next(
        item
        for item in client.get("/api/benchmark-instances").json()
        if item["id"] == "external-gsm8k"
    )
    assert instance["prepared_path"] == str(external)
    assert instance["managed"] is False
    assert instance["provenance"][0]["path"] == str(external)

    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_OVERRIDES", json.dumps({"gsm8k": str(external)}))
    assert prepared_benchmark_dir("gsm8k") == external
    removed = client.delete("/api/benchmark-instances/external-gsm8k")
    assert removed.status_code == 200
    assert (external / "test-00000-of-00001.parquet").read_bytes() == b"external-data"
    assert not (tmp_path / "configs/eval_studio/benchmarks/references.json").exists()


def test_studio_api_unregisters_model_instance_but_preserves_external_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    external = tmp_path.parent / f"{tmp_path.name}-external-model"
    _complete_test_model(external)
    _register_test_model_spec(tmp_path)
    client = TestClient(create_app(tmp_path))
    registered = client.post(
        "/api/model-instances/external-test-qwen/instantiate",
        json={
            "model_spec_id": "test-qwen",
            "mode": "local",
            "source_path": str(external),
        },
    )
    assert registered.status_code == 200
    assert registered.json()["instance"]["id"] == "external-test-qwen"
    removed = client.delete("/api/model-instances/external-test-qwen")
    assert removed.status_code == 200
    assert (external / "model.safetensors").is_file()


def test_benchmark_spec_resolves_runtime_image_and_proxy_defaults(
    tmp_path: Path,
) -> None:
    registry = BenchmarkRegistry(tmp_path)
    registry.save_spec(_benchmark_spec("gaia"))
    prepared = tmp_path / "data/benchmarks/prepared/gaia"
    prepared.mkdir(parents=True)
    registry.save_instance(
        {
            "id": "instance-gaia",
            "benchmark_spec_id": "gaia",
            "prepared_path": str(prepared),
            "supported_tasks": ["gaia_validation"],
            "status": "ready",
        }
    )
    persisted_instance = json.loads(
        (tmp_path / "configs/eval_studio/benchmarks/instances/instance-gaia.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_instance["prepared_path"] == "data/benchmarks/prepared/gaia"
    assert registry.instances()[0]["prepared_path"] == str(prepared)
    project = default_project(tmp_path)
    project["benchmark"].update(
        task="gaia_validation",
        benchmark_spec_id="gaia",
        benchmark_instance_id="instance-gaia",
        prepare_target="gaia",
    )

    normalized = ExecutionPlanCompiler(tmp_path).normalize(project)

    assert normalized["runtime"]["docker_image"] == "lychee-agbench-gaia:local"
    assert normalized["benchmark"]["benchmark_instance_id"] == "instance-gaia"
    assert normalized["network"]["mode"] == "proxy"
    assert normalized["network"]["container_proxy_url"] == ("http://host.docker.internal:17897")
    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="gaia-proxy")
    assert "lychee-agbench-gaia:local" in plan.files["run.sh"]
    assert "--web-proxy-url" in plan.files["run.sh"]
    assert "--proxy-relay-upstream-url" in plan.files["run.sh"]


def test_experiment_instances_are_persistent_priority_queue_entries(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "nightly"
    registry.save_spec("nightly", spec)
    low = registry.create_instance(
        instance_id="nightly-low",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=200,
    )
    high = registry.create_instance(
        instance_id="nightly-high",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
    )

    tmux = registry.create_instance(
        instance_id="nightly-tmux",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        launcher={
            "type": "new_tmux_session",
            "tmux_session": "nightly",
            "tmux_window": "benchmark",
        },
    )
    assert low["status"] == high["status"] == tmux["status"] == "ready"
    assert tmux["launcher"]["type"] == "new_tmux_session"
    registry.enqueue("nightly-low")
    registry.enqueue("nightly-high")
    assert registry.next_queued()["id"] == "nightly-high"
    updated = registry.update_instance("nightly-high", status="running")
    assert updated["status"] == "running"
    assert (tmp_path / "configs/eval_studio/experiments/instances/nightly-high.json").is_file()


def test_experiment_registry_rejects_duplicate_instance_id(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "nightly"
    registry.save_spec("nightly", spec)
    kwargs = {
        "instance_id": "nightly-run",
        "spec_id": "nightly",
        "benchmark_instance_id": "benchmark-data",
        "team_instance_id": "team-runtime",
    }
    registry.create_instance(**kwargs)

    with pytest.raises(ValueError, match="already exists"):
        registry.create_instance(**kwargs)


def test_orphaned_launch_can_be_recovered_as_experiment_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "nightly"
    registry.save_spec("nightly", spec)
    launch_dir = tmp_path / "runs/eval_studio/launches/nightly-orphan"
    run_dir = tmp_path / "runs/benchmarks/nightly-orphan"
    state = {
        "launch_id": "nightly-orphan",
        "status": "running",
        "mode": "subprocess",
        "pid": 123,
        "launch_dir": str(launch_dir),
        "run_dir": str(run_dir),
        "created_at_utc": "2026-08-12T00:00:00+00:00",
    }
    snapshot = {
        "schema_version": 3,
        "id": "nightly-orphan",
        "experiment_spec_id": "nightly",
        "benchmark_instance_id": "benchmark-data",
        "team_instance_id": "team-runtime",
        "launcher": {"type": "subprocess"},
        "queue": {"priority": 50},
        "run_dir": str(run_dir),
    }

    class Jobs:
        stopped = False

        @staticmethod
        def launches():
            return [dict(state)]

        @staticmethod
        def is_active_state(value):
            return value.get("status") in {"starting", "running"}

        @staticmethod
        def experiment_instance_snapshot(_launch_dir):
            return dict(snapshot)

        @staticmethod
        def progress(_launch_dir, *, tail):
            return {"state": dict(state), "percent": 20, "lines": []}

        @staticmethod
        def status(_launch_dir):
            return dict(state)

        def stop(self, _launch_dir):
            self.stopped = True
            return {**state, "status": "stopped"}

    jobs = Jobs()
    manager = ExperimentQueueManager(registry, None, jobs, None)
    monkeypatch.setattr(manager, "_start_monitor", lambda *_args: None)

    orphan = manager.orphaned_launches()[0]
    assert orphan["launch_id"] == "nightly-orphan"
    assert orphan["recoverable"] is True
    recovered = manager.recover_orphaned_launch("nightly-orphan")["instance"]
    assert recovered["status"] == "running"
    assert recovered["launch_dir"] == str(launch_dir)
    assert manager.orphaned_launches() == []


def test_active_launch_blocks_experiment_instance_deletion(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "nightly"
    registry.save_spec("nightly", spec)
    launch_dir = tmp_path / "runs/eval_studio/launches/nightly-run"
    launch_dir.mkdir(parents=True)
    instance = registry.create_instance(
        instance_id="nightly-run",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
    )
    registry.update_instance(
        instance["id"],
        status="completed",
        launch_dir=str(launch_dir),
        launch_id=instance["id"],
    )
    (launch_dir / "job.json").write_text("{}", encoding="utf-8")

    class Jobs:
        @staticmethod
        def status(_launch_dir):
            return {"status": "running"}

        @staticmethod
        def is_active_state(value):
            return value.get("status") == "running"

    manager = ExperimentQueueManager(registry, None, Jobs(), None)
    with pytest.raises(ValueError, match="active launch"):
        manager.delete_instance(instance["id"])
    assert registry.get_instance(instance["id"])["id"] == instance["id"]


def test_experiment_queue_runs_persistent_instances_in_priority_order(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "nightly"
    registry.save_spec("nightly", spec)
    registry.create_instance(
        instance_id="nightly-low",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=200,
        launcher={"type": "subprocess"},
    )
    registry.create_instance(
        instance_id="nightly-high",
        spec_id="nightly",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
        launcher={
            "type": "new_tmux_session",
            "tmux_session": "nightly",
            "tmux_window": "high",
        },
    )

    class Compiler:
        launch_order: list[str] = []

        def compile(self, value, *, launch_id):
            self.launch_order.append(launch_id)
            launch_dir = tmp_path / "runs/eval_studio/launches" / launch_id
            launch_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                launch_id=launch_id,
                launch_dir=launch_dir,
                run_dir=tmp_path / "runs/benchmarks" / launch_id,
                project=value,
            )

    class Jobs:
        launch_modes: list[str] = []
        launch_order: list[str] = []
        validated_modes: list[str] = []

        def validate(self, _plan, mode):
            self.validated_modes.append(mode)
            return {"mode": mode}

        def launch(self, plan, mode):
            self.launch_order.append(plan.launch_id)
            self.launch_modes.append(mode)
            return {"status": "running"}

        def status(self, _launch_dir):
            return {"status": "completed", "return_code": 0}

        @staticmethod
        def launches():
            return []

    compiler = Compiler()
    manager = ExperimentQueueManager(
        registry,
        compiler,
        Jobs(),
        lambda value, **_instances: value,
    )
    manager.enqueue("nightly-low")
    manager.enqueue("nightly-high")
    assert manager.jobs.validated_modes == ["subprocess", "new_tmux_session"]
    manager.start()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert manager.jobs.launch_order == ["nightly-high", "nightly-low"]
    assert manager.jobs.launch_modes == ["new_tmux_session", "subprocess"]
    assert [item["status"] for item in registry.instances()] == [
        "completed",
        "completed",
    ]


def test_experiment_queue_dynamically_shares_deployment_capacity(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "parallel"
    spec["runtime"]["case_concurrency"] = 32
    spec["runtime"]["concurrency_policy"] = {
        "mode": "fixed",
        "initial": 32,
        "minimum": 32,
        "maximum": 32,
    }
    spec["benchmark"]["cases"] = 10
    registry.save_spec("parallel", spec)
    for index in range(4):
        registry.create_instance(
            instance_id=f"parallel-{index}",
            spec_id="parallel",
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=index,
        )

    class Compiler:
        launch_order: list[str] = []

        def compile(self, _value, *, launch_id):
            self.launch_order.append(launch_id)
            launch_dir = tmp_path / "runs/eval_studio/launches" / launch_id
            run_dir = tmp_path / "runs/benchmarks" / launch_id
            launch_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                launch_id=launch_id,
                launch_dir=launch_dir,
                run_dir=run_dir,
                project={
                    "runtime": {
                        "case_concurrency": 32,
                        "concurrency_policy": {"maximum": 32},
                        "samples": 1,
                    },
                    "benchmark": {"n": 10},
                    "deployment_instances": [
                        {
                            "id": "shared-vllm",
                            "request_limits": {"max_concurrency": 32},
                        }
                    ],
                },
            )

    class Jobs:
        states: dict[str, str] = {}
        launch_order: list[str] = []

        @staticmethod
        def validate(_plan, mode):
            return {"mode": mode}

        def launch(self, plan, mode):
            self.states[plan.launch_id] = "running"
            self.launch_order.append(plan.launch_id)
            return {"status": "running", "mode": mode}

        def status(self, launch_dir):
            launch_id = Path(launch_dir).name
            status = self.states.get(launch_id, "running")
            return {"status": status, "return_code": 0 if status == "completed" else None}

        @staticmethod
        def launches():
            return []

    compiler = Compiler()
    jobs = Jobs()
    manager = ExperimentQueueManager(
        registry,
        compiler,
        jobs,
        lambda value, **_instances: value,
    )
    for index in range(4):
        manager.enqueue(f"parallel-{index}")
    compiler.launch_order.clear()
    manager.start(max_parallel_instances=8)

    deadline = time.time() + 5
    while len(jobs.launch_order) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert jobs.launch_order == ["parallel-0", "parallel-1", "parallel-2"]
    assert manager.status()["resource_usage"]["shared-vllm"]["used"] == 30

    run_status = {
        "expected_predictions": 10,
        "successful_predictions": 8,
        "error_predictions": 0,
        "skipped_predictions": 0,
    }
    (tmp_path / "runs/benchmarks/parallel-0/run_status.json").write_text(
        json.dumps(run_status), encoding="utf-8"
    )
    manager._wake_event.set()
    deadline = time.time() + 5
    while len(jobs.launch_order) < 4 and time.time() < deadline:
        time.sleep(0.02)
    assert jobs.launch_order == [
        "parallel-0",
        "parallel-1",
        "parallel-2",
        "parallel-3",
    ]
    status = manager.status()
    assert len(status["active_instance_ids"]) == 4
    assert status["resource_usage"]["shared-vllm"]["used"] == 32

    jobs.states.update({f"parallel-{index}": "completed" for index in range(4)})
    manager._wake_event.set()
    assert manager._thread is not None
    manager._thread.join(timeout=8)
    assert not manager._thread.is_alive()
    assert all(item["status"] == "completed" for item in registry.instances())


def test_experiment_queue_rejects_start_without_queued_instances(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    manager = ExperimentQueueManager(registry, None, None, None)

    with pytest.raises(ValueError, match="没有 queued ExperimentInstance"):
        manager.start()


def test_experiment_spec_rejects_task_outside_registered_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    benchmarks = BenchmarkRegistry(tmp_path)
    benchmarks.save_spec(_benchmark_spec("gsm8k"))
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(
        {
            "schema_version": 4,
            "id": "single",
            "participants": [{"id": "solver", "name": "solver"}],
            "group_chat": {"type": "round_robin"},
        }
    )
    response = TestClient(create_app(tmp_path)).put(
        "/api/experiment-specs/invalid-experiment",
        json={
            "schema_version": 3,
            "id": "invalid-experiment",
            "benchmark": {
                "benchmark_spec_id": "gsm8k",
                "runnable_task": "gaia_validation",
                "cases": 1,
            },
            "team_spec_id": "single",
        },
    )
    assert response.status_code == 422
    assert "is not registered by Benchmark" in response.json()["detail"]


def test_experiment_instance_requires_available_benchmark_instance(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    benchmarks = BenchmarkRegistry(tmp_path)
    benchmarks.save_spec(_benchmark_spec("gsm8k"))
    benchmarks.save_instance(
        {
            "id": "instance-gsm8k",
            "benchmark_spec_id": "gsm8k",
            "prepared_path": str(tmp_path / "missing-prepared-data"),
            "status": "ready",
        }
    )
    TeamSpecRegistry(tmp_path).save(
        {
            "schema_version": 4,
            "id": "single",
            "participants": [{"id": "solver", "name": "solver"}],
            "group_chat": {"type": "round_robin"},
        }
    )
    registry = ExperimentRegistry(tmp_path)
    registry.save_spec(
        "gsm8k-smoke",
        {
            "schema_version": 3,
            "id": "gsm8k-smoke",
            "benchmark": {
                "benchmark_spec_id": "gsm8k",
                "runnable_task": "gsm8k",
                "cases": 1,
            },
            "team_spec_id": "single",
        },
    )
    response = TestClient(create_app(tmp_path)).post(
        "/api/experiment-instances",
        json={
            "id": "gsm8k-smoke-instance",
            "experiment_spec_id": "gsm8k-smoke",
            "benchmark_instance_id": "instance-gsm8k",
            "team_instance_id": "missing-team-instance",
        },
    )
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_api_resource_manual_key_is_never_persisted(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.apis import APIRegistry

    spec = tmp_path / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "deployment_kind": "api",
                "model_id": "test-model",
                "default_base_url": "https://example.invalid/v1",
                "auth_modes": ["env"],
                "default_api_key_env": "MODEL_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    secret = "unit-test-secret-value"
    registry = APIRegistry(tmp_path)
    saved = registry.save_instance(
        {
            "id": "test-model",
            "api_spec_id": "test-api",
            "model_id": "test-model",
            "base_url": "https://example.invalid/v1",
            "auth_mode": "env",
            "api_key_env": "MODEL_API_KEY",
            "api_key": secret,
        }
    )
    path = tmp_path / "configs/eval_studio/apis/instances/test-model.json"
    persisted = path.read_text(encoding="utf-8")

    try:
        assert secret not in persisted
        assert "api_key" not in json.loads(persisted)
        assert saved["credential_source"] == "manual_session"
        assert os.environ[saved["api_key_env"]] == secret
    finally:
        os.environ.pop(saved["api_key_env"], None)


def test_deployment_api_rejects_unregistered_direct_endpoint(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    client = TestClient(create_app(tmp_path))
    response = client.put(
        "/api/deployments/unregistered-api",
        json={
            "id": "unregistered-api",
            "kind": "api",
            "model_id": "arbitrary-model",
            "base_url": "https://example.invalid/v1",
            "auth_mode": "none",
        },
    )

    assert response.status_code == 422
    assert "source_spec.type must be model or api" in response.json()["detail"]


def test_benchmark_spec_does_not_create_an_instance_by_itself(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path)
    saved = registry.save_spec(_benchmark_spec("gaia"))
    persisted = json.loads(
        (tmp_path / "configs/eval_studio/benchmarks/specs/gaia.json").read_text(encoding="utf-8")
    )

    assert persisted == _benchmark_spec("gaia")
    assert saved["schema_version"] == 4
    assert saved["prepare_target"] == "gaia"
    assert saved["implementation"]["prepare_target"] == "gaia"
    assert registry.instances() == []


def test_registered_benchmark_spec_derives_implementation_owned_fields(tmp_path: Path) -> None:
    registry = BenchmarkRegistry(tmp_path)

    saved = registry.save_spec(_benchmark_spec("gsm8k"))
    persisted = json.loads(
        (tmp_path / "configs/eval_studio/benchmarks/specs/gsm8k.json").read_text(encoding="utf-8")
    )

    assert saved["name"] == "GSM8K"
    assert saved["category"] == "math"
    assert saved["prepare_target"] == "gsm8k"
    assert saved["runnable_tasks"] == ["gsm8k"]
    assert saved["task_contracts"]["gsm8k"]["kind"] == "exact"
    assert saved["sources"]["huggingface"]["default_ids"]
    assert set(persisted) == {"schema_version", "id", "name", "category"}

    with pytest.raises(ValueError, match="stores identity metadata only"):
        registry.save_spec(
            {
                "id": "gsm8k",
                "prepare_target": "wrong-target",
            }
        )


def test_benchmark_download_instance_uses_only_full_prepare_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    commands: list[list[str]] = []

    def fake_launch_utility(_self, **kwargs):
        commands.append(kwargs["command"])
        return {"launch_id": kwargs["job_id"], "status": "running"}

    monkeypatch.setattr(JobManager, "launch_utility", fake_launch_utility)
    BenchmarkRegistry(tmp_path).save_spec(_benchmark_spec("gaia"))
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/benchmark-instances/gaia-full-local/instantiate",
        json={
            "benchmark_spec_id": "gaia",
            "mode": "download",
            "source": "auto",
            "python": sys.executable,
        },
    )

    assert response.status_code == 200
    assert len(commands) == 1
    task_index = commands[0].index("--tasks")
    assert commands[0][task_index + 1] == "gaia"
    assert not any("level_" in part for part in commands[0])
    instance = response.json()["instance"]
    assert instance["benchmark_spec_id"] == "gaia"
    assert instance["acquisition"]["mode"] == "download"
    assert instance["status"] == "preparing"


def test_benchmark_instance_rejects_an_unregistered_provider(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    BenchmarkRegistry(tmp_path).save_spec(_benchmark_spec("gsm8k"))
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/benchmark-instances/gsm8k-github/instantiate",
        json={
            "benchmark_spec_id": "gsm8k",
            "mode": "download",
            "source": "github",
            "python": sys.executable,
        },
    )

    assert response.status_code == 422
    assert "has no registered github source" in response.json()["detail"]


def test_direct_benchmark_path_import_requires_benchmark_instance(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    source = tmp_path / "external"
    source.mkdir()
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/resources/import",
        json={"kind": "benchmark", "source_path": str(source), "target": "gsm8k"},
    )

    assert response.status_code == 422
    assert "registered BenchmarkSpec, ModelSpec or APISpec" in response.json()["detail"]


def test_spec_documents_never_persist_instance_bindings(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    deployment = DeploymentRegistry(tmp_path).save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
            "capabilities": {"text_generation": True},
        }
    )
    team = TeamSpecRegistry(tmp_path).save(_single_team_spec())
    experiment = ExperimentRegistry(tmp_path).default_spec()
    experiment.update(id="gsm8k-reason", team_spec_id="single")
    saved_experiment = ExperimentRegistry(tmp_path).save_spec("gsm8k-reason", experiment)

    deployment_document = json.loads(Path(deployment["registry_path"]).read_text(encoding="utf-8"))
    team_document = json.loads(Path(team["registry_path"]).read_text(encoding="utf-8"))
    experiment_document = json.loads(
        Path(saved_experiment["registry_path"]).read_text(encoding="utf-8")
    )

    assert deployment_document["source_spec"] == {"type": "model", "id": "qwen-model"}
    assert "source_instance" not in deployment_document
    assert "model_instance_id" not in deployment_document
    assert "api_instance_id" not in deployment_document
    assert "inference_bindings" not in team_document
    assert "benchmark_instance_id" not in experiment_document
    assert "team_instance_id" not in experiment_document


def test_instance_documents_bind_only_their_corresponding_instances(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry
    from lychee_mas.eval.studio.team_instances import TeamInstanceRegistry
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    team_saved = TeamInstanceRegistry(tmp_path).save(
        {
            "id": "single-local",
            "team_spec_id": "single",
            "inference_bindings": [
                {"slot_id": "Solver", "deployment_instance_id": "instance-local"}
            ],
        }
    )
    deployment_registry = DeploymentRegistry(tmp_path)
    deployment_registry.save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
        }
    )
    deployment_registry._write_instance(
        _instance_record(
            "instance-local",
            "local-hf",
            kind="hf",
            model_id="Qwen",
            source_type="model",
            source_id="qwen-model-local",
            status="ready_on_run",
        )
    )
    experiment_registry = ExperimentRegistry(tmp_path)
    experiment = experiment_registry.default_spec()
    experiment.update(id="gsm8k-single", team_spec_id="single")
    experiment_registry.save_spec("gsm8k-single", experiment)
    experiment_saved = experiment_registry.create_instance(
        instance_id="gsm8k-single-run",
        spec_id="gsm8k-single",
        benchmark_instance_id="gsm8k-data",
        team_instance_id="single-local",
    )

    team_document = json.loads(Path(team_saved["registry_path"]).read_text(encoding="utf-8"))
    deployment_document = json.loads(
        (tmp_path / "configs/eval_studio/deployments/instances/instance-local.json").read_text(
            encoding="utf-8"
        )
    )
    experiment_document = json.loads(
        Path(experiment_saved["registry_path"]).read_text(encoding="utf-8")
    )

    assert team_document == {
        "schema_version": 3,
        "id": "single-local",
        "team_spec_id": "single",
        "team_spec_fingerprint": team_document["team_spec_fingerprint"],
        "creation_source": "studio",
        "inference_bindings": [{"slot_id": "Solver", "deployment_instance_id": "instance-local"}],
        "created_at_utc": team_document["created_at_utc"],
    }
    assert deployment_document["source_instance"] == {
        "type": "model",
        "id": "qwen-model-local",
    }
    assert experiment_document["benchmark_instance_id"] == "gsm8k-data"
    assert experiment_document["team_instance_id"] == "single-local"
    assert "benchmark" not in experiment_document
    assert "team_spec" not in experiment_document


def test_instance_registries_pin_the_spec_version_used_at_creation(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.apis import APIRegistry
    from lychee_mas.eval.studio.models import ModelRegistry
    from lychee_mas.eval.studio.team_instances import TeamInstanceRegistry
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    _register_test_model_spec(tmp_path)
    model_path = tmp_path / "models/test-qwen"
    _complete_test_model(model_path)
    model_registry = ModelRegistry(tmp_path)
    model = model_registry.save_instance(
        {
            "id": "model-instance",
            "model_spec_id": "test-qwen",
            "path": str(model_path),
        }
    )

    api_spec_path = tmp_path / "configs/eval_studio/apis/specs/test-api.json"
    api_spec_path.parent.mkdir(parents=True, exist_ok=True)
    api_spec_path.write_text(
        json.dumps(
            {
                "id": "test-api",
                "model_id": "remote-model",
                "default_base_url": "https://example.invalid/v1",
                "auth_modes": ["none"],
            }
        ),
        encoding="utf-8",
    )
    api_registry = APIRegistry(tmp_path)
    api_instance = api_registry.save_instance(
        {
            "id": "api-instance",
            "api_spec_id": "test-api",
            "model_id": "remote-model",
            "base_url": "https://example.invalid/v1",
            "auth_mode": "none",
        }
    )

    benchmarks = BenchmarkRegistry(tmp_path)
    benchmarks.save_spec(_benchmark_spec("gsm8k"))
    prepared = tmp_path / "prepared/test-benchmark"
    prepared.mkdir(parents=True)
    (prepared / "data.jsonl").write_text("{}\n", encoding="utf-8")
    benchmark = benchmarks.save_instance(
        {
            "id": "benchmark-instance",
            "benchmark_spec_id": "gsm8k",
            "prepared_path": str(prepared),
        }
    )

    teams = TeamSpecRegistry(tmp_path)
    teams.save(_single_team_spec())
    team_instances = TeamInstanceRegistry(tmp_path)
    team = team_instances.save(
        {
            "id": "team-instance",
            "team_spec_id": "single",
            "inference_bindings": [
                {"slot_id": "Solver", "deployment_instance_id": "deployment-instance"}
            ],
        }
    )

    experiments = ExperimentRegistry(tmp_path)
    experiment_spec = experiments.default_spec()
    experiment_spec["id"] = "experiment"
    experiments.save_spec("experiment", experiment_spec)
    experiment = experiments.create_instance(
        instance_id="experiment-instance",
        spec_id="experiment",
        benchmark_instance_id="benchmark-instance",
        team_instance_id="team-instance",
    )

    for prefix, instance in (
        ("model", model),
        ("api", api_instance),
        ("benchmark", benchmark),
        ("team", team),
        ("experiment", experiment),
    ):
        assert instance[f"{prefix}_spec_fingerprint"]
        assert instance["created_at_utc"]
        assert instance["creation_source"] == "studio"

    model_spec_path = tmp_path / "configs/eval_studio/models/specs/test-qwen.json"
    model_spec = json.loads(model_spec_path.read_text(encoding="utf-8"))
    model_spec["capabilities"] = ["hf"]
    model_spec_path.write_text(json.dumps(model_spec), encoding="utf-8")

    api_spec = json.loads(api_spec_path.read_text(encoding="utf-8"))
    api_spec["default_base_url"] = "https://changed.invalid/v1"
    api_spec_path.write_text(json.dumps(api_spec), encoding="utf-8")

    changed_benchmark = _benchmark_spec("gsm8k")
    changed_benchmark["name"] = "GSM8K changed"
    benchmarks.save_spec(changed_benchmark)

    changed_team = _single_team_spec()
    changed_team["participants"][0]["system_prompt"] = "changed"
    teams.save(changed_team)

    changed_experiment = experiments.get_spec("experiment")
    changed_experiment["runtime"]["max_turns"] = 99
    experiments.save_spec("experiment", changed_experiment)

    assert model_registry.instances()[0]["configuration_status"] == "invalid"
    assert api_registry.instances()[0]["configuration_status"] == "invalid"
    assert benchmarks.instances()[0]["configuration_status"] == "invalid"
    assert team_instances.all()[0]["configuration_status"] == "invalid"
    assert experiments.instances()[0]["configuration_status"] == "invalid"
    with pytest.raises(ValueError, match="is invalid"):
        experiments.enqueue("experiment-instance")


def test_benchmark_spec_rejects_an_unregistered_implementation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no registered Benchmark implementation"):
        BenchmarkRegistry(tmp_path).save_spec(
            {
                "id": "custom",
                "name": "Custom",
                "category": "reasoning",
            }
        )


def test_magentic_one_controller_is_an_inference_slot_but_terminal_is_not(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.teams import TeamSpecRegistry

    saved = TeamSpecRegistry(tmp_path).save(
        {
            "schema_version": 4,
            "id": "gaia",
            "participants": [
                {"id": "Coder", "name": "Coder", "agent_type": "coder"},
                {
                    "id": "ComputerTerminal",
                    "name": "ComputerTerminal",
                    "agent_type": "computer_terminal",
                },
                {"id": "WebSurfer", "name": "WebSurfer", "agent_type": "web_surfer"},
            ],
            "group_chat": {"type": "magentic_one"},
        }
    )

    slots = {item["id"]: item for item in saved["inference_slots"]}
    assert slots["Orchestrator"]["kind"] == "controller"
    assert slots["Coder"]["kind"] == "participant"
    assert slots["WebSurfer"]["kind"] == "participant"
    assert "ComputerTerminal" not in slots


def test_deployment_instance_is_invalid_when_its_spec_is_missing(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry._write_instance(
        _instance_record(
            "orphan",
            "missing-spec",
            kind="api",
            model_id="missing",
            source_type="api",
            source_id="missing-api",
            deployment_spec_fingerprint="0" * 64,
            status="running",
        )
    )

    assert registry.instance("orphan")["status"] == "invalid"
    assert registry.instance("orphan")["available"] is False


def test_deployment_instance_is_invalid_when_its_pricing_instance_is_missing(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "api",
            "kind": "api",
            "source_spec": {"type": "api", "id": "test-api"},
        }
    )
    registry._write_instance(
        _instance_record(
            "orphan-pricing",
            "api",
            kind="api",
            model_id="remote",
            source_type="api",
            source_id="test-api-instance",
            actual_pricing_instance_id="missing-pricing",
            status="running",
        )
    )

    checked = registry.instance("orphan-pricing")
    assert checked["status"] == "invalid"
    assert checked["available"] is False
    assert "invalid pricing binding" in checked["health_detail"]


def test_deployment_instance_becomes_invalid_after_its_spec_changes(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import (
        DeploymentRegistry,
        _deployment_spec_fingerprint,
    )

    registry = DeploymentRegistry(tmp_path)
    original = registry.save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "source"},
            "cuda_visible_devices": "0",
        }
    )
    registry._write_instance(
        _instance_record(
            "instance-local-hf",
            "local-hf",
            kind="hf",
            model_id="model",
            model_path=str(tmp_path / "model"),
            python=sys.executable,
            cuda_visible_devices="0",
            deployment_spec_fingerprint=_deployment_spec_fingerprint(original),
            status="ready_on_run",
        )
    )
    registry.save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "source"},
            "cuda_visible_devices": "1",
        }
    )

    checked = registry.instance("instance-local-hf")
    assert checked["status"] == "invalid"
    assert checked["available"] is False
    assert "Spec changed" in checked["health_detail"]


@pytest.mark.parametrize(
    ("kind", "managed", "pricing_spec_id", "message"),
    [
        ("hf", None, "api-token-usage", "allocated_gpu_time"),
        ("vllm", True, "api-token-usage", "allocated_gpu_time"),
        ("vllm", False, "allocated-gpu-time", "token_usage"),
        ("api", False, "allocated-gpu-time", "token_usage"),
    ],
)
def test_deployment_spec_enforces_actual_pricing_by_lifecycle(
    tmp_path: Path,
    kind: str,
    managed: bool | None,
    pricing_spec_id: str,
    message: str,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    spec = {
        "id": f"invalid-{kind}-{str(managed).lower()}",
        "kind": kind,
        "source_spec": {
            "type": "model" if kind == "hf" or managed is True else "api",
            "id": "source",
        },
        "actual_pricing_spec_id": pricing_spec_id,
        **({"managed": managed} if managed is not None else {}),
    }
    with pytest.raises(ValueError, match=message):
        DeploymentRegistry(tmp_path).save(spec)


def test_external_provider_deployment_rejects_api_equivalent_pricing(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    with pytest.raises(ValueError, match="only valid for local HF or managed vLLM"):
        DeploymentRegistry(tmp_path).save(
            {
                "id": "external-vllm",
                "kind": "vllm",
                "managed": False,
                "source_spec": {"type": "api", "id": "source"},
                "actual_pricing_spec_id": "api-token-usage",
                "api_equivalent_pricing_spec_id": "api-token-usage",
            }
        )


def test_pricing_contract_rejects_wrong_units_and_unknown_rates(tmp_path: Path) -> None:
    registry = PricingRegistry(tmp_path)
    with pytest.raises(ValueError, match="per_million_tokens"):
        registry.save_spec(
            "wrong-unit",
            {
                "basis": "token_usage",
                "rate_unit": "per_thousand_tokens",
                "supported_deployment_kinds": ["api"],
                "required_rates": ["input", "output"],
                "currency": "CNY",
                "rates": {"input": 1.0, "output": 2.0},
                "metadata": {"model_ids": ["model"]},
            },
        )
    with pytest.raises(ValueError, match="undeclared"):
        registry.save_spec(
            "bad-rate",
            {
                "basis": "token_usage",
                "supported_deployment_kinds": ["api"],
                "required_rates": ["input", "output"],
                "currency": "CNY",
                "rates": {"input": 1.0, "output": 2.0, "image": 3.0},
                "metadata": {"model_ids": ["model"]},
            },
        )


def test_pricing_instance_supports_strictly_ordered_input_token_tiers(
    tmp_path: Path,
) -> None:
    registry = PricingRegistry(tmp_path)
    registry.save_spec(
        "tiered-token-pricing",
        {
            "basis": "token_usage",
            "supported_deployment_kinds": ["api", "vllm"],
            "required_rates": ["input", "output"],
            "optional_rates": [],
            "currency": "CNY",
            "rates": {},
            "rate_tiers": [
                {
                    "up_to_input_tokens": 128_000,
                    "rates": {"input": 0.8, "output": 2.0},
                },
                {
                    "up_to_input_tokens": 256_000,
                    "rates": {"input": 2.4, "output": 20.0},
                },
            ],
            "metadata": {"model_ids": ["model"], "source_spec_ids": ["source"]},
        },
    )
    saved = registry.instantiate_instance("tiered-token-pricing", "tiered-token-pricing")

    assert saved["rate_tiers"][1]["rates"]["output"] == 20.0
    assert registry.get_instance("tiered-token-pricing")["status"] == "ready"
    with pytest.raises(ValueError, match="strictly increasing"):
        registry.save_spec(
            "unordered-token-pricing",
            {
                "basis": "token_usage",
                "supported_deployment_kinds": ["api", "vllm"],
                "required_rates": ["input", "output"],
                "currency": "CNY",
                "rates": {},
                "rate_tiers": [
                    {
                        "up_to_input_tokens": 256_000,
                        "rates": {"input": 1.0, "output": 2.0},
                    },
                    {
                        "up_to_input_tokens": 128_000,
                        "rates": {"input": 1.0, "output": 2.0},
                    },
                ],
                "metadata": {"model_ids": ["model"]},
            },
        )


def test_monthly_gpu_pricing_requires_an_exact_allocated_gpu_hour_rate(
    tmp_path: Path,
) -> None:
    registry = PricingRegistry(tmp_path)
    registry.save_spec(
        "allocated-gpu-monthly-amortized",
        {
            "basis": "allocated_gpu_time",
            "billing_mode": "monthly_node_amortized",
            "supported_deployment_kinds": ["hf", "vllm"],
            "required_rates": ["gpu_hour"],
            "currency": "CNY",
            "rates": {"gpu_hour": 32657.28 / 8 / 730},
            "metadata": {
                "accelerator": "NVIDIA A800",
                "provider": "test-provider",
                "quoted_node_month": 32657.28,
                "node_gpu_count": 8,
            },
            "amortization_policy": "allocated_gpu_share",
            "amortization_hours_per_month": 730,
        },
    )
    saved = registry.instantiate_instance("allocated-gpu-monthly-amortized", "a800-monthly")
    assert saved["rates"]["gpu_hour"] == pytest.approx(5.592)

    with pytest.raises(ValueError, match="must equal quoted_node_month"):
        registry.save_spec(
            "a800-monthly-invalid",
            {
                **registry.get_spec("allocated-gpu-monthly-amortized"),
                "id": "a800-monthly-invalid",
                "rates": {"gpu_hour": 6.99},
            },
        )


def test_pricing_instance_is_instantiated_once_from_its_pricing_spec(
    tmp_path: Path,
) -> None:
    registry = PricingRegistry(tmp_path)

    saved = registry.instantiate_instance(
        "allocated-gpu-time",
        "internal-a800-v1",
    )

    assert saved["pricing_spec_id"] == "allocated-gpu-time"
    assert saved["rates"]["gpu_hour"] == 1.0
    assert "rates" not in json.loads(
        (registry.instance_root / "internal-a800-v1.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="already exists"):
        registry.instantiate_instance(
            "allocated-gpu-time",
            "internal-a800-v1",
        )


def test_pricing_instance_becomes_invalid_after_its_spec_changes(
    tmp_path: Path,
) -> None:
    registry = PricingRegistry(tmp_path)
    registry.instantiate_instance("allocated-gpu-time", "gpu-price-v1")
    current = registry.get_spec("allocated-gpu-time")
    registry.save_spec(
        "allocated-gpu-time",
        {**current, "rates": {"gpu_hour": 2.0}},
    )

    observed = next(item for item in registry.instances() if item["id"] == "gpu-price-v1")
    assert observed["status"] == "invalid"
    with pytest.raises(ValueError, match="stale"):
        registry.get_instance("gpu-price-v1")


def test_pricing_api_uses_spec_to_instance_instantiation(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    client = TestClient(create_app(tmp_path))
    payload = {"id": "internal-a800-v1"}

    response = client.post(
        "/api/pricing-specs/allocated-gpu-time/instances",
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["pricing_spec_id"] == "allocated-gpu-time"
    assert (
        client.post(
            "/api/pricing-specs/allocated-gpu-time/instances",
            json=payload,
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/pricing-instances/internal-a800-v1",
            json=payload,
        ).status_code
        == 405
    )


def test_token_pricing_instance_must_match_deployment_model(tmp_path: Path) -> None:
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    pricing = PricingRegistry(tmp_path)
    pricing.save_spec(
        "other-model-token-pricing",
        {
            "basis": "token_usage",
            "supported_deployment_kinds": ["api", "vllm"],
            "required_rates": ["input", "output"],
            "currency": "CNY",
            "rates": {"input": 1.0, "output": 2.0},
            "metadata": {
                "model_ids": ["other-model"],
                "source_spec_ids": ["remote-source"],
            },
        },
    )
    pricing.instantiate_instance("other-model-token-pricing", "other-model-token-pricing")
    spec = DeploymentRegistry(tmp_path).save(
        {
            "id": "remote-api",
            "kind": "api",
            "source_spec": {"type": "api", "id": "remote-source"},
            "actual_pricing_spec_id": "other-model-token-pricing",
        }
    )
    with pytest.raises(ValueError, match="does not apply to deployment model_id"):
        DeploymentRegistry(tmp_path).deploy(
            spec,
            instance_id="remote-instance",
            source_instance={"type": "api", "id": "remote-source-instance"},
            source_fields={
                "model_id": "expected-model",
                "base_url": "https://example.invalid/v1",
                "auth_mode": "none",
            },
            actual_pricing_instance_id="other-model-token-pricing",
        )


def test_pricing_spec_cannot_be_deleted_while_deployment_spec_references_it(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app
    from lychee_mas.eval.studio.deployments import DeploymentRegistry

    DeploymentRegistry(tmp_path).save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "source"},
            "actual_pricing_spec_id": "allocated-gpu-time",
        }
    )
    response = TestClient(create_app(tmp_path)).delete("/api/pricing-specs/allocated-gpu-time")
    assert response.status_code == 422
    assert "referenced by a DeploymentSpec" in response.json()["detail"]
