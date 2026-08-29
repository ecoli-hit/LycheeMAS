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
from lychee_mas.eval.application.read_models.run_events import read_jsonl_events, read_run_events
from lychee_mas.eval.benchmarks.assets import BenchmarkRegistry
from lychee_mas.eval.contracts.specs import TeamSpec
from lychee_mas.eval.environment.service import readme_environment
from lychee_mas.eval.experiments.compiler import (
    CompiledPlan,
    ExecutionPlanCompiler,
    default_project,
)
from lychee_mas.eval.experiments.lifecycle import finished_instance_projection
from lychee_mas.eval.experiments.registry import ExperimentRegistry
from lychee_mas.eval.pricing.registry import PricingRegistry
from lychee_mas.eval.scheduling.jobs import JobManager
from lychee_mas.eval.scheduling.manager import ExperimentQueueManager
from lychee_mas.eval.teams.compiler import load_team_spec
from lychee_mas.runtime.adapters.inference.deployment_pool import DeploymentPool
from lychee_mas.runtime.adapters.inference.hf import HFBackend
from lychee_mas.runtime.adapters.inference.openai_compatible import RequestRateLimiter
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


@pytest.fixture(autouse=True)
def _register_test_pricing(tmp_path: Path) -> None:
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

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
    canonical_reason = Path(__file__).resolve().parents[1] / (
        "configs/eval_studio/teams/specs/reason.json"
    )
    TeamSpecRegistry(tmp_path).save(json.loads(canonical_reason.read_text(encoding="utf-8")))


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
    _register_test_model_spec(root)
    spec = root / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "provider_key": "test",
                "organization": "Test Organization",
                "deployment_kind": "api",
                "allowed_model_spec_ids": ["test-qwen"],
                "default_base_url": "https://example.invalid/v1",
                "auth_modes": ["env"],
                "default_api_key_env": "MODEL_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    from lychee_mas.eval.apis.registry import APIRegistry

    APIRegistry(root).save_instance(
        {
            "id": "test-qwen",
            "api_spec_id": "test-api",
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
                "capabilities": {"text_generation": True, "latent": True},
                "supported_backends": ["hf", "vllm", "api"],
                "provider_model_ids": {"test": "qwen-plus"},
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
                    "role_id": node_id,
                    "deployment_instance_id": deployment_instance_id,
                }
                for node_id in [
                    node["id"]
                    for node in project["team"]["nodes"]
                    if node.get("kind") == "model_agent"
                ]
            ],
        }
    )


def _port(
    port_id: str,
    kind: str,
    semantic_type: str,
    *,
    required: bool | None = None,
    contract: dict | None = None,
) -> dict:
    value = {
        "id": port_id,
        "kind": kind,
        "semantic_type": semantic_type,
        "description": "",
        "schema": {},
        "contract": dict(contract or {}),
    }
    if required is not None:
        value["required"] = required
    return value


def _team_node(
    node_id: str,
    *,
    tools: list[str] | None = None,
) -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "kind": "model_agent",
        "behavior": {"type": "assistant", "options": {}},
        "purpose": node_id,
        "instructions": "",
        "capabilities": [],
        "operations": [],
        "tools": [{"id": item, "required": True} for item in (tools or [])],
        "context": {"type": "unbounded"},
        "limits": {"max_tool_iterations": 1},
    }


def _store_node(node_id: str, store_type: str) -> dict:
    semantic = {
        "input": "task",
        "artifact": "artifact",
        "message": "message",
        "state": "state",
        "result": "result",
    }[store_type]
    input_port = {
        "input": "initialize",
        "artifact": "write",
        "message": "append",
        "state": "merge",
        "result": "commit",
    }[store_type]
    output_port = {
        "input": "task",
        "artifact": "artifacts",
        "message": "messages",
        "state": "state",
        "result": "result",
    }[store_type]
    control_port = (
        "ready" if store_type == "input" else "committed" if store_type == "result" else "updated"
    )
    return {
        "id": node_id,
        "node_type": "store",
        "identity": {"name": node_id, "description": node_id},
        "interface": {
            "input_ports": [_port(input_port, "data", semantic, required=False)],
            "output_ports": [
                _port(output_port, "data", semantic),
                _port(control_port, "control", control_port),
            ],
            "activation": {"mode": "any"},
        },
        "implementation": {
            "kind": "store",
            "store_type": store_type,
            "lifetime": "trial",
            "update_policy": {
                "input": "initialize",
                "artifact": "append",
                "message": "append",
                "state": "merge",
                "result": "commit",
            }[store_type],
            "schema": {},
            "conditions": [],
        },
    }


def _edge(
    edge_id: str,
    edge_type: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    *,
    operation: str = "transfer",
    semantic_type: str = "generic",
) -> dict:
    value = {
        "id": edge_id,
        "edge_type": edge_type,
        "source": {"node": source_node, "port": source_port},
        "target": {"node": target_node, "port": target_port},
        "enabled": True,
        "description": "",
        "condition": {},
        "mapping": {},
        "policy": {},
    }
    if edge_type == "control":
        value["control"] = {
            "trigger": "source_emitted",
            "priority": 0,
            "fallback": "error",
            "on_failure": "fail_trial",
        }
    else:
        value["data"] = {"operation": operation, "semantic_type": semantic_type}
    return value


def _single_team_spec(team_id: str = "single", participant_id: str = "Solver") -> dict:
    return {
        "schema_version": 14,
        "id": team_id,
        "metadata": {
            "name": team_id,
            "description": "",
            "tags": [],
            "provenance": {
                "track": "custom",
                "evidence_level": "hypothesis",
                "sources": [],
                "notes": "",
            },
        },
        "nodes": [_team_node(participant_id)],
        "relations": [],
        "shared_state": [
            {
                "id": "SharedConversation",
                "kind": "message_channel",
                "description": "Trial conversation",
                "readers": [participant_id],
                "writers": [participant_id],
                "update": "append",
                "lifetime": "trial",
                "initial": [],
                "schema": {},
                "retention": {},
            },
            {
                "id": "TrialWorkspace",
                "kind": "artifact_store",
                "description": "Trial artifacts",
                "readers": [participant_id],
                "writers": [participant_id],
                "update": "commit",
                "lifetime": "trial",
                "initial": {},
                "schema": {},
                "retention": {},
            },
        ],
        "lifecycle": {
            "entry": [
                {
                    "node": participant_id,
                    "inputs": [
                        {
                            "source": "trial.task",
                            "target": "task",
                            "required": True,
                            "view": "all",
                            "schema": {},
                        }
                    ],
                }
            ],
            "result": {
                "submissions": [
                    {"from": participant_id, "source": "source.output", "key": "final_answer"}
                ],
                "mode": "first_valid",
                "schema": {},
            },
            "termination": {"condition": "result_submitted"},
            "failure": {"unhandled": "fail_trial", "deadlock": "fail_trial"},
            "limits": {},
        },
    }


def test_team_spec_value_object_matches_v14_persisted_contract() -> None:
    value = _single_team_spec()
    document = TeamSpec(
        id=value["id"],
        metadata=value["metadata"],
        nodes=value["nodes"],
        relations=value["relations"],
        shared_state=value["shared_state"],
        lifecycle=value["lifecycle"],
    ).to_dict()

    assert list(document) == [
        "schema_version",
        "id",
        "metadata",
        "nodes",
        "relations",
        "shared_state",
        "lifecycle",
    ]
    assert not ({"pattern", "edges", "execution_policy", "resources"} & set(document))


def _team_instance_document(
    instance_id: str,
    team_spec_id: str,
    node_ids: list[str],
    deployment_instance_id: str,
    *,
    runtime_framework: str = "autogen",
) -> dict:
    return {
        "schema_version": 5,
        "id": instance_id,
        "team_spec_id": team_spec_id,
        "runtime_framework": runtime_framework,
        "framework_options": {},
        "resource_bindings": [
            {
                "node_id": node_id,
                "requirement": "model_inference",
                "resource_instance_type": "DeploymentInstance",
                "resource_instance_id": deployment_instance_id,
            }
            for node_id in node_ids
        ],
    }


def _selector_team_spec(team_id: str = "selector-team") -> dict:
    members = ["Planner", "Solver"]
    selector = _team_node("Selector")
    selector["capabilities"] = ["coordinate", "select_next"]
    selector["operations"] = [
        {
            "type": "select_next",
            "options": {
            "mode": "model",
            "max_attempts": 3,
            "allow_repeat": False,
            "finish": {"allowed": True, "requires_result": True},
            },
        }
    ]
    value = _single_team_spec(team_id, members[0])
    participants = [_team_node(node_id) for node_id in members]
    value["nodes"] = [*participants, selector]
    value["relations"] = [
        {
            "id": f"selector-to-{node_id}",
            "from": "Selector",
            "to": node_id,
            "control": {
                "trigger": "completed",
                "action": "select_next",
                "priority": index,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "trial.task", "target": "task", "required": True},
                    {"source": "shared.SharedConversation", "target": "messages"},
                ]
            },
        }
        for index, node_id in enumerate(members)
    ]
    for state in value["shared_state"]:
        state["readers"] = [*members, "Selector"]
        state["writers"] = [*members, "Selector"]
    value["lifecycle"]["entry"] = [
        {
            "node": "Selector",
            "inputs": [{"source": "trial.task", "target": "task", "required": True}],
        }
    ]
    value["lifecycle"]["result"]["submissions"] = [
        {"from": "Solver", "source": "source.output", "key": "final_answer"}
    ]
    return value


def _instance_record(
    instance_id: str,
    deployment_spec_id: str,
    *,
    kind: str,
    model_id: str,
    source_type: str = "model",
    source_id: str = "test-source",
    model_spec_id: str | None = None,
    **options,
) -> dict:
    actual_pricing_instance_id = (
        "test-token-pricing"
        if source_type == "api" or options.get("managed") is False
        else "test-gpu-pricing"
    )
    return {
        "schema_version": 3,
        "id": instance_id,
        "deployment_spec_id": deployment_spec_id,
        "source_instance": {"type": source_type, "id": source_id},
        "kind": kind,
        "model_id": model_id,
        "model_spec_id": model_spec_id or (source_id if source_type == "model" else "test-qwen"),
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
    from lychee_mas.runtime.adapters.inference import openai_compatible as openai_api_backend

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
    assert "--trace-detail-level" not in plan.files["run.sh"]
    assert "--run-dir" in plan.files["run.sh"]
    assert "--event-log-max-events-per-file" in plan.files["run.sh"]
    assert "--event-log-max-mib-per-file" in plan.files["run.sh"]
    assert str(plan.run_dir) in plan.files["run.sh"]
    assert plan.run_dir.parent == (
        tmp_path / "runs/benchmarks/gsm8k/gsm8k/reason/none/eval-experiment"
    )
    assert re.fullmatch(r"\d{8}T\d{9}Z", plan.run_dir.name)
    assert "trace_detail_level" not in plan.files["runtime_config.yaml"]
    runtime_config = json.loads(plan.files["runtime_config.yaml"])
    assert runtime_config["experiment"]["instance_id"] == "unit-plan"
    assert runtime_config["runtime"]["max_turns"] == 12
    assert runtime_config["runtime"]["code_timeout"] == 60
    assert runtime_config["runtime"]["work_root"] == "runs/lychee_tool_workspaces"
    assert runtime_config["runtime"]["web_headless"] is True
    assert runtime_config["runtime"]["save_screenshots"] is False
    assert runtime_config["run"]["max_model_calls_per_case"] is None
    assert runtime_config["run"]["max_case_wall_time_s"] is None
    assert runtime_config["run"]["max_attempts_per_trial"] == 1
    assert runtime_config["run"]["on_trial_error"] == "continue"
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
        "event_log_max_events_per_file": 10000,
        "event_log_max_mib_per_file": 64.0,
    }
    assert "--collect-vllm-metrics" in plan.files["run.sh"]
    assert "--vllm-metrics-interval-s \\\n  5.0" in plan.files["run.sh"]
    assert "--max-model-calls-per-case \\\n  unlimited" in plan.files["run.sh"]
    assert "--max-case-wall-time-s \\\n  unlimited" in plan.files["run.sh"]
    assert "--max-attempts-per-trial \\\n  1" in plan.files["run.sh"]
    assert "--on-trial-error \\\n  continue" in plan.files["run.sh"]
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


def test_runtime_snapshot_names_the_selected_framework(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["runtime"]["framework"] = "langgraph"

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="langgraph-plan")
    runtime_config = json.loads(plan.files["runtime_config.yaml"])

    assert runtime_config["runtime"]["name"] == "langgraph"
    assert runtime_config["runtime"]["framework"] == "langgraph"


def test_hle_external_evaluator_requires_declared_judge_config(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["benchmark"].update(
        task="hle_verified",
        benchmark_spec_id="hle_verified",
        evaluation_requirements={
            "config_key": "hle_judge",
            "required_when_external_evaluator": ["auto", "run"],
            "required_fields": ["model", "base_url"],
        },
    )
    project["evaluation"] = {
        "external_evaluator": "run",
        "profile_id": "core",
        "hle_judge": {},
    }

    with pytest.raises(ValueError, match="requires evaluation.hle_judge"):
        ExecutionPlanCompiler(tmp_path).compile(project, launch_id="missing-hle-judge")


def test_execution_plan_preserves_hle_judge_auth_contract(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["evaluation"] = {
        "external_evaluator": "run",
        "profile_id": "core",
        "hle_judge": {
            "model": "local-judge",
            "base_url": "http://127.0.0.1:6200/v1",
            "auth_mode": "none",
            "api_key_env": "UNSET_LOCAL_KEY",
            "workers": 2,
            "timeout_s": 30.0,
            "max_tokens": 512,
            "thinking_mode": "disabled",
            "output_mode": "local_json_object",
            "max_attempts": 3,
        },
    }

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="hle-auth")

    assert "--hle-judge-auth-mode" in plan.files["analyze.sh"]
    assert "--hle-judge-thinking-mode" in plan.files["analyze.sh"]
    assert "--hle-judge-output-mode" in plan.files["analyze.sh"]
    assert "none" in plan.files["analyze.sh"]
    assert plan.project["evaluation"]["hle_judge"]["auth_mode"] == "none"
    assert plan.project["evaluation"]["hle_judge"]["thinking_mode"] == "disabled"
    assert plan.project["evaluation"]["hle_judge"]["output_mode"] == "local_json_object"


def test_experiment_registry_preserves_hle_judge_auth_contract(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "hle-local-judge"
    spec["evaluation"] = {
        "external_evaluator": "run",
        "profile_id": "core",
        "hle_judge": {
            "model": "local-judge",
            "base_url": "http://127.0.0.1:6200/v1",
            "auth_mode": "none",
            "api_key_env": "UNSET_LOCAL_KEY",
            "workers": 2,
            "timeout_s": 30.0,
            "max_tokens": 512,
            "thinking_mode": "disabled",
            "output_mode": "local_json_object",
            "max_attempts": 3,
        },
    }

    saved = registry.save_spec(spec["id"], spec)

    assert saved["evaluation"]["hle_judge"]["auth_mode"] == "none"
    assert saved["evaluation"]["hle_judge"]["thinking_mode"] == "disabled"
    assert saved["evaluation"]["hle_judge"]["output_mode"] == "local_json_object"


def test_dynamic_coordination_uses_max_turns_and_model_call_limit(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    selector = _team_node("Selector")
    selector["capabilities"] = ["coordinate", "select_next"]
    selector["operations"] = [
        {
            "type": "select_next",
            "options": {
                "mode": "model",
                "max_attempts": 3,
                "allow_repeat": False,
                "finish": {"allowed": True, "requires_result": True},
            },
        }
    ]
    project["team"]["nodes"].append(selector)
    project["deployment_bindings"]["role_bindings"].append(
        {
            "role_id": "Selector",
            "deployment_instance_id": project["deployment_bindings"][
                "default_deployment_instance_id"
            ],
        }
    )
    project["team"]["relations"].extend(
        {
            "id": f"control-Selector-{member_id}",
            "from": "Selector",
            "to": member_id,
            "control": {
                "trigger": "completed",
                "action": "select_next",
                "priority": index,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "trial.task", "target": "task", "required": True},
                    {"source": "shared.SharedConversation", "target": "messages"},
                ]
            },
        }
        for index, member_id in enumerate(("planner", "solver", "verifier"))
    )
    for state in project["team"]["shared_state"]:
        if "Selector" not in state["readers"]:
            state["readers"].append("Selector")
        if "Selector" not in state["writers"]:
            state["writers"].append("Selector")
    project["team"]["lifecycle"]["entry"] = [
        {
            "node": "Selector",
            "inputs": [
                {"source": "trial.task", "target": "task", "required": True}
            ],
        }
    ]
    project["runtime"].update(
        max_rounds=99,
        max_turns=27,
        max_model_calls_per_case=41,
        max_case_wall_time_s=1234,
        max_attempts_per_trial=3,
        on_trial_error="fail-fast",
    )

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="dynamic-limits")
    runtime_config = json.loads(plan.files["runtime_config.yaml"])

    assert plan.project["runtime"]["effective_max_turns"] == 27
    assert runtime_config["runtime"]["max_turns"] == 27
    assert runtime_config["run"]["max_model_calls_per_case"] == 41
    assert runtime_config["run"]["max_case_wall_time_s"] == 1234
    assert runtime_config["run"]["max_attempts_per_trial"] == 3
    assert runtime_config["run"]["on_trial_error"] == "fail-fast"
    assert "--max-turns \\\n  27" in plan.files["run.sh"]
    assert "--max-model-calls-per-case \\\n  41" in plan.files["run.sh"]
    assert "--max-case-wall-time-s \\\n  1234.0" in plan.files["run.sh"]
    assert "--max-attempts-per-trial \\\n  3" in plan.files["run.sh"]
    assert "--on-trial-error \\\n  fail-fast" in plan.files["run.sh"]


def test_direct_team_respects_explicit_max_turns_below_round_limit(
    tmp_path: Path,
) -> None:
    project = default_project(tmp_path)
    project["team"] = _single_team_spec("reason", "planner")
    project["deployment_bindings"]["role_bindings"] = [
        item
        for item in project["deployment_bindings"]["role_bindings"]
        if item["role_id"] == "planner"
    ]
    project["runtime"].update(max_rounds=4, max_turns=1)

    plan = ExecutionPlanCompiler(tmp_path).compile(
        project,
        launch_id="explicit-direct-turn-limit",
    )
    runtime_config = json.loads(plan.files["runtime_config.yaml"])

    assert plan.project["runtime"]["effective_max_turns"] == 1
    assert runtime_config["runtime"]["max_turns"] == 1
    assert "--max-turns \\" + "\n  1" in plan.files["run.sh"]


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


def test_execution_plan_has_one_lossless_event_recording_mode(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    plan = ExecutionPlanCompiler(tmp_path).compile(project)
    assert "trace_detail_level" not in plan.project["runtime"]
    assert "trace-detail-level" not in plan.files["run.sh"]


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


def test_experiment_download_proxy_is_exported_to_runtime_scripts(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["network"] = {
        "mode": "proxy",
        "proxy_url": "http://127.0.0.1:7897",
        "no_proxy": "127.0.0.1,localhost,::1",
        "targets": {"downloads": True},
    }

    plan = ExecutionPlanCompiler(tmp_path).compile(project, launch_id="download-proxy")

    for script_name in ("prepare.sh", "run.sh"):
        script = plan.files[script_name]
        assert "export HTTP_PROXY=http://127.0.0.1:7897" in script
        assert "export HTTPS_PROXY=http://127.0.0.1:7897" in script
        assert "export NO_PROXY=127.0.0.1,localhost,::1" in script


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


def test_vllm_and_api_are_rejected_for_latent(tmp_path: Path) -> None:
    project = default_project(tmp_path)
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
        ExecutionPlanCompiler(tmp_path).normalize(project)


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


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "name"),
        ("benchmark", "display_name"),
        ("runtime", "max_stalls"),
        ("observability", "compact"),
        ("network", "use_proxy"),
        ("environment", "cache_root"),
        ("environment", "run_dir"),
    ],
)
def test_experiment_spec_rejects_fields_without_runtime_effect(
    tmp_path: Path,
    section: str | None,
    field: str,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "strict-contract"
    if section is None:
        spec[field] = True
    else:
        spec[section][field] = True

    with pytest.raises(ValueError, match=field):
        registry.save_spec("strict-contract", spec)


def test_experiment_spec_accepts_all_cases_and_validates_trials(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "all-cases"
    spec["benchmark"]["cases"] = "ALL"
    spec["runtime"]["trials_per_case"] = 3

    saved = registry.save_spec("all-cases", spec)

    assert saved["benchmark"]["cases"] == "all"
    assert saved["runtime"]["trials_per_case"] == 3
    assert saved["observability"] == {
        "collect_vllm_metrics": True,
        "vllm_metrics_interval_s": 5.0,
        "event_log_max_events_per_file": 10000,
        "event_log_max_mib_per_file": 64.0,
    }

    spec["runtime"]["trials_per_case"] = 0
    with pytest.raises(ValueError, match="runtime.trials_per_case must be positive"):
        registry.save_spec("all-cases", spec)


def test_experiment_spec_rejects_runtime_fields_that_cannot_be_configured(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "no-phantom-runtime-controls"
    spec["runtime"]["max_stalls"] = 3

    with pytest.raises(ValueError, match="unsupported fields: max_stalls"):
        registry.save_spec("no-phantom-runtime-controls", spec)


def test_experiment_spec_rejects_unknown_concurrency_policy_fields(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "strict-concurrency"
    spec["runtime"]["concurrency_policy"]["gpu_target"] = 0.9

    with pytest.raises(ValueError, match="unsupported fields: gpu_target"):
        registry.save_spec("strict-concurrency", spec)


def test_saved_custom_runs_root_is_visible_through_runs_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    custom_root = tmp_path.parent / f"{tmp_path.name}-custom-runs"
    run_dir = custom_root / "model/team_none/gsm8k"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")
    writer = RunEventWriter(run_events_path(run_dir))
    writer.log_event("trial.started", case_id="case-1", dataset_index=0)
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
    events = client.get(f"/api/runs/{row['id']}/run-events").json()
    assert events["events"][0]["event_type"] == "trial.started"


def test_custom_team_spec_preserves_edges_without_deployments(tmp_path: Path) -> None:
    path = tmp_path / "team.json"
    selector = _team_node("Selector")
    selector.update(
        kind="function",
        behavior={
            "type": "round_robin_selector",
            "options": {},
        },
        capabilities=["coordinate", "select_next"],
        operations=[
            {
                "type": "select_next",
                "options": {
                "mode": "deterministic",
                "max_attempts": 3,
                "allow_repeat": False,
                "finish": {"allowed": False, "requires_result": True},
                },
            },
        ],
    )
    value = _single_team_spec("custom", "Planner")
    members = ["Planner", "Solver"]
    nodes = [_team_node(member) for member in members] + [selector]
    relations = [
        {
            "id": "planner-to-solver",
            "from": "Planner",
            "to": "Solver",
            "control": {
                "trigger": "completed",
                "action": "activate",
                "priority": 0,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "source.output", "target": "messages", "view": "latest"}
                ]
            },
        },
        *[
            {
                "id": f"selector-to-{member}",
                "from": "Selector",
                "to": member,
                "control": {
                    "trigger": "completed",
                    "action": "select_next",
                    "priority": index,
                    "on_failure": "fail_trial",
                },
            }
            for index, member in enumerate(members)
        ],
    ]
    for state in value["shared_state"]:
        state["readers"] = [*members, "Selector"]
        state["writers"] = [*members, "Selector"]
    value["lifecycle"]["entry"] = [
        {"node": "Selector", "inputs": [{"source": "trial.task", "target": "task"}]}
    ]
    value["lifecycle"]["result"]["submissions"] = [
        {"from": "Solver", "source": "source.output", "key": "final_answer"}
    ]
    path.write_text(
        json.dumps({**value, "nodes": nodes, "relations": relations}),
        encoding="utf-8",
    )
    graph, _ = load_team_spec(path, rounds=3)
    assert graph.names == ["Planner", "Solver"]
    assert graph.edges == {"Planner": ["Solver"], "Solver": []}
    assert graph.nodes[1].meta["receives_from"] == ["Planner"]
    assert "deployment_id" not in graph.nodes[1].meta
    assert "control_deployment_id" not in graph.meta
    assert graph.meta["adapter_plans"]["autogen"]["config"] == {
        "type": "selector",
        "selector_func_factory": "operation_selector",
        "max_selector_attempts": 3,
        "allow_repeated_speaker": False,
        "candidate_node_ids": ["Planner", "Solver"],
    }
    assert graph.meta["context_visibility"] == "shared"


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
    from lychee_mas.eval.teams.compiler import load_team_spec

    graph = load_team_spec(
        Path(__file__).resolve().parents[1]
        / "configs/eval_studio/teams/specs/reason.json",
        rounds=2,
    )[0]
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
    project["team"] = _single_team_spec("single", "Solver")
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
    project["team"] = _single_team_spec("single", "Solver")
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


def test_team_spec_rejects_obsolete_persisted_coordination(tmp_path: Path) -> None:
    project = default_project(tmp_path)
    project["team"]["coordination"] = {
        "pattern": "fixed_order",
        "members": ["planner", "solver", "verifier"],
        "operation_bindings": [],
        "options": {},
    }
    with pytest.raises(ValueError, match="unsupported fields: coordination"):
        ExecutionPlanCompiler(tmp_path).normalize(project)


def test_incremental_event_reader(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["repo_root"] == str(tmp_path)


def test_studio_api_derives_results_and_trace_from_run_events(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "runs/benchmarks/model/team/task"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "task": "task"}), encoding="utf-8"
    )
    writer = RunEventWriter(run_events_path(run_dir))
    trial_id = writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    model_id = writer.log_event("model_call.started", case_id="case-1", dataset_index=0)
    writer.log_event(
        "model_call.completed",
        operation_id=model_id,
        case_id="case-1",
        dataset_index=0,
        input_total_positions=2,
        output_total_tokens=1,
        model_latency_s=0.1,
    )
    writer.log_event(
        "agent.message.published",
        case_id="case-1",
        dataset_index=0,
        source="Coder",
        content="hello",
    )
    writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "operation_id": trial_id,
            "final_output": "hello",
        }
    )

    client = TestClient(create_app(tmp_path))
    row = client.get("/api/runs").json()[0]
    assert row["has_result_projection"] is True
    assert row["has_execution_trace"] is True
    results = client.get(f"/api/runs/{row['id']}/results").json()["trials"]
    trace = client.get(f"/api/runs/{row['id']}/execution-trace").json()["nodes"]
    assert results[0]["prediction"] == "hello"
    assert results[0]["model_call_count"] == 1
    assert any(node["event_type"] == "model_call" for node in trace)


def test_studio_api_reports_unsupported_run_event_schema(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "runs/benchmarks/model/team/legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "complete", "task": "legacy"}), encoding="utf-8"
    )
    run_events_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    run_events_path(run_dir).write_text(
        json.dumps({"schema_version": 1, "event_type": "trial.completed"}) + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(tmp_path))
    row = client.get("/api/runs").json()[0]
    for endpoint in ("results", "execution-trace"):
        response = client.get(f"/api/runs/{row['id']}/{endpoint}")
        assert response.status_code == 422
        assert "unsupported or invalid RunEvent log" in response.json()["detail"]


def test_studio_pages_run_events_across_segments(tmp_path: Path) -> None:
    writer = RunEventWriter(
        run_events_path(tmp_path),
        max_events_per_file=2,
        max_bytes_per_file=1024 * 1024,
    )
    for index in range(5):
        writer.log_event("agent.message.published", source="Solver", content=str(index))

    page = read_run_events(tmp_path, start_line=1, limit=3)
    assert [event["seq"] for event in page["events"]] == [2, 3, 4]
    assert page["next_line"] == 4
    assert page["total_lines"] == 5
    assert page["segment_count"] == 3
    assert page["has_more"] is True


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
                "control_protocol_version": 2,
                "status": "running",
                "expected_trials": 10,
                "successful_trials": 3,
                "failed_trials": 1,
                "skipped_trials": 0,
                "effective_trial_concurrency": 2,
                "started_at_unix_s": 100.0,
                "finished_at_unix_s": 112.5,
                "active_cases": [
                    {
                        "worker_id": 1,
                        "case_order": 5,
                        "num_cases": 10,
                        "case_id": "case-5",
                        "trial_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)

    assert progress["percent"] == 40
    assert progress["completed_trials"] == 4
    assert progress["remaining_trials"] == 6
    assert progress["trial_concurrency"] == 2
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
                "completed_trials": 2,
                "expected_trials": 2,
                "lines": ["done"],
            }

    manager = ExperimentQueueManager(registry, None, Jobs(), None)
    observed = manager.instances()[0]

    assert observed["status"] == "completed"
    assert observed["progress"]["percent"] == 100
    assert registry.get_instance("recovered")["status"] == "completed"


def test_experiment_progress_uses_instance_case_target_for_percent(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "segment-progress"
    registry.save_spec(spec["id"], spec)
    created = registry.create_instance(
        instance_id="segment-progress-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir=str(tmp_path / "runs/benchmarks/segment-progress"),
        execution={"next_segment_case_limit": 3, "case_completion_target": 3},
    )
    launch_dir = tmp_path / "runs/eval_studio/launches/segment-progress"
    registry.update_instance(
        created["id"],
        status="paused",
        launch_dir=str(launch_dir),
    )

    class Jobs:
        @staticmethod
        def progress(_launch_dir, *, tail):
            return {
                "state": {"status": "completed", "return_code": 0},
                "percent": 1,
                "completed_distinct_cases": 3,
                "successful_trials": 3,
                "failed_trials": 0,
                "lines": [],
            }

    observed = ExperimentQueueManager(registry, None, Jobs(), None).instances()[0]

    assert observed["progress"]["dataset_percent"] == 1
    assert observed["progress"]["percent"] == 100
    assert observed["progress"]["remaining_to_case_target"] == 0


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
                "expected_trials": 10,
                "successful_trials": 2,
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
    assert status["successful_trials"] == 2
    assert status["finished_at_unix_s"] >= status["updated_at_unix_s"]
    assert list(run_dir.glob(".run_status.*.tmp")) == []


def test_job_request_drain_preserves_active_trials_and_persists_control(
    tmp_path: Path,
) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/draining"
    run_dir = tmp_path / "runs/benchmarks/draining"
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    JobManager._write_state(
        launch_dir,
        {
            "launch_id": "draining",
            "status": "running",
            "run_dir": str(run_dir),
        },
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "control_protocol_version": 2,
                "status": "running",
                "expected_trials": 10,
                "successful_trials": 3,
                "failed_trials": 0,
                "active_cases": [
                    {"case_id": "case-4", "worker_id": 0},
                    {"case_id": "case-5", "worker_id": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    manager = JobManager(ExecutionPlanCompiler(tmp_path))
    state = manager.request_drain(launch_dir)
    control = json.loads((run_dir / "run_control.json").read_text(encoding="utf-8"))
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    progress = manager.progress(launch_dir)

    assert state["status"] == "running"
    assert state["drain_requested"] is True
    assert control["action"] == "stop_after_active_trials"
    assert run_status["status"] == "draining"
    assert run_status["accepting_new_trials"] is False
    assert run_status["active_cases"] == [
        {"case_id": "case-4", "worker_id": 0},
        {"case_id": "case-5", "worker_id": 1},
    ]
    assert run_status["active_trials_at_stop_request"] == 2
    assert progress["drain_requested"] is True
    assert progress["drain_available"] is True
    assert progress["current_cases"] == run_status["active_cases"]


def test_job_request_drain_rejects_runner_without_control_protocol(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/legacy"
    run_dir = tmp_path / "runs/benchmarks/legacy"
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    JobManager._write_state(
        launch_dir,
        {"launch_id": "legacy", "status": "running", "run_dir": str(run_dir)},
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "active_cases": [{"case_id": "old"}]}),
        encoding="utf-8",
    )

    manager = JobManager(ExecutionPlanCompiler(tmp_path))
    with pytest.raises(ValueError, match="started before graceful Trial draining"):
        manager.request_drain(launch_dir)

    assert not (run_dir / "run_control.json").exists()


def test_forced_stop_projects_persisted_run_terminal_state(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "stop-projection"
    registry.save_spec(spec["id"], spec)
    run_dir = tmp_path / "runs/benchmarks/stop-projection"
    launch_dir = tmp_path / "runs/eval_studio/launches/stop-projection-run"
    run_dir.mkdir(parents=True)
    launch_dir.mkdir(parents=True)
    created = registry.create_instance(
        instance_id="stop-projection-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir=str(run_dir),
    )
    registry.update_instance(
        created["id"],
        status="running",
        launch_dir=str(launch_dir),
        queue={"enabled": True, "priority": 10},
        run_status="running",
    )

    class Jobs:
        @staticmethod
        def stop(_launch_dir):
            (run_dir / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "stopped",
                        "completed_distinct_cases": 7,
                    }
                ),
                encoding="utf-8",
            )
            return {
                "status": "stopped",
                "return_code": -15,
                "run_dir": str(run_dir),
            }

    manager = ExperimentQueueManager(registry, None, Jobs(), None)
    stopped = manager.stop_instance(created["id"])["instance"]

    assert stopped["status"] == "stopped"
    assert stopped["job_status"] == "stopped"
    assert stopped["run_status"] == "stopped"
    assert stopped["queue"]["enabled"] is False
    assert stopped["evaluation_status"] == "pending"


def test_forced_stop_cleanup_only_removes_launch_workspace_containers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lychee_mas.eval.scheduling import jobs as jobs_module

    run_dir = tmp_path / "runs/benchmarks/example"
    event_dir = run_dir / "events"
    event_dir.mkdir(parents=True)
    workspace = tmp_path / "runs/lychee_tool_workspaces/ws_owned"
    (event_dir / "run_events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "workspace.prepared",
                "payload": {
                    "workspace": str(workspace.relative_to(tmp_path)),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if command[:3] == ["docker", "ps", "-aq"]:
            return SimpleNamespace(returncode=0, stdout="owned-container\n", stderr="")
        if command[:3] == ["docker", "rm", "-f"]:
            return SimpleNamespace(returncode=0, stdout="owned-container\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(jobs_module.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(jobs_module.subprocess, "run", fake_run)
    result = JobManager(ExecutionPlanCompiler(tmp_path))._cleanup_launch_docker_sandboxes(
        {"run_dir": str(run_dir)}
    )

    assert result == {
        "status": "completed",
        "workspace_count": 1,
        "removed": ["owned-container"],
        "errors": [],
    }
    assert commands == [
        ["docker", "ps", "-aq", "--filter", f"volume={workspace}"],
        ["docker", "rm", "-f", "owned-container"],
    ]


def test_gracefully_stopped_job_keeps_partial_progress(tmp_path: Path) -> None:
    launch_dir = tmp_path / "runs/eval_studio/launches/drained"
    run_dir = tmp_path / "runs/benchmarks/drained"
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    JobManager._write_state(
        launch_dir,
        {"launch_id": "drained", "status": "running", "run_dir": str(run_dir)},
    )
    (launch_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "expected_trials": 10,
                "successful_trials": 3,
                "failed_trials": 0,
                "active_cases": [],
            }
        ),
        encoding="utf-8",
    )

    progress = JobManager(ExecutionPlanCompiler(tmp_path)).progress(launch_dir)

    assert progress["state"]["status"] == "completed"
    assert progress["run_status"] == "stopped"
    assert progress["percent"] == 30
    assert progress["remaining_trials"] == 7


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


def test_run_mas_sigterm_allows_async_runtime_cleanup(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    cleaned = tmp_path / "cleaned"
    script_path = Path(__file__).parents[1] / "scripts/run_mas.py"
    code = f"""
import argparse
import asyncio
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("lychee_test_run_mas", {str(script_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

async def fake_run_one(_cfg, _args):
    Path({str(ready)!r}).write_text("ready", encoding="utf-8")
    try:
        await asyncio.Event().wait()
    finally:
        Path({str(cleaned)!r}).write_text("cleaned", encoding="utf-8")

module.run_one = fake_run_one
raise SystemExit(
    asyncio.run(module._run_with_termination_cleanup({{}}, argparse.Namespace()))
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
        assert cleaned.read_text(encoding="utf-8") == "cleaned"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
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
    from lychee_mas.eval.models.registry import ModelRegistry

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
    from lychee_mas.eval.benchmarks.task_config import TASK_CONFIG

    assert TASK_CONFIG
    assert all("team" not in config for config in TASK_CONFIG.values())


def test_model_spec_catalog_separates_platform_sources_without_creating_instance(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.models.registry import ModelRegistry

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
    from lychee_mas.eval.benchmarks import assets as benchmark_module

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
    from lychee_mas.eval.environment.service import _PROBE

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
    from lychee_mas.eval.models.registry import ModelRegistry

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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from lychee_mas.eval.apis.registry import APIRegistry

    _register_test_model_spec(tmp_path)
    spec = tmp_path / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "provider_key": "test",
                "deployment_kind": "api",
                "allowed_model_spec_ids": ["test-qwen"],
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from lychee_mas.eval.application.read_models.catalog import EvalCatalog

    rows = EvalCatalog(tmp_path).benchmarks()
    by_key = {item["benchmark_key"]: item for item in rows}
    assert by_key["gsm8k"]["scenario"] == "math"
    assert by_key["gaia"]["scenario"] == "general_assistant"
    assert by_key["bbeh"]["download_sources"]["github"] == [
        "https://github.com/google-deepmind/bbeh.git"
    ]
    assert by_key["hle"]["download_sources"]["github"] == []
    assert by_key["hle"]["reference_sources"][0]["provider"] == "github"


def test_persistent_deployment_registry_round_trip(tmp_path: Path) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    saved = registry.save(
        {
            "id": "shared-vllm",
            "kind": "vllm",
            "model_spec_id": "qwen-model",
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
            "model_spec_id": "glm-model",
            "source_spec": {"type": "api", "id": "shared-glm-api"},
            "managed": False,
            "shared": True,
        },
        instance_id="instance-shared-glm",
        source_instance={"type": "api", "id": "shared-glm-api-instance"},
        source_fields={
            "model_id": "GLM-4.7-Flash",
            "model_info": {},
            "capabilities": {"text_generation": True},
            "thinking_protocol": None,
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
    from lychee_mas.eval.teams.instances import TeamInstanceRegistry
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    teams = TeamSpecRegistry(tmp_path)
    team = teams.save(_single_team_spec("research-team"))
    registry = TeamInstanceRegistry(tmp_path)
    saved = registry.save(
        {
            **_team_instance_document(
                "research-team-local",
                "research-team",
                ["Solver"],
                "instance-local",
            ),
            "resource_bindings": [
                {
                    "node_id": "Solver",
                    "requirement": "model_inference",
                    "resource_instance_type": "DeploymentInstance",
                    "resource_instance_id": "instance-local",
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
    changed_node = {
        **team["nodes"][0],
        "name": "Changed",
    }
    teams.save({**team, "nodes": [changed_node, *team["nodes"][1:]]})
    loaded = registry.all()[0]
    assert loaded["team_spec"]["nodes"][0]["name"] == "Changed"
    assert loaded["resource_bindings"][0]["generation_overrides"]["max_new_tokens"] == 2048
    assert loaded["resource_bindings"][0]["generation_overrides"]["thinking_mode"] == "disabled"
    assert loaded["resource_bindings"][0]["generation_overrides"]["preserve_thinking"] is False
    assert saved["team_spec_id"] == "research-team"
    assert registry.delete("research-team-local")["deleted"] is True


def test_persistent_team_spec_registry_round_trip(tmp_path: Path) -> None:
    from lychee_mas.eval.teams.registry import TeamSpecRegistry
    from lychee_mas.runtime.coordination.compiler import compile_coordination

    registry = TeamSpecRegistry(tmp_path)
    saved = registry.save(_selector_team_spec("research-team"))
    assert Path(saved["registry_path"]).is_file()
    assert saved["schema_version"] == 14
    assert set(saved) >= {
        "metadata",
        "nodes",
        "relations",
        "shared_state",
        "lifecycle",
    }
    assert not ({"pattern", "edges", "execution_policy", "resources"} & set(saved))
    coordination = compile_coordination(registry.get("research-team"))
    assert "pattern" not in coordination
    assert coordination["operations"]["select_next"][0]["options"]["mode"] == "model"
    assert registry.delete("research-team")["deleted"] is True
    assert [item["id"] for item in registry.all()] == ["reason"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda spec: spec.update(typo_field=True), "TeamSpec contains unsupported fields"),
        (
            lambda spec: spec["nodes"][0].update(max_tool_iteration=3),
            "nodes\\[0\\] contains unsupported fields: max_tool_iteration",
        ),
        (
            lambda spec: spec["nodes"][0]["limits"].update(max_tool_iteration=3),
            "limits contains unsupported fields: max_tool_iteration",
        ),
        (
            lambda spec: spec["nodes"][0]["context"].update(message_filter="shared"),
            "unbounded model context does not accept fields: message_filter",
        ),
    ],
)
def test_team_spec_rejects_fields_that_runtime_would_ignore(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    spec = _single_team_spec("strict-team")
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        TeamSpecRegistry(tmp_path).save(spec)


def test_manual_api_key_is_session_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
            "model_spec_id": "test-qwen",
            "source_spec": {"type": "api", "id": "manual-api-resource"},
        },
        instance_id="instance-manual-api",
        source_instance={"type": "api", "id": "manual-api-resource-instance"},
        source_fields={
            "model_id": "qwen-plus",
            "model_info": {},
            "capabilities": {"text_generation": True},
            "thinking_protocol": None,
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    with pytest.raises(ValueError, match="cannot bind resource Instance fields"):
        registry.save(
            {
                "id": "unsafe-api",
                "kind": "api",
                "model_spec_id": "test-model",
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path))
    payload = _single_team_spec("api-team")
    saved = client.put("/api/team-specs/api-team", json=payload)
    assert saved.status_code == 200
    listing = client.get("/api/team-specs").json()
    assert [item["id"] for item in listing["specs"]] == ["api-team", "reason"]
    assert client.delete("/api/team-specs/api-team").status_code == 200


def test_team_instance_api_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path))
    assert (
        client.put(
            "/api/team-specs/api-team",
            json=_single_team_spec("api-team"),
        ).status_code
        == 200
    )
    payload = {
        **_team_instance_document(
            "api-team-instance",
            "api-team",
            ["Solver"],
            "instance-local-hf",
        ),
        "resource_bindings": [
            {
                "node_id": "Solver",
                "requirement": "model_inference",
                "resource_instance_type": "DeploymentInstance",
                "resource_instance_id": "instance-local-hf",
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.deployments.registry import DeploymentRegistry
    from lychee_mas.eval.teams.instances import TeamInstanceRegistry
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    _register_gsm8k_benchmark(tmp_path)
    deployments = DeploymentRegistry(tmp_path)
    deployments.save(
        {
            "id": "local",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "test-model"},
        }
    )
    deployments._write_instance(
        _instance_record(
            "instance-local",
            "local",
            kind="hf",
            model_id="Qwen",
            model_spec_id="test-model",
            model_path=str(tmp_path),
            python=sys.executable,
            status="ready_on_run",
            capabilities={"text_generation": True},
        )
    )
    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    TeamInstanceRegistry(tmp_path).save(
        _team_instance_document("single-local", "single", ["Solver"], "instance-local")
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
    experiment["observability"].update(
        event_log_max_events_per_file=321,
        event_log_max_mib_per_file=7.5,
    )
    experiment["evaluation"] = {
        "external_evaluator": "run",
        "profile_id": "core",
        "hle_judge": {
            "model": "local-judge",
            "base_url": "http://127.0.0.1:6200/v1",
            "auth_mode": "none",
            "api_key_env": "UNSET_LOCAL_KEY",
            "workers": 2,
            "timeout_s": 30.0,
            "max_tokens": 512,
            "thinking_mode": "disabled",
            "output_mode": "local_json_object",
            "max_attempts": 3,
        },
    }
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
    assert created.status_code == 200, created.text
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
    assert plan["project"]["observability"]["event_log_max_events_per_file"] == 321
    assert plan["project"]["evaluation"]["hle_judge"]["auth_mode"] == "none"
    assert plan["project"]["evaluation"]["hle_judge"]["thinking_mode"] == "disabled"
    assert (
        plan["project"]["evaluation"]["hle_judge"]["output_mode"]
        == "local_json_object"
    )
    assert "--event-log-max-events-per-file \\\n  321" in plan["files"]["run.sh"]
    assert "--hle-judge-auth-mode" in plan["files"]["analyze.sh"]
    assert "none" in plan["files"]["analyze.sh"]
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
    import lychee_mas.eval.deployments.registry as module

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
            "host": "127.0.0.2",
            "port": 8123,
            "max_model_len": 131072,
            "gpu_memory_utilization": 0.85,
            "dtype": "bfloat16",
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
            "enforce_eager": True,
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
    assert command[command.index("--host") + 1] == "127.0.0.2"
    assert command[command.index("--port") + 1] == "8123"
    assert command[command.index("--max-model-len") + 1] == "131072"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-num-seqs") + 1] == "8"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    reasoning_config = json.loads(command[command.index("--reasoning-config") + 1])
    assert reasoning_config["reasoning_start_str"] == "<think>"
    assert reasoning_config["reasoning_end_str"] == "Stop reasoning now.</think>"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--enable-prefix-caching" in command
    assert "--enforce-eager" in command
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_model_len", 0, "max_model_len must be positive"),
        (
            "gpu_memory_utilization",
            0,
            "gpu_memory_utilization must be greater than 0 and at most 1",
        ),
        (
            "gpu_memory_utilization",
            1.1,
            "gpu_memory_utilization must be greater than 0 and at most 1",
        ),
        ("port", 70000, "port must be between 1 and 65535"),
    ],
)
def test_managed_vllm_rejects_invalid_runtime_limits(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    spec = {
        "id": "local-vllm",
        "kind": "vllm",
        "source_spec": {"type": "model", "id": "qwen-model"},
        "python": "/env/bin/python",
        "cuda_visible_devices": "4,5",
        "tensor_parallel_size": 1,
        "data_parallel_size": 2,
        "max_num_seqs": 8,
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        DeploymentRegistry(tmp_path).validate(spec)


def test_vllm_enabled_per_request_metrics_rejects_unsupported_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.deployments.registry as module

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
    import lychee_mas.eval.deployments.registry as module

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


def test_managed_vllm_deploy_retry_reuses_live_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.deployments.registry as module

    model = tmp_path / "models/Qwen"
    model.mkdir(parents=True)
    python = tmp_path / "envs/vllm/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    launches: list[list[str]] = []

    class Process:
        pid = os.getpid()

    monkeypatch.setattr(module, "_vllm_supports_flag", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **_kwargs: launches.append(command) or Process(),
    )
    monkeypatch.setattr(
        module.DeploymentRegistry,
        "_probe_endpoint",
        staticmethod(lambda *_args, **_kwargs: {"status": "unreachable"}),
    )
    registry = module.DeploymentRegistry(tmp_path)
    spec = {
        "id": "local-vllm",
        "kind": "vllm",
        "source_spec": {"type": "model", "id": "qwen-model"},
        "python": str(python),
    }
    kwargs = {
        "instance_id": "instance-local-vllm",
        "source_instance": {"type": "model", "id": "qwen-model-instance"},
        "source_fields": {
            "model_id": "Qwen",
            "model_path": str(model),
            "auth_mode": "none",
        },
        "actual_pricing_instance_id": "test-gpu-pricing",
    }

    first = registry.deploy(spec, **kwargs)
    second = registry.deploy(spec, **kwargs)

    assert first["pid"] == os.getpid()
    assert second["pid"] == first["pid"]
    assert len(launches) == 1


def test_vllm_enabled_prompt_token_details_rejects_unsupported_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lychee_mas.eval.deployments.registry as module

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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    saved = DeploymentRegistry(tmp_path).save(
        {
            "id": "shared-api",
            "kind": "api",
            "model_spec_id": "test-model",
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "declared",
            "kind": "vllm",
            "model_spec_id": "test-model",
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
            model_spec_id="test-model",
            base_url="http://127.0.0.1:8000/v1",
            status="running",
        )
    )

    assert [item["id"] for item in registry.instances()] == ["instance-declared"]


def test_vllm_probe_updates_its_instance_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
        model_spec_id="qwen-model",
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "api",
            "kind": "api",
            "model_spec_id": "test-model",
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
            model_spec_id="test-model",
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry
    from lychee_mas.eval.teams.instances import TeamInstanceRegistry

    deployments = DeploymentRegistry(tmp_path)
    deployments.save(
        {
            "id": "local",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
        }
    )
    deployments._write_instance(
        _instance_record(
            "instance-local",
            "local",
            kind="hf",
            model_id="Qwen",
            model_spec_id="qwen-model",
            model_path=str(tmp_path),
            python=sys.executable,
            status="ready_on_run",
            capabilities={"text_generation": True},
        )
    )
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    teams = TeamInstanceRegistry(tmp_path)
    teams.save(_team_instance_document("team-local", "single", ["Solver"], "instance-local"))
    assert teams.all(deployments.instances())[0]["available"] is True
    deployments.delete_instance("instance-local")
    row = teams.all(deployments.instances())[0]
    assert row["available"] is False
    assert row["missing_deployment_instance_ids"] == ["instance-local"]


def test_environment_installation_profiles_are_composable(tmp_path: Path) -> None:
    from lychee_mas.eval.environment.service import (
        installation_command,
        installation_profiles,
    )

    ids = {item["id"] for item in installation_profiles()}
    assert {"basic", "eval", "runtime", "agentinit", "docs", "full"} <= ids
    command = installation_command(tmp_path, sys.executable, ["eval", "agentinit"])
    assert command[-2:] == ["-e", f"{tmp_path}[benchmark,studio,construct]"]


def test_environment_download_network_maps_proxy_and_mirrors() -> None:
    from lychee_mas.eval.environment.service import build_child_environment

    environment = build_child_environment(
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
        "model_spec_id": "test-qwen",
        "source_spec": {"type": "api", "id": "test-api"},
    }
    response = client.put("/api/deployments/api-shared", json=deployment)
    assert response.status_code == 200
    assert client.get("/api/deployments").json()["specs"][0]["id"] == "api-shared"
    from lychee_mas.eval.models.registry import ModelRegistry

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
        "model_spec_id": "test-qwen",
        "source_spec": {"type": "model", "id": "test-qwen"},
        "python": sys.executable,
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.benchmarks.common import prepared_benchmark_dir

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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
        def launches(*, active_only=False):
            assert active_only is True
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


def test_job_manager_resumes_existing_launch_as_new_execution_segment(tmp_path: Path) -> None:
    launch_id = "resume-demo"
    launch_dir = tmp_path / "runs/eval_studio/launches" / launch_id
    run_dir = tmp_path / "runs/benchmarks" / launch_id
    launch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    project = {
        "name": launch_id,
        "environment": {"repo_root": str(tmp_path), "python": sys.executable},
        "execution": {"mode": "subprocess"},
        "runtime": {
            "case_concurrency": 6,
            "concurrency_policy": {
                "mode": "auto",
                "minimum": 1,
                "initial": 2,
                "maximum": 6,
                "increase_step": 1,
                "decrease_factor": 0.5,
                "control_window_cases": 4,
            },
        },
    }
    (launch_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
    (launch_dir / "launch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "launch_id": launch_id,
                "run_dir": str(run_dir),
                "commands": {},
            }
        ),
        encoding="utf-8",
    )
    for name, marker in (
        ("verify_deployment_instances.sh", "verified"),
        ("resume.sh", "resumed"),
        ("analyze.sh", "analyzed"),
    ):
        (launch_dir / name).write_text(f"printf '%s\\n' {marker}\n", encoding="utf-8")

    class Compiler:
        repo_root = tmp_path
        compiled_projects = []

        @classmethod
        def compile(cls, value, *, launch_id):
            cls.compiled_projects.append(value)
            files = {
                name: (launch_dir / name).read_text(encoding="utf-8")
                for name in (
                    "verify_deployment_instances.sh",
                    "resume.sh",
                    "analyze.sh",
                )
            }
            files["runtime_config.yaml"] = json.dumps(value["runtime"])
            return CompiledPlan(
                launch_id=launch_id,
                launch_dir=launch_dir,
                project=value,
                files=files,
                commands={},
                run_dir=run_dir,
            )

    manager = JobManager(Compiler())
    manager._write_state(
        launch_dir,
        {
            "schema_version": 2,
            "launch_id": launch_id,
            "mode": "subprocess",
            "status": "completed",
            "created_at_utc": "2026-08-17T00:00:00+00:00",
            "updated_at_utc": "2026-08-17T00:00:01+00:00",
            "launch_dir": str(launch_dir),
            "run_dir": str(run_dir),
            "pid": None,
            "tmux_target": None,
        },
    )
    (launch_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (run_dir / "run_status.json").write_text(json.dumps({"status": "stopped"}), encoding="utf-8")
    (run_dir / "run_finalization.json").write_text(
        json.dumps({"status": "finalized", "segment": 0}), encoding="utf-8"
    )

    plan = manager.existing_plan(launch_dir)
    assert isinstance(plan, CompiledPlan)
    started = manager.resume(plan)
    assert started["status"] == "running"
    assert started["resume_count"] == 1
    assert not (run_dir / "run_finalization.json").exists()
    archived_receipt = run_dir / "run_finalization_history/segment_000000.json"
    assert json.loads(archived_receipt.read_text(encoding="utf-8"))["segment"] == 0

    deadline = time.time() + 5
    while time.time() < deadline:
        state = manager.status(launch_dir)
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.02)
    assert state["status"] == "completed"
    assert state["return_code"] == 0
    assert state["resume_history"][0]["resume_index"] == 1
    assert state["resume_history"][0]["return_code"] == 0
    migrated_runtime = Compiler.compiled_projects[-1]["runtime"]
    assert migrated_runtime["trial_concurrency"] == 6
    assert "case_concurrency" not in migrated_runtime
    assert migrated_runtime["concurrency_policy"]["control_window_trials"] == 4
    assert "control_window_cases" not in migrated_runtime["concurrency_policy"]
    refreshed_runtime = json.loads((launch_dir / "runtime_config.yaml").read_text())
    assert refreshed_runtime == migrated_runtime
    log = (launch_dir / "launch.log").read_text(encoding="utf-8")
    assert "resume segment started" in log
    assert "verified\nresumed\nanalyzed" in log


def test_completed_run_is_not_reclassified_as_failed_by_post_run_job_failure() -> None:
    projected = finished_instance_projection(
        {"queue": {"enabled": True}},
        {
            "status": "failed",
            "return_code": 1,
            "failure_reason": "analysis command failed",
        },
        {"status": "complete", "completed_distinct_cases": 10},
        case_completion_target=10,
        finished_at_utc="2026-08-29T00:00:00+00:00",
    )

    assert projected["status"] == "completed"
    assert projected["lifecycle_consistent"] is True
    assert projected["launcher_failure_after_run"] is True
    assert projected["post_run_failure"]["failure_reason"] == "analysis command failed"
    assert projected["queue"]["enabled"] is False


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
        def launches(*, active_only=False):
            assert active_only is True
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


def _rebalance_trial_admission_ticks(
    manager: ExperimentQueueManager,
    count: int = 32,
) -> None:
    """Advance deterministic Scheduler pressure-sampling ticks in unit tests."""

    for _ in range(count):
        manager._last_trial_admission_monotonic = 0.0
        before = {
            key: int(value.get("admission_total_issued") or 0)
            for key, value in {
                **manager._pending_allocations,
                **manager._active_allocations,
            }.items()
        }
        manager._rebalance_active_allocations()
        after = {
            key: int(value.get("admission_total_issued") or 0)
            for key, value in {
                **manager._pending_allocations,
                **manager._active_allocations,
            }.items()
        }
        if after == before:
            break


def test_scheduler_builds_priority_waiting_and_running_trial_sequences(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "priority-sequences"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id in enumerate(("A", "B", "C", "D", "E"), 10):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status="queued")

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 5
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    settings = {
        "A": ("auto", 5, 5),
        "B": ("fixed", 5, 5),
        "C": ("auto", 32, 40),
        "D": ("fixed", 5, 10),
        "E": ("auto", 5, 5),
    }
    manager._pending_allocations = {
        instance_id: {
            "instance_id": instance_id,
            "trial_concurrency_mode": mode,
            "max_trial_slots": maximum,
            "remaining_work": remaining,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 32},
        }
        for instance_id, (mode, maximum, remaining) in settings.items()
    }

    _rebalance_trial_admission_ticks(manager)

    assert {
        instance_id: allocation["current_trial_slots"]
        for instance_id, allocation in manager._pending_allocations.items()
    } == {"A": 5, "B": 5, "C": 32, "D": 5, "E": 5}
    queues = manager._trial_queue_snapshot_locked(registry.instances())
    sequences = manager._trial_sequences_locked(queues)
    assert len(sequences["running_trial_sequence"]) == 52
    assert sequences["waiting_trial_segments"] == [
        {"instance_id": "C", "priority": 12, "count": 8},
        {"instance_id": "D", "priority": 13, "count": 5},
    ]

    # Once B and D are active, both fixed experiments are replenished before
    # pressure is checked. General admission then stops for A, C, and E.
    for instance_id in ("A", "B", "C", "D", "E"):
        manager._pending_allocations[instance_id]["segment_started_trials"] = int(
            manager._pending_allocations[instance_id]["admission_total_issued"]
        )
    for instance_id in ("B", "D"):
        allocation = manager._pending_allocations.pop(instance_id)
        allocation["in_flight_trial_slots"] = 4
        manager._active_allocations[instance_id] = allocation
    manager._deployment_health["shared-vllm"] = {
        "vllm_admission_blocked": True,
        "gpu_admission_blocked": False,
    }
    _rebalance_trial_admission_ticks(manager, count=2)

    assert manager._active_allocations["B"]["current_trial_slots"] == 5
    assert manager._active_allocations["D"]["current_trial_slots"] == 5
    for instance_id in ("A", "C", "E"):
        assert manager._pending_allocations[instance_id]["current_trial_slots"] == 0
        assert (
            manager._pending_allocations[instance_id]["admission_wait_reason"] == "pressure_paused"
        )


def test_experiment_queue_dynamically_admits_trials_by_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lychee_mas.eval.scheduling.manager._TRIAL_ADMISSION_SAMPLE_INTERVAL_S",
        0.0,
    )
    monkeypatch.setattr(
        "lychee_mas.eval.scheduling.manager._MAX_NEW_TRIALS_PER_TICK",
        64,
    )
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "parallel"
    spec["runtime"]["trial_concurrency"] = 32
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
                        "trial_concurrency": 32,
                        "concurrency_policy": {"mode": "auto", "maximum": 32},
                        "trials_per_case": 1,
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
        initial_case_limits: list[tuple[str, int, int]] = []

        @staticmethod
        def validate(_plan, mode):
            return {"mode": mode}

        def launch(self, plan, mode):
            self.states[plan.launch_id] = "running"
            self.launch_order.append(plan.launch_id)
            return {"status": "running", "mode": mode}

        def configure_run_segment(
            self,
            plan,
            *,
            segment_case_limit,
            scheduler_trial_concurrency,
            scheduler_trial_admission_total,
        ):
            self.initial_case_limits.append(
                (
                    plan.launch_id,
                    scheduler_trial_concurrency,
                    scheduler_trial_admission_total,
                )
            )
            return {"segment_case_limit": segment_case_limit}

        def status(self, launch_dir):
            launch_id = Path(launch_dir).name
            status = self.states.get(launch_id, "running")
            return {"status": status, "return_code": 0 if status == "completed" else None}

        @staticmethod
        def launches(*, active_only=False):
            assert active_only is True
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
    manager.start(max_parallel_instances=8, max_running_trials=32)
    restarted = manager.start(max_parallel_instances=16, max_running_trials=32)
    assert restarted["max_parallel_instances"] == 16
    assert restarted["max_running_trials"] == 32
    assert restarted["running_trial_pool"]["hard_limit"] == 32

    deadline = time.time() + 5
    while len(jobs.launch_order) < 4 and time.time() < deadline:
        time.sleep(0.02)
    assert jobs.launch_order == [
        "parallel-0",
        "parallel-1",
        "parallel-2",
        "parallel-3",
    ]
    assert jobs.initial_case_limits == [
        ("parallel-0", 10, 10),
        ("parallel-1", 10, 10),
        ("parallel-2", 10, 10),
        ("parallel-3", 10, 2),
    ]
    assert manager.status()["running_trial_pool"]["occupied"] == 32
    assert manager.stop_after_current()["stop_after_current"] is True
    assert (
        manager.start(
            max_parallel_instances=16,
            max_running_trials=32,
        )["stop_after_current"]
        is False
    )

    run_status = {
        "expected_trials": 10,
        "successful_trials": 8,
        "failed_trials": 0,
        # Resume reports retained successes in both successful_trials and the
        # informational skipped_trials subset; capacity must not double count it.
        "skipped_trials": 8,
    }
    (tmp_path / "runs/benchmarks/parallel-0/run_status.json").write_text(
        json.dumps(run_status), encoding="utf-8"
    )
    manager._wake_event.set()
    deadline = time.time() + 5
    while (
        manager._active_allocations["parallel-0"]["current_trial_slots"] != 2
        and time.time() < deadline
    ):
        time.sleep(0.02)
    status = manager.status()
    assert len(status["active_instance_ids"]) == 4
    assert status["running_trial_pool"]["occupied"] == 32
    assert manager._active_allocations["parallel-0"]["current_trial_slots"] == 2

    jobs.states.update({f"parallel-{index}": "completed" for index in range(4)})
    manager._wake_event.set()
    assert manager._thread is not None
    manager._thread.join(timeout=8)
    assert not manager._thread.is_alive()
    assert all(item["status"] == "completed" for item in registry.instances())


def test_experiment_queue_priority_fill_redistributes_unused_capacity(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "progressive-fill"
    registry.save_spec("progressive-fill", spec)
    for index in range(2):
        registry.create_instance(
            instance_id=f"progressive-fill-{index}",
            spec_id="progressive-fill",
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=index,
        )

    class Jobs:
        limits: dict[str, int] = {}

        def set_trial_admission_total(self, launch_dir, value):
            self.limits[Path(launch_dir).name] = value

    jobs = Jobs()
    manager = ExperimentQueueManager(registry, None, jobs, None)
    manager._max_running_trials = 32
    manager._active_allocations = {
        "progressive-fill-0": {
            "instance_id": "progressive-fill-0",
            "launch_dir": str(tmp_path / "progressive-fill-0"),
            "max_trial_slots": 32,
            "current_trial_slots": 0,
            "remaining_work": 9,
            "deployment_capacities": {"shared-vllm": 32},
        },
        "progressive-fill-1": {
            "instance_id": "progressive-fill-1",
            "launch_dir": str(tmp_path / "progressive-fill-1"),
            "max_trial_slots": 32,
            "current_trial_slots": 0,
            "remaining_work": 50,
            "deployment_capacities": {"shared-vllm": 32},
        },
    }

    _rebalance_trial_admission_ticks(manager)

    assert manager._active_allocations["progressive-fill-0"]["current_trial_slots"] == 9
    assert manager._active_allocations["progressive-fill-1"]["current_trial_slots"] == 23
    assert jobs.limits == {"progressive-fill-0": 9, "progressive-fill-1": 23}


def test_experiment_queue_reclaims_ramping_and_draining_trial_slots(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "observed-demand"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id, status in (
        (10, "draining", "running"),
        (20, "ramping", "running"),
        (30, "next", "queued"),
    ):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status=status)

    draining_dir = tmp_path / "draining-run"
    ramping_dir = tmp_path / "ramping-run"
    draining_dir.mkdir()
    ramping_dir.mkdir()
    (draining_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "draining",
                "expected_trials": 50,
                "successful_trials": 15,
                "failed_trials": 0,
                "completed_distinct_cases": 15,
                "active_cases": [{"case_id": "in-flight"}],
                "current_trial_concurrency": 4,
                "concurrency_policy": {"mode": "auto", "maximum": 16},
            }
        ),
        encoding="utf-8",
    )
    (ramping_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "expected_trials": 50,
                "successful_trials": 20,
                "failed_trials": 0,
                "completed_distinct_cases": 20,
                "active_cases": [{"case_id": f"active-{index}"} for index in range(5)],
                "current_trial_concurrency": 5,
                "adaptive_trial_concurrency_target": 6,
                "concurrency_policy": {"mode": "auto", "maximum": 16},
            }
        ),
        encoding="utf-8",
    )

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 3
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._active_allocations = {
        "draining": {
            "instance_id": "draining",
            "run_dir": str(draining_dir),
            "max_trial_slots": 16,
            "current_trial_slots": 16,
            "deployment_capacities": {"shared-vllm": 32},
        },
        "ramping": {
            "instance_id": "ramping",
            "run_dir": str(ramping_dir),
            "max_trial_slots": 16,
            "current_trial_slots": 16,
            "deployment_capacities": {"shared-vllm": 32},
        },
    }
    manager._pending_allocations = {
        "next": {
            "instance_id": "next",
            "max_trial_slots": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 32},
        }
    }

    manager._refresh_active_demands(rebalance=False)
    _rebalance_trial_admission_ticks(manager)

    assert manager._active_allocations["draining"]["current_trial_slots"] == 1
    assert manager._active_allocations["ramping"]["current_trial_slots"] == 6
    assert manager._pending_allocations["next"]["current_trial_slots"] == 5
    with manager._lock:
        usage = manager._resource_usage_locked()
    assert usage["shared-vllm"]["associated_running_trials"] == 6


def test_experiment_queue_keeps_new_segment_launch_allocation_until_status_refresh(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "resuming-run"
    run_dir.mkdir()
    (run_dir / "run_segment.json").write_text(
        json.dumps(
            {
                "segment_index": 2,
                "scheduler_trial_concurrency": 5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "segment_index": 1,
                "active_cases": [],
            }
        ),
        encoding="utf-8",
    )

    from lychee_mas.eval.scheduling.runner_state import observed_trial_state

    demand = observed_trial_state({"run_dir": str(run_dir)})["requested_trial_slots"]

    assert demand == 5


def test_scheduler_managed_runner_reports_hard_demand_not_local_aimd_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "scheduler-managed"
    run_dir.mkdir()
    (run_dir / "run_segment.json").write_text(
        json.dumps({"concurrency_authority": "scheduler"}),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "concurrency_authority": "scheduler",
                "current_trial_concurrency": 2,
                "adaptive_trial_concurrency_target": None,
                "active_cases": [{"case_id": "active"}],
                "concurrency_policy": {"mode": "auto", "maximum": 16},
            }
        ),
        encoding="utf-8",
    )

    from lychee_mas.eval.scheduling.runner_state import observed_trial_state

    state = observed_trial_state({"run_dir": str(run_dir)})

    assert state["concurrency_authority"] == "scheduler"
    assert state["requested_trial_slots"] is None
    assert state["in_flight_trial_slots"] == 1

    registry = ExperimentRegistry(tmp_path / "registry")
    spec = registry.default_spec()
    spec["id"] = "scheduler-demand"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="scheduler-demand-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
    )
    registry.update_instance("scheduler-demand-run", status="running")
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._active_allocations = {
        "scheduler-demand-run": {
            "instance_id": "scheduler-demand-run",
            "max_trial_slots": 16,
            "remaining_work": 8,
            "observed_trial_slot_demand": 1,
            "concurrency_authority": "scheduler",
            "current_trial_slots": 0,
            "in_flight_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 16},
        }
    }

    _rebalance_trial_admission_ticks(manager, count=2)

    assert manager._active_allocations["scheduler-demand-run"]["current_trial_slots"] == 8


def test_experiment_queue_priority_fill_partially_admits_boundary_run(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "priority-fill"
    registry.save_spec("priority-fill", spec)
    case_counts = {"A": 5, "B": 10, "C": 50, "D": 10}
    for priority, instance_id in enumerate(case_counts):
        registry.create_instance(
            instance_id=instance_id,
            spec_id="priority-fill",
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.enqueue(instance_id)

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 4
    manager._max_running_trials = 20
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._pending_allocations = {
        instance_id: {
            "instance_id": instance_id,
            "max_trial_slots": case_count,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 20},
        }
        for instance_id, case_count in case_counts.items()
    }

    _rebalance_trial_admission_ticks(manager)

    assert {
        instance_id: allocation["current_trial_slots"]
        for instance_id, allocation in manager._pending_allocations.items()
    } == {"A": 5, "B": 10, "C": 5, "D": 0}

    # A and B complete: all released capacity goes to the earlier C, while D
    # remains queued. D is admitted only after C's remaining demand shrinks.
    manager._active_allocations = {"C": manager._pending_allocations.pop("C")}
    manager._pending_allocations = {"D": manager._pending_allocations["D"]}
    manager._active_allocations["C"]["remaining_work"] = 50
    _rebalance_trial_admission_ticks(manager)
    assert manager._active_allocations["C"]["current_trial_slots"] == 20
    assert manager._pending_allocations["D"]["current_trial_slots"] == 0

    manager._active_allocations["C"]["remaining_work"] = 3
    _rebalance_trial_admission_ticks(manager)
    assert manager._active_allocations["C"]["current_trial_slots"] == 3
    assert manager._pending_allocations["D"]["current_trial_slots"] == 10


def test_experiment_queue_refills_active_fixed_before_resource_pressure_check(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "gang-admission"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id, status in (
        (10, "fixed-head", "running"),
        (20, "elastic-running", "running"),
        (30, "unrelated", "queued"),
    ):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status=status)

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 3
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._active_allocations = {
        "fixed-head": {
            "instance_id": "fixed-head",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "in_flight_trial_slots": 2,
            "current_trial_slots": 2,
            "deployment_capacities": {"shared-vllm": 10},
        },
        "elastic-running": {
            "instance_id": "elastic-running",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 10,
            "remaining_work": 20,
            "in_flight_trial_slots": 8,
            "current_trial_slots": 8,
            "deployment_capacities": {"shared-vllm": 10},
        },
    }
    manager._pending_allocations = {
        "unrelated": {
            "instance_id": "unrelated",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 4,
            "remaining_work": 4,
            "current_trial_slots": 0,
            "deployment_capacities": {"other-api": 4},
        },
    }
    manager._deployment_health = {
        "shared-vllm": {
            "vllm_admission_blocked": True,
            "gpu_admission_blocked": False,
        },
        "other-api": {
            "vllm_admission_blocked": True,
            "gpu_admission_blocked": False,
        },
    }

    _rebalance_trial_admission_ticks(manager, count=2)

    fixed = manager._active_allocations["fixed-head"]
    assert fixed["current_trial_slots"] == 5
    assert manager._active_allocations["elastic-running"]["current_trial_slots"] == 8
    assert manager._pending_allocations["unrelated"]["current_trial_slots"] == 0
    assert manager._pending_allocations["unrelated"]["admission_wait_reason"] == "pressure_paused"
    admitted, reason = manager._can_admit_locked(fixed)
    assert admitted is False
    assert "pressure relief" in reason


def test_experiment_queue_does_not_start_pending_fixed_under_pressure(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "pending-fixed-pressure"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="pending-fixed",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
    )
    registry.update_instance("pending-fixed", status="queued")
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._pending_allocations = {
        "pending-fixed": {
            "instance_id": "pending-fixed",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 10},
        }
    }
    manager._deployment_health["shared-vllm"] = {
        "vllm_admission_blocked": True,
        "gpu_admission_blocked": False,
    }

    _rebalance_trial_admission_ticks(manager, count=2)

    allocation = manager._pending_allocations["pending-fixed"]
    assert allocation["current_trial_slots"] == 0
    assert allocation["admission_wait_reason"] == "pressure_paused"
    admitted, reason = manager._can_admit_locked(allocation)
    assert admitted is False
    assert "pressure relief" in reason


def test_experiment_queue_refills_active_fixed_before_higher_priority_dynamic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lychee_mas.eval.scheduling.manager._TRIAL_ADMISSION_SAMPLE_INTERVAL_S",
        0.0,
    )
    monkeypatch.setattr(
        "lychee_mas.eval.scheduling.manager._MAX_NEW_TRIALS_PER_TICK",
        4,
    )
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "fixed-refill-first"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id, status in (
        (10, "higher-dynamic", "queued"),
        (20, "lower-fixed", "running"),
    ):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status=status)
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 2
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._active_allocations = {
        "lower-fixed": {
            "instance_id": "lower-fixed",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "in_flight_trial_slots": 1,
            "current_trial_slots": 1,
            "deployment_capacities": {"shared-vllm": 10},
        }
    }
    manager._pending_allocations = {
        "higher-dynamic": {
            "instance_id": "higher-dynamic",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 10},
        }
    }

    manager._rebalance_active_allocations()

    assert manager._active_allocations["lower-fixed"]["current_trial_slots"] == 5
    assert manager._pending_allocations["higher-dynamic"]["current_trial_slots"] == 0
    assert (
        manager._pending_allocations["higher-dynamic"]["admission_wait_reason"]
        == "next_admission_tick"
    )


def test_experiment_queue_fixed_mode_admits_short_tail_incrementally(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "short-tail"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="short-tail-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
    )
    registry.update_instance("short-tail-run", status="queued")
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 1
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._pending_allocations = {
        "short-tail-run": {
            "instance_id": "short-tail-run",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 3,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 3},
        }
    }

    _rebalance_trial_admission_ticks(manager)

    allocation = manager._pending_allocations["short-tail-run"]
    assert allocation["current_trial_slots"] == 3


def test_experiment_queue_fixed_mode_uses_hard_capacity_not_adaptive_limit(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "fixed-hard-capacity"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="fixed-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
    )
    registry.update_instance("fixed-run", status="queued")
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._max_parallel_instances = 1
    manager._deployment_health["shared-vllm"] = {
        "effective_capacity": 3,
        "hard_capacity": 10,
        "vllm_admission_blocked": False,
        "gpu_admission_blocked": False,
    }
    manager._pending_allocations = {
        "fixed-run": {
            "instance_id": "fixed-run",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 10},
        }
    }

    _rebalance_trial_admission_ticks(manager, count=2)

    allocation = manager._pending_allocations["fixed-run"]
    assert allocation["current_trial_slots"] == 5
    assert allocation.get("admission_wait_reason") is None
    admitted, reason = manager._can_admit_locked(allocation)
    assert admitted is True, reason


def test_experiment_queue_global_trial_pool_caps_unrelated_deployments(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "global-pool"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id in ((10, "first"), (20, "second")):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status="queued")

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 2
    manager._max_running_trials = 8
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._pending_allocations = {
        "first": {
            "instance_id": "first",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"vllm-a": 5},
        },
        "second": {
            "instance_id": "second",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"vllm-b": 5},
        },
    }

    _rebalance_trial_admission_ticks(manager)

    assert manager._pending_allocations["first"]["current_trial_slots"] == 5
    assert manager._pending_allocations["second"]["current_trial_slots"] == 3
    with manager._lock:
        pool = manager._running_trial_pool_locked()
    assert pool == {
        "occupied": 8,
        "running": 0,
        "launching": 8,
        "hard_limit": 8,
        "available_before_hard_limit": 0,
    }


def test_experiment_queue_fixed_mode_uses_available_global_capacity_incrementally(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "global-gang"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id, status in (
        (10, "fixed-head", "queued"),
        (20, "existing", "running"),
        (30, "lower", "queued"),
    ):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status=status)

    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_parallel_instances = 3
    manager._max_running_trials = 10
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._active_allocations = {
        "existing": {
            "instance_id": "existing",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 6,
            "remaining_work": 6,
            "in_flight_trial_slots": 6,
            "current_trial_slots": 6,
            "deployment_capacities": {"vllm-b": 6},
        }
    }
    manager._pending_allocations = {
        "fixed-head": {
            "instance_id": "fixed-head",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "current_trial_slots": 0,
            "deployment_capacities": {"vllm-a": 5},
        },
        "lower": {
            "instance_id": "lower",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 4,
            "remaining_work": 4,
            "current_trial_slots": 0,
            "deployment_capacities": {"api-c": 4},
        },
    }

    _rebalance_trial_admission_ticks(manager)

    assert manager._pending_allocations["fixed-head"]["current_trial_slots"] == 4
    assert manager._pending_allocations["lower"]["current_trial_slots"] == 0
    assert (
        manager._pending_allocations["lower"]["admission_wait_reason"] == "running_trial_hard_limit"
    )

    manager._active_allocations["existing"]["in_flight_trial_slots"] = 5
    _rebalance_trial_admission_ticks(manager)
    assert manager._pending_allocations["fixed-head"]["current_trial_slots"] == 5
    assert manager._active_allocations["existing"]["current_trial_slots"] == 5
    assert manager._pending_allocations["lower"]["current_trial_slots"] == 0


def test_experiment_queue_rejects_fixed_concurrency_above_running_trial_limit(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._max_running_trials = 4
    instance = {
        "id": "oversized-global-fixed-run",
        "execution": {
            "next_segment_case_limit": None,
            "case_completion_target": None,
        },
    }
    plan = SimpleNamespace(
        project={
            "runtime": {
                "trial_concurrency": 5,
                "trials_per_case": 1,
                "concurrency_policy": {"mode": "fixed", "maximum": 5},
            },
            "benchmark": {"n": 20},
            "deployment_instances": [],
        },
        launch_dir=tmp_path / "launch",
        run_dir=tmp_path / "run",
    )

    with pytest.raises(ValueError, match="Scheduler running Trial hard limit 4"):
        manager._allocation_for_plan(instance, plan, launch_origin="queue")


def test_experiment_queue_allows_more_trials_than_model_request_concurrency(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    manager = ExperimentQueueManager(registry, None, None, None)
    instance = {
        "id": "oversized-fixed-run",
        "execution": {
            "next_segment_case_limit": None,
            "case_completion_target": None,
        },
    }
    plan = SimpleNamespace(
        project={
            "runtime": {
                "trial_concurrency": 8,
                "trials_per_case": 1,
                "concurrency_policy": {"mode": "fixed", "maximum": 8},
            },
            "benchmark": {"n": 20},
            "deployment_instances": [
                {
                    "id": "small-vllm",
                    "kind": "vllm",
                    "request_limits": {"max_concurrency": 4},
                }
            ],
        },
        launch_dir=tmp_path / "launch",
        run_dir=tmp_path / "run",
    )

    allocation = manager._allocation_for_plan(instance, plan, launch_origin="queue")

    assert allocation["max_trial_slots"] == 8
    assert allocation["deployment_capacities"] == {"small-vllm": 4}


def test_dynamic_deployment_starts_from_configured_admission_initial(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "dynamic-initial"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="dynamic-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=10,
    )
    registry.update_instance("dynamic-run", status="queued")
    manager = ExperimentQueueManager(registry, None, None, None)
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._max_parallel_instances = 1
    manager._pending_allocations = {
        "dynamic-run": {
            "instance_id": "dynamic-run",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 64,
            "remaining_work": 64,
            "current_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 64},
            "deployment_admission_policies": {
                "shared-vllm": {
                    "mode": "auto",
                    "minimum": 8,
                    "initial": 32,
                    "maximum": 64,
                }
            },
        }
    }

    manager._rebalance_active_allocations()

    assert manager._pending_allocations["dynamic-run"]["current_trial_slots"] == 4
    assert manager._pending_allocations["dynamic-run"]["admission_total_issued"] == 4
    usage = manager._resource_usage_locked()
    assert usage == {}


def test_experiment_queue_excludes_invalid_instances_from_runnable_queue(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "mutable-spec"
    registry.save_spec(spec["id"], spec)
    registry.create_instance(
        instance_id="stale-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
    )
    registry.enqueue("stale-run")
    spec["runtime"]["max_turns"] = 99
    registry.save_spec(spec["id"], spec)

    class Jobs:
        @staticmethod
        def launches(*, active_only=False):
            assert active_only is True
            return []

    manager = ExperimentQueueManager(registry, None, Jobs(), None)

    status = manager.status()

    assert status["queued_instance_ids"] == []
    assert status["invalid_queued_instance_ids"] == ["stale-run"]


def test_experiment_queue_refills_from_highest_priority_after_each_trial(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "running-pool"
    registry.save_spec(spec["id"], spec)
    for priority, instance_id in enumerate(("A", "B", "C", "D")):
        registry.create_instance(
            instance_id=instance_id,
            spec_id=spec["id"],
            benchmark_instance_id="benchmark-data",
            team_instance_id="team-runtime",
            priority=priority,
        )
        registry.update_instance(instance_id, status="running")

    class Jobs:
        limits: dict[str, int] = {}

        def set_trial_admission_total(self, launch_dir, value):
            self.limits[Path(launch_dir).name] = value

    jobs = Jobs()
    manager = ExperimentQueueManager(registry, None, jobs, None)
    manager._max_running_trials = 8
    manager._active_allocations = {
        "A": {
            "instance_id": "A",
            "launch_dir": str(tmp_path / "A"),
            "max_trial_slots": 5,
            "remaining_work": 5,
            "observed_trial_slot_demand": 5,
            "in_flight_trial_slots": 4,
            "current_trial_slots": 5,
            "deployment_capacities": {"shared-vllm": 8},
        },
        "B": {
            "instance_id": "B",
            "launch_dir": str(tmp_path / "B"),
            "max_trial_slots": 5,
            "remaining_work": 10,
            "observed_trial_slot_demand": 5,
            "in_flight_trial_slots": 3,
            "current_trial_slots": 3,
            "deployment_capacities": {"shared-vllm": 8},
        },
        "C": {
            "instance_id": "C",
            "launch_dir": str(tmp_path / "C"),
            "max_trial_slots": 8,
            "remaining_work": 40,
            "observed_trial_slot_demand": 8,
            "in_flight_trial_slots": 0,
            "current_trial_slots": 2,
            "deployment_capacities": {"shared-vllm": 8},
        },
        "D": {
            "instance_id": "D",
            "launch_dir": str(tmp_path / "D"),
            "max_trial_slots": 5,
            "remaining_work": 5,
            "observed_trial_slot_demand": 5,
            "in_flight_trial_slots": 0,
            "current_trial_slots": 1,
            "deployment_capacities": {"shared-vllm": 8},
        },
    }

    manager._rebalance_active_allocations()

    # Seven non-preemptible Trials already occupy the pool. The only free slot
    # returns to A, while B keeps its three in-flight Trials and C/D receive a
    # zero admission cap without terminating their runner processes.
    assert {
        key: value["current_trial_slots"] for key, value in manager._active_allocations.items()
    } == {"A": 5, "B": 3, "C": 0, "D": 0}
    assert jobs.limits == {"A": 1}

    # One B Trial finishes. Refill restarts from A; A is already full, so the
    # newly free slot goes back to B before C or D can enter the running pool.
    manager._active_allocations["B"]["in_flight_trial_slots"] = 2
    manager._last_trial_admission_monotonic = 0.0
    manager._rebalance_active_allocations()
    assert {
        key: value["current_trial_slots"] for key, value in manager._active_allocations.items()
    } == {"A": 5, "B": 3, "C": 0, "D": 0}
    assert jobs.limits == {"A": 1, "B": 1}


def test_experiment_queue_rejects_start_without_queued_instances(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    manager = ExperimentQueueManager(registry, None, None, None)

    with pytest.raises(ValueError, match="没有 queued ExperimentInstance"):
        manager.start()


def test_experiment_queue_keeps_scheduling_recovered_running_instances(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "recover-running"
    registry.save_spec(spec["id"], spec)
    launch_dir = tmp_path / "launches/recover-running-run"
    launch_dir.mkdir(parents=True)
    created = registry.create_instance(
        instance_id="recover-running-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
    )
    registry.update_instance(
        created["id"],
        status="running",
        launch_dir=str(launch_dir),
        launch_origin="queue_resume",
        run_status="paused",
        evaluation_status="completed",
        finalized_at_utc="2026-08-28T00:00:00+00:00",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )

    class Jobs:
        state = "running"
        archived = False

        def status(self, _launch_dir):
            return {
                "status": self.state,
                "run_dir": str(run_dir),
                "return_code": 0 if self.state == "completed" else None,
            }

        @classmethod
        def archive_finalization_receipt(cls, _run_dir, *, resume_count):
            assert resume_count == 1
            cls.archived = True

        @staticmethod
        def launches(*, active_only=False):
            assert active_only is True
            return []

    jobs = Jobs()
    manager = ExperimentQueueManager(registry, None, jobs, None)
    started = manager.start(max_parallel_instances=4)

    assert started["recoverable_running_instance_ids"] == ["recover-running-run"]
    deadline = time.time() + 3
    while registry.get_instance("recover-running-run").get("run_status") != "running":
        assert time.time() < deadline
        time.sleep(0.02)
    assert "recover-running-run" in manager.status()["active_instance_ids"]
    assert manager._thread is not None and manager._thread.is_alive()
    recovered = registry.get_instance("recover-running-run")
    assert recovered["run_status"] == "running"
    assert recovered["evaluation_status"] == "pending"
    assert recovered["finalized_at_utc"] is None
    assert jobs.archived is True

    jobs.state = "completed"
    manager._wake_event.set()
    manager._thread.join(timeout=5)
    assert not manager._thread.is_alive()


def test_experiment_spec_rejects_task_outside_registered_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    benchmarks = BenchmarkRegistry(tmp_path)
    benchmarks.save_spec(_benchmark_spec("gsm8k"))
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec("single", "solver"))
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


def test_team_binding_report_uses_authoritative_runtime_contract(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    team = _single_team_spec("portable-team", "solver")
    team["nodes"].append(_team_node("solver_2"))
    team["relations"].append(
        {
            "id": "solver-to-solver-2",
            "from": "solver",
            "to": "solver_2",
            "control": {
                "trigger": "completed",
                "action": "activate",
                "priority": 0,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "source.output", "target": "messages", "view": "latest"},
                    {
                        "source": "shared.TrialWorkspace",
                        "target": "state",
                        "filter": {"content_type": "artifact"},
                    },
                ]
            },
        }
    )
    for state in team["shared_state"]:
        state["readers"].append("solver_2")
        state["writers"].append("solver_2")
    team["lifecycle"]["result"]["submissions"] = [
        {"from": "solver_2", "source": "source.output", "key": "final_answer"}
    ]
    response = TestClient(create_app(tmp_path)).post(
        "/api/team-binding-report",
        json={"runtime_framework": "autogen", "team_spec": team},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["implementation"] == "RoundRobinGroupChat"
    assert report["mapping_level"] == "approximated"
    assert report["controlled_comparison_eligible"] is False
    assert any("DataTransfer filters" in item for item in report["semantic_deltas"])


def test_team_binding_report_accepts_team_spec_read_model(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec("portable-team", "solver"))
    client = TestClient(create_app(tmp_path))
    read_model = next(
        item
        for item in client.get("/api/team-specs").json()["specs"]
        if item["id"] == "portable-team"
    )

    assert "registry_path" in read_model
    response = client.post(
        "/api/team-binding-report",
        json={"runtime_framework": "autogen", "team_spec": read_model},
    )

    assert response.status_code == 200
    assert response.json()["implementation"] == "RoundRobinGroupChat"


def test_experiment_instance_requires_available_benchmark_instance(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

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
    TeamSpecRegistry(tmp_path).save(_single_team_spec("single", "solver"))
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
    from lychee_mas.eval.apis.registry import APIRegistry

    _register_test_model_spec(tmp_path)
    spec = tmp_path / "configs/eval_studio/apis/specs/test-api.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        json.dumps(
            {
                "id": "test-api",
                "name": "Test API",
                "provider": "Test Provider",
                "provider_key": "test",
                "deployment_kind": "api",
                "allowed_model_spec_ids": ["test-qwen"],
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    tastrial_index = commands[0].index("--tasks")
    assert commands[0][tastrial_index + 1] == "gaia"
    assert not any("level_" in part for part in commands[0])
    instance = response.json()["instance"]
    assert instance["benchmark_spec_id"] == "gaia"
    assert instance["acquisition"]["mode"] == "download"
    assert instance["status"] == "preparing"


def test_benchmark_instance_rejects_an_unregistered_provider(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    deployment = DeploymentRegistry(tmp_path).save(
        {
            "id": "local-hf",
            "kind": "hf",
            "source_spec": {"type": "model", "id": "qwen-model"},
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
    assert "resource_bindings" not in team_document
    assert "benchmark_instance_id" not in experiment_document
    assert "team_instance_id" not in experiment_document


def test_instance_documents_bind_only_their_corresponding_instances(tmp_path: Path) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry
    from lychee_mas.eval.teams.instances import TeamInstanceRegistry
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    TeamSpecRegistry(tmp_path).save(_single_team_spec())
    team_saved = TeamInstanceRegistry(tmp_path).save(
        _team_instance_document("single-local", "single", ["Solver"], "instance-local")
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
            model_spec_id="qwen-model",
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

    assert team_document["schema_version"] == 5
    assert team_document["id"] == "single-local"
    assert team_document["team_spec_id"] == "single"
    assert team_document["resource_bindings"] == [
        {
            "node_id": "Solver",
            "requirement": "model_inference",
            "resource_instance_type": "DeploymentInstance",
            "resource_instance_id": "instance-local",
        }
    ]
    assert team_document["framework_binding_report"]["mapping_level"] == "exact"
    assert deployment_document["source_instance"] == {
        "type": "model",
        "id": "qwen-model-local",
    }
    assert experiment_document["benchmark_instance_id"] == "gsm8k-data"
    assert experiment_document["team_instance_id"] == "single-local"
    assert "benchmark" not in experiment_document
    assert "team_spec" not in experiment_document


def test_instance_registries_pin_the_spec_version_used_at_creation(tmp_path: Path) -> None:
    from lychee_mas.eval.apis.registry import APIRegistry
    from lychee_mas.eval.models.registry import ModelRegistry
    from lychee_mas.eval.teams.instances import TeamInstanceRegistry
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

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
                "provider_key": "test",
                "allowed_model_spec_ids": ["test-qwen"],
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
        _team_instance_document("team-instance", "single", ["Solver"], "deployment-instance")
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
    changed_team["nodes"][0]["instructions"] = "changed"
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


def test_magentic_one_operation_node_requires_model_but_terminal_does_not(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.teams.registry import TeamSpecRegistry

    canonical = Path(__file__).resolve().parents[1] / ("configs/eval_studio/teams/specs/gaia.json")
    saved = TeamSpecRegistry(tmp_path).save(json.loads(canonical.read_text(encoding="utf-8")))
    required = {
        node["id"]
        for node in saved["nodes"]
        if node["kind"] == "model_agent"
    }
    assert required == {"Orchestrator", "Coder", "FileSurfer", "WebSurfer"}
    assert "ComputerTerminal" not in required


def test_builtin_swe_team_routes_coder_messages_to_terminal():
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "configs/eval_studio/teams/specs/swe_bench_verified.json"
    )
    graph, spec = load_team_spec(spec_path, rounds=1)
    terminal = next(item for item in graph.nodes if item.name == "ComputerTerminal")

    assert terminal.meta["sources"] == ["Coder"]
    assert spec["lifecycle"]["termination"] == {"condition": "result_submitted"}
    assert spec["lifecycle"]["result"]["submissions"] == [
        {"from": "Coder", "source": "source.output", "key": "final_answer"}
    ]


def test_deployment_instance_is_invalid_when_its_spec_is_missing(tmp_path: Path) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry._write_instance(
        _instance_record(
            "orphan",
            "missing-spec",
            kind="api",
            model_id="missing",
            source_type="api",
            source_id="missing-api",
            model_spec_id="missing-model",
            deployment_spec_fingerprint="0" * 64,
            status="running",
        )
    )

    assert registry.instance("orphan")["status"] == "invalid"
    assert registry.instance("orphan")["available"] is False


def test_deployment_instance_is_invalid_when_its_pricing_instance_is_missing(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    registry = DeploymentRegistry(tmp_path)
    registry.save(
        {
            "id": "api",
            "kind": "api",
            "model_spec_id": "test-model",
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
            model_spec_id="test-model",
            actual_pricing_instance_id="missing-pricing",
            status="running",
        )
    )

    checked = registry.instance("orphan-pricing")
    assert checked["status"] == "invalid"
    assert checked["available"] is False
    assert "invalid pricing binding" in checked["health_detail"]


def test_deployment_instance_becomes_invalid_after_its_spec_changes(tmp_path: Path) -> None:
    from lychee_mas.eval.deployments.registry import (
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
            model_spec_id="source",
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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    spec = {
        "id": f"invalid-{kind}-{str(managed).lower()}",
        "kind": kind,
        "source_spec": {
            "type": "model" if kind == "hf" or managed is True else "api",
            "id": "source",
        },
        **({"model_spec_id": "test-model"} if kind == "api" or managed is False else {}),
        "actual_pricing_spec_id": pricing_spec_id,
        **({"managed": managed} if managed is not None else {}),
    }
    with pytest.raises(ValueError, match=message):
        DeploymentRegistry(tmp_path).save(spec)


def test_external_provider_deployment_rejects_api_equivalent_pricing(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

    with pytest.raises(ValueError, match="only valid for local HF or managed vLLM"):
        DeploymentRegistry(tmp_path).save(
            {
                "id": "external-vllm",
                "kind": "vllm",
                "managed": False,
                "model_spec_id": "test-model",
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

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
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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
            "model_spec_id": "test-model",
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
                "model_info": {},
                "capabilities": {"text_generation": True},
                "thinking_protocol": None,
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
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient
    from lychee_mas.eval.deployments.registry import DeploymentRegistry

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


def test_swarm_team_spec_preserves_validated_handoffs(tmp_path: Path) -> None:
    path = tmp_path / "swarm.json"
    value = _single_team_spec("swarm-demo", "Researcher")
    researcher = _team_node("Researcher")
    verifier = _team_node("Verifier")
    for node in (researcher, verifier):
        node["capabilities"] = ["handoff"]
        node["operations"] = [
            {
                "type": "handoff",
                "options": {"mode": "model", "max_attempts": 3, "fallback": "retry"},
            }
        ]
    members = ("Researcher", "Verifier")
    relations = [
        {
            "id": "researcher-to-verifier",
            "from": "Researcher",
            "to": "Verifier",
            "control": {
                "trigger": "completed",
                "action": "handoff",
                "priority": 0,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "source.output", "target": "messages", "view": "latest"}
                ]
            },
        },
        {
            "id": "verifier-to-researcher",
            "from": "Verifier",
            "to": "Researcher",
            "control": {
                "trigger": "completed",
                "action": "handoff",
                "priority": 0,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "source.output", "target": "messages", "view": "latest"}
                ]
            },
        },
    ]
    for state in value["shared_state"]:
        state["readers"] = list(members)
        state["writers"] = list(members)
    value["lifecycle"]["entry"] = [
        {"node": "Researcher", "inputs": [{"source": "trial.task", "target": "task"}]}
    ]
    value["lifecycle"]["result"]["submissions"] = [
        {"from": "Verifier", "source": "source.output", "key": "final_answer"}
    ]
    path.write_text(
        json.dumps({**value, "nodes": [researcher, verifier], "relations": relations}),
        encoding="utf-8",
    )

    graph, document = load_team_spec(path, rounds=1)

    assert "coordination" not in document
    coordination = graph.meta["coordination_ir"]
    assert coordination["members"] == ["Researcher", "Verifier"]
    assert [item["type"] for item in coordination["handoff_relations"]] == [
        "handoff",
        "handoff",
    ]
    assert "pattern" not in coordination
    assert graph.meta["adapter_plans"]["autogen"]["config"]["type"] == "swarm"
    assert graph.order()[0].meta["handoffs"] == ["Verifier"]
    assert graph.order()[1].meta["handoffs"] == ["Researcher"]


def test_graph_flow_team_spec_compiles_explicit_execution_graph(tmp_path: Path) -> None:
    path = tmp_path / "graph-flow.json"
    value = _single_team_spec("parallel-peer-demo", "Analyst")
    members = ("Analyst", "Solver", "Verifier")
    relations = [
        {
            "id": f"{source.lower()}-to-verifier",
            "from": source,
            "to": "Verifier",
            "control": {
                "trigger": "completed",
                "action": "activate",
                "priority": 0,
                "on_failure": "fail_trial",
            },
            "data": {
                "transfers": [
                    {"source": "source.output", "target": "messages", "view": "latest"}
                ]
            },
        }
        for source in ("Analyst", "Solver")
    ]
    for state in value["shared_state"]:
        state["readers"] = list(members)
        state["writers"] = list(members)
    value["lifecycle"]["entry"] = [
        {"node": source, "inputs": [{"source": "trial.task", "target": "task"}]}
        for source in ("Analyst", "Solver")
    ]
    value["lifecycle"]["result"]["submissions"] = [
        {"from": "Verifier", "source": "source.output", "key": "final_answer"}
    ]
    path.write_text(
        json.dumps(
            {
                **value,
                "nodes": [_team_node(member) for member in members],
                "relations": relations,
            }
        ),
        encoding="utf-8",
    )

    graph, document = load_team_spec(path, rounds=99)

    assert "coordination" not in document
    coordination = graph.meta["coordination_ir"]
    assert coordination["members"] == ["Analyst", "Solver", "Verifier"]
    assert coordination["roots"] == ["Analyst", "Solver"]
    assert coordination["sinks"] == ["Verifier"]
    assert "pattern" not in coordination
    assert graph.meta["adapter_plans"]["langgraph"]["strategy"] == "explicit_dependency_graph"
    assert graph.edges == {
        "Analyst": ["Verifier"],
        "Solver": ["Verifier"],
        "Verifier": [],
    }
    assert graph.meta["speaking_order"] == ["Analyst", "Solver", "Verifier"]


def test_benchmark_tool_bundle_is_shared_by_agents_in_one_trial() -> None:
    from lychee_mas.eval.benchmarks.base import ToolBundle
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import _resolve_function_tools

    resource = SimpleNamespace(close=lambda: None)

    class Benchmark:
        calls = 0

        def create_tools(self, slot, _meta):
            self.calls += 1

            def mutate_shared_state():
                return slot

            return ToolBundle(tools=[mutate_shared_state], resources=[resource])

    benchmark = Benchmark()
    cache = {}
    resources = []
    first = _resolve_function_tools(
        ["benchmark:tools"],
        None,
        benchmark=benchmark,
        resources=resources,
        tool_bundles=cache,
    )
    second = _resolve_function_tools(
        ["benchmark:tools"],
        None,
        benchmark=benchmark,
        resources=resources,
        tool_bundles=cache,
    )

    assert benchmark.calls == 1
    assert first[0] is second[0]
    assert resources == [resource]


def test_experiment_segment_quota_can_pause_update_and_requeue(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "segmented"
    registry.save_spec("segmented", spec)
    created = registry.create_instance(
        instance_id="segmented-run",
        spec_id="segmented",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        priority=20,
        execution={"next_segment_case_limit": 50, "case_completion_target": 100},
    )
    assert created["execution"] == {
        "next_segment_case_limit": 50,
        "case_completion_target": 100,
    }

    paused = registry.update_instance(
        created["id"],
        status="paused",
        launch_dir=str(tmp_path / "runs/eval_studio/launches/segmented-run"),
    )
    assert paused["status"] == "paused"
    edited = registry.update_controls(
        created["id"],
        priority=5,
        next_segment_case_limit=10,
        case_completion_target=75,
    )
    assert edited["queue"]["priority"] == 5
    assert edited["execution"] == {
        "next_segment_case_limit": 10,
        "case_completion_target": 75,
    }

    queued = registry.enqueue(created["id"])
    assert queued["status"] == "queued"
    assert queued["queue"]["resume_from_status"] == "paused"
    restored = registry.dequeue(created["id"])
    assert restored["status"] == "paused"
    assert "resume_from_status" not in restored["queue"]


def test_draining_instance_can_queue_one_automatic_next_segment(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "draining"
    registry.save_spec("draining", spec)
    created = registry.create_instance(
        instance_id="draining-run",
        spec_id="draining",
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        execution={"next_segment_case_limit": 13},
    )
    registry.update_instance(
        created["id"],
        status="running",
        drain_requested=True,
    )

    queued = registry.enqueue(created["id"])

    assert queued["status"] == "running"
    assert queued["queue"]["enabled"] is True
    assert queued["queue"]["resume_from_status"] == "paused"


def test_scheduler_derives_next_segment_from_additional_count_and_cumulative_target(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "automatic-next-segment"
    registry.save_spec(spec["id"], spec)
    run_dir = tmp_path / "runs/benchmarks/automatic-next-segment"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "expected_trials": 100,
                "successful_trials": 40,
                "failed_trials": 0,
                "completed_distinct_cases": 40,
            }
        ),
        encoding="utf-8",
    )
    created = registry.create_instance(
        instance_id="automatic-next-segment-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir=str(run_dir),
        execution={
            "next_segment_case_limit": 13,
            "case_completion_target": 50,
        },
    )

    class Jobs:
        configured = None

        def configure_run_segment(
            self,
            _plan,
            *,
            segment_case_limit,
            scheduler_trial_concurrency,
            scheduler_trial_admission_total,
        ):
            self.configured = (
                segment_case_limit,
                scheduler_trial_concurrency,
                scheduler_trial_admission_total,
            )
            return {"segment_case_limit": segment_case_limit}

    jobs = Jobs()
    manager = ExperimentQueueManager(registry, None, jobs, None)
    plan = SimpleNamespace(run_dir=run_dir)
    manager._configure_segment(
        created,
        plan,
        {"max_trial_slots": 8, "admission_total_issued": 3},
    )

    # The next segment asks for 13 more Cases, but the cumulative target has
    # only 10 Cases remaining.
    assert jobs.configured == (10, 8, 3)


def test_scheduler_settles_stale_queued_instance_after_case_target(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "target-reached"
    registry.save_spec(spec["id"], spec)
    run_dir = tmp_path / "runs/benchmarks/target-reached"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "paused",
                "expected_trials": 100,
                "successful_trials": 3,
                "failed_trials": 0,
                "completed_distinct_cases": 3,
            }
        ),
        encoding="utf-8",
    )
    created = registry.create_instance(
        instance_id="target-reached-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir=str(run_dir),
        execution={"next_segment_case_limit": 3, "case_completion_target": 3},
    )
    queued = registry.update_instance(
        created["id"],
        status="queued",
        queue={"enabled": True, "priority": 10, "resume_from_status": "paused"},
    )
    manager = ExperimentQueueManager(registry, None, None, None)

    assert manager._settle_reached_case_target(queued, SimpleNamespace(run_dir=run_dir)) is True
    settled = registry.get_instance(created["id"])
    assert settled["status"] == "paused"
    assert settled["queue"]["enabled"] is False
    assert "resume_from_status" not in settled["queue"]


def test_drained_instance_automatically_becomes_queued_when_resume_is_pending(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry(tmp_path)
    spec = registry.default_spec()
    spec["id"] = "queued-after-drain"
    registry.save_spec(spec["id"], spec)
    run_dir = tmp_path / "runs/benchmarks/queued-after-drain"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "expected_trials": 100,
                "successful_trials": 20,
                "failed_trials": 0,
                "completed_distinct_cases": 20,
            }
        ),
        encoding="utf-8",
    )
    created = registry.create_instance(
        instance_id="queued-after-drain-run",
        spec_id=spec["id"],
        benchmark_instance_id="benchmark-data",
        team_instance_id="team-runtime",
        run_dir=str(run_dir),
        execution={"next_segment_case_limit": 13},
    )
    registry.update_instance(
        created["id"],
        status="running",
        drain_requested=True,
    )
    registry.enqueue(created["id"])
    manager = ExperimentQueueManager(registry, None, None, None)

    finished = manager._finish(
        created["id"],
        {"status": "completed", "return_code": 0, "run_dir": str(run_dir)},
    )

    assert finished["status"] == "queued"
    assert finished["queue"]["enabled"] is True
    assert finished["queue"]["resume_from_status"] == "stopped"


def test_running_allocation_recovers_resources_from_launch_snapshot(tmp_path: Path) -> None:
    launch_dir = tmp_path / "launch"
    run_dir = tmp_path / "run"
    launch_dir.mkdir()
    run_dir.mkdir()
    (launch_dir / "project.json").write_text(
        json.dumps(
            {
                "environment": {"run_dir": str(run_dir)},
                "benchmark": {"n": "all"},
                "runtime": {
                    # A live launch can predate the Trial terminology migration.
                    "case_concurrency": 32,
                    "concurrency_policy": {"maximum": 32},
                },
                "deployment_instances": [
                    {
                        "id": "shared-vllm",
                        "kind": "vllm",
                        "request_limits": {"max_concurrency": 64},
                        "admission_control": {"maximum": 64},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = ExperimentQueueManager(ExperimentRegistry(tmp_path), None, None, None)

    allocation = manager._allocation_from_launch_snapshot(
        {
            "id": "running-experiment",
            "run_dir": str(run_dir),
            "execution": {},
        },
        launch_dir,
        compile_error=ValueError("spec fingerprint changed"),
    )

    assert allocation["max_trial_slots"] == 32
    assert allocation["deployment_capacities"] == {"shared-vllm": 64}
    assert allocation["recovered_from_launch_snapshot"] is True
    assert "spec fingerprint changed" in allocation["compile_recovery_reason"]
