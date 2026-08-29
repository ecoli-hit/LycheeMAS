"""Compile one Eval Studio project into auditable shell scripts."""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...runtime.adapters.frameworks.bindings import build_framework_binding_report
from ...runtime.coordination.compiler import compile_coordination
from ...runtime.model.token_budget import (
    BUDGET_POLICY_FIELDS,
    normalize_token_budget_policy,
)
from ..application.read_models.catalog import repository_root
from ..benchmarks.assets import BenchmarkRegistry
from ..contracts.specs import normalize_invocation_overrides, validate_project_document
from ..teams.contracts import model_resource_requirements, normalize_team_spec_document

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _q(value: Any) -> str:
    return shlex.quote(str(value))


def _slug(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-.")
    return text[:64] or fallback


def default_project(repo_root: str | os.PathLike | None = None) -> dict[str, Any]:
    from ..environment.service import readme_environment

    root = Path(repo_root or repository_root()).resolve()
    readme_env = readme_environment(root)
    runtime_environment = {
        "python": readme_env["python"],
        "conda_sh": readme_env["conda_sh"],
        "conda_env": readme_env["conda_env"],
    }
    return {
        "schema_version": 4,
        "name": "eval-experiment",
        "environment": {
            "repo_root": str(root),
            **runtime_environment,
            "raw_root": str(root / "data/benchmarks/raw"),
            "prepared_root": str(root / "data/benchmarks/prepared"),
            "models_root": str(root / "models"),
            "runs_root": str(root / "runs/benchmarks"),
            "run_dir": "",
        },
        "benchmark": {
            "benchmark_spec_id": "gsm8k",
            "benchmark_instance_id": None,
            "prepare_target": "gsm8k",
            "task": "gsm8k",
            "source": "auto",
            "n": 10,
            "start_index": 0,
            "case_selection": "head",
            "case_strata_field": None,
            "prepare_before_run": False,
        },
        "team": normalize_team_spec_document(
            json.loads(
                (root / "configs/eval_studio/teams/specs/reason.json").read_text(encoding="utf-8")
            )
        ),
        "deployment_instances": [
            {
                "id": "instance-local-hf",
                "deployment_spec_id": "local-hf",
                "source_instance": {"type": "model", "id": "local-model"},
                "kind": "hf",
                "model_id": "Qwen3-4B-Instruct-2507",
                "model_path": str(root / "models/Qwen3-4B-Instruct-2507"),
                "device": "cuda:0",
                "cuda_visible_devices": "0",
                "status": "ready_on_run",
                "available": True,
                "actual_pricing_instance_id": "reference-a800-on-demand-cny",
                "api_equivalent_pricing_instance_id": "dashscope-qwen3-4b-instruct-2507-unpriced",
            }
        ],
        "deployment_bindings": {
            "default_deployment_instance_id": "instance-local-hf",
            "control_deployment_instance_id": "instance-local-hf",
            "role_bindings": [
                {
                    "role_id": role_id,
                    "deployment_instance_id": "instance-local-hf",
                }
                for role_id in ("planner", "solver", "verifier")
            ],
        },
        "team_instance": {
            "id": "reason-instance",
            "team_spec_id": "reason",
        },
        "runtime": {
            "method": "none",
            "trials_per_case": 1,
            "max_rounds": 4,
            "max_turns": 20,
            "max_model_calls_per_case": None,
            "max_case_wall_time_s": None,
            "max_attempts_per_trial": 1,
            "on_trial_error": "continue",
            "max_new_tokens": 16384,
            "max_input_tokens": None,
            "min_output_reserve_tokens": 2048,
            "min_thinking_reserve_tokens": 0,
            "max_thinking_budget_tokens": None,
            "min_final_reserve_tokens": 1024,
            "safety_margin_tokens": 256,
            "trial_concurrency": 1,
            "concurrency_policy": {
                "mode": "fixed",
                "initial": 1,
                "minimum": 1,
                "maximum": 1,
                "increase_step": 1,
                "decrease_factor": 0.5,
                "control_window_trials": 8,
            },
            "seed": 0,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": 1.0,
            "code_executor": "docker",
            "code_timeout": 60,
            "work_root": "runs/lychee_tool_workspaces",
            "web_headless": True,
            "save_screenshots": False,
            "docker_image": None,
            "trace_model_calls": True,
        },
        "observability": {
            "collect_vllm_metrics": True,
            "vllm_metrics_interval_s": 5.0,
            "event_log_max_events_per_file": 10000,
            "event_log_max_mib_per_file": 64.0,
        },
        "evaluation": {
            "external_evaluator": "auto",
            "profile_id": "core",
            "hle_judge": {},
        },
        "network": {
            "mode": "benchmark_default",
            "proxy_url": "",
            "no_proxy": "127.0.0.1,localhost,::1",
            "targets": {
                "downloads": False,
                "web_surfer": False,
                "code_executor": False,
                "model_backend": False,
            },
            "docker_bridge_host": "172.17.0.1",
            "container_proxy_port": 17897,
            "probe_url": "https://www.google.com/generate_204",
        },
        "execution": {
            "mode": "command_only",
            "tmux_session": "lychee-eval",
            "tmux_window": "benchmark",
            "segment_case_limit": None,
        },
    }


@dataclass
class CompiledPlan:
    launch_id: str
    launch_dir: Path
    project: dict[str, Any]
    files: dict[str, str]
    commands: dict[str, str]
    run_dir: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "launch_id": self.launch_id,
            "launch_dir": str(self.launch_dir),
            "run_dir": str(self.run_dir),
            "project": self.project,
            "files": self.files,
            "commands": self.commands,
        }


class ExecutionPlanCompiler:
    """Generate scripts and machine-readable launch metadata from one project."""

    def __init__(self, repo_root: str | os.PathLike | None = None) -> None:
        self.repo_root = Path(repo_root or repository_root()).resolve()
        self.benchmarks = BenchmarkRegistry(self.repo_root)

    def normalize(self, value: dict[str, Any]) -> dict[str, Any]:
        base = default_project(self.repo_root)
        project = json.loads(json.dumps(base))
        for section, section_value in validate_project_document(value).items():
            if isinstance(section_value, dict) and isinstance(project.get(section), dict):
                project[section].update(section_value)
            else:
                project[section] = section_value
        name = _slug(project.get("name"), "eval-experiment")
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("project name must contain only letters, numbers, '.', '_' or '-'")
        project["name"] = name
        environment = project["environment"]
        configured_repo = Path(environment.get("repo_root") or self.repo_root).resolve()
        if configured_repo != self.repo_root:
            raise ValueError(f"repo_root must be {self.repo_root}")
        environment["repo_root"] = str(self.repo_root)
        for key in ("raw_root", "prepared_root", "models_root"):
            path = Path(environment[key]).expanduser().resolve()
            try:
                path.relative_to(self.repo_root)
            except ValueError as exc:
                message = f"{key} must stay inside repository root {self.repo_root}"
                raise ValueError(message) from exc
            environment[key] = str(path)
        runs_root = Path(environment["runs_root"]).expanduser()
        if not runs_root.is_absolute():
            runs_root = self.repo_root / runs_root
        environment["runs_root"] = str(runs_root.resolve())
        configured_run_dir = str(environment.get("run_dir") or "").strip()
        if configured_run_dir:
            run_dir = Path(configured_run_dir).expanduser()
            if not run_dir.is_absolute():
                run_dir = self.repo_root / run_dir
            environment["run_dir"] = str(run_dir.resolve())
        else:
            environment["run_dir"] = ""

        benchmark = project["benchmark"]
        task = str(benchmark.get("task") or "").strip()
        if not task:
            raise ValueError("ExperimentSpec requires benchmark.task")
        benchmark_spec = None
        benchmark_instance = None
        try:
            benchmark_spec, benchmark_instance = self.benchmarks.resolve_task(
                task,
                instance_id=str(benchmark.get("benchmark_instance_id") or "") or None,
            )
        except ValueError:
            if self.benchmarks.specs():
                raise
        if benchmark_spec:
            benchmark["benchmark_spec_id"] = benchmark_spec["id"]
            benchmark["prepare_target"] = (
                benchmark.get("prepare_target") or benchmark_spec["prepare_target"]
            )
            benchmark["capabilities"] = benchmark_spec.get("capabilities") or {}
            benchmark["evaluation_requirements"] = (
                benchmark_spec.get("evaluation_requirements") or {}
            )
            if benchmark_instance:
                benchmark["benchmark_instance_id"] = benchmark_instance["id"]
                benchmark["prepared_path"] = benchmark_instance["prepared_path"]
                benchmark["instance_status"] = benchmark_instance.get("status")

        runtime = project["runtime"]
        from ...runtime.execution.concurrency import normalize_concurrency_policy

        runtime["trial_concurrency"] = int(runtime.get("trial_concurrency", 1))
        runtime["concurrency_policy"] = normalize_concurrency_policy(
            runtime.get("concurrency_policy"),
            configured_concurrency=runtime["trial_concurrency"],
        )
        runtime_defaults = (
            dict(benchmark_spec.get("runtime_defaults") or {}) if benchmark_spec else {}
        )
        for key, value in runtime_defaults.items():
            if runtime.get(key) is None:
                runtime[key] = value
        if not runtime.get("docker_image"):
            runtime["docker_image"] = "lychee-python-sandbox:local"

        network = project.setdefault("network", {})
        network_defaults = (
            dict(benchmark_spec.get("network_defaults") or {}) if benchmark_spec else {}
        )
        if network.get("mode") == "benchmark_default":
            network = {**network, **network_defaults}
            if network.get("mode") == "benchmark_default":
                network["mode"] = "direct"
            project["network"] = network
        mode = str(network.get("mode") or "direct")
        if mode not in {"direct", "proxy", "offline"}:
            raise ValueError("network.mode must be benchmark_default, direct, proxy, or offline")
        network["mode"] = mode
        network["access"] = str(
            network.get("access") or network_defaults.get("access") or "optional"
        )
        targets = {
            "downloads": False,
            "web_surfer": False,
            "code_executor": False,
            "model_backend": False,
            **dict(network.get("targets") or {}),
        }
        network["targets"] = {key: bool(value) for key, value in targets.items()}
        network["no_proxy"] = str(network.get("no_proxy") or "127.0.0.1,localhost,::1")
        network["docker_bridge_host"] = str(network.get("docker_bridge_host") or "172.17.0.1")
        network["container_proxy_port"] = int(network.get("container_proxy_port") or 17897)
        network["probe_url"] = str(
            network.get("probe_url") or "https://www.google.com/generate_204"
        )
        if mode == "proxy":
            proxy_url = str(network.get("proxy_url") or "").strip()
            if not re.match(r"^https?://[^\s]+$", proxy_url):
                raise ValueError("network.proxy_url must be an HTTP(S) URL in proxy mode")
            network["proxy_url"] = proxy_url
            network["container_proxy_url"] = (
                f"http://host.docker.internal:{network['container_proxy_port']}"
            )
            if network["targets"].get("model_backend"):
                for instance in project.get("deployment_instances") or []:
                    if instance.get("kind") in {"api", "vllm"}:
                        instance["proxy_url"] = proxy_url
        else:
            network["proxy_url"] = ""
            network["container_proxy_url"] = ""
        method = str(project["runtime"].get("method", "none"))
        if method not in {"none", "nl_only", "latent_only", "both"}:
            raise ValueError(f"unsupported memory method {method!r}")
        observability = project.setdefault("observability", {})
        observability["collect_vllm_metrics"] = bool(
            observability.get("collect_vllm_metrics", True)
        )
        try:
            observability["vllm_metrics_interval_s"] = float(
                observability.get("vllm_metrics_interval_s", 5.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("vllm_metrics_interval_s must be positive") from exc
        if observability["vllm_metrics_interval_s"] <= 0:
            raise ValueError("vllm_metrics_interval_s must be positive")
        try:
            observability["event_log_max_events_per_file"] = int(
                observability.get("event_log_max_events_per_file", 10000)
            )
            observability["event_log_max_mib_per_file"] = float(
                observability.get("event_log_max_mib_per_file", 64.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("EventLog segment limits must be positive") from exc
        if (
            observability["event_log_max_events_per_file"] <= 0
            or observability["event_log_max_mib_per_file"] <= 0
        ):
            raise ValueError("EventLog segment limits must be positive")
        evaluation = project.setdefault("evaluation", {})
        external_evaluator = str(evaluation.get("external_evaluator") or "auto")
        if external_evaluator not in {"auto", "run", "skip"}:
            raise ValueError("evaluation.external_evaluator must be auto, run, or skip")
        evaluation["external_evaluator"] = external_evaluator
        evaluation["profile_id"] = str(evaluation.get("profile_id") or "core")
        evaluation_requirements = dict(benchmark.get("evaluation_requirements") or {})
        required_when = {
            str(item)
            for item in evaluation_requirements.get(
                "required_when_external_evaluator", []
            )
        }
        required_config_key = str(evaluation_requirements.get("config_key") or "")
        if required_config_key and external_evaluator in required_when:
            required_config = evaluation.get(required_config_key)
            if not isinstance(required_config, dict) or not required_config:
                raise ValueError(
                    f"benchmark {benchmark.get('benchmark_spec_id')!r} requires "
                    f"evaluation.{required_config_key} when external_evaluator="
                    f"{external_evaluator!r}"
                )
            for key in evaluation_requirements.get("required_fields") or []:
                if not str(required_config.get(str(key)) or "").strip():
                    raise ValueError(
                        f"evaluation.{required_config_key}.{key} is required for "
                        f"benchmark {benchmark.get('benchmark_spec_id')!r}"
                    )
        hle_judge = dict(evaluation.get("hle_judge") or {})
        if hle_judge:
            for key in ("model", "base_url"):
                if not str(hle_judge.get(key) or "").strip():
                    raise ValueError(f"evaluation.hle_judge.{key} is required")
            hle_judge["workers"] = max(1, int(hle_judge.get("workers", 8)))
            hle_judge["timeout_s"] = max(1.0, float(hle_judge.get("timeout_s", 300.0)))
            hle_judge["max_tokens"] = max(1, int(hle_judge.get("max_tokens", 4096)))
            hle_judge["thinking_mode"] = str(
                hle_judge.get("thinking_mode") or "inherit"
            )
            if hle_judge["thinking_mode"] not in {"inherit", "enabled", "disabled"}:
                raise ValueError(
                    "evaluation.hle_judge.thinking_mode must be inherit, enabled or disabled"
                )
            hle_judge["max_attempts"] = max(1, int(hle_judge.get("max_attempts", 3)))
            hle_judge["output_mode"] = str(
                hle_judge.get("output_mode") or "official_schema"
            )
            if hle_judge["output_mode"] not in {
                "official_schema",
                "local_json_object",
            }:
                raise ValueError(
                    "evaluation.hle_judge.output_mode must be official_schema or "
                    "local_json_object"
                )
            hle_judge["auth_mode"] = str(hle_judge.get("auth_mode") or "env")
            if hle_judge["auth_mode"] not in {"env", "none"}:
                raise ValueError("evaluation.hle_judge.auth_mode must be env or none")
            hle_judge["api_key_env"] = str(hle_judge.get("api_key_env") or "OPENAI_API_KEY")
        evaluation["hle_judge"] = hle_judge
        execution = project.setdefault("execution", {})
        raw_segment_limit = execution.get("segment_case_limit")
        if isinstance(raw_segment_limit, str) and raw_segment_limit.strip().lower() in {
            "",
            "all",
            "none",
            "null",
            "unlimited",
            "unbounded",
        }:
            raw_segment_limit = None
        if raw_segment_limit is None:
            execution["segment_case_limit"] = None
        else:
            execution["segment_case_limit"] = int(raw_segment_limit)
            if execution["segment_case_limit"] < 1:
                raise ValueError("execution.segment_case_limit must be positive or null")
        instances = project.get("deployment_instances") or []
        if not instances:
            raise ValueError("at least one DeploymentInstance is required")
        instance_ids = set()
        for instance in instances:
            instance_id = _slug(instance.get("id"), "deployment-instance")
            if instance_id in instance_ids:
                raise ValueError(f"duplicate DeploymentInstance id {instance_id!r}")
            instance["id"] = instance_id
            instance_ids.add(instance_id)
            kind = instance.get("kind")
            if kind not in {"hf", "vllm", "api"}:
                raise ValueError(
                    f"DeploymentInstance {instance_id!r} has unsupported kind {kind!r}"
                )
            if not instance.get("model_id"):
                raise ValueError(f"DeploymentInstance {instance_id!r} requires model_id")
            if not instance.get("actual_pricing_instance_id"):
                raise ValueError(
                    f"DeploymentInstance {instance_id!r} requires actual_pricing_instance_id"
                )
            if kind == "hf" and not instance.get("model_path"):
                raise ValueError(f"HF DeploymentInstance {instance_id!r} requires model_path")
            if kind == "vllm":
                instance.setdefault("host", "127.0.0.1")
                instance.setdefault("port", 8000)
                instance["base_url"] = instance.get("base_url") or (
                    f"http://{instance['host']}:{instance['port']}/v1"
                )
            if kind in {"vllm", "api"}:
                auth_mode = str(instance.get("auth_mode") or "env")
                if auth_mode not in {"env", "none"}:
                    raise ValueError(
                        f"DeploymentInstance {instance_id!r} has invalid auth_mode {auth_mode!r}"
                    )
                instance["auth_mode"] = auth_mode
                instance["trust_env"] = bool(instance.get("trust_env", True))
                if auth_mode == "env":
                    instance.setdefault(
                        "api_key_env", "DASHSCOPE_API_KEY" if kind == "api" else "VLLM_API_KEY"
                    )
                    api_key_env = instance.get("api_key_env")
                    if not _ENV_NAME.fullmatch(str(api_key_env or "")):
                        raise ValueError(
                            f"DeploymentInstance {instance_id!r} has invalid "
                            f"api_key_env {api_key_env!r}"
                        )
                else:
                    instance.pop("api_key_env", None)
            status = str(instance.get("status") or "")
            if status not in {"running", "ready_on_run"} or instance.get("available") is False:
                raise ValueError(
                    f"DeploymentInstance {instance_id!r} is not available (status={status!r})"
                )
        team = normalize_team_spec_document(project["team"])
        project["team"] = team
        coordination = compile_coordination(team)
        framework = str(runtime.get("framework") or "autogen")
        support = build_framework_binding_report(framework, coordination, team)
        if not support["supported"]:
            raise ValueError(str(support["reason"]))
        runtime["framework"] = framework
        runtime["framework_implementation"] = support["implementation"]
        runtime["framework_binding_report"] = support
        model_requirements = model_resource_requirements(team)
        if not model_requirements:
            raise ValueError("resolved TeamSpec requires at least one model-inference Node")
        try:
            max_rounds = int(runtime.get("max_rounds", 4))
            max_turns = int(runtime.get("max_turns", 20))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_rounds and max_turns must be positive integers") from exc
        if max_rounds < 1 or max_turns < 1:
            raise ValueError("max_rounds and max_turns must be positive integers")
        runtime["max_rounds"] = max_rounds
        runtime["max_turns"] = max_turns
        try:
            runtime["max_attempts_per_trial"] = int(runtime.get("max_attempts_per_trial", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_attempts_per_trial must be a positive integer") from exc
        if runtime["max_attempts_per_trial"] < 1:
            raise ValueError("max_attempts_per_trial must be a positive integer")
        try:
            runtime["trials_per_case"] = int(runtime.get("trials_per_case", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("trials_per_case must be a positive integer") from exc
        if runtime["trials_per_case"] < 1:
            raise ValueError("trials_per_case must be a positive integer")
        runtime["on_trial_error"] = str(runtime.get("on_trial_error") or "continue")
        if runtime["on_trial_error"] not in {"continue", "fail-fast"}:
            raise ValueError("on_trial_error must be continue or fail-fast")
        if not coordination.get("operations") and not coordination.get("handoff_relations"):
            round_derived_limit = max(1, len(coordination["members"])) * max_rounds
            runtime["effective_max_turns"] = min(max_turns, round_derived_limit)
        else:
            runtime["effective_max_turns"] = max_turns
        raw_model_call_limit = runtime.get("max_model_calls_per_case")
        if isinstance(raw_model_call_limit, str) and raw_model_call_limit.strip().lower() in {
            "",
            "none",
            "null",
            "unlimited",
            "unbounded",
        }:
            raw_model_call_limit = None
        if raw_model_call_limit is None:
            runtime["max_model_calls_per_case"] = None
        else:
            try:
                runtime["max_model_calls_per_case"] = int(raw_model_call_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_model_calls_per_case must be a positive integer or unlimited"
                ) from exc
            if runtime["max_model_calls_per_case"] < 1:
                raise ValueError("max_model_calls_per_case must be a positive integer or unlimited")
        raw_wall_time_limit = runtime.get("max_case_wall_time_s")
        if isinstance(raw_wall_time_limit, str) and raw_wall_time_limit.strip().lower() in {
            "",
            "none",
            "null",
            "unlimited",
            "unbounded",
        }:
            raw_wall_time_limit = None
        if raw_wall_time_limit is None:
            runtime["max_case_wall_time_s"] = None
        else:
            try:
                runtime["max_case_wall_time_s"] = float(raw_wall_time_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_case_wall_time_s must be a positive number or unlimited"
                ) from exc
            if runtime["max_case_wall_time_s"] <= 0:
                raise ValueError("max_case_wall_time_s must be a positive number or unlimited")
        runtime_max_new_tokens = int(runtime.get("max_new_tokens", 16384))
        runtime["max_new_tokens"] = runtime_max_new_tokens
        runtime_budget = normalize_token_budget_policy(
            {key: runtime[key] for key in BUDGET_POLICY_FIELDS if runtime.get(key) is not None},
            max_new_tokens=runtime_max_new_tokens,
        )
        runtime.update(runtime_budget.to_dict())

        bindings = project.setdefault("deployment_bindings", {})
        default_id = str(bindings.get("default_deployment_instance_id") or instances[0]["id"])
        control_id = str(bindings.get("control_deployment_instance_id") or default_id)
        if default_id not in instance_ids:
            raise ValueError(f"unknown default DeploymentInstance {default_id!r}")
        if control_id not in instance_ids:
            raise ValueError(f"unknown control DeploymentInstance {control_id!r}")
        bindings["default_deployment_instance_id"] = default_id
        bindings["control_deployment_instance_id"] = control_id
        role_ids = {str(requirement["node_id"]) for requirement in model_requirements}
        seen_roles: set[str] = set()
        normalized_bindings = []
        for item in bindings.get("role_bindings") or []:
            role_id = str(item.get("role_id") or "")
            deployment_instance_id = str(item.get("deployment_instance_id") or default_id)
            if not role_id:
                raise ValueError("role DeploymentInstance binding requires role_id")
            if role_ids and role_id not in role_ids:
                raise ValueError(f"deployment binding references unknown role {role_id!r}")
            if role_id in seen_roles:
                raise ValueError(f"duplicate deployment binding for role {role_id!r}")
            if deployment_instance_id not in instance_ids:
                raise ValueError(
                    f"role {role_id!r} references unknown DeploymentInstance "
                    f"{deployment_instance_id!r}"
                )
            seen_roles.add(role_id)
            generation_overrides = normalize_invocation_overrides(
                item.get("generation_overrides"), label=f"role {role_id}"
            )
            normalize_token_budget_policy(
                {
                    key: generation_overrides.get(key, getattr(runtime_budget, key))
                    for key in BUDGET_POLICY_FIELDS
                    if generation_overrides.get(key, getattr(runtime_budget, key)) is not None
                },
                max_new_tokens=int(
                    generation_overrides.get("max_new_tokens", runtime_max_new_tokens)
                ),
            )
            normalized_bindings.append(
                {
                    "role_id": role_id,
                    "deployment_instance_id": deployment_instance_id,
                    **(
                        {"generation_overrides": generation_overrides}
                        if generation_overrides
                        else {}
                    ),
                }
            )
        bindings["role_bindings"] = normalized_bindings
        missing_roles = sorted(role_ids - seen_roles)
        if missing_roles:
            raise ValueError(
                "TeamInstance requires one DeploymentInstance binding for every "
                f"model-inference Node; missing: {', '.join(missing_roles)}"
            )
        control_overrides = normalize_invocation_overrides(
            bindings.get("control_generation_overrides"),
            label="coordination-operation Node",
        )
        normalize_token_budget_policy(
            {
                key: control_overrides.get(key, getattr(runtime_budget, key))
                for key in BUDGET_POLICY_FIELDS
                if control_overrides.get(key, getattr(runtime_budget, key)) is not None
            },
            max_new_tokens=int(control_overrides.get("max_new_tokens", runtime_max_new_tokens)),
        )
        bindings["control_generation_overrides"] = control_overrides
        instance_by_id = {item["id"]: item for item in instances}
        for item in normalized_bindings:
            raw_generation_overrides = item.get("generation_overrides")
            item_generation_overrides = (
                raw_generation_overrides if isinstance(raw_generation_overrides, dict) else {}
            )
            effective_overrides = {
                **{
                    key: runtime.get(key)
                    for key in BUDGET_POLICY_FIELDS
                    if runtime.get(key) is not None
                },
                **item_generation_overrides,
            }
            self._validate_invocation_capabilities(
                instance_by_id[item["deployment_instance_id"]],
                effective_overrides,
                label=f"Node {item['role_id']}",
            )
        effective_control_overrides = {
            **{
                key: runtime.get(key)
                for key in BUDGET_POLICY_FIELDS
                if runtime.get(key) is not None
            },
            **control_overrides,
        }
        self._validate_invocation_capabilities(
            instance_by_id[control_id],
            effective_control_overrides,
            label="coordination function Node",
        )
        team_instance = project.setdefault("team_instance", {})
        team_instance["id"] = _slug(
            team_instance.get("id") or f"{team.get('id', 'team')}-instance",
            "team-instance",
        )
        team_instance["team_spec_id"] = str(team["id"])
        if method in {"latent_only", "both"}:
            non_hf = [item["id"] for item in instances if item["kind"] != "hf"]
            if non_hf:
                raise ValueError(
                    "latent_only/both currently require HF DeploymentInstances; incompatible: "
                    + ", ".join(non_hf)
                )
            if len(instances) != 1:
                raise ValueError("latent_only/both require one shared HF DeploymentInstance")
        self._resolve_hf_devices(project)
        return project

    @staticmethod
    def _resolve_hf_devices(project: dict[str, Any]) -> None:
        """Map physical HF allocations to one process-wide logical CUDA namespace."""
        hf_instances = [
            item for item in project.get("deployment_instances") or [] if item.get("kind") == "hf"
        ]
        physical_devices: list[str] = []
        for instance in hf_instances:
            configured = [
                item.strip()
                for item in str(instance.get("cuda_visible_devices") or "0").split(",")
                if item.strip()
            ]
            if len(configured) != 1:
                raise ValueError(
                    f"HF DeploymentInstance {instance['id']!r} must allocate exactly one "
                    "physical CUDA device; create separate DeploymentInstances for separate GPUs"
                )
            physical = configured[0]
            if physical in physical_devices:
                raise ValueError(
                    f"HF DeploymentInstances cannot share physical CUDA device {physical!r} "
                    "inside one experiment process"
                )
            physical_devices.append(physical)
            instance["physical_cuda_device"] = physical
        for instance in hf_instances:
            instance["runtime_device"] = (
                f"cuda:{physical_devices.index(instance['physical_cuda_device'])}"
            )
        project["runtime"]["hf_cuda_visible_devices"] = ",".join(physical_devices)
        snapshot = project.get("configuration_snapshot") or {}
        snapshot_by_id = {
            str(item.get("id") or ""): item for item in snapshot.get("deployment_instances") or []
        }
        for instance in hf_instances:
            snapshotted = snapshot_by_id.get(str(instance["id"]))
            if snapshotted is not None:
                snapshotted["physical_cuda_device"] = instance["physical_cuda_device"]
                snapshotted["runtime_device"] = instance["runtime_device"]

    @staticmethod
    def _validate_invocation_capabilities(
        deployment: dict[str, Any],
        overrides: dict[str, Any],
        *,
        label: str,
    ) -> None:
        """Reject per-call thinking controls that the selected provider cannot honor."""

        mode = overrides.get("thinking_mode")
        preserve = overrides.get("preserve_thinking")
        budget = overrides.get("max_thinking_budget_tokens")
        if mode is None and preserve is None and budget is None:
            return
        kind = str(deployment.get("kind") or "")
        protocol = str(deployment.get("thinking_protocol") or "").strip()
        if not protocol:
            model = str(deployment.get("model_id") or "").lower()
            base_url = str(deployment.get("base_url") or "").lower()
            if kind == "vllm" and "qwen" in model:
                protocol = "qwen_chat_template"
            elif "dashscope" in base_url or "modelstudio" in base_url or "maas" in base_url:
                protocol = "dashscope"
            elif "api.deepseek.com" in base_url:
                protocol = "deepseek"
        capabilities = dict(deployment.get("capabilities") or {})
        if mode is not None and capabilities.get("thinking_toggle") is not True:
            raise ValueError(
                f"{label} requests thinking_mode, but deployment "
                f"{deployment['id']!r} does not declare thinking_toggle support"
            )
        if preserve is not None and capabilities.get("preserve_thinking") is not True:
            raise ValueError(
                f"{label} requests preserve_thinking, but deployment "
                f"{deployment['id']!r} does not declare preserve_thinking support"
            )
        if protocol not in {"qwen_chat_template", "dashscope", "deepseek"}:
            raise ValueError(
                f"{label} requests thinking control, but deployment "
                f"{deployment['id']!r} has no supported thinking_protocol"
            )
        if budget is not None and not bool(capabilities.get("native_thinking_budget")):
            raise ValueError(
                f"{label} requests max_thinking_budget_tokens={budget}, but deployment "
                f"{deployment['id']!r} does not declare native thinking budget support"
            )

    def compile(
        self,
        value: dict[str, Any],
        *,
        launch_id: str | None = None,
        launch_root: str | os.PathLike | None = None,
    ) -> CompiledPlan:
        project = self.normalize(value)
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
        launch_id = (
            _slug(launch_id, f"{project['name']}-{stamp}")
            if launch_id
            else (f"{project['name']}-{stamp}")
        )
        env = project["environment"]
        launch_base = Path(
            launch_root or Path(env["repo_root"]) / "runs/eval_studio/launches"
        ).resolve()
        launch_dir = launch_base / launch_id
        runtime = project["runtime"]
        benchmark = project["benchmark"]
        team = project["team"]
        instances = project["deployment_instances"]
        bindings = project["deployment_bindings"]
        instance_by_id = {item["id"]: item for item in instances}
        control = instance_by_id[bindings["control_deployment_instance_id"]]
        model_tag = _slug(project.get("model_tag") or self._model_tag(instances), "mixed-models")
        benchmark_id = _slug(benchmark.get("benchmark_spec_id"), "benchmark")
        task_id = _slug(benchmark.get("task"), "task")
        team_id = _slug(team.get("id"), "studio-team")
        method = _slug(runtime["method"], "none")
        experiment_id = _slug(project.get("name"), "experiment")
        automatic_run_dir = (
            Path(env["runs_root"])
            / benchmark_id
            / task_id
            / team_id
            / method
            / experiment_id
            / stamp
        )
        run_dir = Path(env.get("run_dir") or automatic_run_dir)
        env["run_dir"] = str(run_dir)
        snapshot = project.get("configuration_snapshot") or {}
        if snapshot:
            snapshot["captured_at_utc"] = now.isoformat()
            snapshot.setdefault("experiment_instance", {})["run_dir"] = str(run_dir)

        files = {
            "project.json": json.dumps(project, ensure_ascii=False, indent=2) + "\n",
            "team_spec.json": json.dumps(self._team_spec(project), ensure_ascii=False, indent=2)
            + "\n",
            "deployments.json": json.dumps(
                self._runtime_deployment_config(instances, bindings),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "runtime_config.yaml": json.dumps(
                self._runtime_config(
                    project,
                    control,
                    experiment_instance_id=str(
                        (snapshot.get("experiment_instance") or {}).get("id") or launch_id
                    ),
                ),
                indent=2,
            )
            + "\n",
        }
        files.update(self._configuration_snapshot_files(project))
        header = self._shell_header(project, launch_dir, run_dir)
        files["prepare.sh"] = header + self._prepare_body(project)
        files["verify_deployment_instances.sh"] = header + self._verification_body(project)
        files["run.sh"] = header + self._run_body(project, launch_dir, model_tag)
        files["resume.sh"] = header + self._run_body(project, launch_dir, model_tag, resume=True)
        files["analyze.sh"] = header + self._analyze_body(project, run_dir)
        files["run_all.sh"] = (
            header
            + 'bash "$LAUNCH_DIR/prepare.sh"\n'
            + 'bash "$LAUNCH_DIR/verify_deployment_instances.sh"\n'
            + 'bash "$LAUNCH_DIR/run.sh"\n'
            + 'bash "$LAUNCH_DIR/analyze.sh"\n'
        )
        commands = self._launch_commands(project, launch_dir)
        manifest = {
            "schema_version": 1,
            "launch_id": launch_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_name": project["name"],
            "run_dir": str(run_dir),
            "launcher": dict(project["execution"]),
            "files": sorted([*files, "launch_manifest.json", "launch.log"]),
            "commands": commands,
        }
        files["launch_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        return CompiledPlan(launch_id, launch_dir, project, files, commands, run_dir)

    def materialize(self, plan: CompiledPlan) -> None:
        plan.launch_dir.mkdir(parents=True, exist_ok=True)
        for name, content in plan.files.items():
            path = plan.launch_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
            if path.suffix == ".sh":
                path.chmod(0o750)

    @staticmethod
    def _configuration_snapshot_files(project: dict[str, Any]) -> dict[str, str]:
        snapshot = dict(project.get("configuration_snapshot") or {})
        if not snapshot:
            return {}
        entries = {
            "configuration_snapshot.json": snapshot,
            "experiment_spec.json": snapshot.get("experiment_spec") or {},
            "experiment_instance.json": snapshot.get("experiment_instance") or {},
            "benchmark_spec.json": snapshot.get("benchmark_spec") or {},
            "benchmark_instance.json": snapshot.get("benchmark_instance") or {},
            "team_spec.json": snapshot.get("team_spec") or {},
            "team_instance.json": snapshot.get("team_instance") or {},
            "deployment_specs.json": snapshot.get("deployment_specs") or [],
            "deployment_instances.json": snapshot.get("deployment_instances") or [],
            "pricing_specs.json": snapshot.get("pricing_specs") or [],
            "pricing_instances.json": snapshot.get("pricing_instances") or [],
            "resource_specs.json": snapshot.get("resource_specs") or {},
            "resource_instances.json": snapshot.get("resource_instances") or {},
            "resolved_project.json": project,
        }
        return {
            f"config_snapshot/{name}": json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            for name, value in entries.items()
        }

    @staticmethod
    def _model_tag(instances: list[dict[str, Any]]) -> str:
        names = {
            str(item.get("model_id") or Path(str(item.get("model_path", "model"))).name)
            for item in instances
        }
        return next(iter(names)) if len(names) == 1 else "mixed-models"

    @staticmethod
    def _runtime_deployment_config(
        instances: list[dict[str, Any]], bindings: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate Studio instance names to the stable run_mas deployment contract."""

        runtime_bindings = {
            "default_deployment_id": bindings["default_deployment_instance_id"],
            "control_deployment_id": bindings["control_deployment_instance_id"],
            "role_bindings": [
                {
                    "role_id": item["role_id"],
                    "deployment_id": item["deployment_instance_id"],
                    **(
                        {"generation_overrides": item["generation_overrides"]}
                        if item.get("generation_overrides")
                        else {}
                    ),
                }
                for item in bindings.get("role_bindings") or []
            ],
        }
        if bindings.get("control_generation_overrides"):
            runtime_bindings["control_generation_overrides"] = bindings[
                "control_generation_overrides"
            ]
        return {
            "deployments": instances,
            "team_deployment_bindings": runtime_bindings,
        }

    @staticmethod
    def _team_spec(project: dict[str, Any]) -> dict[str, Any]:
        team = project["team"]
        return {"schema_version": 4, **team}

    @staticmethod
    def _runtime_config(
        project: dict[str, Any],
        control: dict[str, Any],
        *,
        experiment_instance_id: str,
    ) -> dict[str, Any]:
        runtime = project["runtime"]
        network = project.get("network") or {}
        provider = "hf" if control["kind"] == "hf" else "api"
        return {
            "experiment": {"instance_id": experiment_instance_id},
            "runtime": {
                "name": runtime.get("framework", "autogen"),
                "framework": runtime.get("framework", "autogen"),
                "max_turns": runtime.get("effective_max_turns", runtime.get("max_turns", 20)),
                "code_executor": runtime.get("code_executor", "docker"),
                "code_timeout": runtime.get("code_timeout", 60),
                "work_root": runtime.get("work_root", "runs/lychee_tool_workspaces"),
                "web_headless": runtime.get("web_headless", True),
                "save_screenshots": runtime.get("save_screenshots", False),
                "docker_image": runtime.get("docker_image", "lychee-python-sandbox:local"),
                "trace_model_calls": runtime.get("trace_model_calls", True),
            },
            "observability": dict(project.get("observability") or {}),
            "network": network,
            "backend": {
                "provider": provider,
                "model_path": control.get("model_path"),
                "model": control.get("model_id"),
                "base_url": control.get("base_url"),
                "api_key_env": control.get("api_key_env", "VLLM_API_KEY"),
                "auth_mode": control.get("auth_mode", "env"),
                "trust_env": control.get("trust_env", True),
                "proxy_url": control.get("proxy_url"),
                "do_sample": runtime.get("do_sample", False),
                "temperature": runtime.get("temperature", 0.7),
                "top_p": runtime.get("top_p", 0.8),
                "top_k": runtime.get("top_k"),
                "min_p": runtime.get("min_p"),
                "presence_penalty": runtime.get("presence_penalty"),
                "repetition_penalty": runtime.get("repetition_penalty", 1.0),
                "seed": runtime.get("seed", 0),
            },
            "memory": {"name": "cdm", "P": runtime.get("P", 16)},
            "router": {"method": runtime.get("method", "none")},
            "run": {
                "task": project["benchmark"]["task"],
                "n": project["benchmark"].get("n", 10),
                "trials_per_case": runtime.get("trials_per_case", 1),
                "max_rounds": runtime.get("max_rounds", 4),
                "max_model_calls_per_case": runtime.get("max_model_calls_per_case"),
                "max_case_wall_time_s": runtime.get("max_case_wall_time_s"),
                "max_attempts_per_trial": runtime.get("max_attempts_per_trial", 1),
                "on_trial_error": runtime.get("on_trial_error", "continue"),
                "max_new_tokens": runtime.get("max_new_tokens", 16384),
                **{
                    key: runtime.get(key)
                    for key in BUDGET_POLICY_FIELDS
                    if runtime.get(key) is not None
                },
                "trial_concurrency": runtime.get("trial_concurrency", 1),
                "concurrency_policy": dict(runtime.get("concurrency_policy") or {}),
            },
            "eval": {"runs_root": project["environment"]["runs_root"]},
        }

    @staticmethod
    def _shell_header(project: dict[str, Any], launch_dir: Path, run_dir: Path) -> str:
        env = project["environment"]
        benchmark = project["benchmark"]
        network = project.get("network") or {}
        download_proxy_exports = ""
        if network.get("mode") == "proxy" and (network.get("targets") or {}).get("downloads"):
            proxy_url = _q(network["proxy_url"])
            no_proxy = _q(network.get("no_proxy") or "127.0.0.1,localhost,::1")
            download_proxy_exports = (
                f"export HTTP_PROXY={proxy_url}\n"
                f"export HTTPS_PROXY={proxy_url}\n"
                f"export http_proxy={proxy_url}\n"
                f"export https_proxy={proxy_url}\n"
                f"export NO_PROXY={no_proxy}\n"
                f"export no_proxy={no_proxy}\n"
            )
        prepared_override = (
            {str(benchmark["prepare_target"]): str(benchmark["prepared_path"])}
            if benchmark.get("prepared_path")
            else {}
        )
        return (
            "#!/usr/bin/env bash\nset -eo pipefail\n\n"
            f"REPO_ROOT={_q(env['repo_root'])}\n"
            f"LAUNCH_DIR={_q(launch_dir)}\n"
            f"CONDA_SH={_q(env.get('conda_sh', ''))}\n"
            f"CONDA_ENV={_q(env.get('conda_env', ''))}\n"
            f"PY={_q(env['python'])}\n"
            f"export PYTHONPATH={_q(str(Path(env['repo_root']) / 'src'))}\n"
            f"export LYCHEE_BENCHMARK_RAW_ROOT={_q(env['raw_root'])}\n"
            f"export LYCHEE_BENCHMARK_PREPARED_ROOT={_q(env['prepared_root'])}\n"
            "export LYCHEE_BENCHMARK_PREPARED_OVERRIDES="
            f"{_q(json.dumps(prepared_override, ensure_ascii=False))}\n"
            f"export LYCHEE_BENCHMARK_RUNS_ROOT={_q(env['runs_root'])}\n"
            f"export LYCHEE_BENCHMARK_RUN_DIR={_q(run_dir)}\n"
            f"{download_proxy_exports}"
            'if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then source "$CONDA_SH"; fi\n'
            'if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then '
            'conda activate "$CONDA_ENV"; fi\n'
            "set -u\n"
            'cd "$REPO_ROOT"\n\n'
            'mkdir -p "$LYCHEE_BENCHMARK_RUN_DIR/config_snapshot"\n'
            'if [[ -d "$LAUNCH_DIR/config_snapshot" ]]; then\n'
            '  cp -a "$LAUNCH_DIR/config_snapshot/." '
            '"$LYCHEE_BENCHMARK_RUN_DIR/config_snapshot/"\n'
            "fi\n\n"
        )

    @staticmethod
    def _prepare_body(project: dict[str, Any]) -> str:
        benchmark = project["benchmark"]
        if not benchmark.get("prepare_before_run", False):
            return 'echo "[studio] data preparation skipped by project config"\n'
        return (
            '"$PY" scripts/prepare_benchmarks.py \\\n'
            '  --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" \\\n'
            '  --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" \\\n'
            f"  --source {_q(benchmark.get('source', 'auto'))} \\\n"
            f"  --tasks {_q(benchmark['prepare_target'])}\n"
        )

    @staticmethod
    def _verification_body(project: dict[str, Any]) -> str:
        """Emit defensive launch-time checks without creating new deployments."""

        blocks = []
        network = project.get("network") or {}
        if network.get("mode") == "proxy":
            proxy_url = _q(network["proxy_url"])
            probe_url = _q(network.get("probe_url", "https://www.google.com/generate_204"))
            blocks.append(
                f"curl -fsS --max-time 12 --proxy {proxy_url} {probe_url} >/dev/null || "
                "{ echo '[studio] configured experiment proxy is unreachable' >&2; exit 2; }\n"
            )
            if network.get("targets", {}).get("code_executor"):
                blocks.append(
                    "command -v socat >/dev/null 2>&1 || "
                    "{ echo '[studio] socat is required for Docker proxy relay' >&2; exit 2; }\n"
                )
        for instance in project["deployment_instances"]:
            instance_id = instance["id"]
            if instance["kind"] == "hf":
                blocks.append(
                    f"test -e {_q(instance['model_path'])} || "
                    f"{{ echo '[studio] unavailable DeploymentInstance {instance_id}: "
                    "model path missing' >&2; exit 2; }\n"
                )
                continue
            base_url = str(instance.get("base_url") or "").rstrip("/")
            proxy_option = (
                f"--proxy {_q(instance['proxy_url'])}"
                if instance.get("proxy_url")
                else "--noproxy '*'"
            )
            blocks.append(
                f"curl {proxy_option} -fsS {_q(base_url + '/models')} >/dev/null || "
                f"{{ echo '[studio] unavailable DeploymentInstance {instance_id}: "
                "endpoint probe failed' >&2; exit 2; }\n"
            )
        return "".join(blocks) or 'echo "[studio] no DeploymentInstance to verify"\n'

    @staticmethod
    def _run_body(
        project: dict[str, Any], launch_dir: Path, model_tag: str, *, resume: bool = False
    ) -> str:
        benchmark = project["benchmark"]
        runtime = project["runtime"]
        network = project.get("network") or {}
        instances = {item["id"]: item for item in project["deployment_instances"]}
        control = instances[project["deployment_bindings"]["control_deployment_instance_id"]]
        backend = "hf" if control["kind"] == "hf" else "api"
        args = [
            '"$PY"',
            "scripts/run_mas.py",
            "--config",
            '"$LAUNCH_DIR/runtime_config.yaml"',
            "--runtime",
            _q(runtime.get("framework", "autogen")),
            "--backend",
            backend,
            "--task",
            _q(benchmark["task"]),
            "--team-spec",
            '"$LAUNCH_DIR/team_spec.json"',
            "--deployment-config",
            '"$LAUNCH_DIR/deployments.json"',
            "--method",
            _q(runtime["method"]),
            "--n",
            _q(benchmark.get("n", 10)),
            "--start-index",
            _q(benchmark.get("start_index", 0)),
            "--case-selection",
            _q(benchmark.get("case_selection", "head")),
            *(
                ["--case-strata-field", _q(benchmark["case_strata_field"])]
                if benchmark.get("case_strata_field")
                else []
            ),
            "--trials-per-case",
            _q(runtime.get("trials_per_case", 1)),
            "--max-rounds",
            _q(runtime.get("max_rounds", 4)),
            "--max-turns",
            _q(runtime.get("effective_max_turns", runtime.get("max_turns", 20))),
            "--max-model-calls-per-case",
            _q(runtime.get("max_model_calls_per_case") or "unlimited"),
            "--max-case-wall-time-s",
            _q(runtime.get("max_case_wall_time_s") or "unlimited"),
            "--max-attempts-per-trial",
            _q(runtime.get("max_attempts_per_trial", 1)),
            "--on-trial-error",
            _q(runtime.get("on_trial_error", "continue")),
            "--max-new-tokens",
            _q(runtime.get("max_new_tokens", 16384)),
            "--min-output-reserve-tokens",
            _q(runtime.get("min_output_reserve_tokens", 2048)),
            "--min-thinking-reserve-tokens",
            _q(runtime.get("min_thinking_reserve_tokens", 0)),
            "--min-final-reserve-tokens",
            _q(runtime.get("min_final_reserve_tokens", 1024)),
            "--safety-margin-tokens",
            _q(runtime.get("safety_margin_tokens", 256)),
            "--trial-concurrency",
            _q(runtime.get("trial_concurrency", 1)),
            "--concurrency-mode",
            _q((runtime.get("concurrency_policy") or {}).get("mode", "fixed")),
            "--concurrency-initial",
            _q((runtime.get("concurrency_policy") or {}).get("initial", 1)),
            "--concurrency-minimum",
            _q((runtime.get("concurrency_policy") or {}).get("minimum", 1)),
            "--concurrency-maximum",
            _q(
                (runtime.get("concurrency_policy") or {}).get(
                    "maximum", runtime.get("trial_concurrency", 1)
                )
            ),
            "--concurrency-increase-step",
            _q((runtime.get("concurrency_policy") or {}).get("increase_step", 1)),
            "--concurrency-decrease-factor",
            _q((runtime.get("concurrency_policy") or {}).get("decrease_factor", 0.5)),
            "--concurrency-control-window-trials",
            _q((runtime.get("concurrency_policy") or {}).get("control_window_trials", 8)),
            "--seed",
            _q(runtime.get("seed", 0)),
            "--model-tag",
            _q(model_tag),
            "--code-executor",
            _q(runtime.get("code_executor", "docker")),
            "--code-timeout",
            _q(runtime.get("code_timeout", 60)),
            "--work-root",
            _q(runtime.get("work_root", "runs/lychee_tool_workspaces")),
            "--docker-image",
            _q(runtime.get("docker_image", "lychee-python-sandbox:local")),
            "--runs-root",
            '"$LYCHEE_BENCHMARK_RUNS_ROOT"',
            "--run-dir",
            '"$LYCHEE_BENCHMARK_RUN_DIR"',
        ]
        if network.get("mode") == "proxy":
            targets = network.get("targets") or {}
            if targets.get("web_surfer"):
                args.extend(["--web-proxy-url", _q(network["proxy_url"])])
            if targets.get("code_executor"):
                args.extend(
                    [
                        "--container-proxy-url",
                        _q(network["container_proxy_url"]),
                        "--proxy-relay-upstream-url",
                        _q(network["proxy_url"]),
                        "--proxy-relay-listen-host",
                        _q(network["docker_bridge_host"]),
                        "--proxy-relay-port",
                        _q(network["container_proxy_port"]),
                    ]
                )
            args.extend(["--proxy-no-proxy", _q(network["no_proxy"])])
        if runtime.get("max_input_tokens") is not None:
            args.extend(["--max-input-tokens", _q(runtime["max_input_tokens"])])
        if runtime.get("max_thinking_budget_tokens") is not None:
            args.extend(
                [
                    "--max-thinking-budget-tokens",
                    _q(runtime["max_thinking_budget_tokens"]),
                ]
            )
        if runtime.get("do_sample"):
            args.append("--do-sample")
        else:
            args.append("--no-do-sample")
        args.extend(["--temperature", _q(runtime.get("temperature", 0.7))])
        args.extend(["--top-p", _q(runtime.get("top_p", 0.8))])
        if runtime.get("top_k") is not None:
            args.extend(["--top-k", _q(runtime["top_k"])])
        if runtime.get("min_p") is not None:
            args.extend(["--min-p", _q(runtime["min_p"])])
        if runtime.get("presence_penalty") is not None:
            args.extend(["--presence-penalty", _q(runtime["presence_penalty"])])
        args.extend(["--repetition-penalty", _q(runtime.get("repetition_penalty", 1.0))])
        observability = project.get("observability") or {}
        if observability.get("collect_vllm_metrics", True):
            args.append("--collect-vllm-metrics")
        else:
            args.append("--no-collect-vllm-metrics")
        args.extend(
            [
                "--vllm-metrics-interval-s",
                _q(observability.get("vllm_metrics_interval_s", 5.0)),
            ]
        )
        args.extend(
            [
                "--event-log-max-events-per-file",
                _q(observability.get("event_log_max_events_per_file", 10000)),
                "--event-log-max-mib-per-file",
                _q(observability.get("event_log_max_mib_per_file", 64.0)),
            ]
        )
        if backend == "hf":
            args.extend(["--model-path", _q(control["model_path"])])
            args.extend(["--device", _q(control.get("runtime_device", "cuda:0"))])
        else:
            args.extend(["--api-model", _q(control["model_id"])])
            args.extend(["--api-base-url", _q(control.get("base_url", ""))])
            args.extend(["--api-key-env", _q(control.get("api_key_env", "VLLM_API_KEY"))])
        hf_devices = str(runtime.get("hf_cuda_visible_devices") or "")
        prefix = f"CUDA_VISIBLE_DEVICES={_q(hf_devices)} " if hf_devices else ""
        if resume:
            args.append("--resume")
        return prefix + " \\\n  ".join(args) + "\n"

    @staticmethod
    def _analyze_body(project: dict[str, Any], run_dir: Path) -> str:
        evaluation = dict(project.get("evaluation") or {})
        args = [
            '"$PY"',
            "scripts/analyze_benchmark_run.py",
            '"$RUN_DIR"',
            "--external-evaluator",
            _q(evaluation.get("external_evaluator", "auto")),
            "--evaluation-profile",
            _q(evaluation.get("profile_id", "core")),
        ]
        hle_judge = dict(evaluation.get("hle_judge") or {})
        if hle_judge:
            args.extend(
                [
                    "--hle-judge-model",
                    _q(hle_judge["model"]),
                    "--hle-judge-base-url",
                    _q(hle_judge["base_url"]),
                    "--hle-judge-auth-mode",
                    _q(hle_judge.get("auth_mode", "env")),
                    "--hle-judge-api-key-env",
                    _q(hle_judge.get("api_key_env", "OPENAI_API_KEY")),
                    "--hle-judge-workers",
                    _q(hle_judge.get("workers", 8)),
                    "--hle-judge-timeout",
                    _q(hle_judge.get("timeout_s", 300.0)),
                    "--hle-judge-max-tokens",
                    _q(hle_judge.get("max_tokens", 4096)),
                    "--hle-judge-thinking-mode",
                    _q(hle_judge.get("thinking_mode", "inherit")),
                    "--hle-judge-max-attempts",
                    _q(hle_judge.get("max_attempts", 3)),
                    "--hle-judge-output-mode",
                    _q(hle_judge.get("output_mode", "official_schema")),
                ]
            )
            if hle_judge.get("api_key") is not None:
                args.extend(["--hle-judge-api-key", _q(hle_judge["api_key"])])
        command = " \\\n  ".join(args) + "\n"
        return (
            f"RUN_DIR={_q(run_dir)}\n"
            'if [[ ! -f "$RUN_DIR/events/run_events.jsonl" ]]; then\n'
            '  echo "[studio] run events not found; skip analysis: $RUN_DIR" >&2\n'
            "  exit 2\n"
            "fi\n" + command
        )

    @staticmethod
    def _launch_commands(project: dict[str, Any], launch_dir: Path) -> dict[str, str]:
        execution = project["execution"]
        session = _slug(execution.get("tmux_session"), "lychee-eval")
        window = _slug(execution.get("tmux_window"), project["name"])
        run_all = launch_dir / "run_all.sh"
        return {
            "foreground": f"bash {_q(run_all)}",
            "new_tmux_session": (
                f"tmux new-session -d -s {_q(session)} "
                f"-c {_q(project['environment']['repo_root'])} "
                f"bash {_q(run_all)}"
            ),
            "existing_tmux_session": (
                f"tmux new-window -t {_q(session)} -n {_q(window)} "
                f"-c {_q(project['environment']['repo_root'])} bash {_q(run_all)}"
            ),
            "attach": f"tmux attach -t {_q(session)}",
            "resume": f"bash {_q(launch_dir / 'resume.sh')}",
            "analyze": f"bash {_q(launch_dir / 'analyze.sh')}",
        }
