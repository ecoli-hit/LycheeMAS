"""Persistence and validation for ExperimentSpec and ExperimentInstance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..contracts.lifecycle import bind_instance_to_spec, lifecycle_error, utc_now
from ..infrastructure.json_store import atomic_write_json as _write_json

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STATUSES = {"ready", "queued", "running", "paused", "completed", "failed", "stopped"}
_RESUMABLE_STATUSES = {"paused", "stopped", "failed"}
_LAUNCHER_TYPES = {"subprocess", "new_tmux_session", "existing_tmux_session"}
_EXPERIMENT_SPEC_FIELDS = {
    "schema_version",
    "id",
    "benchmark",
    "team_spec_id",
    "runtime",
    "observability",
    "evaluation",
    "network",
    "environment",
    "notes",
    "updated_at_utc",
    "registry_path",
}


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: " + ", ".join(sorted(unknown)))


def _utc_now() -> str:
    return utc_now()


def normalize_launcher(value: dict[str, Any] | None) -> dict[str, Any]:
    launcher = dict(value or {})
    launcher_type = str(launcher.get("type") or "subprocess")
    if launcher_type not in _LAUNCHER_TYPES:
        raise ValueError(
            "ExperimentInstance launcher.type must be subprocess, new_tmux_session, "
            "or existing_tmux_session"
        )
    result = {"type": launcher_type}
    if launcher_type in {"new_tmux_session", "existing_tmux_session"}:
        session = str(launcher.get("tmux_session") or "lychee-eval")
        window = str(launcher.get("tmux_window") or "benchmark")
        if not _SAFE_ID.fullmatch(session) or not _SAFE_ID.fullmatch(window):
            raise ValueError(
                "tmux session and window may contain only letters, numbers, '.', '_' and '-'"
            )
        result.update(tmux_session=session, tmux_window=window)
    return result


def normalize_instance_execution(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize mutable next-segment and cumulative Case controls."""

    execution = dict(value or {})
    unlimited_values = {"", "all", "none", "null", "unlimited", "unbounded"}

    def optional_positive(raw: Any, *, name: str) -> int | None:
        if isinstance(raw, str) and raw.strip().lower() in unlimited_values:
            raw = None
        if raw is None:
            return None
        try:
            normalized = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ExperimentInstance execution.{name} must be positive or null"
            ) from exc
        if normalized < 1:
            raise ValueError(f"ExperimentInstance execution.{name} must be positive or null")
        return normalized

    return {
        "next_segment_case_limit": optional_positive(
            execution.get(
                "next_segment_case_limit",
                execution.get("segment_case_limit"),
            ),
            name="next_segment_case_limit",
        ),
        "case_completion_target": optional_positive(
            execution.get("case_completion_target"),
            name="case_completion_target",
        ),
    }


class ExperimentRegistry:
    """Persist reusable recipes and concrete queue entries under configs/."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/experiments"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        rows = []
        for item in self._all(self.spec_root):
            try:
                normalized = self.normalize_spec(item)
            except (TypeError, ValueError):
                continue
            rows.append({**normalized, "registry_path": item["registry_path"]})
        return rows

    def instances(self) -> list[dict[str, Any]]:
        specs = {item["id"]: item for item in self.specs()}
        return sorted(
            [
                self._with_lifecycle_status(
                    item,
                    specs.get(str(item.get("experiment_spec_id") or "")),
                )
                for item in self._all(self.instance_root)
                if int(item.get("schema_version") or 0) == 3
                and item.get("experiment_spec_id")
                and item.get("benchmark_instance_id")
                and item.get("team_instance_id")
            ],
            key=lambda item: (
                int(item.get("queue", {}).get("priority", 100)),
                str(item.get("created_at_utc") or ""),
                item["id"],
            ),
        )

    @staticmethod
    def _all(root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []
        rows = []
        for path in sorted(root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict) or not _SAFE_ID.fullmatch(str(value.get("id") or "")):
                continue
            rows.append({**value, "registry_path": str(path)})
        return rows

    def default_spec(self) -> dict[str, Any]:
        from ..environment.service import readme_environment

        environment = readme_environment(self.repo_root)
        return {
            "schema_version": 3,
            "id": "eval-experiment",
            "benchmark": {
                "benchmark_spec_id": "gsm8k",
                "runnable_task": "gsm8k",
                "cases": 10,
                "start_index": 0,
                "case_selection": "head",
                "case_strata_field": None,
            },
            "team_spec_id": "reason",
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
                    "control_window_trials": 1,
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
            "environment": {
                "repo_root": str(self.repo_root),
                "python": environment["python"],
                "conda_sh": environment["conda_sh"],
                "conda_env": environment["conda_env"],
                "raw_root": str(self.repo_root / "data/benchmarks/raw"),
                "prepared_root": str(self.repo_root / "data/benchmarks/prepared"),
                "models_root": str(self.repo_root / "models"),
                "runs_root": str(self.repo_root / "runs/benchmarks"),
            },
        }

    def normalize_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("ExperimentSpec must be a JSON object")
        _reject_unknown_fields(value, _EXPERIMENT_SPEC_FIELDS, label="ExperimentSpec")
        spec = self.default_spec()
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(spec.get(key), dict):
                spec[key] = {**spec[key], **item}
            else:
                spec[key] = item
        if int(spec.get("schema_version") or 0) != 3:
            raise ValueError("ExperimentSpec requires schema_version 3")
        spec_id = str(spec.get("id") or "")
        self._validate_id(spec_id)
        benchmark = dict(spec.get("benchmark") or {})
        _reject_unknown_fields(
            benchmark,
            {
                "benchmark_spec_id",
                "runnable_task",
                "cases",
                "start_index",
                "case_selection",
                "case_strata_field",
                "scoring_profile",
            },
            label="ExperimentSpec benchmark",
        )
        benchmark_spec_id = str(benchmark.get("benchmark_spec_id") or "")
        runnable_task = str(benchmark.get("runnable_task") or "")
        self._validate_id(benchmark_spec_id)
        if not runnable_task:
            raise ValueError("ExperimentSpec requires benchmark.runnable_task")
        cases = benchmark.get("cases", 10)
        if isinstance(cases, str) and cases.strip().lower() == "all":
            cases = "all"
        else:
            try:
                cases = int(cases)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ExperimentSpec benchmark.cases must be positive or 'all'"
                ) from exc
        if cases != "all" and cases < 1:
            raise ValueError("ExperimentSpec benchmark.cases must be positive or 'all'")
        start_index = int(benchmark.get("start_index") or 0)
        if start_index < 0:
            raise ValueError("ExperimentSpec benchmark.start_index must be non-negative")
        case_selection = str(benchmark.get("case_selection") or "head")
        if case_selection not in {"head", "uniform", "stratified"}:
            raise ValueError(
                "ExperimentSpec benchmark.case_selection must be head, uniform or stratified"
            )
        team_spec_id = str(spec.get("team_spec_id") or "")
        self._validate_id(team_spec_id)
        normalized_benchmark = {
            "benchmark_spec_id": benchmark_spec_id,
            "runnable_task": runnable_task,
            "cases": cases if cases == "all" else int(cases),
            "start_index": start_index,
            "case_selection": case_selection,
            "case_strata_field": (
                str(benchmark["case_strata_field"]) if benchmark.get("case_strata_field") else None
            ),
        }
        if benchmark.get("scoring_profile"):
            normalized_benchmark["scoring_profile"] = str(benchmark["scoring_profile"])
        runtime = dict(spec.get("runtime") or {})
        allowed_runtime_fields = set(self.default_spec()["runtime"])
        unknown_runtime_fields = set(runtime) - allowed_runtime_fields
        if unknown_runtime_fields:
            raise ValueError(
                "ExperimentSpec runtime contains unsupported fields: "
                + ", ".join(sorted(unknown_runtime_fields))
            )
        try:
            runtime["trials_per_case"] = int(runtime.get("trials_per_case", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("ExperimentSpec runtime.trials_per_case must be positive") from exc
        if runtime["trials_per_case"] < 1:
            raise ValueError("ExperimentSpec runtime.trials_per_case must be positive")
        try:
            runtime["max_attempts_per_trial"] = int(runtime.get("max_attempts_per_trial", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ExperimentSpec runtime.max_attempts_per_trial must be positive"
            ) from exc
        if runtime["max_attempts_per_trial"] < 1:
            raise ValueError("ExperimentSpec runtime.max_attempts_per_trial must be positive")
        runtime["do_sample"] = bool(runtime.get("do_sample", False))
        for key in ("temperature", "top_p", "repetition_penalty"):
            try:
                runtime[key] = float(runtime[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"ExperimentSpec runtime.{key} must be numeric") from exc
        if runtime["temperature"] < 0:
            raise ValueError("ExperimentSpec runtime.temperature must be non-negative")
        if not 0.0 < runtime["top_p"] <= 1.0:
            raise ValueError("ExperimentSpec runtime.top_p must be greater than 0 and at most 1")
        if runtime["repetition_penalty"] <= 0:
            raise ValueError("ExperimentSpec runtime.repetition_penalty must be positive")
        if runtime.get("top_k") in (None, ""):
            runtime["top_k"] = None
        else:
            runtime["top_k"] = int(runtime["top_k"])
            if runtime["top_k"] < 1:
                raise ValueError("ExperimentSpec runtime.top_k must be positive or null")
        for key in ("min_p", "presence_penalty"):
            if runtime.get(key) in (None, ""):
                runtime[key] = None
            else:
                runtime[key] = float(runtime[key])
        if runtime["min_p"] is not None and not 0.0 <= runtime["min_p"] <= 1.0:
            raise ValueError("ExperimentSpec runtime.min_p must be between 0 and 1")
        if runtime["presence_penalty"] is not None and not (
            -2.0 <= runtime["presence_penalty"] <= 2.0
        ):
            raise ValueError("ExperimentSpec runtime.presence_penalty must be between -2 and 2")
        runtime["on_trial_error"] = str(runtime.get("on_trial_error") or "continue")
        if runtime["on_trial_error"] not in {"continue", "fail-fast"}:
            raise ValueError("ExperimentSpec runtime.on_trial_error must be continue or fail-fast")
        runtime["method"] = str(runtime.get("method") or "none")
        if runtime["method"] not in {"none", "nl_only", "latent_only", "both"}:
            raise ValueError(
                "ExperimentSpec runtime.method must be none, nl_only, latent_only or both"
            )
        for key, minimum in (
            ("max_new_tokens", 1),
            ("min_output_reserve_tokens", 0),
            ("min_thinking_reserve_tokens", 0),
            ("min_final_reserve_tokens", 0),
            ("safety_margin_tokens", 0),
            ("code_timeout", 1),
        ):
            try:
                runtime[key] = int(runtime[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"ExperimentSpec runtime.{key} must be an integer") from exc
            if runtime[key] < minimum:
                raise ValueError(f"ExperimentSpec runtime.{key} must be >= {minimum}")
        for key in ("max_input_tokens", "max_thinking_budget_tokens"):
            raw_value = runtime.get(key)
            if raw_value is None or raw_value == "":
                runtime[key] = None
                continue
            try:
                runtime[key] = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ExperimentSpec runtime.{key} must be non-negative or null"
                ) from exc
            if runtime[key] < 0:
                raise ValueError(f"ExperimentSpec runtime.{key} must be non-negative or null")
        runtime["seed"] = int(runtime.get("seed", 0))
        runtime["code_executor"] = str(runtime.get("code_executor") or "docker")
        if runtime["code_executor"] not in {"docker", "local"}:
            raise ValueError("ExperimentSpec runtime.code_executor must be docker or local")
        runtime["work_root"] = str(runtime.get("work_root") or "").strip()
        if not runtime["work_root"]:
            raise ValueError("ExperimentSpec runtime.work_root must be non-empty")
        for key in ("web_headless", "save_screenshots", "trace_model_calls"):
            runtime[key] = bool(runtime.get(key, False))
        runtime["docker_image"] = (
            str(runtime["docker_image"]).strip() if runtime.get("docker_image") else None
        )
        from ...runtime.execution.concurrency import normalize_concurrency_policy

        runtime["trial_concurrency"] = int(runtime.get("trial_concurrency", 1))
        runtime["concurrency_policy"] = normalize_concurrency_policy(
            runtime.get("concurrency_policy"),
            configured_concurrency=runtime["trial_concurrency"],
        )
        for key, default in (("max_rounds", 4), ("max_turns", 20)):
            try:
                runtime[key] = int(runtime.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ExperimentSpec runtime.{key} must be positive") from exc
            if runtime[key] < 1:
                raise ValueError(f"ExperimentSpec runtime.{key} must be positive")
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
                    "ExperimentSpec runtime.max_model_calls_per_case must be positive or null"
                ) from exc
            if runtime["max_model_calls_per_case"] < 1:
                raise ValueError(
                    "ExperimentSpec runtime.max_model_calls_per_case must be positive or null"
                )
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
                    "ExperimentSpec runtime.max_case_wall_time_s must be positive or null"
                ) from exc
            if runtime["max_case_wall_time_s"] <= 0:
                raise ValueError(
                    "ExperimentSpec runtime.max_case_wall_time_s must be positive or null"
                )
        observability = dict(spec.get("observability") or {})
        _reject_unknown_fields(
            observability,
            {
                "collect_vllm_metrics",
                "vllm_metrics_interval_s",
                "event_log_max_events_per_file",
                "event_log_max_mib_per_file",
            },
            label="ExperimentSpec observability",
        )
        observability["collect_vllm_metrics"] = bool(
            observability.get("collect_vllm_metrics", True)
        )
        try:
            observability["vllm_metrics_interval_s"] = float(
                observability.get("vllm_metrics_interval_s", 5.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ExperimentSpec observability.vllm_metrics_interval_s must be positive"
            ) from exc
        if observability["vllm_metrics_interval_s"] <= 0:
            raise ValueError(
                "ExperimentSpec observability.vllm_metrics_interval_s must be positive"
            )
        try:
            observability["event_log_max_events_per_file"] = int(
                observability.get("event_log_max_events_per_file", 10000)
            )
            observability["event_log_max_mib_per_file"] = float(
                observability.get("event_log_max_mib_per_file", 64.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ExperimentSpec EventLog segment limits must be positive") from exc
        if (
            observability["event_log_max_events_per_file"] <= 0
            or observability["event_log_max_mib_per_file"] <= 0
        ):
            raise ValueError("ExperimentSpec EventLog segment limits must be positive")
        evaluation = dict(spec.get("evaluation") or {})
        _reject_unknown_fields(
            evaluation,
            {"external_evaluator", "profile_id", "hle_judge"},
            label="ExperimentSpec evaluation",
        )
        evaluation["external_evaluator"] = str(evaluation.get("external_evaluator") or "auto")
        if evaluation["external_evaluator"] not in {"auto", "run", "skip"}:
            raise ValueError(
                "ExperimentSpec evaluation.external_evaluator must be auto, run or skip"
            )
        evaluation["profile_id"] = str(evaluation.get("profile_id") or "core")
        hle_judge = dict(evaluation.get("hle_judge") or {})
        _reject_unknown_fields(
            hle_judge,
            {
                "model",
                "base_url",
                "auth_mode",
                "api_key_env",
                "workers",
                "timeout_s",
                "max_tokens",
                "thinking_mode",
                "output_mode",
                "max_attempts",
            },
            label="ExperimentSpec evaluation.hle_judge",
        )
        if hle_judge:
            for key in ("model", "base_url"):
                hle_judge[key] = str(hle_judge.get(key) or "").strip()
                if not hle_judge[key]:
                    raise ValueError(f"ExperimentSpec evaluation.hle_judge.{key} must be non-empty")
            hle_judge["auth_mode"] = str(hle_judge.get("auth_mode") or "env")
            if hle_judge["auth_mode"] not in {"env", "none"}:
                raise ValueError(
                    "ExperimentSpec evaluation.hle_judge.auth_mode must be env or none"
                )
            hle_judge["api_key_env"] = str(hle_judge.get("api_key_env") or "OPENAI_API_KEY")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", hle_judge["api_key_env"]):
                raise ValueError(
                    "ExperimentSpec evaluation.hle_judge.api_key_env must be an "
                    "environment variable name"
                )
            hle_judge["workers"] = max(1, int(hle_judge.get("workers", 8)))
            hle_judge["timeout_s"] = max(1.0, float(hle_judge.get("timeout_s", 300.0)))
            hle_judge["max_tokens"] = max(1, int(hle_judge.get("max_tokens", 4096)))
            hle_judge["thinking_mode"] = str(
                hle_judge.get("thinking_mode") or "inherit"
            )
            if hle_judge["thinking_mode"] not in {"inherit", "enabled", "disabled"}:
                raise ValueError(
                    "ExperimentSpec evaluation.hle_judge.thinking_mode must be "
                    "inherit, enabled or disabled"
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
                    "ExperimentSpec evaluation.hle_judge.output_mode must be "
                    "official_schema or local_json_object"
                )
        evaluation["hle_judge"] = hle_judge
        network = dict(spec.get("network") or {})
        _reject_unknown_fields(
            network,
            {
                "mode",
                "proxy_url",
                "no_proxy",
                "targets",
                "docker_bridge_host",
                "container_proxy_port",
                "probe_url",
            },
            label="ExperimentSpec network",
        )
        network["mode"] = str(network.get("mode") or "benchmark_default")
        if network["mode"] not in {"benchmark_default", "direct", "proxy", "offline"}:
            raise ValueError(
                "ExperimentSpec network.mode must be benchmark_default, direct, proxy or offline"
            )
        network["proxy_url"] = str(network.get("proxy_url") or "").strip()
        if network["mode"] == "proxy" and not re.match(r"^https?://[^\s]+$", network["proxy_url"]):
            raise ValueError(
                "ExperimentSpec network.proxy_url must be an HTTP(S) URL in proxy mode"
            )
        targets = dict(network.get("targets") or {})
        allowed_targets = {"downloads", "web_surfer", "code_executor", "model_backend"}
        _reject_unknown_fields(targets, allowed_targets, label="ExperimentSpec network.targets")
        network["targets"] = {
            name: bool(targets.get(name, False)) for name in sorted(allowed_targets)
        }
        network["no_proxy"] = str(network.get("no_proxy") or "127.0.0.1,localhost,::1")
        network["docker_bridge_host"] = str(network.get("docker_bridge_host") or "172.17.0.1")
        network["container_proxy_port"] = int(network.get("container_proxy_port") or 17897)
        if not 1 <= network["container_proxy_port"] <= 65535:
            raise ValueError(
                "ExperimentSpec network.container_proxy_port must be between 1 and 65535"
            )
        network["probe_url"] = str(network.get("probe_url") or "").strip()
        if not network["probe_url"]:
            raise ValueError("ExperimentSpec network.probe_url must be non-empty")

        normalized_environment = dict(spec.get("environment") or {})
        _reject_unknown_fields(
            normalized_environment,
            {
                "repo_root",
                "python",
                "conda_sh",
                "conda_env",
                "raw_root",
                "prepared_root",
                "models_root",
                "runs_root",
            },
            label="ExperimentSpec environment",
        )
        for key in (
            "repo_root",
            "python",
            "raw_root",
            "prepared_root",
            "models_root",
            "runs_root",
        ):
            normalized_environment[key] = str(normalized_environment.get(key) or "").strip()
            if not normalized_environment[key]:
                raise ValueError(f"ExperimentSpec environment.{key} must be non-empty")
        for key in ("conda_sh", "conda_env"):
            normalized_environment[key] = str(normalized_environment.get(key) or "").strip()

        return {
            "schema_version": 3,
            "id": spec_id,
            "benchmark": normalized_benchmark,
            "team_spec_id": team_spec_id,
            "runtime": runtime,
            "observability": observability,
            "evaluation": evaluation,
            "network": network,
            "environment": normalized_environment,
            "notes": str(spec.get("notes") or ""),
        }

    def save_spec(self, spec_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        self._validate_id(spec_id)
        if spec.get("id") and str(spec["id"]) != spec_id:
            raise ValueError("ExperimentSpec id does not match URL")
        value = self.normalize_spec({**spec, "id": spec_id})
        value["updated_at_utc"] = _utc_now()
        path = self.spec_root / f"{spec_id}.json"
        _write_json(path, value)
        return {**value, "registry_path": str(path)}

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        raw = self._read(self.spec_root, spec_id)
        normalized = self.normalize_spec(raw)
        return {**normalized, "registry_path": raw["registry_path"]}

    def delete_spec(self, spec_id: str) -> dict[str, Any]:
        if any(item.get("experiment_spec_id") == spec_id for item in self.instances()):
            raise ValueError("ExperimentSpec is referenced by an ExperimentInstance")
        return self._delete(self.spec_root, spec_id)

    def create_instance(
        self,
        *,
        instance_id: str,
        spec_id: str,
        benchmark_instance_id: str,
        team_instance_id: str,
        run_dir: str | None = None,
        priority: int = 100,
        launcher: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_id(instance_id)
        if (self.instance_root / f"{instance_id}.json").exists():
            raise ValueError(f"ExperimentInstance {instance_id!r} already exists")
        self._validate_id(spec_id)
        spec = self.get_spec(spec_id)
        self._validate_id(benchmark_instance_id)
        self._validate_id(team_instance_id)
        normalized_launcher = normalize_launcher(launcher)
        value = bind_instance_to_spec(
            {
                "schema_version": 3,
                "id": instance_id,
                "benchmark_instance_id": benchmark_instance_id,
                "team_instance_id": team_instance_id,
                "launcher": normalized_launcher,
                "execution": normalize_instance_execution(execution),
                "status": "ready",
                "queue": {"priority": int(priority)},
                "updated_at_utc": _utc_now(),
                "launch_id": None,
                "launch_dir": None,
                "run_dir": str(run_dir or "") or None,
                "error": None,
            },
            spec=spec,
            prefix="experiment",
            creation_source="studio",
        )
        return self._save_instance(value)

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        value = self._read(self.instance_root, instance_id)
        try:
            spec = self.get_spec(str(value.get("experiment_spec_id") or ""))
        except (FileNotFoundError, ValueError):
            spec = None
        return self._with_lifecycle_status(value, spec)

    def update_instance(self, instance_id: str, **changes: Any) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        value.pop("registry_path", None)
        if "status" in changes and changes["status"] not in _STATUSES:
            raise ValueError(f"invalid ExperimentInstance status {changes['status']!r}")
        if "execution" in changes:
            changes["execution"] = normalize_instance_execution(changes["execution"])
        if "queue" in changes:
            queue = dict(changes["queue"] or {})
            try:
                queue["priority"] = int(queue.get("priority", 100))
            except (TypeError, ValueError) as exc:
                raise ValueError("ExperimentInstance queue.priority must be an integer") from exc
            changes["queue"] = queue
        value.update(changes, updated_at_utc=_utc_now())
        return self._save_instance(value)

    def update_controls(
        self,
        instance_id: str,
        *,
        priority: Any = None,
        next_segment_case_limit: Any = ...,
        case_completion_target: Any = ...,
    ) -> dict[str, Any]:
        """Update persistent scheduling controls without changing an active segment."""

        value = self.get_instance(instance_id)
        changes: dict[str, Any] = {}
        if priority is not None:
            queue = dict(value.get("queue") or {})
            queue["priority"] = int(priority)
            changes["queue"] = queue
        if next_segment_case_limit is not ... or case_completion_target is not ...:
            execution = normalize_instance_execution(value.get("execution"))
            if next_segment_case_limit is not ...:
                execution["next_segment_case_limit"] = next_segment_case_limit
            if case_completion_target is not ...:
                execution["case_completion_target"] = case_completion_target
            changes["execution"] = normalize_instance_execution(execution)
        if not changes:
            return value
        return self.update_instance(instance_id, **changes)

    def next_queued(self) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.instances()
                if item.get("status") == "queued" and item.get("configuration_status") == "current"
            ),
            None,
        )

    def enqueue(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        self._require_current(value)
        previous_status = str(value.get("status") or "")
        queue = dict(value.get("queue") or {})
        if previous_status == "running":
            if not value.get("drain_requested"):
                raise ValueError(
                    "a running ExperimentInstance can be queued only after graceful stop "
                    "has been requested"
                )
            queue.update(
                enabled=True,
                resume_from_status="paused",
            )
            return self.update_instance(
                instance_id,
                queue=queue,
                queued_at_utc=_utc_now(),
            )
        if previous_status not in {"ready", *_RESUMABLE_STATUSES}:
            raise ValueError(
                "only a ready, paused, stopped, or failed ExperimentInstance can be queued"
            )
        queue["enabled"] = True
        queue["resume_from_status"] = (
            previous_status if previous_status in _RESUMABLE_STATUSES else None
        )
        return self.update_instance(
            instance_id,
            status="queued",
            queue=queue,
            queued_at_utc=_utc_now(),
        )

    def dequeue(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        queue = dict(value.get("queue") or {})
        if value.get("status") == "running" and queue.get("enabled"):
            queue["enabled"] = False
            queue.pop("resume_from_status", None)
            return self.update_instance(
                instance_id,
                queue=queue,
                queued_at_utc=None,
            )
        if value.get("status") != "queued":
            raise ValueError("only a queued ExperimentInstance can be removed from the queue")
        queue["enabled"] = False
        restored = str(queue.pop("resume_from_status", "") or "ready")
        if restored not in {"ready", *_RESUMABLE_STATUSES}:
            restored = "ready"
        return self.update_instance(
            instance_id,
            status=restored,
            queue=queue,
            queued_at_utc=None,
        )

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        value = self.get_instance(instance_id)
        if value.get("status") in {"queued", "running"}:
            raise ValueError("a queued or running ExperimentInstance cannot be deleted")
        return self._delete(self.instance_root, instance_id)

    def _save_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(value.get("id") or "")
        self._validate_id(instance_id)
        path = self.instance_root / f"{instance_id}.json"
        derived = {
            "registry_path",
            "experiment_spec",
            "configuration_status",
            "validation_error",
        }
        clean = {key: item for key, item in value.items() if key not in derived}
        _write_json(path, clean)
        return {**clean, "registry_path": str(path)}

    @staticmethod
    def _with_lifecycle_status(
        value: dict[str, Any], spec: dict[str, Any] | None
    ) -> dict[str, Any]:
        error = lifecycle_error(value, spec=spec, prefix="experiment")
        return {
            **value,
            "experiment_spec": spec,
            "configuration_status": "invalid" if error else "current",
            "validation_error": error,
        }

    @staticmethod
    def _require_current(value: dict[str, Any]) -> None:
        if value.get("configuration_status") == "invalid":
            raise ValueError(
                f"ExperimentInstance {value.get('id')!r} is invalid: "
                f"{value.get('validation_error')}"
            )

    def _read(self, root: Path, value_id: str) -> dict[str, Any]:
        self._validate_id(value_id)
        path = root / f"{value_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid registry document {path}")
        return {**value, "registry_path": str(path)}

    def _delete(self, root: Path, value_id: str) -> dict[str, Any]:
        value = self._read(root, value_id)
        path = Path(value["registry_path"])
        path.unlink()
        return {"id": value_id, "deleted": True, "path": str(path)}

    @staticmethod
    def _validate_id(value_id: str) -> None:
        if not _SAFE_ID.fullmatch(str(value_id or "")):
            raise ValueError(
                "registry id must be 1-64 characters and contain only letters, "
                "numbers, '.', '_' and '-'"
            )
