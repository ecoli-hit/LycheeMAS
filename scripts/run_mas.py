"""真实 MAS 推理驱动（新框架 lychee_mas 版）—— 把 backend + CDM 记忆 + 固定通道路由 + RoutingContext
+ TeamSpec/RoleProfile + AutoGenRuntime 接成端到端可跑的实验，逐样本生成 prediction 并落盘。

为什么需要本文件：`Orchestrator` 只接 runtime/topology/aggregator，不组装记忆层；
`scripts/run_experiment.py` 是离线自检入口（argparse，无 YAML/记忆）。要在 AIME 上跑「带 CDM
记忆通道」的真实实验，必须自己把 backend（HFBackend，模型只加载一次）、记忆管理器（cdm 双通道）、
路由器（fixed 固定通道，对应 none/nl_only/latent_only/both 四种消融）、RoutingContext（跨 agent
共享状态）、AutoGenRuntime 拼起来。逻辑对齐旧原型 src-bak/LycheeMAS/run_mas.py。

跑法（需 extras：autogen + torch + transformers；数据走 benchmark raw/prepared roots）：
  LYCHEE_BENCHMARK_RAW_ROOT=data/benchmarks/raw \
  LYCHEE_BENCHMARK_PREPARED_ROOT=data/benchmarks/prepared \
  CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_mas.py --config configs/aime_both.yaml

命令行同名项覆盖 YAML（如 --n 3 / --method latent_only / --runs-root <dir>）。
重依赖（torch/autogen）全部惰性导入在 run_one 内部，import 本模块不触发它们（黄金法则 2）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# 让 `python scripts/run_mas.py` 直接可用（把 src/ 加进路径）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
for _pkg in ("autogen-core", "autogen-agentchat", "autogen-ext"):
    _pkg_src = os.path.join(_ROOT, "src", "autogen", "python", "packages", _pkg, "src")
    if os.path.isdir(_pkg_src) and _pkg_src not in sys.path:
        sys.path.append(_pkg_src)

# router.method name -> fixed_channel_router channel value.
FIXED = {"none": "none", "nl_only": "nl", "latent_only": "latent", "both": "both"}


def _profile_group_chat(profile: str) -> dict:
    """Resolve the explicit GroupChat recommended by one role profile."""
    from lychee_mas.layers.construct.templates import ROLE_PROFILE_META

    value = ROLE_PROFILE_META.get(profile, {}).get("recommended_group_chat")
    if not isinstance(value, dict) or not value.get("type"):
        raise SystemExit(f"role profile {profile!r} does not define recommended_group_chat")
    return dict(value)


def _team_chat_mode(profile: str) -> str:
    """Return the concrete AutoGen GroupChat used by a role profile."""
    group_chat = _profile_group_chat(profile)
    labels = {
        "round_robin": "RoundRobinGroupChat",
        "selector": "SelectorGroupChat",
        "magentic_one": "MagenticOneGroupChat",
    }
    label = labels[str(group_chat["type"])]
    if group_chat.get("selector_func_factory"):
        label += f" (selector_func={group_chat['selector_func_factory']})"
    if group_chat.get("candidate_func_factory"):
        label += f" (candidate_func={group_chat['candidate_func_factory']})"
    return label


def _team_details(profile: str) -> dict:
    """Resolve one profile without loading a model or benchmark dataset."""
    from lychee_mas.layers.construct.templates import ROLE_PROFILES

    if profile not in ROLE_PROFILES:
        choices = ", ".join(sorted(ROLE_PROFILES))
        raise SystemExit(f"unknown role profile {profile!r}; choices: {choices}")
    roles = [{"name": role.name, "agent_type": role.agent_type} for role in ROLE_PROFILES[profile]]
    group_chat = _profile_group_chat(profile)
    runtime_roles = list(roles)
    if group_chat["type"] == "magentic_one":
        runtime_roles = [{"name": "Orchestrator", "agent_type": "orchestrator"}, *roles]
    elif group_chat["type"] == "selector":
        runtime_roles = [
            {
                "name": "Selector",
                "agent_type": "selector",
                "control_mode": (
                    "selector_func" if group_chat.get("selector_func_factory") else "model"
                ),
            },
            *roles,
        ]
    return {
        "profile": profile,
        "roles": roles,
        "group_chat": group_chat,
        "group_chat_type": group_chat["type"],
        "chat_mode": _team_chat_mode(profile),
        "runtime_roles": runtime_roles,
        "context_visibility": "shared",
    }


def _print_team_structure(*, task: str | None = None, team_override: str | None = None) -> None:
    """Print runnable-task -> role profile -> concrete AutoGen GroupChat."""
    from lychee_mas.eval.benchmark_requirements import requirements_for_task
    from lychee_mas.eval.benchmarks import LOADERS
    from lychee_mas.layers.construct.templates import ROLE_PROFILES

    if task and task not in LOADERS:
        choices = ", ".join(LOADERS)
        raise SystemExit(f"unknown runnable task {task!r}; choices: {choices}")
    tasks = [task] if task else list(LOADERS)
    rows = []
    for task_name in tasks:
        profile = team_override or requirements_for_task(task_name)["suggested_role_profile"]
        if task_name.startswith("agent_collab_") and team_override is None:
            rows.append(
                (
                    task_name,
                    f"{profile} (fallback)",
                    "SelectorGroupChat (selector_func=topology_selector)",
                    "<from record topology.agents>",
                )
            )
            continue
        details = _team_details(profile)
        participants = ", ".join(
            f"{role['name']}[{role['agent_type']}]" for role in details["runtime_roles"]
        )
        rows.append((task_name, profile, details["chat_mode"], participants))

    headers = (
        "RUNNABLE TASK",
        "SUGGESTED ROLE PROFILE",
        "AUTOGEN GROUP CHAT",
        "RUNTIME ROLES",
    )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    used_profiles = {
        team_override or requirements_for_task(task_name)["suggested_role_profile"]
        for task_name in tasks
    }
    unused = [name for name in ROLE_PROFILES if name not in used_profiles]
    if not task and not team_override and unused:
        print("\nAvailable role profiles not selected as a runnable task default:")
        for profile in unused:
            details = _team_details(profile)
            roles = ", ".join(
                f"{role['name']}[{role['agent_type']}]" for role in details["runtime_roles"]
            )
            print(f"  {profile}: {details['chat_mode']}; {roles}")


class TeeStream:
    """Mirror stdout/stderr to the terminal and a run-local console log."""

    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, text):
        self.primary.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _start_console_log(out_dir: str, *, append: bool = False) -> str:
    path = os.path.join(out_dir, "console_log.txt")
    log_file = open(path, "a" if append else "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    if append:
        print("\n" + "=" * 88, flush=True)
        print(f"[resume] appending to existing console_log={path}", flush=True)
    print(f"[log] console_log={path}", flush=True)
    return path


def _load_yaml(path: str) -> dict:
    import yaml  # 惰性导入

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict, dotted: str, default=None):
    """按「a.b.c」分层取值；任一层缺失返回 default。"""
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _apply_benchmark_contract(cfg: dict, args) -> dict:
    """Resolve task-owned runtime and network defaults for every CLI entry path.

    A benchmark-specific YAML may override its BenchmarkSpec defaults. When ``--task``
    changes a generic YAML to a different benchmark, however, carrying the generic
    sandbox/network settings forward is incorrect. Explicit CLI arguments continue to
    win later in ``run_one``.
    """

    from lychee_mas.eval.studio.benchmarks import BenchmarkRegistry

    resolved = json.loads(json.dumps(cfg))
    configured_task = str(_get(resolved, "run.task", "") or "")
    task = str(args.task or configured_task).strip()
    if not task:
        return resolved
    try:
        benchmark_spec, benchmark_instance = BenchmarkRegistry(Path(_ROOT)).resolve_task(task)
    except ValueError:
        return resolved

    run_cfg = resolved.setdefault("run", {})
    run_cfg["task"] = task
    run_cfg["benchmark_spec_id"] = benchmark_spec["id"]
    if benchmark_instance is not None:
        run_cfg["benchmark_instance_id"] = benchmark_instance["id"]

    task_changed = bool(args.task and configured_task and task != configured_task)
    runtime_cfg = resolved.setdefault("runtime", {})
    for key, value in dict(benchmark_spec.get("runtime_defaults") or {}).items():
        if task_changed or runtime_cfg.get(key) is None:
            runtime_cfg[key] = value

    network_defaults = dict(benchmark_spec.get("network_defaults") or {})
    network_cfg = dict(resolved.get("network") or {})
    if task_changed or not network_cfg or network_cfg.get("mode") == "benchmark_default":
        network_cfg = {**network_cfg, **network_defaults}
    else:
        for key, value in network_defaults.items():
            if network_cfg.get(key) is None:
                network_cfg[key] = value
    resolved["network"] = network_cfg

    if network_cfg.get("mode") == "proxy":
        relay_port = int(network_cfg.get("container_proxy_port") or 17897)
        network_cfg.setdefault("container_proxy_url", f"http://host.docker.internal:{relay_port}")
        network_cfg.setdefault("proxy_relay_upstream_url", network_cfg.get("proxy_url"))
    return resolved


def _resolve_prefix_length(cfg: dict, cli_value: int | None) -> int:
    """Resolve latent-prefix length from CLI, current config, then legacy config."""

    if cli_value is not None:
        return int(cli_value)
    return int(_get(cfg, "router.P", _get(cfg, "memory.P", 16)))


def _as_bool(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_positive_limit(value, *, name: str) -> int | None:
    """Parse a positive integer limit; None/unlimited means no limit."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "unlimited",
        "unbounded",
    }:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{name} must be a positive integer or 'unlimited'") from exc
    if parsed < 1:
        raise SystemExit(f"{name} must be >= 1 or 'unlimited'")
    return parsed


def _decision_number(decision: dict, *names: str, default=0):
    for name in names:
        if name in decision and decision[name] is not None:
            return decision[name]
    return default


def _normalize_model_call(decision: dict) -> dict:
    input_positions = int(_decision_number(decision, "input_positions", "prompt_pos", default=0))
    latent_positions = int(
        _decision_number(decision, "latent_prefix_positions", "prefix_len", default=0)
    )
    text_tokens = int(
        _decision_number(
            decision, "text_input_tokens", default=max(0, input_positions - latent_positions)
        )
    )
    output_tokens = int(_decision_number(decision, "output_tokens", "gen_tokens", default=0))
    latency = float(
        _decision_number(decision, "model_generation_latency_s", "latency_s", default=0.0)
    )
    normalized = dict(decision)
    normalized.update(
        {
            "input_total_positions": input_positions,
            "input_text_tokens": text_tokens,
            "input_latent_positions": latent_positions,
            "output_text_tokens": output_tokens,
            "model_latency_s": round(latency, 3),
            "input_positions": input_positions,
            "text_input_tokens": text_tokens,
            "latent_prefix_positions": latent_positions,
            "output_tokens": output_tokens,
            "model_generation_latency_s": round(latency, 3),
        }
    )
    return normalized


_MODEL_CALL_DURATION_FIELDS = (
    "client_rate_limiter_wait_s",
    "client_http_request_latency_s",
    "client_response_postprocess_latency_s",
    "client_model_call_wall_time_s",
    "provider_request_queue_latency_s",
    "provider_scheduled_to_first_token_s",
    "provider_generation_latency_s",
    "provider_mean_inter_token_latency_s",
)


def _optional_call_values(model_calls: list[dict], field: str) -> list[float]:
    return [float(call[field]) for call in model_calls if call.get(field) is not None]


def _case_id(item: dict, index: int) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("safe_task_id") or index)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"status": "error", "error_type": "InvalidJSONL", "raw_line": line})
    return records


def _read_json(path: str) -> dict:
    """Read one JSON object without making resume depend on a pristine status file."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_status_elapsed_seconds(status: dict) -> float:
    """Return elapsed wall time across every completed and current resume segment."""
    segment_started = status.get("current_segment_started_at_unix_s")
    accumulated = status.get("accumulated_run_elapsed_before_segment_s")
    segment_finished = status.get("finished_at_unix_s") or status.get("updated_at_unix_s")
    if accumulated is not None and segment_started is not None and segment_finished is not None:
        return max(0.0, float(accumulated)) + max(
            0.0, float(segment_finished) - float(segment_started)
        )
    cumulative = status.get("cumulative_run_elapsed_s")
    if cumulative is not None:
        return max(0.0, float(cumulative))
    started = status.get("started_at_unix_s")
    if started is not None and segment_finished is not None:
        return max(0.0, float(segment_finished) - float(started))
    return 0.0


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_write_json(path: str, value: dict) -> None:
    """Replace one JSON file atomically so interrupted runs never leave partial state."""
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(temporary, path)


def _backup_file(path: str) -> str | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    backup = f"{path}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, backup)
    return backup


def _prepare_predictions_for_run(
    predictions_path: str,
    data: list[dict],
    *,
    start_index: int,
    resume: bool,
) -> tuple[dict[str, set[int]], dict]:
    """Prepare predictions.jsonl and return completed samples per case.

    Returns ``{case_id: {k_index, ...}}``; a (case, sample) is skipped on resume
    only if a successful prediction already exists for that exact k_index. Error
    records are backed up and dropped so those samples are retried cleanly.
    K=1 runs use k_index=0, so this stays backward compatible with old files.
    """
    if not resume:
        open(predictions_path, "w", encoding="utf-8").close()
        return {}, {
            "enabled": False,
            "existing_predictions": 0,
            "kept_success_predictions": 0,
            "dropped_predictions": 0,
            "backup_path": None,
        }

    existing = _read_jsonl(predictions_path)
    valid_case_ids = {_case_id(item, start_index + offset) for offset, item in enumerate(data)}
    kept: list[dict] = []
    completed: dict[str, set[int]] = {}
    dropped: list[dict] = []
    for record in existing:
        case_id = str(record.get("case_id") or "")
        status = str(record.get("status") or "ok")
        k_index = int(record.get("k_index", 0) or 0)
        if case_id not in valid_case_ids:
            dropped.append(record)
            continue
        if status == "error":
            dropped.append(record)
            continue
        if k_index in completed.get(case_id, set()):
            dropped.append(record)
            continue
        kept.append(record)
        completed.setdefault(case_id, set()).add(k_index)

    kept.sort(
        key=lambda r: (int(r.get("sample_index", 10**18) or 10**18), int(r.get("k_index", 0) or 0))
    )
    backup_path = _backup_file(predictions_path) if existing and dropped else None
    _write_jsonl(predictions_path, kept)
    return completed, {
        "enabled": True,
        "existing_predictions": len(existing),
        "kept_success_predictions": len(kept),
        "dropped_predictions": len(dropped),
        "backup_path": backup_path,
        "skipped_sample_count": sum(len(ks) for ks in completed.values()),
    }


def _print_resolved_config(
    snapshot: dict,
    *,
    out_dir: str,
    predictions_path: str,
    spans_path: str,
    group_chat_path: str,
    resume_info: dict,
) -> None:
    resolved = dict(snapshot.get("resolved") or {})
    resolved["out_dir"] = out_dir
    resolved["predictions_path"] = predictions_path
    resolved["spans_path"] = spans_path
    resolved["group_chat_path"] = group_chat_path
    resolved["resume"] = resume_info
    print("[config] resolved parameters:", flush=True)
    print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _model_tag(
    args,
    cfg: dict,
    *,
    backend_provider: str | None = None,
    api_model: str | None = None,
    model_path: str | None = None,
    suffix: str = "",
) -> str:
    """Resolve a truthful model tag for result directories."""
    if args.model_tag:
        return args.model_tag
    if args.api_model and api_model:
        return f"{api_model}{suffix}"
    if args.model_path and model_path:
        return os.path.basename(os.path.normpath(model_path))
    configured = _get(cfg, "backend.model_tag")
    if configured:
        return configured
    if backend_provider == "api" and api_model:
        return f"{api_model}{suffix}"
    if model_path:
        return os.path.basename(os.path.normpath(model_path))
    return api_model or "model"


def _require_local_docker_image(image: str) -> dict | None:
    """Fail early when a project-owned sandbox is absent, stale, or incomplete."""
    if not image.endswith(":local"):
        return None
    from lychee_mas.runtime.docker_sandbox import (
        SandboxVerificationError,
        verify_local_sandbox,
    )

    try:
        report = verify_local_sandbox(image, repo_root=_ROOT)
    except SandboxVerificationError as exc:
        raise SystemExit(
            f"Docker sandbox {image!r} is not ready: {exc}. "
            "Rebuild and verify it with: python scripts/prepare_benchmarks.py "
            "--tasks human_eval,gaia --docker-images always"
        ) from exc
    print(
        f"[sandbox] image={image} profile={report.get('profile')} "
        f"fingerprint={str(report.get('fingerprint') or '')[:12]} verified=true",
        flush=True,
    )
    return report


def _usage_summary(model_calls: list[dict]) -> dict:
    input_positions = sum(call["input_positions"] for call in model_calls)
    text_tokens = sum(call["text_input_tokens"] for call in model_calls)
    latent_positions = sum(call["latent_prefix_positions"] for call in model_calls)
    output_tokens = sum(call["output_tokens"] for call in model_calls)
    latency = round(sum(call["model_generation_latency_s"] for call in model_calls), 3)
    timing_summary: dict[str, object] = {}
    for field in _MODEL_CALL_DURATION_FIELDS:
        values = _optional_call_values(model_calls, field)
        timing_summary[f"{field}_available_calls"] = len(values)
        timing_summary[f"{field}_total"] = round(sum(values), 6) if values else None
        timing_summary[f"{field}_mean"] = round(sum(values) / len(values), 6) if values else None
    provider_tps = _optional_call_values(model_calls, "provider_output_tokens_per_second")
    timing_summary["provider_output_tokens_per_second_available_calls"] = len(provider_tps)
    timing_summary["provider_output_tokens_per_second_mean"] = (
        round(sum(provider_tps) / len(provider_tps), 6) if provider_tps else None
    )
    timing_summary["provider_request_metrics_available_calls"] = sum(
        bool(call.get("provider_request_metrics_available")) for call in model_calls
    )
    routing_trace = [
        {
            key: call.get(key)
            for key in (
                "role",
                "turn_index",
                "sender_role",
                "memory_channel",
                "routing_reason",
                "nl_strategy",
                "latent_strategy",
            )
        }
        for call in model_calls
    ]
    return {
        "num_model_calls": len(model_calls),
        "sum_input_total_positions": input_positions,
        "sum_input_text_tokens": text_tokens,
        "sum_input_latent_positions": latent_positions,
        "sum_output_text_tokens": output_tokens,
        "sum_model_latency_s": latency,
        "model_call_count": len(model_calls),
        "input_positions_total": input_positions,
        "text_input_tokens_total": text_tokens,
        "latent_prefix_positions_total": latent_positions,
        "output_tokens_total": output_tokens,
        "model_generation_latency_s_total": latency,
        **timing_summary,
        "model_calls": model_calls,
        "routing_trace": routing_trace,
        "cost_prompt_pos": input_positions,
        "gen_tokens": output_tokens,
        "latency_s": latency,
    }


async def _execute_case_sample(
    *,
    case_order: int,
    num_cases: int,
    item: dict,
    sample_index: int,
    k_index: int,
    samples_per_case: int,
    task: str,
    kind: str,
    benchmark_id: str,
    method: str,
    profile: str,
    max_rounds: int,
    max_case_retries: int,
    memory,
    ctx,
    runtime,
    base_graph=None,
) -> dict:
    """Execute one isolated ``(case_id, k_index)`` and return a prediction record."""
    from lychee_mas.core.types import TaskQuery
    from lychee_mas.eval.benchmarks.topology import graph_from_benchmark_record
    from lychee_mas.runtime.seeding import (
        PREDICTION_SEED_DERIVATION,
        derive_prediction_seed,
    )
    from lychee_mas.runtime.spans import exception_record

    case_id = _case_id(item, sample_index)
    k_tag = f" k={k_index + 1}/{samples_per_case}" if samples_per_case > 1 else ""
    sample_extra = (
        {"k_index": k_index, "num_samples": samples_per_case} if samples_per_case > 1 else {}
    )
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    record_topology = item.get("topology") or metadata.get("topology") or {}
    has_record_agents = bool(
        isinstance(record_topology, dict) and record_topology.get("agents")
    ) or bool(item.get("agent_specs"))
    if base_graph is not None and not has_record_agents:
        case_graph, dynamic_topology = base_graph, bool(base_graph.meta.get("dynamic_topology"))
    else:
        case_graph, dynamic_topology = graph_from_benchmark_record(
            item,
            fallback_profile=profile,
            rounds=max_rounds,
        )
    runtime_query_id = case_id if samples_per_case == 1 else f"{case_id}__k{k_index}"
    query = TaskQuery(
        question=item["question"],
        context=item.get("context"),
        gold=None,
        id=runtime_query_id,
        meta={**metadata, "kind": kind, "dynamic_topology": dynamic_topology},
    )
    print(f"  [{case_order}/{num_cases}{k_tag}] start case_id={case_id}", flush=True)
    case_t0 = time.time()
    ctx.set_case(case_id, sample_index)
    ctx.current_k_index = k_index
    ctx.current_attempt = None
    ctx.set_case_span(None)
    base_seed = int(getattr(ctx, "base_seed", 0))
    prediction_seed = derive_prediction_seed(
        base_seed=base_seed,
        benchmark_id=benchmark_id,
        task=task,
        case_id=case_id,
        k_index=k_index,
    )
    ctx.prediction_seed = prediction_seed
    ctx.seed_derivation = PREDICTION_SEED_DERIVATION
    # Every model call and transport retry in one prediction uses the same seed.
    # The prompt still changes between agent turns; only the random stream identity is stable.
    ctx.generation_seed = prediction_seed
    case_span_id = ctx.log_span(
        "case_start",
        case_order=case_order,
        num_cases=num_cases,
        k_index=k_index,
        question=item["question"],
        has_context=bool(item.get("context")),
        scorer_kind=kind,
        benchmark_id=benchmark_id,
        dynamic_topology=dynamic_topology,
        topology_type=case_graph.meta.get("topology_type"),
        topology_roles=[node.name for node in case_graph.nodes],
        topology_edges=case_graph.edges,
        speaking_order=case_graph.meta.get("speaking_order"),
        base_seed=base_seed,
        prediction_seed=prediction_seed,
        seed_derivation=PREDICTION_SEED_DERIVATION,
    )
    ctx.set_case_span(case_span_id)

    trajectory = None
    error = None
    last_exception = None
    retry_count = 0
    for attempt in range(max_case_retries + 1):
        ctx.reset(preserve_model_call_budget=attempt > 0)
        ctx.set_case(case_id, sample_index)
        ctx.current_k_index = k_index
        ctx.current_attempt = attempt + 1
        ctx.log_span(
            "case_attempt_start",
            parent_span_id=case_span_id,
            attempt=attempt + 1,
            max_attempts=max_case_retries + 1,
            k_index=k_index,
        )
        try:
            if item.get("context") and hasattr(memory, "seed"):
                memory.seed(item["context"])
            trajectory = await runtime.run(case_graph, query)
            break
        except Exception as exc:
            last_exception = exc
            error = exception_record(exc)
            will_retry = attempt < max_case_retries
            ctx.log_span(
                "case_attempt_error",
                parent_span_id=case_span_id,
                attempt=attempt + 1,
                max_attempts=max_case_retries + 1,
                will_retry=will_retry,
                k_index=k_index,
                **error,
            )
            if will_retry:
                retry_count += 1
                print(
                    f"  [{case_order}/{num_cases}{k_tag}] retry "
                    f"attempt={attempt + 2}/{max_case_retries + 1} "
                    f"after {error['error_type']}: {error['error_message']}",
                    flush=True,
                )

    if trajectory is None:
        assert error is not None and last_exception is not None
        model_calls = [_normalize_model_call(value) for value in ctx.decisions]
        usage = _usage_summary(model_calls)
        record = {
            "case_id": case_id,
            "sample_index": sample_index,
            "base_seed": base_seed,
            "prediction_seed": prediction_seed,
            "seed_derivation": PREDICTION_SEED_DERIVATION,
            **sample_extra,
            "task": task,
            "method": method,
            "scorer_kind": kind,
            "benchmark_id": benchmark_id,
            "question": item["question"][:500],
            "final_answer": "",
            "status": "error",
            "error_type": error["error_type"],
            "error_message": error["error_message"],
            "traceback": error["traceback"],
            **usage,
            "num_messages": 0,
            "case_wall_time_s": round(time.time() - case_t0, 3),
            "num_tool_calls": 0,
            "num_tool_errors": 0,
            "num_tool_requests": 0,
            "num_tool_executions": 0,
            "num_tool_agent_errors": 0,
            "message_count": 0,
            "tool_call_count": 0,
            "tool_error_count": 0,
            "tool_request_count": 0,
            "tool_execution_count": 0,
            "tool_agent_error_count": 0,
            "tool_calls": [],
            "tool_requests": [],
            "tool_executions": [],
            "tool_agent_errors": [],
            "n_messages": 0,
        }
        ctx.log_span(
            "case_error",
            parent_span_id=case_span_id,
            case_order=case_order,
            num_cases=num_cases,
            k_index=k_index,
            partial_model_call_count=len(model_calls),
            case_wall_time_s=record["case_wall_time_s"],
            **error,
        )
        print(
            f"  [{case_order}/{num_cases}{k_tag}] error case_id={case_id} "
            f"{error['error_type']}: {error['error_message']}",
            flush=True,
        )
        return {
            "record": record,
            "retry_count": retry_count,
            "exception": last_exception,
        }

    prediction = trajectory.final_answer.content if trajectory.final_answer else ""
    model_calls = [_normalize_model_call(value) for value in trajectory.meta.get("decisions", [])]
    usage = _usage_summary(model_calls)
    tool_calls = list(trajectory.meta.get("tool_calls", []))
    wall_time = float(trajectory.meta.get("case_wall_time_s", 0.0) or 0.0)
    tool_call_count = int(trajectory.meta.get("tool_call_count", len(tool_calls)) or 0)
    tool_error_count = int(trajectory.meta.get("tool_error_count", 0) or 0)
    tool_request_count = int(trajectory.meta.get("tool_request_count", 0) or 0)
    tool_execution_count = int(trajectory.meta.get("tool_execution_count", 0) or 0)
    tool_agent_error_count = int(trajectory.meta.get("tool_agent_error_count", 0) or 0)
    message_count = len(trajectory.messages)
    record = {
        "case_id": case_id,
        "sample_index": sample_index,
        "base_seed": base_seed,
        "prediction_seed": prediction_seed,
        "seed_derivation": PREDICTION_SEED_DERIVATION,
        **sample_extra,
        "task": task,
        "method": method,
        "scorer_kind": kind,
        "benchmark_id": benchmark_id,
        "question": item["question"][:500],
        "final_answer": prediction,
        "stop_reason": trajectory.meta.get("stop_reason"),
        "max_model_calls_per_case": trajectory.meta.get("max_model_calls_per_case"),
        "model_calls_started": trajectory.meta.get("model_calls_started", len(model_calls)),
        "model_call_budget_exhausted": bool(
            trajectory.meta.get("model_call_budget_exhausted", False)
        ),
        **usage,
        "num_messages": message_count,
        "case_wall_time_s": round(wall_time or (time.time() - case_t0), 3),
        "num_tool_calls": tool_call_count,
        "num_tool_errors": tool_error_count,
        "num_tool_requests": tool_request_count,
        "num_tool_executions": tool_execution_count,
        "num_tool_agent_errors": tool_agent_error_count,
        "message_count": message_count,
        "tool_call_count": tool_call_count,
        "tool_error_count": tool_error_count,
        "tool_request_count": tool_request_count,
        "tool_execution_count": tool_execution_count,
        "tool_agent_error_count": tool_agent_error_count,
        "workspace": trajectory.meta.get("workspace"),
        "workspace_info": trajectory.meta.get("workspace_info", {}),
        "copied_files": trajectory.meta.get("copied_files", []),
        "dynamic_topology": dynamic_topology,
        "topology": {
            "type": case_graph.meta.get("topology_type"),
            "roles": [node.name for node in case_graph.nodes],
            "edges": case_graph.edges,
            "speaking_order": case_graph.meta.get("speaking_order"),
        }
        if dynamic_topology
        else None,
        "tool_calls": tool_calls,
        "tool_requests": list(trajectory.meta.get("tool_requests", [])),
        "tool_executions": list(trajectory.meta.get("tool_executions", [])),
        "tool_agent_errors": list(trajectory.meta.get("tool_agent_errors", [])),
        "n_messages": message_count,
    }
    ctx.log_span(
        "case_end",
        parent_span_id=case_span_id,
        case_order=case_order,
        num_cases=num_cases,
        k_index=k_index,
        status="ok",
        final_answer=prediction,
        stop_reason=trajectory.meta.get("stop_reason"),
        max_model_calls_per_case=trajectory.meta.get("max_model_calls_per_case"),
        model_calls_started=trajectory.meta.get("model_calls_started", len(model_calls)),
        model_call_budget_exhausted=bool(trajectory.meta.get("model_call_budget_exhausted", False)),
        num_model_calls=len(model_calls),
        num_messages=message_count,
        num_tool_calls=tool_call_count,
        num_tool_errors=tool_error_count,
        num_tool_requests=tool_request_count,
        num_tool_executions=tool_execution_count,
        num_tool_agent_errors=tool_agent_error_count,
        case_wall_time_s=record["case_wall_time_s"],
    )
    print(
        f"  [{case_order}/{num_cases}{k_tag}] msgs={message_count} "
        f"input_pos={usage['input_positions_total']} ans={prediction[:60]!r}",
        flush=True,
    )
    return {"record": record, "retry_count": retry_count, "exception": None}


async def run_one(cfg: dict, args) -> dict:
    from lychee_mas.eval import metrics as M
    from lychee_mas.eval.benchmarks import get_benchmark
    from lychee_mas.eval.benchmarks.common import runs_root as default_runs_root
    from lychee_mas.layers.construct.templates import RoleProfileTeamBuilder
    from lychee_mas.memory.context import RoutingContext
    from lychee_mas.memory.managers.DualChannelMemory import DualChannelMemoryManager
    from lychee_mas.memory.routing.static import fixed_channel_router
    from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime, _effective_max_turns
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend
    from lychee_mas.runtime.seeding import (
        PREDICTION_SEED_DERIVATION,
        derive_prediction_seed,
    )
    from lychee_mas.runtime.spans import JsonlGroupChatLogger, JsonlSpanLogger
    from lychee_mas.runtime.vllm_metrics import (
        VLLMMetricsMonitor,
        metrics_url,
        targets_from_deployments,
    )

    runtime_name = args.runtime or _get(cfg, "runtime.name", "autogen")
    if runtime_name != "autogen":
        raise SystemExit(f"unknown runtime.name {runtime_name!r}; use autogen")

    network_mode = str(_get(cfg, "network.mode", "direct") or "direct")
    network_proxy_url = str(_get(cfg, "network.proxy_url", "") or "").strip()
    network_targets = dict(_get(cfg, "network.targets", {}) or {})
    model_proxy_url = (
        network_proxy_url
        if network_mode == "proxy" and _as_bool(network_targets.get("model_backend"), default=False)
        else None
    )

    # ---- 解析参数（命令行 > YAML）----
    deployment_config_path = args.deployment_config or _get(cfg, "runtime.deployment_config")
    deployment_document = None
    if deployment_config_path:
        with open(deployment_config_path, encoding="utf-8") as handle:
            deployment_document = json.load(handle)
        if model_proxy_url:
            for deployment in deployment_document.get("deployments", []):
                if deployment.get("kind") in {"api", "vllm"}:
                    deployment["proxy_url"] = model_proxy_url
        deployment_by_id = {
            str(item["id"]): item for item in deployment_document.get("deployments", [])
        }
        binding_document = deployment_document.get("team_deployment_bindings") or {}
        control_id = str(
            binding_document.get("control_deployment_id")
            or deployment_document.get("control_deployment_id")
            or binding_document.get("default_deployment_id")
            or next(iter(deployment_by_id), "")
        )
        if control_id not in deployment_by_id:
            raise SystemExit(f"unknown control deployment {control_id!r}")
        control_kind = str(deployment_by_id[control_id].get("kind"))
        resolved_provider = "hf" if control_kind == "hf" else "api"
        if args.backend and args.backend != resolved_provider:
            raise SystemExit(
                f"--backend {args.backend!r} conflicts with control deployment kind "
                f"{control_kind!r}"
            )
        backend_provider = resolved_provider
    else:
        backend_provider = args.backend or _get(cfg, "backend.provider", "hf")
    method = args.method or _get(cfg, "router.method", "nl_only")
    if method not in FIXED:
        raise SystemExit(f"本驱动只支持固定通道 method {list(FIXED)}；got {method!r}")
    if backend_provider == "api" and method in {"latent_only", "both"}:
        raise SystemExit(
            "API backend 不支持 latent prefix；请使用 --method none 或 --method nl_only"
        )
    task = args.task or _get(cfg, "run.task", "aime_2024")
    model_path = (
        args.model_path or _get(cfg, "backend.model_path") or os.environ.get("LYCHEE_HF_MODEL")
    )
    api_model = args.api_model or _get(cfg, "backend.model")
    if backend_provider == "hf" and not model_path and not deployment_document:
        raise SystemExit("需要 backend.model_path 或 --model-path 或环境变量 LYCHEE_HF_MODEL")
    model_tag = _model_tag(
        args,
        cfg,
        backend_provider=backend_provider,
        api_model=api_model,
        model_path=model_path,
        suffix="-api" if backend_provider == "api" else "",
    )
    device = args.device or _get(cfg, "backend.device", "cuda:0")
    enable_thinking = _as_bool(_get(cfg, "backend.enable_thinking", False), default=False)
    do_sample = (
        args.do_sample
        if args.do_sample is not None
        else _as_bool(_get(cfg, "backend.do_sample", False), default=False)
    )
    temperature = float(
        args.temperature if args.temperature is not None else _get(cfg, "backend.temperature", 0.7)
    )
    top_p = float(args.top_p if args.top_p is not None else _get(cfg, "backend.top_p", 0.8))
    raw_top_k = args.top_k if args.top_k is not None else _get(cfg, "backend.top_k", None)
    top_k = None if raw_top_k in (None, "") else int(raw_top_k)
    raw_min_p = args.min_p if args.min_p is not None else _get(cfg, "backend.min_p", None)
    min_p = None if raw_min_p in (None, "") else float(raw_min_p)
    raw_presence_penalty = (
        args.presence_penalty
        if args.presence_penalty is not None
        else _get(cfg, "backend.presence_penalty", None)
    )
    presence_penalty = None if raw_presence_penalty in (None, "") else float(raw_presence_penalty)
    seed = int(args.seed if args.seed is not None else _get(cfg, "backend.seed", 0))
    max_input_tokens = (
        args.max_input_tokens
        if args.max_input_tokens is not None
        else _get(cfg, "backend.max_input_tokens", None)
    )
    max_input_tokens = None if max_input_tokens in (None, "", 0, "0") else int(max_input_tokens)
    max_repeated_token_run = int(_get(cfg, "backend.max_repeated_token_run", 0))
    repetition_penalty = float(
        args.repetition_penalty
        if args.repetition_penalty is not None
        else _get(cfg, "backend.repetition_penalty", 1.0)
    )
    if top_k is not None and top_k < 1:
        raise SystemExit("top_k must be >= 1 when configured")
    if not 0.0 < top_p <= 1.0:
        raise SystemExit("top_p must be greater than 0 and at most 1")
    if min_p is not None and not 0.0 <= min_p <= 1.0:
        raise SystemExit("min_p must be between 0 and 1")
    if presence_penalty is not None and not -2.0 <= presence_penalty <= 2.0:
        raise SystemExit("presence_penalty must be between -2 and 2")
    if repetition_penalty <= 0.0:
        raise SystemExit("repetition_penalty must be positive")
    P = _resolve_prefix_length(cfg, args.P)
    latent_strategy = _get(cfg, "memory.latent_strategy", "soft_token")
    nl_strategy = _get(cfg, "memory.nl_strategy", "prev_output")
    max_encode_tokens = int(_get(cfg, "memory.max_encode_tokens", 4096))
    include_transcript = bool(_get(cfg, "memory.include_transcript", True))
    c2c_ckpt = args.c2c_ckpt or _get(cfg, "memory.c2c_ckpt")  # latent_strategy=c2c 时必填
    c2c_gate = _get(cfg, "memory.c2c_gate", "soft")
    nl_simplemem = _get(cfg, "memory.nl_simplemem")  # simplemem 策略传给 SimpleMem(...) 的 kwargs
    raw_n = args.n if args.n is not None else _get(cfg, "run.n", "all")
    n_samples = None if str(raw_n) in ("None", "all", "full", "0") else int(raw_n)
    start_index = int(
        args.start_index if args.start_index is not None else _get(cfg, "run.start_index", 0)
    )
    max_rounds = int(
        args.max_rounds if args.max_rounds is not None else _get(cfg, "run.max_rounds", 2)
    )
    if max_rounds < 1:
        raise SystemExit("max_rounds must be >= 1")
    k_samples = int(args.samples if args.samples is not None else _get(cfg, "run.samples", 1))
    if k_samples < 1:
        raise SystemExit(f"--samples/run.samples 必须 >=1；got {k_samples}")
    if k_samples > 1 and not do_sample:
        print(
            "[warn] samples>1 但 backend.do_sample=false，多次采样可能相同；"
            "建议开 do_sample 让 pass@K 有意义",
            flush=True,
        )
    max_turns = (
        args.max_turns if args.max_turns is not None else _get(cfg, "runtime.max_turns", None)
    )
    max_turns = _optional_positive_limit(max_turns, name="max_turns")
    raw_max_model_calls = (
        args.max_model_calls_per_case
        if args.max_model_calls_per_case is not None
        else _get(cfg, "run.max_model_calls_per_case", None)
    )
    max_model_calls_per_case = _optional_positive_limit(
        raw_max_model_calls,
        name="max_model_calls_per_case",
    )
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else _get(cfg, "run.max_new_tokens", 4096)
    )
    from lychee_mas.runtime.token_budget import normalize_token_budget_policy

    def budget_value(argument_name: str, config_name: str, default=None):
        value = getattr(args, argument_name)
        return value if value is not None else _get(cfg, f"run.{config_name}", default)

    configured_budget = {
        "max_input_tokens": budget_value("max_input_tokens", "max_input_tokens", max_input_tokens),
        "min_output_reserve_tokens": budget_value(
            "min_output_reserve_tokens", "min_output_reserve_tokens", 1024
        ),
        "min_thinking_reserve_tokens": budget_value(
            "min_thinking_reserve_tokens", "min_thinking_reserve_tokens", 0
        ),
        "max_thinking_budget_tokens": budget_value(
            "max_thinking_budget_tokens", "max_thinking_budget_tokens"
        ),
        "min_final_reserve_tokens": budget_value(
            "min_final_reserve_tokens", "min_final_reserve_tokens", 512
        ),
        "safety_margin_tokens": budget_value("safety_margin_tokens", "safety_margin_tokens", 256),
    }
    configured_budget = {
        key: value for key, value in configured_budget.items() if value is not None
    }
    try:
        token_budget_policy = normalize_token_budget_policy(
            configured_budget,
            max_new_tokens=max_new_tokens,
        ).to_dict()
    except ValueError as exc:
        raise SystemExit(f"invalid token budget policy: {exc}") from exc
    # Team selection belongs to the experiment, never to the benchmark loader.
    team_profile = args.team or _get(cfg, "run.team")
    runs_root = args.runs_root or _get(cfg, "eval.runs_root") or default_runs_root()
    explicit_run_dir = args.run_dir or _get(cfg, "eval.run_dir")
    code_executor = args.code_executor or _get(cfg, "runtime.code_executor", "docker")
    docker_image = args.docker_image or _get(cfg, "runtime.docker_image", "python:3.11-slim")
    code_timeout = int(
        args.code_timeout
        if args.code_timeout is not None
        else _get(cfg, "runtime.code_timeout", 60)
    )
    work_root = args.work_root or _get(cfg, "runtime.work_root", "runs/lychee_tool_workspaces")
    web_proxy_url = args.web_proxy_url or (
        network_proxy_url
        if network_mode == "proxy" and _as_bool(network_targets.get("web_surfer"), default=False)
        else None
    )
    container_proxy_url = args.container_proxy_url or (
        _get(cfg, "network.container_proxy_url")
        if network_mode == "proxy" and _as_bool(network_targets.get("code_executor"), default=False)
        else None
    )
    proxy_no_proxy = args.proxy_no_proxy or _get(cfg, "network.no_proxy", "127.0.0.1,localhost,::1")
    trace_model_calls = (
        args.trace_model_calls
        if args.trace_model_calls is not None
        else _get(cfg, "runtime.trace_model_calls", None)
    )
    trace_model_calls = _as_bool(trace_model_calls, default=True)
    trace_detail_level = str(
        args.trace_detail_level or _get(cfg, "runtime.trace_detail_level", "compact")
    )
    if trace_detail_level not in {"compact", "full"}:
        raise SystemExit("runtime.trace_detail_level must be compact or full")
    collect_vllm_metrics = (
        args.collect_vllm_metrics
        if args.collect_vllm_metrics is not None
        else _as_bool(_get(cfg, "observability.collect_vllm_metrics", False), default=False)
    )
    vllm_metrics_interval_s = float(
        args.vllm_metrics_interval_s
        if args.vllm_metrics_interval_s is not None
        else _get(cfg, "observability.vllm_metrics_interval_s", 5.0)
    )
    if vllm_metrics_interval_s <= 0:
        raise SystemExit("observability.vllm_metrics_interval_s must be positive")
    on_case_error = args.on_case_error or _get(cfg, "run.on_case_error", "continue")
    max_case_retries = int(
        args.max_case_retries
        if args.max_case_retries is not None
        else _get(cfg, "run.max_case_retries", 0)
    )
    if max_case_retries < 0:
        raise SystemExit("--max-case-retries must be >= 0")
    case_concurrency = int(
        args.case_concurrency
        if args.case_concurrency is not None
        else _get(cfg, "run.case_concurrency", 1)
    )
    if case_concurrency < 1:
        raise SystemExit("--case-concurrency must be >= 1")
    from lychee_mas.runtime.concurrency import normalize_concurrency_policy

    concurrency_policy_input = dict(_get(cfg, "run.concurrency_policy", {}) or {})
    for key, arg_name in (
        ("mode", "concurrency_mode"),
        ("initial", "concurrency_initial"),
        ("minimum", "concurrency_minimum"),
        ("maximum", "concurrency_maximum"),
        ("increase_step", "concurrency_increase_step"),
        ("decrease_factor", "concurrency_decrease_factor"),
        ("control_window_cases", "concurrency_control_window_cases"),
    ):
        argument = getattr(args, arg_name, None)
        if argument is not None:
            concurrency_policy_input[key] = argument
    try:
        concurrency_policy = normalize_concurrency_policy(
            concurrency_policy_input,
            configured_concurrency=case_concurrency,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    worker_capacity = (
        concurrency_policy["maximum"] if concurrency_policy["mode"] == "auto" else case_concurrency
    )
    deployment_has_hf = bool(
        deployment_document
        and any(item.get("kind") == "hf" for item in deployment_document.get("deployments", []))
    )
    if (backend_provider == "hf" or deployment_has_hf) and worker_capacity > 1:
        raise SystemExit(
            "A single HF model instance cannot safely execute concurrent cases. "
            "Use --case-concurrency 1. To run in parallel, create one DeploymentInstance "
            "per GPU and queue separate ExperimentInstances in Eval Studio."
        )

    # ---- 拓扑 + 结果目录：先确定 out_dir，随后所有终端输出都会 tee 到 console_log.txt ----
    team_spec_path = args.team_spec or _get(cfg, "run.team_spec")
    if team_spec_path:
        from lychee_mas.eval.studio.team_spec import load_team_spec, team_details_from_graph

        graph, _team_document = load_team_spec(team_spec_path, rounds=max_rounds)
        profile = str(graph.meta.get("team") or "studio-team")
        team_details = team_details_from_graph(graph)
    else:
        profile = team_profile or "default"
        graph = RoleProfileTeamBuilder(
            team=profile,
            model=None,
            rounds=max_rounds,
        ).build()
        team_details = _team_details(profile)
    needs_code_executor = any(
        str(node.meta.get("agent_type") or "assistant").lower()
        in {"computer_terminal", "terminal", "code_executor"}
        for node in graph.nodes
    )
    effective_max_turns = _effective_max_turns(
        str(team_details["group_chat_type"]),
        len(graph.nodes),
        max_rounds=max_rounds,
        max_turns=max_turns,
    )
    out_dir = (
        os.path.abspath(os.path.expanduser(explicit_run_dir))
        if explicit_run_dir
        else M.result_dir(model_tag, method, task, root=runs_root, team=profile)
    )
    os.makedirs(out_dir, exist_ok=True)
    console_log_path = _start_console_log(out_dir, append=bool(args.resume))
    sandbox_info = None
    if needs_code_executor and code_executor == "docker":
        sandbox_info = _require_local_docker_image(docker_image)

    # ---- 数据：放在 console log 启动之后，保留 dataset/cache 的终端输出 ----
    load_n = None if n_samples is None else start_index + n_samples
    benchmark = get_benchmark(task)
    data = benchmark.load(task, n=load_n)
    if start_index:
        data = data[start_index:]
    if n_samples is not None:
        data = data[:n_samples]
    if not data:
        raise SystemExit(
            f"no samples loaded for task={task!r}, start_index={start_index}, n={raw_n!r}"
        )
    kind = benchmark.scorer_kinds[task]
    benchmark_id = benchmark.id

    # ---- backend：模型只加载一次，记忆/各 agent 复用（apples-to-apples）----
    deployment_pool = None
    backend_startup_reports = []
    generation_options = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "max_input_tokens": max_input_tokens,
        **token_budget_policy,
        "max_repeated_token_run": max_repeated_token_run,
        "repetition_penalty": repetition_penalty,
    }
    if deployment_document:
        from lychee_mas.runtime.backends.deployment_pool import DeploymentPool

        deployment_pool = DeploymentPool(deployment_document, generation=generation_options)
        deployment_pool.bind_graph(graph)
        if method in {"latent_only", "both"} and (
            len(deployment_pool.configs) != 1
            or deployment_pool.has_kind("api")
            or deployment_pool.has_kind("vllm")
        ):
            raise SystemExit(
                "latent_only/both currently require one shared HF deployment for every role"
            )
        backend = deployment_pool.resolve()
        backend_startup_reports = deployment_pool.prepare_observability()
    elif backend_provider == "api":
        if not api_model:
            raise SystemExit("API backend 需要 backend.model 或 --api-model，例如 qwen-plus")
        api_extra_body = dict(_get(cfg, "backend.extra_body", {}) or {})
        api_extra_body.update(
            {
                key: value
                for key, value in {
                    "top_k": top_k,
                    "min_p": min_p,
                    "presence_penalty": presence_penalty,
                    "repetition_penalty": repetition_penalty,
                }.items()
                if value is not None
            }
        )
        backend = OpenAICompatibleBackend(
            api_model,
            base_url=args.api_base_url or _get(cfg, "backend.base_url"),
            api_key_env=args.api_key_env or _get(cfg, "backend.api_key_env", "DASHSCOPE_API_KEY"),
            auth_mode=str(_get(cfg, "backend.auth_mode", "env")),
            trust_env=_as_bool(_get(cfg, "backend.trust_env", True), default=True),
            proxy_url=model_proxy_url or _get(cfg, "backend.proxy_url"),
            timeout=float(_get(cfg, "backend.timeout", 120.0)),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            model_info=_get(cfg, "backend.model_info", {}),
            request_limits=_get(cfg, "backend.request_limits", {}),
            extra_body=api_extra_body,
            tokenizer_path=_get(cfg, "backend.tokenizer_path") or _get(cfg, "backend.model_path"),
            reasoning_token_accounting=_get(
                cfg, "backend.reasoning_token_accounting", "inconclusive"
            ),
        )
        backend_startup_reports = [
            {
                "deployment_instance_id": "cli-api-backend",
                "kind": "api",
                "model_id": api_model,
                **backend.startup_observability(),
            }
        ]
    elif backend_provider == "hf":
        import torch
        from lychee_mas.runtime.backends.hf_backend import HFBackend

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtypes.get(_get(cfg, "backend.dtype", "bfloat16"), torch.bfloat16)
        backend = HFBackend(
            model_path,
            device=device,
            dtype=dtype,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            seed=seed,
            vision=_get(cfg, "backend.vision", _get(cfg, "backend.model_info.vision")),
            max_input_tokens=max_input_tokens,
            max_repeated_token_run=max_repeated_token_run,
            repetition_penalty=repetition_penalty,
        )
        backend_startup_reports = [
            {
                "deployment_instance_id": "cli-hf-backend",
                "kind": "hf",
                "model_id": model_tag,
                "reasoning_token_accounting": "backend_native",
                "tokenizer_warmup_status": "not_applicable",
            }
        ]
    else:
        raise SystemExit(f"unknown backend.provider {backend_provider!r}; choose hf or api")

    # ---- worker factory：并发 worker 共享 backend，但隔离 memory/router/ctx/runtime ----
    def make_worker():
        worker_memory = DualChannelMemoryManager(
            backend,
            latent_strategy=latent_strategy,
            nl_strategy=nl_strategy,
            max_encode_tokens=max_encode_tokens,
            include_transcript=include_transcript,
            P=P,
            c2c_ckpt=c2c_ckpt,
            c2c_gate=c2c_gate,
            nl_simplemem=nl_simplemem,
        )
        worker_router = fixed_channel_router(FIXED[method])
        worker_ctx = RoutingContext(
            task=task,
            router=worker_router,
            memory=worker_memory,
            team=profile,
        )
        worker_ctx.base_seed = seed
        worker_ctx.trace_detail_level = trace_detail_level
        worker_runtime = AutoGenRuntime(
            backend=backend,
            ctx=worker_ctx,
            max_new_tokens=max_new_tokens,
            max_rounds=max_rounds,
            model_id=model_tag,
            max_turns=effective_max_turns,
            max_model_calls_per_case=max_model_calls_per_case,
            max_stalls=int(_get(cfg, "runtime.max_stalls", 3)),
            work_root=work_root,
            code_executor=code_executor,
            docker_image=docker_image,
            code_timeout=code_timeout,
            web_headless=_as_bool(_get(cfg, "runtime.web_headless", True), default=True),
            save_screenshots=_as_bool(_get(cfg, "runtime.save_screenshots", False), default=False),
            web_proxy_url=web_proxy_url,
            container_proxy_url=container_proxy_url,
            proxy_no_proxy=proxy_no_proxy,
            trace_model_calls=trace_model_calls,
            backend_resolver=deployment_pool.resolve if deployment_pool else None,
            model_resolver=deployment_pool.model_id if deployment_pool else None,
            control_deployment_id=(
                deployment_pool.control_deployment_id if deployment_pool else None
            ),
            max_new_tokens_resolver=(
                deployment_pool.max_new_tokens_for_role if deployment_pool else None
            ),
            control_max_new_tokens=(
                deployment_pool.control_max_new_tokens(max_new_tokens) if deployment_pool else None
            ),
            invocation_overrides_resolver=(
                deployment_pool.invocation_overrides_for_role
                if deployment_pool
                else lambda _role: {"token_budget_policy": token_budget_policy}
            ),
            control_invocation_overrides=(
                deployment_pool.control_invocation_overrides()
                if deployment_pool
                else {"token_budget_policy": token_budget_policy}
            ),
        )
        return worker_memory, worker_router, worker_ctx, worker_runtime

    memory, router, ctx, runtime = make_worker()
    # Team 与 Method 使用独立目录层级，便于按任一维度筛选运行记录。
    out_dir = (
        os.path.abspath(os.path.expanduser(explicit_run_dir))
        if explicit_run_dir
        else M.result_dir(model_tag, method, task, root=runs_root, team=profile)
    )
    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    spans_path = os.path.join(out_dir, "spans.jsonl")
    group_chat_path = os.path.join(out_dir, "group_chat.jsonl")
    vllm_metrics_path = os.path.join(out_dir, "vllm_metrics.jsonl")
    vllm_metrics_summary_path = os.path.join(out_dir, "vllm_metrics_summary.json")
    vllm_metrics_targets = targets_from_deployments(deployment_document)
    if args.vllm_metrics_url:
        vllm_metrics_targets = [
            {
                "deployment_id": "cli-vllm-endpoint",
                "model_id": api_model or model_tag,
                "base_url": args.api_base_url or _get(cfg, "backend.base_url", ""),
                "metrics_url": metrics_url("", args.vllm_metrics_url),
                "trust_env": _as_bool(_get(cfg, "backend.trust_env", True), default=True),
                "exclusive_run_attribution": False,
                "auth_mode": "none",
            }
        ]
    vllm_metrics_monitor = (
        VLLMMetricsMonitor(
            targets=vllm_metrics_targets,
            output_path=vllm_metrics_path,
            summary_path=vllm_metrics_summary_path,
            interval_s=vllm_metrics_interval_s,
            append=bool(args.resume),
        )
        if collect_vllm_metrics
        else None
    )
    completed_samples, resume_info = _prepare_predictions_for_run(
        predictions_path, data, start_index=start_index, resume=bool(args.resume)
    )
    stale_removed = []
    for stale_name in ("outputs.jsonl", "metrics.json"):
        stale_path = os.path.join(out_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            stale_removed.append(stale_name)
    resume_info["stale_removed"] = stale_removed
    span_logger = JsonlSpanLogger(
        spans_path,
        run_fields={
            "task": task,
            "method": method,
            "team": profile,
            "backend_provider": backend_provider,
            "model": model_tag,
            "trace_detail_level": trace_detail_level,
            "benchmark_id": benchmark_id,
        },
        append=bool(args.resume),
    )
    group_chat_logger = JsonlGroupChatLogger(
        group_chat_path,
        run_fields={
            "task": task,
            "method": method,
            "team": profile,
            "backend_provider": backend_provider,
            "model": model_tag,
            "trace_detail_level": trace_detail_level,
            "benchmark_id": benchmark_id,
        },
        append=bool(args.resume),
    )
    ctx.span_logger = span_logger
    ctx.group_chat_logger = group_chat_logger
    n = len(data)
    snapshot = {
        "config_file": args.config,
        "team_spec": team_spec_path,
        "deployment_config": deployment_config_path,
        "config": cfg,
        "resolved": {
            "task": task,
            "backend_provider": backend_provider,
            "method": method,
            "team": profile,
            "group_chat": team_details["group_chat"],
            "group_chat_type": team_details["group_chat_type"],
            "team_chat_mode": team_details["chat_mode"],
            "team_roles": team_details["runtime_roles"],
            "context_visibility": team_details["context_visibility"],
            "P": P,
            "n": n,
            "start_index": start_index,
            "samples": k_samples,
            "model": model_tag,
            "model_path": model_path,
            "memory": memory.name,
            "router": router.name,
            "scorer_kind": kind,
            "benchmark_id": benchmark_id,
            "max_new_tokens": max_new_tokens,
            "token_budget_policy": token_budget_policy,
            "max_rounds": max_rounds,
            "configured_max_turns": max_turns,
            "max_turns": effective_max_turns,
            "max_model_calls_per_case": max_model_calls_per_case,
            "code_executor": code_executor,
            "docker_image": docker_image,
            "docker_sandbox": sandbox_info,
            "code_timeout": code_timeout,
            "work_root": work_root,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_input_tokens": max_input_tokens,
            "max_repeated_token_run": max_repeated_token_run,
            "repetition_penalty": repetition_penalty,
            "base_seed": seed,
            "seed_derivation": PREDICTION_SEED_DERIVATION,
            "trace_model_calls": trace_model_calls,
            "trace_detail_level": trace_detail_level,
            "observability": {
                "collect_vllm_metrics": bool(collect_vllm_metrics),
                "vllm_metrics_interval_s": vllm_metrics_interval_s,
                "vllm_metrics_targets": vllm_metrics_targets,
                "vllm_metrics_path": vllm_metrics_path if collect_vllm_metrics else None,
                "vllm_metrics_summary_path": (
                    vllm_metrics_summary_path if collect_vllm_metrics else None
                ),
                "backend_startup": backend_startup_reports,
            },
            "on_case_error": on_case_error,
            "max_case_retries": max_case_retries,
            "case_concurrency": case_concurrency,
            "concurrency_policy": concurrency_policy,
            "console_log": console_log_path,
            "spans": spans_path,
            "group_chat_path": group_chat_path,
            "predictions": predictions_path,
            "out_dir": out_dir,
            "resume": resume_info,
            "command": sys.argv,
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "LYCHEE_BENCHMARK_RAW_ROOT": os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT"),
                "LYCHEE_BENCHMARK_PREPARED_ROOT": os.environ.get("LYCHEE_BENCHMARK_PREPARED_ROOT"),
                "LYCHEE_BENCHMARK_RUNS_ROOT": os.environ.get("LYCHEE_BENCHMARK_RUNS_ROOT"),
                "CDM_DATA_ROOT": os.environ.get("CDM_DATA_ROOT"),
            },
        },
    }
    M.write_config(out_dir, snapshot)
    _print_resolved_config(
        snapshot,
        out_dir=out_dir,
        predictions_path=predictions_path,
        spans_path=spans_path,
        group_chat_path=group_chat_path,
        resume_info=resume_info,
    )
    if args.resume:
        span_logger.log("resume_start", **resume_info)
    span_logger.log(
        "run_start",
        out_dir=out_dir,
        config_file=args.config,
        num_cases=n,
        start_index=start_index,
        scorer_kind=kind,
        benchmark_id=benchmark_id,
        base_seed=seed,
        seed_derivation=PREDICTION_SEED_DERIVATION,
        max_new_tokens=max_new_tokens,
        token_budget_policy=token_budget_policy,
        max_rounds=max_rounds,
        configured_max_turns=max_turns,
        max_turns=effective_max_turns,
        max_model_calls_per_case=max_model_calls_per_case,
        group_chat=team_details["group_chat"],
        group_chat_type=team_details["group_chat_type"],
        team_chat_mode=team_details["chat_mode"],
        team_roles=team_details["runtime_roles"],
        context_visibility=team_details["context_visibility"],
        on_case_error=on_case_error,
        max_case_retries=max_case_retries,
        case_concurrency=case_concurrency,
        concurrency_policy=concurrency_policy,
        code_executor=code_executor,
        docker_image=docker_image,
        trace_model_calls=trace_model_calls,
        trace_detail_level=trace_detail_level,
        group_chat_path=group_chat_path,
        observability=snapshot["resolved"]["observability"],
        resume=resume_info,
    )
    for report in backend_startup_reports:
        span_logger.log("backend_startup", **report)
        print(
            "[backend:start] "
            f"deployment={report.get('deployment_instance_id')} "
            f"accounting={report.get('reasoning_token_accounting')} "
            f"tokenizer_warmup={report.get('tokenizer_warmup_status')} "
            f"latency_s={report.get('tokenizer_warmup_latency_s', 0.0)}",
            flush=True,
        )
    print(
        f"[MAS] task={task} backend={backend_provider} method={method} "
        f"team={profile} router={router.name} "
        f"memory={memory.name} n={n} kind={kind} P={P} max_new_tokens={max_new_tokens} "
        f"max_model_calls_per_case={max_model_calls_per_case or 'unlimited'}",
        flush=True,
    )
    print(
        "[MAS:generation] "
        f"do_sample={do_sample} temperature={temperature} top_p={top_p} "
        f"top_k={top_k} min_p={min_p} presence_penalty={presence_penalty} "
        f"repetition_penalty={repetition_penalty} base_seed={seed} "
        f"seed_derivation={PREDICTION_SEED_DERIVATION}",
        flush=True,
    )
    role_summary = ", ".join(
        f"{role['name']}[{role['agent_type']}]" for role in team_details["runtime_roles"]
    )
    print(f"[MAS:team] chat={team_details['chat_mode']} roles={role_summary}", flush=True)

    run_status_path = os.path.join(out_dir, "run_status.json")
    previous_run_status = _read_json(run_status_path) if args.resume else {}
    previous_cumulative_elapsed_s = _run_status_elapsed_seconds(previous_run_status)
    current_segment_started_at = round(time.time(), 6)
    predictions = _read_jsonl(predictions_path)
    run_status = {
        "schema_version": 1,
        "status": "running",
        "task": task,
        "model": model_tag,
        "method": method,
        "team": profile,
        "expected_distinct_cases": n,
        "requested_samples_per_case": k_samples,
        "base_seed": seed,
        "seed_derivation": PREDICTION_SEED_DERIVATION,
        "expected_predictions": n * k_samples,
        "successful_predictions": sum(
            str(record.get("status") or "ok") != "error" for record in predictions
        ),
        "error_predictions": 0,
        "skipped_predictions": sum(len(ks) for ks in completed_samples.values()),
        "retry_attempts": 0,
        "on_case_error": on_case_error,
        "max_case_retries": max_case_retries,
        "concurrency_policy": concurrency_policy,
        "current_case_concurrency": concurrency_policy["initial"],
        "concurrency_history": [],
        "started_at_unix_s": current_segment_started_at,
        "current_segment_started_at_unix_s": current_segment_started_at,
        "accumulated_run_elapsed_before_segment_s": round(previous_cumulative_elapsed_s, 6),
        "cumulative_run_elapsed_s": round(previous_cumulative_elapsed_s, 6),
        "updated_at_unix_s": round(time.time(), 6),
        "predictions_path": predictions_path,
        "spans_path": spans_path,
        "group_chat_path": group_chat_path,
        "vllm_metrics_path": vllm_metrics_path if collect_vllm_metrics else None,
        "vllm_metrics_summary_path": (vllm_metrics_summary_path if collect_vllm_metrics else None),
        "active_cases": [],
        "last_completed_case": None,
    }
    _atomic_write_json(run_status_path, run_status)

    work_items: list[tuple[int, dict, int]] = []
    for case_offset, item in enumerate(data):
        sample_index = start_index + case_offset
        case_id = _case_id(item, sample_index)
        completed_k = completed_samples.get(case_id, set())
        for k_index in range(k_samples):
            if k_index in completed_k:
                ctx.set_case(case_id, sample_index)
                ctx.current_k_index = k_index
                ctx.current_attempt = None
                ctx.set_case_span(None)
                ctx.prediction_seed = derive_prediction_seed(
                    base_seed=seed,
                    benchmark_id=benchmark_id,
                    task=task,
                    case_id=case_id,
                    k_index=k_index,
                )
                ctx.generation_seed = ctx.prediction_seed
                ctx.seed_derivation = PREDICTION_SEED_DERIVATION
                k_tag = f" k={k_index + 1}/{k_samples}" if k_samples > 1 else ""
                print(
                    f"  [{case_offset + 1}/{len(data)}{k_tag}] skip completed case_id={case_id}",
                    flush=True,
                )
                ctx.log_span(
                    "case_skip",
                    case_order=case_offset + 1,
                    num_cases=len(data),
                    k_index=k_index,
                    reason="resume_existing_success_prediction",
                )
                continue
            work_items.append((case_offset, item, k_index))

    workers = [(memory, router, ctx, runtime)]
    for _ in range(1, min(worker_capacity, max(1, len(work_items)))):
        worker = make_worker()
        worker[2].span_logger = span_logger
        worker[2].group_chat_logger = group_chat_logger
        workers.append(worker)
    run_status["effective_case_concurrency"] = len(workers)
    _atomic_write_json(run_status_path, run_status)

    queue: asyncio.Queue = asyncio.Queue()
    for work_item in work_items:
        queue.put_nowait(work_item)
    for _ in workers:
        queue.put_nowait(None)
    output_lock = asyncio.Lock()
    from lychee_mas.runtime.concurrency import CaseConcurrencyController

    concurrency_controller = CaseConcurrencyController(concurrency_policy)
    run_status["current_case_concurrency"] = concurrency_controller.target
    run_status["concurrency_history"] = list(concurrency_controller.history)

    async def worker_loop(worker_id: int, worker) -> None:
        worker_memory, _worker_router, worker_ctx, worker_runtime = worker
        while True:
            work_item = await queue.get()
            if work_item is None:
                queue.task_done()
                return
            await concurrency_controller.acquire()
            case_offset, item, k_index = work_item
            sample_index = start_index + case_offset
            case_id = _case_id(item, sample_index)
            async with output_lock:
                active_cases = {
                    int(active["worker_id"]): active
                    for active in run_status.get("active_cases") or []
                }
                active_cases[worker_id] = {
                    "worker_id": worker_id,
                    "case_order": case_offset + 1,
                    "num_cases": len(data),
                    "case_id": case_id,
                    "k_index": k_index,
                    "prediction_seed": derive_prediction_seed(
                        base_seed=seed,
                        benchmark_id=benchmark_id,
                        task=task,
                        case_id=case_id,
                        k_index=k_index,
                    ),
                    "samples_per_case": k_samples,
                    "started_at_unix_s": round(time.time(), 6),
                }
                run_status["active_cases"] = list(active_cases.values())
                run_status["updated_at_unix_s"] = round(time.time(), 6)
                _atomic_write_json(run_status_path, run_status)
            try:
                result = await _execute_case_sample(
                    case_order=case_offset + 1,
                    num_cases=len(data),
                    item=item,
                    sample_index=sample_index,
                    k_index=k_index,
                    samples_per_case=k_samples,
                    task=task,
                    kind=kind,
                    benchmark_id=benchmark_id,
                    method=method,
                    profile=profile,
                    max_rounds=max_rounds,
                    max_case_retries=max_case_retries,
                    memory=worker_memory,
                    ctx=worker_ctx,
                    runtime=worker_runtime,
                    base_graph=graph if team_spec_path else None,
                )
            finally:
                await concurrency_controller.release()
            record = result["record"]
            concurrency_event = await concurrency_controller.observe(
                record=record,
                service_metrics=(
                    vllm_metrics_monitor.latest_snapshots()
                    if vllm_metrics_monitor is not None
                    else []
                ),
            )
            async with output_lock:
                run_status["active_cases"] = [
                    active
                    for active in run_status.get("active_cases") or []
                    if int(active.get("worker_id", -1)) != worker_id
                ]
                predictions.append(record)
                with open(predictions_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_status["retry_attempts"] += int(result["retry_count"])
                if record.get("status") == "error":
                    run_status["error_predictions"] += 1
                    run_status["last_error"] = {
                        "case_id": record["case_id"],
                        "k_index": record.get("k_index", 0),
                        "error_type": record.get("error_type"),
                        "error_message": record.get("error_message"),
                    }
                else:
                    run_status["successful_predictions"] += 1
                run_status["last_completed_case"] = {
                    "worker_id": worker_id,
                    "case_order": case_offset + 1,
                    "num_cases": len(data),
                    "case_id": record["case_id"],
                    "k_index": record.get("k_index", 0),
                    "status": record.get("status", "ok"),
                    "score": record.get("score"),
                    "finished_at_unix_s": round(time.time(), 6),
                }
                run_status["current_case_concurrency"] = concurrency_controller.target
                if concurrency_event is not None:
                    run_status["concurrency_history"] = list(concurrency_controller.history)
                    worker_ctx.log_span(
                        "case_concurrency_change",
                        **concurrency_event,
                        policy=concurrency_policy,
                    )
                    print(
                        "[concurrency] "
                        f"reason={concurrency_event['reason']} "
                        f"target={concurrency_event['previous']}->"
                        f"{concurrency_event['target']}",
                        flush=True,
                    )
                run_status["updated_at_unix_s"] = round(time.time(), 6)
                if record.get("status") == "error" and on_case_error == "fail-fast":
                    run_status["status"] = "failed_fast"
                _atomic_write_json(run_status_path, run_status)
            queue.task_done()
            if record.get("status") == "error" and on_case_error == "fail-fast":
                raise result["exception"]

    vllm_metrics_summary = None
    if vllm_metrics_monitor is not None:
        vllm_metrics_monitor.start()
        print(
            f"[observability] vllm_metrics targets={len(vllm_metrics_targets)} "
            f"interval_s={vllm_metrics_interval_s} path={vllm_metrics_path}",
            flush=True,
        )
    try:
        await asyncio.gather(
            *(worker_loop(worker_id, worker) for worker_id, worker in enumerate(workers))
        )
    finally:
        if vllm_metrics_monitor is not None:
            vllm_metrics_summary = vllm_metrics_monitor.stop()
            run_status["vllm_metrics_summary"] = vllm_metrics_summary
            run_status["updated_at_unix_s"] = round(time.time(), 6)
            _atomic_write_json(run_status_path, run_status)
    predictions.sort(
        key=lambda record: (
            int(record.get("sample_index", 10**18) or 10**18),
            int(record.get("k_index", 0) or 0),
        )
    )
    _write_jsonl(predictions_path, predictions)

    skipped_total = sum(len(ks) for ks in completed_samples.values())
    completed_total = run_status["successful_predictions"] + run_status["error_predictions"]
    run_status["missing_predictions"] = max(0, n * k_samples - completed_total)
    run_status["status"] = (
        "complete"
        if run_status["error_predictions"] == 0 and run_status["missing_predictions"] == 0
        else "complete_with_errors"
    )
    run_status["active_cases"] = []
    run_status["finished_at_unix_s"] = round(time.time(), 6)
    run_status["cumulative_run_elapsed_s"] = round(
        previous_cumulative_elapsed_s
        + max(0.0, run_status["finished_at_unix_s"] - current_segment_started_at),
        6,
    )
    run_status["updated_at_unix_s"] = run_status["finished_at_unix_s"]
    _atomic_write_json(run_status_path, run_status)
    print(
        f"[done] task={task} team={profile} method={method}: predictions={len(predictions)} "
        f"samples_per_case={k_samples} skipped={skipped_total} "
        f"errors={run_status['error_predictions']} status={run_status['status']} -> {out_dir}",
        flush=True,
    )
    span_logger.log(
        "run_end",
        num_predictions=len(predictions),
        samples_per_case=k_samples,
        skipped_predictions=skipped_total,
        error_predictions=run_status["error_predictions"],
        run_status=run_status["status"],
        vllm_metrics_summary=vllm_metrics_summary,
        out_dir=out_dir,
    )
    return {
        "out_dir": out_dir,
        "num_predictions": len(predictions),
        "run_status": run_status["status"],
        "vllm_metrics_summary": vllm_metrics_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LycheeMAS 真实 MAS 推理驱动（新框架版）")
    ap.add_argument("--config", default=None, help="YAML 配置（backend/memory/router/run/eval）")
    ap.add_argument(
        "--list-team-structure",
        action="store_true",
        help="列出 runnable task、默认 team、AutoGen 调度方式和实际角色；不加载模型/数据",
    )
    ap.add_argument(
        "--runtime",
        default=None,
        choices=("autogen",),
        help="覆盖 runtime.name；工具能力也走 autogen 主链路",
    )
    ap.add_argument("--task", default=None, help="覆盖 run.task")
    ap.add_argument("--method", default=None, help="none|nl_only|latent_only|both")
    ap.add_argument("--backend", default=None, choices=("hf", "api"), help="覆盖 backend.provider")
    ap.add_argument("--team", default=None, help="显式覆盖 YAML/task_config.py 默认 team")
    ap.add_argument(
        "--team-spec",
        dest="team_spec",
        default=None,
        help="Eval Studio team JSON；可定义自定义节点、边、角色和 deployment 绑定",
    )
    ap.add_argument(
        "--deployment-config",
        dest="deployment_config",
        default=None,
        help="Eval Studio deployment JSON；同一 deployment ID 在所有角色间复用 backend",
    )
    ap.add_argument("--n", default=None, help="样本数；all/0=全量")
    ap.add_argument(
        "--start-index",
        dest="start_index",
        type=int,
        default=None,
        help="从 loader 返回列表的第几个样本开始跑，0-based",
    )
    ap.add_argument("--P", type=int, default=None, help="latent prefix 长度")
    ap.add_argument(
        "--c2c-ckpt",
        dest="c2c_ckpt",
        default=None,
        help="latent_strategy=c2c 时的 projector 栈 ckpt 路径（覆盖 memory.c2c_ckpt）",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=None,
        help="每题采样次数 K（pass@K；K>1 建议 backend.do_sample=true；打分侧按 case 聚合）",
    )
    ap.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=None,
        help="启用采样解码；pass@K 通常应开启",
    )
    ap.add_argument(
        "--no-do-sample", dest="do_sample", action="store_false", help="强制使用确定性解码"
    )
    ap.add_argument("--temperature", type=float, default=None, help="覆盖 backend.temperature")
    ap.add_argument("--top-p", dest="top_p", type=float, default=None, help="覆盖 backend.top_p")
    ap.add_argument("--top-k", dest="top_k", type=int, default=None, help="覆盖 backend.top_k")
    ap.add_argument("--min-p", dest="min_p", type=float, default=None, help="覆盖 backend.min_p")
    ap.add_argument(
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=None,
        help="覆盖 backend.presence_penalty",
    )
    ap.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="覆盖 backend.repetition_penalty",
    )
    ap.add_argument("--seed", type=int, default=None, help="覆盖 backend.seed")
    ap.add_argument("--model-path", dest="model_path", default=None)
    ap.add_argument("--model-tag", dest="model_tag", default=None)
    ap.add_argument(
        "--api-model", dest="api_model", default=None, help="OpenAI-compatible model name"
    )
    ap.add_argument("--api-base-url", dest="api_base_url", default=None)
    ap.add_argument("--api-key-env", dest="api_key_env", default=None)
    ap.add_argument("--device", default=None, help="覆盖 backend.device（本地 HF 路径）")
    ap.add_argument(
        "--max-rounds", dest="max_rounds", type=int, default=None, help="覆盖 run.max_rounds"
    )
    ap.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="覆盖 run.max_new_tokens（每次模型调用最多生成多少 token）",
    )
    ap.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="单次模型调用允许保留的最大输入 token 数",
    )
    ap.add_argument(
        "--min-output-reserve-tokens",
        type=int,
        default=None,
        help="裁剪输入时为思考与最终答案合计保留的最小输出容量",
    )
    ap.add_argument(
        "--min-thinking-reserve-tokens",
        type=int,
        default=None,
        help="思考模式开启时希望保留的最小 reasoning 容量",
    )
    ap.add_argument(
        "--max-thinking-budget-tokens",
        type=int,
        default=None,
        help="provider 支持时对 reasoning token 的独立硬上限",
    )
    ap.add_argument(
        "--min-final-reserve-tokens",
        type=int,
        default=None,
        help="从本次输出预算中为最终回答保留的最小容量",
    )
    ap.add_argument(
        "--safety-margin-tokens",
        type=int,
        default=None,
        help="模型上下文窗口末端不参与输入或输出分配的安全边距",
    )
    ap.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=None,
        help="覆盖 runtime.max_turns（AutoGen group chat 最大发言上限）",
    )
    ap.add_argument(
        "--max-model-calls-per-case",
        dest="max_model_calls_per_case",
        default=None,
        help="每个 case 允许的真实模型调用数；正整数或 unlimited",
    )
    ap.add_argument(
        "--code-executor",
        dest="code_executor",
        default=None,
        choices=("docker", "local"),
        help="需要执行代码时使用的执行器",
    )
    ap.add_argument(
        "--docker-image",
        dest="docker_image",
        default=None,
        help="覆盖 runtime.docker_image（docker executor）",
    )
    ap.add_argument(
        "--code-timeout",
        dest="code_timeout",
        type=int,
        default=None,
        help="覆盖 runtime.code_timeout",
    )
    ap.add_argument(
        "--work-root", dest="work_root", default=None, help="每个 case 的工具/附件 workspace 根目录"
    )
    ap.add_argument(
        "--web-proxy-url",
        default=None,
        help="WebSurfer/Playwright 使用的显式 HTTP(S) 代理地址",
    )
    ap.add_argument(
        "--container-proxy-url",
        default=None,
        help="代码执行容器内可访问的代理地址，例如 http://host.docker.internal:17897",
    )
    ap.add_argument(
        "--proxy-no-proxy",
        default=None,
        help="容器和显式网络客户端绕过代理的主机列表",
    )
    ap.add_argument(
        "--proxy-relay-upstream-url",
        default=None,
        help="需要为 Docker 建立网桥转发时的宿主机上游代理",
    )
    ap.add_argument(
        "--proxy-relay-listen-host",
        default=None,
        help="Docker 网桥上的 relay 监听地址",
    )
    ap.add_argument(
        "--proxy-relay-port",
        type=int,
        default=None,
        help="Docker 网桥上的 relay 监听端口",
    )
    ap.add_argument(
        "--trace-model-calls",
        dest="trace_model_calls",
        action="store_true",
        default=None,
        help="打印每次 LLM 调用的 start/done 摘要；默认开启",
    )
    ap.add_argument(
        "--no-trace-model-calls",
        dest="trace_model_calls",
        action="store_false",
        help="关闭每次 LLM 调用的 start/done 摘要",
    )
    ap.add_argument(
        "--trace-detail-level",
        choices=("compact", "full"),
        default=None,
        help=(
            "span 记录详细度；compact 省略 model_call_start 的三层消息，"
            "full 保存 autogen_model_messages/role_visible_messages/backend_messages"
        ),
    )
    ap.add_argument(
        "--collect-vllm-metrics",
        dest="collect_vllm_metrics",
        action="store_true",
        default=None,
        help="定时采集 vLLM /metrics，并写入独立 JSONL 时间序列",
    )
    ap.add_argument(
        "--no-collect-vllm-metrics",
        dest="collect_vllm_metrics",
        action="store_false",
        help="关闭 vLLM 服务级指标采集",
    )
    ap.add_argument(
        "--vllm-metrics-interval-s",
        type=float,
        default=None,
        help="vLLM /metrics 采样间隔（秒，默认 5）",
    )
    ap.add_argument(
        "--vllm-metrics-url",
        default=None,
        help="无 DeploymentConfig 时显式指定 vLLM Prometheus endpoint",
    )
    ap.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        help="benchmark run result root",
    )
    ap.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help=(
            "exact directory for this run; overrides the automatic "
            "<runs-root>/<model>/<team>/<method>/<task> path"
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume this run directory: append log/spans, keep successful predictions, "
            "retry failed/missing cases"
        ),
    )
    ap.add_argument(
        "--on-case-error",
        choices=("continue", "fail-fast"),
        default=None,
        help="case 运行失败时继续后续 case（默认）或立即终止",
    )
    ap.add_argument(
        "--max-case-retries",
        type=int,
        default=None,
        help="单个 (case_id,k_index) 失败后的额外重试次数，默认 0",
    )
    ap.add_argument(
        "--case-concurrency",
        type=int,
        default=None,
        help="同一进程并发执行的 case 数；API/vLLM 可大于 1，本地 HF 固定为 1",
    )
    ap.add_argument("--concurrency-mode", choices=("fixed", "auto"), default=None)
    ap.add_argument("--concurrency-initial", type=int, default=None)
    ap.add_argument("--concurrency-minimum", type=int, default=None)
    ap.add_argument("--concurrency-maximum", type=int, default=None)
    ap.add_argument("--concurrency-increase-step", type=int, default=None)
    ap.add_argument("--concurrency-decrease-factor", type=float, default=None)
    ap.add_argument("--concurrency-control-window-cases", type=int, default=None)
    args = ap.parse_args()
    if args.list_team_structure:
        _print_team_structure(
            task=args.task,
            team_override=args.team,
        )
        return
    if not args.config:
        ap.error("--config is required unless --list-team-structure is used")
    cfg = _apply_benchmark_contract(_load_yaml(args.config), args)
    relay_context = nullcontext()
    relay_upstream = args.proxy_relay_upstream_url or _get(cfg, "network.proxy_relay_upstream_url")
    relay_requested = bool(
        args.container_proxy_url
        or (
            _as_bool(_get(cfg, "network.targets.code_executor", False), default=False)
            and _get(cfg, "network.container_proxy_url")
        )
    )
    if relay_upstream and relay_requested:
        from lychee_mas.runtime.proxy import ProxyRelay

        relay_context = ProxyRelay(
            relay_upstream,
            listen_host=args.proxy_relay_listen_host
            or _get(cfg, "network.docker_bridge_host", "172.17.0.1"),
            listen_port=args.proxy_relay_port
            or int(_get(cfg, "network.container_proxy_port", 17897)),
        )
    with relay_context:
        asyncio.run(run_one(cfg, args))


if __name__ == "__main__":
    main()
