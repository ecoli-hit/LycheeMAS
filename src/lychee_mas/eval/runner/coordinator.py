"""Coordinate one benchmark Run from immutable configuration to terminal events."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from . import configuration as _runner_config
from . import run_state as _runner_state
from .backend_assembly import BackendAssemblyRequest, assemble_backend
from .run_artifacts import (
    model_tag as resolve_model_tag,
)
from .run_artifacts import (
    prepare_trials_for_resume,
    print_resolved_config,
    require_local_docker_image,
    start_console_log,
)
from .run_finalization import (
    finalize_run_segment,
    record_cancelled_run,
    record_failed_run,
)
from .trial_identity import case_id_from_item, dataset_index_from_item
from .trial_pool import TrialPoolRequest, execute_trial_pool

_ROOT = str(Path(__file__).resolve().parents[4])
FIXED = {"none": "none", "nl_only": "nl", "latent_only": "latent", "both": "both"}
RUN_CONTROL_FILENAME = "run_control.json"
RUN_SEGMENT_FILENAME = "run_segment.json"

_as_bool = _runner_config.as_bool
_get = _runner_config.get_config_value
_resolve_prefix_length = _runner_config.resolve_prefix_length
_atomic_write_json = _runner_state.atomic_write_json
_read_json = _runner_state.read_json
_run_status_elapsed_seconds = _runner_state.run_status_elapsed_seconds
_scheduler_trial_admission_total = _runner_state.scheduler_trial_admission_total
_scheduler_trial_concurrency = _runner_state.scheduler_trial_concurrency


def _optional_positive_limit(value, *, name: str) -> int | None:
    try:
        return _runner_config.optional_positive_limit(value, name=name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc




async def run_one(cfg: dict, args) -> dict:
    from lychee_mas.eval.benchmarks import get_benchmark
    from lychee_mas.eval.benchmarks.common import runs_root as default_runs_root
    from lychee_mas.eval.evaluation import metrics as M
    from lychee_mas.memory.context import RoutingContext
    from lychee_mas.memory.managers.DualChannelMemory import DualChannelMemoryManager
    from lychee_mas.memory.routing.static import fixed_channel_router
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import _effective_max_turns
    from lychee_mas.runtime.adapters.frameworks.factory import build_runtime
    from lychee_mas.runtime.adapters.inference.vllm_metrics import (
        VLLMMetricsMonitor,
        metrics_url,
        targets_from_deployments,
    )
    from lychee_mas.runtime.events.store import (
        RunEventWriter,
        read_unfinished_trial_operations,
    )
    from lychee_mas.runtime.events.store import (
        run_events_path as resolve_events_path,
    )
    from lychee_mas.runtime.execution.seeding import (
        TRIAL_SEED_DERIVATION,
        derive_trial_seed,
    )

    runtime_name = args.runtime or _get(
        cfg, "runtime.framework", _get(cfg, "runtime.name", "autogen")
    )

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
    model_tag = resolve_model_tag(
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
    case_limit = None if str(raw_n) in ("None", "all", "full", "0") else int(raw_n)
    start_index = int(
        args.start_index if args.start_index is not None else _get(cfg, "run.start_index", 0)
    )
    case_selection = str(
        args.case_selection
        if args.case_selection is not None
        else _get(cfg, "run.case_selection", "head")
    )
    case_strata_field = (
        args.case_strata_field
        if args.case_strata_field is not None
        else _get(cfg, "run.case_strata_field", None)
    )
    max_rounds = int(
        args.max_rounds if args.max_rounds is not None else _get(cfg, "run.max_rounds", 2)
    )
    if max_rounds < 1:
        raise SystemExit("max_rounds must be >= 1")
    trials_per_case = int(
        args.trials_per_case
        if args.trials_per_case is not None
        else _get(cfg, "run.trials_per_case", 1)
    )
    if trials_per_case < 1:
        raise SystemExit(f"--trials-per-case/run.trials_per_case 必须 >=1；got {trials_per_case}")
    if trials_per_case > 1 and not do_sample:
        print(
            "[warn] trials_per_case>1 但 backend.do_sample=false，多次 Trial 可能相同；"
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
    raw_case_wall_time = (
        args.max_case_wall_time_s
        if args.max_case_wall_time_s is not None
        else _get(cfg, "run.max_case_wall_time_s", None)
    )
    if isinstance(raw_case_wall_time, str) and raw_case_wall_time.strip().lower() in {
        "",
        "none",
        "null",
        "unlimited",
        "unbounded",
    }:
        raw_case_wall_time = None
    max_case_wall_time_s = (
        None if raw_case_wall_time is None else float(raw_case_wall_time)
    )
    if max_case_wall_time_s is not None and max_case_wall_time_s <= 0:
        raise SystemExit("max_case_wall_time_s must be positive or unlimited")
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else _get(cfg, "run.max_new_tokens", 4096)
    )
    from lychee_mas.runtime.model.token_budget import normalize_token_budget_policy

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
    event_log_max_events_per_file = (
        args.event_log_max_events_per_file
        if args.event_log_max_events_per_file is not None
        else _get(cfg, "observability.event_log_max_events_per_file")
    )
    event_log_max_mib_per_file = (
        args.event_log_max_mib_per_file
        if args.event_log_max_mib_per_file is not None
        else _get(cfg, "observability.event_log_max_mib_per_file")
    )
    if event_log_max_events_per_file is not None:
        event_log_max_events_per_file = int(event_log_max_events_per_file)
        if event_log_max_events_per_file <= 0:
            raise SystemExit("event log max events per file must be positive")
    if event_log_max_mib_per_file is not None:
        event_log_max_mib_per_file = float(event_log_max_mib_per_file)
        if event_log_max_mib_per_file <= 0:
            raise SystemExit("event log max MiB per file must be positive")
    on_trial_error = args.on_trial_error or _get(cfg, "run.on_trial_error", "continue")
    max_attempts_per_trial = int(
        args.max_attempts_per_trial
        if args.max_attempts_per_trial is not None
        else _get(cfg, "run.max_attempts_per_trial", 1)
    )
    if max_attempts_per_trial < 1:
        raise SystemExit("--max-attempts-per-trial must be >= 1")
    trial_concurrency = int(
        args.trial_concurrency
        if args.trial_concurrency is not None
        else _get(cfg, "run.trial_concurrency", 1)
    )
    if trial_concurrency < 1:
        raise SystemExit("--trial-concurrency must be >= 1")
    from lychee_mas.runtime.execution.concurrency import normalize_concurrency_policy

    concurrency_policy_input = dict(_get(cfg, "run.concurrency_policy", {}) or {})
    for key, arg_name in (
        ("mode", "concurrency_mode"),
        ("initial", "concurrency_initial"),
        ("minimum", "concurrency_minimum"),
        ("maximum", "concurrency_maximum"),
        ("increase_step", "concurrency_increase_step"),
        ("decrease_factor", "concurrency_decrease_factor"),
        ("control_window_trials", "concurrency_control_window_trials"),
    ):
        argument = getattr(args, arg_name, None)
        if argument is not None:
            concurrency_policy_input[key] = argument
    try:
        concurrency_policy = normalize_concurrency_policy(
            concurrency_policy_input,
            configured_concurrency=trial_concurrency,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    worker_capacity = (
        concurrency_policy["maximum"] if concurrency_policy["mode"] == "auto" else trial_concurrency
    )
    deployment_has_hf = bool(
        deployment_document
        and any(item.get("kind") == "hf" for item in deployment_document.get("deployments", []))
    )
    if (backend_provider == "hf" or deployment_has_hf) and worker_capacity > 1:
        raise SystemExit(
            "A single HF model instance cannot safely execute concurrent cases. "
            "Use --trial-concurrency 1. To run in parallel, create one DeploymentInstance "
            "per GPU and queue separate ExperimentInstances in Eval Studio."
        )

    # ---- 拓扑 + 结果目录：先确定 out_dir，随后所有终端输出都会 tee 到 console_log.txt ----
    team_spec_path = args.team_spec or _get(cfg, "run.team_spec")
    if not team_spec_path:
        raise SystemExit(
            "Eval runs require an explicit --team-spec. RoleProfile/agent_type graphs are "
            "not part of the framework-neutral Eval runtime contract."
        )
    from lychee_mas.eval.teams.compiler import load_team_spec, team_details_from_graph

    graph, _team_document = load_team_spec(team_spec_path, rounds=max_rounds)
    profile = str(graph.meta.get("team") or "studio-team")
    team_details = team_details_from_graph(graph)
    needs_code_executor = any(
        (
            str(node.meta.get("execution_kind") or "") == "executor"
            and str(node.meta.get("executor_type") or "") == "code"
        )
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
    console_log_path = start_console_log(out_dir, append=bool(args.resume))
    sandbox_info = None
    if needs_code_executor and code_executor == "docker":
        sandbox_info = require_local_docker_image(docker_image, repo_root=_ROOT)

    # ---- 数据：放在 console log 启动之后，保留 dataset/cache 的终端输出 ----
    load_n = (
        None
        if case_limit is None or case_selection != "head"
        else start_index + case_limit
    )
    benchmark = get_benchmark(task)
    loaded_data = benchmark.load(task, n=load_n)
    from lychee_mas.eval.benchmarks.sampling import select_cases

    data, case_selection_info = select_cases(
        loaded_data,
        count=case_limit,
        start_index=start_index,
        method=case_selection,
        strata_field=case_strata_field,
    )
    if not data:
        raise SystemExit(
            f"no samples loaded for task={task!r}, start_index={start_index}, n={raw_n!r}"
        )
    kind = benchmark.scorer_kinds[task]
    benchmark_id = benchmark.id

    # ---- backend：模型只加载一次，记忆/各 agent 复用（apples-to-apples）----
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
    assembly = assemble_backend(
        BackendAssemblyRequest(
            backend_provider=backend_provider,
            method=method,
            model_tag=model_tag,
            api_model=api_model,
            model_path=model_path,
            device=device,
            enable_thinking=enable_thinking,
            dtype_name=str(_get(cfg, "backend.dtype", "bfloat16")),
            vision=_get(cfg, "backend.vision", _get(cfg, "backend.model_info.vision")),
            deployment_document=deployment_document,
            graph=graph,
            generation=generation_options,
            api_options={
                "base_url": args.api_base_url or _get(cfg, "backend.base_url"),
                "api_key_env": args.api_key_env
                or _get(cfg, "backend.api_key_env", "DASHSCOPE_API_KEY"),
                "auth_mode": str(_get(cfg, "backend.auth_mode", "env")),
                "trust_env": _as_bool(_get(cfg, "backend.trust_env", True), default=True),
                "proxy_url": model_proxy_url or _get(cfg, "backend.proxy_url"),
                "timeout": float(_get(cfg, "backend.timeout", 120.0)),
                "model_info": _get(cfg, "backend.model_info", {}),
                "request_limits": _get(cfg, "backend.request_limits", {}),
                "extra_body": dict(_get(cfg, "backend.extra_body", {}) or {}),
                "tokenizer_path": _get(cfg, "backend.tokenizer_path")
                or _get(cfg, "backend.model_path"),
                "reasoning_token_accounting": _get(
                    cfg, "backend.reasoning_token_accounting", "inconclusive"
                ),
            },
        )
    )
    backend = assembly.backend
    deployment_pool = assembly.deployment_pool
    backend_startup_reports = assembly.startup_reports

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
        worker_runtime = build_runtime(
            runtime_name,
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
            memory_method=method,
            framework_options=dict(_get(cfg, "runtime.framework_options", {}) or {}),
        )
        return worker_memory, worker_router, worker_ctx, worker_runtime

    memory, router, ctx, runtime = make_worker()
    # Team 与 Method 使用独立目录层级，便于按任一维度筛选运行记录。
    out_dir = (
        os.path.abspath(os.path.expanduser(explicit_run_dir))
        if explicit_run_dir
        else M.result_dir(model_tag, method, task, root=runs_root, team=profile)
    )
    run_events_path = str(resolve_events_path(out_dir))
    run_control_path = os.path.join(out_dir, RUN_CONTROL_FILENAME)
    Path(run_control_path).unlink(missing_ok=True)
    run_segment_path = os.path.join(out_dir, RUN_SEGMENT_FILENAME)
    segment_request = _read_json(run_segment_path)
    raw_segment_case_limit = (
        args.segment_case_limit
        if args.segment_case_limit is not None
        else segment_request.get("segment_case_limit")
    )
    segment_case_limit = _optional_positive_limit(
        raw_segment_case_limit,
        name="segment_case_limit",
    )
    initial_scheduler_trial_concurrency = _scheduler_trial_concurrency(
        run_segment_path,
        fallback=None,
    )
    initial_scheduler_trial_admission_total = _scheduler_trial_admission_total(
        run_segment_path,
        fallback=None,
    )
    concurrency_authority = str(
        segment_request.get("concurrency_authority") or "runner"
    ).strip().lower()
    if concurrency_authority not in {"runner", "scheduler"}:
        raise SystemExit("run segment concurrency_authority must be runner or scheduler")
    scheduler_controls_concurrency = concurrency_authority == "scheduler"
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
    completed_trials, resume_info = prepare_trials_for_resume(
        out_dir, data, start_index=start_index, resume=bool(args.resume)
    )
    unfinished_trial_operations = (
        read_unfinished_trial_operations(out_dir) if args.resume else []
    )
    resume_info["interrupted_trial_operations_closed"] = len(
        unfinished_trial_operations
    )
    stale_removed = []
    for stale_name in (
        "metrics.json",
        "evidence.jsonl",
        "metric_observations.jsonl",
        "metric_coverage.json",
        "metric_evidence.jsonl",
    ):
        stale_path = os.path.join(out_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            stale_removed.append(stale_name)
    resume_info["stale_removed"] = stale_removed
    event_writer = RunEventWriter(
        run_events_path,
        experiment_instance_id=_get(cfg, "experiment.instance_id"),
        append=bool(args.resume),
        max_events_per_file=event_log_max_events_per_file,
        max_bytes_per_file=(
            int(event_log_max_mib_per_file * 1024 * 1024)
            if event_log_max_mib_per_file is not None
            else None
        ),
    )
    ctx.event_writer = event_writer
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
            "case_selection": case_selection_info,
            "trials_per_case": trials_per_case,
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
            "max_case_wall_time_s": max_case_wall_time_s,
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
            "seed_derivation": TRIAL_SEED_DERIVATION,
            "trace_model_calls": trace_model_calls,
            "observability": {
                "collect_vllm_metrics": bool(collect_vllm_metrics),
                "vllm_metrics_interval_s": vllm_metrics_interval_s,
                "vllm_metrics_targets": vllm_metrics_targets,
                "vllm_metrics_path": vllm_metrics_path if collect_vllm_metrics else None,
                "vllm_metrics_summary_path": (
                    vllm_metrics_summary_path if collect_vllm_metrics else None
                ),
                "event_log_max_events_per_file": event_writer.max_events_per_file,
                "event_log_max_bytes_per_file": event_writer.max_bytes_per_file,
                "event_log_max_mib_per_file": round(
                    event_writer.max_bytes_per_file / (1024 * 1024), 6
                ),
                "backend_startup": backend_startup_reports,
            },
            "on_trial_error": on_trial_error,
            "max_attempts_per_trial": max_attempts_per_trial,
            "trial_concurrency": trial_concurrency,
            "concurrency_policy": concurrency_policy,
            "segment_case_limit": segment_case_limit,
            "segment_request": segment_request,
            "console_log": console_log_path,
            "run_events_path": run_events_path,
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
    print_resolved_config(
        snapshot,
        out_dir=out_dir,
        run_events_path=run_events_path,
        resume_info=resume_info,
    )
    if args.resume:
        resumed_event_id = event_writer.log_event("run.resumed", **resume_info)
        for event in unfinished_trial_operations:
            event_writer.log_event(
                "trial.interrupted",
                case_id=event.get("case_id"),
                dataset_index=event.get("dataset_index"),
                trial_index=event.get("trial_index"),
                operation_id=event.get("operation_id"),
                parent_event_id=resumed_event_id,
                interrupted_started_event_id=event.get("event_id"),
                previous_parent_event_id=event.get("parent_event_id"),
                reason="previous_process_ended_without_trial_terminal",
            )
    run_event_id = event_writer.log_event(
        "run.started",
        out_dir=out_dir,
        config_file=args.config,
        num_cases=n,
        start_index=start_index,
        case_selection=case_selection_info,
        scorer_kind=kind,
        benchmark_id=benchmark_id,
        base_seed=seed,
        seed_derivation=TRIAL_SEED_DERIVATION,
        max_new_tokens=max_new_tokens,
        token_budget_policy=token_budget_policy,
        max_rounds=max_rounds,
        configured_max_turns=max_turns,
        max_turns=effective_max_turns,
        max_model_calls_per_case=max_model_calls_per_case,
        max_case_wall_time_s=max_case_wall_time_s,
        group_chat=team_details["group_chat"],
        group_chat_type=team_details["group_chat_type"],
        team_chat_mode=team_details["chat_mode"],
        team_roles=team_details["runtime_roles"],
        context_visibility=team_details["context_visibility"],
        on_trial_error=on_trial_error,
        max_attempts_per_trial=max_attempts_per_trial,
        trial_concurrency=trial_concurrency,
        concurrency_policy=concurrency_policy,
        segment_case_limit=segment_case_limit,
        segment_request=segment_request,
        code_executor=code_executor,
        docker_image=docker_image,
        trace_model_calls=trace_model_calls,
        run_events_path=run_events_path,
        observability=snapshot["resolved"]["observability"],
        resume=resume_info,
    )
    ctx.current_run_event_id = run_event_id
    for report in backend_startup_reports:
        event_writer.log_event("backend.probed", parent_event_id=run_event_id, **report)
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
        f"max_case_wall_time_s={max_case_wall_time_s or 'unlimited'}",
        flush=True,
    )
    print(
        "[MAS:generation] "
        f"do_sample={do_sample} temperature={temperature} top_p={top_p} "
        f"top_k={top_k} min_p={min_p} presence_penalty={presence_penalty} "
        f"repetition_penalty={repetition_penalty} base_seed={seed} "
        f"seed_derivation={TRIAL_SEED_DERIVATION}",
        flush=True,
    )
    role_summary = ", ".join(
        f"{role['name']}["
        f"{role.get('execution_kind') or 'model'}]"
        for role in team_details["runtime_roles"]
    )
    print(f"[MAS:team] chat={team_details['chat_mode']} roles={role_summary}", flush=True)

    run_status_path = os.path.join(out_dir, "run_status.json")
    previous_run_status = _read_json(run_status_path) if args.resume else {}
    previous_cumulative_elapsed_s = _run_status_elapsed_seconds(previous_run_status)
    current_segment_started_at = round(time.time(), 6)
    from lychee_mas.runtime.events.store import event_data, read_trial_records

    trial_records = (
        [event_data(event) for event in read_trial_records(out_dir)] if args.resume else []
    )
    successful_trials_by_case = {
        case_id: set(indices) for case_id, indices in completed_trials.items()
    }
    incomplete_case_ids: list[str] = []
    for case_offset, item in enumerate(data):
        dataset_index = dataset_index_from_item(item, start_index + case_offset)
        case_id = case_id_from_item(item, dataset_index)
        if len(successful_trials_by_case.get(case_id, set())) < trials_per_case:
            incomplete_case_ids.append(case_id)
    selected_case_ids = (
        incomplete_case_ids[:segment_case_limit]
        if segment_case_limit is not None
        else incomplete_case_ids
    )
    segment_truncated = len(selected_case_ids) < len(incomplete_case_ids)
    selected_case_id_set = set(selected_case_ids)
    completed_distinct_cases = sum(
        len(indices) >= trials_per_case for indices in successful_trials_by_case.values()
    )
    run_status = {
        "schema_version": 1,
        "control_protocol_version": 2,
        "status": "running",
        "task": task,
        "model": model_tag,
        "method": method,
        "team": profile,
        "expected_distinct_cases": n,
        "requested_trials_per_case": trials_per_case,
        "base_seed": seed,
        "seed_derivation": TRIAL_SEED_DERIVATION,
        "expected_trials": n * trials_per_case,
        "successful_trials": sum(
            record.get("event_type") != "trial.failed" for record in trial_records
        ),
        "failed_trials": 0,
        "skipped_trials": sum(len(indices) for indices in completed_trials.values()),
        "completed_distinct_cases": completed_distinct_cases,
        "remaining_distinct_cases": max(0, n - completed_distinct_cases),
        "segment_index": int(segment_request.get("segment_index") or 1),
        "segment_case_limit": segment_case_limit,
        "segment_selected_distinct_cases": len(selected_case_ids),
        "segment_selected_case_ids": selected_case_ids,
        "retry_attempt_count": 0,
        "on_trial_error": on_trial_error,
        "max_attempts_per_trial": max_attempts_per_trial,
        "concurrency_policy": concurrency_policy,
        "concurrency_authority": concurrency_authority,
        "current_trial_concurrency": concurrency_policy["initial"],
        "adaptive_trial_concurrency_target": (
            None if scheduler_controls_concurrency else concurrency_policy["initial"]
        ),
        "scheduler_trial_concurrency": initial_scheduler_trial_concurrency,
        "scheduler_trial_admission_total": initial_scheduler_trial_admission_total,
        "segment_started_trials": 0,
        "concurrency_history": [],
        "started_at_unix_s": current_segment_started_at,
        "current_segment_started_at_unix_s": current_segment_started_at,
        "accumulated_run_elapsed_before_segment_s": round(previous_cumulative_elapsed_s, 6),
        "cumulative_run_elapsed_s": round(previous_cumulative_elapsed_s, 6),
        "updated_at_unix_s": round(time.time(), 6),
        "run_events_path": run_events_path,
        "run_control_path": run_control_path,
        "run_segment_path": run_segment_path,
        "accepting_new_trials": True,
        "stop_requested_at_utc": None,
        "stop_reason": None,
        "vllm_metrics_path": vllm_metrics_path if collect_vllm_metrics else None,
        "vllm_metrics_summary_path": (vllm_metrics_summary_path if collect_vllm_metrics else None),
        "active_cases": [],
        "last_completed_case": None,
    }
    _atomic_write_json(run_status_path, run_status)

    work_items: list[tuple[int, dict, int]] = []
    for case_offset, item in enumerate(data):
        dataset_index = dataset_index_from_item(item, start_index + case_offset)
        case_id = case_id_from_item(item, dataset_index)
        completed_k = completed_trials.get(case_id, set())
        if case_id not in selected_case_id_set:
            continue
        for trial_index in range(trials_per_case):
            if trial_index in completed_k:
                ctx.set_case(case_id, dataset_index)
                ctx.current_trial_index = trial_index
                ctx.current_attempt = None
                ctx.set_trial_event(None)
                ctx.trial_seed = derive_trial_seed(
                    base_seed=seed,
                    benchmark_id=benchmark_id,
                    task=task,
                    case_id=case_id,
                    trial_index=trial_index,
                )
                ctx.generation_seed = ctx.trial_seed
                ctx.seed_derivation = TRIAL_SEED_DERIVATION
                trial_tag = f" k={trial_index + 1}/{trials_per_case}" if trials_per_case > 1 else ""
                print(
                    f"  [{case_offset + 1}/{len(data)}{trial_tag}] "
                    f"skip completed case_id={case_id}",
                    flush=True,
                )
                ctx.log_event(
                    "trial.skipped",
                    case_order=case_offset + 1,
                    num_cases=len(data),
                    trial_index=trial_index,
                    reason="resume_existing_successful_trial",
                )
                continue
            work_items.append((case_offset, item, trial_index))

    trial_pool_request = TrialPoolRequest(
        work_items=work_items,
        initial_worker=(memory, router, ctx, runtime),
        worker_factory=make_worker,
        worker_capacity=worker_capacity,
        run_status=run_status,
        run_status_path=run_status_path,
        run_control_path=run_control_path,
        run_event_id=run_event_id,
        event_writer=event_writer,
        concurrency_policy=concurrency_policy,
        initial_scheduler_trial_concurrency=initial_scheduler_trial_concurrency,
        initial_scheduler_trial_admission_total=initial_scheduler_trial_admission_total,
        scheduler_controls_concurrency=scheduler_controls_concurrency,
        data_count=n,
        start_index=start_index,
        trials_per_case=trials_per_case,
        base_seed=seed,
        benchmark_id=benchmark_id,
        task=task,
        kind=kind,
        method=method,
        profile=profile,
        max_rounds=max_rounds,
        max_case_wall_time_s=max_case_wall_time_s,
        max_attempts_per_trial=max_attempts_per_trial,
        on_trial_error=on_trial_error,
        successful_trials_by_case=successful_trials_by_case,
        trial_records=trial_records,
        base_graph=graph,
        vllm_metrics_monitor=vllm_metrics_monitor,
    )
    pool_result = None

    vllm_metrics_summary = None
    if vllm_metrics_monitor is not None:
        vllm_metrics_monitor.start()
        print(
            f"[observability] vllm_metrics targets={len(vllm_metrics_targets)} "
            f"interval_s={vllm_metrics_interval_s} path={vllm_metrics_path}",
            flush=True,
        )
    try:
        try:
            pool_result = await execute_trial_pool(trial_pool_request)
        finally:
            if vllm_metrics_monitor is not None:
                vllm_metrics_summary = vllm_metrics_monitor.stop()
                run_status["vllm_metrics_summary"] = vllm_metrics_summary
                run_status["updated_at_unix_s"] = round(time.time(), 6)
                _atomic_write_json(run_status_path, run_status)
    except asyncio.CancelledError:
        record_cancelled_run(
            run_status=run_status,
            run_status_path=run_status_path,
            event_writer=event_writer,
            operation_id=run_event_id,
            previous_elapsed_s=previous_cumulative_elapsed_s,
            segment_started_at_unix_s=current_segment_started_at,
            vllm_metrics_summary=vllm_metrics_summary,
            out_dir=out_dir,
        )
        raise
    except BaseException as exc:
        record_failed_run(
            run_status=run_status,
            run_status_path=run_status_path,
            event_writer=event_writer,
            operation_id=run_event_id,
            error=exc,
            vllm_metrics_summary=vllm_metrics_summary,
            out_dir=out_dir,
        )
        raise
    settlement = finalize_run_segment(
        run_status=run_status,
        run_status_path=run_status_path,
        run_control_path=run_control_path,
        event_writer=event_writer,
        operation_id=run_event_id,
        trial_records=trial_records,
        completed_trials=completed_trials,
        expected_cases=n,
        trials_per_case=trials_per_case,
        segment_truncated=segment_truncated,
        graceful_stop_requested=bool(
            pool_result and pool_result.graceful_stop_requested
        ),
        previous_elapsed_s=previous_cumulative_elapsed_s,
        segment_started_at_unix_s=current_segment_started_at,
        vllm_metrics_summary=vllm_metrics_summary,
        out_dir=out_dir,
    )
    skipped_total = int(settlement["skipped_trials"])
    print(
        f"[done] task={task} team={profile} method={method}: trial_records={len(trial_records)} "
        f"trials_per_case={trials_per_case} skipped={skipped_total} "
        f"errors={run_status['failed_trials']} status={run_status['status']} -> {out_dir}",
        flush=True,
    )
    return {
        "out_dir": out_dir,
        "num_trials": len(trial_records),
        "run_status": run_status["status"],
        "vllm_metrics_summary": vllm_metrics_summary,
    }
