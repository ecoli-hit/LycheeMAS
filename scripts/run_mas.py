"""真实 MAS 推理驱动（新框架 lychee_mas 版）—— 把 backend + CDM 记忆 + 固定通道路由 + RoutingContext
+ 静态拓扑 + AutoGenRuntime 接成端到端可跑的实验，逐样本生成 prediction 并落盘。

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
import subprocess
import sys
import time

# 让 `python scripts/run_mas.py` 直接可用（把 src/ 加进路径）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
for _pkg in ("autogen-core", "autogen-agentchat", "autogen-ext"):
    _pkg_src = os.path.join(_ROOT, "src", "autogen", "python", "packages", _pkg, "src")
    if os.path.isdir(_pkg_src) and _pkg_src not in sys.path:
        sys.path.append(_pkg_src)

# router.method 名 -> fixed_channel_router 的通道值：四种「强制某通道」的消融基线
FIXED = {"none": "none", "nl_only": "nl", "latent_only": "latent", "both": "both"}


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


def _as_bool(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _decision_number(decision: dict, *names: str, default=0):
    for name in names:
        if name in decision and decision[name] is not None:
            return decision[name]
    return default


def _normalize_model_call(decision: dict) -> dict:
    input_positions = int(_decision_number(decision, "input_positions", "prompt_pos", default=0))
    latent_positions = int(_decision_number(
        decision, "latent_prefix_positions", "prefix_len", default=0))
    text_tokens = int(_decision_number(
        decision, "text_input_tokens", default=max(0, input_positions - latent_positions)))
    output_tokens = int(_decision_number(decision, "output_tokens", "gen_tokens", default=0))
    latency = float(_decision_number(
        decision, "model_generation_latency_s", "latency_s", default=0.0))
    normalized = dict(decision)
    normalized.update({
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
    })
    return normalized


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


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    valid_case_ids = {
        _case_id(item, start_index + offset)
        for offset, item in enumerate(data)
    }
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

    kept.sort(key=lambda r: (int(r.get("sample_index", 10**18) or 10**18),
                             int(r.get("k_index", 0) or 0)))
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


def _print_resolved_config(snapshot: dict, *, out_dir: str, predictions_path: str,
                           spans_path: str, resume_info: dict) -> None:
    resolved = dict(snapshot.get("resolved") or {})
    resolved["out_dir"] = out_dir
    resolved["predictions_path"] = predictions_path
    resolved["spans_path"] = spans_path
    resolved["resume"] = resume_info
    print("[config] resolved parameters:", flush=True)
    print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _model_tag(args, cfg: dict, *, backend_provider: str | None = None,
               api_model: str | None = None, model_path: str | None = None,
               suffix: str = "") -> str:
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


def _require_local_docker_image(image: str) -> None:
    """Fail early for local benchmark images so AutoGen does not try a registry pull."""
    if not image.endswith(":local"):
        return
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Docker CLI not found while checking local image {image!r}. "
            "Install Docker or use --code-executor local only for trusted tasks."
        ) from exc
    if result.returncode == 0:
        return
    raise SystemExit(
        f"Docker local image {image!r} was not found. Build the official benchmark images before running this "
        "benchmark; otherwise AutoGen will try to pull the name from a registry and may fail "
        "with proxy/registry errors. Check with: docker image inspect "
        f"{image}. Prepare command: scripts/prepare_benchmarks.py --tasks human_eval,gaia"
    )


async def run_one(cfg: dict, args) -> dict:
    from lychee_mas.core.types import TaskQuery
    from lychee_mas.eval import metrics as M
    from lychee_mas.eval.benchmarks import load as load_task
    from lychee_mas.eval.benchmarks.common import runs_root as default_runs_root
    from lychee_mas.eval.task_config import team_name_for_task
    from lychee_mas.layers.construct.templates import StaticTopology
    from lychee_mas.memory.context import RoutingContext
    from lychee_mas.memory.managers.DualChannelMemory import DualChannelMemoryManager
    from lychee_mas.memory.routing.static import fixed_channel_router
    from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend
    from lychee_mas.runtime.spans import JsonlSpanLogger, exception_record

    runtime_name = args.runtime or _get(cfg, "runtime.name", "autogen")
    if runtime_name != "autogen":
        raise SystemExit(f"unknown runtime.name {runtime_name!r}; use autogen")

    # ---- 解析参数（命令行 > YAML）----
    backend_provider = args.backend or _get(cfg, "backend.provider", "hf")
    method = args.method or _get(cfg, "router.method", "nl_only")
    if method not in FIXED:
        raise SystemExit(f"本驱动只支持固定通道 method {list(FIXED)}；got {method!r}")
    if backend_provider == "api" and method in {"latent_only", "both"}:
        raise SystemExit("API backend 不支持 latent prefix；请使用 --method none 或 --method nl_only")
    task = args.task or _get(cfg, "run.task", "aime_2024")
    model_path = (args.model_path or _get(cfg, "backend.model_path")
                  or os.environ.get("LYCHEE_HF_MODEL"))
    api_model = args.api_model or _get(cfg, "backend.model")
    if backend_provider == "hf" and not model_path:
        raise SystemExit("需要 backend.model_path 或 --model-path 或环境变量 LYCHEE_HF_MODEL")
    model_tag = _model_tag(args, cfg, backend_provider=backend_provider,
                           api_model=api_model, model_path=model_path,
                           suffix="-api" if backend_provider == "api" else "")
    device = args.device or _get(cfg, "backend.device", "cuda:0")
    enable_thinking = _as_bool(_get(cfg, "backend.enable_thinking", False), default=False)
    do_sample = _as_bool(_get(cfg, "backend.do_sample", False), default=False)
    temperature = float(_get(cfg, "backend.temperature", 0.7))
    top_p = float(_get(cfg, "backend.top_p", 0.8))
    seed = int(_get(cfg, "backend.seed", 0))
    max_input_tokens = _get(cfg, "backend.max_input_tokens", None)
    max_input_tokens = None if max_input_tokens in (None, "", 0, "0") else int(max_input_tokens)
    max_repeated_token_run = int(_get(cfg, "backend.max_repeated_token_run", 128))
    repetition_penalty = float(_get(cfg, "backend.repetition_penalty", 1.0))
    # P（latent prefix 长度）现归 latent 通道：优先 memory.P，回退旧位置 router.P（向后兼容）
    P = args.P if args.P is not None else int(_get(cfg, "memory.P", _get(cfg, "router.P", 16)))
    latent_strategy = _get(cfg, "memory.latent_strategy", "soft_token")
    nl_strategy = _get(cfg, "memory.nl_strategy", "prev_output")
    max_encode_tokens = int(_get(cfg, "memory.max_encode_tokens", 4096))
    include_transcript = bool(_get(cfg, "memory.include_transcript", True))
    c2c_ckpt = args.c2c_ckpt or _get(cfg, "memory.c2c_ckpt")  # latent_strategy=c2c 时必填
    c2c_gate = _get(cfg, "memory.c2c_gate", "soft")
    nl_simplemem = _get(cfg, "memory.nl_simplemem")  # simplemem 策略传给 SimpleMem(...) 的 kwargs
    raw_n = args.n if args.n is not None else _get(cfg, "run.n", "all")
    n_samples = None if str(raw_n) in ("None", "all", "full", "0") else int(raw_n)
    start_index = int(args.start_index if args.start_index is not None else _get(
        cfg, "run.start_index", 0))
    max_rounds = int(args.max_rounds if args.max_rounds is not None else _get(cfg, "run.max_rounds", 2))
    k_samples = int(args.samples if args.samples is not None else _get(cfg, "run.samples", 1))
    if k_samples < 1:
        raise SystemExit(f"--samples/run.samples 必须 >=1；got {k_samples}")
    if k_samples > 1 and not do_sample:
        print("[warn] samples>1 但 backend.do_sample=false，多次采样可能相同；建议开 do_sample 让 pass@K 有意义",
              flush=True)
    max_turns = args.max_turns if args.max_turns is not None else _get(cfg, "runtime.max_turns", None)
    max_turns = None if max_turns is None else int(max_turns)
    cli_max_new_tokens = args.max_new_tokens
    if cli_max_new_tokens is None and args.max_tokens is not None:
        cli_max_new_tokens = args.max_tokens
    max_new_tokens = int(cli_max_new_tokens if cli_max_new_tokens is not None else _get(
        cfg, "run.max_new_tokens", 4096))
    team_profile = args.team or _get(cfg, "run.team")  # None=按 task 自动选
    results_root = (
        args.runs_root
        or args.results_root
        or _get(cfg, "eval.runs_root")
        or _get(cfg, "eval.results_root")
        or default_runs_root()
    )
    code_executor = args.code_executor or _get(cfg, "runtime.code_executor", "docker")
    docker_image = args.docker_image or _get(cfg, "runtime.docker_image", "python:3.11-slim")
    code_timeout = int(args.code_timeout if args.code_timeout is not None else _get(
        cfg, "runtime.code_timeout", 120))
    work_root = args.work_root or _get(cfg, "runtime.work_root", "runs/lychee_tool_workspaces")
    trace_model_calls = (
        args.trace_model_calls
        if args.trace_model_calls is not None
        else _get(cfg, "runtime.trace_model_calls", None)
    )
    trace_model_calls = _as_bool(trace_model_calls, default=True)

    # ---- 拓扑 + 结果目录：先确定 out_dir，随后所有终端输出都会 tee 到 console_log.txt ----
    profile = team_profile or team_name_for_task(task)
    graph = StaticTopology(team=profile, model=None, rounds=max_rounds).build()
    uses_tool_team = any((node.meta.get("agent_type") or "assistant") != "assistant" or node.tools
                         for node in graph.nodes)
    run_label = f"{profile}_{method}" if (team_profile or uses_tool_team) else method
    out_dir = M.result_dir(model_tag, run_label, task, root=results_root)
    console_log_path = _start_console_log(out_dir, append=bool(args.resume))
    if uses_tool_team and code_executor == "docker":
        _require_local_docker_image(docker_image)

    # ---- 数据：放在 console log 启动之后，保留 dataset/cache 的终端输出 ----
    load_n = None if n_samples is None else start_index + n_samples
    data = load_task(task, n=load_n)
    if start_index:
        data = data[start_index:]
    if n_samples is not None:
        data = data[:n_samples]
    if not data:
        raise SystemExit(
            f"no samples loaded for task={task!r}, start_index={start_index}, n={raw_n!r}"
        )
    kind = data[0]["kind"]

    # ---- backend：模型只加载一次，记忆/各 agent 复用（apples-to-apples）----
    if backend_provider == "api":
        if not api_model:
            raise SystemExit("API backend 需要 backend.model 或 --api-model，例如 qwen-plus")
        backend = OpenAICompatibleBackend(
            api_model,
            base_url=args.api_base_url or _get(cfg, "backend.base_url"),
            api_key_env=args.api_key_env or _get(cfg, "backend.api_key_env", "DASHSCOPE_API_KEY"),
            timeout=float(_get(cfg, "backend.timeout", 120.0)),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            model_info=_get(cfg, "backend.model_info", {}),
        )
    elif backend_provider == "hf":
        import torch
        from lychee_mas.runtime.backends.hf_backend import HFBackend

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtypes.get(_get(cfg, "backend.dtype", "bfloat16"), torch.bfloat16)
        backend = HFBackend(model_path, device=device, dtype=dtype, enable_thinking=enable_thinking,
                            do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed,
                            max_input_tokens=max_input_tokens,
                            max_repeated_token_run=max_repeated_token_run,
                            repetition_penalty=repetition_penalty)
    else:
        raise SystemExit(f"unknown backend.provider {backend_provider!r}; choose hf or api")
    # ---- 记忆方法（CDM 双通道）+ 通道决策（固定通道路由）----
    memory = DualChannelMemoryManager(backend, latent_strategy=latent_strategy,
                                      nl_strategy=nl_strategy,
                                      max_encode_tokens=max_encode_tokens,
                                      include_transcript=include_transcript,
                                      P=P, c2c_ckpt=c2c_ckpt, c2c_gate=c2c_gate,
                                      nl_simplemem=nl_simplemem)
    router = fixed_channel_router(FIXED[method])
    # ---- 共享 ctx + runtime（每样本 ctx.reset；同一 backend/graph 复用）----
    ctx = RoutingContext(task=task, router=router, memory=memory, team=profile)
    runtime = AutoGenRuntime(backend=backend, ctx=ctx, max_new_tokens=max_new_tokens,
                             max_rounds=max_rounds, model_id=model_tag,
                             max_turns=max_turns,
                             max_stalls=int(_get(cfg, "runtime.max_stalls", 3)),
                             work_root=work_root,
                             code_executor=code_executor,
                             docker_image=docker_image,
                             code_timeout=code_timeout,
                             web_headless=_as_bool(_get(cfg, "runtime.web_headless", True), default=True),
                             save_screenshots=_as_bool(_get(cfg, "runtime.save_screenshots", False), default=False),
                             trace_model_calls=trace_model_calls)
    # 落盘标签：强制队伍时用 "{team}_{method}"（如单模型 baseline=single_none），避免撞目录
    run_label = f"{profile}_{method}" if (team_profile or uses_tool_team) else method
    out_dir = M.result_dir(model_tag, run_label, task, root=results_root)
    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    spans_path = os.path.join(out_dir, "spans.jsonl")
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
    span_logger = JsonlSpanLogger(spans_path, run_fields={
        "task": task,
        "method": method,
        "team": profile,
        "backend_provider": backend_provider,
        "model": model_tag,
    }, append=bool(args.resume))
    ctx.span_logger = span_logger
    n = len(data)
    snapshot = {"config_file": args.config, "config": cfg,
                "resolved": {"task": task, "backend_provider": backend_provider,
                             "method": method, "team": profile,
                             "P": P, "n": n, "start_index": start_index,
                             "samples": k_samples,
                             "model": model_tag, "model_path": model_path,
                             "memory": memory.name, "router": router.name,
                             "scorer_kind": kind,
                             "max_new_tokens": max_new_tokens, "max_rounds": max_rounds,
                             "max_turns": max_turns, "code_executor": code_executor,
                             "docker_image": docker_image, "code_timeout": code_timeout,
                             "work_root": work_root,
                             "do_sample": do_sample, "temperature": temperature,
                             "top_p": top_p, "max_input_tokens": max_input_tokens,
                             "max_repeated_token_run": max_repeated_token_run,
                             "repetition_penalty": repetition_penalty,
                             "trace_model_calls": trace_model_calls,
                             "console_log": console_log_path,
                             "spans": spans_path,
                             "predictions": predictions_path,
                             "out_dir": out_dir,
                             "resume": resume_info,
                             "command": sys.argv,
                             "environment": {
                                 "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "LYCHEE_BENCHMARK_RAW_ROOT": os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT"),
                                 "LYCHEE_BENCHMARK_PREPARED_ROOT": os.environ.get(
                                     "LYCHEE_BENCHMARK_PREPARED_ROOT"),
                                 "LYCHEE_BENCHMARK_RUNS_ROOT": os.environ.get(
                                     "LYCHEE_BENCHMARK_RUNS_ROOT"),
                                 "CDM_DATA_ROOT": os.environ.get("CDM_DATA_ROOT"),
                             }}}
    M.write_config(out_dir, snapshot)
    _print_resolved_config(snapshot, out_dir=out_dir, predictions_path=predictions_path,
                           spans_path=spans_path, resume_info=resume_info)
    if args.resume:
        span_logger.log("resume_start", **resume_info)
    span_logger.log(
        "run_start",
        out_dir=out_dir,
        config_file=args.config,
        num_cases=n,
        start_index=start_index,
        scorer_kind=kind,
        max_new_tokens=max_new_tokens,
        max_rounds=max_rounds,
        max_turns=max_turns,
        code_executor=code_executor,
        docker_image=docker_image,
        trace_model_calls=trace_model_calls,
        resume=resume_info,
    )
    print(f"[MAS] task={task} backend={backend_provider} method={method} team={profile} router={router.name} "
          f"memory={memory.name} n={n} kind={kind} P={P} max_new_tokens={max_new_tokens}",
          flush=True)

    predictions = _read_jsonl(predictions_path)

    for i, it in enumerate(data):
        sample_index = start_index + i
        case_id = _case_id(it, sample_index)
        done_ks = completed_samples.get(case_id, set())
        q = TaskQuery(question=it["question"], context=it.get("context"),
                      gold=None, id=case_id, meta={"kind": kind})
        for k in range(k_samples):
            # k_index/num_samples 仅在 pass@K（K>1）时写入，保证 K=1 的 prediction 与旧格式逐字一致
            k_tag = f" k={k + 1}/{k_samples}" if k_samples > 1 else ""
            sample_extra = {"k_index": k, "num_samples": k_samples} if k_samples > 1 else {}
            if k in done_ks:
                ctx.set_case(case_id, sample_index)
                print(f"  [{i + 1}/{len(data)}{k_tag}] skip completed case_id={case_id}", flush=True)
                ctx.log_span(
                    "case_skip",
                    case_order=i + 1,
                    num_cases=len(data),
                    k_index=k,
                    reason="resume_existing_success_prediction",
                )
                continue
            ctx.reset()  # 清 turn/决策/记忆库
            ctx.set_case(case_id, sample_index)
            print(f"  [{i + 1}/{len(data)}{k_tag}] start case_id={case_id}", flush=True)
            case_t0 = time.time()
            ctx.log_span(
                "case_start",
                case_order=i + 1,
                num_cases=len(data),
                k_index=k,
                question=it["question"],
                has_context=bool(it.get("context")),
                scorer_kind=kind,
            )
            try:
                # 长程记忆任务把对话历史预载进记忆库（AIME 无 context，此处不触发）
                if it.get("context") and hasattr(memory, "seed"):
                    memory.seed(it["context"])
                traj = await runtime.run(graph, q)
            except Exception as exc:
                err = exception_record(exc)
                model_calls = [_normalize_model_call(d) for d in ctx.decisions]
                case_generation_latency = sum(c["model_generation_latency_s"] for c in model_calls)
                error_sample = {
                    "case_id": case_id, "sample_index": sample_index,
                    **sample_extra,
                    "task": task, "method": method,
                    "scorer_kind": kind, "question": it["question"][:500],
                    "final_answer": "",
                    "status": "error",
                    "error_type": err["error_type"],
                    "error_message": err["error_message"],
                    "traceback": err["traceback"],
                    "num_model_calls": len(model_calls),
                    "num_messages": 0,
                    "sum_input_total_positions": sum(c["input_positions"] for c in model_calls),
                    "sum_input_text_tokens": sum(c["text_input_tokens"] for c in model_calls),
                    "sum_input_latent_positions": sum(
                        c["latent_prefix_positions"] for c in model_calls),
                    "sum_output_text_tokens": sum(c["output_tokens"] for c in model_calls),
                    "sum_model_latency_s": round(case_generation_latency, 3),
                    "case_wall_time_s": round(time.time() - case_t0, 3),
                    "num_tool_calls": 0,
                    "num_tool_errors": 0,
                    "model_call_count": len(model_calls),
                    "message_count": 0,
                    "tool_call_count": 0,
                    "tool_error_count": 0,
                    "input_positions_total": sum(c["input_positions"] for c in model_calls),
                    "text_input_tokens_total": sum(c["text_input_tokens"] for c in model_calls),
                    "latent_prefix_positions_total": sum(
                        c["latent_prefix_positions"] for c in model_calls),
                    "output_tokens_total": sum(c["output_tokens"] for c in model_calls),
                    "model_generation_latency_s_total": round(case_generation_latency, 3),
                    "model_calls": model_calls,
                    "tool_calls": [],
                    "routing_trace": model_calls,
                    "n_messages": 0,
                    "cost_prompt_pos": sum(c["input_positions"] for c in model_calls),
                    "gen_tokens": sum(c["output_tokens"] for c in model_calls),
                    "latency_s": round(case_generation_latency, 3),
                }
                with open(predictions_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(error_sample, ensure_ascii=False) + "\n")
                ctx.log_span(
                    "case_error",
                    case_order=i + 1,
                    num_cases=len(data),
                    k_index=k,
                    partial_model_call_count=len(model_calls),
                    case_wall_time_s=round(time.time() - case_t0, 3),
                    **err,
                )
                print(
                    f"  [{i + 1}/{len(data)}{k_tag}] error case_id={case_id} "
                    f"{err['error_type']}: {err['error_message']}",
                    flush=True,
                )
                raise
            pred = traj.final_answer.content if traj.final_answer else ""
            model_calls = [_normalize_model_call(d) for d in traj.meta.get("decisions", [])]

            case_input_positions = sum(c["input_positions"] for c in model_calls)
            case_text_input_tokens = sum(c["text_input_tokens"] for c in model_calls)
            case_latent_prefix_positions = sum(c["latent_prefix_positions"] for c in model_calls)
            case_output_tokens = sum(c["output_tokens"] for c in model_calls)
            case_generation_latency = sum(c["model_generation_latency_s"] for c in model_calls)
            tool_calls = list(traj.meta.get("tool_calls", []))
            case_wall_time = float(traj.meta.get("case_wall_time_s", 0.0) or 0.0)
            case_tool_call_count = int(traj.meta.get("tool_call_count", len(tool_calls)) or 0)
            case_tool_error_count = int(traj.meta.get("tool_error_count", 0) or 0)
            case_message_count = len(traj.messages)
            case_model_call_count = len(model_calls)

            sample = {
                "case_id": case_id, "sample_index": sample_index,
                **sample_extra,
                "task": task, "method": method,
                "scorer_kind": kind, "question": it["question"][:500],
                "final_answer": pred,
                "num_model_calls": case_model_call_count,
                "num_messages": case_message_count,
                "sum_input_total_positions": case_input_positions,
                "sum_input_text_tokens": case_text_input_tokens,
                "sum_input_latent_positions": case_latent_prefix_positions,
                "sum_output_text_tokens": case_output_tokens,
                "sum_model_latency_s": round(case_generation_latency, 3),
                "case_wall_time_s": round(case_wall_time, 3),
                "num_tool_calls": case_tool_call_count,
                "num_tool_errors": case_tool_error_count,
                "model_call_count": case_model_call_count,
                "message_count": case_message_count,
                "tool_call_count": case_tool_call_count,
                "tool_error_count": case_tool_error_count,
                "input_positions_total": case_input_positions,
                "text_input_tokens_total": case_text_input_tokens,
                "latent_prefix_positions_total": case_latent_prefix_positions,
                "output_tokens_total": case_output_tokens,
                "model_generation_latency_s_total": round(case_generation_latency, 3),
                "workspace": traj.meta.get("workspace"),
                "copied_files": traj.meta.get("copied_files", []),
                "tool_calls": tool_calls,
                "model_calls": model_calls,
                # Backward-compatible aliases for older result readers.
                "n_messages": case_message_count,
                "routing_trace": model_calls,
                "cost_prompt_pos": case_input_positions,
                "gen_tokens": case_output_tokens,
                "latency_s": round(case_generation_latency, 3)}
            predictions.append(sample)
            with open(predictions_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            ctx.log_span(
                "case_end",
                case_order=i + 1,
                num_cases=len(data),
                k_index=k,
                status="ok",
                final_answer=pred,
                num_model_calls=case_model_call_count,
                num_messages=case_message_count,
                num_tool_calls=case_tool_call_count,
                num_tool_errors=case_tool_error_count,
                case_wall_time_s=round(case_wall_time or (time.time() - case_t0), 3),
            )
            print(f"  [{i + 1}/{len(data)}{k_tag}] msgs={case_message_count} "
                  f"input_pos={case_input_positions} ans={pred[:60]!r}", flush=True)

    skipped_total = sum(len(ks) for ks in completed_samples.values())
    print(f"[done] {task}/{run_label}: predictions={len(predictions)} "
          f"samples_per_case={k_samples} skipped={skipped_total} -> {out_dir}", flush=True)
    span_logger.log("run_end", num_predictions=len(predictions),
                    samples_per_case=k_samples,
                    skipped_predictions=skipped_total, out_dir=out_dir)
    return {"out_dir": out_dir, "num_predictions": len(predictions)}


def main() -> None:
    ap = argparse.ArgumentParser(description="LycheeMAS 真实 MAS 推理驱动（新框架版）")
    ap.add_argument("--config", required=True, help="YAML 配置（backend/memory/router/run/eval）")
    ap.add_argument("--runtime", default=None, choices=("autogen",),
                    help="覆盖 runtime.name；工具能力也走 autogen 主链路")
    ap.add_argument("--task", default=None, help="覆盖 run.task")
    ap.add_argument("--method", default=None, help="none|nl_only|latent_only|both")
    ap.add_argument("--backend", default=None, choices=("hf", "api"), help="覆盖 backend.provider")
    ap.add_argument("--team", default=None, help="显式覆盖 YAML/task_config.py 默认 team")
    ap.add_argument("--n", default=None, help="样本数；all/0=全量")
    ap.add_argument("--start-index", dest="start_index", type=int, default=None,
                    help="从 loader 返回列表的第几个样本开始跑，0-based")
    ap.add_argument("--P", type=int, default=None, help="latent prefix 长度")
    ap.add_argument("--c2c-ckpt", dest="c2c_ckpt", default=None,
                    help="latent_strategy=c2c 时的 projector 栈 ckpt 路径（覆盖 memory.c2c_ckpt）")
    ap.add_argument("--samples", type=int, default=None,
                    help="每题采样次数 K（pass@K；K>1 建议 backend.do_sample=true；打分侧按 case 聚合）")
    ap.add_argument("--model-path", dest="model_path", default=None)
    ap.add_argument("--model-tag", dest="model_tag", default=None)
    ap.add_argument("--api-model", dest="api_model", default=None, help="OpenAI-compatible model name")
    ap.add_argument("--api-base-url", dest="api_base_url", default=None)
    ap.add_argument("--api-key-env", dest="api_key_env", default=None)
    ap.add_argument("--device", default=None, help="覆盖 backend.device（本地 HF 路径）")
    ap.add_argument("--max-rounds", dest="max_rounds", type=int, default=None,
                    help="覆盖 run.max_rounds")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None,
                    help="覆盖 run.max_new_tokens（每次模型调用最多生成多少 token）")
    ap.add_argument("--max-turns", dest="max_turns", type=int, default=None,
                    help="覆盖 runtime.max_turns（AutoGen group chat 最大发言上限）")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--code-executor", dest="code_executor", default=None,
                    choices=("docker", "local"), help="需要执行代码时使用的执行器")
    ap.add_argument("--docker-image", dest="docker_image", default=None,
                    help="覆盖 runtime.docker_image（docker executor）")
    ap.add_argument("--code-timeout", dest="code_timeout", type=int, default=None,
                    help="覆盖 runtime.code_timeout")
    ap.add_argument("--work-root", dest="work_root", default=None,
                    help="每个 case 的工具/附件 workspace 根目录")
    ap.add_argument("--trace-model-calls", dest="trace_model_calls", action="store_true",
                    default=None,
                    help="打印每次 LLM 调用的 start/done 摘要；默认开启")
    ap.add_argument("--no-trace-model-calls", dest="trace_model_calls", action="store_false",
                    help="关闭每次 LLM 调用的 start/done 摘要")
    ap.add_argument("--runs-root", dest="runs_root", default=None,
                    help="benchmark run result root; preferred over --results-root")
    ap.add_argument("--results-root", dest="results_root", default=None,
                    help="deprecated compatibility alias for --runs-root")
    ap.add_argument("--resume", action="store_true",
                    help="Resume this run directory: append log/spans, keep successful predictions, retry failed/missing cases")
    args = ap.parse_args()
    asyncio.run(run_one(_load_yaml(args.config), args))


if __name__ == "__main__":
    main()
