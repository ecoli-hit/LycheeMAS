"""真实 MAS 实验驱动（新框架 lychee_mas 版）—— 把 backend + CDM 记忆 + 固定通道路由 + RoutingContext
+ 静态拓扑 + AutoGenRuntime 接成端到端可跑的实验，逐样本跑分并落盘。

为什么需要本文件：`Orchestrator` 只接 runtime/topology/aggregator，不组装记忆层；
`scripts/run_experiment.py` 是离线自检入口（argparse，无 YAML/记忆）。要在 AIME 上跑「带 CDM
记忆通道」的真实实验，必须自己把 backend（HFBackend，模型只加载一次）、记忆管理器（cdm 双通道）、
路由器（fixed 固定通道，对应 none/nl_only/latent_only/both 四种消融）、RoutingContext（跨 agent
共享状态）、AutoGenRuntime 拼起来。逻辑对齐旧原型 src-bak/LycheeMAS/run_mas.py。

跑法（需 extras：autogen + torch + transformers；数据走 CDM_DATA_ROOT）：
  CDM_DATA_ROOT=/data/mxy/Project/CDM/Data/raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_mas.py --config configs/aime_both.yaml

命令行同名项覆盖 YAML（如 --n 3 / --method latent_only / --results-root <dir>）。
重依赖（torch/autogen）全部惰性导入在 run_one 内部，import 本模块不触发它们（黄金法则 2）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# 让 `python scripts/run_mas.py` 直接可用（把 src/ 加进路径）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# router.method 名 -> fixed_channel_router 的通道值：四种「强制某通道」的消融基线
FIXED = {"none": "none", "nl_only": "nl", "latent_only": "latent", "both": "both"}


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


async def run_one(cfg: dict, args) -> dict:
    import torch
    from lychee_mas.core.types import TaskQuery
    from lychee_mas.eval import metrics as M
    from lychee_mas.eval.benchmarks import load as load_task
    from lychee_mas.eval.task_config import team_name_for_task
    from lychee_mas.layers.construct.templates import StaticTopology
    from lychee_mas.layers.memory.managers.cdm import DualChannelMemoryManager
    from lychee_mas.layers.memory.routing.context import RoutingContext
    from lychee_mas.layers.memory.routing.static import fixed_channel_router
    from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime
    from lychee_mas.runtime.backends.hf_backend import HFBackend

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    # ---- 解析参数（命令行 > YAML）----
    method = args.method or _get(cfg, "router.method", "nl_only")
    if method not in FIXED:
        raise SystemExit(f"本驱动只支持固定通道 method {list(FIXED)}；got {method!r}")
    task = args.task or _get(cfg, "run.task", "aime2024")
    model_path = (args.model_path or _get(cfg, "backend.model_path")
                  or os.environ.get("LYCHEE_HF_MODEL"))
    if not model_path:
        raise SystemExit("需要 backend.model_path 或 --model-path 或环境变量 LYCHEE_HF_MODEL")
    model_tag = args.model_tag or _get(cfg, "backend.model_tag", "model")
    device = _get(cfg, "backend.device", "cuda:0")
    dtype = dtypes.get(_get(cfg, "backend.dtype", "bfloat16"), torch.bfloat16)
    enable_thinking = bool(_get(cfg, "backend.enable_thinking", False))
    do_sample = bool(_get(cfg, "backend.do_sample", False))
    temperature = float(_get(cfg, "backend.temperature", 0.7))
    top_p = float(_get(cfg, "backend.top_p", 0.8))
    seed = int(_get(cfg, "backend.seed", 0))
    P = args.P if args.P is not None else int(_get(cfg, "router.P", 16))
    latent_strategy = _get(cfg, "memory.latent_strategy", "soft_token")
    nl_strategy = _get(cfg, "memory.nl_strategy", "prev_output")
    max_encode_tokens = int(_get(cfg, "memory.max_encode_tokens", 4096))
    include_transcript = bool(_get(cfg, "memory.include_transcript", True))
    raw_n = args.n if args.n is not None else _get(cfg, "run.n", "all")
    n_samples = None if str(raw_n) in ("None", "all", "full", "0") else int(raw_n)
    max_rounds = int(_get(cfg, "run.max_rounds", 2))
    max_new_tokens = int(_get(cfg, "run.max_new_tokens", 4096))
    team_profile = args.team or _get(cfg, "run.team")  # None=按 task 自动选
    results_root = args.results_root or _get(cfg, "eval.results_root", "runs/lychee")

    # ---- backend：模型只加载一次，记忆/各 agent 复用（apples-to-apples）----
    backend = HFBackend(model_path, device=device, dtype=dtype, enable_thinking=enable_thinking,
                        do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed)
    # ---- 记忆方法（CDM 双通道）+ 通道决策（固定通道路由）----
    memory = DualChannelMemoryManager(backend, latent_strategy=latent_strategy,
                                      nl_strategy=nl_strategy,
                                      max_encode_tokens=max_encode_tokens,
                                      include_transcript=include_transcript)
    router = fixed_channel_router(FIXED[method], P=P)
    # ---- 数据 + 拓扑（按 team）----
    data = load_task(task, n=n_samples)
    kind = data[0]["kind"]
    profile = team_profile or team_name_for_task(task)
    graph = StaticTopology(team=profile, model=None, rounds=max_rounds).build()
    # ---- 共享 ctx + runtime（每样本 ctx.reset；同一 backend/graph 复用）----
    ctx = RoutingContext(task=task, router=router, memory=memory, team=team_profile)
    runtime = AutoGenRuntime(backend=backend, ctx=ctx, max_new_tokens=max_new_tokens,
                             max_rounds=max_rounds, model_id=model_tag)
    # 落盘标签：强制队伍时用 "{team}_{method}"（如单模型 baseline=single_none），避免撞目录
    run_label = f"{team_profile}_{method}" if team_profile else method
    print(f"[MAS] task={task} method={method} team={profile} router={router.name} "
          f"memory={memory.name} n={len(data)} kind={kind} P={P} max_new_tokens={max_new_tokens}",
          flush=True)

    samples, q_sum, pos_sum, gen_sum, lat_sum, msg_sum = [], 0.0, 0, 0, 0.0, 0
    for i, it in enumerate(data):
        ctx.reset()  # 清 turn/决策/记忆库
        # 长程记忆任务把对话历史预载进记忆库（AIME 无 context，此处不触发）
        if it.get("context") and hasattr(memory, "seed"):
            memory.seed(it["context"])
        q = TaskQuery(question=it["question"], context=it.get("context"),
                      gold=it["gold"], meta={"kind": kind})
        traj = await runtime.run(graph, q)
        pred = traj.final_answer.content if traj.final_answer else ""
        correct = M.score(kind, pred, it["gold"])
        # 成本三元组：把本样本所有 agent 轮次的决策逐项求和（routing_trace 来自 ctx.decisions）
        decs = traj.meta.get("decisions", [])
        item_pos = sum(int(d.get("prompt_pos", 0)) for d in decs)
        item_gen = sum(int(d.get("gen_tokens", 0)) for d in decs)
        item_lat = sum(float(d.get("latency_s", 0.0)) for d in decs)
        n_msgs = len(traj.messages)
        q_sum += correct
        pos_sum += item_pos
        gen_sum += item_gen
        lat_sum += item_lat
        msg_sum += n_msgs
        samples.append({
            "question": it["question"][:500], "method": method,
            "final_answer": pred, "gold": it["gold"], "correct": correct,
            "n_messages": n_msgs, "routing_trace": decs,
            "cost_prompt_pos": item_pos, "gen_tokens": item_gen,
            "latency_s": round(item_lat, 3)})
        print(f"  [{i + 1}/{len(data)}] correct={correct:.2f} msgs={n_msgs} "
              f"pos={item_pos} ans={pred[:60]!r}", flush=True)

    n = len(data)
    metrics = {
        "model": model_tag, "method": method, "team": team_profile, "task": task, "probe": "mas",
        "memory": memory.name, "router": router.name, "n": n,
        "quality": round(q_sum / n, 4) if n else None, "quality_metric": kind,
        "cost_prompt_pos_mean": round(pos_sum / n, 1) if n else 0,
        "gen_tokens_mean": round(gen_sum / n, 1) if n else 0,
        "latency_s_mean": round(lat_sum / n, 3) if n else 0,
        "messages_mean": round(msg_sum / n, 2) if n else 0}
    out_dir = M.result_dir(model_tag, run_label, task, root=results_root)
    snapshot = {"config_file": args.config, "config": cfg,
                "resolved": {"task": task, "method": method, "team": profile,
                             "P": P, "n": n, "model_path": model_path,
                             "max_new_tokens": max_new_tokens, "max_rounds": max_rounds}}
    M.write_results(out_dir, samples, metrics, snapshot)
    print(f"[done] {task}/{run_label}: q={metrics['quality']} "
          f"cost={metrics['cost_prompt_pos_mean']}pos -> {out_dir}", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="LycheeMAS 真实 AIME/MAS 实验驱动（新框架版）")
    ap.add_argument("--config", required=True, help="YAML 配置（backend/memory/router/run/eval）")
    ap.add_argument("--task", default=None, help="覆盖 run.task")
    ap.add_argument("--method", default=None, help="none|nl_only|latent_only|both")
    ap.add_argument("--team", default=None, help="覆盖 run.team（如 single）")
    ap.add_argument("--n", default=None, help="样本数；all/0=全量")
    ap.add_argument("--P", type=int, default=None, help="latent prefix 长度")
    ap.add_argument("--model-path", dest="model_path", default=None)
    ap.add_argument("--model-tag", dest="model_tag", default=None)
    ap.add_argument("--results-root", dest="results_root", default=None)
    args = ap.parse_args()
    asyncio.run(run_one(_load_yaml(args.config), args))


if __name__ == "__main__":
    main()
