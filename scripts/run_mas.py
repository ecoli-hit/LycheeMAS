"""真实 MAS 实验驱动（新框架 lychee_mas 版）—— 把 backend + CDM 记忆 + 固定通道路由 + RoutingContext
+ 静态拓扑 + AutoGenRuntime 接成端到端可跑的实验，逐样本跑分并落盘。

为什么需要本文件：`Orchestrator` 只接 runtime/topology/aggregator，不组装记忆层；
`scripts/run_experiment.py` 是离线自检入口（argparse，无 YAML/记忆）。要在 AIME 上跑「带 CDM
记忆通道」的真实实验，必须自己把 backend（HFBackend，模型只加载一次）、记忆管理器（cdm 双通道）、
路由器（fixed 固定通道，对应 none/nl_only/latent_only/both 四种消融）、RoutingContext（跨 agent
共享状态）、AutoGenRuntime 拼起来。逻辑对齐旧原型 src-bak/LycheeMAS/run_mas.py。

**跑几次由 处理层决定**：本驱动把「跑一次 MAS 产一条轨迹」封成 runner 交给 processor——
K=1 用 `processor/serial`（跑 1 次），K>1 用 `processor/parallel`（并发跑 K 次 + aggregator 聚合）。
pass@1 仍对每条轨迹单独评分；parallel 的聚合答案在配了 aggregator 时作为 vote_acc 上报。

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
    from lychee_mas.core.registry import REGISTRY
    from lychee_mas.core.types import TaskQuery
    from lychee_mas.eval import metrics as M
    from lychee_mas.eval.benchmarks import load as load_task
    from lychee_mas.eval.task_config import team_name_for_task
    from lychee_mas.layers.construct.templates import StaticTopology
    from lychee_mas.memory.context import RoutingContext
    from lychee_mas.memory.managers.DualChannelMemory import DualChannelMemoryManager
    from lychee_mas.memory.routing.static import fixed_channel_router
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
    max_rounds = int(_get(cfg, "run.max_rounds", 2))
    max_new_tokens = int(_get(cfg, "run.max_new_tokens", 4096))
    team_profile = args.team or _get(cfg, "run.team")  # None=按 task 自动选
    results_root = args.results_root or _get(cfg, "eval.results_root", "runs/lychee")
    n_runs = args.samples if args.samples is not None else int(_get(cfg, "run.samples", 1))
    agg_name = args.aggregator or _get(cfg, "run.aggregator")  # K>1 时多数表决用

    # ---- backend：模型只加载一次，记忆/各 agent 复用（apples-to-apples）----
    backend = HFBackend(model_path, device=device, dtype=dtype, enable_thinking=enable_thinking,
                        do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed)
    # ---- 记忆方法（CDM 双通道）+ 通道决策（固定通道路由）----
    memory = DualChannelMemoryManager(backend, latent_strategy=latent_strategy,
                                      nl_strategy=nl_strategy,
                                      max_encode_tokens=max_encode_tokens,
                                      include_transcript=include_transcript,
                                      P=P, c2c_ckpt=c2c_ckpt, c2c_gate=c2c_gate,
                                      nl_simplemem=nl_simplemem)
    router = fixed_channel_router(FIXED[method])
    # ---- 数据 + 拓扑（按 team）----
    data = load_task(task, n=n_samples)
    kind = data[0]["kind"]
    profile = team_profile or team_name_for_task(task)
    graph = StaticTopology(team=profile, model=None, rounds=max_rounds).build()
    # ---- 共享 ctx + runtime（每次 run 在 runner 内 ctx.reset；同一 backend/graph 复用）----
    ctx = RoutingContext(task=task, router=router, memory=memory, team=team_profile)
    runtime = AutoGenRuntime(backend=backend, ctx=ctx, max_new_tokens=max_new_tokens,
                             max_rounds=max_rounds, model_id=model_tag)
    # 处理层：串/并行由采样数 K 决定——
    #   K=1 → processor/serial   ：跑 1 次、产 1 条轨迹（无聚合）。
    #   K>1 → processor/parallel ：并发跑 K 次、产 K 条轨迹，用 aggregator 聚合出 res.answer。
    # pass@1 仍对**每条轨迹单独评分**（不投票）；res.answer（并行聚合答案）仅当配置了 aggregator
    # 时作为 vote_acc 上报。
    if n_runs > 1:
        processor = REGISTRY.create("processor", "parallel",
                                    k=n_runs, aggregator=agg_name or "self_consistency")
    else:
        processor = REGISTRY.create("processor", "serial")
    if n_runs > 1 and not do_sample:
        print("[warn] samples>1 但 backend.do_sample=false，多次采样会相同；建议开 do_sample",
              flush=True)
    # 共享 ctx/memory/backend（单模型）非并发安全：用锁把每次 reset→run 变成原子临界区
    # （parallel 处理器 gather 下由此串行化；单 GPU 生成本就串行，正确性优先）。
    run_lock = asyncio.Lock()
    # 落盘标签：多采样加 _passK 避免与贪心结果撞目录
    base_label = f"{team_profile}_{method}" if team_profile else method
    run_label = f"{base_label}_pass{n_runs}" if n_runs > 1 else base_label
    print(f"[MAS] task={task} method={method} team={profile} router={router.name} "
          f"memory={memory.name} n={len(data)} kind={kind} P={P} K={n_runs} "
          f"max_new_tokens={max_new_tokens}", flush=True)

    samples = []
    correct_samples = total_samples = passk_hits = pos_sum = gen_sum = msg_sum = 0
    lat_sum = vote_correct = 0.0
    for i, it in enumerate(data):
        q = TaskQuery(question=it["question"], context=it.get("context"),
                      gold=it["gold"], meta={"kind": kind})
        async def _runner(it=it, q=q):
            # 跑一次 MAS 产一条轨迹；reset→run 在锁内原子完成（清 turn/决策/记忆库 + 可选 seed）
            async with run_lock:
                ctx.reset()
                if it.get("context") and hasattr(memory, "seed"):
                    memory.seed(it["context"])  # 长程记忆任务预载（AIME 无 context 不触发）
                return await runtime.run(graph, q)
        # 交给处理层执行：serial 调 1 次 / parallel 并发调 K 次并聚合
        res = await processor.run(_runner)
        trajs = res.trajectories
        per_correct = [M.score(kind, (t.final_answer.content if t.final_answer else ""),
                               it["gold"]) for t in trajs]
        answers = [(t.final_answer.content if t.final_answer else "") for t in trajs]
        n_ok = sum(per_correct)
        correct_samples += n_ok
        total_samples += len(per_correct)
        passk_hits += int(n_ok > 0)
        if agg_name and n_runs > 1:  # 并行聚合答案（res.answer）作为可选投票准确率
            vote_correct += M.score(kind, res.answer.content, it["gold"])
        for t in trajs:  # 成本：K 条轨迹所有 agent 轮次决策求和
            decs = t.meta.get("decisions", [])
            pos_sum += sum(int(d.get("prompt_pos", 0)) for d in decs)
            gen_sum += sum(int(d.get("gen_tokens", 0)) for d in decs)
            lat_sum += sum(float(d.get("latency_s", 0.0)) for d in decs)
            msg_sum += len(t.messages)
        samples.append({
            "question": it["question"][:500], "method": method, "gold": it["gold"],
            "k": n_runs, "answers": answers, "per_sample_correct": per_correct,
            "pass1": round(n_ok / len(per_correct), 4), "any_correct": int(n_ok > 0),
            "routing_trace": trajs[0].meta.get("decisions", [])})
        print(f"  [{i + 1}/{len(data)}] pass@1={n_ok}/{n_runs} any={int(n_ok > 0)} "
              f"gold={it['gold']!r} answers={answers}", flush=True)

    n = len(data)
    ts = total_samples or 1
    metrics = {
        "model": model_tag, "method": method, "team": team_profile, "task": task, "probe": "mas",
        "memory": memory.name, "router": router.name, "n": n, "samples_k": n_runs,
        "total_samples": total_samples,
        "pass@1": round(correct_samples / ts, 4),
        f"pass@{n_runs}": round(passk_hits / n, 4) if n else None,
        "quality": round(correct_samples / ts, 4), "quality_metric": "pass@1",
        "cost_prompt_pos_mean": round(pos_sum / ts, 1),
        "gen_tokens_mean": round(gen_sum / ts, 1),
        "latency_s_mean": round(lat_sum / ts, 3),
        "messages_mean": round(msg_sum / ts, 2)}
    if agg_name and n_runs > 1:
        metrics["vote_acc"] = round(vote_correct / n, 4) if n else None
    out_dir = M.result_dir(model_tag, run_label, task, root=results_root)
    snapshot = {"config_file": args.config, "config": cfg,
                "resolved": {"task": task, "method": method, "team": profile,
                             "P": P, "n": n, "model_path": model_path,
                             "max_new_tokens": max_new_tokens, "max_rounds": max_rounds}}
    M.write_results(out_dir, samples, metrics, snapshot)
    print(f"[done] {task}/{run_label}: pass@1={metrics['pass@1']} "
          f"pass@{n_runs}={metrics.get(f'pass@{n_runs}')} -> {out_dir}", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="LycheeMAS 真实 AIME/MAS 实验驱动（新框架版）")
    ap.add_argument("--config", required=True, help="YAML 配置（backend/memory/router/run/eval）")
    ap.add_argument("--task", default=None, help="覆盖 run.task")
    ap.add_argument("--method", default=None, help="none|nl_only|latent_only|both")
    ap.add_argument("--team", default=None, help="覆盖 run.team（如 single）")
    ap.add_argument("--n", default=None, help="样本数；all/0=全量")
    ap.add_argument("--P", type=int, default=None, help="latent prefix 长度")
    ap.add_argument("--c2c-ckpt", dest="c2c_ckpt", default=None,
                    help="latent_strategy=c2c 的 projector ckpt（覆盖 memory.c2c_ckpt）")
    ap.add_argument("--samples", type=int, default=None, help="每题采样轨迹数 K（自一致性，>1）")
    ap.add_argument("--aggregator", default=None, help="K>1 时聚合器名（如 self_consistency）")
    ap.add_argument("--model-path", dest="model_path", default=None)
    ap.add_argument("--model-tag", dest="model_tag", default=None)
    ap.add_argument("--results-root", dest="results_root", default=None)
    args = ap.parse_args()
    asyncio.run(run_one(_load_yaml(args.config), args))


if __name__ == "__main__":
    main()
