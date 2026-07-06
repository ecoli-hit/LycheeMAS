"""实验入口（CLI + 落盘）—— 迁移自旧 run_mas.py 的「编排循环之外」部分（CLAUDE.md §7/§12）。

编排循环在 `lychee_mas.pipeline.Orchestrator`（Runtime + REGISTRY，默认 runtime=mock 离线可跑）；
本脚本只做：解析命令行 -> 组装组件 -> 跑样本 -> 评分 -> 汇总指标 -> 落盘 config 快照 + git SHA。

离线快速自检（零重依赖、无需 API）：
  PYTHONPATH=src python scripts/run_experiment.py --runtime mock --team default --n 3 \
      --questions "2 plus 2 is 4" "answer is 7"

真实跑分（需 extras [all]：autogen + torch + datasets ...）：
  PYTHONPATH=src python scripts/run_experiment.py --runtime autogen --benchmark gsm8k --n 20
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from typing import List, Optional

# 让 `python scripts/run_experiment.py` 直接可用（把 src/ 加进路径）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas import REGISTRY  # noqa: E402
from lychee_mas.core.types import TaskQuery, Trajectory  # noqa: E402
from lychee_mas.pipeline import Orchestrator  # noqa: E402
from lychee_mas.stores import TraceStore  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _load_questions(args) -> List[TaskQuery]:
    """构造待跑查询：优先 --questions（离线自检）；否则从 benchmark 惰性加载（需数据/依赖）。"""
    if args.questions:
        return [TaskQuery(question=q, gold=None) for q in args.questions]
    if args.benchmark:
        bench = REGISTRY.create("benchmark", args.benchmark, n=args.n)
        items = bench.load()
        if args.n:
            items = items[: args.n]
        return [TaskQuery(question=it["question"], context=it.get("context"),
                          gold=it.get("gold"), meta={"kind": it.get("kind")})
                for it in items]
    # 兜底：一个内置示例查询，保证脚本始终可跑（离线）
    return [TaskQuery(question="If a train travels 60 miles in 1 hour, speed? Answer: 60",
                      gold="60")]


def _score(query: TaskQuery, trajectory: Trajectory) -> Optional[float]:
    """有 gold + kind 时按 eval.metrics 评分；否则返回 None（离线自检无 gold）。"""
    kind = query.meta.get("kind")
    if query.gold is None or not kind:
        return None
    from lychee_mas.eval.metrics import score

    pred = trajectory.final_answer.content if trajectory.final_answer else ""
    try:
        return score(kind, pred, query.gold)
    except Exception:
        return None


def _selector_kwargs(args) -> Optional[dict]:
    """从 CLI 组装 selector 构造参数。mode 只此一个来源（--selector-mode），无歧义。"""
    if not args.selector:
        return None
    kw = {"mode": args.selector_mode, "critique_rounds": args.selector_critique_rounds}
    if args.selector_embedder:
        kw["embedder_model"] = args.selector_embedder
    if args.selector_min_roles is not None:
        kw["min_roles"] = args.selector_min_roles
    if args.selector_max_roles is not None:
        kw["max_roles"] = args.selector_max_roles
    return kw


async def _run(args) -> dict:
    trace = TraceStore()
    orch = Orchestrator(runtime=args.runtime, team=args.team,
                        aggregator=args.aggregator, trace_store=trace, rounds=args.rounds,
                        selector=args.selector, selector_kwargs=_selector_kwargs(args))

    queries = _load_questions(args)
    samples, q_sum, scored, tok_sum, team_sum = [], 0.0, 0, 0, 0
    for i, q in enumerate(queries):
        trace.reset()
        traj = await orch.run(q)
        # 聚合器开启时 orch.run 返回 Answer；统一取文本
        final = traj.final_answer.content if isinstance(traj, Trajectory) and traj.final_answer \
            else getattr(traj, "content", "")
        s = _score(q, traj) if isinstance(traj, Trajectory) else None
        tok = traj.total_tokens if isinstance(traj, Trajectory) else 0
        # 团队规模 = 该轨迹里出现过的不同发言者数（对齐论文消融的 team size 指标）
        team = len({m.sender for m in traj.messages}) if isinstance(traj, Trajectory) else 0
        tok_sum += tok
        team_sum += team
        if s is not None:
            q_sum += s
            scored += 1
        samples.append({"question": q.question[:300], "final_answer": final,
                        "gold": q.gold, "correct": s, "total_tokens": tok, "team_size": team})
        print(f"  [{i + 1}/{len(queries)}] correct={s} tokens={tok} team={team} ans={final[:50]!r}")

    metrics = {
        "runtime": args.runtime, "team": args.team, "aggregator": args.aggregator,
        "selector": args.selector, "selector_mode": args.selector_mode if args.selector else None,
        "benchmark": args.benchmark, "n": len(queries),
        "quality": round(q_sum / scored, 4) if scored else None,
        "total_tokens_mean": round(tok_sum / len(queries), 1) if queries else 0,
        "team_size_mean": round(team_sum / len(queries), 2) if queries else 0,
        "git_sha": _git_sha(), "seed": args.seed,
    }
    return {"samples": samples, "metrics": metrics}


def _persist(args, result: dict) -> str:
    method = (f"{args.selector}-{args.selector_mode}" if args.selector
              else args.aggregator or args.team)
    bench = args.benchmark or "adhoc"
    from lychee_mas.eval.metrics import result_dir, write_results

    out_dir = result_dir(args.model_tag, method, bench, root=args.results_root)
    snapshot = {"args": vars(args), "registry": REGISTRY.snapshot()}
    write_results(out_dir, result["samples"], result["metrics"], snapshot)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="LycheeMAS 实验入口（默认 runtime=mock 可离线跑）")
    ap.add_argument("--runtime", default="mock", help="runtime 名（mock|autogen|...）")
    ap.add_argument("--team", default="default", help="静态拓扑队伍 profile")
    ap.add_argument("--aggregator", default=None, help="可选聚合器名（如 self_consistency）")
    # --- L1 团队组建 selector（给出则由 selector 决定成员，覆盖 --team 的角色）---
    ap.add_argument("--selector", default=None,
                    help="agent_selector 名（如 agentinit）；不给=沿用 --team 模板")
    ap.add_argument("--selector-mode", dest="selector_mode", default="pool",
                    choices=["pool", "generate"], help="agentinit 模式（mode 的唯一来源）")
    ap.add_argument("--selector-embedder", dest="selector_embedder", default=None,
                    help="generate 模式的句向量编码器路径（覆盖 config）")
    ap.add_argument("--selector-critique-rounds", dest="selector_critique_rounds",
                    type=int, default=3, help="generate 模式 CreateRoles↔Check 迭代上限")
    ap.add_argument("--selector-min-roles", dest="selector_min_roles", type=int, default=None,
                    help="团队规模下界（消融/扫描用；默认走 selector 默认 1）")
    ap.add_argument("--selector-max-roles", dest="selector_max_roles", type=int, default=None,
                    help="团队规模上界（消融/扫描用；默认走 selector 默认 5）")
    ap.add_argument("--benchmark", default=None, help="benchmark 名（gsm8k|aime2024|...）")
    ap.add_argument("--questions", nargs="*", default=None, help="离线自检：直接给若干问题")
    ap.add_argument("--n", type=int, default=5, help="样本数上限")
    ap.add_argument("--rounds", type=int, default=1, help="每轮发言轮数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（可复现）")
    ap.add_argument("--model-tag", dest="model_tag", default="mock", help="落盘目录用的模型标签")
    ap.add_argument("--results-root", dest="results_root", default=os.path.join("runs", "lychee"),
                    help="结果根目录")
    ap.add_argument("--no-save", action="store_true", help="不落盘（仅打印）")
    args = ap.parse_args()

    _sel = f"{args.selector}({args.selector_mode})" if args.selector else None
    print(f"[LycheeMAS] runtime={args.runtime} team={args.team} selector={_sel} "
          f"aggregator={args.aggregator} benchmark={args.benchmark}")
    result = asyncio.run(_run(args))
    print("[metrics]", result["metrics"])
    if not args.no_save:
        out = _persist(args, result)
        print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
