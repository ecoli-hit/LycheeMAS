"""AgentInit 消融 runner —— 跑对照矩阵并汇总成表（对应 01-agentinit.md §10 / 论文 Table 4）。

复用 `run_experiment._run`（同一套建图/评分/记账），逐配置跑 N 样本，收集 quality/token/team_size，
打印对照表并落盘 JSON。

离线可跑（`--runtime mock`）：baseline/pool/max_roles 扫描都确定性、秒级出——但 mock 无 gold，
`quality` 为 None，真实 acc 需 `--runtime autogen --benchmark mmlu` + 真实模型。
`generate` 配置需 env（LYCHEE_LLM_* + 编码器）；未配则**自动跳过并打印说明**，不报错。

用法：
  # 离线冒烟（确认矩阵能跑、看 team_size/token 对比）
  PYTHONPATH=src python scripts/ablation_agentinit.py --runtime mock --questions "6 times 7 is 42"
  # 真实（对齐论文：Qwen2.5-72B / Deepseek-V3 backbone + all-MiniLM-L6-v2 编码器）
  LYCHEE_LLM_MODEL=... LYCHEE_LLM_API_KEY=... LYCHEE_EMBED_MODEL=/path/to/all-MiniLM-L6-v2 \
      PYTHONPATH=src python scripts/ablation_agentinit.py --runtime autogen --benchmark mmlu --n 20

⚠️ 尚未覆盖的论文消融维度（需先给 selector 加能力，见文末 TODO）：
  - Random / Pareto-Worst / Global-Worst 选择策略（当前 selector 只有默认 Pareto-Best）
  - 关 diversity / 关 expertise 的单目标（selector 收了 objectives 但暂未按它裁剪）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 让 import run_experiment 可用

import run_experiment as RE  # noqa: E402

# 配置矩阵：(名称, 覆盖项)。覆盖项合并进 base args。
CONFIGS: list[tuple[str, dict]] = [
    ("baseline_no_selector", dict(selector=None)),
    ("agentinit_pool", dict(selector="agentinit", selector_mode="pool")),
    ("pool_maxroles3", dict(selector="agentinit", selector_mode="pool", selector_max_roles=3)),
    ("pool_maxroles7", dict(selector="agentinit", selector_mode="pool", selector_max_roles=7)),
    ("agentinit_generate", dict(selector="agentinit", selector_mode="generate")),  # 需 env
]


def _base_args(cli) -> dict:
    """run_experiment._run 需要的完整 args 字段（与其 argparse 默认对齐）。"""
    return dict(
        runtime=cli.runtime, team="default", aggregator=None, benchmark=cli.benchmark,
        questions=cli.questions, n=cli.n, rounds=cli.rounds, seed=cli.seed,
        model_tag=cli.model_tag, results_root=cli.results_root, no_save=True,
        selector=None, selector_mode="pool", selector_embedder=cli.selector_embedder,
        selector_critique_rounds=cli.selector_critique_rounds,
        selector_min_roles=None, selector_max_roles=None,
    )


def _needs_llm_env(overrides: dict) -> bool:
    return overrides.get("selector_mode") == "generate"


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentInit 消融矩阵 runner")
    ap.add_argument("--runtime", default="mock")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--questions", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selector-embedder", dest="selector_embedder", default=None)
    ap.add_argument("--selector-critique-rounds", dest="selector_critique_rounds",
                    type=int, default=3)
    ap.add_argument("--model-tag", dest="model_tag", default="ablation")
    ap.add_argument("--results-root", dest="results_root",
                    default=os.path.join("runs", "ablation"))
    ap.add_argument("--out", default=os.path.join("runs", "ablation", "agentinit_ablation.json"))
    cli = ap.parse_args()

    has_llm = bool(os.getenv("LYCHEE_LLM_API_KEY"))
    rows = []
    for name, overrides in CONFIGS:
        if _needs_llm_env(overrides) and not has_llm:
            print(f"[skip] {name}: 需 env LYCHEE_LLM_*（generate 模式），未配置 -> 跳过")
            continue
        args = argparse.Namespace(**{**_base_args(cli), **overrides})
        print(f"\n=== [{name}] selector={args.selector} mode={args.selector_mode} "
              f"maxroles={args.selector_max_roles} ===")
        result = asyncio.run(RE._run(args))
        m = result["metrics"]
        rows.append({"config": name, **{k: m[k] for k in
                     ("quality", "total_tokens_mean", "team_size_mean", "n")}})

    # 汇总表
    print("\n" + "=" * 72)
    print(f"{'config':<24}{'quality':>10}{'tokens':>12}{'team_size':>12}{'n':>6}")
    print("-" * 72)
    for r in rows:
        q = "None" if r["quality"] is None else f"{r['quality']:.4f}"
        print(f"{r['config']:<24}{q:>10}{r['total_tokens_mean']:>12}"
              f"{r['team_size_mean']:>12}{r['n']:>6}")
    print("=" * 72)

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump({"runtime": cli.runtime, "benchmark": cli.benchmark, "rows": rows}, f, indent=2)
    print(f"[done] -> {cli.out}")


if __name__ == "__main__":
    main()
