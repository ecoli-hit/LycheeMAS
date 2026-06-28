"""离线端到端示例：mock runtime + 静态团队 + 一个 TaskQuery 跑通 Orchestrator。

零重依赖、纯离线可跑（CLAUDE.md §12 的 P0 验收）：
  PYTHONPATH=src python examples/01_static_chain_e2e.py

打印：最终答案、轨迹概览、TraceStore 收到的消息条数，以及 REGISTRY.snapshot()。
"""
from __future__ import annotations

import asyncio
import pprint

from lychee_mas import REGISTRY
from lychee_mas.core.types import TaskQuery
from lychee_mas.pipeline import Orchestrator
from lychee_mas.stores import TraceStore


async def main() -> None:
    # 一个简单的可验证查询（mock 后端会回显 query 中的数字作为最终答案）
    query = TaskQuery(
        question="If Alice has 3 apples and Bob gives her 5 more, how many apples? Answer: 8",
        gold="8",
    )

    # TraceStore 接到 runtime.intercept 的每条 Message（L3/L4/L5 的数据来源）
    trace = TraceStore()

    # Orchestrator：runtime=mock（离线确定性）+ 静态 default 团队（manager->worker->verifier）
    orch = Orchestrator(runtime="mock", team="default", trace_store=trace, rounds=1)
    trajectory = await orch.run(query)

    print("=== 端到端结果（runtime=mock, team=default）===")
    print("final_answer :", trajectory.final_answer.content if trajectory.final_answer else None)
    print("num messages :", len(trajectory.messages))
    print("num_rounds   :", trajectory.num_rounds)
    print("total_tokens :", trajectory.total_tokens)
    print("trace_store  :", len(trace), "messages intercepted")
    print("transcript   :")
    for m in trajectory.messages:
        print(f"  [{m.round}] {m.sender}: {m.content[:60]}")

    # 顺带演示一个真实可跑的 L4 聚合器（多数投票）
    agg = REGISTRY.create("aggregator", "self_consistency")
    voted = agg.aggregate([trajectory, trajectory])
    print("\n=== L4 self_consistency 多数投票 ===")
    print("voted answer :", voted.content, "| confidence:", round(voted.confidence, 3))

    print("\n=== REGISTRY.snapshot() ===")
    pprint.pprint(REGISTRY.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
