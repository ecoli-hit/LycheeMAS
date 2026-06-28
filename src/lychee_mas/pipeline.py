"""Orchestrator —— 端到端编排器（CLAUDE.md §0/§12）。

只按 config 从 REGISTRY 取组件，绝不硬编码实现（黄金法则 2：换单一组件即一组对照实验，不改本文件）。
默认 runtime=mock，可完全离线跑通：构建一张静态 MASGraph + 一个 TaskQuery -> Runtime.run ->
Trajectory。

闭环（后续）：在 run 之后接 L4 聚合 / L5 归因→信用→训练→反哺；当前 P0 先打通 construct + runtime。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .core.registry import REGISTRY
from .core.types import Message, TaskQuery, Trajectory
from .runtime.base import MASGraph


class Orchestrator:
    """最小编排器：按名字从 REGISTRY 取 topology + runtime，跑一条 query。

    参数：
      - runtime: 运行时名（REGISTRY "runtime" 类别），默认 "mock"（离线确定性）。
      - team:    静态拓扑的队伍 profile 名（topology_generator/static），默认 "default"。
      - aggregator: 可选聚合器名（REGISTRY "aggregator"）；给出则对 [trajectory] 聚合出最终 Answer。
      - trace_store: 可选 stores.TraceStore；给出则把每条 Message 通过 runtime.intercept 写入。
      - runtime_kwargs: 透传给 runtime 构造（如 autogen 后端的 backend/ctx）。
    """

    def __init__(self, runtime: str = "mock", team: str = "default",
                 aggregator: Optional[str] = None, trace_store: Any = None,
                 rounds: int = 1, model: Optional[str] = None,
                 runtime_kwargs: Optional[dict] = None):
        self.runtime_name = runtime
        self.team = team
        self.aggregator_name = aggregator
        self.trace_store = trace_store
        self.rounds = rounds
        self.model = model
        self.runtime_kwargs = runtime_kwargs or {}

    def build_graph(self) -> MASGraph:
        """用 topology_generator/static 按 team 名构建一张顺序链 MASGraph。"""
        topo = REGISTRY.create("topology_generator", "static",
                               team=self.team, model=self.model, rounds=self.rounds)
        return topo.build()

    async def run(self, query: TaskQuery, hook: Optional[Callable[[Message], None]] = None):
        """端到端：建图 -> 取 runtime -> intercept -> run -> (可选)聚合，返回 Trajectory（或
        Answer）。"""
        graph = self.build_graph()
        runtime = REGISTRY.create("runtime", self.runtime_name, **self.runtime_kwargs)

        # intercept：把每条 Message 喂给 TraceStore / 自定义 hook（CLAUDE.md §8）
        if self.trace_store is not None and hasattr(runtime, "intercept"):
            runtime.intercept(self.trace_store.hook)
        if hook is not None and hasattr(runtime, "intercept"):
            runtime.intercept(hook)

        trajectory: Trajectory = await runtime.run(graph, query)

        if self.aggregator_name:
            agg = REGISTRY.create("aggregator", self.aggregator_name)
            return agg.aggregate([trajectory])
        return trajectory
