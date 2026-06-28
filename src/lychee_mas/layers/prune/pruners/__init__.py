"""L2 图剪枝（GraphPruner 接缝）。

注册占位实现（统一报错文案 `<name>: not wired yet (TODO)`），但能被 REGISTRY.list 看到：
  graph_pruner/agentdropout      逐轮邻接淘汰（本组已发表）
  graph_pruner/agentdropout_v2   运行时在线淘汰（在研）
  graph_pruner/agentprune        基线（AgentPrune / AGP 对标）

迁移逻辑时：实现 prune(graph, context) -> 剪枝后图，并保留原始引用/许可证说明。
"""
from __future__ import annotations

from typing import Any

from ....core.registry import REGISTRY


class _StubPruner:
    name = "pruner"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def prune(self, graph: Any, context: Any = None) -> Any:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("graph_pruner", "agentdropout")
class AgentDropout(_StubPruner):
    name = "agentdropout"  # 逐轮邻接淘汰（已发表）


@REGISTRY.register("graph_pruner", "agentdropout_v2")
class AgentDropoutV2(_StubPruner):
    name = "agentdropout_v2"  # 运行时在线淘汰（在研）


@REGISTRY.register("graph_pruner", "agentprune")
class AgentPrune(_StubPruner):
    name = "agentprune"  # 基线


__all__ = ["AgentDropout", "AgentDropoutV2", "AgentPrune"]
