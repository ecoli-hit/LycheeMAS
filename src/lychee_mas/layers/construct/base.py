"""构建层协议（CLAUDE.md §5）。

- AgentSelector       团队组建（多样性×专长 Pareto）：从候选池选出一组 AgentSpec。
- TopologyGenerator   静态/动态图：产出节点（AgentSpec 列表）+ 边（邻接），封装成 MASGraph。

实现一个新组件 = 实现协议 + `@REGISTRY.register(category, name)` + 加 config + 加 test。
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ...core.types import AgentSpec, Budget, TaskQuery


@runtime_checkable
class AgentSelector(Protocol):
    """团队组建：给定任务/预算，从候选池选出一组智能体画像。"""

    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]: ...


@runtime_checkable
class TopologyGenerator(Protocol):
    """拓扑生成：产出一张 MASGraph（节点 + 边 + 轮数）。返回类型用字符串前向引用避免环依赖。"""

    def build(self, agents: list[AgentSpec], query: TaskQuery) -> "object": ...
