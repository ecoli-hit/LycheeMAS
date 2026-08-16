"""构建层协议（CLAUDE.md §5）。

- AgentSelector       团队组建（多样性×专长 Pareto）：从候选池选出一组 AgentSpec。

实现一个新组件 = 实现协议 + `@REGISTRY.register(category, name)` + 加 config + 加 test。
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ...core.types import AgentSpec, Budget, TaskQuery


@runtime_checkable
class AgentSelector(Protocol):
    """团队组建：给定任务/预算，从候选池选出一组智能体画像。"""

    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]: ...
