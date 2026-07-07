"""团队组建（AgentSelector 接缝）。

注册 `agent_selector/agentinit`（团队组建·多样性×专长 Pareto，本组已发表，当前为接口桩）。
后续把已发表的 AgentInit 选择逻辑迁移进来：实现 select() -> list[AgentSpec]。
"""
from __future__ import annotations

from typing import Optional

from ....core.registry import REGISTRY
from ....core.types import AgentSpec, Budget, TaskQuery


@REGISTRY.register("agent_selector", "agentinit")
class AgentInitSelector:
    """团队组建：多样性×专长 Pareto 选择（桩）。统一报错文案：not wired yet (TODO)。"""

    name = "agentinit"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]:
        raise NotImplementedError("agentinit: not wired yet (TODO)")


__all__ = ["AgentInitSelector"]
