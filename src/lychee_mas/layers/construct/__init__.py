"""L1 多智能体网络构建（Construct，CLAUDE.md §0）。

- AgentSelector       团队组建（agent_selector/agentinit，桩）
- TopologyGenerator   静态/动态图（topology_generator/static，已实现按 team 产 AgentSpec）

import 本包触发上述组件注册（不触发 torch/autogen）。
"""
from __future__ import annotations

from .base import AgentSelector, TopologyGenerator
from .selectors import AgentInitSelector
from .templates import (
    ROLE_SYSTEM,
    TEAMS,
    Role,
    StaticTopology,
    team_to_agentspecs,
)

__all__ = [
    "AgentSelector",
    "TopologyGenerator",
    "AgentInitSelector",
    "Role",
    "ROLE_SYSTEM",
    "TEAMS",
    "StaticTopology",
    "team_to_agentspecs",
]
