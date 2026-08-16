"""多智能体 Team 构建（Construct，CLAUDE.md §0）。

- AgentSelector       团队组建（agent_selector/agentinit，桩）
- RoleProfileTeamBuilder 把 RoleProfile 实例化为显式 AutoGen GroupChat Team

import 本包触发上述组件注册（不触发 torch/autogen）。
"""
from __future__ import annotations

from .base import AgentSelector
from .selectors import AgentInitSelector
from .templates import (
    ROLE_PROFILE_META,
    ROLE_PROFILES,
    ROLE_SYSTEM,
    Role,
    RoleProfileTeamBuilder,
    team_to_agentspecs,
)

__all__ = [
    "AgentSelector",
    "AgentInitSelector",
    "Role",
    "ROLE_SYSTEM",
    "ROLE_PROFILES",
    "ROLE_PROFILE_META",
    "RoleProfileTeamBuilder",
    "team_to_agentspecs",
]
