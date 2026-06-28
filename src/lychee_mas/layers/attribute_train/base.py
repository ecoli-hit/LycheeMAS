"""L5 错误归因 + 强化学习训练协议（CLAUDE.md §5/§10）。

- FailureAttributor  错误归因：把一条失败轨迹定位到「哪个 agent / 哪一步」出错（Attribution 列表）。
- CreditAssigner     信用分配：把归因转成 per-agent 稠密信用（核心贡献：attribution_guided）。
- Trainer            训练：先 maspo（提示级，无权重更新）；topology_rl/marl 留接口（
NotImplementedError）。

接口先稳定，RL 库以后接（放 optional extra [train]，只在 trainer/* 实现里依赖，不污染基座）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.types import Trajectory


@dataclass
class Attribution:
    """一次归因结果：定位到某 agent / 某步，附理由与置信度。"""

    agent: str = ""
    step: int = -1
    is_fault: bool = False
    reason: str = ""
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FailureAttributor(Protocol):
    """错误归因：给定一条（失败）轨迹，产出 Attribution 列表。"""

    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]: ...


@runtime_checkable
class CreditAssigner(Protocol):
    """信用分配：把归因 + 标量奖励转成 per-agent 信用（agent -> float）。"""

    def credits(self, attributions: list[Attribution], reward: float) -> dict[str, float]: ...


@runtime_checkable
class Trainer(Protocol):
    """训练：把信用与轨迹喂给优化过程，产出新参数/提示（Params）。"""

    def credits(self, attrs: list[Attribution], reward: float) -> dict[str, float]: ...

    def train(self, generator, policies, mem_policies, traces, credits) -> Any: ...
