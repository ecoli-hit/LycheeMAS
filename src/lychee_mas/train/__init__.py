"""Train —— 强化学习/提示优化训练（从 attribute_train 拆出，CLAUDE.md §0）。

「写」参数/提示的一侧。注册占位实现（`<name>: not wired yet (TODO)`），但能被 REGISTRY.list 看到：
  trainer/maspo    联合提示优化（已发表，廉价基线/暖启动）
topology_rl / marl 先留 stub；RL 训练器接 RL 库时再加。import 本包触发注册（不触发 torch/RL 库）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.registry import REGISTRY
from .base import Trainer

if TYPE_CHECKING:  # Attribution 仅用于注解（来自 trace 包）
    from ..trace.base import Attribution


@REGISTRY.register("trainer", "maspo")
class MASPOTrainer:
    """MASPO 联合提示优化（已发表，桩）：提示级优化，无需权重更新，作为廉价基线/暖启动。"""

    name = "maspo"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def credits(self, attrs: list[Attribution], reward: float) -> dict[str, float]:
        raise NotImplementedError("maspo: not wired yet (TODO)")

    def train(self, generator, policies, mem_policies, traces, credits) -> Any:
        raise NotImplementedError("maspo: not wired yet (TODO)")


__all__ = ["Trainer", "MASPOTrainer"]
