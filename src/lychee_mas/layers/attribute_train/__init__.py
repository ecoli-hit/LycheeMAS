"""L5 错误归因 + 强化学习训练（Attribute-and-Train，CLAUDE.md §0）。

注册占位实现（统一报错文案 `<name>: not wired yet (TODO)`），但能被 REGISTRY.list 看到：
  attributor/{all_at_once, step_by_step, binary_search}   错误归因（Who&When / MAST 对标）
  credit_assigner/attribution_guided                       归因引导信用分配（核心贡献）
  trainer/maspo                                            联合提示优化（已发表，廉价基线/暖启动）

topology_rl / marl 先留 stub（CLAUDE.md §10）——本文件先登 maspo；RL 训练器接 RL 库时再加。
import 本包触发上述注册（不触发 torch/RL 库）。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY
from ...core.types import Trajectory
from .base import (
    Attribution,
    CreditAssigner,
    FailureAttributor,
    Trainer,
)


class _StubAttributor:
    name = "attributor"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("attributor", "all_at_once")
class AllAtOnceAttributor(_StubAttributor):
    name = "all_at_once"


@REGISTRY.register("attributor", "step_by_step")
class StepByStepAttributor(_StubAttributor):
    name = "step_by_step"


@REGISTRY.register("attributor", "binary_search")
class BinarySearchAttributor(_StubAttributor):
    name = "binary_search"


@REGISTRY.register("credit_assigner", "attribution_guided")
class AttributionGuidedCredit:
    """归因引导信用分配（核心贡献，桩）：把 L5 归因转为 per-agent 稠密信用喂给 train。"""

    name = "attribution_guided"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def credits(self, attributions: list[Attribution], reward: float) -> dict[str, float]:
        raise NotImplementedError("attribution_guided: not wired yet (TODO)")


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


__all__ = [
    "FailureAttributor",
    "CreditAssigner",
    "Trainer",
    "Attribution",
    "AllAtOnceAttributor",
    "StepByStepAttributor",
    "BinarySearchAttributor",
    "AttributionGuidedCredit",
    "MASPOTrainer",
]
