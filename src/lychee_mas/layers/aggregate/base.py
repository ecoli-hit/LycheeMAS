"""L4 聚合层协议（CLAUDE.md §5）。

TrajectoryAggregator：把多条轨迹/候选答案融合成一个 Answer（同构/异构，过程+完成时）。
输入 `list[Trajectory]` 或 `list[Answer]`，输出 `Answer`。

实现一个新聚合器 = 实现协议 + `@REGISTRY.register("aggregator", name)` + 加 config + 加 test。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.types import Answer, Trajectory


@runtime_checkable
class TrajectoryAggregator(Protocol):
    """多轨迹聚合：list[Trajectory] -> Answer。子类可另接受 list[Answer]。"""

    def aggregate(self, trajectories: list[Trajectory]) -> Answer: ...
