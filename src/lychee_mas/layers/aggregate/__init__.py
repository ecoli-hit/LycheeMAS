"""L4 多轨迹聚合与融合（Aggregate，CLAUDE.md §0）。

- TrajectoryAggregator  aggregator/{self_consistency(可跑), dynamicagg(桩)}

import 本包触发聚合器注册（纯标准库）。
"""
from __future__ import annotations

from .aggregators import DynamicAggregator, SelfConsistencyVote
from .base import TrajectoryAggregator

__all__ = ["TrajectoryAggregator", "SelfConsistencyVote", "DynamicAggregator"]
