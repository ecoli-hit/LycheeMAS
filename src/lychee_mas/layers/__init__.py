"""五层变换（CLAUDE.md §0）：L1 Construct / L2 Prune / L3 Memory / L4 Aggregate / L5
Attribute-and-Train。

import 本包会依次 import 各层（触发其组件注册），但均不触发 torch/autogen（惰性导入约定）。
"""
from __future__ import annotations

from . import (
    aggregate,
    attribute_train,
    construct,
    memory,
    prune,
)

__all__ = [
    "construct",
    "prune",
    "memory",
    "aggregate",
    "attribute_train",
]
