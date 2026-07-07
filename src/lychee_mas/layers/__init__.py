"""层变换（CLAUDE.md §0）：Construct / Prune / Processing。

注：Memory（CDM 主线）与归因训练已提升为顶层子包，不再属于 `layers/`；后者更拆成
`lychee_mas.trace`（归因/信用 + TraceStore）与 `lychee_mas.train`（RL/提示优化）。其注册由
`lychee_mas/__init__.py` 直接 import 触发。原 `aggregate` 层已改名 `processing`，内分
`serial`（跑 1 次）与 `parallel`（并发 K 次 + `aggregator` 聚合）两个子模块（`processor` 类别）。

import 本包会依次 import 各层（触发其组件注册），但均不触发 torch/autogen（惰性导入约定）。
"""
from __future__ import annotations

from . import (
    construct,
    processing,
    prune,
)

__all__ = [
    "construct",
    "prune",
    "processing",
]
