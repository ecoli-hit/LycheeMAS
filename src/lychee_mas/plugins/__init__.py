"""插件系统（顶层包）—— 运行前 / 运行后 / 离线优化三接缝。

- base.py     三协议（PreRunPlugin / PostRunPlugin / Optimizer）+ RunContext + Metric
- program.py  MASProgram（系统的可变异文本组件集合，Optimizer 的操作对象）
- adapters.py 旧层适配器：pre_run_plugin/prune、post_run_plugin/attribution
- gepa/       optimizer/gepa（Pareto 选择 + 反思变异 compile 循环）

import 本包触发全部插件注册（纯标准库，不触发重依赖）。
"""
from __future__ import annotations

from . import adapters  # noqa: F401  触发 pre_run_plugin/prune + post_run_plugin/attribution 注册
from .base import Metric, Optimizer, PostRunPlugin, PreRunPlugin, RunContext
from .gepa import GEPAOptimizer  # noqa: F401  触发 optimizer/gepa 注册
from .program import MASProgram

__all__ = [
    "PreRunPlugin",
    "PostRunPlugin",
    "Optimizer",
    "RunContext",
    "Metric",
    "MASProgram",
    "GEPAOptimizer",
]
