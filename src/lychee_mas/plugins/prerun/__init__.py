"""prerun —— LangGraph 原生「运行前优化」接缝（REGISTRY 类别 ``pre_run_optimizer``）。

- base.py         PreRunOptimizer 协议 + optimize_langgraph 统一入口（method 按名分发）
- graphview.py    GraphView / extract_view / rebuild（StateGraph 读写的唯一通道 + 节点契约）
- agentprune_lg.py  pre_run_optimizer/agentprune（复用 graph_pruner/agentprune）
- maspo/          pre_run_optimizer/maspo（MASPO 联合提示优化，ICML 2026）

import 本包触发上述注册（纯标准库；langgraph 只在函数内部惰性导入）。
"""
from __future__ import annotations

from . import agentprune_lg  # noqa: F401  触发 pre_run_optimizer/agentprune 注册
from .base import PreRunOptimizer, optimize_langgraph
from .graphview import GraphView, extract_view, rebuild
from .maspo import MASPOOptimizer  # noqa: F401  触发 pre_run_optimizer/maspo 注册

__all__ = [
    "PreRunOptimizer",
    "optimize_langgraph",
    "GraphView",
    "extract_view",
    "rebuild",
    "MASPOOptimizer",
]
