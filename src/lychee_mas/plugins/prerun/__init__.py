"""prerun —— 运行前优化接缝（REGISTRY 类别 ``pre_run_optimizer``）。

- base.py         PreRunOptimizer 协议 + optimize_langgraph 统一入口（method 按名分发）
- graphview.py    GraphView / extract_view / rebuild（StateGraph 读写的唯一通道 + 节点契约）
- agentprune_lg.py  pre_run_optimizer/agentprune（薄适配，算法在 methods/prerun/agentprune）
- agentdropout_lg.py  pre_run_optimizer/agentdropout（薄适配，逐轮 apply + dropped 元数据）

方法实现（maspo/ gepa/ agentprune 训练机器）在 ``methods/prerun/``；import methods 包
触发其注册（本包只触发薄适配件）。langgraph 惰性导入。
"""
from __future__ import annotations

from . import agentdropout_lg, agentprune_lg  # noqa: F401  触发 pre_run_optimizer/* 薄适配注册
from .base import PreRunOptimizer, optimize_langgraph
from .graphview import GraphView, extract_view, rebuild

__all__ = [
    "PreRunOptimizer",
    "optimize_langgraph",
    "GraphView",
    "extract_view",
    "rebuild",
]
