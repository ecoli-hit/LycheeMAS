"""runtime —— 记忆线兼容层（P3 退役预定；新主线不经此处）。

保留给 scripts/run_mas.py（CDM 记忆线）的执行路径：
  base.py                 MASGraph 容器 + Runtime 协议（也是 agentprune 的 MASGraph 剪枝入参）
  injection.py            记忆注入六步引擎（P3 将改造为 plugins/memory 的节点包裹挂载）
  backends/langgraph_runtime.py   runtime/langgraph 后端

AutoGen 线（autogen_runtime / injection_client）与 mock/vllm 已随五接缝重构退役。
"""
from __future__ import annotations

from . import backends  # noqa: F401  触发 runtime/langgraph 注册
from .base import BaseRuntime, MASGraph, MASTeam, Runtime

__all__ = ["Runtime", "BaseRuntime", "MASGraph", "MASTeam"]
