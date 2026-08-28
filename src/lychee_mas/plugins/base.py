"""插件系统三接口 —— 运行前 / 运行后 / 离线优化（GEPA 式）。

把「优化器」挂在执行外部的三种生命周期：

- **PreRunPlugin**（类别 `pre_run_plugin`）：每次执行前变换图 G（如剪枝）。
- **PostRunPlugin**（类别 `post_run_plugin`）：每次执行后消费轨迹 τ（如归因/信用）。
- **Optimizer**（类别 `optimizer`）：离线 compile 循环——反复 rollout + 打分 + 变异，
  迭代改进一个 `MASProgram`（系统的可变异文本组件集合）。

约定（显式错误原则）：`before_run` 必须返回 MASGraph（编排器校验，非法返回直接 raise）；
插件对不支持的输入显式报错，不静默降级。纯标准库、零重依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from ..core.types import TaskQuery, Trajectory
from ..runtime.base import MASGraph
from .program import MASProgram


@dataclass
class RunContext:
    """插件可见的运行上下文（编排器构造并在整条插件链上共享）。"""

    task: str = ""
    trace_store: Any = None  # 可选 trace.TraceStore
    routing_ctx: Any = None  # 可选 memory.RoutingContext（真实执行链路才有）
    backend: Any = None  # 可选生成后端（需要 LLM 的插件用）
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PreRunPlugin(Protocol):
    """运行前插件：每次执行前变换图（必须返回 MASGraph）。"""

    def before_run(self, graph: MASGraph, query: TaskQuery, ctx: RunContext) -> MASGraph: ...


@runtime_checkable
class PostRunPlugin(Protocol):
    """运行后插件：每次执行后消费轨迹（score 可为 None——无度量的普通运行）。"""

    def after_run(self, trajectory: Trajectory, score: Optional[float],
                  ctx: RunContext) -> None: ...


# 评分函数：一条轨迹在一条 query 上的得分（越大越好）
Metric = Callable[[Trajectory, TaskQuery], float]


@runtime_checkable
class Optimizer(Protocol):
    """离线优化器：GEPA 式 compile 循环，返回改进后的 MASProgram。"""

    def optimize(self, system: MASProgram, trainset: Sequence[TaskQuery],
                 metric: Metric) -> MASProgram: ...
