"""旧层能力的插件适配器：prune → 运行前插件；trace（归因+信用）→ 运行后插件。

适配器本身只做「取组件 + 调协议 + 校验/写回」；具体算法仍由被包装的
graph_pruner / attributor / credit_assigner 组件提供（桩组件会显式 NotImplementedError，
错误如实向上抛，不静默吞掉）。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from ..core.registry import REGISTRY
from ..core.types import TaskQuery, Trajectory
from ..runtime.base import MASGraph
from .base import RunContext


@REGISTRY.register("pre_run_plugin", "prune")
class PrunePlugin:
    """包装任意已注册 `graph_pruner` 为运行前插件（如 pruner="agentdropout"）。"""

    name = "prune"

    def __init__(self, pruner: str, **pruner_kwargs: Any) -> None:
        self.pruner_name = pruner
        self._pruner = REGISTRY.create("graph_pruner", pruner, **pruner_kwargs)

    def before_run(self, graph: MASGraph, query: TaskQuery, ctx: RunContext) -> MASGraph:
        result = self._pruner.prune(graph, context={"query": query, "run": ctx})
        if not isinstance(result, MASGraph):
            raise TypeError(
                f"graph_pruner/{self.pruner_name}.prune 必须返回 MASGraph，"
                f"得到 {type(result).__name__}")
        return result


@REGISTRY.register("post_run_plugin", "attribution")
class AttributionPlugin:
    """串接 attributor + credit_assigner 为运行后插件：归因 → 信用 → 写回轨迹/TraceStore。"""

    name = "attribution"

    def __init__(self, attributor: str = "all_at_once",
                 credit_assigner: str = "attribution_guided",
                 **attributor_kwargs: Any) -> None:
        self.attributor_name = attributor
        self.credit_assigner_name = credit_assigner
        self._attributor = REGISTRY.create("attributor", attributor, **attributor_kwargs)
        self._assigner = REGISTRY.create("credit_assigner", credit_assigner)

    def after_run(self, trajectory: Trajectory, score: Optional[float],
                  ctx: RunContext) -> None:
        attributions = self._attributor.attribute(trajectory, context=ctx)
        credits = self._assigner.credits(attributions, float(score) if score is not None else 0.0)
        trajectory.meta["attribution"] = [asdict(a) for a in attributions]
        trajectory.meta["credits"] = dict(credits)
        if ctx.trace_store is not None:
            ctx.trace_store.log_decision({
                "type": "attribution",
                "trajectory_id": trajectory.id,
                "task_id": trajectory.task_id,
                "score": score,
                "credits": dict(credits),
            })
