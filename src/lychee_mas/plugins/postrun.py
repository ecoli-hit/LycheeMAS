"""运行后接缝（postrun）—— 第五模块（归因训练）的双入口。

- **读侧** ``analyze_run(trajectory, score, method, ...)``：逐次运行的归因 → 信用，
  写回 ``trajectory.meta``（attributor + credit_assigner，实现/桩在 ``methods/postrun/``）。
- **写侧** ``train_from_runs(method, ...)``：离线消费轨迹与信用信号产训练产物
  （trainer 类别，当前无实现——占名待接 RL 线）；产物统一经 prerun 的 apply 挂载回图。

方法自带训练（MASPO/GEPA/AgentPrune 的 optimize 模式）不走本接缝——训练素材自给的方法，
训练代码跟方法走（methods/prerun/）；素材来自执行轨迹的才在这里。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Protocol, runtime_checkable

from ..core.registry import REGISTRY
from ..core.types import Trajectory
from ..methods.postrun.base import (  # noqa: F401  协议 re-export
    Attribution,
    CreditAssigner,
    FailureAttributor,
)
from ..methods.postrun.store import TraceStore  # noqa: F401


@runtime_checkable
class Trainer(Protocol):
    """离线训练器协议（写侧）：消费信用与轨迹，产出新参数/提示（RL 库只在实现里依赖）。"""

    def train(self, credits: dict[str, float], trajectories: list[Trajectory],
              **kwargs: Any) -> Any: ...


def analyze_run(trajectory: Trajectory, score: Optional[float] = None,
                method: str = "all_at_once", credit_assigner: str = "attribution_guided",
                trace_store: Any = None, **kwargs: Any) -> dict[str, float]:
    """读侧：归因（attributor/<method>）→ 信用（credit_assigner）→ 写回轨迹与 TraceStore。

    桩组件的 NotImplementedError 如实上抛（显式错误原则）。返回 per-agent 信用。
    """
    attributor = REGISTRY.create("attributor", method, **kwargs)
    assigner = REGISTRY.create("credit_assigner", credit_assigner)
    attributions = attributor.attribute(trajectory, context=None)
    credits = assigner.credits(attributions, float(score) if score is not None else 0.0)
    trajectory.meta["attribution"] = [asdict(a) for a in attributions]
    trajectory.meta["credits"] = dict(credits)
    if trace_store is not None:
        trace_store.log_decision({
            "type": "attribution", "trajectory_id": trajectory.id,
            "task_id": trajectory.task_id, "score": score, "credits": dict(credits),
        })
    return dict(credits)


def train_from_runs(method: str, **kwargs: Any) -> Any:
    """写侧：按名取 trainer 执行训练（当前无实现，占名待接 RL 线）。"""
    trainer = REGISTRY.create("trainer", method, **kwargs)  # 未注册 → 显式 KeyError
    return trainer
