"""Train —— 强化学习/提示优化训练协议（从 attribute_train 拆出；CLAUDE.md §5/§10）。

本包 = 「写」参数/提示的一侧：Trainer 把信用（来自姊妹包 `lychee_mas.trace` 的 CreditAssigner）
与轨迹喂给优化过程，产出新参数/提示。先 maspo（提示级，无权重更新）；topology_rl / marl 留接口。

RL 库放 optional extra `[train]`，只在 trainer/* 实现里依赖，不污染基座（黄金法则 2/10）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # 仅类型注解需要 Attribution（来自 trace）；避免运行期跨包硬依赖
    from ..trace.base import Attribution


@runtime_checkable
class Trainer(Protocol):
    """训练：把信用与轨迹喂给优化过程，产出新参数/提示（Params）。"""

    def credits(self, attrs: list[Attribution], reward: float) -> dict[str, float]: ...

    def train(self, generator, policies, mem_policies, traces, credits) -> Any: ...
