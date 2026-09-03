"""Train —— 强化学习训练（从 attribute_train 拆出，CLAUDE.md §0）。

「写」参数/提示的一侧。RL 训练器（topology_rl / marl 等）接 RL 库时在此注册；
提示级优化不在本包：离线 compile 走 `optimizer/gepa`，运行前联合提示优化走
`pre_run_optimizer/maspo`（`plugins/prerun/maspo/`，原 trainer/maspo 桩已迁出）。
import 本包触发注册（不触发 torch/RL 库）。
"""
from __future__ import annotations

from .base import Trainer

__all__ = ["Trainer"]
