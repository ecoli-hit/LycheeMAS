"""MemoryManager = 记忆「方法」接缝（CLAUDE.md §5，L3 主战场）。

一个 manager 负责：存储 + 召回 + 各通道如何被物化：
  - observe(messages): 从目前的对话历史更新内部记忆库
  - recall(decision, query): 按路由器选的通道 + 成本旋钮，产出一个 MemoryBundle
    （nl_text 和/或 latent_prefix）
  - reset(): 清空单次对话的状态

要加一种新记忆方法（mem0 / AMA / LatentMem），继承 MemoryManager +
`@REGISTRY.register("memory_manager", name)`。
注入 client 与 MAS 消费的是 MemoryBundle，不需任何改动（接缝隔离）。

⚠️ 惰性导入：本文件不在 import 时触发 torch；`latent_prefix` 类型注解用字符串前向引用，
torch 张量只在运行期出现（黄金法则 4：核心骨架零运行依赖）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .routing.base import RouteDecision


@dataclass
class MemoryBundle:
    """本轮要注入的东西。按通道不同，两个字段可各自为 None（none 通道则都为 None）。"""

    nl_text: Optional[str] = None  # 注入进 prompt 的文本（NL 通道）
    # (1,P,H) 的 embedding 层 prefix（latent 通道，torch.Tensor）
    latent_prefix: Optional[Any] = None
    meta: dict = field(default_factory=dict)  # 附带元信息（如所选 channel、是否空 source）

    @property
    def prefix_len(self) -> int:
        # latent prefix 的位置数 P（无 latent 则为 0）；用于成本记账
        return 0 if self.latent_prefix is None else int(self.latent_prefix.shape[1])


class MemoryManager(ABC):
    name: str = "memory"  # 子类覆盖，用于日志/落盘目录命名

    @abstractmethod
    def observe(self, messages: list[Any]) -> None:
        """从对话历史（role/content 字典列表）更新记忆库。每轮 create() 调一次。"""

    @abstractmethod
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle:
        """把路由器选定的通道物化成 MemoryBundle（真正决定注入什么）。"""

    def reset(self) -> None:
        pass  # 默认无状态；有状态的子类（如 DualChannel）覆盖它
