"""Runtime 抽象（隔离 AutoGen，CLAUDE.md §8）。

`runtime/base.py` 定义最小 `Runtime` 协议；所有运行时能力（run/intercept/spawn/route）只能通过它使
用。
唯一允许 `import autogen_*` 的位置是 `runtime/backends/autogen_*.py`——这隔离了 AutoGen 维护风险并
预留 MAF 迁移。

本文件还提供轻量的 `MASTeam`/`MASGraph` 容器（节点=AgentSpec 列表，边=简单邻接/顺序），供 mock 与
autogen 后端共用。静态团队先用顺序/轮转即可，不必实现完整图。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from ..core.types import AgentSpec, Message, TaskQuery, Trajectory


@dataclass
class MASGraph:
    """MAS 有向图 G 的轻量容器：节点=AgentSpec，边=邻接表（role->下游 roles）。

    边为空时默认按 `nodes` 顺序链式串联（manager->worker->verifier ...）。
    这是 P0 阶段的最小图；完整的 W/T/M（边权/时序/记忆）与 apply_mask/to_runtime_team 后续在
    `core/graph.py` 补全（CLAUDE.md §4）。
    """

    nodes: list[AgentSpec] = field(default_factory=list)
    edges: dict[str, list[str]] = field(default_factory=dict)
    rounds: int = 1
    meta: dict = field(default_factory=dict)

    def order(self) -> list[AgentSpec]:
        """按拓扑/声明顺序返回节点；无显式边时即声明顺序（顺序链）。"""
        return list(self.nodes)

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.nodes]


# MASTeam 当前等价于「按顺序执行的 MASGraph」；保留独立别名以便后端按团队语义消费。
MASTeam = MASGraph


@runtime_checkable
class Runtime(Protocol):
    """运行时协议（最小集）。

    - run(team, query) -> Trajectory：把图/团队跑一遍，产出执行轨迹 τ。
    - intercept(hook)：消息级拦截——把每条 Message 喂给 hook（写 TraceStore + L3 记忆抽取）。
    """

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory: ...

    def intercept(self, hook: Callable[[Message], None]) -> None: ...


class BaseRuntime:
    """可选基类：实现 intercept 的样板（维护 hook 列表 + 逐消息回调）。后端可继承复用。"""

    def __init__(self) -> None:
        self._hooks: list[Callable[[Message], None]] = []

    def intercept(self, hook: Callable[[Message], None]) -> None:
        self._hooks.append(hook)

    def _emit(self, message: Message) -> None:
        # 逐条 Message 回调所有 hook（CLAUDE.md §8：L3/L4/L5 的数据来源）
        for hook in self._hooks:
            hook(message)

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:  # pragma: no cover
        raise NotImplementedError

    def intercept_or_none(self) -> Optional[Callable]:  # pragma: no cover - 兼容占位
        return self._hooks[-1] if self._hooks else None
