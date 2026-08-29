"""Framework-neutral runtime protocol and compiled Team graph container.

TeamSpec validation lives in ``eval.teams``. ``runtime.coordination`` derives an
adapter-facing execution plan from explicit Nodes, Operations, and Control/Data
Relations.
Framework imports remain inside their adapters. A Runtime executes one Attempt
and returns a framework-neutral Trajectory; Trial retries and final projection
belong to ``eval.runner.TrialExecutor``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from lychee_mas.core.types import AgentSpec, Message, TaskQuery, Trajectory


@dataclass
class MASGraph:
    """Compiled graph passed from the Team layer to one RuntimeAdapter.

    ``nodes`` contains executable runtime Nodes and ``edges`` is only the compiled
    control adjacency used by adapters. The normalized TeamSpec, typed Control/Data
    Edges, Coordination IR, result contract, and binding metadata travel through
    ``meta``. Adapters must never infer data visibility from control adjacency.
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


# Compatibility alias used by callers that name the compiled graph as a team.
MASTeam = MASGraph


@runtime_checkable
class Runtime(Protocol):
    """Minimal Attempt-level adapter protocol shared by all frameworks."""

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory: ...

    def intercept(self, hook: Callable[[Message], None]) -> None: ...


class BaseRuntime:
    """可选基类：实现 intercept 的样板（维护 hook 列表 + 逐消息回调）。后端可继承复用。"""

    def __init__(self) -> None:
        self._hooks: list[Callable[[Message], None]] = []

    def intercept(self, hook: Callable[[Message], None]) -> None:
        self._hooks.append(hook)

    def _emit(self, message: Message) -> None:
        # 逐条 Message 回调所有 hook（CLAUDE.md §8：记忆/处理/训练 的数据来源）
        for hook in self._hooks:
            hook(message)

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:  # pragma: no cover
        raise NotImplementedError

    def intercept_or_none(self) -> Optional[Callable]:  # pragma: no cover - 兼容占位
        return self._hooks[-1] if self._hooks else None
