"""Orchestrator —— 端到端编排器（CLAUDE.md §0/§12）。

只按 config 从 REGISTRY 取组件，绝不硬编码实现（黄金法则 2：换单一组件即一组对照实验，不改本文件）。
默认 runtime=mock，可完全离线跑通：构建一张静态 MASGraph + 一个 TaskQuery -> Runtime.run ->
Trajectory。

闭环（后续）：在 run 之后接处理层聚合 / 归因→信用→训练→反哺；当前 P0 先打通 construct + runtime。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .core.registry import REGISTRY
from .core.types import Message, TaskQuery, Trajectory
from .runtime.base import MASGraph

_log = logging.getLogger("lychee_mas.pipeline")


class Orchestrator:
    """最小编排器：按名字从 REGISTRY 取 topology + runtime，跑一条 query。

    参数：
      - runtime: 运行时名（REGISTRY "runtime" 类别），默认 "mock"（离线确定性）。
      - team:    静态拓扑的队伍 profile 名（topology_generator/static），默认 "default"。
      - aggregator: 可选聚合器名（REGISTRY "aggregator"）；给出则对 [trajectory] 聚合出最终 Answer。
      - trace_store: 可选 trace.TraceStore；给出则把每条 Message 通过 runtime.intercept 写入。
      - runtime_kwargs: 透传给 runtime 构造（如 autogen 后端的 backend/ctx）。
      - selector: 可选 agent_selector 名（如 "agentinit"）。给出时**由 selector 决定团队成员**，
        `team` 的角色被覆盖、仅余 `meta["team"]` 标签（rounds 由 `--rounds` 决定，与 team 无关），
        会打 warning。默认 None=沿用 team 模板。
      - selector_kwargs: 透传给 selector 构造（如 {"mode": "pool"}）。`mode` 只此一个来源，无歧义。
      - pre_plugins / post_plugins: 可选插件链（REGISTRY "pre_run_plugin" / "post_run_plugin"），
        每项为名字或 (名字, kwargs)。pre 在 runtime.run 之前按序变换图（必须返回 MASGraph，
        非法返回显式报错）；post 在 runtime.run 之后按序消费轨迹（score=None）。
        不配置时行为与无插件完全一致。
    """

    def __init__(self, runtime: str = "mock", team: str = "default",
                 aggregator: Optional[str] = None, trace_store: Any = None,
                 rounds: int = 1, model: Optional[str] = None,
                 runtime_kwargs: Optional[dict] = None,
                 selector: Optional[str] = None, selector_kwargs: Optional[dict] = None,
                 pre_plugins: Optional[list] = None, post_plugins: Optional[list] = None):
        self.runtime_name = runtime
        self.team = team
        self.aggregator_name = aggregator
        self.trace_store = trace_store
        self.rounds = rounds
        self.model = model
        self.runtime_kwargs = runtime_kwargs or {}
        self.selector_name = selector
        self.selector_kwargs = selector_kwargs or {}
        self.pre_plugins = list(pre_plugins or [])
        self.post_plugins = list(post_plugins or [])

    @staticmethod
    def _create_plugins(category: str, specs: list) -> list[tuple[str, Any]]:
        """把 [名字 | (名字, kwargs)] 解析成 [(名字, 实例)]（按名从 REGISTRY 取）。"""
        out: list[tuple[str, Any]] = []
        for spec in specs:
            if isinstance(spec, str):
                name, kwargs = spec, {}
            else:
                name, kwargs = spec[0], dict(spec[1] or {})
            out.append((name, REGISTRY.create(category, name, **kwargs)))
        return out

    def build_graph(self, query: Optional[TaskQuery] = None) -> MASGraph:
        """构建一张 MASGraph。

        - 无 selector：按 `team` 名产出顺序链（原行为，零回归）。
        - 有 selector：先 `selector.select(query)` 选出团队成员（AgentSpec 列表），再交给
          StaticTopology 封装成图。**selector 优先**：此时 `team` 的角色被覆盖，仅余 meta 标签
          （rounds 由 Orchestrator 的 rounds 决定，与 team 无关）。
        """
        agents = None
        if self.selector_name:
            if self.team != "default":
                _log.warning(
                    "selector=%r 决定团队成员，--team=%r 的角色被覆盖（仅余 meta 标签；"
                    "rounds 由 --rounds 决定，与 team 无关）。",
                    self.selector_name, self.team)
            selector = REGISTRY.create("agent_selector", self.selector_name, **self.selector_kwargs)
            agents = selector.select(query) if query is not None else selector.select(TaskQuery())
        topo = REGISTRY.create("topology_generator", "static",
                               team=self.team, model=self.model, rounds=self.rounds)
        return topo.build(agents=agents)

    async def run(self, query: TaskQuery, hook: Optional[Callable[[Message], None]] = None):
        """端到端：建图 -> 运行前插件链 -> 取 runtime -> intercept -> run
        -> 运行后插件链 -> (可选)聚合，返回 Trajectory（或 Answer）。"""
        from .plugins.base import RunContext

        graph = self.build_graph(query)
        run_ctx = RunContext(task=str((query.meta or {}).get("task", "")),
                             trace_store=self.trace_store,
                             routing_ctx=self.runtime_kwargs.get("ctx"),
                             backend=self.runtime_kwargs.get("backend"))
        for name, plugin in self._create_plugins("pre_run_plugin", self.pre_plugins):
            graph = plugin.before_run(graph, query, run_ctx)
            if not isinstance(graph, MASGraph):
                raise TypeError(
                    f"pre_run_plugin/{name}.before_run 必须返回 MASGraph，"
                    f"得到 {type(graph).__name__}")

        runtime = REGISTRY.create("runtime", self.runtime_name, **self.runtime_kwargs)

        # intercept：把每条 Message 喂给 TraceStore / 自定义 hook
        if self.trace_store is not None and hasattr(runtime, "intercept"):
            runtime.intercept(self.trace_store.hook)
        if hook is not None and hasattr(runtime, "intercept"):
            runtime.intercept(hook)

        trajectory: Trajectory = await runtime.run(graph, query)

        for _name, plugin in self._create_plugins("post_run_plugin", self.post_plugins):
            plugin.after_run(trajectory, None, run_ctx)

        if self.aggregator_name:
            agg = REGISTRY.create("aggregator", self.aggregator_name)
            return agg.aggregate([trajectory])
        return trajectory
