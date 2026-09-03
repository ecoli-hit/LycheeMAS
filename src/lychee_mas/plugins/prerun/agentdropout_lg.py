"""pre_run_optimizer/agentdropout —— AgentDropout 接上统一接口（逐轮 apply 语义）。

薄适配：算法（两阶段淘汰训练/实现矩阵）全在 ``methods/prerun/agentdropout.py``，本类只做
「视图取节点 → 加载训练状态 → 第 ``round`` 轮 threshold 实现写回图」。AgentDropout 的
拓扑是**逐轮不同**的：多轮实验由脚本按轮取 ``realized_matrices(r)`` 驱动执行；本适配器
把指定一轮的通信结构挂到契约图上，并把该轮被淘汰的节点写入
``spec.meta["dropped"]=True``（节点工厂据此让该 agent 本轮不执行，原版输出 'None.'）。
"""
from __future__ import annotations

from typing import Any, Optional

from ...core.registry import REGISTRY
from ...methods.prerun.agentdropout import AgentDropoutOptimizer


@REGISTRY.register("pre_run_optimizer", "agentdropout")
class AgentDropoutLG:
    """AgentDropout 动态节点/边淘汰（LangGraph 统一接口版，apply 单轮语义）。"""

    name = "agentdropout"

    def __init__(self, state_file: Optional[str] = None, round: int = 0,
                 rounds: int = 2, **kwargs: Any) -> None:
        self.state_file = state_file
        self.round = int(round)
        self.rounds = int(rounds)
        self.kwargs = dict(kwargs)
        self.last_meta: Optional[dict] = None

    def optimize(self, graph: Any) -> Any:
        from .graphview import extract_view, rebuild  # 函数内导入避免包内环

        view = extract_view(graph)
        opt = AgentDropoutOptimizer(n_agents=len(view.names), rounds=self.rounds,
                                    state_file=self.state_file, **self.kwargs)
        sm, tm = opt.realized_matrices(self.round, "threshold")
        names = view.names
        dropped = opt.skip_nodes.get(self.round)
        for i, name in enumerate(names):
            view.specs[name].meta["dropped"] = (i == dropped)
        adjacency = {names[i]: [names[j] for j in range(len(names)) if sm[i][j]]
                     for i in range(len(names))}
        self.last_meta = {"round": self.round, "spatial": sm, "temporal": tm,
                          "dropped": None if dropped is None else names[dropped],
                          "skip_nodes": dict(opt.skip_nodes), "names": names}
        return rebuild(graph, adjacency=adjacency)
