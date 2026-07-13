"""AutoGenRuntime —— 把 MASGraph 映射为 AutoGen 群聊并跑任务（迁移自 teams/groupchat.py）。

发言者选择走 AutoGen 原生 SelectorGroupChat 的 selector_func：默认提供轮转 selector_func（
round-robin）；
要换选择算法，传入自定义 selector_func(messages)->下一个发言者名（返回 None 则回退到内置 LLM 选择
）。
最后一个角色用 'APPROVE: <answer>' 结束（TextMentionTermination），并由最大轮数兜底。

路由信号经共享 RoutingContext 流动——每个 agent 的 InjectionClient 都读它。答案提取按任务可定制
（见 eval/task_config.extractor_for_task）。`run` 返回统一的 `Trajectory`，并把每条消息 + 决策喂给
intercept hook（写 TraceStore / 抽取）。

⚠️ 这是【唯一允许 import autogen 的文件之一】（CLAUDE.md 黄金法则 1：runtime/backends/autogen_*.py
）。
autogen_agentchat 的导入全部惰性化到方法内部，保证 import 本模块（与注册）不需要 autogen。
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from ...core.registry import REGISTRY
from ...core.types import Answer, Message, TaskQuery, Trajectory
from ..base import BaseRuntime, MASGraph


def round_robin_selector(agent_names: Sequence[str]) -> Callable:
    """生成一个轮转的 selector_func：按 agent 顺序循环选下一个发言者（默认选择算法）。"""
    state = {"i": 0}

    def _selector(messages) -> str:
        name = agent_names[state["i"] % len(agent_names)]
        state["i"] += 1
        return name

    return _selector


def build_groupchat(agents, max_rounds: int = 2, selector_func: Optional[Callable] = None,
                    selector_client=None):
    """把已构造好的 agents 包成 AutoGen SelectorGroupChat，发言者选择由 selector_func 定义。

    - selector_func: 选下一个发言者的函数 (messages)->name；None 时用默认轮转。
    - selector_client: selector_func 返回 None（交给 LLM 选）时所需的 model_client。
    单 agent 队伍 SelectorGroupChat 不支持（要求 >=2），回退 RoundRobinGroupChat。
    """
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat

    n_turns = len(agents) * max_rounds
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(n_turns + 1)
    if len(agents) < 2:
        return RoundRobinGroupChat(agents, termination_condition=termination, max_turns=n_turns)
    if selector_func is None:
        selector_func = round_robin_selector([a.name for a in agents])
    return SelectorGroupChat(agents, model_client=selector_client, selector_func=selector_func,
                             termination_condition=termination, max_turns=n_turns,
                             allow_repeated_speaker=True)


def _build_agents(graph: MASGraph, backend, ctx, max_new_tokens: int = 256,
                  model_id: str = "qwen-3-4b") -> List[Any]:
    """把 MASGraph 的节点（AgentSpec）实例化成 AutoGen AssistantAgent（各绑定一个 InjectionClient
    ）。"""
    from autogen_agentchat.agents import AssistantAgent

    from .autogen_injection_client import make_injection_client

    agents = []
    for node in graph.order():
        mid = node.model or model_id
        ctx.model_of_role[node.name] = mid  # 登记角色->模型（驱动约束 #2 的同模型对判断）
        client = make_injection_client(backend, role=node.name, ctx=ctx,
                                       max_new_tokens=max_new_tokens, model_id=mid)
        desc = node.meta.get("description") or node.name
        agents.append(AssistantAgent(node.name, model_client=client,
                                     system_message=node.system_prompt, description=desc))
    return agents


@REGISTRY.register("runtime", "autogen")
class AutoGenRuntime(BaseRuntime):
    """AutoGen 后端：把 MASGraph 跑成 SelectorGroupChat，产出统一 Trajectory。"""

    name = "autogen"

    def __init__(self, backend=None, ctx=None, max_new_tokens: int = 256,
                 max_rounds: int = 2, model_id: str = "qwen-3-4b",
                 selector_func: Optional[Callable] = None):
        super().__init__()
        self.backend = backend  # 共享 HFBackend（None 时延迟到 run 前必须由调用方注入）
        self.ctx = ctx  # 共享 RoutingContext（携带 router/memory/task）
        self.max_new_tokens = max_new_tokens
        self.max_rounds = max_rounds
        self.model_id = model_id
        self.selector_func = selector_func

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if self.backend is None or self.ctx is None:
            raise ValueError("AutoGenRuntime.run 需要 backend 与 ctx（构造时或运行前注入）")
        from .autogen_injection_client import make_plain_client

        agents = _build_agents(team, self.backend, self.ctx,
                               max_new_tokens=self.max_new_tokens, model_id=self.model_id)
        gc = build_groupchat(agents, max_rounds=self.max_rounds,
                             selector_func=self.selector_func,
                             selector_client=make_plain_client(self.backend))
        result = await gc.run(task=query.question)
        msgs = result.messages

        # 组装统一 Trajectory：逐条 AutoGen 消息 -> core.Message，并 emit 给 hook
        traj = Trajectory(task_id=query.id, meta={"runtime": self.name, "task": self.ctx.task})
        for r, m in enumerate(msgs):
            msg = Message(
                sender=getattr(m, "source", "?"),
                content=getattr(m, "content", "") or "",
                round=r,
                role="assistant",
            )
            traj.add(msg)
            self._emit(msg)

        # 答案提取按任务定制（eval/task_config）
        from ...eval.task_config import extractor_for_task

        final_text = extractor_for_task(self.ctx.task)(msgs)
        traj.final_answer = Answer(content=final_text, source="verifier",
                                   meta={"n_messages": len(msgs)})
        traj.candidates = [traj.final_answer]
        traj.meta["decisions"] = list(getattr(self.ctx, "decisions", []))
        return traj
