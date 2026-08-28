"""LangGraph 后端（runtime/langgraph）的离线测试。

需要安装 langgraph（`.[langgraph]` extra）；未安装则整文件跳过（importorskip 模式）。
FakeBackend 脚本化输出（verifier 轮出 APPROVE），零 GPU / 零 API。
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langgraph")

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import TaskQuery
from lychee_mas.layers.construct.templates import StaticTopology
from lychee_mas.memory.base import MemoryBundle
from lychee_mas.memory.context import RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router


class FakeGen:
    def __init__(self, text: str):
        self.text = text
        self.n_prompt_pos = 7
        self.n_gen_tokens = 3
        self.prefix_len = 0
        self.latency_s = 0.001


class ScriptedBackend:
    """按调用次序输出脚本文本；记录每次收到的消息。"""

    def __init__(self, script: list[str]):
        self.script = script
        self.calls: list[list] = []

    def generate_chat(self, msgs, max_new_tokens):
        self.calls.append(list(msgs))
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        return FakeGen(self.script[idx])


class NullMemory:
    def observe(self, messages):
        pass

    def recall(self, decision, query):
        return MemoryBundle()

    def reset(self):
        pass


def make_runtime(backend, max_rounds: int = 2):
    ctx = RoutingContext(task="gsm8k", router=fixed_channel_router("none"),
                         memory=NullMemory())
    ctx.trace_model_calls = False
    rt = REGISTRY.create("runtime", "langgraph", backend=backend, ctx=ctx,
                         max_new_tokens=16, max_rounds=max_rounds)
    return rt, ctx


def build_graph():
    return StaticTopology(team="default").build()


def test_approve_terminates_and_extracts_answer():
    graph = build_graph()
    n_agents = len(graph.order())
    # 第二个发言的 agent 直接给出 APPROVE 答案
    backend = ScriptedBackend(["step one", "APPROVE: 42"])
    rt, ctx = make_runtime(backend)
    seen: list = []
    rt.intercept(seen.append)
    traj = asyncio.run(rt.run(graph, TaskQuery(question="1 plus 41?")))
    # 消息 = 首条 task + 2 条 agent 发言（APPROVE 消息保留后终止）
    assert len(traj.messages) == 3
    assert traj.final_answer is not None and traj.final_answer.content == "42"
    assert traj.meta["stop_reason"] == "APPROVE"
    assert len(seen) == len(traj.messages)  # intercept 收到全部消息
    assert n_agents >= 2


def test_max_turns_caps_when_no_approve():
    graph = build_graph()
    n_agents = len(graph.order())
    backend = ScriptedBackend(["thinking..."])  # 永不 APPROVE
    rt, ctx = make_runtime(backend, max_rounds=2)
    traj = asyncio.run(rt.run(graph, TaskQuery(question="q")))
    # agent 发言数 = len(agents) * max_rounds；消息数再 +1（首条 task）
    assert len(traj.messages) == n_agents * 2 + 1
    assert traj.meta["stop_reason"] == "max_turns"
    # 每次 agent 发言都落了一条决策
    assert len(traj.meta["decisions"]) == n_agents * 2


def test_context_assembly_matches_assistant_semantics():
    graph = build_graph()
    first = graph.order()[0]
    backend = ScriptedBackend(["A-says", "APPROVE: done"])
    rt, ctx = make_runtime(backend)
    asyncio.run(rt.run(graph, TaskQuery(question="the task")))
    # 首个 agent 首轮：system=自己的 prompt + task 消息（user, source=user）
    first_call = backend.calls[0]
    assert first_call[0]["role"] == "system"
    assert first_call[0]["content"] == first.system_prompt
    assert first_call[1] == {"role": "user", "content": "the task", "source": "user"}
    # 第二个 agent 看到第一个 agent 的输出（user + source=该 agent 名）
    second_call = backend.calls[1]
    assert second_call[-1] == {"role": "user", "content": "A-says", "source": first.name}


def test_tool_team_raises_not_implemented():
    graph = build_graph()
    graph.order()[0].meta["agent_type"] = "web_surfer"
    rt, ctx = make_runtime(ScriptedBackend(["x"]))
    with pytest.raises(NotImplementedError):
        asyncio.run(rt.run(graph, TaskQuery(question="q")))


def test_registered_in_registry():
    assert "langgraph" in REGISTRY.list("runtime")
