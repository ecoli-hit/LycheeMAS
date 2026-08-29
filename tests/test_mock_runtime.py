"""MockRuntime + Orchestrator：离线跑出 Trajectory（无重依赖）。"""
from __future__ import annotations

import asyncio

from lychee_mas.core.types import AgentSpec, TaskQuery
from lychee_mas.pipeline import Orchestrator
from lychee_mas.runtime.adapters.frameworks.mock import MockRuntime
from lychee_mas.runtime.contracts.runtime import MASGraph
from lychee_mas.trace import TraceStore


def test_mock_runtime_direct():
    rt = MockRuntime()
    graph = MASGraph(nodes=[
        AgentSpec(name="manager", role="manager"),
        AgentSpec(name="verifier", role="verifier"),
    ], rounds=1)
    seen = []
    rt.intercept(seen.append)
    traj = asyncio.run(rt.run(graph, TaskQuery(question="2 plus 2 is 4", gold="4")))
    assert traj.final_answer is not None
    assert traj.final_answer.content == "4"  # 回显 query 中最后一个数字
    assert len(traj.messages) == 2
    assert len(seen) == 2  # intercept 收到每条 Message


def test_orchestrator_end_to_end_mock():
    trace = TraceStore()
    orch = Orchestrator(runtime="mock", team="default", trace_store=trace, rounds=1)
    traj = asyncio.run(orch.run(TaskQuery(question="answer is 7", gold="7")))
    assert traj.final_answer.content == "7"
    assert len(traj.messages) == 3  # manager -> worker -> verifier
    assert len(trace) == 3


def test_orchestrator_with_aggregator_returns_answer():
    orch = Orchestrator(runtime="mock", team="single", aggregator="self_consistency", rounds=1)
    ans = asyncio.run(orch.run(TaskQuery(question="result 9", gold="9")))
    assert ans.content == "9"
