"""插件系统（plugins/）的离线测试：mock runtime 端到端 + 适配器，零重依赖。"""
from __future__ import annotations

import asyncio

import pytest
from lychee_mas import Orchestrator
from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import TaskQuery
from lychee_mas.plugins.adapters import AttributionPlugin, PrunePlugin
from lychee_mas.plugins.base import RunContext
from lychee_mas.runtime.base import MASGraph
from lychee_mas.trace.base import Attribution


@REGISTRY.register("pre_run_plugin", "_test_drop_last")
class DropLastPlugin:
    """测试用：删掉图的最后一个节点（产出新图，不原地改）。"""

    def before_run(self, graph, query, ctx):
        return MASGraph(nodes=list(graph.nodes[:-1]), edges=dict(graph.edges),
                        rounds=graph.rounds, meta=dict(graph.meta))


@REGISTRY.register("pre_run_plugin", "_test_bad_return")
class BadReturnPlugin:
    def before_run(self, graph, query, ctx):
        return None  # 非法：必须返回 MASGraph


_captured: list = []


@REGISTRY.register("post_run_plugin", "_test_capture")
class CapturePlugin:
    def after_run(self, trajectory, score, ctx):
        _captured.append((trajectory, score))


@REGISTRY.register("graph_pruner", "_test_identity_pruner")
class IdentityPruner:
    def __init__(self, **kwargs):
        self.cfg = kwargs

    def prune(self, graph, context=None):
        return MASGraph(nodes=list(graph.nodes), edges=dict(graph.edges),
                        rounds=graph.rounds, meta=dict(graph.meta))


@REGISTRY.register("attributor", "_test_blame_first")
class BlameFirstAttributor:
    def __init__(self, **kwargs):
        self.cfg = kwargs

    def attribute(self, trajectory, context=None):
        first = trajectory.messages[0].sender if trajectory.messages else "unknown"
        return [Attribution(agent=first, step=0, is_fault=True, reason="test")]


@REGISTRY.register("credit_assigner", "_test_uniform")
class UniformAssigner:
    def __init__(self, **kwargs):
        self.cfg = kwargs

    def credits(self, attributions, reward):
        return {a.agent: reward for a in attributions}


def run_orch(**kwargs):
    orch = Orchestrator(runtime="mock", team="default", **kwargs)
    return asyncio.run(orch.run(TaskQuery(question="2 plus 2 is 4")))


def test_no_plugins_behavior_unchanged():
    base = run_orch()
    with_empty = run_orch(pre_plugins=[], post_plugins=[])
    assert len(base.messages) == len(with_empty.messages)
    assert base.final_answer.content == with_empty.final_answer.content


def test_pre_plugin_transforms_graph():
    base = run_orch()
    pruned = run_orch(pre_plugins=["_test_drop_last"])
    # mock runtime 每个节点产一条消息：少一个节点 => 少一条消息
    assert len(pruned.messages) == len(base.messages) - 1


def test_pre_plugin_bad_return_raises_type_error():
    with pytest.raises(TypeError, match="_test_bad_return"):
        run_orch(pre_plugins=["_test_bad_return"])


def test_post_plugin_receives_trajectory():
    _captured.clear()
    traj = run_orch(post_plugins=["_test_capture"])
    assert len(_captured) == 1
    got, score = _captured[0]
    assert got is traj
    assert score is None  # Orchestrator 层无 metric


def test_prune_adapter_wraps_registered_pruner():
    plugin = PrunePlugin(pruner="_test_identity_pruner")
    graph = MASGraph(nodes=[], edges={}, rounds=1, meta={})
    out = plugin.before_run(graph, TaskQuery(), RunContext())
    assert isinstance(out, MASGraph)


def test_prune_adapter_stub_pruner_raises_not_implemented():
    plugin = PrunePlugin(pruner="agentdropout")  # 真桩：显式 NotImplementedError，不静默跳过
    graph = MASGraph(nodes=[], edges={}, rounds=1, meta={})
    with pytest.raises(NotImplementedError):
        plugin.before_run(graph, TaskQuery(), RunContext())


def test_attribution_adapter_writes_meta_and_trace_store():
    class MiniStore:
        def __init__(self):
            self.records = []

        def log_decision(self, record):
            self.records.append(record)

    traj = run_orch()
    store = MiniStore()
    plugin = AttributionPlugin(attributor="_test_blame_first",
                               credit_assigner="_test_uniform")
    plugin.after_run(traj, 1.0, RunContext(trace_store=store))
    assert traj.meta["credits"] == {traj.messages[0].sender: 1.0}
    assert traj.meta["attribution"][0]["agent"] == traj.messages[0].sender
    assert store.records[0]["type"] == "attribution"


def test_new_categories_registered():
    assert "prune" in REGISTRY.list("pre_run_plugin")
    assert "attribution" in REGISTRY.list("post_run_plugin")
    assert "gepa" in REGISTRY.list("optimizer")
