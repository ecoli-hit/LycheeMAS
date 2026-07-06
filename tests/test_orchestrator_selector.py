"""M3：Orchestrator 接入 agent_selector（selector 优先 + 覆盖 warning + 零回归）。

pool 模式需 numpy + vendi_score（select() 时惰性）；缺则跳过。generate 模式的端到端走 env 门控，
这里只测 pool 接入 + 不带 selector 的零回归。
"""
from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("numpy")
pytest.importorskip("vendi_score")

from lychee_mas.core.types import TaskQuery  # noqa: E402
from lychee_mas.layers.construct.selectors.pool import CANDIDATE_ROLES  # noqa: E402
from lychee_mas.pipeline import Orchestrator  # noqa: E402
from lychee_mas.stores import TraceStore  # noqa: E402

_POOL_NAMES = {r[0] for r in CANDIDATE_ROLES}
_DEFAULT_TEAM = {"manager", "worker", "verifier"}


def _q(text="Solve this algebra problem step by step"):
    return TaskQuery(question=text, gold=None)


# ---- 零回归：不带 selector 时行为与原来一致 ----
def test_no_selector_unchanged():
    orch = Orchestrator(runtime="mock", team="default", rounds=1)
    graph = orch.build_graph(_q())
    assert graph.names == ["manager", "worker", "verifier"]  # 仍走 team 模板


def test_no_selector_end_to_end_unchanged():
    orch = Orchestrator(runtime="mock", team="default", rounds=1)
    traj = asyncio.run(orch.run(TaskQuery(question="answer is 7", gold="7")))
    assert traj.final_answer.content == "7"
    assert len(traj.messages) == 3  # manager -> worker -> verifier（与 test_mock_runtime 一致）


# ---- pool selector 接入 ----
def test_pool_selector_build_graph_uses_pool_roles():
    orch = Orchestrator(runtime="mock", selector="agentinit",
                        selector_kwargs={"mode": "pool"}, rounds=1)
    graph = orch.build_graph(_q())
    assert 1 <= len(graph.names) <= 5
    assert set(graph.names) <= _POOL_NAMES          # 成员来自候选池
    assert not (set(graph.names) & _DEFAULT_TEAM)   # 不再是 default team


def test_pool_selector_end_to_end_mock():
    trace = TraceStore()
    orch = Orchestrator(runtime="mock", selector="agentinit",
                        selector_kwargs={"mode": "pool"}, trace_store=trace, rounds=1)
    traj = asyncio.run(orch.run(TaskQuery(question="compute 6 times 7 is 42", gold="42")))
    assert traj.final_answer is not None
    assert len(traj.messages) >= 1
    assert {m.sender for m in traj.messages} <= _POOL_NAMES  # 每条消息来自被选角色


def test_build_graph_without_query_still_works():
    # selector 存在但 build_graph 未收到 query → 用空 TaskQuery，仍产出合法团队。
    orch = Orchestrator(runtime="mock", selector="agentinit", selector_kwargs={"mode": "pool"})
    graph = orch.build_graph()
    assert len(graph.names) >= 1 and set(graph.names) <= _POOL_NAMES


# ---- selector 优先 + 覆盖 warning ----
def test_selector_overrides_team_with_warning(caplog):
    orch = Orchestrator(runtime="mock", team="reason", selector="agentinit",
                        selector_kwargs={"mode": "pool"}, rounds=1)
    with caplog.at_level(logging.WARNING, logger="lychee_mas.pipeline"):
        graph = orch.build_graph(_q())
    assert set(graph.names) <= _POOL_NAMES          # selector 赢，reason 角色被覆盖
    assert any("覆盖" in r.message or "selector" in r.message for r in caplog.records)


def test_default_team_plus_selector_no_warning(caplog):
    # team 是默认值时不必 warning（用户没显式挑 team）。
    orch = Orchestrator(runtime="mock", team="default", selector="agentinit",
                        selector_kwargs={"mode": "pool"}, rounds=1)
    with caplog.at_level(logging.WARNING, logger="lychee_mas.pipeline"):
        orch.build_graph(_q())
    assert not caplog.records
