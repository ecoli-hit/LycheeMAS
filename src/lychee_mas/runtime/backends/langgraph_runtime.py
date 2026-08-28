"""LangGraphRuntime —— 把 MASGraph 映射为 LangGraph StateGraph 并跑任务。

映射关系：
- 每个 AgentSpec = 一个 LangGraph 节点（节点函数内调共享注入引擎 run_injection_step）。
- 边 = 顺序轮转（START → 首节点；每个节点后接条件边：终止则 END，否则下一个节点）。
- 终止 = 任意 agent 输出含 "APPROVE"（该消息保留）或 agent 发言数达上限
  （max_turns = max_turns or len(agents) × max_rounds），与 autogen 后端的
  TextMentionTermination("APPROVE") | MaxMessageTermination 语义对齐。

上下文组装与 AutoGen AssistantAgent 对齐：每个 agent 看到
[system=自己的 system_prompt] + 全部历史（自己的消息为 assistant，他人消息为 user，带 source）；
history[0] 恒为初始 task 消息（source="user"）。

MVP 仅支持纯文本 assistant 团队：工具型节点 / magentic_one / coder_executor 预设
显式 raise NotImplementedError（不静默降级）。

⚠️ langgraph 是重依赖：只在 `_build_app` 内部 import（惰性导入约定），
import 本模块与注册不需要 langgraph。
"""
from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from ...core.registry import REGISTRY
from ...core.types import AgentSpec, Answer, Message, TaskQuery, Trajectory
from ..base import BaseRuntime, MASGraph
from ..injection import ChatMessage, InjectionRequest, run_injection_step
from ._common import extract_final_answer


class MASState(TypedDict):
    """LangGraph 状态：任务 + 消息历史 + 终止信号。

    history 条目：{"source": 发言者, "content": 文本, "prompt_tokens", "completion_tokens"}。
    agent_turns 只计 agent 发言（不含首条 task 消息）。
    """

    task: str
    history: List[Dict[str, Any]]
    agent_turns: int
    done: bool


def _agent_type(node: AgentSpec) -> str:
    return str(node.meta.get("agent_type") or node.meta.get("type") or "assistant").lower()


def _has_tool_agents(graph: MASGraph) -> bool:
    return any(_agent_type(node) != "assistant" or node.tools or node.meta.get("tools")
               for node in graph.order())


def _graph_preset(graph: MASGraph) -> str:
    return str(graph.meta.get("team_preset") or graph.meta.get("team") or "").lower()


@REGISTRY.register("runtime", "langgraph")
class LangGraphRuntime(BaseRuntime):
    """LangGraph 执行后端（构造参数对齐 AutoGenRuntime 的纯文本子集）。"""

    name = "langgraph"

    def __init__(self, backend: Any = None, ctx: Any = None, max_new_tokens: int = 256,
                 max_rounds: int = 2, model_id: str = "qwen-3-4b",
                 max_turns: Optional[int] = None,
                 trace_model_calls: Optional[bool] = None) -> None:
        super().__init__()
        self.backend = backend
        self.ctx = ctx
        self.max_new_tokens = max_new_tokens
        self.max_rounds = max_rounds
        self.model_id = model_id
        self.max_turns = max_turns
        self.trace_model_calls = trace_model_calls

    # ---------------- 图构建 ----------------

    def _chat_for(self, spec: AgentSpec, state: MASState) -> List[ChatMessage]:
        """对齐 AutoGen AssistantAgent 的上下文组装（system + 按 source 映射的历史）。"""
        chat: List[ChatMessage] = [{"role": "system", "content": spec.system_prompt}]
        for entry in state["history"]:
            role = "assistant" if entry["source"] == spec.name else "user"
            chat.append({"role": role, "content": entry["content"], "source": entry["source"]})
        return chat

    def _make_node(self, spec: AgentSpec) -> Callable[[MASState], Awaitable[Dict[str, Any]]]:
        async def node(state: MASState) -> Dict[str, Any]:
            chat = self._chat_for(spec, state)
            res = run_injection_step(self.backend, self.ctx, InjectionRequest(
                role=spec.name, chat=chat, max_new_tokens=self.max_new_tokens))
            entry = {
                "source": spec.name,
                "content": res.text,
                "prompt_tokens": res.input_positions,
                "completion_tokens": res.output_tokens,
            }
            return {
                "history": state["history"] + [entry],  # 返回新列表，不原地改 state
                "agent_turns": state["agent_turns"] + 1,
                "done": state["done"] or ("APPROVE" in res.text),
            }

        return node

    def _build_app(self, team: MASGraph, max_turns: int) -> Any:
        """唯一 import langgraph 的位置（惰性导入约定）。"""
        from langgraph.graph import END, START, StateGraph

        names = [a.name for a in team.order()]

        def _route(state: MASState) -> str:
            return "end" if (state["done"] or state["agent_turns"] >= max_turns) else "continue"

        g = StateGraph(MASState)
        for spec in team.order():
            g.add_node(spec.name, self._make_node(spec))
        g.add_edge(START, names[0])
        for i, name in enumerate(names):
            nxt = names[(i + 1) % len(names)]
            g.add_conditional_edges(name, _route, {"continue": nxt, "end": END})
        return g.compile()

    # ---------------- 执行 ----------------

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if self.backend is None or self.ctx is None:
            raise ValueError("LangGraphRuntime.run 需要 backend 与 ctx（构造时或运行前注入）")
        preset = _graph_preset(team)
        if _has_tool_agents(team) or preset in {"magentic_one", "coder_executor"}:
            raise NotImplementedError(
                "runtime/langgraph 目前仅支持纯文本 assistant 团队；"
                "工具型团队 / magentic_one / coder_executor 预设请用 runtime/autogen")

        agents = team.order()
        if not agents:
            raise ValueError("LangGraphRuntime.run 需要至少一个 agent 节点")
        max_turns = self.max_turns or (len(agents) * self.max_rounds)
        self.ctx.trace_model_calls = (
            True if self.trace_model_calls is None else self.trace_model_calls
        )
        for node in agents:
            # 登记角色->模型（驱动同模型对判断），与 autogen 后端一致
            self.ctx.model_of_role[node.name] = node.model or self.model_id

        t0 = time.time()
        self.ctx.log_span(
            "runtime_start",
            runtime=self.name,
            team=team.meta.get("team"),
            team_preset=team.meta.get("team_preset"),
            uses_tools=False,
            max_turns=self.max_turns,
            max_rounds=self.max_rounds,
        )
        state: MASState = {
            "task": query.question,
            "history": [{"source": "user", "content": query.question,
                         "prompt_tokens": 0, "completion_tokens": 0}],
            "agent_turns": 0,
            "done": False,
        }
        try:
            app = self._build_app(team, max_turns)
            # langgraph 默认 recursion_limit=25，长对话必须显式放宽
            final = await app.ainvoke(state, config={"recursion_limit": 2 * max_turns + 10})
        except Exception as exc:
            from ..spans import exception_record

            self.ctx.log_span("runtime_error", **exception_record(exc))
            raise
        elapsed = time.time() - t0

        history = final["history"]
        stop_reason = "APPROVE" if final["done"] else "max_turns"
        traj = Trajectory(task_id=query.id, meta={
            "runtime": self.name,
            "task": self.ctx.task,
            "team": team.meta.get("team"),
            "team_preset": team.meta.get("team_preset"),
            "workspace": None,
            "copied_files": [],
            "stop_reason": stop_reason,
            "case_wall_time_s": round(elapsed, 3),
        })
        for r, entry in enumerate(history):
            msg = Message(
                sender=entry["source"],
                content=entry["content"],
                round=r,
                role="assistant",
                prompt_tokens=int(entry.get("prompt_tokens", 0)),
                completion_tokens=int(entry.get("completion_tokens", 0)),
                meta={"message_type": "text"},
            )
            traj.add(msg)
            self._emit(msg)

        from ...eval.task_config import extractor_for_task

        final_text = extract_final_answer(history) or extractor_for_task(self.ctx.task)(history)
        traj.final_answer = Answer(content=final_text, source="verifier",
                                   meta={"n_messages": len(history),
                                         "case_wall_time_s": round(elapsed, 3)})
        traj.candidates = [traj.final_answer]
        traj.meta["decisions"] = list(getattr(self.ctx, "decisions", []))
        traj.meta["tool_calls"] = []
        traj.meta["tool_call_count"] = 0
        traj.meta["tool_error_count"] = 0
        self.ctx.log_span(
            "runtime_end",
            stop_reason=stop_reason,
            message_count=len(history),
            tool_call_count=0,
            tool_error_count=0,
            case_wall_time_s=round(elapsed, 3),
        )
        return traj
