"""MockRuntime —— 确定性离线后端（测试/CI 默认，CLAUDE.md §8）。

无需任何模型/网络：按团队顺序为每个 agent 产出一条伪 Message，最后一个 agent 产出 final_answer
（回显或取 query 中的数字），返回 Trajectory，并把每条 Message 通过 intercept hook 写出。纯标准库。
这是 examples/tests 的执行后端。
"""
from __future__ import annotations

import re

from ...core.registry import REGISTRY
from ...core.types import Answer, Message, TaskQuery, Trajectory
from ..base import BaseRuntime, MASGraph


def _last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


@REGISTRY.register("runtime", "mock")
class MockRuntime(BaseRuntime):
    """离线确定性运行时：按节点顺序轮转发言，无 LLM 调用。"""

    name = "mock"

    def __init__(self, prompt_tokens: int = 8, completion_tokens: int = 8) -> None:
        super().__init__()
        # 固定的伪 token 记账，保证可复现的成本数字（黄金法则 6）
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        traj = Trajectory(task_id=query.id, meta={"runtime": self.name})
        nodes = team.order()
        if not nodes:
            raise ValueError("MockRuntime.run: 团队没有任何节点")

        rounds = max(1, int(getattr(team, "rounds", 1)))
        prev = query.question
        round_idx = 0
        for r in range(rounds):
            for node in nodes:
                role = node.role or node.name or "agent"
                content = f"[mock:{role}] echo: {prev}"
                msg = Message(
                    sender=node.name or role,
                    content=content,
                    round=round_idx,
                    role="assistant",
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    meta={"mock": True},
                )
                traj.add(msg)
                self._emit(msg)  # 逐消息回调（写 TraceStore / 抽取）
                prev = content
                round_idx += 1

        # 最后一个 agent 产出最终答案：优先回显 query 中的数字，否则回显问题
        final_text = _last_number(query.question) or query.question
        last_role = (nodes[-1].role or nodes[-1].name or "agent")
        traj.final_answer = Answer(
            content=final_text,
            source=nodes[-1].name or last_role,
            confidence=1.0,
            meta={"mock": True},
        )
        traj.candidates = [traj.final_answer]
        return traj
