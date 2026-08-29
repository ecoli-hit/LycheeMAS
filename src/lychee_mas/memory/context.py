"""RoutingContext —— 把路由信号从 GroupChat 层下传到各 agent 的 InjectionClient 的共享状态
（CLAUDE.md §8，「AutoGen 版唯一要接仔细的链路」）。

GroupChat 里 agent 串行发言，所以单个共享 context 是安全的：每个 agent 的 client 从这里读
（自己的 role、当前 task/turn、上一个发言者），问 router（记忆通道决策）、向 memory 召回，再把决策
写回。
router（记忆通道决策）与 memory（记忆方法）是两个可替换接缝；其余固定。

决策日志默认存在 `self.decisions`（可解释性 + 落盘）；若构造时传入 `trace_store`，则同时
写入 `trace.TraceStore`（CLAUDE.md §8：统一落点）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .routing.base import MemoryRouter, RouteDecision

if TYPE_CHECKING:  # 仅类型检查期需要；避免与 memory.base 形成 import 环
    from .base import MemoryManager


class ModelCallBudgetExceeded(RuntimeError):
    """Raised before a model request would exceed the per-case hard limit."""


class TrialDeadlineExceeded(TimeoutError):
    """Raised before a model request would start after the Trial deadline."""


@dataclass
class RoutingContext:
    task: str  # 当前任务名（下传给 RouterInputs）
    router: MemoryRouter  # 记忆通道决策（所有 agent 共用一个）
    memory: MemoryManager  # 记忆方法（所有 agent 共用一个）
    team: Optional[str] = None  # 强制用的队伍 profile；None=按 task 自动选
    turns: Dict[str, int] = field(default_factory=dict)  # role -> 该角色已发言轮次
    availability: Dict[str, bool] = field(
        default_factory=lambda: {"none": True, "nl": True, "latent": True, "both": True}
    )  # 各通道是否可用（约束 #2 用）
    model_of_role: Dict[str, str] = field(default_factory=dict)  # role -> model id（判同模型对）
    deployment_of_role: Dict[str, str] = field(default_factory=dict)
    decisions: List[dict] = field(default_factory=list)  # 决策日志（可解释性 + 落盘 routing_trace）
    trace_store: Any = None  # 可选 trace.TraceStore：把决策同时写入统一落点
    event_writer: Any = None  # 可选 RunEventWriter：实时写入唯一 EventLog
    current_case_id: Optional[str] = None
    current_dataset_index: Optional[int] = None
    current_worker_id: Optional[int] = None
    current_trial_index: Optional[int] = None
    current_attempt: Optional[int] = None
    current_run_event_id: Optional[str] = None
    current_trial_event_id: Optional[str] = None
    current_attempt_event_id: Optional[str] = None
    current_runtime_event_id: Optional[str] = None
    current_group_chat_event_id: Optional[str] = None
    base_seed: int = 0
    trial_seed: Optional[int] = None
    seed_derivation: Optional[str] = None
    generation_seed: Optional[int] = None
    trace_model_calls: bool = True  # 是否在终端打印每次 LLM 调用的 start/done 摘要
    max_model_calls_per_case: Optional[int] = None  # None=不限制；包含 participant/controller
    model_calls_started: int = 0  # 已经发送或尝试发送的真实后端请求
    model_call_budget_exhausted: bool = False
    case_deadline_monotonic_s: Optional[float] = None
    case_deadline_exceeded: bool = False
    context_visibility: str = "shared"  # AutoGen 默认 shared；可显式 topology_filtered
    last_model_exchange: Optional[dict[str, Any]] = None  # C2C 内存态，不写入 Trial 结果
    graph_edges: Dict[str, List[str]] = field(default_factory=dict)
    tool_request_event_ids: Dict[str, str] = field(default_factory=dict)
    tool_request_records: Dict[str, dict[str, Any]] = field(default_factory=dict)
    observed_tool_executions: List[dict[str, Any]] = field(default_factory=list)
    last_controller_exchange: Optional[dict[str, Any]] = None
    # TeamSpec shared-state memory. This is intentionally separate from
    # ``memory``, which is LycheeMAS's NL/latent communication method.
    team_memory: Any = None

    def turn_of(self, role: str) -> int:
        return self.turns.get(role, 0)  # 读取某角色当前轮次（默认 0）

    def bump_turn(self, role: str) -> None:
        self.turns[role] = self.turns.get(role, 0) + 1  # 该角色发言后 +1

    def reserve_model_call(self, role: str, *, controller: bool = False) -> int:
        """Reserve one real backend request and return its one-based case index."""

        deadline = self.case_deadline_monotonic_s
        if deadline is not None and time.monotonic() >= deadline:
            self.case_deadline_exceeded = True
            self.log_event(
                "trial.deadline_exceeded",
                role=role,
                controller=controller,
                model_calls_started=self.model_calls_started,
            )
            raise TrialDeadlineExceeded(
                f"Trial deadline reached before model call for role {role!r}"
            )
        limit = self.max_model_calls_per_case
        if limit is not None and self.model_calls_started >= limit:
            self.model_call_budget_exhausted = True
            self.log_event(
                "model_call.budget_exhausted",
                role=role,
                controller=controller,
                model_calls_started=self.model_calls_started,
                max_model_calls_per_case=limit,
            )
            raise ModelCallBudgetExceeded(
                f"Maximum model calls per case {limit} reached before role {role!r}"
            )
        self.model_calls_started += 1
        return self.model_calls_started

    def same_model_pair(self, sender: Optional[str], receiver: Optional[str]) -> bool:
        """约束 #2：latent 只能在对齐（同模型）的 agent 对之间传递。

        缺信息时（无 sender/receiver 或未登记模型）保守返回 True（交给路由器后续判断）。
        """
        if not sender or not receiver:
            return True
        ms, mr = self.model_of_role.get(sender), self.model_of_role.get(receiver)
        if ms is None or mr is None:
            return True
        return ms == mr  # 同模型才允许 latent 跨 agent

    def log_decision(
        self, role: str, turn: int, sender: Optional[str], decision: RouteDecision, extra: dict
    ) -> None:
        # 把一次路由决策 + 额外信息（成本记账 + 本 agent 的输入 input_messages / 输出 output）
        # 追加进日志；落盘为每样本的 routing_trace（供核查 / MAST judge / 反事实蒸馏复用）
        record = {
            # Backward-compatible short names.
            "role": role,
            "turn": turn,
            "sender": sender,
            "channel": decision.channel,  # P 已移出 RouteDecision（归 latent 通道）
            "reason": decision.reason,
            # Clear names for downstream metric/trace consumers.
            "turn_index": turn,
            "sender_role": sender,
            "memory_channel": decision.channel,
            "routing_reason": decision.reason,
            **extra,
        }
        self.decisions.append(record)
        if self.trace_store is not None:
            self.trace_store.log_decision(record)

    def set_case(self, case_id: str, dataset_index: int) -> None:
        self.current_case_id = str(case_id)
        self.current_dataset_index = int(dataset_index)

    def log_event(self, event_type: str, **fields: Any) -> Optional[str]:
        if self.event_writer is None:
            return None
        if "parent_event_id" not in fields:
            fields["parent_event_id"] = self._event_parent(event_type)
        fields.setdefault("trial_index", self.current_trial_index)
        fields.setdefault("attempt", self.current_attempt)
        fields.setdefault("base_seed", self.base_seed)
        fields.setdefault("trial_seed", self.trial_seed)
        fields.setdefault("seed_derivation", self.seed_derivation)
        fields.setdefault("case_id", self.current_case_id)
        fields.setdefault("dataset_index", self.current_dataset_index)
        fields.setdefault("worker_id", self.current_worker_id)
        return self.event_writer.log_event(event_type, **fields)

    def _event_parent(self, event_type: str) -> Optional[str]:
        """Return the nearest causal parent for a newly emitted event."""

        if event_type.startswith(("run.", "backend.", "concurrency.")):
            return self.current_run_event_id
        if event_type.startswith("trial."):
            return self.current_run_event_id
        if event_type.startswith("attempt."):
            return self.current_trial_event_id
        if event_type.startswith("runtime."):
            return self.current_attempt_event_id or self.current_trial_event_id
        if event_type.startswith(("workspace.", "code_executor.", "web.")):
            return self.current_runtime_event_id or self.current_attempt_event_id
        if event_type.startswith("memory."):
            return self.current_group_chat_event_id or self.current_runtime_event_id
        if event_type.startswith("coordination."):
            return self.current_group_chat_event_id or self.current_runtime_event_id
        if event_type.startswith(
            ("node_invocation.", "control_edge.", "data_edge.", "data_item.", "result.")
        ):
            return self.current_group_chat_event_id or self.current_runtime_event_id
        if event_type.startswith(("group_chat.", "framework.", "model_call.", "agent.", "tool_")):
            return (
                self.current_group_chat_event_id
                or self.current_runtime_event_id
                or self.current_attempt_event_id
                or self.current_trial_event_id
            )
        return self.current_trial_event_id or self.current_run_event_id

    def set_trial_event(self, event_id: Optional[str]) -> None:
        self.current_trial_event_id = event_id

    def remember_tool_request_event(
        self,
        tool_call_id: str,
        event_id: Optional[str],
        record: Optional[dict[str, Any]] = None,
    ) -> None:
        if tool_call_id and event_id:
            self.tool_request_event_ids[str(tool_call_id)] = str(event_id)
        if tool_call_id and record is not None:
            self.tool_request_records[str(tool_call_id)] = dict(record)

    def tool_request_event_id(self, tool_call_id: str) -> Optional[str]:
        return self.tool_request_event_ids.get(str(tool_call_id)) if tool_call_id else None

    def tool_request_record(self, tool_call_id: str) -> dict[str, Any]:
        return dict(self.tool_request_records.get(str(tool_call_id)) or {})

    def record_tool_execution(self, record: dict[str, Any]) -> None:
        self.observed_tool_executions.append(dict(record))

    def configure_graph(self, graph: Any) -> None:
        """Install per-case communication visibility from a normalized MASGraph."""
        self.graph_edges = {
            str(source): [str(target) for target in targets]
            for source, targets in (getattr(graph, "edges", {}) or {}).items()
        }
        meta = getattr(graph, "meta", {}) or {}
        extensions = meta.get("extensions") or {}
        visibility = str(
            meta.get("context_visibility")
            or (extensions.get("context_visibility") or {}).get("type")
            or "shared"
        )
        if visibility not in {"shared", "topology_filtered"}:
            raise ValueError(f"unsupported context visibility {visibility!r}")
        self.context_visibility = visibility

    def reset(self, *, preserve_model_call_budget: bool = False) -> None:
        # 每个样本开始前清空轮次/日志，并重置记忆库（清 transcript/缓存）
        self.turns.clear()
        self.decisions.clear()
        self.tool_request_event_ids.clear()
        self.tool_request_records.clear()
        self.observed_tool_executions.clear()
        self.last_model_exchange = None
        self.last_controller_exchange = None
        self.team_memory = None
        self.current_attempt_event_id = None
        self.current_runtime_event_id = None
        self.current_group_chat_event_id = None
        if not preserve_model_call_budget:
            self.model_calls_started = 0
            self.model_call_budget_exhausted = False
        self.case_deadline_exceeded = False
        self.memory.reset()
