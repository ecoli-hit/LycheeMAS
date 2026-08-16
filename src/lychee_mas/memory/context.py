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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .routing.base import MemoryRouter, RouteDecision

if TYPE_CHECKING:  # 仅类型检查期需要；避免与 memory.base 形成 import 环
    from .base import MemoryManager


class ModelCallBudgetExceeded(RuntimeError):
    """Raised before a model request would exceed the per-case hard limit."""


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
    span_logger: Any = None  # 可选 JsonlSpanLogger：运行中实时写 spans.jsonl
    group_chat_logger: Any = None  # 可选 JsonlGroupChatLogger：实时写 group_chat.jsonl
    current_case_id: Optional[str] = None
    current_sample_index: Optional[int] = None
    current_k_index: Optional[int] = None
    current_attempt: Optional[int] = None
    current_case_span_id: Optional[str] = None
    base_seed: int = 0
    prediction_seed: Optional[int] = None
    seed_derivation: Optional[str] = None
    generation_seed: Optional[int] = None
    trace_model_calls: bool = True  # 是否在终端打印每次 LLM 调用的 start/done 摘要
    trace_detail_level: str = "compact"  # compact/full；仅控制 model_call_start 三层消息
    max_model_calls_per_case: Optional[int] = None  # None=不限制；包含 participant/controller
    model_calls_started: int = 0  # 已经发送或尝试发送的真实后端请求
    model_call_budget_exhausted: bool = False
    context_visibility: str = "shared"  # AutoGen 默认 shared；可显式 topology_filtered
    last_model_exchange: Optional[dict[str, Any]] = None  # C2C 内存态，不写入 predictions
    graph_edges: Dict[str, List[str]] = field(default_factory=dict)

    def turn_of(self, role: str) -> int:
        return self.turns.get(role, 0)  # 读取某角色当前轮次（默认 0）

    def bump_turn(self, role: str) -> None:
        self.turns[role] = self.turns.get(role, 0) + 1  # 该角色发言后 +1

    def reserve_model_call(self, role: str, *, controller: bool = False) -> int:
        """Reserve one real backend request and return its one-based case index."""

        limit = self.max_model_calls_per_case
        if limit is not None and self.model_calls_started >= limit:
            self.model_call_budget_exhausted = True
            self.log_span(
                "model_call_budget_exhausted",
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

    def set_case(self, case_id: str, sample_index: int) -> None:
        self.current_case_id = str(case_id)
        self.current_sample_index = int(sample_index)

    def log_span(self, span_type: str, **fields: Any) -> Optional[str]:
        if self.span_logger is None:
            return None
        if "parent_span_id" not in fields and span_type != "case_start":
            fields["parent_span_id"] = self.current_case_span_id
        fields.setdefault("k_index", self.current_k_index)
        fields.setdefault("attempt", self.current_attempt)
        fields.setdefault("base_seed", self.base_seed)
        fields.setdefault("prediction_seed", self.prediction_seed)
        fields.setdefault("seed_derivation", self.seed_derivation)
        return self.span_logger.log(
            span_type,
            case_id=self.current_case_id,
            sample_index=self.current_sample_index,
            **fields,
        )

    def set_case_span(self, span_id: Optional[str]) -> None:
        self.current_case_span_id = span_id

    def log_group_chat(self, event_type: str, **fields: Any) -> Optional[str]:
        if self.group_chat_logger is None:
            return None
        fields.setdefault("k_index", self.current_k_index)
        fields.setdefault("attempt", self.current_attempt)
        fields.setdefault("base_seed", self.base_seed)
        fields.setdefault("prediction_seed", self.prediction_seed)
        fields.setdefault("seed_derivation", self.seed_derivation)
        return self.group_chat_logger.log(
            event_type,
            case_id=self.current_case_id,
            sample_index=self.current_sample_index,
            **fields,
        )

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
        self.last_model_exchange = None
        if not preserve_model_call_budget:
            self.model_calls_started = 0
            self.model_call_budget_exhausted = False
        self.memory.reset()
