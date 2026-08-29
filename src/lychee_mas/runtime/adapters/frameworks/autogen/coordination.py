"""Compile framework-neutral coordination into AutoGen callbacks and limits."""

from __future__ import annotations

from typing import Any, Callable, Optional

from lychee_mas.runtime.contracts.runtime import MASGraph
from lychee_mas.runtime.coordination.group_chat import normalize_group_chat_config


def group_chat_config(graph: MASGraph) -> dict[str, Any]:
    plan = dict((graph.meta.get("adapter_plans") or {}).get("autogen") or {})
    return normalize_group_chat_config(plan.get("config"))


def operation_selector(graph: MASGraph, candidate_node_ids: list[str]) -> Callable:
    names_by_id = {str(node.id): node.name for node in graph.nodes}
    candidates = [names_by_id[item] for item in candidate_node_ids if item in names_by_id]
    if not candidates:
        raise ValueError("deterministic select_next requires at least one explicit candidate")
    state = {"index": 0}

    def select(_messages) -> str:
        selected = candidates[state["index"] % len(candidates)]
        state["index"] += 1
        return selected

    return select


def bounded_handoff_selector(graph: MASGraph) -> Callable:
    coordination = dict(graph.meta.get("coordination_ir") or {})
    members = [str(item) for item in (coordination.get("ordered_members") or [])]
    if not members:
        raise ValueError("bounded handoff requires at least one executable member")
    member_set = set(members)
    relations = sorted(
        coordination.get("handoff_relations") or [],
        key=lambda item: (
            int(((item.get("contract") or {}).get("priority") or 0)),
            str(item.get("id") or ""),
        ),
    )
    targets: dict[str, list[str]] = {}
    for relation in relations:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        if source in member_set and target in member_set:
            targets.setdefault(source, []).append(target)
    options_by_node = {
        str(binding.get("node_id") or ""): dict(binding.get("options") or {})
        for binding in ((coordination.get("operations") or {}).get("handoff") or [])
    }
    state: dict[str, str | int] = {"current": members[0], "attempts": 0}

    def select(messages) -> str:
        from autogen_agentchat.messages import HandoffMessage

        if not messages:
            return str(state["current"])
        latest = messages[-1]
        if isinstance(latest, HandoffMessage) and latest.target in member_set:
            state["current"] = str(latest.target)
            state["attempts"] = 0
            return str(state["current"])
        source = str(getattr(latest, "source", "") or "")
        if source not in member_set:
            return str(state["current"])
        if source != state["current"]:
            state["current"] = source
            state["attempts"] = 0
        state["attempts"] = int(state["attempts"]) + 1
        options = options_by_node.get(source) or {}
        max_attempts = max(1, int(options.get("max_attempts") or 1))
        if int(state["attempts"]) < max_attempts:
            return source
        fallback = str(options.get("fallback") or "error")
        if fallback != "next_priority":
            raise ValueError(
                f"bounded handoff for {source!r} requires fallback='next_priority'"
            )
        candidates = targets.get(source) or []
        if not candidates:
            raise ValueError(f"bounded handoff for {source!r} has no legal target")
        state["current"] = candidates[0]
        state["attempts"] = 0
        return str(state["current"])

    return select


def operation_candidates(graph: MASGraph, candidate_node_ids: list[str]) -> Callable:
    names_by_id = {str(node.id): node.name for node in graph.nodes}
    candidates = [names_by_id[item] for item in candidate_node_ids if item in names_by_id]
    if not candidates:
        raise ValueError("model select_next requires at least one explicit candidate")

    def candidates_for_turn(_messages) -> list[str]:
        return list(candidates)

    return candidates_for_turn


def topology_selector(graph: MASGraph) -> Callable:
    """Preserve original per-record topology routing outside TeamSpec v12."""

    order = list(graph.meta.get("speaking_order") or graph.names)
    indexes = {name: index for index, name in enumerate(order)}
    state = {"fallback": 0}

    def source(message) -> str:
        return str(getattr(message, "source", "") or getattr(message, "name", ""))

    def select(messages) -> str:
        previous = next((source(item) for item in reversed(messages) if source(item)), "")
        allowed = list(graph.edges.get(previous) or [])
        if allowed:
            return min(allowed, key=lambda name: (indexes.get(name, len(order)), name))
        if previous in indexes and order:
            return order[(indexes[previous] + 1) % len(order)]
        selected = order[state["fallback"] % len(order)]
        state["fallback"] += 1
        return selected

    return select


def topology_candidates(graph: MASGraph) -> Callable:
    order = list(graph.meta.get("speaking_order") or graph.names)
    indexes = {name: index for index, name in enumerate(order)}

    def source(message) -> str:
        return str(getattr(message, "source", "") or getattr(message, "name", ""))

    def candidates_for_turn(messages) -> list[str]:
        previous = next((source(item) for item in reversed(messages) if source(item)), "")
        allowed = list(graph.edges.get(previous) or [])
        if allowed:
            return sorted(allowed, key=lambda name: (indexes.get(name, len(order)), name))
        if previous in indexes and order:
            return [order[(indexes[previous] + 1) % len(order)]]
        return list(order)

    return candidates_for_turn


def termination_condition(team: MASGraph, ctx: Any | None = None):
    from autogen_agentchat.base import TerminatedException, TerminationCondition
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.messages import StopMessage

    conditions = list((team.meta.get("termination") or {}).get("conditions") or [])
    compiled: list[Any] = []
    for raw in conditions:
        item = dict(raw) if isinstance(raw, dict) else {"type": str(raw)}
        condition_type = str(item.get("type") or "").lower()
        if condition_type in {"approve", "terminate"}:
            item = {"type": "text_mention", "text": condition_type.upper()}
            condition_type = "text_mention"
        if condition_type != "text_mention":
            raise ValueError(f"unsupported termination condition {condition_type!r}")
        text = str(item.get("text") or "")
        if not text:
            raise ValueError("text_mention termination requires non-empty text")
        sources = item.get("sources")
        compiled.append(
            TextMentionTermination(
                text,
                sources=[str(source) for source in sources] if sources else None,
            )
        )
    result_contract = dict(team.meta.get("result_contract") or {})
    result_conditions = list(result_contract.get("conditions") or [])
    if any(
        str(item.get("type") or "") == "result_submitted"
        if isinstance(item, dict)
        else str(item) == "result_submitted"
        for item in result_conditions
    ):
        submitters = {
            str(item) for item in (result_contract.get("submitters") or []) if str(item)
        }

        class ResultSubmittedTermination(TerminationCondition):
            """Stop after an approved submitter publishes a consumable final message."""

            def __init__(self) -> None:
                self._terminated = False

            @property
            def terminated(self) -> bool:
                return self._terminated

            async def __call__(self, messages):
                if self._terminated:
                    raise TerminatedException("Result submission already terminated the team")
                for message in messages:
                    source = str(getattr(message, "source", "") or "")
                    message_type = type(message).__name__
                    content = getattr(message, "content", None)
                    has_content = (
                        bool(content.strip()) if isinstance(content, str) else bool(content)
                    )
                    if (
                        source in submitters
                        and message_type
                        in {"TextMessage", "MultiModalMessage", "StructuredMessage"}
                        and has_content
                    ):
                        self._terminated = True
                        if ctx is not None:
                            ctx.log_event(
                                "result.submission_detected",
                                source=source,
                                message_type=message_type,
                                submitters=sorted(submitters),
                                origin="autogen_termination_adapter",
                            )
                        return StopMessage(
                            content=f"Result submitted by {source}.",
                            source="ResultSubmittedTermination",
                        )
                return None

            async def reset(self) -> None:
                self._terminated = False

        compiled.append(ResultSubmittedTermination())
    if not compiled:
        return None
    condition = compiled[0]
    for extra in compiled[1:]:
        condition = condition | extra
    return condition


def model_call_limit_termination(ctx: Any, limit: Optional[int]):
    if limit is None:
        return None
    from autogen_agentchat.base import TerminatedException, TerminationCondition
    from autogen_agentchat.messages import StopMessage

    class ModelCallLimitTermination(TerminationCondition):
        def __init__(self) -> None:
            self._terminated = False

        @property
        def terminated(self) -> bool:
            return self._terminated

        async def __call__(self, _messages):
            if self._terminated:
                raise TerminatedException("Model-call limit termination already reached")
            if int(getattr(ctx, "model_calls_started", 0)) >= limit:
                self._terminated = True
                ctx.model_call_budget_exhausted = True
                ctx.log_event(
                    "model_call.budget_stopped",
                    model_calls_started=ctx.model_calls_started,
                    max_model_calls_per_case=limit,
                )
                return StopMessage(
                    content=f"Maximum model calls per case {limit} reached.",
                    source="ModelCallLimitTermination",
                )
            return None

        async def reset(self) -> None:
            self._terminated = False

    return ModelCallLimitTermination()


def effective_max_turns(
    group_chat_type: str,
    participant_count: int,
    *,
    max_rounds: int,
    max_turns: Optional[int],
) -> int:
    if max_turns is not None:
        return int(max_turns)
    if group_chat_type == "round_robin":
        return max(1, participant_count) * int(max_rounds)
    return 20


__all__ = [
    "bounded_handoff_selector",
    "effective_max_turns",
    "group_chat_config",
    "model_call_limit_termination",
    "operation_candidates",
    "operation_selector",
    "termination_condition",
    "topology_candidates",
    "topology_selector",
]
