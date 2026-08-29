"""Pure validation for the AutoGen GroupChat subset supported by LycheeMAS."""

from __future__ import annotations

from typing import Any

GROUP_CHAT_TYPES = {"round_robin", "selector", "magentic_one", "swarm", "graph_flow"}
SELECTOR_FUNC_FACTORIES = {
    "operation_selector",
    "topology_selector",
    "bounded_handoff_selector",
}
CANDIDATE_FUNC_FACTORIES = {"operation_candidates", "topology_candidates"}

_FIELDS_BY_TYPE = {
    "round_robin": {"type", "max_turns"},
    "selector": {
        "type",
        "selector_func_factory",
        "candidate_func_factory",
        "selector_prompt",
        "allow_repeated_speaker",
        "max_selector_attempts",
        "model_client_streaming",
        "max_turns",
        "candidate_node_ids",
    },
    "magentic_one": {
        "type",
        "max_stalls",
        "final_answer_prompt",
        "max_turns",
    },
    "swarm": {"type", "max_turns"},
    "graph_flow": {"type", "max_turns"},
}


def normalize_group_chat_config(value: Any) -> dict[str, Any]:
    """Validate and normalize one explicit TeamSpec ``group_chat`` object."""

    if not isinstance(value, dict):
        raise ValueError("TeamSpec requires a group_chat object")
    group_chat = dict(value)
    group_chat_type = str(group_chat.get("type") or "")
    if group_chat_type not in GROUP_CHAT_TYPES:
        raise ValueError(
            f"unsupported group_chat type {group_chat_type!r}; "
            f"choose {sorted(GROUP_CHAT_TYPES)}"
        )
    unknown = set(group_chat) - _FIELDS_BY_TYPE[group_chat_type]
    if unknown:
        raise ValueError(
            f"{group_chat_type} GroupChat does not support fields: "
            + ", ".join(sorted(unknown))
        )
    group_chat["type"] = group_chat_type
    if group_chat.get("max_turns") is not None:
        max_turns = int(group_chat["max_turns"])
        if max_turns < 1:
            raise ValueError("group_chat.max_turns must be >= 1")
        group_chat["max_turns"] = max_turns

    if group_chat_type == "selector":
        selector_factory = group_chat.get("selector_func_factory")
        candidate_factory = group_chat.get("candidate_func_factory")
        if selector_factory not in {None, "", *SELECTOR_FUNC_FACTORIES}:
            raise ValueError(f"unsupported selector_func_factory {selector_factory!r}")
        if candidate_factory not in {None, "", *CANDIDATE_FUNC_FACTORIES}:
            raise ValueError(f"unsupported candidate_func_factory {candidate_factory!r}")
        if selector_factory and candidate_factory:
            raise ValueError(
                "SelectorGroupChat accepts one selection strategy: "
                "selector_func_factory or candidate_func_factory"
            )
        attempts = int(group_chat.get("max_selector_attempts", 3))
        if attempts < 1:
            raise ValueError("max_selector_attempts must be >= 1")
        group_chat["max_selector_attempts"] = attempts
        group_chat["allow_repeated_speaker"] = bool(
            group_chat.get("allow_repeated_speaker", False)
        )
        streaming = bool(group_chat.get("model_client_streaming", False))
        if streaming:
            raise ValueError(
                "model_client_streaming=true is not supported by LycheeMAS model clients"
            )
        group_chat["model_client_streaming"] = False
        candidate_node_ids = group_chat.get("candidate_node_ids") or []
        if not isinstance(candidate_node_ids, list):
            raise ValueError("selector candidate_node_ids must be a list")
        if (
            selector_factory == "operation_selector"
            or candidate_factory == "operation_candidates"
        ) and not candidate_node_ids:
            raise ValueError("operation selector requires non-empty candidate_node_ids")
        group_chat["candidate_node_ids"] = list(dict.fromkeys(map(str, candidate_node_ids)))
    elif group_chat_type == "magentic_one":
        max_stalls = int(group_chat.get("max_stalls", 3))
        if max_stalls < 1:
            raise ValueError("max_stalls must be >= 1")
        group_chat["max_stalls"] = max_stalls

    return group_chat
