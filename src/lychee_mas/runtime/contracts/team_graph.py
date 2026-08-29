"""Read-only indexes over one normalized TeamSpec v14 document.

These helpers never create a second persisted team schema. They derive small
lookups from the canonical Node, Relation, SharedState, and Lifecycle facts and
may be rebuilt by each adapter at negligible cost.
"""

from __future__ import annotations

from typing import Any

from .operations import ORCHESTRATION_OPERATION_KINDS, declared_operations


def node_map(team_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in team_spec.get("nodes") or []}


def executable_nodes(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Every v14 Node is executable; persistent state lives in shared_state."""

    return list(team_spec.get("nodes") or [])


def node_is_executable(node: dict[str, Any]) -> bool:
    return str(node.get("kind") or "") in {
        "model_agent",
        "tool_executor",
        "function",
        "human",
        "team",
        "remote",
    }


def node_is_store(node: dict[str, Any], store_type: str | None = None) -> bool:
    del node, store_type
    return False


def node_uses_model(node: dict[str, Any]) -> bool:
    return str(node.get("kind") or "") == "model_agent"


def control_nodes(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Nodes that explicitly own central coordination operations."""

    values: list[dict[str, Any]] = []
    for node in executable_nodes(team_spec):
        operations = set(declared_operations(node))
        if "select_next" in operations or ORCHESTRATION_OPERATION_KINDS <= operations:
            values.append(node)
    return values


def member_nodes(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return workers while preserving control Nodes in the canonical document."""

    controllers = {str(node["id"]) for node in control_nodes(team_spec)}
    members = [node for node in executable_nodes(team_spec) if str(node["id"]) not in controllers]
    return members or executable_nodes(team_spec)


def relation_map(team_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in team_spec.get("relations") or []}


def control_relations(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in team_spec.get("relations") or [] if item.get("control")]


def data_relations(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in team_spec.get("relations") or [] if item.get("data")]


def shared_state_map(team_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in team_spec.get("shared_state") or []}


def result_submitter_ids(team_spec: dict[str, Any]) -> list[str]:
    result = dict((team_spec.get("lifecycle") or {}).get("result") or {})
    return [str(item["from"]) for item in result.get("submissions") or []]


__all__ = [
    "control_nodes",
    "control_relations",
    "data_relations",
    "executable_nodes",
    "member_nodes",
    "node_is_executable",
    "node_is_store",
    "node_map",
    "node_uses_model",
    "relation_map",
    "result_submitter_ids",
    "shared_state_map",
]
