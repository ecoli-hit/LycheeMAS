"""Compile framework-neutral executable Nodes to LangGraph callables."""

from __future__ import annotations

from typing import Any

from lychee_mas.runtime.coordination.compiler import (
    operation_node_id,
    orchestration_node_id,
)


def bind_node(node: dict[str, Any], coordination: dict[str, Any]) -> dict[str, Any]:
    node_id = str(node["id"])
    kind = str(node.get("kind") or "")
    behavior = str((node.get("behavior") or {}).get("type") or "custom")
    runtime_role = (
        "coordination"
        if node_id in {
            orchestration_node_id(coordination),
            operation_node_id(coordination, "select_next"),
        }
        else "member"
    )
    if kind == "model_agent":
        implementation = "langgraph.graph.StateGraph model callable via ModelGateway"
        specialization = (
            "ledger_orchestration"
            if node_id == orchestration_node_id(coordination)
            else "model_selector"
            if node_id == operation_node_id(coordination, "select_next")
            else "model_node"
        )
        level = "composed" if runtime_role == "coordination" else "exact"
    elif kind == "tool_executor" and behavior == "code_executor":
        implementation = "langgraph.graph.StateGraph deterministic code-executor callable"
        specialization = "code_executor"
        level = "exact"
    else:
        implementation = "unsupported"
        specialization = "unsupported"
        level = "unsupported"
    return {
        "node_id": node_id,
        "framework": "langgraph",
        "node_kind": kind,
        "runtime_role": runtime_role,
        "runtime_implementation": implementation,
        "specialization": specialization,
        "mapping_level": level,
        "selection_basis": [
            f"kind={kind}",
            f"behavior={behavior}",
            f"runtime_role={runtime_role}",
        ],
        "declared_capabilities": list(node.get("capabilities") or []),
        "declared_tools": [str(item["id"]) for item in node.get("tools") or []],
    }


__all__ = ["bind_node"]
