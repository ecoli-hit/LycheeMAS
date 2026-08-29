"""Compile framework-neutral executable Nodes to AutoGen runtime components."""

from __future__ import annotations

from typing import Any

from lychee_mas.runtime.coordination.compiler import (
    operation_node_id,
    orchestration_node_id,
)


def _base_binding(node: dict[str, Any], *, implementation: str) -> dict[str, Any]:
    return {
        "node_id": str(node["id"]),
        "framework": "autogen",
        "node_kind": str(node.get("kind") or ""),
        "runtime_implementation": implementation,
        "mapping_level": "exact",
        "declared_capabilities": list(node.get("capabilities") or []),
        "declared_tools": [str(item["id"]) for item in node.get("tools") or []],
    }


def bind_node(node: dict[str, Any], coordination: dict[str, Any]) -> dict[str, Any]:
    """Select an AutoGen component from explicit portable Node semantics.

    Node identity and display names deliberately do not participate in this
    decision. Specialized AutoGen components are selected only from the Node's
    execution kind, operation ownership, capabilities, and explicit tools.
    """

    node_id = str(node["id"])
    kind = str(node.get("kind") or "")
    behavior = str((node.get("behavior") or {}).get("type") or "custom")
    tools = {str(item["id"]) for item in node.get("tools") or []}

    if node_id == orchestration_node_id(coordination):
        binding = _base_binding(
            node,
            implementation="autogen_agentchat.teams.MagenticOneGroupChat.internal_orchestrator",
        )
        binding.update(
            {
                "runtime_role": "coordination",
                "specialization": "magentic_one_orchestrator",
                "selection_basis": ["operation_bundle=orchestration"],
            }
        )
        return binding
    if node_id == operation_node_id(coordination, "select_next"):
        binding = _base_binding(
            node,
            implementation="autogen_agentchat.teams.SelectorGroupChat.model_selector",
        )
        binding.update(
            {
                "runtime_role": "coordination",
                "specialization": "model_selector",
                "selection_basis": ["operation=select_next"],
            }
        )
        return binding
    if kind == "tool_executor":
        executor_type = "code" if behavior == "code_executor" else behavior
        implementation = (
            "autogen_agentchat.agents.CodeExecutorAgent"
            if executor_type == "code"
            else "unsupported"
        )
        binding = _base_binding(node, implementation=implementation)
        binding.update(
            {
                "runtime_role": "member",
                "specialization": "code_executor" if executor_type == "code" else "unsupported",
                "selection_basis": ["kind=executor", f"executor_type={executor_type}"],
                "mapping_level": "exact" if executor_type == "code" else "unsupported",
            }
        )
        return binding
    if kind != "model_agent":
        binding = _base_binding(node, implementation="unsupported")
        binding.update(
            {
                "runtime_role": "member",
                "specialization": "unsupported",
                "selection_basis": [f"kind={kind}"],
                "mapping_level": "unsupported",
            }
        )
        return binding

    specialization = "assistant"
    runtime_implementation = "autogen_agentchat.agents.AssistantAgent"
    basis = ["kind=model_agent", f"behavior={behavior}"]
    if behavior == "web_navigator":
        specialization = "multimodal_web_surfer"
        runtime_implementation = "autogen_ext.agents.web_surfer.MultimodalWebSurfer"
    elif behavior == "file_navigator":
        specialization = "file_surfer"
        runtime_implementation = "autogen_ext.agents.file_surfer.FileSurfer"
    elif behavior == "code_author" and not tools:
        specialization = "magentic_one_coder"
        runtime_implementation = "autogen_agentchat.agents.AssistantAgent"

    binding = _base_binding(node, implementation=runtime_implementation)
    binding.update(
        {
            "runtime_role": "member",
            "specialization": specialization,
            "selection_basis": basis,
        }
    )
    return binding


__all__ = ["bind_node"]
