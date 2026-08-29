"""Load canonical TeamSpec v14 and build one transient runtime view.

The JSON document remains the only persisted Team definition. ``MASGraph`` is
an attempt-local index containing executable member handles and adapter reports;
it is not serialized and does not expand v14 back into Port/Store/Edge objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...core.types import AgentSpec
from ...runtime.adapters.frameworks.bindings import build_framework_binding_report
from ...runtime.contracts.operations import declared_operations
from ...runtime.contracts.runtime import MASGraph
from ...runtime.contracts.team_graph import member_nodes, node_map, result_submitter_ids
from ...runtime.coordination.compiler import (
    compile_coordination,
    coordination_control_node_id,
    coordination_topology,
    message_policies,
    message_sources,
    message_visibility,
)
from .contracts import normalize_team_spec_document


def _agent_from_node(node: dict[str, Any], index: int) -> AgentSpec:
    tools = [str(item["id"]) for item in node.get("tools") or []]
    behavior = dict(node.get("behavior") or {})
    node_kind = str(node["kind"])
    return AgentSpec(
        id=str(node["id"]),
        name=str(node["id"]),
        role=str(node.get("purpose") or node["id"]),
        system_prompt=str(node.get("instructions") or ""),
        tools=tools,
        meta={
            "node_id": str(node["id"]),
            "node_name": str(node.get("name") or node["id"]),
            "node_kind": node_kind,
            "execution_kind": "executor" if node_kind == "tool_executor" else "model",
            "behavior": behavior,
            "behavior_type": str(behavior.get("type") or "custom"),
            "description": str(node.get("purpose") or node["id"]),
            "capabilities": list(node.get("capabilities") or []),
            "operations": declared_operations(node),
            "model_context": dict(node.get("context") or {"type": "unbounded"}),
            "max_tool_iterations": int((node.get("limits") or {}).get("max_tool_iterations", 1)),
            "on_tool_limit": str((node.get("limits") or {}).get("on_tool_limit") or "finalize"),
            "declaration_index": index,
        },
    )


def _control_adjacency(
    document: dict[str, Any], member_ids: set[str]
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    adjacency = {node_id: [] for node_id in member_ids}
    typed: list[dict[str, Any]] = []
    for relation in document.get("relations") or []:
        control = relation.get("control")
        source = str(relation["from"])
        target = str(relation["to"])
        if (
            control
            and source in member_ids
            and target in member_ids
            and target not in adjacency[source]
        ):
            adjacency[source].append(target)
        if control:
            typed.append(
                {
                    "id": str(relation["id"]),
                    "edge_type": "control",
                    "source": {"node": source},
                    "target": {"node": target},
                    "control": dict(control),
                }
            )
        if relation.get("data"):
            typed.append(
                {
                    "id": f"{relation['id']}:data",
                    "edge_type": "data",
                    "source": {"node": source},
                    "target": {"node": target},
                    "data": dict(relation["data"]),
                }
            )
    return adjacency, typed


def _result_contract(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document["lifecycle"]["result"])
    return {
        "submitters": result_submitter_ids(document),
        "submissions": list(result.get("submissions") or []),
        "mode": str(result.get("mode") or "first_valid"),
        "schema": dict(result.get("schema") or {}),
        "conditions": [{"type": "result_submitted"}],
    }


def load_team_spec(path: str | Path, *, rounds: int) -> tuple[MASGraph, dict[str, Any]]:
    document = normalize_team_spec_document(json.loads(Path(path).read_text(encoding="utf-8")))
    coordination = compile_coordination(document)
    raw_by_id = {node["id"]: node for node in member_nodes(document)}
    raw_members = [raw_by_id[node_id] for node_id in coordination["ordered_members"]]
    nodes = [_agent_from_node(node, index) for index, node in enumerate(raw_members)]
    names_by_id = {str(node["id"]): str(node["id"]) for node in raw_members}
    member_ids = set(names_by_id)
    edges, typed_relations = _control_adjacency(document, member_ids)

    visible_sources = message_sources(document, coordination["members"])
    visible_policies = message_policies(document, coordination["members"])
    for raw, node in zip(raw_members, nodes, strict=True):
        node_id = str(raw["id"])
        node.meta["receives_from"] = [
            names_by_id[source]
            for source in visible_sources.get(node_id, [])
            if source in names_by_id
        ]
        node.meta["message_policies"] = {
            names_by_id[source]: dict(policy)
            for source, policy in visible_policies.get(node_id, {}).items()
            if source in names_by_id
        }
        node.meta["sources"] = list(node.meta["receives_from"])
        node.meta["handoffs"] = [
            str(relation["to"])
            for relation in document.get("relations") or []
            if relation.get("from") == node_id
            and (relation.get("control") or {}).get("action") == "handoff"
            and relation.get("to") in member_ids
        ]
        node.meta["handoff_contracts"] = {
            str(relation["to"]): dict(relation["control"])
            for relation in document.get("relations") or []
            if relation.get("from") == node_id
            and (relation.get("control") or {}).get("action") == "handoff"
        }
        node.meta["control_predecessors"] = [
            str(relation["from"])
            for relation in document.get("relations") or []
            if relation.get("to") == node_id and relation.get("control")
        ]
        node.meta["resource_access"] = [
            {
                "id": str(relation["id"]),
                "source": str(relation["from"]),
                "target": str(relation["to"]),
                "transfers": list((relation.get("data") or {}).get("transfers") or []),
            }
            for relation in document.get("relations") or []
            if relation.get("data") and node_id in {str(relation["from"]), str(relation["to"])}
        ]

    binding_reports = {
        framework: build_framework_binding_report(framework, coordination, document)
        for framework in ("autogen", "langgraph", "crewai")
    }
    bindings_by_framework = {
        framework: {
            str(binding["node_id"]): dict(binding) for binding in report.get("node_bindings") or []
        }
        for framework, report in binding_reports.items()
    }
    for node in nodes:
        node.meta["framework_bindings"] = {
            framework: dict(bindings[node.id])
            for framework, bindings in bindings_by_framework.items()
            if node.id in bindings
        }

    control_node_id = coordination_control_node_id(coordination)
    all_nodes = node_map(document)
    control_node = all_nodes.get(str(control_node_id)) if control_node_id else None
    result_contract = _result_contract(document)
    operation_bindings = list(coordination.get("operation_bindings") or [])
    operation_nodes = {
        str(binding["operation"]): all_nodes.get(str(binding["node_id"]))
        for binding in operation_bindings
    }
    visibility = message_visibility(document, coordination["members"])
    message_states = [
        str(item["id"])
        for item in document.get("shared_state") or []
        if item.get("kind") == "message_channel"
    ]
    artifact_states = [
        str(item["id"])
        for item in document.get("shared_state") or []
        if item.get("kind") == "artifact_store"
    ]
    memory_states = [
        dict(item) for item in document.get("shared_state") or [] if item.get("kind") == "memory"
    ]
    default_context = next(
        (dict(node.get("context") or {}) for node in raw_members if node.get("context")),
        {"type": "unbounded"},
    )
    graph = MASGraph(
        nodes=nodes,
        edges=edges,
        rounds=rounds,
        meta={
            "team": document["id"],
            "team_spec_schema_version": document["schema_version"],
            "team_spec": document,
            "provenance": dict(document["metadata"]["provenance"]),
            "coordination_ir": coordination,
            "adapter_plans": {
                framework: dict(report["adapter_plan"])
                for framework, report in binding_reports.items()
            },
            "framework_binding_reports": binding_reports,
            "relations": list(document["relations"]),
            "lifecycle": dict(document["lifecycle"]),
            "communication": {
                "mode": "group_chat",
                "visibility": {"type": visibility},
                "stores": message_states,
                "channels": message_states,
                "tool_results": {"max_inline_chars": 50000, "overflow": "head_tail"},
            },
            "state": {
                "scope": "trial",
                "artifact_stores": artifact_states,
                "memory_states": memory_states,
            },
            "result_contract": result_contract,
            "typed_edges": typed_relations,
            "operation_bindings": operation_bindings,
            "operation_nodes": operation_nodes,
            "control_node_id": control_node_id,
            "model_context": default_context,
            "controller_model_context": (
                dict(control_node.get("context") or default_context)
                if control_node
                else default_context
            ),
            "context_visibility": visibility,
            "dynamic_topology": visibility == "topology_filtered",
            "speaking_order": [names_by_id[item] for item in coordination["ordered_members"]],
            "topology_type": coordination_topology(coordination),
        },
    )
    return graph, document


def team_details_from_graph(graph: MASGraph) -> dict[str, Any]:
    document = dict(graph.meta["team_spec"])
    details_by_id = node_map(document)
    members = []
    for node in graph.nodes:
        raw = details_by_id[str(node.id)]
        members.append(
            {
                "id": str(node.id),
                "name": str(raw.get("name") or node.name),
                "kind": str(raw["kind"]),
                "behavior": dict(raw.get("behavior") or {}),
                "execution_kind": str(node.meta.get("execution_kind") or raw["kind"]),
                "purpose": str(raw.get("purpose") or ""),
                "instructions": str(raw.get("instructions") or ""),
                "capabilities": list(raw.get("capabilities") or []),
                "operations": list(raw.get("operations") or []),
                "tools": list(raw.get("tools") or []),
                "context": dict(raw.get("context") or {}),
                "framework_bindings": dict(node.meta.get("framework_bindings") or {}),
                "deployment_id": node.meta.get("deployment_id"),
                "receives_from": list(node.meta.get("receives_from") or []),
            }
        )
    control_id = graph.meta.get("control_node_id")
    control = details_by_id.get(str(control_id)) if control_id else None
    group_chat_type = str(
        ((graph.meta.get("adapter_plans") or {}).get("autogen") or {}).get("strategy")
        or "graph_flow"
    )
    topology_type = str(graph.meta.get("topology_type") or "explicit_graph")
    return {
        "id": document["id"],
        "schema_version": document["schema_version"],
        "metadata": dict(document["metadata"]),
        "members": members,
        "control_node": control,
        "nodes": list(document["nodes"]),
        "relations": list(document["relations"]),
        "shared_state": list(document["shared_state"]),
        "lifecycle": dict(document["lifecycle"]),
        "group_chat_type": group_chat_type,
        "adapter_plans": dict(graph.meta.get("adapter_plans") or {}),
        "framework_binding_reports": dict(graph.meta.get("framework_binding_reports") or {}),
        "result_contract": dict(graph.meta.get("result_contract") or {}),
        "topology_type": topology_type,
        "speaking_order": list(graph.meta.get("speaking_order") or []),
        # Runner/event projection fields. They describe the normalized v14 graph;
        # they are not accepted as persisted TeamSpec inputs.
        "group_chat": group_chat_type,
        "chat_mode": topology_type,
        "runtime_roles": members,
        "context_visibility": "explicit_data_relations",
    }


__all__ = ["load_team_spec", "normalize_team_spec_document", "team_details_from_graph"]
