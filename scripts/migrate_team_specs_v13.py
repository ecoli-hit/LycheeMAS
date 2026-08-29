#!/usr/bin/env python3
"""One-time mechanical migration of registered TeamSpec v12 documents to v13."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "configs/eval_studio/teams/specs"


def _control_contract(value: dict[str, Any] | None) -> dict[str, Any]:
    """Keep decision policy only; candidates and cardinality come from graph ports/edges."""

    contract = dict(value or {})
    contract.pop("candidate_source", None)
    contract.pop("cardinality", None)
    return contract


def _port(
    port_id: str,
    kind: str,
    semantic_type: str,
    description: str,
    *,
    required: bool | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": port_id,
        "kind": kind,
        "semantic_type": semantic_type,
        "description": description,
        "schema": {},
        "contract": dict(contract or {}),
    }
    if required is not None:
        value["required"] = required
    return value


def _ensure_port(ports: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if not any(item["id"] == value["id"] for item in ports):
        ports.append(value)


def _store_type(resource_kind: str) -> tuple[str, str | None]:
    return {
        "message_channel": ("message", None),
        "artifact_store": ("artifact", None),
        "task_board": ("state", "task_board"),
        "plan_store": ("state", "plan"),
        "progress_ledger": ("state", "progress"),
        "result_sink": ("result", None),
    }[resource_kind]


def _store_ports(store_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic = {
        "input": "task",
        "message": "message",
        "state": "state",
        "artifact": "artifact",
        "result": "result",
    }[store_type]
    input_id = {
        "input": "initialize",
        "message": "append",
        "state": "merge",
        "artifact": "write",
        "result": "commit",
    }[store_type]
    output_id = {
        "input": "task",
        "message": "messages",
        "state": "state",
        "artifact": "artifacts",
        "result": "result",
    }[store_type]
    event_id = (
        "ready" if store_type == "input" else "committed" if store_type == "result" else "updated"
    )
    return (
        [_port(input_id, "data", semantic, f"Accept {semantic} data", required=False)],
        [
            _port(output_id, "data", semantic, f"Provide {semantic} data"),
            _port(event_id, "control", event_id, f"Store {event_id} event"),
        ],
    )


def _convert_store(node: dict[str, Any]) -> dict[str, Any]:
    resource = dict(node["resource"])
    store_type, state_role = _store_type(str(resource["kind"]))
    input_ports, output_ports = _store_ports(store_type)
    implementation: dict[str, Any] = {
        "kind": "store",
        "store_type": store_type,
        "lifetime": str(resource.get("scope") or "trial"),
        "update_policy": {
            "input": "initialize",
            "message": "append",
            "state": "merge",
            "artifact": "append",
            "result": "commit",
        }[store_type],
        "schema": dict(resource.get("schema") or resource.get("output_schema") or {}),
        "conditions": list(resource.get("conditions") or []),
    }
    if state_role:
        implementation["state_role"] = state_role
    if resource.get("tool_results"):
        implementation["tool_results"] = dict(resource["tool_results"])
    role = node.get("role") or {}
    return {
        "id": str(node["id"]),
        "node_type": "store",
        "identity": {
            "name": str(role.get("name") or node["id"]),
            "description": str(role.get("description") or role.get("name") or node["id"]),
        },
        "interface": {
            "input_ports": input_ports,
            "output_ports": output_ports,
            "activation": {"mode": "any"},
        },
        "implementation": implementation,
    }


def _convert_executable(node: dict[str, Any]) -> dict[str, Any]:
    execution = dict(node["execution"])
    old_kind = str(execution["kind"])
    kind = {
        "model_agent": "model",
        "deterministic": "function",
        "executor": "executor",
        "team": "team",
        "remote": "remote",
        "human": "human",
    }[old_kind]
    role = node.get("role") or {}
    capabilities = list(dict.fromkeys(str(item) for item in node.get("capabilities") or []))
    operations = dict(node.get("operations") or {})
    if set(operations) & {
        "select_next",
        "plan",
        "delegate",
        "monitor_progress",
        "detect_stall",
        "replan",
        "aggregate",
    }:
        capabilities.append("coordinate")
    capabilities.extend(item for item in operations if item not in capabilities)
    implementation: dict[str, Any] = {
        "kind": kind,
        "capabilities": list(dict.fromkeys(capabilities)),
        "tools": list(node.get("tools") or []),
    }
    if kind == "model":
        implementation.update(
            {
                "system_prompt": str(role.get("instructions") or ""),
                "context": dict(node.get("context") or {"type": "unbounded"}),
                "max_tool_iterations": int(execution.get("max_tool_iterations") or 1),
            }
        )
    for field in (
        "implementation_ref",
        "executor_type",
        "team_spec_ref",
        "endpoint_ref",
    ):
        if execution.get(field):
            implementation[field] = execution[field]

    inputs = [
        _port("activate", "control", "activate", "Activate this Node", required=False),
        _port("task", "data", "task", "Task visible to this Node", required=True),
    ]
    outputs = [
        _port("completed", "control", "completed", "Node completed"),
        _port("failed", "control", "failed", "Node failed"),
    ]
    for operation, options in operations.items():
        if operation in {"select_next", "delegate", "handoff", "replan"}:
            _ensure_port(
                outputs,
                _port(
                    operation,
                    "control",
                    operation,
                    f"Emit {operation} decision",
                    contract=_control_contract(options),
                ),
            )
        elif operation == "detect_stall":
            _ensure_port(
                outputs,
                _port(
                    "stall_detected",
                    "control",
                    "stall_detected",
                    "Signal that progress has stalled",
                    contract=dict(options or {}),
                ),
            )
        elif operation in {"plan", "decompose", "monitor_progress"}:
            _ensure_port(
                outputs,
                _port(
                    f"{operation}_update",
                    "data",
                    "state",
                    f"Structured {operation} update",
                    contract={"state_role": operation, **dict(options or {})},
                ),
            )
        elif operation in {"aggregate", "submit"}:
            _ensure_port(
                outputs,
                _port(
                    "result",
                    "data",
                    "result",
                    "Final team result",
                    contract=dict(options or {}),
                ),
            )
    return {
        "id": str(node["id"]),
        "node_type": "executable",
        "identity": {
            "name": str(role.get("name") or node["id"]),
            "description": str(role.get("description") or role.get("name") or node["id"]),
        },
        "interface": {
            "input_ports": inputs,
            "output_ports": outputs,
            "activation": {"mode": "any"},
        },
        "implementation": implementation,
    }


def _node(nodes: dict[str, dict[str, Any]], node_id: str) -> dict[str, Any]:
    return nodes[node_id]


def _ensure_input(node: dict[str, Any], port_id: str, semantic: str, description: str) -> None:
    _ensure_port(
        node["interface"]["input_ports"],
        _port(port_id, "data", semantic, description, required=False),
    )


def _ensure_output(node: dict[str, Any], port_id: str, semantic: str, description: str) -> None:
    _ensure_port(
        node["interface"]["output_ports"],
        _port(port_id, "data", semantic, description),
    )


def _edge(
    edge_id: str,
    edge_type: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    *,
    description: str = "",
    condition: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    operation: str | None = None,
    semantic_type: str | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": edge_id,
        "edge_type": edge_type,
        "source": {"node": source_node, "port": source_port},
        "target": {"node": target_node, "port": target_port},
        "enabled": True,
        "description": description,
        "condition": dict(condition or {}),
        "mapping": {},
        "policy": dict(policy or {}),
    }
    if edge_type == "control":
        value["control"] = dict(control or {})
    else:
        value["data"] = {
            "operation": str(operation or "transfer"),
            "semantic_type": str(semantic_type or "generic"),
        }
    return value


def convert(document: dict[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) == 13:
        cleaned = json.loads(json.dumps(document))
        for node in cleaned.get("nodes") or []:
            for port in (node.get("interface") or {}).get("output_ports") or []:
                if port.get("kind") == "control":
                    port["contract"] = _control_contract(port.get("contract"))
        return cleaned
    if int(document.get("schema_version") or 0) != 12:
        raise ValueError(f"cannot migrate TeamSpec schema {document.get('schema_version')!r}")

    converted_nodes = [
        _convert_executable(node) if node.get("execution") else _convert_store(node)
        for node in document.get("nodes") or []
    ]
    task_input = {
        "id": "TaskInput",
        "node_type": "store",
        "identity": {
            "name": "Task Input",
            "description": "Trial-scoped original task input",
        },
        "interface": {
            "input_ports": [
                _port("initialize", "data", "task", "Initialize the Trial task", required=False)
            ],
            "output_ports": [
                _port("task", "data", "task", "Provide the Trial task"),
                _port("ready", "control", "ready", "Task input is ready"),
            ],
            "activation": {"mode": "any"},
        },
        "implementation": {
            "kind": "store",
            "store_type": "input",
            "lifetime": "trial",
            "update_policy": "initialize",
            "schema": {},
            "conditions": [],
        },
    }
    if not any(node["id"] == "TaskInput" for node in converted_nodes):
        converted_nodes.insert(0, task_input)
    by_id = {str(node["id"]): node for node in converted_nodes}
    executable_ids = [
        str(node["id"]) for node in converted_nodes if node["node_type"] == "executable"
    ]
    old_nodes = {str(node["id"]): node for node in document.get("nodes") or []}

    edges: list[dict[str, Any]] = []
    for node_id in executable_ids:
        edges.append(
            _edge(
                f"task-TaskInput-{node_id}",
                "data",
                "TaskInput",
                "task",
                node_id,
                "task",
                operation="read",
                semantic_type="task",
            )
        )

    old_relations = [
        relation for relation in document.get("relations") or [] if relation.get("enabled", True)
    ]
    for relation in old_relations:
        relation_id = str(relation["id"])
        relation_type = str(relation["type"])
        source = str(relation["source"])
        target = str(relation["target"])
        contract = dict(relation.get("contract") or {})
        if relation_type in {"control", "dependency", "handoff"}:
            source_port = (
                str(contract.get("operation") or "select_next")
                if relation_type == "control"
                else "handoff"
                if relation_type == "handoff"
                else "completed"
            )
            if source_port == "activate":
                source_port = "completed"
            semantic = source_port
            if semantic not in {
                "select_next",
                "delegate",
                "handoff",
                "replan",
                "completed",
            }:
                semantic = "completed"
                source_port = "completed"
            _ensure_port(
                _node(by_id, source)["interface"]["output_ports"],
                _port(source_port, "control", semantic, f"Emit {semantic}"),
            )
            edges.append(
                _edge(
                    relation_id,
                    "control",
                    source,
                    source_port,
                    target,
                    "activate",
                    description=str(relation.get("description") or ""),
                    condition=dict(contract.get("guard") or {}),
                    control={
                        "trigger": str(contract.get("trigger") or "source_emitted"),
                        "priority": int(contract.get("priority") or 0),
                        "fallback": str(contract.get("fallback") or "error"),
                        "on_failure": str(contract.get("on_failure") or "fail_trial"),
                    },
                )
            )
            continue
        if relation_type in {"publish", "subscribe", "message"}:
            if relation_type == "publish":
                _ensure_output(_node(by_id, source), "response", "message", "Published message")
                source_port, target_port, operation = "response", "append", "append"
            elif relation_type == "subscribe":
                _ensure_input(_node(by_id, target), "messages", "message", "Visible messages")
                source_port, target_port, operation = "messages", "messages", "read"
            else:
                _ensure_output(_node(by_id, source), "response", "message", "Direct message")
                _ensure_input(_node(by_id, target), "messages", "message", "Visible messages")
                source_port, target_port, operation = "response", "messages", "transfer"
            edges.append(
                _edge(
                    relation_id,
                    "data",
                    source,
                    source_port,
                    target,
                    target_port,
                    description=str(relation.get("description") or ""),
                    policy=contract,
                    operation=operation,
                    semantic_type="message",
                )
            )
            continue
        if relation_type in {"read", "write", "artifact", "submit"}:
            if relation_type == "submit":
                _ensure_output(_node(by_id, source), "result", "result", "Final result")
                source_port, target_port = "result", "commit"
                operation, semantic = "commit", "result"
                policy = contract
            elif relation_type == "read":
                store_type = str(_node(by_id, source)["implementation"]["store_type"])
                semantic = "artifact" if store_type == "artifact" else "state"
                source_port = "artifacts" if semantic == "artifact" else "state"
                target_port = "artifacts" if semantic == "artifact" else "state"
                _ensure_input(_node(by_id, target), target_port, semantic, f"Readable {semantic}")
                operation, policy = "read", contract
            elif relation_type == "write":
                store_type = str(_node(by_id, target)["implementation"]["store_type"])
                semantic = "artifact" if store_type == "artifact" else "state"
                source_port = "artifacts" if semantic == "artifact" else "state_update"
                target_port = "write" if semantic == "artifact" else "merge"
                _ensure_output(_node(by_id, source), source_port, semantic, f"Produced {semantic}")
                operation, policy = "write" if semantic == "artifact" else "merge", contract
            else:
                semantic = "artifact"
                source_port, target_port = "artifacts", "artifacts"
                _ensure_output(_node(by_id, source), source_port, semantic, "Produced artifact")
                _ensure_input(_node(by_id, target), target_port, semantic, "Received artifact")
                operation, policy = "transfer", contract
            edges.append(
                _edge(
                    relation_id,
                    "data",
                    source,
                    source_port,
                    target,
                    target_port,
                    description=str(relation.get("description") or ""),
                    policy=policy,
                    operation=operation,
                    semantic_type=semantic,
                )
            )

    operation_nodes = [
        str(node_id)
        for node_id, node in old_nodes.items()
        if node.get("operations")
        and set(node.get("operations") or {})
        & {"select_next", "delegate", "plan", "monitor_progress", "detect_stall", "replan"}
    ]
    if operation_nodes:
        entry_nodes = operation_nodes
    else:
        participant_ids = [
            node_id
            for node_id, node in old_nodes.items()
            if node.get("execution") and "participant" in set(node.get("activation_modes") or [])
        ]
        dependency_targets = {
            str(item["target"]) for item in old_relations if item.get("type") == "dependency"
        }
        entry_nodes = [node_id for node_id in participant_ids if node_id not in dependency_targets]
        if not entry_nodes and participant_ids:
            entry_nodes = participant_ids[:1]
    for node_id in entry_nodes:
        edges.insert(
            0,
            _edge(
                f"control-TaskInput-{node_id}",
                "control",
                "TaskInput",
                "ready",
                node_id,
                "activate",
                control={
                    "trigger": "source_emitted",
                    "priority": 0,
                    "fallback": "error",
                    "on_failure": "fail_trial",
                },
            ),
        )

    for controller_id in operation_nodes:
        candidates = {
            edge["target"]["node"]
            for edge in edges
            if edge["edge_type"] == "control"
            and edge["source"]["node"] == controller_id
            and edge["target"]["node"] != controller_id
        }
        for worker_id in sorted(candidates):
            return_id = f"control-{worker_id}-return-{controller_id}"
            if not any(edge["id"] == return_id for edge in edges):
                edges.append(
                    _edge(
                        return_id,
                        "control",
                        worker_id,
                        "completed",
                        controller_id,
                        "activate",
                        control={
                            "trigger": "source_emitted",
                            "priority": 0,
                            "fallback": "error",
                            "on_failure": "fail_trial",
                        },
                    )
                )

    result_node = next(
        node
        for node in converted_nodes
        if node["node_type"] == "store" and node["implementation"]["store_type"] == "result"
    )
    identity = dict(document.get("identity") or {})
    return {
        "schema_version": 13,
        "id": str(document["id"]),
        "metadata": {
            "name": str(identity.get("name") or document["id"]),
            "description": str(identity.get("description") or ""),
            "tags": [],
            "provenance": dict(document.get("provenance") or {}),
        },
        "nodes": converted_nodes,
        "edges": edges,
        "execution_policy": {
            "ready_order": "priority_fifo",
            "unhandled_failure": "fail_trial",
            "deadlock_policy": "fail_trial",
            "completion": {"node": result_node["id"], "port": "committed"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = 0
    for path in sorted(args.root.glob("*.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        target = convert(source)
        if target == source:
            continue
        changed += 1
        if not args.check:
            path.write_text(
                json.dumps(target, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps({"root": str(args.root), "changed": changed, "check": args.check}))


if __name__ == "__main__":
    main()
