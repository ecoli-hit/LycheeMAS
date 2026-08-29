#!/usr/bin/env python3
"""One-time TeamSpec v13 to canonical v14 migration.

The Eval runtime does not load v13. This script exists only to convert the
repository's checked-in definitions and may also be used on an archived v13
file before importing it into a v14 workspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _operation_list(raw: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for kind, raw_options in raw.items():
        if kind == "submit":
            continue
        options = dict(raw_options or {})
        for obsolete in ("candidate_source", "cardinality", "output_target"):
            options.pop(obsolete, None)
        values.append({"type": str(kind), "options": options})
    return values


def _behavior(node: dict[str, Any]) -> str:
    implementation = dict(node.get("implementation") or {})
    kind = str(implementation.get("kind") or "model")
    capabilities = set(map(str, implementation.get("capabilities") or []))
    operations = set((implementation.get("operations") or {}).keys())
    if kind == "executor" and implementation.get("executor_type") == "code":
        return "code_executor"
    orchestration_operations = {
        "plan",
        "delegate",
        "monitor_progress",
        "detect_stall",
        "replan",
        "aggregate",
    }
    if orchestration_operations <= operations:
        return "orchestrator"
    if "select_next" in operations:
        return "selector"
    if {"browse", "search_web"} <= capabilities:
        return "web_navigator"
    if {"read_files", "inspect_artifacts"} <= capabilities:
        return "file_navigator"
    if "write_code" in capabilities:
        return "code_author"
    return "assistant" if kind == "model" else "custom"


def _node(node: dict[str, Any]) -> dict[str, Any]:
    implementation = dict(node.get("implementation") or {})
    old_kind = str(implementation.get("kind") or "model")
    kind = {
        "model": "model_agent",
        "executor": "tool_executor",
        "function": "function",
        "human": "human",
        "team": "team",
        "remote": "remote",
    }[old_kind]
    identity = dict(node.get("identity") or {})
    return {
        "id": str(node["id"]),
        "name": str(identity.get("name") or node["id"]),
        "kind": kind,
        "behavior": {"type": _behavior(node), "options": {}},
        "purpose": str(identity.get("description") or identity.get("name") or node["id"]),
        "instructions": str(implementation.get("system_prompt") or ""),
        "capabilities": list(map(str, implementation.get("capabilities") or [])),
        "operations": _operation_list(dict(implementation.get("operations") or {})),
        "tools": [
            {"id": str(item), "required": True}
            for item in implementation.get("tools") or []
        ],
        "context": dict(implementation.get("context") or {"type": "unbounded"}),
        "limits": {
            "max_tool_iterations": max(1, int(implementation.get("max_tool_iterations", 1)))
        },
    }


def _state_access(
    document: dict[str, Any], state_id: str, executable_ids: set[str]
) -> tuple[list[str], list[str]]:
    readers: list[str] = []
    writers: list[str] = []
    for edge in document.get("edges") or []:
        if not edge.get("enabled", True) or edge.get("edge_type") != "data":
            continue
        source = str((edge.get("source") or {}).get("node") or "")
        target = str((edge.get("target") or {}).get("node") or "")
        operation = str((edge.get("data") or {}).get("operation") or "transfer")
        if source == state_id and target in executable_ids and operation == "read":
            readers.append(target)
        if target == state_id and source in executable_ids and operation in {
            "write",
            "append",
            "merge",
            "commit",
        }:
            writers.append(source)
    return list(dict.fromkeys(readers)), list(dict.fromkeys(writers))


def _shared_state(document: dict[str, Any], executable_ids: set[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for node in document.get("nodes") or []:
        if node.get("node_type") != "store":
            continue
        implementation = dict(node.get("implementation") or {})
        old_kind = str(implementation.get("store_type") or "state")
        if old_kind in {"input", "result"}:
            continue
        state_id = str(node["id"])
        readers, writers = _state_access(document, state_id, executable_ids)
        identity = dict(node.get("identity") or {})
        kind = {
            "message": "message_channel",
            "state": "structured_state",
            "artifact": "artifact_store",
        }.get(old_kind, "structured_state")
        values.append(
            {
                "id": state_id,
                "kind": kind,
                "description": str(identity.get("description") or identity.get("name") or ""),
                "readers": readers,
                "writers": writers,
                "update": str(implementation.get("update_policy") or "merge"),
                "lifetime": str(implementation.get("lifetime") or "trial"),
                "initial": [] if kind == "message_channel" else {},
                "schema": dict(implementation.get("schema") or {}),
                "retention": {
                    "max_items": None,
                    "max_tokens": None,
                    "overflow": "drop_oldest",
                },
            }
        )
    return values


def _condition(raw: dict[str, Any]) -> dict[str, Any] | None:
    value = dict(raw or {})
    if not value or value.get("type") == "always":
        return None
    if {"source", "operator"} <= set(value):
        return {key: value[key] for key in ("source", "operator", "value") if key in value}
    return None


def _relation_data(target: str, states: list[dict[str, Any]]) -> dict[str, Any]:
    transfers = [
        {
            "source": "trial.task",
            "target": "task",
            "required": True,
            "view": "all",
            "latest_n": None,
            "filter": None,
            "transform": None,
            "schema": {},
        }
    ]
    for state in states:
        if target not in state["readers"]:
            continue
        transfers.append(
            {
                "source": f"shared.{state['id']}",
                "target": "messages" if state["kind"] == "message_channel" else "state",
                "required": False,
                "view": "all",
                "latest_n": None,
                "filter": None,
                "transform": None,
                "schema": {},
            }
        )
    return {"transfers": transfers}


def _relations(
    document: dict[str, Any], executable_ids: set[str], states: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for edge in document.get("edges") or []:
        if not edge.get("enabled", True) or edge.get("edge_type") != "control":
            continue
        source = str((edge.get("source") or {}).get("node") or "")
        target = str((edge.get("target") or {}).get("node") or "")
        if source not in executable_ids or target not in executable_ids:
            continue
        source_port = str((edge.get("source") or {}).get("port") or "completed")
        action = {
            "select_next": "select_next",
            "delegate": "delegate",
            "handoff": "handoff",
            "return": "return",
            "finish": "finish",
        }.get(source_port, "activate")
        trigger = "failed" if source_port == "failed" else "completed"
        control = dict(edge.get("control") or {})
        values.append(
            {
                "id": str(edge["id"]),
                "from": source,
                "to": target,
                "control": {
                    "trigger": trigger,
                    "action": action,
                    "condition": _condition(dict(edge.get("condition") or {})),
                    "priority": int(control.get("priority") or 0),
                    "on_failure": str(control.get("on_failure") or "fail_trial"),
                    "max_uses": None,
                },
                "data": _relation_data(target, states),
            }
        )
    return values


def _entry_nodes(document: dict[str, Any], executable_ids: set[str]) -> list[str]:
    values: list[str] = []
    input_ids = {
        str(node["id"])
        for node in document.get("nodes") or []
        if node.get("node_type") == "store"
        and (node.get("implementation") or {}).get("store_type") == "input"
    }
    for edge in document.get("edges") or []:
        source = str((edge.get("source") or {}).get("node") or "")
        target = str((edge.get("target") or {}).get("node") or "")
        if edge.get("edge_type") == "control" and source in input_ids and target in executable_ids:
            values.append(target)
    return list(dict.fromkeys(values))


def _submissions(document: dict[str, Any], executable_ids: set[str]) -> list[dict[str, str]]:
    result_ids = {
        str(node["id"])
        for node in document.get("nodes") or []
        if node.get("node_type") == "store"
        and (node.get("implementation") or {}).get("store_type") == "result"
    }
    values: list[dict[str, str]] = []
    for edge in document.get("edges") or []:
        source = str((edge.get("source") or {}).get("node") or "")
        target = str((edge.get("target") or {}).get("node") or "")
        if (
            edge.get("edge_type") != "data"
            or source not in executable_ids
            or target not in result_ids
        ):
            continue
        values.append({"from": source, "source": "source.output", "key": "final_answer"})
    return values


def migrate(document: dict[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != 13:
        raise ValueError("migration input must be TeamSpec v13")
    executable = [
        node for node in document.get("nodes") or [] if node.get("node_type") == "executable"
    ]
    executable_ids = {str(node["id"]) for node in executable}
    states = _shared_state(document, executable_ids)
    relations = _relations(document, executable_ids, states)
    entry = _entry_nodes(document, executable_ids)
    incoming = {node_id: 0 for node_id in executable_ids}
    for relation in relations:
        incoming[relation["to"]] += 1
    entry = entry or [node_id for node_id in executable_ids if incoming[node_id] == 0]
    submissions = _submissions(document, executable_ids)
    submissions = submissions or [
        {"from": str(executable[-1]["id"]), "source": "source.output", "key": "final_answer"}
    ]
    policy = dict(document.get("execution_policy") or {})
    return {
        "schema_version": 14,
        "id": str(document["id"]),
        "metadata": dict(document.get("metadata") or {}),
        "nodes": [_node(node) for node in executable],
        "relations": relations,
        "shared_state": states,
        "lifecycle": {
            "entry": [
                {
                    "node": node_id,
                    "inputs": [
                        {
                            "source": "trial.task",
                            "target": "task",
                            "required": True,
                            "view": "all",
                            "latest_n": None,
                            "filter": None,
                            "transform": None,
                            "schema": {},
                        }
                    ],
                }
                for node_id in entry
            ],
            "result": {"submissions": submissions, "mode": "first_valid", "schema": {}},
            "termination": {"condition": "result_submitted"},
            "failure": {
                "unhandled": str(policy.get("unhandled_failure") or "fail_trial"),
                "deadlock": str(policy.get("deadlock_policy") or "fail_trial"),
            },
            "limits": {
                "max_turns": None,
                "max_node_calls": None,
                "max_stalls": None,
                "timeout_seconds": None,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate(document)
        if args.check:
            print(path)
            continue
        path.write_text(json.dumps(migrated, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
