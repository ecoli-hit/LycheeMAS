"""Canonical framework-neutral TeamSpec v14 contracts.

TeamSpec v14 is already normalized authoring data. It is the single persisted
truth consumed by the Reference Executor and every framework adapter. Runtime
indexes and native framework objects are derived in memory and are never saved
as another team schema.
"""

from __future__ import annotations

import re
from typing import Any

from ...runtime.contracts.memory import normalize_memory_policy
from ...runtime.contracts.operations import declared_operations, normalize_operations
from ...runtime.contracts.team_graph import (
    control_nodes,
    data_relations,
    executable_nodes,
    member_nodes,
    node_map,
    node_uses_model,
    result_submitter_ids,
    shared_state_map,
)
from ...runtime.model.token_budget import normalize_model_context_policy

TEAM_SPEC_SCHEMA_VERSION = 14

NODE_KINDS = {"model_agent", "tool_executor", "function", "human", "team", "remote"}
CONTROL_TRIGGERS = {
    "trial_started",
    "completed",
    "failed",
    "output_emitted",
    "result_submitted",
    "always",
}
CONTROL_ACTIONS = {
    "activate",
    "select_next",
    "delegate",
    "handoff",
    "return",
    "finish",
    "cancel",
}
CONTROL_FAILURES = {"fail_trial", "continue", "try_next", "return_to_source"}
DATA_VIEWS = {"all", "latest", "latest_n", "from_source", "summary"}
SHARED_STATE_KINDS = {"message_channel", "structured_state", "artifact_store", "memory"}
SHARED_STATE_UPDATES = {"append", "replace", "merge", "commit"}
STATE_LIFETIMES = {"invocation", "trial", "run"}

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_TEAM_FIELDS = {
    "schema_version",
    "id",
    "metadata",
    "nodes",
    "relations",
    "shared_state",
    "lifecycle",
}


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _list(value: Any, *, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _reject_unknown(value: dict[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: " + ", ".join(sorted(unknown)))


def _identifier(value: Any, *, label: str, runtime_safe: bool = False) -> str:
    text = str(value or "").strip()
    pattern = _SAFE_NAME if runtime_safe else _SAFE_ID
    if not text or not pattern.fullmatch(text):
        raise ValueError(f"{label} has an invalid identifier {text!r}")
    return text


def safe_runtime_name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(value or fallback)).strip("_")
    if not text:
        text = fallback
    return f"node_{text}" if text[0].isdigit() else text


def _unique_ids(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        duplicates = sorted({item for item in values if values.count(item) > 1})
        raise ValueError(f"{label} contains duplicate ids: " + ", ".join(duplicates))


def _normalize_provenance(raw: Any) -> dict[str, Any]:
    value = _object(raw, label="TeamSpec metadata.provenance")
    _reject_unknown(value, {"track", "evidence_level", "sources", "notes"}, label="provenance")
    track = str(value.get("track") or "custom")
    if track not in {
        "controlled_portability",
        "native_replication",
        "literature_replication",
        "custom",
    }:
        raise ValueError(f"unsupported TeamSpec provenance track {track!r}")
    evidence = str(value.get("evidence_level") or "hypothesis")
    if evidence not in {"official", "paper_reproduction", "paper_inspired", "hypothesis"}:
        raise ValueError(f"unsupported TeamSpec provenance evidence_level {evidence!r}")
    sources: list[dict[str, str]] = []
    for index, item in enumerate(_list(value.get("sources"), label="provenance.sources")):
        source = _object(item, label=f"provenance.sources[{index}]")
        _reject_unknown(source, {"title", "url", "scope"}, label=f"provenance.sources[{index}]")
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        if not title or not url:
            raise ValueError("provenance source requires title and url")
        sources.append({"title": title, "url": url, "scope": str(source.get("scope") or "")})
    return {
        "track": track,
        "evidence_level": evidence,
        "sources": sources,
        "notes": str(value.get("notes") or ""),
    }


def _normalize_metadata(raw: Any, *, team_id: str) -> dict[str, Any]:
    value = _object(raw, label="TeamSpec metadata")
    _reject_unknown(value, {"name", "description", "tags", "provenance"}, label="metadata")
    return {
        "name": str(value.get("name") or team_id),
        "description": str(value.get("description") or ""),
        "tags": list(dict.fromkeys(map(str, _list(value.get("tags"), label="metadata.tags")))),
        "provenance": _normalize_provenance(value.get("provenance")),
    }


def _normalize_behavior(raw: Any, *, node_id: str, kind: str) -> dict[str, Any]:
    value = _object(raw, label=f"Node {node_id!r} behavior")
    _reject_unknown(value, {"type", "options"}, label=f"Node {node_id!r} behavior")
    default = "assistant" if kind == "model_agent" else "custom"
    behavior_type = _identifier(
        value.get("type") or default,
        label=f"Node {node_id!r} behavior.type",
        runtime_safe=True,
    )
    options = _object(value.get("options"), label=f"Node {node_id!r} behavior.options")
    return {"type": behavior_type, "options": options}


def _normalize_tools(raw: Any, *, node_id: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for index, item in enumerate(_list(raw, label=f"Node {node_id!r} tools")):
        value = _object(item, label=f"Node {node_id!r} tools[{index}]")
        _reject_unknown(value, {"id", "required"}, label=f"Node {node_id!r} tools[{index}]")
        tools.append(
            {
                "id": _identifier(value.get("id"), label=f"Node {node_id!r} tool id"),
                "required": bool(value.get("required", True)),
            }
        )
    _unique_ids([item["id"] for item in tools], label=f"Node {node_id!r} tools")
    return tools


def _normalize_node(raw: Any, *, index: int) -> dict[str, Any]:
    value = _object(raw, label=f"nodes[{index}]")
    _reject_unknown(
        value,
        {
            "id",
            "name",
            "kind",
            "behavior",
            "purpose",
            "instructions",
            "capabilities",
            "operations",
            "tools",
            "context",
            "limits",
        },
        label=f"nodes[{index}]",
    )
    node_id = _identifier(value.get("id"), label=f"nodes[{index}].id", runtime_safe=True)
    kind = str(value.get("kind") or "").strip()
    if kind not in NODE_KINDS:
        raise ValueError(f"Node {node_id!r} has unsupported kind {kind!r}")
    capabilities = list(
        dict.fromkeys(
            map(str, _list(value.get("capabilities"), label=f"Node {node_id!r} capabilities"))
        )
    )
    limits = _object(value.get("limits"), label=f"Node {node_id!r} limits")
    _reject_unknown(
        limits,
        {"max_tool_iterations", "on_tool_limit"},
        label=f"Node {node_id!r} limits",
    )
    max_tool_iterations = int(limits.get("max_tool_iterations", 1))
    if max_tool_iterations < 1:
        raise ValueError(f"Node {node_id!r} limits.max_tool_iterations must be >= 1")
    on_tool_limit = str(limits.get("on_tool_limit") or "finalize").strip()
    if on_tool_limit not in {"finalize", "return_tool_result"}:
        raise ValueError(
            f"Node {node_id!r} limits.on_tool_limit must be one of "
            "'finalize' or 'return_tool_result'"
        )
    normalized_limits: dict[str, Any] = {"max_tool_iterations": max_tool_iterations}
    if "on_tool_limit" in limits:
        normalized_limits["on_tool_limit"] = on_tool_limit
    return {
        "id": node_id,
        "name": str(value.get("name") or node_id),
        "kind": kind,
        "behavior": _normalize_behavior(value.get("behavior"), node_id=node_id, kind=kind),
        "purpose": str(value.get("purpose") or node_id),
        "instructions": str(value.get("instructions") or ""),
        "capabilities": capabilities,
        "operations": normalize_operations(value.get("operations"), node_id=node_id),
        "tools": _normalize_tools(value.get("tools"), node_id=node_id),
        "context": normalize_model_context_policy(value.get("context") or {"type": "unbounded"}),
        "limits": normalized_limits,
    }


def _normalize_condition(raw: Any, *, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _object(raw, label=label)
    _reject_unknown(value, {"source", "operator", "value"}, label=label)
    source = str(value.get("source") or "").strip()
    operator = str(value.get("operator") or "").strip()
    if not source:
        raise ValueError(f"{label}.source is required")
    if operator not in {
        "eq",
        "ne",
        "in",
        "not_in",
        "exists",
        "truthy",
        "falsy",
        "lt",
        "lte",
        "gt",
        "gte",
    }:
        raise ValueError(f"{label}.operator is unsupported")
    if operator not in {"exists", "truthy", "falsy"} and "value" not in value:
        raise ValueError(f"{label}.value is required for operator {operator!r}")
    normalized = {"source": source, "operator": operator}
    if "value" in value:
        normalized["value"] = value["value"]
    return normalized


def _normalize_transfer(raw: Any, *, label: str) -> dict[str, Any]:
    value = _object(raw, label=label)
    _reject_unknown(
        value,
        {"source", "target", "required", "view", "latest_n", "filter", "transform", "schema"},
        label=label,
    )
    source = str(value.get("source") or "").strip()
    target = str(value.get("target") or "").strip()
    if not source or not target:
        raise ValueError(f"{label} requires source and target")
    view = str(value.get("view") or "all")
    if view not in DATA_VIEWS:
        raise ValueError(f"{label}.view is unsupported: {view!r}")
    latest_n = value.get("latest_n")
    if view == "latest_n":
        latest_n = int(latest_n or 0)
        if latest_n < 1:
            raise ValueError(f"{label}.latest_n must be >= 1 when view=latest_n")
    elif latest_n is not None:
        raise ValueError(f"{label}.latest_n is only valid when view=latest_n")
    transform = value.get("transform")
    return {
        "source": source,
        "target": target,
        "required": bool(value.get("required", False)),
        "view": view,
        "latest_n": latest_n,
        "filter": _object(value.get("filter"), label=f"{label}.filter") or None,
        "transform": None
        if transform is None
        else _identifier(transform, label=f"{label}.transform"),
        "schema": _object(value.get("schema"), label=f"{label}.schema"),
    }


def _normalize_control(raw: Any, *, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _object(raw, label=label)
    _reject_unknown(
        value,
        {"trigger", "action", "condition", "priority", "on_failure", "max_uses"},
        label=label,
    )
    trigger = str(value.get("trigger") or "completed")
    action = str(value.get("action") or "activate")
    if trigger not in CONTROL_TRIGGERS:
        raise ValueError(f"{label}.trigger is unsupported: {trigger!r}")
    if action not in CONTROL_ACTIONS:
        raise ValueError(f"{label}.action is unsupported: {action!r}")
    on_failure = str(value.get("on_failure") or "fail_trial")
    if on_failure not in CONTROL_FAILURES:
        raise ValueError(f"{label}.on_failure is unsupported: {on_failure!r}")
    max_uses = value.get("max_uses")
    if max_uses is not None and int(max_uses) < 1:
        raise ValueError(f"{label}.max_uses must be >= 1")
    return {
        "trigger": trigger,
        "action": action,
        "condition": _normalize_condition(value.get("condition"), label=f"{label}.condition"),
        "priority": int(value.get("priority", 0)),
        "on_failure": on_failure,
        "max_uses": None if max_uses is None else int(max_uses),
    }


def _normalize_data(raw: Any, *, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _object(raw, label=label)
    _reject_unknown(value, {"transfers"}, label=label)
    transfers = [
        _normalize_transfer(item, label=f"{label}.transfers[{index}]")
        for index, item in enumerate(_list(value.get("transfers"), label=f"{label}.transfers"))
    ]
    if not transfers:
        raise ValueError(f"{label}.transfers must not be empty")
    return {"transfers": transfers}


def _normalize_relation(raw: Any, *, index: int) -> dict[str, Any]:
    value = _object(raw, label=f"relations[{index}]")
    _reject_unknown(value, {"id", "from", "to", "control", "data"}, label=f"relations[{index}]")
    relation_id = _identifier(value.get("id"), label=f"relations[{index}].id")
    source = _identifier(
        value.get("from"), label=f"Relation {relation_id!r}.from", runtime_safe=True
    )
    target = _identifier(value.get("to"), label=f"Relation {relation_id!r}.to", runtime_safe=True)
    control = _normalize_control(value.get("control"), label=f"Relation {relation_id!r}.control")
    data = _normalize_data(value.get("data"), label=f"Relation {relation_id!r}.data")
    if control is None and data is None:
        raise ValueError(f"Relation {relation_id!r} requires control and/or data")
    return {"id": relation_id, "from": source, "to": target, "control": control, "data": data}


def _normalize_shared_state(raw: Any, *, index: int) -> dict[str, Any]:
    value = _object(raw, label=f"shared_state[{index}]")
    _reject_unknown(
        value,
        {
            "id",
            "kind",
            "description",
            "readers",
            "writers",
            "update",
            "lifetime",
            "initial",
            "schema",
            "retention",
            "memory",
        },
        label=f"shared_state[{index}]",
    )
    state_id = _identifier(value.get("id"), label=f"shared_state[{index}].id")
    kind = str(value.get("kind") or "").strip()
    if kind not in SHARED_STATE_KINDS:
        raise ValueError(f"SharedState {state_id!r} has unsupported kind {kind!r}")
    default_update = {
        "message_channel": "append",
        "structured_state": "merge",
        "artifact_store": "commit",
        "memory": "append",
    }[kind]
    update = str(value.get("update") or default_update)
    if update not in SHARED_STATE_UPDATES:
        raise ValueError(f"SharedState {state_id!r} has unsupported update {update!r}")
    lifetime = str(value.get("lifetime") or "trial")
    if lifetime not in STATE_LIFETIMES:
        raise ValueError(f"SharedState {state_id!r} has unsupported lifetime {lifetime!r}")
    retention = _object(value.get("retention"), label=f"SharedState {state_id!r} retention")
    _reject_unknown(
        retention,
        {"max_items", "max_tokens", "overflow"},
        label=f"SharedState {state_id!r} retention",
    )
    overflow = str(retention.get("overflow") or "drop_oldest")
    if overflow not in {"drop_oldest", "reject", "error"}:
        raise ValueError(f"SharedState {state_id!r} retention.overflow is unsupported")
    memory = value.get("memory")
    if kind == "memory":
        memory = normalize_memory_policy(memory, state_id=state_id)
    elif memory is not None:
        raise ValueError(f"SharedState {state_id!r} memory is only valid when kind='memory'")
    normalized = {
        "id": state_id,
        "kind": kind,
        "description": str(value.get("description") or ""),
        "readers": list(
            dict.fromkeys(
                map(str, _list(value.get("readers"), label=f"SharedState {state_id!r} readers"))
            )
        ),
        "writers": list(
            dict.fromkeys(
                map(str, _list(value.get("writers"), label=f"SharedState {state_id!r} writers"))
            )
        ),
        "update": update,
        "lifetime": lifetime,
        "initial": value.get("initial", [] if kind in {"message_channel", "memory"} else {}),
        "schema": _object(value.get("schema"), label=f"SharedState {state_id!r} schema"),
        "retention": {
            "max_items": retention.get("max_items"),
            "max_tokens": retention.get("max_tokens"),
            "overflow": overflow,
        },
    }
    if memory is not None:
        normalized["memory"] = memory
    return normalized


def _normalize_lifecycle(raw: Any) -> dict[str, Any]:
    value = _object(raw, label="TeamSpec lifecycle")
    _reject_unknown(
        value, {"entry", "result", "termination", "failure", "limits"}, label="lifecycle"
    )
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(_list(value.get("entry"), label="lifecycle.entry")):
        entry = _object(item, label=f"lifecycle.entry[{index}]")
        _reject_unknown(entry, {"node", "inputs"}, label=f"lifecycle.entry[{index}]")
        entries.append(
            {
                "node": _identifier(
                    entry.get("node"), label=f"lifecycle.entry[{index}].node", runtime_safe=True
                ),
                "inputs": [
                    _normalize_transfer(
                        transfer, label=f"lifecycle.entry[{index}].inputs[{offset}]"
                    )
                    for offset, transfer in enumerate(
                        _list(entry.get("inputs"), label=f"lifecycle.entry[{index}].inputs")
                    )
                ],
            }
        )
    if not entries:
        raise ValueError("TeamSpec lifecycle.entry must contain at least one activation")
    result = _object(value.get("result"), label="lifecycle.result")
    _reject_unknown(result, {"submissions", "mode", "schema"}, label="lifecycle.result")
    submissions: list[dict[str, str]] = []
    for index, item in enumerate(
        _list(result.get("submissions"), label="lifecycle.result.submissions")
    ):
        submission = _object(item, label=f"lifecycle.result.submissions[{index}]")
        _reject_unknown(
            submission, {"from", "source", "key"}, label=f"lifecycle.result.submissions[{index}]"
        )
        submissions.append(
            {
                "from": _identifier(
                    submission.get("from"),
                    label=f"lifecycle.result.submissions[{index}].from",
                    runtime_safe=True,
                ),
                "source": str(submission.get("source") or "source.output"),
                "key": str(submission.get("key") or "final_answer"),
            }
        )
    if not submissions:
        raise ValueError("TeamSpec lifecycle.result.submissions must not be empty")
    mode = str(result.get("mode") or "first_valid")
    if mode not in {"first_valid", "all", "aggregate"}:
        raise ValueError(f"unsupported lifecycle.result.mode {mode!r}")
    termination = _object(value.get("termination"), label="lifecycle.termination")
    _reject_unknown(termination, {"condition"}, label="lifecycle.termination")
    condition = str(termination.get("condition") or "result_submitted")
    if condition != "result_submitted":
        raise ValueError("TeamSpec v14 currently requires termination.condition=result_submitted")
    failure = _object(value.get("failure"), label="lifecycle.failure")
    _reject_unknown(failure, {"unhandled", "deadlock"}, label="lifecycle.failure")
    unhandled = str(failure.get("unhandled") or "fail_trial")
    deadlock = str(failure.get("deadlock") or "fail_trial")
    if unhandled not in {"fail_trial", "continue"}:
        raise ValueError("lifecycle.failure.unhandled is unsupported")
    if deadlock not in {"fail_trial", "submit_best_effort"}:
        raise ValueError("lifecycle.failure.deadlock is unsupported")
    limits = _object(value.get("limits"), label="lifecycle.limits")
    _reject_unknown(
        limits,
        {"max_turns", "max_node_calls", "max_stalls", "timeout_seconds"},
        label="lifecycle.limits",
    )
    normalized_limits: dict[str, int | float | None] = {}
    for field in ("max_turns", "max_node_calls", "max_stalls"):
        raw_limit = limits.get(field)
        if raw_limit is not None and int(raw_limit) < 1:
            raise ValueError(f"lifecycle.limits.{field} must be >= 1")
        normalized_limits[field] = None if raw_limit is None else int(raw_limit)
    timeout = limits.get("timeout_seconds")
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("lifecycle.limits.timeout_seconds must be > 0")
    normalized_limits["timeout_seconds"] = None if timeout is None else float(timeout)
    return {
        "entry": entries,
        "result": {
            "submissions": submissions,
            "mode": mode,
            "schema": _object(result.get("schema"), label="lifecycle.result.schema"),
        },
        "termination": {"condition": condition},
        "failure": {"unhandled": unhandled, "deadlock": deadlock},
        "limits": normalized_limits,
    }


def _validate_references(document: dict[str, Any]) -> None:
    nodes = set(node_map(document))
    states = set(shared_state_map(document))
    for relation in document["relations"]:
        if relation["from"] not in nodes or relation["to"] not in nodes:
            raise ValueError(f"Relation {relation['id']!r} references an unknown Node")
        for transfer in (relation.get("data") or {}).get("transfers") or []:
            source = str(transfer["source"])
            if source.startswith("shared.") and source.split(".", 1)[1] not in states:
                raise ValueError(
                    f"Relation {relation['id']!r} references unknown SharedState {source!r}"
                )
    lifecycle = document["lifecycle"]
    referenced = [item["node"] for item in lifecycle["entry"]]
    referenced.extend(item["from"] for item in lifecycle["result"]["submissions"])
    unknown = sorted(set(referenced) - nodes)
    if unknown:
        raise ValueError("TeamSpec lifecycle references unknown Nodes: " + ", ".join(unknown))
    for state in document["shared_state"]:
        unknown_access = sorted((set(state["readers"]) | set(state["writers"])) - nodes)
        if unknown_access:
            raise ValueError(
                f"SharedState {state['id']!r} references unknown Nodes: "
                + ", ".join(unknown_access)
            )
    for node in document["nodes"]:
        for options in declared_operations(node).values():
            state_refs = [options.get("state_target"), options.get("state_source")]
            state_refs.extend(options.get("state_sources") or [])
            for state_id in (str(item) for item in state_refs if item is not None):
                if state_id not in states:
                    raise ValueError(
                        f"Node {node['id']!r} operation references unknown SharedState {state_id!r}"
                    )


def normalize_team_spec_document(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the only persisted TeamSpec v14 representation."""

    document = _object(value, label="TeamSpec")
    _reject_unknown(document, _TEAM_FIELDS, label="TeamSpec")
    if int(document.get("schema_version") or 0) != TEAM_SPEC_SCHEMA_VERSION:
        raise ValueError(f"TeamSpec requires schema_version {TEAM_SPEC_SCHEMA_VERSION}")
    team_id = _identifier(document.get("id"), label="TeamSpec.id")
    nodes = [
        _normalize_node(item, index=index)
        for index, item in enumerate(_list(document.get("nodes"), label="TeamSpec.nodes"))
    ]
    if not nodes:
        raise ValueError("TeamSpec requires at least one Node")
    _unique_ids([item["id"] for item in nodes], label="TeamSpec nodes")
    relations = [
        _normalize_relation(item, index=index)
        for index, item in enumerate(_list(document.get("relations"), label="TeamSpec.relations"))
    ]
    _unique_ids([item["id"] for item in relations], label="TeamSpec relations")
    states = [
        _normalize_shared_state(item, index=index)
        for index, item in enumerate(
            _list(document.get("shared_state"), label="TeamSpec.shared_state")
        )
    ]
    _unique_ids([item["id"] for item in states], label="TeamSpec shared_state")
    normalized = {
        "schema_version": TEAM_SPEC_SCHEMA_VERSION,
        "id": team_id,
        "metadata": _normalize_metadata(document.get("metadata"), team_id=team_id),
        "nodes": nodes,
        "relations": relations,
        "shared_state": states,
        "lifecycle": _normalize_lifecycle(document.get("lifecycle")),
    }
    _validate_references(normalized)
    return normalized


def model_resource_requirements(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for node in executable_nodes(team_spec):
        if not node_uses_model(node):
            continue
        behavior = str((node.get("behavior") or {}).get("type") or "assistant")
        required_capabilities = ["text_generation"]
        if behavior == "web_navigator" or "vision" in set(node.get("capabilities") or []):
            required_capabilities.append("vision")
        requirements.append(
            {
                "node_id": str(node["id"]),
                "requirement": "model_inference",
                "resource_instance_type": "DeploymentInstance",
                "required": True,
                "behavior": behavior,
                "required_capabilities": required_capabilities,
            }
        )
    return requirements


__all__ = [
    "CONTROL_ACTIONS",
    "CONTROL_TRIGGERS",
    "NODE_KINDS",
    "TEAM_SPEC_SCHEMA_VERSION",
    "control_nodes",
    "data_relations",
    "executable_nodes",
    "member_nodes",
    "model_resource_requirements",
    "node_map",
    "normalize_team_spec_document",
    "result_submitter_ids",
    "safe_runtime_name",
    "shared_state_map",
]
