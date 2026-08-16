"""Convert a deployment-independent Eval Studio TeamSpec to MASGraph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...core.types import AgentSpec
from ...layers.construct.templates import (
    ROLE_PROFILE_META,
    ROLE_PROFILES,
    team_to_agentspecs,
)
from ...runtime.base import MASGraph
from ...runtime.group_chat import normalize_group_chat_config
from ...runtime.token_budget import normalize_model_context_policy

_CONTEXT_VISIBILITY_TYPES = {"shared", "topology_filtered"}
_OBSOLETE_FIELDS = {
    "roles",
    "nodes",
    "graph",
    "protocol",
    "coordination",
    "communication",
    "scheduler",
    "team_preset",
}
_NON_MODEL_PARTICIPANT_TYPES = {"computer_terminal"}
_CONTROL_SLOT_BY_GROUP_CHAT = {
    "magentic_one": ("Orchestrator", "orchestrator"),
    "selector": ("Selector", "selector"),
}


def _name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(value or fallback)).strip("_")
    if not text:
        text = fallback
    return f"agent_{text}" if text[0].isdigit() else text


def _group_chat(document: dict[str, Any]) -> dict[str, Any]:
    return normalize_group_chat_config(document.get("group_chat"))


def _extensions(document: dict[str, Any]) -> dict[str, Any]:
    extensions = dict(document.get("extensions") or {})
    if "topology" in extensions:
        topology = dict(extensions.get("topology") or {})
        topology.setdefault("edges", {})
        topology.setdefault("speaking_order", [])
        extensions["topology"] = topology
    if "context_visibility" in extensions:
        visibility = dict(extensions.get("context_visibility") or {})
        visibility_type = str(visibility.get("type") or "shared")
        if visibility_type not in _CONTEXT_VISIBILITY_TYPES:
            raise ValueError(
                f"unsupported context visibility {visibility_type!r}; "
                f"choose {sorted(_CONTEXT_VISIBILITY_TYPES)}"
            )
        visibility["type"] = visibility_type
        extensions["context_visibility"] = visibility
    return extensions


def _participant_from_document(raw: dict[str, Any], index: int) -> AgentSpec:
    node_name = _name(raw.get("id") or raw.get("name"), f"agent_{index + 1}")
    meta = dict(raw.get("meta") or {})
    meta.pop("deployment_id", None)
    meta.update(
        {
            "description": raw.get("description") or raw.get("role") or node_name,
            "agent_type": raw.get("agent_type", "assistant"),
            "model_context": normalize_model_context_policy(raw.get("model_context")),
        }
    )
    return AgentSpec(
        id=str(raw.get("id") or node_name),
        name=node_name,
        role=str(raw.get("role") or node_name),
        system_prompt=str(raw.get("system_prompt") or ""),
        model=raw.get("model_id"),
        tools=[str(tool) for tool in (raw.get("tools") or [])],
        meta=meta,
    )


def _default_inference_slots(
    participants: list[dict[str, Any]], group_chat: dict[str, Any]
) -> list[dict[str, Any]]:
    slots = [
        {
            "id": str(item["id"]),
            "kind": "participant",
            "participant_id": str(item["id"]),
            "required_capabilities": ["text_generation"],
        }
        for item in participants
        if str(item.get("agent_type") or "assistant") not in _NON_MODEL_PARTICIPANT_TYPES
    ]
    control = _CONTROL_SLOT_BY_GROUP_CHAT.get(group_chat["type"])
    if control and not (
        group_chat["type"] == "selector" and group_chat.get("selector_func_factory")
    ):
        slot_id, controller_type = control
        slots.append(
            {
                "id": slot_id,
                "kind": "controller",
                "controller_type": controller_type,
                "required_capabilities": ["text_generation"],
            }
        )
    return slots


def _normalize_inference_slots(
    raw_slots: Any,
    *,
    participants: list[dict[str, Any]],
    group_chat: dict[str, Any],
) -> list[dict[str, Any]]:
    defaults = _default_inference_slots(participants, group_chat)
    slots = defaults if raw_slots is None else raw_slots
    if not isinstance(slots, list) or not slots:
        raise ValueError("TeamSpec requires at least one inference slot")
    participants_by_id = {str(item["id"]): item for item in participants}
    expected_controller = _CONTROL_SLOT_BY_GROUP_CHAT.get(group_chat["type"])
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(slots):
        if not isinstance(raw, dict):
            raise ValueError(f"inference slot at index {index} must be an object")
        slot_id = str(raw.get("id") or "").strip()
        if not slot_id or slot_id in seen:
            raise ValueError(f"invalid or duplicate inference slot id {slot_id!r}")
        seen.add(slot_id)
        kind = str(raw.get("kind") or "participant")
        required = [str(item) for item in raw.get("required_capabilities") or ["text_generation"]]
        if kind == "participant":
            participant_id = str(raw.get("participant_id") or slot_id)
            participant = participants_by_id.get(participant_id)
            if participant is None:
                raise ValueError(
                    f"inference slot {slot_id!r} references unknown participant {participant_id!r}"
                )
            if str(participant.get("agent_type") or "assistant") in _NON_MODEL_PARTICIPANT_TYPES:
                raise ValueError(
                    f"non-model participant {participant_id!r} cannot be an inference slot"
                )
            normalized.append(
                {
                    "id": slot_id,
                    "kind": kind,
                    "participant_id": participant_id,
                    "required_capabilities": required,
                }
            )
        elif kind == "controller":
            if expected_controller is None or (
                group_chat["type"] == "selector" and group_chat.get("selector_func_factory")
            ):
                raise ValueError(
                    f"TeamSpec group chat {group_chat['type']!r} has no model controller"
                )
            controller_type = str(raw.get("controller_type") or expected_controller[1])
            if controller_type != expected_controller[1]:
                raise ValueError(
                    f"controller slot {slot_id!r} must use {expected_controller[1]!r}"
                )
            normalized.append(
                {
                    "id": slot_id,
                    "kind": kind,
                    "controller_type": controller_type,
                    "required_capabilities": required,
                }
            )
        else:
            raise ValueError("inference slot kind must be participant or controller")
    expected_participants = {
        str(item["id"])
        for item in participants
        if str(item.get("agent_type") or "assistant") not in _NON_MODEL_PARTICIPANT_TYPES
    }
    bound_participants = {
        str(item["participant_id"])
        for item in normalized
        if item["kind"] == "participant"
    }
    if bound_participants != expected_participants:
        missing = sorted(expected_participants - bound_participants)
        extra = sorted(bound_participants - expected_participants)
        raise ValueError(
            "TeamSpec inference slots must cover every model participant exactly once"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + (f"; extra: {', '.join(extra)}" if extra else "")
        )
    controller_slots = [item for item in normalized if item["kind"] == "controller"]
    needs_controller = expected_controller is not None and not (
        group_chat["type"] == "selector" and group_chat.get("selector_func_factory")
    )
    if len(controller_slots) != int(needs_controller):
        raise ValueError(
            f"TeamSpec group chat {group_chat['type']!r} requires "
            f"{int(needs_controller)} controller inference slot"
        )
    return normalized


def normalize_team_spec_document(value: dict[str, Any]) -> dict[str, Any]:
    """Validate one standalone TeamSpec v4 without adding deployment state."""

    if not isinstance(value, dict):
        raise ValueError("team spec must be a JSON object")
    document = dict(value)
    if int(document.get("schema_version") or 0) != 4:
        raise ValueError("TeamSpec requires schema_version 4")
    obsolete = _OBSOLETE_FIELDS & set(document)
    if obsolete:
        raise ValueError(
            "obsolete TeamSpec fields are not supported: " + ", ".join(sorted(obsolete))
        )
    team_id = str(document.get("id") or "").strip()
    if not team_id:
        raise ValueError("TeamSpec requires a non-empty id")
    participants = document.get("participants") or []
    if not isinstance(participants, list):
        raise ValueError("team participants must be a list")
    profile = document.get("role_profile")
    if not participants and str(profile or "") not in ROLE_PROFILES:
        raise ValueError("TeamSpec requires participants or a known role_profile")
    names: set[str] = set()
    normalized_participants = []
    default_model_context = normalize_model_context_policy(document.get("model_context"))
    controller_model_context = normalize_model_context_policy(
        document.get("controller_model_context") or default_model_context
    )
    for index, raw in enumerate(participants):
        if not isinstance(raw, dict):
            raise ValueError(f"team participant at index {index} must be an object")
        participant = dict(raw)
        participant_id = str(participant.get("id") or participant.get("name") or "").strip()
        if not participant_id:
            raise ValueError(f"team participant at index {index} requires id or name")
        safe_name = _name(participant_id, f"agent_{index + 1}")
        if safe_name in names:
            raise ValueError(f"duplicate team participant name {safe_name!r}")
        names.add(safe_name)
        participant["id"] = participant_id
        participant["model_context"] = normalize_model_context_policy(
            participant.get("model_context") or default_model_context
        )
        normalized_participants.append(participant)
    group_chat = normalize_group_chat_config(document.get("group_chat"))
    participant_count = len(participants) or len(ROLE_PROFILES.get(str(profile), []))
    if group_chat["type"] == "selector" and participant_count < 2:
        raise ValueError("SelectorGroupChat requires at least two participants")
    termination = dict(document.get("termination") or {"conditions": []})
    conditions = termination.get("conditions") or []
    if not isinstance(conditions, list):
        raise ValueError("termination.conditions must be a list")
    for condition in conditions:
        if not isinstance(condition, dict) or condition.get("type") != "text_mention":
            raise ValueError("only AutoGen text_mention termination is supported")
        if not str(condition.get("text") or "").strip():
            raise ValueError("text_mention termination requires non-empty text")
    return {
        "schema_version": 4,
        "id": team_id,
        "role_profile": profile,
        "participants": normalized_participants,
        "group_chat": group_chat,
        "termination": termination,
        "model_context": default_model_context,
        "controller_model_context": controller_model_context,
        "inference_slots": _normalize_inference_slots(
            document.get("inference_slots"),
            participants=normalized_participants,
            group_chat=group_chat,
        ),
        "extensions": _extensions(document),
    }


def load_team_spec(path: str | Path, *, rounds: int) -> tuple[MASGraph, dict[str, Any]]:
    document = normalize_team_spec_document(json.loads(Path(path).read_text(encoding="utf-8")))
    profile = str(
        document.get("role_profile") or document.get("profile") or document.get("id") or "default"
    )
    raw_participants = document.get("participants") or []
    default_model_context = normalize_model_context_policy(document.get("model_context"))
    if raw_participants:
        if not isinstance(raw_participants, list):
            raise ValueError("team participants must be a list")
        nodes = []
        names: set[str] = set()
        for index, raw in enumerate(raw_participants):
            if not isinstance(raw, dict):
                raise ValueError(f"team participant at index {index} must be an object")
            node = _participant_from_document(raw, index)
            node.meta["model_context"] = normalize_model_context_policy(
                raw.get("model_context") or default_model_context
            )
            if node.name in names:
                raise ValueError(f"duplicate team participant name {node.name!r}")
            names.add(node.name)
            nodes.append(node)
    else:
        if profile not in ROLE_PROFILES:
            raise ValueError(f"team spec has no participants and unknown role profile {profile!r}")
        nodes = team_to_agentspecs(profile)
        for node in nodes:
            node.meta["model_context"] = default_model_context
        names = {node.name for node in nodes}

    group_chat = _group_chat(document)
    extensions = _extensions(document)
    topology = dict(extensions.get("topology") or {})
    topology.setdefault("edges", {})
    topology.setdefault("speaking_order", [])

    raw_edges = topology.get("edges") or {}
    edges: dict[str, list[str]] = {node.name: [] for node in nodes}
    if isinstance(raw_edges, dict):
        pairs = [
            (source, target)
            for source, targets in raw_edges.items()
            for target in (targets if isinstance(targets, list) else [targets])
        ]
    elif isinstance(raw_edges, list):
        pairs = [edge[:2] for edge in raw_edges if isinstance(edge, list) and len(edge) >= 2]
    else:
        raise ValueError("topology edges must be an adjacency object or a list of pairs")
    for source, target in pairs:
        source_name = _name(source, "source")
        target_name = _name(target, "target")
        if source_name not in names or target_name not in names:
            raise ValueError(
                f"topology edge references unknown participant: {source!r} -> {target!r}"
            )
        if target_name not in edges[source_name]:
            edges[source_name].append(target_name)

    incoming: dict[str, list[str]] = {node.name: [] for node in nodes}
    for source, targets in edges.items():
        for target in targets:
            incoming[target].append(source)
    for node in nodes:
        node.meta["receives_from"] = incoming[node.name]

    speaking_order = [_name(value, "agent") for value in (topology.get("speaking_order") or [])]
    if any(value not in names for value in speaking_order):
        raise ValueError("topology speaking_order references an unknown participant")
    termination = dict(
        document.get("termination")
        or ROLE_PROFILE_META.get(profile, {}).get("termination")
        or {"conditions": []}
    )
    visibility = str((extensions.get("context_visibility") or {}).get("type") or "shared")
    graph = MASGraph(
        nodes=nodes,
        edges=edges,
        rounds=rounds,
        meta={
            "team": str(document.get("id") or profile or "studio-team"),
            "role_profile": profile,
            "studio_team": True,
            "group_chat": group_chat,
            "termination": termination,
            "model_context": default_model_context,
            "controller_model_context": normalize_model_context_policy(
                document.get("controller_model_context") or default_model_context
            ),
            "extensions": extensions,
            "context_visibility": visibility,
            "dynamic_topology": False,
            "speaking_order": speaking_order or [node.name for node in nodes],
        },
    )
    return graph, document


def team_details_from_graph(graph: MASGraph) -> dict[str, Any]:
    roles = [
        {
            "name": node.name,
            "agent_type": str(node.meta.get("agent_type") or "assistant"),
            "deployment_id": node.meta.get("deployment_id"),
            "model": node.model,
            "model_context": node.meta.get("model_context")
            or graph.meta.get("model_context")
            or {"type": "unbounded"},
        }
        for node in graph.nodes
    ]
    group_chat = dict(graph.meta.get("group_chat") or {"type": "round_robin"})
    group_chat_type = str(group_chat.get("type") or "round_robin")
    runtime_roles = list(roles)
    if group_chat_type == "magentic_one":
        runtime_roles = [
            {
                "name": "Orchestrator",
                "agent_type": "orchestrator",
                "deployment_id": graph.meta.get("control_deployment_id"),
                "model": None,
            },
            *roles,
        ]
    elif group_chat_type == "selector":
        runtime_roles = [
            {
                "name": "Selector",
                "agent_type": "selector",
                "deployment_id": (
                    None
                    if group_chat.get("selector_func_factory")
                    else graph.meta.get("control_deployment_id")
                ),
                "model": None,
                "control_mode": (
                    "selector_func" if group_chat.get("selector_func_factory") else "model"
                ),
            },
            *roles,
        ]
    labels = {
        "round_robin": "RoundRobinGroupChat",
        "selector": "SelectorGroupChat",
        "magentic_one": "MagenticOneGroupChat",
    }
    return {
        "profile": graph.meta.get("role_profile") or graph.meta.get("team", "studio-team"),
        "roles": roles,
        "group_chat": group_chat,
        "group_chat_type": group_chat_type,
        "chat_mode": labels[group_chat_type],
        "runtime_roles": runtime_roles,
        "context_visibility": (
            graph.meta.get("context_visibility")
            or graph.meta.get("extensions", {}).get("context_visibility", {}).get("type", "shared")
        ),
        "model_context": graph.meta.get("model_context") or {"type": "unbounded"},
        "controller_model_context": graph.meta.get("controller_model_context")
        or {"type": "unbounded"},
    }
