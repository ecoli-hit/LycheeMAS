"""Build per-record MAS graphs for benchmarks that define their own teams."""

from __future__ import annotations

import re
from typing import Any

from ...core.types import AgentSpec
from ...layers.construct.templates import RoleProfileTeamBuilder
from ...runtime.base import MASGraph


def _safe_name(value: Any, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(value or fallback)).strip("_")
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"agent_{name}"
    return name


def _topology_payload(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    topology = record.get("topology") or metadata.get("topology") or {}
    return topology if isinstance(topology, dict) else {}


def graph_from_benchmark_record(
    record: dict[str, Any], *, fallback_profile: str, rounds: int
) -> tuple[MASGraph, bool]:
    """Return a record-specific graph, or the configured static fallback.

    AgentCollabBench stores agents inside ``topology.agents``. The normalized
    graph keeps the original topology in ``meta`` and exposes ``receives_from``
    on each node so the runtime can enforce message visibility.
    """
    topology = _topology_payload(record)
    raw_agents = topology.get("agents") or record.get("agent_specs") or []
    if not isinstance(raw_agents, list) or not raw_agents:
        return (
            RoleProfileTeamBuilder(
                team=fallback_profile,
                model=None,
                rounds=rounds,
            ).build(),
            False,
        )

    nodes: list[AgentSpec] = []
    original_to_safe: dict[str, str] = {}
    for index, raw in enumerate(raw_agents):
        item = raw if isinstance(raw, dict) else {"role": str(raw)}
        original = str(
            item.get("agent_id") or item.get("id") or item.get("name") or f"agent_{index + 1}"
        )
        name = _safe_name(original, f"agent_{index + 1}")
        original_to_safe[original] = name
        role = str(item.get("role") or item.get("name") or original)
        system_prompt = str(
            item.get("system_prompt") or item.get("system") or item.get("prompt") or ""
        )
        receives_from = item.get("receives_from") or item.get("allowed_senders") or []
        if isinstance(receives_from, str):
            receives_from = [receives_from]
        tools = item.get("tools") if isinstance(item.get("tools"), list) else []
        nodes.append(
            AgentSpec(
                id=original,
                name=name,
                role=role,
                system_prompt=system_prompt,
                model=item.get("model"),
                tools=[str(tool) for tool in tools],
                meta={
                    "description": str(item.get("description") or role),
                    "agent_type": str(item.get("agent_type") or "assistant"),
                    "original_agent_id": original,
                    "receives_from": [str(value) for value in receives_from],
                },
            )
        )

    edges: dict[str, list[str]] = {node.name: [] for node in nodes}
    raw_edges = topology.get("edges") or []
    if isinstance(raw_edges, dict):
        edge_pairs = [
            (source, target)
            for source, targets in raw_edges.items()
            for target in (targets if isinstance(targets, list) else [targets])
        ]
    else:
        edge_pairs = [
            edge for edge in raw_edges if isinstance(edge, (list, tuple)) and len(edge) >= 2
        ]
    for source, target, *_ in edge_pairs:
        safe_source = original_to_safe.get(str(source), _safe_name(source, "source"))
        safe_target = original_to_safe.get(str(target), _safe_name(target, "target"))
        if safe_source in edges and safe_target in edges and safe_target not in edges[safe_source]:
            edges[safe_source].append(safe_target)

    for node in nodes:
        raw_sources = list(node.meta.get("receives_from") or [])
        node.meta["receives_from"] = [
            original_to_safe.get(source, _safe_name(source, source)) for source in raw_sources
        ]

    raw_order = topology.get("speaking_order") or [node.meta["original_agent_id"] for node in nodes]
    if not isinstance(raw_order, list):
        raw_order = [raw_order]
    speaking_order = [
        original_to_safe.get(str(value), _safe_name(value, "agent")) for value in raw_order
    ]
    valid_names = {node.name for node in nodes}
    speaking_order = [name for name in speaking_order if name in valid_names]
    if not speaking_order:
        speaking_order = [node.name for node in nodes]

    graph = MASGraph(
        nodes=nodes,
        edges=edges,
        rounds=rounds,
        meta={
            "team": fallback_profile,
            "role_profile": fallback_profile,
            "group_chat": {
                "type": "selector",
                "selector_func_factory": "topology_selector",
            },
            "dynamic_topology": True,
            "context_visibility": "topology_filtered",
            "topology_type": topology.get("type"),
            "speaking_order": speaking_order,
            "source_topology": topology,
        },
    )
    return graph, True
