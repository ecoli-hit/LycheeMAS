"""Framework-local Node binding compilation and reporting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from lychee_mas.runtime.contracts.memory import compile_memory_bindings
from lychee_mas.runtime.contracts.team_graph import executable_nodes
from lychee_mas.runtime.coordination.compiler import runtime_support

from .autogen.bindings import bind_node as bind_autogen_node
from .crewai.bindings import bind_node as bind_crewai_node
from .langgraph.bindings import bind_node as bind_langgraph_node

NodeBinder = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_BINDERS: dict[str, NodeBinder] = {
    "autogen": bind_autogen_node,
    "langgraph": bind_langgraph_node,
    "crewai": bind_crewai_node,
}


def compile_node_bindings(
    framework: str,
    team_spec: dict[str, Any],
    coordination: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compile every executable Node without using Node names as dispatch keys."""

    framework = str(framework).strip().lower()
    try:
        binder = _BINDERS[framework]
    except KeyError as exc:
        raise ValueError(f"unknown runtime framework {framework!r}") from exc
    return [binder(node, coordination) for node in executable_nodes(team_spec)]


def compile_adapter_plan(
    framework: str,
    team_spec: dict[str, Any],
    coordination: dict[str, Any],
) -> dict[str, Any]:
    report = build_framework_binding_report(framework, coordination, team_spec)
    return deepcopy(report["adapter_plan"])


def build_framework_binding_report(
    framework: str,
    coordination: dict[str, Any],
    team_spec: dict[str, Any],
) -> dict[str, Any]:
    """Combine coordination support with adapter-owned per-Node bindings."""

    report = deepcopy(runtime_support(framework, coordination, team_spec))
    bindings = compile_node_bindings(framework, team_spec, coordination)
    memory_bindings = compile_memory_bindings(framework, team_spec)
    unsupported = [
        binding for binding in bindings if binding.get("mapping_level") == "unsupported"
    ]
    unsupported_memory = [
        binding
        for binding in memory_bindings
        if binding.get("mapping_level") == "unsupported"
    ]
    report["node_bindings"] = bindings
    report["memory_bindings"] = memory_bindings
    report["adapter_plan"] = {
        **dict(report.get("adapter_plan") or {}),
        "node_bindings": bindings,
        "memory_bindings": memory_bindings,
    }
    node_levels = {str(binding.get("mapping_level") or "unsupported") for binding in bindings}
    report["node_binding_mapping_levels"] = sorted(node_levels)
    report["adapter_plan"]["node_binding_mapping_levels"] = sorted(node_levels)
    if unsupported or unsupported_memory:
        deltas = list(report.get("semantic_deltas") or [])
        deltas.extend(
            f"{framework} cannot bind Node {binding['node_id']!r} "
            f"with execution kind {binding['node_kind']!r}"
            for binding in unsupported
        )
        deltas.extend(
            f"{framework} cannot bind Memory {binding['memory_id']!r}: "
            + "; ".join(binding.get("semantic_deltas") or ["unsupported memory semantics"])
            for binding in unsupported_memory
        )
        report.update(
            {
                "supported": False,
                "mapping_level": "unsupported",
                "controlled_comparison_eligible": False,
                "semantic_deltas": deltas,
                "reason": "; ".join(deltas),
            }
        )
    elif report.get("mapping_level") == "exact" and "composed" in node_levels:
        report["mapping_level"] = "composed"
    return report


def binding_for_node(node: Any, framework: str) -> dict[str, Any]:
    """Read one already-compiled binding from an AgentSpec-like runtime Node."""

    bindings = dict(node.meta.get("framework_bindings") or {})
    binding = dict(bindings.get(str(framework).lower()) or {})
    if not binding:
        raise ValueError(
            f"Node {getattr(node, 'name', '<unknown>')!r} has no {framework!r} binding"
        )
    return binding


__all__ = [
    "binding_for_node",
    "build_framework_binding_report",
    "compile_adapter_plan",
    "compile_node_bindings",
]
