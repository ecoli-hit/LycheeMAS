"""Derive framework-local plans directly from canonical TeamSpec v14 facts."""

from __future__ import annotations

from typing import Any

from ..contracts.operations import (
    ORCHESTRATION_OPERATION_KINDS,
    declared_operations,
)
from ..contracts.team_graph import (
    executable_nodes,
    member_nodes,
)


def _execution_nodes(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return executable_nodes(team_spec)


def _member_nodes(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return member_nodes(team_spec)


def _operation_bindings(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile only operations explicitly declared by executable Nodes."""

    bindings: list[dict[str, Any]] = []
    for node in _execution_nodes(team_spec):
        for operation, options in sorted(declared_operations(node).items()):
            bindings.append(
                {
                    "operation": operation,
                    "node_id": str(node["id"]),
                    "options": dict(options),
                }
            )
    return bindings


def _derived_control_relation(relation: dict[str, Any]) -> dict[str, Any]:
    control = dict(relation.get("control") or {})
    operation = str(control.get("action") or "activate")
    relation_type = (
        "handoff"
        if operation == "handoff"
        else "dependency"
        if operation in {"activate", "return"}
        else "control"
    )
    return {
        "id": str(relation["id"]),
        "family": "control",
        "type": relation_type,
        "source": str(relation["from"]),
        "target": str(relation["to"]),
        "enabled": True,
        "contract": {
            "operation": operation,
            "trigger": str(control.get("trigger") or "completed"),
            "guard": dict(control.get("condition") or {}) or {"type": "always"},
            "priority": int(control.get("priority") or 0),
            "fallback": str(control.get("on_failure") or "fail_trial"),
            "on_failure": str(control.get("on_failure") or "fail_trial"),
            "max_uses": control.get("max_uses"),
        },
    }


def _member_relations(team_spec: dict[str, Any], relation_types: set[str]) -> list[dict[str, Any]]:
    member_ids = {node["id"] for node in _member_nodes(team_spec)}
    return [
        relation
        for item in team_spec.get("relations") or []
        if item.get("control")
        for relation in [_derived_control_relation(item)]
        if relation.get("type") in relation_types
        and relation.get("source") in member_ids
        and relation.get("target") in member_ids
    ]


def _chain_order(member_ids: list[str], relations: list[dict[str, Any]]) -> list[str] | None:
    if len(member_ids) < 2 or len(relations) != len(member_ids) - 1:
        return None
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in member_ids}
    indegree = {node_id: 0 for node_id in member_ids}
    for relation in relations:
        source = str(relation["source"])
        target = str(relation["target"])
        outgoing[source].append(target)
        indegree[target] += 1
    if any(len(targets) > 1 for targets in outgoing.values()):
        return None
    roots = [node_id for node_id, degree in indegree.items() if degree == 0]
    if len(roots) != 1 or any(degree > 1 for degree in indegree.values()):
        return None
    order: list[str] = []
    cursor = roots[0]
    while cursor not in order:
        order.append(cursor)
        targets = outgoing[cursor]
        if not targets:
            break
        cursor = targets[0]
    return order if len(order) == len(member_ids) else None


def _topological_order(member_ids: list[str], relations: list[dict[str, Any]]) -> list[str]:
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in member_ids}
    indegree = {node_id: 0 for node_id in member_ids}
    for relation in relations:
        source = str(relation["source"])
        target = str(relation["target"])
        if target not in outgoing[source]:
            outgoing[source].append(target)
            indegree[target] += 1
    queue = [node_id for node_id in member_ids if indegree[node_id] == 0]
    order: list[str] = []
    while queue:
        source = queue.pop(0)
        order.append(source)
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return order if len(order) == len(member_ids) else member_ids


def compile_coordination(team_spec: dict[str, Any]) -> dict[str, Any]:
    """Build a framework-neutral Coordination IR from explicit graph facts.

    The IR intentionally has no shared ``pattern`` field. AutoGen, LangGraph and
    CrewAI compile the same primitives independently instead of inheriting one
    framework-shaped global taxonomy.
    """

    members = [str(node["id"]) for node in _member_nodes(team_spec)]
    if not members:
        raise ValueError("TeamSpec requires at least one executable member Node")

    operation_bindings = _operation_bindings(team_spec)
    operations: dict[str, list[dict[str, Any]]] = {}
    operations_by_node: dict[str, dict[str, dict[str, Any]]] = {}
    for binding in operation_bindings:
        operation = str(binding["operation"])
        node_id = str(binding["node_id"])
        value = {"node_id": node_id, "options": dict(binding["options"])}
        operations.setdefault(operation, []).append(value)
        operations_by_node.setdefault(node_id, {})[operation] = dict(binding["options"])

    control_relations = [
        _derived_control_relation(relation)
        for relation in team_spec.get("relations") or []
        if relation.get("control")
    ]
    operation_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for relation in control_relations:
        contract = dict(relation.get("contract") or {})
        operation = str(contract.get("operation") or "")
        source = str(relation.get("source") or "")
        if not operation or source not in operations_by_node:
            continue
        if operation not in operations_by_node[source]:
            continue
        operation_candidates.setdefault(operation, {}).setdefault(source, []).append(
            {
                "node_id": str(relation["target"]),
                "relation_id": str(relation["id"]),
                "priority": int(contract.get("priority") or 0),
                "guard": dict(contract.get("guard") or {"type": "always"}),
                "fallback": str(contract.get("fallback") or "error"),
            }
        )
    for providers in operation_candidates.values():
        for candidates in providers.values():
            candidates.sort(key=lambda item: (item["priority"], item["relation_id"]))
    member_control_relations = _member_relations(team_spec, {"control", "dependency", "handoff"})
    dependency_relations = [
        relation for relation in member_control_relations if relation["type"] == "dependency"
    ]
    handoff_relations = [
        relation for relation in member_control_relations if relation["type"] == "handoff"
    ]
    chain = _chain_order(members, dependency_relations)
    ordered_members = chain or _topological_order(members, dependency_relations)
    incoming: dict[str, list[str]] = {member: [] for member in members}
    outgoing: dict[str, list[str]] = {member: [] for member in members}
    for relation in dependency_relations:
        source = str(relation["source"])
        target = str(relation["target"])
        if target not in outgoing[source]:
            outgoing[source].append(target)
        if source not in incoming[target]:
            incoming[target].append(source)

    return {
        "ir_version": 2,
        "members": members,
        "ordered_members": ordered_members,
        "operations": operations,
        "operations_by_node": operations_by_node,
        "operation_bindings": operation_bindings,
        "operation_candidates": operation_candidates,
        "control_relations": control_relations,
        "dependency_relations": dependency_relations,
        "handoff_relations": handoff_relations,
        "incoming": incoming,
        "outgoing": outgoing,
        "roots": [member for member in members if not incoming[member]],
        "sinks": [member for member in members if not outgoing[member]],
        "is_dependency_chain": chain is not None,
        "execution_nodes": [str(node["id"]) for node in _execution_nodes(team_spec)],
        "shared_states": [str(item["id"]) for item in team_spec.get("shared_state") or []],
        "derived_from": "teamspec_v14_nodes_relations_shared_state_lifecycle",
    }


def operation_node_ids(coordination: dict[str, Any], operation: str) -> list[str]:
    return [
        str(binding.get("node_id"))
        for binding in (coordination.get("operations") or {}).get(operation) or []
        if binding.get("node_id")
    ]


def operation_node_id(coordination: dict[str, Any], operation: str) -> str | None:
    values = operation_node_ids(coordination, operation)
    return values[0] if values else None


_ORCHESTRATION_BUNDLE = ORCHESTRATION_OPERATION_KINDS


def orchestration_node_id(coordination: dict[str, Any]) -> str | None:
    by_node = coordination.get("operations_by_node") or {}
    for node_id, operations in by_node.items():
        if _ORCHESTRATION_BUNDLE <= set(operations):
            return str(node_id)
    return None


def coordination_control_node_id(coordination: dict[str, Any]) -> str | None:
    return operation_node_id(coordination, "select_next") or orchestration_node_id(coordination)


def coordination_topology(coordination: dict[str, Any]) -> str:
    """Return a research-only presentation label; never use it for dispatch."""

    if operation_node_id(coordination, "select_next") or orchestration_node_id(coordination):
        return "centralized"
    if coordination.get("handoff_relations"):
        return "decentralized"
    if len(coordination.get("members") or []) == 1:
        return "independent"
    if coordination.get("is_dependency_chain"):
        return "sequential"
    return "decentralized"


def operation_options(
    coordination: dict[str, Any], operation: str, *, node_id: str | None = None
) -> dict[str, Any]:
    """Return one Node's declared options for a coordination operation.

    Operation fallbacks describe what to do when an operation itself cannot
    produce a decision. They are intentionally distinct from Relation
    ``on_failure`` policies, which apply only after a concrete Relation has
    already been selected.
    """

    bindings = (coordination.get("operations") or {}).get(operation) or []
    for binding in bindings:
        if node_id is None or str(binding.get("node_id")) == node_id:
            return dict(binding.get("options") or {})
    return {}


def operation_candidates(
    coordination: dict[str, Any], operation: str, *, node_id: str | None = None
) -> list[str]:
    """Return explicitly related candidates in contract priority order."""

    provider = node_id or operation_node_id(coordination, operation)
    if not provider:
        return []
    return [
        str(item["node_id"])
        for item in ((coordination.get("operation_candidates") or {}).get(operation) or {}).get(
            provider, []
        )
    ]


def compile_autogen_plan(coordination: dict[str, Any]) -> dict[str, Any]:
    """Choose official AutoGen primitives directly from Coordination IR."""

    orchestration_id = orchestration_node_id(coordination)
    selector_id = operation_node_id(coordination, "select_next")
    bounded_handoff = any(
        str((binding.get("options") or {}).get("fallback") or "error") == "next_priority"
        for binding in ((coordination.get("operations") or {}).get("handoff") or [])
    )
    if orchestration_id:
        group_chat_type = "magentic_one"
        implementation = "MagenticOneGroupChat"
        options = {
            "max_stalls": operation_options(
                coordination, "detect_stall", node_id=orchestration_id
            ).get("window", 3),
            "final_answer_prompt": operation_options(
                coordination, "aggregate", node_id=orchestration_id
            ).get("prompt"),
        }
    elif selector_id:
        group_chat_type = "selector"
        implementation = "SelectorGroupChat"
        options = operation_options(coordination, "select_next", node_id=selector_id)
    elif coordination.get("handoff_relations") and bounded_handoff:
        group_chat_type = "selector"
        implementation = "SelectorGroupChat with bounded handoff selector"
        options = {}
    elif coordination.get("handoff_relations"):
        group_chat_type = "swarm"
        implementation = "Swarm"
        options = {}
    elif len(coordination.get("members") or []) == 1 or coordination.get("is_dependency_chain"):
        group_chat_type = "round_robin"
        implementation = "RoundRobinGroupChat"
        options = {}
    else:
        group_chat_type = "graph_flow"
        implementation = "GraphFlow"
        options = {}

    config: dict[str, Any] = {"type": group_chat_type}
    if group_chat_type == "selector":
        if bounded_handoff:
            config["candidate_node_ids"] = list(coordination.get("members") or [])
            config["selector_func_factory"] = "bounded_handoff_selector"
            config["allow_repeated_speaker"] = True
            config["max_selector_attempts"] = 1
        else:
            config["candidate_node_ids"] = operation_candidates(
                coordination, "select_next", node_id=selector_id
            )
            mode = options.get("mode", "model")
            if mode == "deterministic":
                config["selector_func_factory"] = "operation_selector"
            else:
                config["candidate_func_factory"] = "operation_candidates"
            aliases = {
                "prompt": "selector_prompt",
                "allow_repeat": "allow_repeated_speaker",
                "max_attempts": "max_selector_attempts",
            }
            for source, target in aliases.items():
                if options.get(source) is not None:
                    config[target] = options[source]
    elif group_chat_type == "magentic_one":
        for name in ("max_stalls", "final_answer_prompt"):
            if options.get(name) is not None:
                config[name] = options[name]
    return {
        "framework": "autogen",
        "strategy": group_chat_type,
        "implementation": implementation,
        "mapping_level": "composed" if bounded_handoff else "exact",
        "config": config,
    }


def compile_langgraph_plan(coordination: dict[str, Any]) -> dict[str, Any]:
    """Choose a LangGraph-local StateGraph construction strategy."""

    if orchestration_node_id(coordination):
        strategy = "supervisor_orchestration"
        implementation = "StateGraph orchestration state machine"
        level = "composed"
    elif operation_node_id(coordination, "select_next"):
        strategy = "supervisor_conditional_edges"
        implementation = "StateGraph supervisor conditional edges"
        level = "composed"
    elif coordination.get("handoff_relations"):
        strategy = "conditional_handoff_edges"
        implementation = "StateGraph conditional handoff adapter"
        level = "composed"
    elif len(coordination.get("members") or []) == 1:
        strategy = "single_node"
        implementation = "StateGraph single node"
        level = "composed"
    elif coordination.get("is_dependency_chain"):
        strategy = "deterministic_dependency_chain"
        implementation = "StateGraph deterministic edges"
        level = "composed"
    else:
        strategy = "explicit_dependency_graph"
        implementation = "StateGraph explicit graph edges"
        level = "exact"
    return {
        "framework": "langgraph",
        "strategy": strategy,
        "implementation": implementation,
        "mapping_level": level,
        "config": {},
    }


def compile_crewai_plan(coordination: dict[str, Any]) -> dict[str, Any]:
    """Choose a CrewAI-local Crew or Flow construction strategy."""

    if orchestration_node_id(coordination):
        strategy = "orchestration_flow"
        implementation = "CrewAI Flow orchestration state machine"
        level = "composed"
    elif operation_node_id(coordination, "select_next"):
        strategy = "selector_flow"
        implementation = "CrewAI Flow model router"
        level = "composed"
    elif coordination.get("handoff_relations"):
        strategy = "handoff_flow"
        implementation = "CrewAI Flow handoff"
        level = "composed"
    elif len(coordination.get("members") or []) == 1:
        strategy = "single_task_crew"
        implementation = "Crew single Agent/Task"
        level = "exact"
    elif coordination.get("is_dependency_chain"):
        strategy = "sequential_crew"
        implementation = "Crew Process.sequential"
        level = "exact"
    else:
        strategy = "dependency_flow"
        implementation = "CrewAI Flow dependency graph"
        level = "composed"
    return {
        "framework": "crewai",
        "strategy": strategy,
        "implementation": implementation,
        "mapping_level": level,
        "config": {},
    }


_PLAN_COMPILERS = {
    "autogen": compile_autogen_plan,
    "langgraph": compile_langgraph_plan,
    "crewai": compile_crewai_plan,
}


def compile_runtime_plan(framework: str, coordination: dict[str, Any]) -> dict[str, Any]:
    framework = str(framework).strip().lower()
    try:
        compiler = _PLAN_COMPILERS[framework]
    except KeyError as exc:
        raise ValueError(f"unknown runtime framework {framework!r}") from exc
    return compiler(coordination)


def parse_next_node_decision(
    value: Any,
    *,
    candidates: list[str] | tuple[str, ...],
    finish_label: str = "FINISH",
) -> str | None:
    """Parse one conservative framework-neutral next-node decision."""

    text = str(value or "").strip()
    allowed = [*map(str, candidates), str(finish_label)]
    if text in allowed:
        return text
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line in allowed:
        return first_line
    matches = [name for name in allowed if text.startswith(f"{name}:")]
    return matches[0] if len(matches) == 1 else None


def adapter_capabilities(framework: str) -> dict[str, Any]:
    """Return primitive capabilities, not a shared pattern catalogue."""

    framework = str(framework).strip().lower()
    if framework not in _PLAN_COMPILERS:
        return {"framework": framework, "supported": False}
    execution_kinds = ["model_agent", "tool_executor"]
    coordination_operations = {
        "autogen": [
            "select_next",
            "plan",
            "decompose",
            "delegate",
            "monitor_progress",
            "detect_stall",
            "replan",
            "handoff",
            "aggregate",
            "submit",
        ],
        "langgraph": [
            "select_next",
            "plan",
            "delegate",
            "monitor_progress",
            "detect_stall",
            "replan",
            "handoff",
            "aggregate",
            "submit",
        ],
        "crewai": [
            "select_next",
            "plan",
            "delegate",
            "monitor_progress",
            "detect_stall",
            "replan",
            "handoff",
            "aggregate",
            "submit",
        ],
    }[framework]
    return {
        "framework": framework,
        "supported": True,
        "execution_kinds": execution_kinds,
        "coordination_operations": coordination_operations,
        "operation_constraints": {"orchestration_bundle": sorted(_ORCHESTRATION_BUNDLE)},
        "node_kinds": execution_kinds,
        "shared_state_kinds": [
            "message_channel",
            "structured_state",
            "artifact_store",
            "memory",
        ],
        "relation_facets": ["control", "data"],
    }


def runtime_support(
    framework: str,
    coordination: dict[str, Any],
    team_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a BindingReport from one adapter's independent runtime plan."""

    framework = str(framework).lower()
    capabilities = adapter_capabilities(framework)
    if not capabilities["supported"]:
        return {
            "supported": False,
            "framework": framework,
            "mapping_level": "unsupported",
            "reason": f"unknown framework {framework!r}",
        }
    plan = compile_runtime_plan(framework, coordination)
    level = str(plan["mapping_level"])
    report: dict[str, Any] = {
        "supported": True,
        "framework": framework,
        "adapter_strategy": plan["strategy"],
        "implementation": plan["implementation"],
        "mapping_level": level,
        "controlled_comparison_eligible": level in {"exact", "composed"},
        "semantic_deltas": (
            []
            if level in {"exact", "composed"}
            else [
                f"{framework} uses {plan['implementation']}; this does not preserve "
                "every coordination semantic in the portable graph contract"
            ]
        ),
        "adapter_plan": plan,
        "adapter_capabilities": capabilities,
    }
    if team_spec is None:
        return report

    semantic_deltas = list(report["semantic_deltas"])
    relation_facets = {
        facet
        for relation in team_spec.get("relations") or []
        for facet in ("control", "data")
        if relation.get(facet)
    }
    unsupported_relations = sorted(relation_facets - set(capabilities.get("relation_facets") or []))
    if unsupported_relations:
        semantic_deltas.append(
            f"{framework} does not execute TeamSpec Relation facets: "
            + ", ".join(unsupported_relations)
        )

    execution_kinds = {str(node.get("kind") or "model_agent") for node in member_nodes(team_spec)}
    unsupported_execution_kinds = sorted(
        execution_kinds - set(capabilities.get("execution_kinds") or [])
    )
    if unsupported_execution_kinds:
        semantic_deltas.append(
            f"{framework} does not natively execute TeamSpec Node kinds: "
            + ", ".join(unsupported_execution_kinds)
        )

    bound_operations = {
        str(operation)
        for node in executable_nodes(team_spec)
        for operation in declared_operations(node)
    }
    unsupported_operations = sorted(
        bound_operations - set(capabilities.get("coordination_operations") or [])
    )
    if unsupported_operations:
        semantic_deltas.append(
            f"{framework} does not execute TeamSpec operations: "
            + ", ".join(unsupported_operations)
        )
    declared_orchestration = bound_operations & _ORCHESTRATION_BUNDLE
    required_orchestration = set(
        (capabilities.get("operation_constraints") or {}).get("orchestration_bundle") or []
    )
    if (
        declared_orchestration
        and required_orchestration
        and not (required_orchestration <= bound_operations)
    ):
        semantic_deltas.append(
            f"{framework} executes orchestration only as the complete operation bundle; "
            "missing: " + ", ".join(sorted(required_orchestration - bound_operations))
        )

    for relation in team_spec.get("relations") or []:
        relation_id = str(relation.get("id") or "relation")
        control = dict(relation.get("control") or {})
        if control and int(control.get("priority") or 0) != 0:
            semantic_deltas.append(
                f"{framework} does not currently order eligible Control relations by "
                f"priority for {relation_id}"
            )
        for transfer in (relation.get("data") or {}).get("transfers") or []:
            filter_policy = dict(transfer.get("filter") or {})
            if filter_policy:
                semantic_deltas.append(
                    f"{framework} does not currently enforce DataTransfer filters for {relation_id}"
                )
            if transfer.get("transform"):
                semantic_deltas.append(
                    f"{framework} does not currently execute DataTransfer transform "
                    f"{transfer['transform']!r} for {relation_id}"
                )

    if semantic_deltas:
        report["semantic_deltas"] = semantic_deltas
        report["controlled_comparison_eligible"] = False
        if report["mapping_level"] in {"exact", "composed"}:
            report["mapping_level"] = "approximated"
    return report


def message_sources(
    team_spec: dict[str, Any], members: list[str] | tuple[str, ...]
) -> dict[str, list[str]]:
    """Resolve message visibility from v14 DataTransfers and SharedState access."""

    member_set = set(map(str, members))
    sources: dict[str, list[str]] = {member: [] for member in member_set}
    states = {str(item["id"]): item for item in team_spec.get("shared_state") or []}
    for state in states.values():
        if state.get("kind") != "message_channel":
            continue
        writers = [str(item) for item in state.get("writers") or [] if str(item) in member_set]
        for reader in state.get("readers") or []:
            reader = str(reader)
            if reader in member_set:
                sources[reader].extend(writer for writer in writers if writer != reader)
    for relation in team_spec.get("relations") or []:
        source = str(relation.get("from") or "")
        target = str(relation.get("to") or "")
        if source not in member_set or target not in member_set:
            continue
        for transfer in (relation.get("data") or {}).get("transfers") or []:
            if transfer.get("target") == "messages" and transfer.get("source") == "source.output":
                sources[target].append(source)
    return {member: list(dict.fromkeys(sources.get(member) or [])) for member in map(str, members)}


def message_policies(
    team_spec: dict[str, Any], members: list[str] | tuple[str, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Resolve per-source policies from canonical v14 DataTransfers."""

    member_list = list(map(str, members))
    policies: dict[str, dict[str, dict[str, Any]]] = {member: {} for member in member_list}
    sources = message_sources(team_spec, member_list)
    view_by_target: dict[str, str] = {}
    for relation in team_spec.get("relations") or []:
        target = str(relation.get("to") or "")
        for transfer in (relation.get("data") or {}).get("transfers") or []:
            if transfer.get("target") == "messages":
                view_by_target[target] = str(transfer.get("view") or "all")
    for target, visible_sources in sources.items():
        view = view_by_target.get(target, "all")
        history = "latest" if view in {"latest", "latest_n"} else "all"
        for source in visible_sources:
            policies[target][source] = {
                "delivery": "immediate",
                "history": history,
                "content_types": ["multimodal", "text", "tool_call", "tool_result"],
            }
    return policies


def message_visibility(team_spec: dict[str, Any], members: list[str] | tuple[str, ...]) -> str:
    """Classify an explicit message graph for capability reporting only."""

    member_list = list(map(str, members))
    sources = message_sources(team_spec, member_list)
    return (
        "shared"
        if all(set(sources[member]) >= (set(member_list) - {member}) for member in member_list)
        else "topology_filtered"
    )


__all__ = [
    "adapter_capabilities",
    "compile_autogen_plan",
    "compile_coordination",
    "compile_crewai_plan",
    "compile_langgraph_plan",
    "compile_runtime_plan",
    "coordination_control_node_id",
    "coordination_topology",
    "operation_node_id",
    "operation_node_ids",
    "operation_candidates",
    "orchestration_node_id",
    "message_sources",
    "message_policies",
    "message_visibility",
    "parse_next_node_decision",
    "runtime_support",
]
