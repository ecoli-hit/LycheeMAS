"""Framework-neutral memory contracts and framework binding descriptions.

TeamSpec owns memory semantics. Framework adapters may use native components to
carry those semantics, but native defaults never become an implicit second
contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MEMORY_RETRIEVAL_MODES = {"chronological", "semantic", "hybrid"}
MEMORY_QUERY_SOURCES = {"task", "node_input", "task_and_node_input"}
MEMORY_WRITE_POLICIES = {"explicit", "node_output"}
MEMORY_INJECTION_TARGETS = {"model_context", "node_input"}


def normalize_memory_policy(raw: Any, *, state_id: str) -> dict[str, Any]:
    """Validate one memory policy without introducing framework vocabulary."""

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Memory {state_id!r} policy must be an object")
    value = dict(raw)
    unknown = set(value) - {"retrieval", "write", "injection"}
    if unknown:
        raise ValueError(
            f"Memory {state_id!r} policy contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )

    retrieval = dict(value.get("retrieval") or {})
    unknown = set(retrieval) - {"mode", "top_k", "score_threshold", "query_source"}
    if unknown:
        raise ValueError(
            f"Memory {state_id!r} retrieval contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    mode = str(retrieval.get("mode") or "chronological")
    if mode not in MEMORY_RETRIEVAL_MODES:
        raise ValueError(f"Memory {state_id!r} retrieval.mode is unsupported: {mode!r}")
    top_k = int(retrieval.get("top_k", 8))
    if top_k < 1:
        raise ValueError(f"Memory {state_id!r} retrieval.top_k must be >= 1")
    score_threshold = retrieval.get("score_threshold")
    if score_threshold is not None:
        score_threshold = float(score_threshold)
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                f"Memory {state_id!r} retrieval.score_threshold must be between 0 and 1"
            )
    query_source = str(retrieval.get("query_source") or "task_and_node_input")
    if query_source not in MEMORY_QUERY_SOURCES:
        raise ValueError(
            f"Memory {state_id!r} retrieval.query_source is unsupported: {query_source!r}"
        )

    write = dict(value.get("write") or {})
    unknown = set(write) - {"policy", "source"}
    if unknown:
        raise ValueError(
            f"Memory {state_id!r} write contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    write_policy = str(write.get("policy") or "node_output")
    if write_policy not in MEMORY_WRITE_POLICIES:
        raise ValueError(
            f"Memory {state_id!r} write.policy is unsupported: {write_policy!r}"
        )
    write_source = str(write.get("source") or "source.output").strip()
    if not write_source:
        raise ValueError(f"Memory {state_id!r} write.source must not be empty")

    injection = dict(value.get("injection") or {})
    unknown = set(injection) - {"target", "template"}
    if unknown:
        raise ValueError(
            f"Memory {state_id!r} injection contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    target = str(injection.get("target") or "model_context")
    if target not in MEMORY_INJECTION_TARGETS:
        raise ValueError(f"Memory {state_id!r} injection.target is unsupported: {target!r}")
    template = str(
        injection.get("template")
        or "Relevant memory from {memory_id}:\n{items}"
    )
    if "{items}" not in template:
        raise ValueError(f"Memory {state_id!r} injection.template must contain {{items}}")

    return {
        "retrieval": {
            "mode": mode,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "query_source": query_source,
        },
        "write": {"policy": write_policy, "source": write_source},
        "injection": {"target": target, "template": template},
    }


def memory_states(team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(item)
        for item in team_spec.get("shared_state") or []
        if str(item.get("kind") or "") == "memory"
    ]


def compile_memory_bindings(framework: str, team_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe how one adapter will carry every explicit memory state.

    The portable read/write policy remains authoritative. Native framework
    components are carriers and lifecycle facilities, not permission to change
    retrieval, visibility, retention, or write timing.
    """

    framework = str(framework).strip().lower()
    carriers = {
        "autogen": {
            "native_component": "autogen_core.memory.Memory",
            "adapter_component": "TeamSpecAutoGenMemory",
            "short_term_carrier": "AssistantAgent model context",
            "long_term_carrier": "Memory implementation selected by adapter",
        },
        "langgraph": {
            "native_component": "StateGraph checkpointer and store",
            "adapter_component": "TeamSpecLangGraphMemory",
            "short_term_carrier": "graph state with checkpointer",
            "long_term_carrier": "LangGraph store namespace",
        },
        "crewai": {
            "native_component": "CrewAI Memory / Flow memory",
            "adapter_component": "TeamSpecCrewAIMemory",
            "short_term_carrier": "Flow state and scoped Memory view",
            "long_term_carrier": "CrewAI Memory scope",
        },
    }
    if framework not in carriers:
        raise ValueError(f"unknown runtime framework {framework!r}")

    bindings: list[dict[str, Any]] = []
    for state in memory_states(team_spec):
        policy = dict(state.get("memory") or {})
        retrieval = dict(policy.get("retrieval") or {})
        lifetime = str(state.get("lifetime") or "trial")
        retrieval_mode = str(retrieval.get("mode") or "chronological")
        portable = lifetime == "trial" and retrieval_mode == "chronological"
        semantic_deltas = []
        if lifetime != "trial":
            semantic_deltas.append(
                "run/persistent lifetime requires a durable MemoryInstance binding"
            )
        if retrieval_mode != "chronological":
            semantic_deltas.append(
                "semantic/hybrid retrieval requires an explicit semantic index binding"
            )
        bindings.append(
            {
                "memory_id": str(state["id"]),
                "framework": framework,
                "readers": list(state.get("readers") or []),
                "writers": list(state.get("writers") or []),
                "lifetime": lifetime,
                "retrieval_mode": retrieval_mode,
                "write_policy": str((policy.get("write") or {}).get("policy") or "node_output"),
                "injection_target": str(
                    (policy.get("injection") or {}).get("target") or "model_context"
                ),
                "mapping_level": "composed" if portable else "unsupported",
                "supported": portable,
                "semantic_deltas": semantic_deltas,
                "semantic_owner": "TeamSpec",
                "read_hook": "before_node_invocation",
                "write_hook": "after_node_invocation",
                **carriers[framework],
            }
        )
    return bindings


__all__ = [
    "MEMORY_INJECTION_TARGETS",
    "MEMORY_QUERY_SOURCES",
    "MEMORY_RETRIEVAL_MODES",
    "MEMORY_WRITE_POLICIES",
    "compile_memory_bindings",
    "memory_states",
    "normalize_memory_policy",
]
