"""Framework-neutral operation declarations used by TeamSpec v14.

An operation is an observable responsibility owned by one executable Node. It
does not name an AutoGen, LangGraph, CrewAI, or provider-specific primitive.
TeamSpec v14 accepts only the normalized list form; authoring shorthands are
rejected so optimizers and adapters see one stable representation.
"""

from __future__ import annotations

from typing import Any

OPERATION_KINDS = {
    "select_next",
    "plan",
    "decompose",
    "delegate",
    "monitor_progress",
    "detect_stall",
    "replan",
    "handoff",
    "aggregate",
    "validate",
}
COORDINATION_OPERATION_KINDS = set(OPERATION_KINDS)
ORCHESTRATION_OPERATION_KINDS = {
    "plan",
    "delegate",
    "monitor_progress",
    "detect_stall",
    "replan",
    "aggregate",
}


def _reject_unknown(value: dict[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: " + ", ".join(sorted(unknown)))


def _positive_int(value: Any, *, default: int, label: str) -> int:
    parsed = int(default if value is None else value)
    if parsed < 1:
        raise ValueError(f"{label} must be >= 1")
    return parsed


def _mode(options: dict[str, Any], *, label: str) -> str:
    mode = str(options.get("mode") or "model").strip().lower()
    if mode not in {"model", "deterministic"}:
        raise ValueError(f"{label}.mode must be 'model' or 'deterministic'")
    return mode


def _common(options: dict[str, Any], *, label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {"mode": _mode(options, label=label)}
    if options.get("prompt"):
        normalized["prompt"] = str(options["prompt"])
    if options.get("max_attempts") is not None:
        normalized["max_attempts"] = _positive_int(
            options["max_attempts"], default=1, label=f"{label}.max_attempts"
        )
    return normalized


def _normalize_select_next(options: dict[str, Any], *, label: str) -> dict[str, Any]:
    _reject_unknown(
        options,
        {"mode", "prompt", "max_attempts", "allow_repeat", "finish"},
        label=label,
    )
    finish = dict(options.get("finish") or {})
    _reject_unknown(finish, {"allowed", "requires_result"}, label=f"{label}.finish")
    return {
        **_common(options, label=label),
        "max_attempts": _positive_int(
            options.get("max_attempts"), default=3, label=f"{label}.max_attempts"
        ),
        "allow_repeat": bool(options.get("allow_repeat", False)),
        "finish": {
            "allowed": bool(finish.get("allowed", True)),
            "requires_result": bool(finish.get("requires_result", True)),
        },
    }


def _normalize_state_operation(kind: str, options: dict[str, Any], *, label: str) -> dict[str, Any]:
    fields_by_kind = {
        "plan": {"mode", "prompt", "max_attempts", "state_target"},
        "decompose": {"mode", "prompt", "max_attempts", "state_target", "max_items"},
        "delegate": {"mode", "prompt", "max_attempts"},
        "monitor_progress": {
            "mode",
            "prompt",
            "max_attempts",
            "state_sources",
            "state_target",
        },
        "detect_stall": {"mode", "prompt", "max_attempts", "state_source", "window"},
        "replan": {
            "mode",
            "prompt",
            "max_attempts",
            "state_sources",
            "state_target",
            "max_replans",
        },
        "aggregate": {"mode", "prompt", "max_attempts", "source"},
        "validate": {"mode", "prompt", "max_attempts", "source", "on_reject"},
    }
    _reject_unknown(options, fields_by_kind[kind], label=label)
    normalized = _common(options, label=label)
    for field in ("state_target", "state_source", "source", "on_reject"):
        if options.get(field) is not None:
            normalized[field] = str(options[field])
    if options.get("state_sources") is not None:
        values = options["state_sources"]
        if not isinstance(values, list):
            raise ValueError(f"{label}.state_sources must be a list")
        normalized["state_sources"] = list(dict.fromkeys(map(str, values)))
    for field in ("max_items", "window", "max_replans"):
        if options.get(field) is not None:
            normalized[field] = _positive_int(options[field], default=1, label=f"{label}.{field}")
    return normalized


def _normalize_handoff(options: dict[str, Any], *, label: str) -> dict[str, Any]:
    _reject_unknown(
        options,
        {"mode", "prompt", "max_attempts", "fallback"},
        label=label,
    )
    fallback = str(options.get("fallback") or "error")
    if fallback not in {"error", "retry", "next_priority", "finish"}:
        raise ValueError(f"{label}.fallback is unsupported")
    return {**_common(options, label=label), "fallback": fallback}


def normalize_operations(raw: Any, *, node_id: str) -> list[dict[str, Any]]:
    """Validate the canonical one-object-per-operation representation."""

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Node {node_id!r} operations must be a list of objects")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        label = f"Node {node_id!r} operations[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        _reject_unknown(item, {"type", "options"}, label=label)
        kind = str(item.get("type") or "").strip()
        if kind not in OPERATION_KINDS:
            raise ValueError(f"{label}.type is unsupported: {kind!r}")
        if kind in seen:
            raise ValueError(f"Node {node_id!r} declares operation {kind!r} more than once")
        seen.add(kind)
        options = item.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError(f"{label}.options must be an object")
        if kind == "select_next":
            value = _normalize_select_next(dict(options), label=label)
        elif kind == "handoff":
            value = _normalize_handoff(dict(options), label=label)
        else:
            value = _normalize_state_operation(kind, dict(options), label=label)
        normalized.append({"type": kind, "options": value})
    return normalized


def declared_operations(node: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return one normalized lookup without introducing a persisted IR."""

    return {
        str(item["type"]): dict(item.get("options") or {})
        for item in node.get("operations") or []
    }


def orchestration_bundle(operations: dict[str, Any]) -> bool:
    return ORCHESTRATION_OPERATION_KINDS <= set(operations)


__all__ = [
    "COORDINATION_OPERATION_KINDS",
    "OPERATION_KINDS",
    "ORCHESTRATION_OPERATION_KINDS",
    "declared_operations",
    "normalize_operations",
    "orchestration_bundle",
]
