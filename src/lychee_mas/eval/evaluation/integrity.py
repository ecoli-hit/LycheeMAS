"""Deterministic integrity checks for the canonical RunEvent journal."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import iter_run_events

_PAIRED_OPERATION_FAMILIES = {
    "trial",
    "attempt",
    "runtime",
    "group_chat",
    "model_call",
    "tool_execution",
    "web.context_startup",
    "node_invocation",
}
_TERMINAL_SUFFIXES = {"completed", "failed", "stopped", "interrupted", "cancelled"}
_MAX_EXAMPLES = 20


def _operation_family(event_type: str) -> tuple[str | None, str | None]:
    for family in _PAIRED_OPERATION_FAMILIES:
        prefix = f"{family}."
        if not event_type.startswith(prefix):
            continue
        suffix = event_type[len(prefix) :]
        if suffix == "started":
            return family, "started"
        if suffix in _TERMINAL_SUFFIXES:
            return family, "terminal"
    return None, None


def _scope(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: event.get(key)
        for key in ("case_id", "dataset_index", "trial_index", "attempt", "worker_id")
    }


def _scope_mismatch(start: Mapping[str, Any], terminal: Mapping[str, Any]) -> bool:
    start_scope = _scope(start)
    terminal_scope = _scope(terminal)
    return any(
        start_scope[key] is not None
        and terminal_scope[key] is not None
        and start_scope[key] != terminal_scope[key]
        for key in start_scope
    )


def _example(event: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "seq": event.get("seq"),
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "operation_id": event.get("operation_id"),
        **_scope(event),
        **extra,
    }


def audit_run_event_integrity(run_dir: str | Path) -> dict[str, Any]:
    """Check pairing, attribution, sequence identity, and Trial boundaries in one pass."""

    issue_names = (
        "missing_event_log",
        "sequence_anomalies",
        "missing_started_operations",
        "missing_terminal_operations",
        "duplicate_started_operations",
        "operation_scope_mismatches",
        "late_operation_starts",
        "late_operation_terminals",
    )
    counts = {name: 0 for name in issue_names}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in issue_names}

    def record_issue(name: str, event: Mapping[str, Any], **extra: Any) -> None:
        counts[name] += 1
        if len(examples[name]) < _MAX_EXAMPLES:
            examples[name].append(_example(event, **extra))

    total_events = 0
    paired_operation_count = 0
    previous_seq = 0
    active_operations: dict[tuple[str, str], dict[str, Any]] = {}
    trial_states: dict[str, dict[str, Any]] = {}
    active_trial_by_key: dict[tuple[str, int], str] = {}

    for event in iter_run_events(run_dir):
        total_events += 1
        seq = int(event.get("seq") or 0)
        if seq != previous_seq + 1:
            record_issue(
                "sequence_anomalies",
                event,
                expected_seq=previous_seq + 1,
                observed_seq=seq,
            )
        previous_seq = seq
        family, phase = _operation_family(str(event.get("event_type") or ""))
        operation_id = str(event.get("operation_id") or "")
        if not family or not phase or not operation_id:
            continue

        operation_key = (family, operation_id)
        scope_key = (str(event.get("case_id") or ""), int(event.get("trial_index") or 0))
        if phase == "started":
            paired_operation_count += 1
            if operation_key in active_operations:
                record_issue(
                    "duplicate_started_operations", event, family=family
                )
                continue
            trial_operation_id = (
                operation_id if family == "trial" else active_trial_by_key.get(scope_key)
            )
            active_operations[operation_key] = {
                "start": dict(event),
                "trial_operation_id": trial_operation_id,
            }
            if family == "trial":
                trial_states[operation_id] = {
                    "start_seq": seq,
                    "terminal_seq": None,
                }
                active_trial_by_key[scope_key] = operation_id
            elif trial_operation_id:
                trial_state = trial_states.get(trial_operation_id) or {}
                trial_terminal_seq = trial_state.get("terminal_seq")
                if trial_terminal_seq is not None and seq > int(trial_terminal_seq):
                    record_issue(
                        "late_operation_starts",
                        event,
                        family=family,
                        trial_operation_id=trial_operation_id,
                        trial_terminal_seq=trial_terminal_seq,
                    )
            continue

        state = active_operations.pop(operation_key, None)
        if state is None:
            record_issue("missing_started_operations", event, family=family)
            continue
        start = state["start"]
        if _scope_mismatch(start, event):
            record_issue(
                "operation_scope_mismatches",
                event,
                family=family,
                started_scope=_scope(start),
                terminal_scope=_scope(event),
            )
        if family == "trial":
            trial_states[operation_id]["terminal_seq"] = seq
        else:
            trial_operation_id = state.get("trial_operation_id")
            trial_state = trial_states.get(str(trial_operation_id)) or {}
            trial_terminal_seq = trial_state.get("terminal_seq")
            if trial_terminal_seq is not None and seq > int(trial_terminal_seq):
                record_issue(
                    "late_operation_terminals",
                    event,
                    family=family,
                    trial_operation_id=trial_operation_id,
                    trial_terminal_seq=trial_terminal_seq,
                )

    for (family, _operation_id), state in active_operations.items():
        record_issue("missing_terminal_operations", state["start"], family=family)

    if total_events == 0:
        counts["missing_event_log"] = 1
        examples["missing_event_log"].append({"run_dir": str(Path(run_dir))})

    fatal_issue_names = tuple(name for name in counts if name != "missing_terminal_operations")
    if any(counts[name] for name in fatal_issue_names):
        status = "invalid"
    elif counts["missing_terminal_operations"]:
        status = "incomplete"
    else:
        status = "valid"
    return {
        "status": status,
        "total_events": total_events,
        "paired_operation_count": paired_operation_count,
        "issue_counts": counts,
        "examples": {
            name: rows for name, rows in examples.items() if rows
        },
    }
