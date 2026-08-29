"""Deterministic read models derived from the canonical RunEvent journal."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import (
    EVENT_TYPES,
    event_payload,
    iter_run_events,
    materialize_payload_value,
)


def _trial_key(event: Mapping[str, Any]) -> tuple[str, int] | None:
    case_id = str(event.get("case_id") or "")
    if not case_id:
        return None
    return case_id, int(event.get("trial_index") or 0)


def _category(event_type: str) -> str:
    try:
        return EVENT_TYPES.resolve(event_type).category
    except ValueError:
        return "unknown"


def _phase(event_type: str) -> str:
    suffix = event_type.rsplit(".", 1)[-1]
    if suffix == "started":
        return "started"
    if suffix in {"completed", "failed", "stopped"}:
        return "terminal"
    return "point"


def _actor(payload: Mapping[str, Any]) -> str | None:
    for key in ("node_id", "role", "source", "agent", "tool_name", "runtime"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _summary(event_type: str, payload: Mapping[str, Any]) -> str:
    for key in (
        "error_message",
        "stop_reason",
        "final_output",
        "content",
        "output",
        "tool_name",
        "runtime",
        "team",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            text = str(value).replace("\n", " ").strip()
            return text if len(text) <= 240 else f"{text[:237]}..."
    return event_type


def _duration(start: Mapping[str, Any], end: Mapping[str, Any] | None) -> float | None:
    if end is None:
        return None
    try:
        return round(
            float(end.get("timestamp_unix_s") or 0)
            - float(start.get("timestamp_unix_s") or 0),
            6,
        )
    except (TypeError, ValueError):
        return None


def _event_view(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event_payload(event)
    event_type = str(event.get("event_type") or "")
    return {
        "node_id": f"event:{event.get('event_id')}",
        "node_type": "event",
        "category": _category(event_type),
        "event_type": event_type,
        "status": "failed" if event_type.endswith(".failed") else "completed",
        "actor": _actor(payload),
        "summary": _summary(event_type, payload),
        "case_id": event.get("case_id"),
        "dataset_index": event.get("dataset_index"),
        "trial_index": event.get("trial_index"),
        "attempt": event.get("attempt"),
        "seq_start": event.get("seq"),
        "seq_end": event.get("seq"),
        "timestamp_start_utc": event.get("timestamp_utc"),
        "timestamp_end_utc": event.get("timestamp_utc"),
        "duration_s": 0.0,
        "parent_event_id": event.get("parent_event_id"),
        "correlation_id": event.get("correlation_id"),
        "operation_id": event.get("operation_id"),
        "events": [dict(event)],
        "children": [],
    }


def build_execution_trace(
    run_dir: str | Path,
    *,
    case_id: str | None = None,
    trial_index: int | None = None,
    filters: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build one causal execution trace without materializing a second log."""

    events = list(iter_run_events(run_dir))
    if case_id is not None:
        events = [event for event in events if str(event.get("case_id") or "") == case_id]
    if trial_index is not None:
        events = [event for event in events if int(event.get("trial_index") or 0) == trial_index]

    run_finished = any(
        str(event.get("event_type") or "") in {"run.completed", "run.failed"}
        for event in events
    )
    operation_nodes: dict[str, dict[str, Any]] = {}
    event_to_node: dict[str, str] = {}
    point_nodes: list[dict[str, Any]] = []

    for event in events:
        event_type = str(event.get("event_type") or "")
        phase = _phase(event_type)
        operation_id = str(event.get("operation_id") or "")
        if phase == "point" or not operation_id:
            node = _event_view(event)
            point_nodes.append(node)
            event_to_node[str(event.get("event_id") or "")] = node["node_id"]
            continue

        if phase == "started":
            payload = event_payload(event)
            node = {
                "node_id": f"operation:{operation_id}",
                "node_type": "operation",
                "category": _category(event_type),
                "event_type": event_type.removesuffix(".started"),
                "status": "running",
                "actor": _actor(payload),
                "summary": _summary(event_type, payload),
                "case_id": event.get("case_id"),
                "dataset_index": event.get("dataset_index"),
                "trial_index": event.get("trial_index"),
                "attempt": event.get("attempt"),
                "seq_start": event.get("seq"),
                "seq_end": None,
                "timestamp_start_utc": event.get("timestamp_utc"),
                "timestamp_end_utc": None,
                "duration_s": None,
                "parent_event_id": event.get("parent_event_id"),
                "correlation_id": event.get("correlation_id"),
                "operation_id": operation_id,
                "events": [dict(event)],
                "children": [],
            }
            operation_nodes[operation_id] = node
            event_to_node[str(event.get("event_id") or "")] = node["node_id"]
            continue

        terminal_node = operation_nodes.get(operation_id)
        if terminal_node is None:
            terminal_node = _event_view(event)
            terminal_node.update(
                {
                    "node_id": f"operation:{operation_id}",
                    "node_type": "operation",
                    "status": "orphaned_terminal",
                }
            )
            operation_nodes[operation_id] = terminal_node
        else:
            terminal_node["status"] = event_type.rsplit(".", 1)[-1]
            terminal_node["seq_end"] = event.get("seq")
            terminal_node["timestamp_end_utc"] = event.get("timestamp_utc")
            terminal_node["duration_s"] = _duration(terminal_node["events"][0], event)
            terminal_node["events"].append(dict(event))
            terminal_payload = event_payload(event)
            terminal_node["summary"] = (
                _summary(event_type, terminal_payload) or terminal_node["summary"]
            )
        event_to_node[str(event.get("event_id") or "")] = terminal_node["node_id"]

    nodes = [*operation_nodes.values(), *point_nodes]
    for node in nodes:
        if node["node_type"] == "operation" and node["status"] == "running" and run_finished:
            node["status"] = "interrupted"

    requested_filters = set(filters or ())

    def visible(node: Mapping[str, Any]) -> bool:
        if not requested_filters:
            return True
        category = str(node.get("category") or "")
        status = str(node.get("status") or "")
        event_type = str(node.get("event_type") or "")
        checks = {
            "messages": category == "agent_message",
            "model_calls": category == "model_operation",
            "tools": category in {"tool_protocol", "tool_operation", "web_operation"},
            "errors": status in {"failed", "interrupted", "orphaned_terminal"}
            or event_type.endswith(".failed"),
        }
        return any(checks.get(name, False) for name in requested_filters)

    visible_nodes = [node for node in nodes if visible(node)]
    visible_ids = {str(node["node_id"]) for node in visible_nodes}
    roots: list[dict[str, Any]] = []
    by_id = {str(node["node_id"]): node for node in visible_nodes}
    for node in visible_nodes:
        parent_node_id = event_to_node.get(str(node.get("parent_event_id") or ""))
        node["parent_node_id"] = parent_node_id if parent_node_id in visible_ids else None
        if node["parent_node_id"]:
            by_id[node["parent_node_id"]]["children"].append(node)
        else:
            roots.append(node)

    def sort_tree(items: list[dict[str, Any]]) -> None:
        items.sort(key=lambda item: int(item.get("seq_start") or 0))
        for item in items:
            sort_tree(item["children"])

    sort_tree(roots)
    return {
        "run_id": events[0].get("run_id") if events else None,
        "case_id": case_id,
        "trial_index": trial_index,
        "filters": sorted(requested_filters),
        "event_count": len(events),
        "node_count": len(visible_nodes),
        "roots": roots,
        "nodes": sorted(visible_nodes, key=lambda item: int(item.get("seq_start") or 0)),
    }


def _number(payload: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = payload.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def build_result_projection(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return one joined result row for every observed Trial."""

    root = Path(run_dir).expanduser().resolve()
    events = list(iter_run_events(root, materialize_payloads=False))

    def materialize(value: Any) -> Any:
        return materialize_payload_value(value, run_dir=root)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = _trial_key(event)
        if key is not None:
            grouped[key].append(event)

    rows: list[dict[str, Any]] = []
    for (case_id, trial_index), trial_events in grouped.items():
        trial_events.sort(key=lambda item: int(item.get("seq") or 0))
        starts = [event for event in trial_events if event.get("event_type") == "trial.started"]
        terminals = [
            event
            for event in trial_events
            if event.get("event_type") in {"trial.completed", "trial.failed"}
        ]
        if not starts and not terminals:
            continue
        start = starts[-1] if starts else None
        terminal = terminals[-1] if terminals else None
        evaluations = [
            event
            for event in trial_events
            if event.get("event_type") in {"evaluation.completed", "evaluation.failed"}
        ]
        evaluation = evaluations[-1] if evaluations else None
        terminal_payload = event_payload(terminal or {})
        evaluation_payload = event_payload(evaluation or {})

        model_calls = [
            event for event in trial_events if event.get("event_type") == "model_call.completed"
        ]
        messages = [
            event for event in trial_events if event.get("event_type") == "agent.message.published"
        ]
        tool_requests = [
            event for event in trial_events if event.get("event_type") == "tool_call.requested"
        ]
        tool_executions = [
            event
            for event in trial_events
            if event.get("event_type")
            in {"tool_execution.completed", "tool_execution.failed", "tool_execution.observed"}
        ]
        failed_events = [
            event
            for event in trial_events
            if str(event.get("event_type") or "").endswith(".failed")
        ]
        attempts = {
            int(event.get("attempt") or 0)
            for event in trial_events
            if str(event.get("event_type") or "").startswith("attempt.")
        }

        model_payloads = [event_payload(event) for event in model_calls]
        input_tokens = sum(
            _number(payload, "input_total_positions", "prompt_tokens")
            for payload in model_payloads
        )
        output_tokens = sum(
            _number(payload, "output_text_tokens", "completion_tokens")
            for payload in model_payloads
        )
        model_latency_s = sum(
            _number(payload, "model_latency_s", "duration_s") for payload in model_payloads
        )
        timing_fields = (
            "client_rate_limiter_wait_s",
            "client_http_request_latency_s",
            "client_response_postprocess_latency_s",
            "client_model_call_wall_time_s",
            "provider_request_queue_latency_s",
            "provider_scheduled_to_first_token_s",
            "provider_generation_latency_s",
            "provider_mean_inter_token_latency_s",
            "provider_output_tokens_per_second",
        )
        timing_summary: dict[str, Any] = {}
        for field in timing_fields:
            values = [
                float(payload[field])
                for payload in model_payloads
                if isinstance(payload.get(field), (int, float))
                and not isinstance(payload.get(field), bool)
            ]
            timing_summary[f"{field}_available_calls"] = len(values)
            timing_summary[f"{field}_sum"] = round(sum(values), 6) if values else None
        input_text_tokens = sum(
            _number(payload, "input_text_tokens") for payload in model_payloads
        )
        input_latent_positions = sum(
            _number(payload, "input_latent_positions") for payload in model_payloads
        )
        output_reasoning_tokens_values: list[int] = [
            int(payload["output_reasoning_tokens"])
            for payload in model_payloads
            if payload.get("output_reasoning_tokens") is not None
        ]
        output_answer_tokens_values: list[int] = [
            int(payload["output_answer_tokens"])
            for payload in model_payloads
            if payload.get("output_answer_tokens") is not None
        ]
        tool_error_count = sum(
            1
            for event in tool_executions
            if str(event.get("event_type") or "") == "tool_execution.failed"
            or bool(event_payload(event).get("is_error"))
        )
        if start is not None and terminal is not None:
            trial_wall_time_s = _duration(start, terminal)
        else:
            trial_wall_time_s = None

        if terminal is None:
            trial_status = "running"
        elif terminal.get("event_type") == "trial.failed":
            trial_status = "failed"
        else:
            trial_status = "completed"
        if evaluation is None:
            evaluation_status = "not_scored"
        elif evaluation.get("event_type") == "evaluation.failed":
            evaluation_status = "failed"
        else:
            evaluation_status = "completed"

        rows.append(
            {
                "run_id": trial_events[0].get("run_id"),
                "case_id": case_id,
                "dataset_index": next(
                    (
                        event.get("dataset_index")
                        for event in (start, terminal)
                        if event is not None and event.get("dataset_index") is not None
                    ),
                    None,
                ),
                "trial_index": trial_index,
                "trial_status": trial_status,
                "trial_event_id": terminal.get("event_id") if terminal else None,
                "trial_started_event_id": start.get("event_id") if start else None,
                "attempt_count": len(attempts),
                "trial_seed": terminal_payload.get("trial_seed")
                or event_payload(start or {}).get("trial_seed"),
                "base_seed": terminal_payload.get("base_seed")
                or event_payload(start or {}).get("base_seed"),
                "trials_per_case": terminal_payload.get("trials_per_case")
                or event_payload(start or {}).get("trials_per_case"),
                "task": terminal_payload.get("task") or event_payload(start or {}).get("task"),
                "method": terminal_payload.get("method")
                or event_payload(start or {}).get("method"),
                "benchmark_id": terminal_payload.get("benchmark_id")
                or event_payload(start or {}).get("benchmark_id"),
                "scorer_kind": terminal_payload.get("scorer_kind")
                or event_payload(start or {}).get("scorer_kind"),
                "prediction": materialize(terminal_payload.get("final_output")),
                "stop_reason": materialize(terminal_payload.get("stop_reason")),
                "error_type": terminal_payload.get("error_type"),
                "error_message": materialize(terminal_payload.get("error_message")),
                "evaluation_status": evaluation_status,
                "evaluation_event_id": evaluation.get("event_id") if evaluation else None,
                "evaluation_trial_event_id": (
                    evaluation.get("parent_event_id") if evaluation else None
                ),
                "score": evaluation_payload.get("score"),
                "correct": evaluation_payload.get("correct"),
                "gold": materialize(evaluation_payload.get("gold")),
                "score_details": materialize(evaluation_payload.get("score_details")),
                "scorer_id": evaluation_payload.get("scorer_id")
                or evaluation_payload.get("scorer_kind"),
                "model_call_count": len(model_calls),
                "message_count": len(messages),
                "tool_call_count": len(tool_requests),
                # Benchmark-native scorers such as WorkBench evaluate the actions
                # actually requested by the model. Keep the canonical tool name and
                # arguments in this derived row instead of reducing them to a count.
                "tool_requests": [
                    {
                        "tool_call_id": event_payload(event).get("tool_call_id"),
                        "tool_name": event_payload(event).get("tool_name"),
                        "arguments": materialize(event_payload(event).get("arguments")),
                    }
                    for event in tool_requests
                ],
                "tool_execution_count": len(tool_executions),
                "tool_error_count": tool_error_count,
                "error_event_count": len(failed_events),
                "input_tokens": int(input_tokens),
                "input_text_tokens": int(input_text_tokens),
                "input_latent_positions": int(input_latent_positions),
                "output_tokens": int(output_tokens),
                "output_reasoning_tokens": (
                    sum(output_reasoning_tokens_values)
                    if len(output_reasoning_tokens_values) == len(model_payloads)
                    else None
                ),
                "output_answer_tokens": (
                    sum(output_answer_tokens_values)
                    if len(output_answer_tokens_values) == len(model_payloads)
                    else None
                ),
                "model_latency_s": round(model_latency_s, 6),
                "trial_wall_time_s": trial_wall_time_s,
                **timing_summary,
                "started_at_utc": start.get("timestamp_utc") if start else None,
                "completed_at_utc": terminal.get("timestamp_utc") if terminal else None,
            }
        )

    return sorted(
        rows,
        key=lambda item: (
            int(item.get("dataset_index") or 0),
            int(item.get("trial_index") or 0),
        ),
    )
