"""Normalize run artifacts into a framework-neutral evidence event stream."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from lychee_mas.runtime.events.store import run_event_paths

from .projectors import AutoGenMagenticOneLedgerProjector

EVIDENCE_SCHEMA_VERSION = 2
EVIDENCE_EVENTS_FILENAME = "evidence.jsonl"
EVIDENCE_COVERAGE_FILENAME = "evidence_coverage.json"

_SOURCE_FILES = (("run_events", "events/run_events.jsonl"),)

_ENVELOPE_FIELDS = {
    "schema_version",
    "run_id",
    "seq",
    "event_id",
    "event_type",
    "timestamp_unix_s",
    "timestamp_utc",
    "case_id",
    "dataset_index",
    "trial_index",
    "attempt",
    "operation_id",
    "parent_event_id",
    "correlation_id",
    "payload",
}

# These fields already remain intact in their source artifact. Copying them into
# evidence.jsonl would multiply large model contexts and binary-like payloads.
_BULKY_FIELDS = {
    "autogen_model_messages",
    "role_visible_messages",
    "backend_messages",
    "provider_request_payload",
    "provider_response_payload",
    "task_text",
    "traceback",
    "code_blocks",
    "output",
    "tool_requests",
    "tool_executions",
}


@dataclass(frozen=True)
class _Requirement:
    key: str
    description: str
    selector: Callable[[dict[str, Any]], bool]
    present: Callable[[dict[str, Any]], bool]
    required: bool = True


def _event_type_is(*event_types: str) -> Callable[[dict[str, Any]], bool]:
    values = set(event_types)
    return lambda event: str(event.get("event_type")) in values


def _event_type_starts(*prefixes: str) -> Callable[[dict[str, Any]], bool]:
    return lambda event: str(event.get("event_type", "")).startswith(prefixes)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _attribute_present(event: dict[str, Any], *keys: str) -> bool:
    attributes = event.get("attributes")
    if not isinstance(attributes, dict):
        return False
    return any(_has_value(attributes.get(key)) for key in keys)


_CASE_EVENT_PREFIXES = (
    "trial.",
    "attempt.",
    "runtime.",
    "coordination.",
    "model_call.",
    "agent.",
    "tool_call.",
    "tool_execution.",
    "result_contract.",
    "evaluation.",
)


def _is_case_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "")
    # Admission totals control a whole Run and intentionally have no case ID.
    # The historical event name starts with ``trial.`` because the permission
    # is eventually consumed by Trials, not because the event is case-scoped.
    if event_type == "trial.admission_updated":
        return False
    return event_type.startswith(_CASE_EVENT_PREFIXES)


_CASE_EVENTS = _is_case_event

_REQUIREMENTS = (
    _Requirement(
        "event.identity",
        "Every normalized event has a stable event ID.",
        lambda event: True,
        lambda event: _has_value(event.get("event_id")),
    ),
    _Requirement(
        "event.provenance",
        "Every event points back to its source artifact and source line.",
        lambda event: True,
        lambda event: (
            _attribute_present(event, "source_artifact")
            and _attribute_present(event, "source_line")
        ),
    ),
    _Requirement(
        "event.timestamp",
        "Runtime and chat events include a timestamp.",
        _event_type_starts(
            "run.",
            "trial.",
            "attempt.",
            "runtime.",
            "coordination.",
            "model_call.",
            "agent.",
            "tool_call.",
            "tool_execution.",
            "group_chat.",
        ),
        lambda event: (
            _has_value(event.get("timestamp_utc")) or _has_value(event.get("timestamp_unix_s"))
        ),
    ),
    _Requirement(
        "case.identity",
        "Case-scoped evidence identifies the benchmark case.",
        _CASE_EVENTS,
        lambda event: _has_value(event.get("case_id")),
    ),
    _Requirement(
        "coordination.actor",
        "Coordination operations identify the control Node that executed them.",
        _event_type_is("coordination.operation.completed"),
        lambda event: _has_value(event.get("actor")),
    ),
    _Requirement(
        "coordination.operation",
        "Coordination operation events identify the framework-neutral operation kind.",
        _event_type_is("coordination.operation.completed"),
        lambda event: _attribute_present(event, "operation", "operation_kind", "kind"),
    ),
    _Requirement(
        "coordination.stalled",
        "Stall-detection operations preserve their boolean decision.",
        lambda event: (
            event.get("event_type") == "coordination.operation.completed"
            and str((event.get("attributes") or {}).get("operation")) == "detect_stall"
        ),
        lambda event: isinstance((event.get("attributes") or {}).get("stalled"), bool),
    ),
    _Requirement(
        "message.actor",
        "Agent messages identify their sender.",
        _event_type_is("agent.message.published"),
        lambda event: _has_value(event.get("actor")),
    ),
    _Requirement(
        "message.content",
        "Agent messages preserve their delivered content.",
        _event_type_is("agent.message.published"),
        lambda event: "content" in event,
    ),
    _Requirement(
        "message.visibility",
        "Node messages identify explicit receivers or known visible member Nodes.",
        _event_type_is("agent.message.published"),
        lambda event: _has_value(event.get("receivers")) or _has_value(event.get("visible_to")),
        required=False,
    ),
    _Requirement(
        "model_call.role",
        "Model calls identify the model-backed Node that issued the request.",
        _event_type_is("model_call.started", "model_call.completed", "model_call.failed"),
        lambda event: _has_value(event.get("actor")),
    ),
    _Requirement(
        "model_call.controller",
        "Completed model calls explicitly identify controller and participant calls.",
        _event_type_is("model_call.completed"),
        lambda event: isinstance((event.get("attributes") or {}).get("controller"), bool),
    ),
    _Requirement(
        "model_call.parent_link",
        "Model call end/error events link to their start event.",
        _event_type_is("model_call.completed", "model_call.failed"),
        lambda event: _has_value(event.get("operation_id")),
    ),
    _Requirement(
        "model_call.input_tokens",
        "Completed model calls report their effective input token count.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(
            event,
            "input_total_positions",
            "input_text_tokens",
            "input_tokens_after_budgeting",
        ),
    ),
    _Requirement(
        "model_call.output_tokens",
        "Completed model calls report their output token count.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(
            event,
            "output_total_tokens",
            "output_text_tokens",
        ),
    ),
    _Requirement(
        "model_call.latency",
        "Completed model calls report an explicit model or request latency.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(
            event,
            "model_latency_s",
            "latency_s",
            "model_generation_latency_s",
            "client_model_call_wall_time_s",
        ),
    ),
    _Requirement(
        "model_call.provider_queue_time",
        "Completed model calls report provider-side request queue time when exposed.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(event, "provider_request_queue_latency_s"),
        required=False,
    ),
    _Requirement(
        "model_call.time_to_first_token",
        "Completed model calls report provider scheduling-to-first-token time when exposed.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(event, "provider_scheduled_to_first_token_s"),
        required=False,
    ),
    _Requirement(
        "model_call.generation_latency",
        "Completed model calls report provider generation time when exposed.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(event, "provider_generation_latency_s"),
        required=False,
    ),
    _Requirement(
        "model_call.reasoning_tokens",
        "Completed model calls report the reasoning-token split when available.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(event, "output_reasoning_tokens"),
        required=False,
    ),
    _Requirement(
        "model_call.answer_tokens",
        "Completed model calls report the answer-token split when available.",
        _event_type_is("model_call.completed"),
        lambda event: _attribute_present(event, "output_answer_tokens"),
        required=False,
    ),
    _Requirement(
        "tool_call.correlation_id",
        "Tool requests preserve a tool-call correlation ID.",
        _event_type_is("tool_call.requested"),
        lambda event: _has_value(event.get("correlation_id")),
    ),
    _Requirement(
        "tool_execution.correlation_id",
        "Tool execution events preserve a tool-call correlation ID.",
        _event_type_starts("tool_execution."),
        lambda event: _has_value(event.get("correlation_id")),
    ),
    _Requirement(
        "tool_execution.parent_link",
        "Tool execution end/error events link to their start event.",
        _event_type_is("tool_execution.completed", "tool_execution.failed"),
        lambda event: _has_value(event.get("operation_id")),
    ),
    _Requirement(
        "tool_execution.status",
        "Completed tool executions report success or error status.",
        _event_type_is(
            "tool_execution.completed",
            "tool_execution.failed",
            "tool_execution.observed",
        ),
        lambda event: _has_value(event.get("status")),
    ),
    _Requirement(
        "tool_execution.latency",
        "Terminal tool executions report execution duration when exposed.",
        _event_type_is(
            "tool_execution.completed",
            "tool_execution.failed",
            "tool_execution.observed",
        ),
        lambda event: _attribute_present(event, "duration_s"),
        required=False,
    ),
    _Requirement(
        "attempt.identity",
        "Every Trial attempt has an explicit attempt-start event.",
        _event_type_is("attempt.started"),
        lambda event: event.get("attempt") is not None,
    ),
    _Requirement(
        "trial.prediction",
        "Completed Trials preserve the model/team final prediction.",
        _event_type_is("trial.completed"),
        lambda event: event.get("content") is not None,
    ),
    _Requirement(
        "trial.terminal_status",
        "Every terminal Trial records whether runtime execution completed or failed.",
        _event_type_is("trial.completed", "trial.failed"),
        lambda event: _has_value(event.get("status")),
    ),
    _Requirement(
        "trial.wall_time",
        "Every terminal Trial records end-to-end active execution time.",
        _event_type_is("trial.completed", "trial.failed"),
        lambda event: _attribute_present(event, "case_wall_time_s"),
    ),
    _Requirement(
        "result_contract.validation",
        "Result projection validation records its terminal validity decision.",
        _event_type_is("result_contract.validated", "result_contract.failed"),
        lambda event: event.get("status") in {"completed", "failed"},
    ),
    _Requirement(
        "result_contract.source",
        "Result projection validation identifies the result source when available.",
        _event_type_is("result_contract.validated", "result_contract.failed"),
        lambda event: _attribute_present(event, "source"),
        required=False,
    ),
    _Requirement(
        "result_contract.collector",
        "Result projection validation identifies the collector when available.",
        _event_type_is("result_contract.validated", "result_contract.failed"),
        lambda event: _attribute_present(event, "collector"),
        required=False,
    ),
    _Requirement(
        "evaluation.score",
        "Scored output records preserve the official benchmark score.",
        _event_type_is("evaluation.completed"),
        lambda event: _attribute_present(event, "score", "correct"),
    ),
)


class _CoverageTracker:
    def __init__(self) -> None:
        self.total_events = 0
        self.by_type: Counter[str] = Counter()
        self.eligible: Counter[str] = Counter()
        self.present: Counter[str] = Counter()

    def observe(self, event: dict[str, Any]) -> None:
        self.total_events += 1
        self.by_type[str(event.get("event_type") or "unknown")] += 1
        for requirement in _REQUIREMENTS:
            if requirement.selector(event):
                self.eligible[requirement.key] += 1
                if requirement.present(event):
                    self.present[requirement.key] += 1

    def report(self) -> tuple[list[dict[str, Any]], Counter[str]]:
        rows: list[dict[str, Any]] = []
        statuses: Counter[str] = Counter()
        for requirement in _REQUIREMENTS:
            eligible = self.eligible[requirement.key]
            present = self.present[requirement.key]
            if eligible == 0:
                status = "not_observed"
                ratio = None
            elif present == eligible:
                status = "complete"
                ratio = 1.0
            elif present == 0:
                status = "missing"
                ratio = 0.0
            else:
                status = "partial"
                ratio = round(present / eligible, 6)
            statuses[status] += 1
            rows.append(
                {
                    "requirement": requirement.key,
                    "description": requirement.description,
                    "required": requirement.required,
                    "status": status,
                    "eligible_events": eligible,
                    "present_events": present,
                    "coverage_ratio": ratio,
                }
            )
        return rows, statuses


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if _has_value(item)]
    return [str(value)]


def _first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if _has_value(record.get(key)):
            return record[key]
    return None


def _message_receivers(payload: dict[str, Any]) -> list[str]:
    direct = _first_value(payload, "receivers", "receiver", "target", "recipient")
    if direct is not None:
        return _list_value(direct)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return _list_value(_first_value(metadata, "receivers", "receiver", "target", "recipient"))
    return []


def _event_status(event_type: str, payload: dict[str, Any]) -> str | None:
    explicit = payload.get("status")
    if _has_value(explicit):
        return str(explicit)
    if event_type.endswith(".failed") or bool(payload.get("is_error")):
        return "failed"
    if event_type == "result_contract.validated":
        return "completed"
    if event_type.endswith(".started"):
        return "started"
    if event_type.endswith((".completed", ".ready", ".stopped")):
        return "completed"
    return None


def _correlation_id(event_type: str, record: dict[str, Any], payload: dict[str, Any]) -> str | None:
    direct = record.get("correlation_id") or _first_value(payload, "tool_call_id", "request_id")
    if direct is not None:
        return str(direct)
    if event_type.startswith("model_call.") and payload.get("model_call_index") is not None:
        return ":".join(
            (
                "model_call",
                str(record.get("case_id") or "unknown"),
                str(record.get("trial_index") or 0),
                str(payload["model_call_index"]),
            )
        )
    return None


def _record_attributes(
    record: dict[str, Any], *, source_artifact: str, source_line: int
) -> tuple[dict[str, Any], list[str]]:
    payload = record.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    attributes = {key: value for key, value in payload.items() if key not in _BULKY_FIELDS}
    attributes["operation_id"] = record.get("operation_id")
    attributes["source_artifact"] = source_artifact
    attributes["source_line"] = source_line
    omitted = sorted(key for key in _BULKY_FIELDS if key in payload)
    return attributes, omitted


def _source_record_id(source: str, record: dict[str, Any], line_number: int) -> str:
    if _has_value(record.get("event_id")):
        return str(record["event_id"])
    case_id = str(record.get("case_id") or "unknown")
    trial_index = int(record.get("trial_index") or 0)
    return f"{case_id}:{trial_index}:{line_number}"


def _normalize_record(
    source: str,
    filename: str,
    record: dict[str, Any],
    line_number: int,
    group_state: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    source_record_id = _source_record_id(source, record, line_number)
    event_type = str(record.get("event_type") or "unknown")
    payload = record.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if event_type.startswith("trial."):
        actor = "benchmark_runner"
        content = payload.get("final_output")
    elif event_type.startswith("evaluation."):
        actor = "benchmark_evaluator"
        content = None
    else:
        actor = _first_value(payload, "actor", "node_id", "role", "source")
        if event_type == "tool_call.requested":
            content = payload.get("arguments")
        elif event_type.startswith("tool_execution."):
            content = payload.get("output")
        else:
            content = payload.get("content")
    parent_event_id = record.get("parent_event_id")

    case_key = (
        str(record.get("case_id") or ""),
        str(record.get("trial_index") or 0),
    )
    if event_type == "group_chat.started":
        group_state[case_key] = {
            "member_nodes": _list_value(payload.get("member_nodes")),
            "operation_bindings": _list_value(payload.get("operation_bindings")),
            "context_visibility": payload.get("context_visibility"),
        }
    chat_context = group_state.get(case_key, {})
    visible_to = _list_value(payload.get("visible_to"))
    if (
        not visible_to
        and event_type == "agent.message.published"
        and chat_context.get("context_visibility") == "shared"
    ):
        visible_to = list(chat_context.get("member_nodes") or [])

    attributes, omitted_fields = _record_attributes(
        record,
        source_artifact=filename,
        source_line=line_number,
    )
    if chat_context:
        attributes.setdefault("context_visibility", chat_context.get("context_visibility"))
        attributes.setdefault("operation_bindings", chat_context.get("operation_bindings"))
    event = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": record.get("run_id"),
        "event_id": source_record_id,
        "event_type": event_type,
        "timestamp_unix_s": record.get("timestamp_unix_s"),
        "timestamp_utc": record.get("timestamp_utc"),
        "case_id": record.get("case_id"),
        "dataset_index": record.get("dataset_index"),
        "trial_index": record.get("trial_index"),
        "attempt": record.get("attempt"),
        "actor": str(actor) if _has_value(actor) else None,
        "receivers": _message_receivers(payload),
        "visible_to": visible_to,
        "operation_id": record.get("operation_id"),
        "parent_event_id": parent_event_id,
        "correlation_id": _correlation_id(event_type, record, payload),
        "status": _event_status(event_type, payload),
        "content": content,
        "attributes": attributes,
        "provenance": {
            "source_kind": "run_event",
            "source_artifact": filename,
            "source_line": line_number,
            "source_record_id": source_record_id,
            "source_schema_version": record.get("schema_version"),
            "omitted_fields": omitted_fields,
        },
    }
    return event


def _iter_source_records(path: Path) -> Iterator[tuple[int, dict[str, Any] | Exception]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSONL record must be an object")
                yield line_number, value
            except (TypeError, ValueError) as exc:
                yield line_number, exc


def _iter_normalized_events(
    run_dir: str | os.PathLike,
    source_reports: dict[str, dict[str, Any]] | None = None,
    projection_reports: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    group_state: dict[tuple[str, str], dict[str, Any]] = {}
    projectors = [AutoGenMagenticOneLedgerProjector()]
    for source, filename in _SOURCE_FILES:
        paths = run_event_paths(root) if source == "run_events" else [root / filename]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            if source_reports is not None:
                source_reports[source] = {
                    "source": source,
                    "artifact": filename,
                    "status": "missing",
                    "record_count": 0,
                    "invalid_record_count": 0,
                    "skipped_duplicate_count": 0,
                    "normalized_event_count": 0,
                }
            continue
        report: dict[str, Any] = {
            "source": source,
            "artifact": filename,
            "artifacts": [str(path.relative_to(root)) for path in paths],
            "segment_count": len(paths),
            "status": "empty",
            "record_count": 0,
            "invalid_record_count": 0,
            "skipped_duplicate_count": 0,
            "normalized_event_count": 0,
            "size_bytes": sum(path.stat().st_size for path in paths),
        }
        if source_reports is not None:
            source_reports[source] = report
        for path in paths:
            for line_number, value in _iter_source_records(path):
                report["record_count"] += 1
                report["status"] = "present"
                if isinstance(value, Exception):
                    report["invalid_record_count"] += 1
                    continue
                event = _normalize_record(
                    source,
                    path.name,
                    value,
                    line_number,
                    group_state,
                )
                report["normalized_event_count"] += 1
                yield event
                for projector in projectors:
                    yield from projector.observe(event)
    if projection_reports is not None:
        projection_reports.extend(projector.report() for projector in projectors)


def iter_normalized_events(run_dir: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """Yield canonical events while leaving all source artifacts untouched."""

    yield from _iter_normalized_events(run_dir)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_run_evidence(
    run_dir: str | os.PathLike,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Normalize one run and return a field-level evidence coverage report."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    events_path = root / EVIDENCE_EVENTS_FILENAME
    temporary_path: Path | None = None
    tracker = _CoverageTracker()
    reports_by_source: dict[str, dict[str, Any]] = {}
    projection_reports: list[dict[str, Any]] = []
    if write:
        event_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=events_path.parent,
            prefix=f".{events_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(event_handle.name)
    else:
        event_handle = None
    try:
        for event in _iter_normalized_events(root, reports_by_source, projection_reports):
            tracker.observe(event)
            if event_handle is not None:
                event_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except BaseException:
        if event_handle is not None:
            event_handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    else:
        if event_handle is not None:
            event_handle.close()
            assert temporary_path is not None
            try:
                os.replace(temporary_path, events_path)
            finally:
                temporary_path.unlink(missing_ok=True)

    source_reports = [reports_by_source[source] for source, _ in _SOURCE_FILES]
    invalid_records = sum(item["invalid_record_count"] for item in source_reports)
    requirements, requirement_statuses = tracker.report()
    required_failures = [
        item
        for item in requirements
        if item["required"] and item["status"] in {"missing", "partial"}
    ]
    if tracker.total_events == 0:
        overall_status = "insufficient"
    elif invalid_records or required_failures:
        overall_status = "partial"
    else:
        overall_status = "complete"

    capabilities = {
        "trial_lifecycle": any(key.startswith("trial.") for key in tracker.by_type),
        "model_calls": any(key.startswith("model_call.") for key in tracker.by_type),
        "agent_messages": tracker.by_type["agent.message.published"] > 0,
        "tool_executions": any(key.startswith("tool_execution.") for key in tracker.by_type),
        "trials": tracker.by_type["trial.completed"] > 0,
        "official_evaluations": tracker.by_type["evaluation.completed"] > 0,
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "run_dir": str(root),
        "overall_status": overall_status,
        "summary": {
            "overall_status": overall_status,
            "total_events": tracker.total_events,
            "event_type_counts": dict(sorted(tracker.by_type.items())),
            "requirement_status_counts": dict(sorted(requirement_statuses.items())),
            "required_failure_count": len(required_failures),
            "invalid_source_records": invalid_records,
            "capabilities_observed": capabilities,
        },
        "sources": source_reports,
        "projections": projection_reports,
        "requirements": requirements,
        "artifacts": {
            "events": EVIDENCE_EVENTS_FILENAME if write else None,
            "coverage": EVIDENCE_COVERAGE_FILENAME if write else None,
        },
    }
    if write:
        _atomic_write_json(root / EVIDENCE_COVERAGE_FILENAME, report)
    return report
