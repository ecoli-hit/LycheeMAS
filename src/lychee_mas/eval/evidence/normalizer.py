"""Normalize run artifacts into a framework-neutral evidence event stream."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_EVENTS_FILENAME = "evidence.jsonl"
EVIDENCE_COVERAGE_FILENAME = "evidence_coverage.json"

_SOURCE_FILES = (
    ("spans", "spans.jsonl"),
    ("group_chat", "group_chat.jsonl"),
    ("predictions", "predictions.jsonl"),
    ("outputs", "outputs.jsonl"),
)

_ENVELOPE_FIELDS = {
    "schema_version",
    "seq",
    "span_id",
    "parent_span_id",
    "span_type",
    "event_id",
    "event_type",
    "timestamp_unix_s",
    "timestamp_utc",
    "case_id",
    "sample_index",
    "role",
    "source",
    "receiver",
    "receivers",
    "target",
    "visible_to",
    "content",
    "final_answer",
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


_CASE_EVENTS = _event_type_starts(
    "case.",
    "case_attempt.",
    "runtime.",
    "model_call.",
    "agent.",
    "tool_",
    "prediction.",
    "evaluation.",
)

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
        lambda event: _attribute_present(event, "source_artifact")
        and _attribute_present(event, "source_line"),
    ),
    _Requirement(
        "event.timestamp",
        "Runtime and chat events include a timestamp.",
        _event_type_starts(
            "run.",
            "case.",
            "case_attempt.",
            "runtime.",
            "model_call.",
            "agent.",
            "tool_",
            "group_chat.",
        ),
        lambda event: _has_value(event.get("timestamp_utc"))
        or _has_value(event.get("timestamp_unix_s")),
    ),
    _Requirement(
        "case.identity",
        "Case-scoped evidence identifies the benchmark case.",
        _CASE_EVENTS,
        lambda event: _has_value(event.get("case_id")),
    ),
    _Requirement(
        "message.actor",
        "Agent messages identify their sender.",
        _event_type_is("agent.message"),
        lambda event: _has_value(event.get("actor")),
    ),
    _Requirement(
        "message.content",
        "Agent messages preserve their delivered content.",
        _event_type_is("agent.message"),
        lambda event: _has_value(event.get("content")),
    ),
    _Requirement(
        "message.visibility",
        "Agent messages identify explicit receivers or known visible participants.",
        _event_type_is("agent.message"),
        lambda event: _has_value(event.get("receivers"))
        or _has_value(event.get("visible_to")),
        required=False,
    ),
    _Requirement(
        "model_call.role",
        "Model calls identify the participant or controller role.",
        _event_type_starts("model_call."),
        lambda event: _has_value(event.get("actor")),
    ),
    _Requirement(
        "model_call.parent_link",
        "Model call end/error events link to their start event.",
        _event_type_is("model_call.end", "model_call.error"),
        lambda event: _has_value(event.get("parent_event_id")),
    ),
    _Requirement(
        "model_call.input_tokens",
        "Completed model calls report their effective input token count.",
        _event_type_is("model_call.end"),
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
        _event_type_is("model_call.end"),
        lambda event: _attribute_present(
            event,
            "output_total_tokens",
            "output_text_tokens",
        ),
    ),
    _Requirement(
        "model_call.reasoning_tokens",
        "Completed model calls report the reasoning-token split when available.",
        _event_type_is("model_call.end"),
        lambda event: _attribute_present(event, "output_reasoning_tokens"),
        required=False,
    ),
    _Requirement(
        "model_call.answer_tokens",
        "Completed model calls report the answer-token split when available.",
        _event_type_is("model_call.end"),
        lambda event: _attribute_present(event, "output_answer_tokens"),
        required=False,
    ),
    _Requirement(
        "tool_call.correlation_id",
        "Tool requests preserve a tool-call correlation ID.",
        _event_type_is("tool_call.request"),
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
        _event_type_is("tool_execution.end", "tool_execution.error"),
        lambda event: _has_value(event.get("parent_event_id")),
    ),
    _Requirement(
        "tool_execution.status",
        "Completed tool executions report success or error status.",
        _event_type_is("tool_execution.end", "tool_execution.error"),
        lambda event: _has_value(event.get("status")),
    ),
    _Requirement(
        "prediction.final_answer",
        "Prediction records preserve the model/team final answer.",
        _event_type_is("prediction.record"),
        lambda event: event.get("content") is not None,
    ),
    _Requirement(
        "evaluation.score",
        "Scored output records preserve the official benchmark score.",
        _event_type_is("evaluation.record"),
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


def _sample_identity(record: dict[str, Any]) -> Any:
    sample_index = record.get("sample_index")
    return sample_index if sample_index is not None else record.get("k_index", 0)


def _message_receivers(record: dict[str, Any]) -> list[str]:
    direct = _first_value(record, "receivers", "receiver", "target", "recipient")
    if direct is not None:
        return _list_value(direct)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return _list_value(
            _first_value(metadata, "receivers", "receiver", "target", "recipient")
        )
    return []


def _canonical_span_type(span_type: str) -> str:
    aliases = {
        "autogen_message": "agent.message",
        "tool_event": "agent.tool_event",
        "group_chat_selected": "group_chat.selected",
        "resume_start": "run.resume",
    }
    if span_type in aliases:
        return aliases[span_type]
    for suffix in ("start", "ready", "end", "error", "stop"):
        marker = f"_{suffix}"
        if span_type.endswith(marker):
            return f"{span_type[: -len(marker)]}.{suffix}"
    return span_type.replace("_", ".")


def _canonical_group_type(event_type: str) -> str:
    if event_type == "message":
        return "agent.message"
    if event_type.startswith("group_chat_"):
        return event_type.replace("group_chat_", "group_chat.", 1)
    return f"group_chat.{event_type.replace('_', '.')}"


def _event_status(event_type: str, record: dict[str, Any]) -> str | None:
    explicit = record.get("status")
    if _has_value(explicit):
        return str(explicit)
    if event_type.endswith(".error") or bool(record.get("is_error")):
        return "error"
    if event_type.endswith(".start"):
        return "started"
    if event_type.endswith((".end", ".ready", ".stop")):
        return "completed"
    return None


def _correlation_id(event_type: str, record: dict[str, Any]) -> str | None:
    direct = _first_value(record, "tool_call_id", "request_id", "correlation_id")
    if direct is not None:
        return str(direct)
    if event_type.startswith("model_call.") and record.get("model_call_index") is not None:
        return ":".join(
            (
                "model_call",
                str(record.get("case_id") or "unknown"),
                str(_sample_identity(record)),
                str(record["model_call_index"]),
            )
        )
    return None


def _record_attributes(
    record: dict[str, Any], *, source_artifact: str, source_line: int
) -> tuple[dict[str, Any], list[str]]:
    attributes = {
        key: value
        for key, value in record.items()
        if key not in _ENVELOPE_FIELDS and key not in _BULKY_FIELDS
    }
    attributes["source_artifact"] = source_artifact
    attributes["source_line"] = source_line
    omitted = sorted(key for key in _BULKY_FIELDS if key in record)
    return attributes, omitted


def _source_record_id(source: str, record: dict[str, Any], line_number: int) -> str:
    if source == "spans" and _has_value(record.get("span_id")):
        return str(record["span_id"])
    if source == "group_chat" and _has_value(record.get("event_id")):
        return str(record["event_id"])
    case_id = str(record.get("case_id") or "unknown")
    sample = _sample_identity(record)
    return f"{case_id}:{sample}:{line_number}"


def _normalize_record(
    source: str,
    filename: str,
    record: dict[str, Any],
    line_number: int,
    group_state: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    source_record_id = _source_record_id(source, record, line_number)
    if source == "spans":
        event_type = _canonical_span_type(str(record.get("span_type") or "unknown"))
        parent = record.get("parent_span_id")
        parent_event_id = f"spans:{parent}" if _has_value(parent) else None
        actor = _first_value(record, "role", "source")
        content = record.get("content")
    elif source == "group_chat":
        raw_type = str(record.get("event_type") or "unknown")
        event_type = _canonical_group_type(raw_type)
        parent_event_id = None
        actor = record.get("source")
        content = record.get("content")
    elif source == "predictions":
        event_type = "prediction.record"
        parent_event_id = None
        actor = "benchmark_runner"
        content = record.get("final_answer")
    else:
        event_type = "evaluation.record"
        parent_event_id = None
        actor = "benchmark_evaluator"
        content = record.get("final_answer")

    case_key = (
        str(record.get("case_id") or ""),
        str(_sample_identity(record)),
    )
    if source == "group_chat" and event_type == "group_chat.start":
        group_state[case_key] = {
            "participants": _list_value(record.get("participants")),
            "context_visibility": record.get("context_visibility"),
        }
    chat_context = group_state.get(case_key, {}) if source == "group_chat" else {}
    visible_to = _list_value(record.get("visible_to"))
    if (
        not visible_to
        and event_type == "agent.message"
        and chat_context.get("context_visibility") == "shared"
    ):
        visible_to = list(chat_context.get("participants") or [])

    attributes, omitted_fields = _record_attributes(
        record,
        source_artifact=filename,
        source_line=line_number,
    )
    if chat_context:
        attributes.setdefault("context_visibility", chat_context.get("context_visibility"))
    event = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "event_id": f"{source}:{source_record_id}",
        "event_type": event_type,
        "timestamp_unix_s": record.get("timestamp_unix_s"),
        "timestamp_utc": record.get("timestamp_utc"),
        "case_id": record.get("case_id"),
        "sample_index": record.get("sample_index"),
        "prediction_index": record.get("k_index"),
        "actor": str(actor) if _has_value(actor) else None,
        "receivers": _message_receivers(record),
        "visible_to": visible_to,
        "parent_event_id": parent_event_id,
        "correlation_id": _correlation_id(event_type, record),
        "status": _event_status(event_type, record),
        "content": content,
        "attributes": attributes,
        "provenance": {
            "source_kind": source,
            "source_artifact": filename,
            "source_line": line_number,
            "source_record_id": source_record_id,
            "source_schema_version": record.get("schema_version"),
            "omitted_fields": omitted_fields,
        },
    }
    return event


def _embedded_tool_events(
    message_event: dict[str, Any],
    record: dict[str, Any],
    tool_request_ids: dict[str, str],
) -> Iterator[dict[str, Any]]:
    provenance = message_event["provenance"]
    base_id = str(message_event["event_id"])
    for index, request in enumerate(record.get("tool_requests") or []):
        if not isinstance(request, dict):
            continue
        correlation_id = str(request.get("tool_call_id") or "") or None
        event_id = f"{base_id}:tool_request:{index}"
        if correlation_id:
            tool_request_ids[correlation_id] = event_id
        yield {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": "tool_call.request",
            "timestamp_unix_s": message_event.get("timestamp_unix_s"),
            "timestamp_utc": message_event.get("timestamp_utc"),
            "case_id": message_event.get("case_id"),
            "sample_index": message_event.get("sample_index"),
            "prediction_index": message_event.get("prediction_index"),
            "actor": str(request.get("source") or message_event.get("actor") or "") or None,
            "receivers": [],
            "visible_to": list(message_event.get("visible_to") or []),
            "parent_event_id": base_id,
            "correlation_id": correlation_id,
            "status": "requested",
            "content": request.get("arguments"),
            "attributes": {
                **request,
                "source_artifact": provenance["source_artifact"],
                "source_line": provenance["source_line"],
            },
            "provenance": {
                **provenance,
                "embedded_field": "tool_requests",
                "embedded_index": index,
            },
        }
    for index, execution in enumerate(record.get("tool_executions") or []):
        if not isinstance(execution, dict):
            continue
        correlation_id = str(execution.get("tool_call_id") or "") or None
        is_error = bool(execution.get("is_error"))
        yield {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "event_id": f"{base_id}:tool_execution:{index}",
            "event_type": "tool_execution.end",
            "timestamp_unix_s": message_event.get("timestamp_unix_s"),
            "timestamp_utc": message_event.get("timestamp_utc"),
            "case_id": message_event.get("case_id"),
            "sample_index": message_event.get("sample_index"),
            "prediction_index": message_event.get("prediction_index"),
            "actor": str(execution.get("source") or message_event.get("actor") or "")
            or None,
            "receivers": [],
            "visible_to": list(message_event.get("visible_to") or []),
            "parent_event_id": (
                tool_request_ids.get(correlation_id, base_id) if correlation_id else base_id
            ),
            "correlation_id": correlation_id,
            "status": "error" if is_error else "completed",
            "content": execution.get("output"),
            "attributes": {
                **execution,
                "source_artifact": provenance["source_artifact"],
                "source_line": provenance["source_line"],
            },
            "provenance": {
                **provenance,
                "embedded_field": "tool_executions",
                "embedded_index": index,
            },
        }


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
) -> Iterator[dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    group_chat_path = root / "group_chat.jsonl"
    has_group_chat = group_chat_path.is_file() and group_chat_path.stat().st_size > 0
    group_state: dict[tuple[str, str], dict[str, Any]] = {}
    tool_request_ids: dict[str, str] = {}
    for source, filename in _SOURCE_FILES:
        path = root / filename
        if not path.is_file():
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
        report = {
            "source": source,
            "artifact": filename,
            "status": "empty",
            "record_count": 0,
            "invalid_record_count": 0,
            "skipped_duplicate_count": 0,
            "normalized_event_count": 0,
            "size_bytes": path.stat().st_size,
        }
        if source_reports is not None:
            source_reports[source] = report
        for line_number, value in _iter_source_records(path):
            report["record_count"] += 1
            report["status"] = "present"
            if isinstance(value, Exception):
                report["invalid_record_count"] += 1
                continue
            if (
                source == "spans"
                and has_group_chat
                and value.get("span_type") in {"autogen_message", "tool_event"}
            ):
                report["skipped_duplicate_count"] += 1
                continue
            event = _normalize_record(
                source,
                filename,
                value,
                line_number,
                group_state,
            )
            report["normalized_event_count"] += 1
            yield event
            if source == "group_chat" and event["event_type"] == "agent.message":
                for tool_event in _embedded_tool_events(event, value, tool_request_ids):
                    report["normalized_event_count"] += 1
                    yield tool_event


def iter_normalized_events(run_dir: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """Yield canonical events while leaving all source artifacts untouched."""

    yield from _iter_normalized_events(run_dir)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
    temporary_path = events_path.with_name(f".{events_path.name}.tmp")
    tracker = _CoverageTracker()
    reports_by_source: dict[str, dict[str, Any]] = {}
    event_handle = temporary_path.open("w", encoding="utf-8") if write else None
    try:
        for event in _iter_normalized_events(root, reports_by_source):
            tracker.observe(event)
            if event_handle is not None:
                event_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except BaseException:
        if event_handle is not None:
            event_handle.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        if event_handle is not None:
            event_handle.close()
            os.replace(temporary_path, events_path)

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
        "case_lifecycle": any(key.startswith("case.") for key in tracker.by_type),
        "model_calls": any(key.startswith("model_call.") for key in tracker.by_type),
        "agent_messages": tracker.by_type["agent.message"] > 0,
        "tool_executions": any(
            key.startswith("tool_execution.") for key in tracker.by_type
        ),
        "predictions": tracker.by_type["prediction.record"] > 0,
        "official_evaluations": tracker.by_type["evaluation.record"] > 0,
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
        "requirements": requirements,
        "artifacts": {
            "events": EVIDENCE_EVENTS_FILENAME if write else None,
            "coverage": EVIDENCE_COVERAGE_FILENAME if write else None,
        },
    }
    if write:
        _atomic_write_json(root / EVIDENCE_COVERAGE_FILENAME, report)
    return report
