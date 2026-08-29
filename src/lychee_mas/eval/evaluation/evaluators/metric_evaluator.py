"""Deterministically evaluate current Metric Contracts from normalized evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ..evidence import EVIDENCE_EVENTS_FILENAME, iter_normalized_events
from ..metrics_registry import MetricContract, MetricObservation, MetricRegistry
from .profiles import EvaluationProfile, EvaluationProfileRegistry

METRIC_EVALUATION_SCHEMA_VERSION = 1
METRIC_OBSERVATIONS_FILENAME = "metric_observations.jsonl"
METRIC_EVALUATION_FILENAME = "metric_evaluation.json"
TrialKey = tuple[str, int | None, int]
Evaluator = Callable[[MetricContract, TrialKey, list[dict[str, Any]], str], MetricObservation]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def _events(run_dir: Path) -> Iterable[dict[str, Any]]:
    path = run_dir / EVIDENCE_EVENTS_FILENAME
    return _read_jsonl(path) if path.is_file() else iter_normalized_events(run_dir)


def _trial_key(event: dict[str, Any]) -> TrialKey | None:
    case_id = event.get("case_id")
    if case_id is None or not str(case_id).strip():
        return None
    dataset_index = event.get("dataset_index")
    trial_index = event.get("trial_index")
    return (
        str(case_id),
        int(dataset_index) if dataset_index is not None else None,
        int(trial_index or 0),
    )


def _event_ids(events: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item["event_id"]) for item in events if item.get("event_id"))


def _attributes(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes")
    return value if isinstance(value, dict) else {}


def _number(event: dict[str, Any], *keys: str) -> float | int | None:
    attributes = _attributes(event)
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _fingerprint(contract: MetricContract, profile: EvaluationProfile) -> str:
    evaluator = contract.evaluator or {}
    payload = {
        "contract_fingerprint": contract.fingerprint,
        "profile_fingerprint": profile.fingerprint,
        "entrypoint": evaluator.get("entrypoint"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _observation(
    contract: MetricContract,
    key: TrialKey,
    fingerprint: str,
    *,
    status: str,
    value: Any = None,
    reason: str | None = None,
    evidence: Iterable[dict[str, Any]] = (),
    attributes: dict[str, Any] | None = None,
) -> MetricObservation:
    return MetricObservation(
        metric_id=contract.metric_id,
        status=status,
        value=value,
        unit=contract.unit,
        level=contract.level,
        case_id=key[0],
        dataset_index=key[1],
        trial_index=key[2],
        reason=reason,
        evidence_event_ids=_event_ids(evidence),
        evaluator_fingerprint=fingerprint,
        attributes=attributes or {},
    )


def _select(events: list[dict[str, Any]], *types: str) -> list[dict[str, Any]]:
    expected = set(types)
    return [event for event in events if event.get("event_type") in expected]


def _official_score(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    records = _select(events, "evaluation.completed")
    if not records:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="no evaluation.completed event exists for this Trial",
        )
    event = records[-1]
    value = _number(event, "score", "correct")
    if value is None:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="evaluation.completed does not contain numeric score/correct",
            evidence=records,
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=float(value),
        evidence=(event,),
        attributes={"record_count": len(records), "selected_record": "last"},
    )


def _message_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        parts = [_message_text(item) for item in value]
        text_parts = [part for part in parts if part]
        return "\n".join(text_parts) if text_parts else None
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return _message_text(value[key])
    return None


def _normalized_message(value: Any) -> str | None:
    text = _message_text(value)
    if text is None:
        return None
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _coordination_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select messages published by team Nodes without counting task input."""

    return [
        event
        for event in _select(events, "agent.message.published")
        if str(event.get("actor") or "").strip().casefold() not in {"user", "system"}
        and not bool((event.get("attributes") or {}).get("is_tool_event"))
    ]


def _message_repetition(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    messages = _coordination_messages(events)
    if not messages:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no delivered agent messages were observed",
        )
    normalized = [_normalized_message(event.get("content")) for event in messages]
    values = [value for value in normalized if value is not None]
    if not values:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no text-bearing agent messages were observed",
            evidence=messages,
        )
    duplicate_count = len(values) - len(set(values))
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(duplicate_count / len(values), 8),
        evidence=messages,
        attributes={
            "message_count": len(values),
            "delivered_message_count": len(messages),
            "non_text_message_count": len(messages) - len(values),
            "duplicate_message_count": duplicate_count,
            "normalization": (
                "multimodal_text_projection_unicode_nfkc_casefold_whitespace_collapse"
            ),
        },
    )


def _role_balance(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    messages = _coordination_messages(events)
    if not messages:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no delivered agent messages were observed",
        )
    if any(not event.get("actor") for event in messages):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more delivered agent messages have no actor",
            evidence=messages,
        )
    counts = Counter(str(event["actor"]) for event in messages)
    if len(counts) < 2:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="participation balance requires at least two observed roles",
            evidence=messages,
            attributes={"role_message_counts": dict(sorted(counts.items()))},
        )
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    balance = entropy / math.log(len(counts))
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(balance, 8),
        evidence=messages,
        attributes={
            "message_count": total,
            "role_count": len(counts),
            "role_message_counts": dict(sorted(counts.items())),
            "formula": "shannon_entropy/log(observed_role_count)",
        },
    )


def _model_call_participation_balance(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    if any(not event.get("actor") for event in calls):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more completed model calls have no actor",
            evidence=calls,
        )
    counts = Counter(str(event["actor"]) for event in calls)
    if len(counts) < 2:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="model-call participation balance requires at least two observed roles",
            evidence=calls,
            attributes={"role_model_call_counts": dict(sorted(counts.items()))},
        )
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(entropy / math.log(len(counts)), 8),
        evidence=calls,
        attributes={
            "model_call_count": total,
            "role_count": len(counts),
            "role_model_call_counts": dict(sorted(counts.items())),
            "formula": "shannon_entropy/log(observed_role_count)",
        },
    )


def _controller_flags(
    contract: MetricContract,
    key: TrialKey,
    calls: list[dict[str, Any]],
    fingerprint: str,
) -> tuple[list[bool] | None, MetricObservation | None]:
    values = [_attributes(event).get("controller") for event in calls]
    if any(not isinstance(value, bool) for value in values):
        return None, _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more completed model calls lack an explicit controller boolean",
            evidence=calls,
            attributes={"completed_model_call_count": len(calls)},
        )
    return [bool(value) for value in values], None


def _controller_model_call_ratio(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    flags, error = _controller_flags(contract, key, calls, fingerprint)
    if error is not None:
        return error
    controller_count = sum(flags or [])
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(controller_count / len(calls), 8),
        evidence=calls,
        attributes={
            "completed_model_call_count": len(calls),
            "controller_model_call_count": controller_count,
            "formula": "controller_model_calls/completed_model_calls",
        },
    )


def _controller_output_token_ratio(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    flags, error = _controller_flags(contract, key, calls, fingerprint)
    if error is not None:
        return error
    values = [_number(event, "output_total_tokens", "output_text_tokens") for event in calls]
    if any(value is None for value in values):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more completed model calls lack output token count",
            evidence=calls,
        )
    total_tokens = sum(float(value) for value in values if value is not None)
    if total_tokens <= 0:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="completed model calls produced no output tokens",
            evidence=calls,
        )
    controller_tokens = sum(
        float(value)
        for value, is_controller in zip(values, flags or [])
        if value is not None and is_controller
    )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(controller_tokens / total_tokens, 8),
        evidence=calls,
        attributes={
            "output_tokens": round(total_tokens, 8),
            "controller_output_tokens": round(controller_tokens, 8),
            "formula": "controller_output_tokens/all_output_tokens",
        },
    )


def _declares_operation(events: list[dict[str, Any]], operation: str | None = None) -> bool:
    for event in events:
        bindings = _attributes(event).get("operation_bindings")
        if not isinstance(bindings, list):
            continue
        for binding in bindings:
            if isinstance(binding, dict):
                candidate = binding.get("operation")
            else:
                candidate = str(binding)
            if operation is None or str(operation) in str(candidate):
                return True
    return False


def _coordination_operations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _select(events, "coordination.operation.completed")


def _operation_kind(event: dict[str, Any]) -> str:
    attributes = _attributes(event)
    return str(
        attributes.get("operation")
        or attributes.get("operation_kind")
        or attributes.get("kind")
        or ""
    )


def _coordination_operation_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    operations = _coordination_operations(events)
    if not operations:
        status = "missing_evidence" if _declares_operation(events) else "not_applicable"
        reason = (
            "TeamSpec declares coordination operations but no operation events were observed"
            if status == "missing_evidence"
            else "the team declares no observable coordination operations"
        )
        return _observation(contract, key, fingerprint, status=status, reason=reason)
    counts = Counter(_operation_kind(event) or "unknown" for event in operations)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=len(operations),
        evidence=operations,
        attributes={"operation_counts": dict(sorted(counts.items()))},
    )


def _replan_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    operations = _coordination_operations(events)
    replans = [event for event in operations if _operation_kind(event) == "replan"]
    if not operations:
        status = "missing_evidence" if _declares_operation(events, "replan") else "not_applicable"
        reason = (
            "TeamSpec declares replan but no coordination operation events were observed"
            if status == "missing_evidence"
            else "the team declares no replan operation"
        )
        return _observation(contract, key, fingerprint, status=status, reason=reason)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=len(replans),
        evidence=operations,
        attributes={"coordination_operation_count": len(operations)},
    )


def _stall_rate(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    checks = [
        event
        for event in _coordination_operations(events)
        if _operation_kind(event) == "detect_stall"
    ]
    if not checks:
        status = (
            "missing_evidence"
            if _declares_operation(events, "detect_stall")
            else "not_applicable"
        )
        reason = (
            "TeamSpec declares detect_stall but no stall-check events were observed"
            if status == "missing_evidence"
            else "the team declares no stall-detection operation"
        )
        return _observation(contract, key, fingerprint, status=status, reason=reason)
    values = [_attributes(event).get("stalled") for event in checks]
    if any(not isinstance(value, bool) for value in values):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more stall-check events lack an explicit stalled boolean",
            evidence=checks,
        )
    stalled = sum(bool(value) for value in values)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(stalled / len(checks), 8),
        evidence=checks,
        attributes={"stall_check_count": len(checks), "stalled_check_count": stalled},
    )


def _post_tool_failure_trial_completion(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    executions = _select(
        events,
        "tool_execution.completed",
        "tool_execution.failed",
        "tool_execution.observed",
    )
    failures = [
        event for event in executions if event.get("status") in {"failed", "error"}
    ]
    if not failures:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no failed tool execution created a recovery opportunity",
            evidence=executions,
        )
    terminal = _select(events, "trial.completed", "trial.failed")
    if not terminal:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="tool failures were observed but the Trial has no terminal event",
            evidence=failures,
        )
    completed = terminal[-1].get("event_type") == "trial.completed"
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=1.0 if completed else 0.0,
        evidence=[*failures, terminal[-1]],
        attributes={
            "failed_tool_execution_count": len(failures),
            "interpretation": "Trial completion after one or more tool failures",
        },
    )


def _sum_model_value(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
    keys: tuple[str, ...],
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    values = [_number(event, *keys) for event in calls]
    if any(value is None for value in values):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason=f"one or more model_call.completed events lack {'/'.join(keys)}",
            evidence=calls,
            attributes={"completed_model_call_count": len(calls)},
        )
    total = sum(float(value) for value in values if value is not None)
    if all(isinstance(value, int) for value in values):
        result: float | int = int(total)
    else:
        result = round(total, 8)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=result,
        evidence=calls,
        attributes={"completed_model_call_count": len(calls), "aggregation": "sum"},
    )


def _model_latency(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    keys = (
        "model_latency_s",
        "latency_s",
        "model_generation_latency_s",
        "client_model_call_wall_time_s",
    )
    values = [_number(event, *keys) for event in calls]
    if any(value is None for value in values):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more model_call.completed events lack an explicit latency",
            evidence=calls,
            attributes={"completed_model_call_count": len(calls)},
        )
    cleaned = [float(value) for value in values if value is not None]
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(sum(cleaned), 8),
        evidence=calls,
        attributes={
            "completed_model_call_count": len(calls),
            "mean_model_call_latency_s": round(statistics.fmean(cleaned), 8),
            "aggregation": "sum_per_trial",
            "timing_scope": "first_available_explicit_model_or_request_latency",
        },
    )


def _sum_explicit_timing(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
    *,
    event_types: tuple[str, ...],
    keys: tuple[str, ...],
    scope: str,
) -> MetricObservation:
    selected = _select(events, *event_types)
    if not selected:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason=f"no {scope} events were observed",
        )
    values = [_number(event, *keys) for event in selected]
    if any(value is None for value in values):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason=f"one or more {scope} events lack {'/'.join(keys)}",
            evidence=selected,
            attributes={"event_count": len(selected)},
        )
    cleaned = [float(value) for value in values if value is not None]
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(sum(cleaned), 8),
        evidence=selected,
        attributes={
            "event_count": len(selected),
            "mean_s": round(statistics.fmean(cleaned), 8),
            "aggregation": "sum_per_trial",
            "timing_scope": scope,
        },
    )


def _generation_throughput(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    calls = _select(events, "model_call.completed")
    if not calls:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no completed model calls were observed",
        )
    token_values = [
        _number(event, "output_total_tokens", "output_text_tokens") for event in calls
    ]
    latency_values = [_number(event, "provider_generation_latency_s") for event in calls]
    if any(value is None for value in [*token_values, *latency_values]):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more model calls lack output tokens or provider generation latency",
            evidence=calls,
        )
    total_tokens = sum(float(value) for value in token_values if value is not None)
    total_latency = sum(float(value) for value in latency_values if value is not None)
    if total_latency <= 0:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="provider generation latency must be greater than zero",
            evidence=calls,
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(total_tokens / total_latency, 8),
        evidence=calls,
        attributes={
            "output_tokens": round(total_tokens, 8),
            "provider_generation_latency_s": round(total_latency, 8),
            "formula": "sum(output_tokens)/sum(provider_generation_latency_s)",
        },
    )


def _model_call_error_rate(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    terminal = _select(events, "model_call.completed", "model_call.failed")
    if not terminal:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no terminal model call events were observed",
        )
    failures = sum(event.get("event_type") == "model_call.failed" for event in terminal)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(failures / len(terminal), 8),
        evidence=terminal,
        attributes={
            "terminal_model_call_count": len(terminal),
            "failed_model_call_count": failures,
        },
    )


def _trial_retry_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    attempts = _select(events, "attempt.started")
    if not attempts:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="no attempt.started event exists for this Trial",
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=max(0, len(attempts) - 1),
        evidence=attempts,
        attributes={"attempt_count": len(attempts), "formula": "attempt_count-1"},
    )


def _role_switch_rate(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    messages = _coordination_messages(events)
    if len(messages) < 2:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="role switch rate requires at least two delivered agent messages",
            evidence=messages,
        )
    if any(not event.get("actor") for event in messages):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more delivered agent messages have no actor",
            evidence=messages,
        )
    actors = [str(event["actor"]) for event in messages]
    switches = sum(left != right for left, right in zip(actors, actors[1:]))
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(switches / (len(actors) - 1), 8),
        evidence=messages,
        attributes={
            "message_count": len(actors),
            "transition_count": len(actors) - 1,
            "role_switch_count": switches,
        },
    )


def _tool_error_rate(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    executions = _select(
        events,
        "tool_execution.completed",
        "tool_execution.failed",
        "tool_execution.observed",
    )
    if not executions:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no tool execution was observed",
        )
    if any(event.get("status") not in {"completed", "failed", "error"} for event in executions):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more tool execution events lack completed/failed status",
            evidence=executions,
        )
    errors = sum(event.get("status") in {"failed", "error"} for event in executions)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(errors / len(executions), 8),
        evidence=executions,
        attributes={
            "tool_execution_count": len(executions),
            "tool_error_count": errors,
        },
    )


def _event_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
    event_types: tuple[str, ...],
    *,
    no_events_status: str = "not_applicable",
) -> MetricObservation:
    selected = _select(events, *event_types)
    if not selected and no_events_status != "measured":
        return _observation(
            contract,
            key,
            fingerprint,
            status=no_events_status,
            reason=f"no {'/'.join(event_types)} events were observed",
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=len(selected),
        evidence=selected,
        attributes={"counted_event_types": list(event_types)},
    )


def _agent_message_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    messages = _coordination_messages(events)
    if not messages:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no team Node messages were observed",
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=len(messages),
        evidence=messages,
        attributes={"excluded_actors": ["system", "user"]},
    )


def _active_role_count(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    messages = _coordination_messages(events)
    if not messages:
        return _observation(
            contract,
            key,
            fingerprint,
            status="not_applicable",
            reason="no delivered agent messages were observed",
        )
    if any(not event.get("actor") for event in messages):
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="one or more delivered agent messages have no actor",
            evidence=messages,
        )
    roles = sorted({str(event["actor"]) for event in messages})
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=len(roles),
        evidence=messages,
        attributes={"active_roles": roles},
    )


def _trial_wall_time(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    terminal = _select(events, "trial.completed", "trial.failed")
    if not terminal:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="no terminal Trial event exists",
        )
    value = _number(terminal[-1], "case_wall_time_s")
    if value is None:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="terminal Trial event lacks case_wall_time_s",
            evidence=terminal,
        )
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=round(float(value), 8),
        evidence=(terminal[-1],),
    )


def _trial_runtime_success(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    terminal = _select(events, "trial.completed", "trial.failed")
    if not terminal:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="no terminal Trial event exists",
        )
    event = terminal[-1]
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=1.0 if event.get("event_type") == "trial.completed" else 0.0,
        evidence=(event,),
    )


def _result_contract_valid(
    contract: MetricContract,
    key: TrialKey,
    events: list[dict[str, Any]],
    fingerprint: str,
) -> MetricObservation:
    records = _select(
        events,
        "result_contract.validated",
        "result_contract.failed",
    )
    if not records:
        return _observation(
            contract,
            key,
            fingerprint,
            status="missing_evidence",
            reason="no benchmark-aware result contract event exists",
        )
    event = records[-1]
    attributes = _attributes(event)
    return _observation(
        contract,
        key,
        fingerprint,
        status="measured",
        value=1.0 if event.get("event_type") == "result_contract.validated" else 0.0,
        reason=str(attributes.get("reason") or "") or None,
        evidence=(event,),
        attributes={
            "result_kind": attributes.get("kind"),
            "result_source": attributes.get("source"),
            "collector": attributes.get("collector"),
            "payload_empty": attributes.get("payload_empty"),
        },
    )


_EVALUATORS: dict[str, Evaluator] = {
    "Benchmark.score": _official_score,
    "coordination.message_repetition_rate": _message_repetition,
    "coordination.role_participation_balance": _role_balance,
    "coordination.model_call_participation_balance": _model_call_participation_balance,
    "coordination.controller_model_call_ratio": _controller_model_call_ratio,
    "coordination.controller_output_token_ratio": _controller_output_token_ratio,
    "coordination.coordination_operation_count": _coordination_operation_count,
    "coordination.replan_count": _replan_count,
    "coordination.stall_rate": _stall_rate,
    "coordination.active_role_count": _active_role_count,
    "coordination.role_switch_rate": _role_switch_rate,
    "coordination.agent_message_count": lambda c, k, e, f: _agent_message_count(
        c, k, e, f
    ),
    "efficiency.input_tokens": lambda c, k, e, f: _sum_model_value(
        c,
        k,
        e,
        f,
        ("input_total_positions", "input_text_tokens", "input_tokens_after_budgeting"),
    ),
    "efficiency.output_tokens": lambda c, k, e, f: _sum_model_value(
        c, k, e, f, ("output_total_tokens", "output_text_tokens")
    ),
    "efficiency.model_call_latency": _model_latency,
    "efficiency.provider_queue_time": lambda c, k, e, f: _sum_explicit_timing(
        c,
        k,
        e,
        f,
        event_types=("model_call.completed",),
        keys=("provider_request_queue_latency_s",),
        scope="provider_request_queue",
    ),
    "efficiency.time_to_first_token": lambda c, k, e, f: _sum_explicit_timing(
        c,
        k,
        e,
        f,
        event_types=("model_call.completed",),
        keys=("provider_scheduled_to_first_token_s",),
        scope="provider_scheduled_to_first_token",
    ),
    "efficiency.generation_throughput": _generation_throughput,
    "efficiency.model_call_count": lambda c, k, e, f: _event_count(
        c, k, e, f, ("model_call.completed",)
    ),
    "efficiency.tool_execution_count": lambda c, k, e, f: _event_count(
        c,
        k,
        e,
        f,
        ("tool_execution.completed", "tool_execution.failed", "tool_execution.observed"),
    ),
    "efficiency.tool_execution_latency": lambda c, k, e, f: _sum_explicit_timing(
        c,
        k,
        e,
        f,
        event_types=(
            "tool_execution.completed",
            "tool_execution.failed",
            "tool_execution.observed",
        ),
        keys=("duration_s",),
        scope="tool_execution",
    ),
    "efficiency.trial_wall_time": _trial_wall_time,
    "reliability.tool_error_rate": _tool_error_rate,
    "reliability.model_call_error_rate": _model_call_error_rate,
    "reliability.trial_retry_count": _trial_retry_count,
    "reliability.trial_runtime_success": _trial_runtime_success,
    "reliability.result_contract_valid": _result_contract_valid,
    "reliability.post_tool_failure_trial_completion": _post_tool_failure_trial_completion,
}


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 8)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 8)


def _summaries(
    observations: list[MetricObservation],
    contracts: list[MetricContract],
) -> list[dict[str, Any]]:
    contract_by_id = {contract.metric_id: contract for contract in contracts}
    grouped: dict[str, list[MetricObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.metric_id].append(observation)
    rows = []
    for metric_id, values in sorted(grouped.items()):
        contract = contract_by_id[metric_id]
        measured = [
            float(item.value)
            for item in values
            if item.status == "measured"
            and isinstance(item.value, (int, float))
            and not isinstance(item.value, bool)
        ]
        rows.append(
            {
                "metric_id": metric_id,
                "name": contract.name,
                "category": contract.category,
                "construct": contract.construct,
                "validation_status": contract.validation_status,
                "unit": values[0].unit,
                "level": values[0].level,
                "observation_count": len(values),
                "status_counts": dict(sorted(Counter(item.status for item in values).items())),
                "measured_mean": round(statistics.fmean(measured), 8) if measured else None,
                "measured_p50": _percentile(measured, 0.5),
                "measured_p95": _percentile(measured, 0.95),
                "evaluator_fingerprint": values[0].evaluator_fingerprint,
            }
        )
    return rows


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, values: list[MetricObservation]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def evaluate_run_metrics(
    run_dir: str | os.PathLike,
    *,
    profile_id: str = "core",
    metric_registry: MetricRegistry | None = None,
    profile_registry: EvaluationProfileRegistry | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate every selected contract for every evidence-bearing Trial."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    metric_registry = metric_registry or MetricRegistry()
    profile_registry = profile_registry or EvaluationProfileRegistry()
    profile = profile_registry.get(profile_id)
    contracts = [
        metric_registry.get(reference.metric_id)
        for reference in profile.metric_refs
    ]
    grouped: dict[TrialKey, list[dict[str, Any]]] = defaultdict(list)
    for event in _events(root):
        key = _trial_key(event)
        if key is not None:
            grouped[key].append(event)

    # Execution Trace may contain a currently active Trial. Result metrics are
    # projections of terminal Trials only; active work remains visible in the
    # trace and will enter this report after trial.completed/trial.failed.
    grouped = defaultdict(
        list,
        {
            key: events
            for key, events in grouped.items()
            if _select(events, "trial.completed", "trial.failed")
        },
    )

    observations: list[MetricObservation] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1] or -1, item[2])):
        events = grouped[key]
        for contract in contracts:
            fingerprint = _fingerprint(contract, profile)
            evaluator_config = contract.evaluator or {}
            entrypoint = str(evaluator_config.get("entrypoint") or "")
            evaluator = _EVALUATORS.get(entrypoint)
            if evaluator is None:
                observations.append(
                    _observation(
                        contract,
                        key,
                        fingerprint,
                        status="unsupported",
                        reason=f"unsupported evaluator entrypoint {entrypoint!r}",
                    )
                )
                continue
            try:
                observations.append(evaluator(contract, key, events, fingerprint))
            except Exception as exc:
                observations.append(
                    _observation(
                        contract,
                        key,
                        fingerprint,
                        status="evaluator_error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )

    summaries = _summaries(observations, contracts)
    statuses = Counter(item.status for item in observations)
    report = {
        "schema_version": METRIC_EVALUATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root),
        "profile": profile.to_dict(),
        "metric_registry_fingerprint": metric_registry.fingerprint,
        "profile_registry_fingerprint": profile_registry.fingerprint,
        "summary": {
            "trial_count": len(grouped),
            "metric_count": len(contracts),
            "observation_count": len(observations),
            "status_counts": dict(sorted(statuses.items())),
            "evaluator_error_count": statuses.get("evaluator_error", 0),
        },
        "metrics": summaries,
        "artifacts": {
            "observations": METRIC_OBSERVATIONS_FILENAME if write else None,
            "evaluation": METRIC_EVALUATION_FILENAME if write else None,
        },
    }
    if write:
        _atomic_write_jsonl(root / METRIC_OBSERVATIONS_FILENAME, observations)
        _atomic_write_json(root / METRIC_EVALUATION_FILENAME, report)
    return report
