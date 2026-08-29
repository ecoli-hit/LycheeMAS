from __future__ import annotations

import json
from pathlib import Path

import pytest
from lychee_mas.eval.evaluation.evaluators import (
    METRIC_EVALUATION_FILENAME,
    METRIC_OBSERVATIONS_FILENAME,
    EvaluationProfile,
    EvaluationProfileRegistry,
    evaluate_run_metrics,
    materialize_run_evaluation,
)
from lychee_mas.eval.evaluation.evidence import normalize_run_evidence
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def _append_events(run_dir: Path, kind: str, rows: list[dict]) -> None:
    writer = RunEventWriter(run_events_path(run_dir), append=(run_events_path(run_dir)).is_file())
    for row in rows:
        value = dict(row)
        event_id = value.pop("event_id", None)
        event_type = value.pop("event_type", None)
        if kind == "runtime":
            if event_type == "model_call.completed":
                operation_id = writer.log_event(
                    "model_call.started",
                    case_id=value.get("case_id"),
                    dataset_index=value.get("dataset_index"),
                    trial_index=value.get("trial_index", 0),
                    role=value.get("role"),
                )
                value["operation_id"] = operation_id
            writer.log_event(event_type, event_id=event_id, **value)
        elif kind == "group_chat":
            requests = list(value.get("tool_requests") or [])
            executions = list(value.get("tool_executions") or [])
            writer.log_event(event_type, event_id=event_id, **value)
            for request in requests:
                writer.log_event(
                    "tool_call.requested",
                    case_id=value.get("case_id"),
                    dataset_index=value.get("dataset_index"),
                    trial_index=value.get("trial_index", 0),
                    correlation_id=request.get("tool_call_id"),
                    **request,
                )
            for execution in executions:
                writer.log_event(
                    "tool_execution.observed",
                    case_id=value.get("case_id"),
                    dataset_index=value.get("dataset_index"),
                    trial_index=value.get("trial_index", 0),
                    correlation_id=execution.get("tool_call_id"),
                    status="error" if execution.get("is_error") else "completed",
                    **execution,
                )
        elif kind == "evaluation":
            trial_started_id = writer.log_event(
                "trial.started",
                case_id=value.get("case_id"),
                dataset_index=value.get("dataset_index"),
                trial_index=value.get("trial_index", 0),
            )
            writer.log_event(
                "attempt.started",
                case_id=value.get("case_id"),
                dataset_index=value.get("dataset_index"),
                trial_index=value.get("trial_index", 0),
                attempt=1,
                parent_event_id=trial_started_id,
            )
            trial_event_id = writer.record_trial(
                {
                    **value,
                    "operation_id": trial_started_id,
                    "final_output": value.get("prediction", ""),
                }
            )
            writer.record_evaluation({**value, "trial_event_id": trial_event_id})


def test_core_profile_evaluates_current_metrics_per_trial(tmp_path: Path) -> None:
    _append_events(
        tmp_path,
        "runtime",
        [
            {
                "event_id": "model-1",
                "event_type": "model_call.completed",
                "timestamp_unix_s": 1.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "role": "Analyst",
                "controller": True,
                "input_total_positions": 10,
                "output_total_tokens": 2,
                "model_latency_s": 0.25,
                "provider_request_queue_latency_s": 0.01,
                "provider_scheduled_to_first_token_s": 0.1,
                "provider_generation_latency_s": 0.2,
            },
            {
                "event_id": "model-2",
                "event_type": "model_call.completed",
                "timestamp_unix_s": 2.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "role": "Verifier",
                "controller": False,
                "input_total_positions": 20,
                "output_total_tokens": 3,
                "model_latency_s": 0.5,
                "provider_request_queue_latency_s": 0.02,
                "provider_scheduled_to_first_token_s": 0.2,
                "provider_generation_latency_s": 0.3,
            },
            {
                "event_id": "operation-plan",
                "event_type": "coordination.operation.completed",
                "timestamp_unix_s": 0.75,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "actor": "Analyst",
                "operation": "plan",
            },
            {
                "event_id": "operation-stall",
                "event_type": "coordination.operation.completed",
                "timestamp_unix_s": 1.75,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "actor": "Analyst",
                "operation": "detect_stall",
                "stalled": True,
            },
            {
                "event_id": "operation-replan",
                "event_type": "coordination.operation.completed",
                "timestamp_unix_s": 1.8,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "actor": "Analyst",
                "operation": "replan",
            },
        ],
    )
    _append_events(
        tmp_path,
        "group_chat",
        [
            {
                "event_id": "chat-start",
                "event_type": "group_chat.started",
                "timestamp_unix_s": 0.5,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "participants": ["Analyst", "Verifier"],
                "context_visibility": "shared",
            },
            {
                "event_id": "message-user",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 1.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "source": "user",
                "content": "Solve the task",
            },
            {
                "event_id": "message-1",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 1.5,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "source": "Analyst",
                "content": "Same answer",
                "tool_executions": [
                    {
                        "tool_call_id": "tool-1",
                        "source": "Analyst",
                        "output": "ok",
                        "duration_s": 0.01,
                    },
                    {
                        "tool_call_id": "tool-2",
                        "source": "Analyst",
                        "output": "failed",
                        "is_error": True,
                        "duration_s": 0.02,
                    },
                ],
            },
            {
                "event_id": "message-2",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 2.5,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "source": "Verifier",
                "content": "same  answer",
            },
            {
                "event_id": "tool-message",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 2.6,
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "source": "Analyst",
                "content": "raw tool result",
                "is_tool_event": True,
            },
        ],
    )
    _append_events(
        tmp_path,
        "evaluation",
        [{"case_id": "case-1", "dataset_index": 0, "trial_index": 0, "score": 1.0}],
    )
    normalize_run_evidence(tmp_path)

    report = evaluate_run_metrics(tmp_path)
    observations = [
        json.loads(line)
        for line in (tmp_path / METRIC_OBSERVATIONS_FILENAME).read_text().splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in observations}

    assert report["summary"]["trial_count"] == 1
    assert report["summary"]["observation_count"] == 28
    assert report["summary"]["status_counts"] == {
        "measured": 26,
        "missing_evidence": 2,
    }
    assert by_metric["task.official_score"]["value"] == 1.0
    assert by_metric["coordination.message_repetition_rate"]["value"] == 0.5
    assert by_metric["coordination.role_participation_balance"]["value"] == 1.0
    assert by_metric["coordination.model_call_participation_balance"]["value"] == 1.0
    assert by_metric["coordination.controller_model_call_ratio"]["value"] == 0.5
    assert by_metric["coordination.controller_output_token_ratio"]["value"] == 0.4
    assert by_metric["coordination.coordination_operation_count"]["value"] == 3
    assert by_metric["coordination.replan_count"]["value"] == 1
    assert by_metric["coordination.stall_rate"]["value"] == 1.0
    assert by_metric["coordination.role_switch_rate"]["value"] == 1.0
    assert by_metric["coordination.active_role_count"]["value"] == 2
    assert by_metric["coordination.agent_message_count"]["value"] == 2
    assert by_metric["efficiency.input_tokens"]["value"] == 30
    assert by_metric["efficiency.output_tokens"]["value"] == 5
    assert by_metric["efficiency.model_call_latency"]["value"] == 0.75
    assert by_metric["efficiency.provider_queue_time"]["value"] == 0.03
    assert by_metric["efficiency.time_to_first_token"]["value"] == 0.3
    assert by_metric["efficiency.generation_throughput"]["value"] == 10.0
    assert by_metric["efficiency.model_call_count"]["value"] == 2
    assert by_metric["efficiency.tool_execution_count"]["value"] == 2
    assert by_metric["efficiency.tool_execution_latency"]["value"] == 0.03
    assert by_metric["efficiency.trial_wall_time"]["status"] == "missing_evidence"
    assert by_metric["reliability.tool_error_rate"]["value"] == 0.5
    assert by_metric["reliability.model_call_error_rate"]["value"] == 0.0
    assert by_metric["reliability.trial_retry_count"]["value"] == 0
    assert by_metric["reliability.trial_runtime_success"]["value"] == 1.0
    assert by_metric["reliability.post_tool_failure_trial_completion"]["value"] == 1.0
    assert all(len(row["evaluator_fingerprint"]) == 64 for row in observations)
    assert (tmp_path / METRIC_EVALUATION_FILENAME).is_file()


def test_profile_registry_contains_current_profile_and_validates_shape() -> None:
    registry = EvaluationProfileRegistry()
    profile = registry.get("core")

    assert len(profile.metric_refs) == 28
    assert len(profile.fingerprint) == 64
    assert "version" not in profile.to_dict()
    assert all("version" not in item for item in profile.to_dict()["metric_refs"])
    payload = profile.to_dict(include_fingerprint=False)
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        EvaluationProfile.from_mapping(payload)


def test_active_trial_is_not_projected_as_a_metric_result(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial_id = writer.log_event(
        "trial.started",
        case_id="case-active",
        dataset_index=0,
        trial_index=0,
    )
    model_id = writer.log_event(
        "model_call.started",
        case_id="case-active",
        dataset_index=0,
        trial_index=0,
        parent_event_id=trial_id,
        role="Solver",
    )
    writer.log_event(
        "model_call.completed",
        operation_id=model_id,
        case_id="case-active",
        dataset_index=0,
        trial_index=0,
        role="Solver",
        input_total_positions=10,
        output_total_tokens=2,
        model_latency_s=0.25,
    )

    report = evaluate_run_metrics(tmp_path)

    assert report["summary"]["trial_count"] == 0
    assert report["summary"]["observation_count"] == 0
    assert (tmp_path / METRIC_OBSERVATIONS_FILENAME).read_text() == ""


def test_message_repetition_projects_multimodal_text_blocks(tmp_path: Path) -> None:
    _append_events(
        tmp_path,
        "group_chat",
        [
            {
                "event_id": "message-1",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 1.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "source": "WebSurfer",
                "content": ["Same answer", {"image": {"url": "screenshot.png"}}],
            },
            {
                "event_id": "message-2",
                "event_type": "agent.message.published",
                "timestamp_unix_s": 2.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "source": "Verifier",
                "content": {"type": "text", "text": "same  answer"},
            },
        ],
    )
    _append_events(
        tmp_path,
        "evaluation",
        [{"case_id": "case-1", "dataset_index": 0, "score": 1.0}],
    )
    normalize_run_evidence(tmp_path)

    evaluate_run_metrics(tmp_path)
    observations = [
        json.loads(line)
        for line in (tmp_path / METRIC_OBSERVATIONS_FILENAME).read_text().splitlines()
    ]
    repetition = next(
        row for row in observations if row["metric_id"] == "coordination.message_repetition_rate"
    )

    assert repetition["status"] == "measured"
    assert repetition["value"] == 0.5
    assert repetition["attributes"]["non_text_message_count"] == 0


def test_materialize_run_evaluation_backfills_existing_metrics(tmp_path: Path) -> None:
    _append_events(
        tmp_path,
        "runtime",
        [
            {
                "event_id": "model-1",
                "event_type": "model_call.completed",
                "timestamp_unix_s": 1.0,
                "case_id": "case-1",
                "dataset_index": 0,
                "role": "Solver",
                "input_total_positions": 10,
                "output_total_tokens": 2,
                "model_latency_s": 0.25,
            }
        ],
    )
    _append_events(
        tmp_path,
        "evaluation",
        [{"case_id": "case-1", "dataset_index": 0, "score": 1.0}],
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"score_mean": 1.0}), encoding="utf-8")

    report = materialize_run_evaluation(tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert report["metric_evaluation"]["summary"]["observation_count"] == 28
    assert metrics["score_mean"] == 1.0
    assert metrics["metric_evaluation"]["metric_count"] == 28
    assert (tmp_path / "evidence.jsonl").is_file()
    assert (tmp_path / METRIC_OBSERVATIONS_FILENAME).is_file()
