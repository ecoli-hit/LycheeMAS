from __future__ import annotations

import json

from lychee_mas.eval.application.read_models.run_events import read_execution_trace
from lychee_mas.eval.evaluation.projections import (
    build_execution_trace,
    build_result_projection,
)
from lychee_mas.runtime.events.store import (
    RUNTIME_WORKSPACES_FILENAME,
    RunEventWriter,
    iter_run_events,
    migrate_legacy_run_events,
    read_unfinished_trial_operations,
    run_event_paths,
    run_events_path,
)


def _write_complete_trial(
    tmp_path,
    *,
    payload_inline_bytes: int | None = None,
    provider_response_payload: dict | None = None,
):
    writer = RunEventWriter(
        run_events_path(tmp_path),
        run_id="run-test",
        **(
            {"payload_inline_bytes": payload_inline_bytes}
            if payload_inline_bytes is not None
            else {}
        ),
    )
    run_id = writer.log_event("run.started", task="demo")
    trial_id = writer.log_event(
        "trial.started",
        parent_event_id=run_id,
        case_id="case-1",
        dataset_index=7,
        trial_index=1,
        task="demo",
        benchmark_id="demo",
        trials_per_case=2,
        trial_seed=123,
    )
    model_id = writer.log_event(
        "model_call.started",
        case_id="case-1",
        dataset_index=7,
        trial_index=1,
        parent_event_id=trial_id,
        role="Solver",
        autogen_model_messages=[{"role": "user", "content": "2+2?"}],
        role_visible_messages=[{"role": "user", "content": "2+2?"}],
        backend_messages=[{"role": "user", "content": "2+2?"}],
    )
    writer.log_event(
        "model_call.completed",
        operation_id=model_id,
        case_id="case-1",
        dataset_index=7,
        trial_index=1,
        parent_event_id=trial_id,
        role="Solver",
        input_total_positions=12,
        input_text_tokens=12,
        output_total_tokens=4,
        output_text_tokens=4,
        output_reasoning_tokens=2,
        output_answer_tokens=2,
        model_latency_s=0.5,
        provider_request_payload={"messages": [{"role": "user", "content": "2+2?"}]},
        provider_response_payload=(
            provider_response_payload
            if provider_response_payload is not None
            else {"choices": [{"message": {"content": "4"}}]}
        ),
        final_content="4",
        delivered_content="4",
    )
    writer.log_event(
        "agent.message.published",
        case_id="case-1",
        dataset_index=7,
        trial_index=1,
        parent_event_id=trial_id,
        source="Solver",
        content="4",
    )
    writer.log_event(
        "model_call.failed",
        case_id="case-1",
        dataset_index=7,
        trial_index=1,
        is_error=True,
    )
    trial_terminal_id = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 7,
            "trial_index": 1,
            "operation_id": trial_id,
            "parent_event_id": run_id,
            "task": "demo",
            "benchmark_id": "demo",
            "trial_seed": 123,
            "trials_per_case": 2,
            "final_output": "4",
        }
    )
    writer.record_evaluation(
        {
            "case_id": "case-1",
            "dataset_index": 7,
            "trial_index": 1,
            "parent_event_id": trial_terminal_id,
            "trial_event_id": trial_terminal_id,
            "scorer_id": "exact",
            "score": 1.0,
            "correct": 1.0,
            "gold": "4",
        }
    )
    writer.log_event("run.completed", operation_id=run_id, num_trials=1)


def test_run_event_envelope_is_stable_and_payload_is_lossless(tmp_path) -> None:
    _write_complete_trial(tmp_path)
    records = [
        json.loads(line)
        for line in (run_events_path(tmp_path)).read_text(encoding="utf-8").splitlines()
    ]
    expected_prefix = [
        "schema_version",
        "run_id",
        "seq",
        "event_id",
        "event_type",
        "timestamp_unix_s",
        "timestamp_utc",
    ]
    assert list(records[0])[: len(expected_prefix)] == expected_prefix
    completed = next(row for row in records if row["event_type"] == "model_call.completed")
    assert completed["payload"]["provider_response_payload"] == {
        "choices": [{"message": {"content": "4"}}]
    }
    assert completed["payload"]["final_content"] == "4"
    assert "event_family" not in completed


def test_large_event_payloads_are_deduplicated_and_transparently_materialized(
    tmp_path,
) -> None:
    large_output = "terminal-output\n" * 10_000
    writer = RunEventWriter(
        run_events_path(tmp_path),
        run_id="run-payload-store",
        payload_inline_bytes=1024,
    )
    writer.log_event(
        "tool_execution.completed",
        source="ComputerTerminal",
        output=large_output,
        delivered_output=large_output[:100],
    )
    writer.log_event(
        "agent.message.published",
        source="ComputerTerminal",
        content=large_output,
    )

    raw_events = list(iter_run_events(tmp_path, materialize_payloads=False))
    output_ref = raw_events[0]["payload"]["output"]["$payload_ref"]
    content_ref = raw_events[1]["payload"]["content"]["$payload_ref"]
    assert output_ref["sha256"] == content_ref["sha256"]
    assert len(list((tmp_path / "payloads" / "sha256").rglob("*.json.gz"))) == 1

    materialized = list(iter_run_events(tmp_path))
    assert materialized[0]["payload"]["output"] == large_output
    assert materialized[1]["payload"]["content"] == large_output


def test_workspace_events_write_a_small_cleanup_manifest(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path), run_id="run-workspace-manifest")
    writer.log_event(
        "workspace.prepared",
        case_id="case-1",
        trial_index=0,
        workspace=str(tmp_path / "workspaces" / "ws_1"),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / RUNTIME_WORKSPACES_FILENAME).read_text().splitlines()
    ]
    assert records == [
        {
            "run_id": "run-workspace-manifest",
            "workspace": str(tmp_path / "workspaces" / "ws_1"),
            "case_id": "case-1",
            "trial_index": 0,
            "attempt": None,
        }
    ]


def test_execution_trace_pairs_operations_and_preserves_point_events(tmp_path) -> None:
    _write_complete_trial(tmp_path)
    trace = build_execution_trace(tmp_path, case_id="case-1", trial_index=1)
    by_type = {node["event_type"]: node for node in trace["nodes"]}

    assert by_type["trial"]["status"] == "completed"
    assert by_type["model_call"]["status"] == "completed"
    assert len(by_type["model_call"]["events"]) == 2
    assert by_type["model_call"]["duration_s"] is not None
    assert by_type["agent.message.published"]["node_type"] == "event"


def test_result_projection_joins_trial_evaluation_and_runtime_usage(tmp_path) -> None:
    _write_complete_trial(tmp_path)
    rows = build_result_projection(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert (row["case_id"], row["dataset_index"], row["trial_index"]) == (
        "case-1",
        7,
        1,
    )
    assert row["prediction"] == "4"
    assert row["score"] == 1.0
    assert row["trial_status"] == "completed"
    assert row["evaluation_status"] == "completed"
    assert row["model_call_count"] == 1
    assert row["message_count"] == 1
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 4
    assert row["output_reasoning_tokens"] == 2
    assert row["output_answer_tokens"] == 2
    assert row["tool_error_count"] == 0


def test_result_projection_does_not_materialize_unused_provider_payload(tmp_path) -> None:
    _write_complete_trial(
        tmp_path,
        payload_inline_bytes=64,
        provider_response_payload={"unused": "x" * 10_000},
    )
    raw = list(iter_run_events(tmp_path, materialize_payloads=False))
    completed = next(
        event for event in raw if event["event_type"] == "model_call.completed"
    )
    extra = completed["payload"]["provider_response_payload"]["$payload_ref"]
    (tmp_path / extra["path"]).unlink()

    rows = build_result_projection(tmp_path)

    assert rows[0]["model_call_count"] == 1
    assert rows[0]["input_tokens"] == 12


def test_trace_marks_unclosed_operation_interrupted_after_run_completion(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    run_id = writer.log_event("run.started")
    writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    writer.log_event("run.completed", operation_id=run_id, num_trials=0)

    trace = build_execution_trace(tmp_path)
    trial = next(node for node in trace["nodes"] if node["event_type"] == "trial")
    assert trial["status"] == "interrupted"


def test_gracefully_stopped_run_is_a_terminal_trace_operation(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    run_id = writer.log_event("run.started", num_cases=3)
    writer.log_event(
        "run.stop_requested",
        parent_event_id=run_id,
        action="stop_after_active_trials",
        active_trials=2,
    )
    writer.log_event(
        "run.stopped",
        operation_id=run_id,
        num_trials=2,
        run_status="stopped",
    )

    trace = build_execution_trace(tmp_path)
    run = next(node for node in trace["nodes"] if node["event_type"] == "run")
    request = next(node for node in trace["nodes"] if node["event_type"] == "run.stop_requested")

    assert run["status"] == "stopped"
    assert request["parent_event_id"] == run_id


def test_execution_trace_preserves_run_trial_attempt_runtime_chat_hierarchy(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    run_id = writer.log_event("run.started", num_cases=1)
    trial_id = writer.log_event(
        "trial.started",
        parent_event_id=run_id,
        case_id="case-1",
        dataset_index=0,
    )
    attempt_id = writer.log_event(
        "attempt.started",
        parent_event_id=trial_id,
        case_id="case-1",
        dataset_index=0,
        attempt=1,
    )
    runtime_id = writer.log_event(
        "runtime.started",
        parent_event_id=attempt_id,
        case_id="case-1",
        dataset_index=0,
    )
    chat_id = writer.log_event(
        "group_chat.started",
        parent_event_id=runtime_id,
        case_id="case-1",
        dataset_index=0,
    )
    writer.log_event(
        "group_chat.completed",
        operation_id=chat_id,
        case_id="case-1",
        dataset_index=0,
    )
    writer.log_event(
        "runtime.completed",
        operation_id=runtime_id,
        case_id="case-1",
        dataset_index=0,
    )
    writer.log_event(
        "attempt.completed",
        operation_id=attempt_id,
        case_id="case-1",
        dataset_index=0,
        attempt=1,
    )
    writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "operation_id": trial_id,
            "parent_event_id": run_id,
            "final_output": "done",
        }
    )
    writer.log_event("run.completed", operation_id=run_id, num_trials=1)

    trace = build_execution_trace(tmp_path)
    by_type = {node["event_type"]: node for node in trace["nodes"]}

    assert by_type["trial"]["parent_node_id"] == by_type["run"]["node_id"]
    assert by_type["attempt"]["parent_node_id"] == by_type["trial"]["node_id"]
    assert by_type["runtime"]["parent_node_id"] == by_type["attempt"]["node_id"]
    assert by_type["group_chat"]["parent_node_id"] == by_type["runtime"]["node_id"]


def test_result_projection_exposes_failed_evaluation_without_losing_prediction(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial_id = writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    terminal_id = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "operation_id": trial_id,
            "final_output": "answer",
        }
    )
    writer.record_evaluation(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_event_id": terminal_id,
            "evaluation_status": "failed",
            "error_type": "ScorerError",
            "error_message": "judge unavailable",
        }
    )

    row = build_result_projection(tmp_path)[0]
    assert row["prediction"] == "answer"
    assert row["evaluation_status"] == "failed"
    assert row["score"] is None


def test_result_projection_preserves_tool_requests_for_native_scorers(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial_id = writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    writer.log_event(
        "tool_call.requested",
        case_id="case-1",
        dataset_index=0,
        trial_index=0,
        tool_call_id="call-1",
        tool_name="analytics_create_plot",
        arguments='{"plot_type":"bar"}',
    )
    writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_index": 0,
            "operation_id": trial_id,
            "final_output": "TASK COMPLETE",
        }
    )

    row = build_result_projection(tmp_path)[0]
    assert row["tool_requests"] == [
        {
            "tool_call_id": "call-1",
            "tool_name": "analytics_create_plot",
            "arguments": '{"plot_type":"bar"}',
        }
    ]


def test_event_log_rotates_by_count_and_reads_as_one_journal(tmp_path) -> None:
    writer = RunEventWriter(
        run_events_path(tmp_path),
        run_id="run-segmented",
        max_events_per_file=2,
        max_bytes_per_file=1024 * 1024,
    )
    for index in range(5):
        writer.log_event("agent.message.published", source="Solver", content=str(index))

    paths = run_event_paths(tmp_path)
    assert [path.name for path in paths] == [
        "run_events.jsonl",
        "run_events.000001.jsonl",
        "run_events.000002.jsonl",
    ]
    assert [event["seq"] for event in iter_run_events(tmp_path)] == [1, 2, 3, 4, 5]

    resumed = RunEventWriter(
        run_events_path(tmp_path),
        append=True,
        max_events_per_file=2,
        max_bytes_per_file=1024 * 1024,
    )
    resumed.log_event("agent.message.published", source="Solver", content="5")
    assert resumed.run_id == "run-segmented"
    assert [event["seq"] for event in iter_run_events(tmp_path)] == [1, 2, 3, 4, 5, 6]


def test_independent_event_writers_share_one_sequence(tmp_path) -> None:
    first = RunEventWriter(run_events_path(tmp_path), run_id="run-shared")
    second = RunEventWriter(run_events_path(tmp_path), append=True)

    first.log_event("agent.message.published", source="Solver", content="one")
    second.log_event(
        "evaluation.completed",
        case_id="case-1",
        score=1.0,
        trial_event_id="trial-1",
    )
    first.log_event("agent.message.published", source="Verifier", content="three")

    events = list(iter_run_events(tmp_path))
    assert [event["seq"] for event in events] == [1, 2, 3]
    assert [event["event_type"] for event in events] == [
        "agent.message.published",
        "evaluation.completed",
        "agent.message.published",
    ]


def test_unfinished_trial_operations_are_closed_once_on_resume(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path), run_id="run-resume")
    run_id = writer.log_event("run.started")
    completed_id = writer.log_event(
        "trial.started",
        parent_event_id=run_id,
        case_id="case-completed",
        dataset_index=0,
        trial_index=0,
    )
    writer.log_event(
        "trial.completed",
        operation_id=completed_id,
        parent_event_id=run_id,
        case_id="case-completed",
        dataset_index=0,
        trial_index=0,
        final_output="done",
    )
    interrupted_id = writer.log_event(
        "trial.started",
        parent_event_id=run_id,
        case_id="case-interrupted",
        dataset_index=1,
        trial_index=0,
    )

    unfinished = read_unfinished_trial_operations(tmp_path)

    assert [event["operation_id"] for event in unfinished] == [interrupted_id]
    resumed = RunEventWriter(run_events_path(tmp_path), append=True)
    resumed_id = resumed.log_event("run.resumed")
    resumed.log_event(
        "trial.interrupted",
        operation_id=interrupted_id,
        parent_event_id=resumed_id,
        case_id="case-interrupted",
        dataset_index=1,
        trial_index=0,
        reason="previous_process_ended_without_trial_terminal",
    )
    assert read_unfinished_trial_operations(tmp_path) == []


def test_event_log_rotates_before_exceeding_byte_limit(tmp_path) -> None:
    writer = RunEventWriter(
        run_events_path(tmp_path),
        max_events_per_file=100,
        max_bytes_per_file=700,
    )
    writer.log_event("agent.message.published", source="Solver", content="a" * 300)
    writer.log_event("agent.message.published", source="Solver", content="b" * 300)

    assert len(run_event_paths(tmp_path)) == 2
    assert len(list(iter_run_events(tmp_path))) == 2


def test_legacy_event_segments_move_to_canonical_directory_without_rewrite(tmp_path) -> None:
    payloads = {
        "run_events.jsonl": b'{"schema_version":2,"run_id":"run-old","seq":1}\n',
        "run_events.000001.jsonl": b'{"schema_version":2,"run_id":"run-old","seq":2}\n',
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    result = migrate_legacy_run_events(tmp_path)

    assert result["migrated"] is True
    assert result["segment_count"] == 2
    assert not list(tmp_path.glob("run_events*.jsonl"))
    assert {
        path.name: path.read_bytes() for path in sorted((tmp_path / "events").glob("*.jsonl"))
    } == payloads


def test_legacy_event_migration_rejects_two_competing_event_logs(tmp_path) -> None:
    (tmp_path / "run_events.jsonl").write_text("legacy\n", encoding="utf-8")
    canonical = run_events_path(tmp_path)
    canonical.parent.mkdir(parents=True)
    canonical.write_text("canonical\n", encoding="utf-8")

    try:
        migrate_legacy_run_events(tmp_path)
    except ValueError as exc:
        assert "both exist" in str(exc)
    else:
        raise AssertionError("competing EventLogs must not be merged automatically")


def test_studio_execution_trace_is_lightweight_and_really_paged(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path), run_id="run-test")
    run_id = writer.log_event("run.started", question="x" * 100_000)
    writer.log_event(
        "run.completed",
        operation_id=run_id,
        final_output="y" * 100_000,
    )

    page = read_execution_trace(tmp_path, start_node=0, limit=1)

    assert page["total_nodes"] == 1
    assert len(page["nodes"]) == 1
    node = page["nodes"][0]
    assert "events" not in node
    assert "children" not in node
    assert [event["event_type"] for event in node["event_refs"]] == [
        "run.started",
        "run.completed",
    ]
    assert node["child_count"] == 0
    assert page["root_node_ids"] == [f"operation:{run_id}"]
