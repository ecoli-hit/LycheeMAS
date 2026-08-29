from __future__ import annotations

import json
from pathlib import Path

import pytest
from lychee_mas.eval.evaluation.evidence import (
    EVIDENCE_COVERAGE_FILENAME,
    EVIDENCE_EVENTS_FILENAME,
    iter_normalized_events,
    normalize_run_evidence,
)
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def test_normalize_run_evidence_materializes_events_and_coverage(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    common = {"case_id": "case-1", "dataset_index": 0, "trial_index": 0}
    writer.log_event("run.started", event_id="run")
    writer.log_event(
        "trial.admission_updated",
        event_id="admission",
        admission_total=1,
        consumed_total=0,
    )
    trial_started_id = writer.log_event("trial.started", event_id="trial", **common)
    model_started_id = writer.log_event(
        "model_call.started", event_id="model-start", role="Coder", model_call_index=1, **common
    )
    writer.log_event(
        "model_call.completed",
        event_id="model-end",
        operation_id=model_started_id,
        role="Coder",
        model_call_index=1,
        controller=False,
        input_total_positions=120,
        output_total_tokens=30,
        output_reasoning_tokens=20,
        output_answer_tokens=10,
        model_latency_s=1.25,
        provider_response_payload={"large": "payload"},
        **common,
    )
    request_id = writer.log_event(
        "tool_call.requested",
        event_id="tool-request",
        correlation_id="call-1",
        source="Coder",
        arguments="{}",
        **common,
    )
    writer.log_event(
        "tool_execution.observed",
        event_id="tool-end",
        parent_event_id=request_id,
        correlation_id="call-1",
        source="ComputerTerminal",
        output="4",
        status="completed",
        **common,
    )
    group_chat_id = writer.log_event(
        "group_chat.started",
        event_id="chat-start",
        member_nodes=["Coder", "ComputerTerminal"],
        operation_bindings=[],
        context_visibility="shared",
        **common,
    )
    writer.log_event(
        "agent.message.published",
        event_id="chat-message-1",
        source="Coder",
        content="Run the code.",
        **common,
    )
    writer.log_event(
        "agent.message.published",
        event_id="chat-message-2",
        source="ComputerTerminal",
        content="The result is 4.",
        **common,
    )
    writer.log_event(
        "group_chat.completed", event_id="chat-end", operation_id=group_chat_id, **common
    )
    writer.log_event(
        "result_contract.validated",
        event_id="result-contract",
        kind="text",
        source="Coder",
        collector="approved_submitter",
        **common,
    )
    trial_event_id = writer.record_trial(
        {
            **common,
            "operation_id": trial_started_id,
            "final_output": "4",
            "case_wall_time_s": 1.5,
        }
    )
    writer.record_evaluation({**common, "trial_event_id": trial_event_id, "score": 1.0})

    report = normalize_run_evidence(tmp_path)
    events = [
        json.loads(line) for line in (tmp_path / EVIDENCE_EVENTS_FILENAME).read_text().splitlines()
    ]
    requirements = {row["requirement"]: row for row in report["requirements"]}

    assert report["overall_status"] == "complete"
    assert report["summary"]["capabilities_observed"] == {
        "trial_lifecycle": True,
        "model_calls": True,
        "agent_messages": True,
        "tool_executions": True,
        "trials": True,
        "official_evaluations": True,
    }
    assert [event["event_type"] for event in events].count("agent.message.published") == 2
    assert any(event["event_type"] == "tool_call.requested" for event in events)
    embedded_execution = next(
        event
        for event in events
        if event["event_type"] == "tool_execution.observed" and event["correlation_id"] == "call-1"
    )
    assert embedded_execution["parent_event_id"] == "tool-request"
    assert embedded_execution["content"] == "4"
    model_end = next(event for event in events if event["event_id"] == "model-end")
    assert "provider_response_payload" not in model_end["attributes"]
    assert "provider_response_payload" in model_end["provenance"]["omitted_fields"]
    assert requirements["message.visibility"]["status"] == "complete"
    assert requirements["case.identity"]["status"] == "complete"
    assert requirements["model_call.reasoning_tokens"]["status"] == "complete"
    assert requirements["result_contract.validation"]["status"] == "complete"
    assert requirements["result_contract.source"]["status"] == "complete"
    assert requirements["result_contract.collector"]["status"] == "complete"
    assert requirements["evaluation.score"]["status"] == "complete"
    assert (tmp_path / EVIDENCE_COVERAGE_FILENAME).is_file()


def test_invalid_source_record_marks_coverage_partial(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    writer.log_event("run.started", event_id="run")
    with (run_events_path(tmp_path)).open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    report = normalize_run_evidence(tmp_path, write=False)

    assert report["overall_status"] == "partial"
    assert report["summary"]["invalid_source_records"] == 1
    source = next(item for item in report["sources"] if item["source"] == "run_events")
    assert source["invalid_record_count"] == 1
    assert not (tmp_path / EVIDENCE_EVENTS_FILENAME).exists()


def test_coordination_actor_requirement_only_applies_to_completed_operations(
    tmp_path: Path,
) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    writer.log_event(
        "coordination.protocol_violation",
        event_id="violation",
        case_id="case-1",
        dataset_index=0,
        trial_index=0,
        protocol="autogen_magentic_one_progress_ledger",
        violation="invalid_next_speaker",
    )
    writer.log_event(
        "coordination.operation.completed",
        event_id="operation",
        case_id="case-1",
        dataset_index=0,
        trial_index=0,
        actor="Orchestrator",
        operation="delegate",
    )

    report = normalize_run_evidence(tmp_path, write=False)
    requirements = {row["requirement"]: row for row in report["requirements"]}

    assert requirements["coordination.actor"]["status"] == "complete"
    assert requirements["coordination.actor"]["eligible_events"] == 1


def test_group_chat_message_is_normalized_from_canonical_journal(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    writer.log_event(
        "agent.message.published",
        event_id="message",
        source="Coder",
        content="canonical content",
        case_id="case-1",
        dataset_index=0,
    )

    events = list(iter_normalized_events(tmp_path))

    assert len(events) == 1
    assert events[0]["event_type"] == "agent.message.published"
    assert events[0]["content"] == "canonical content"
    assert events[0]["provenance"]["source_kind"] == "run_event"


def test_empty_delivered_message_and_budget_control_event_are_not_missing_evidence(
    tmp_path: Path,
) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    writer.log_event(
        "agent.message.published",
        source="Verifier",
        content="",
        case_id="case-1",
    )
    writer.log_event(
        "model_call.budget_stopped",
        model_calls_started=20,
        max_model_calls_per_case=20,
        case_id="case-1",
    )

    report = normalize_run_evidence(tmp_path, write=False)
    requirements = {row["requirement"]: row for row in report["requirements"]}

    assert requirements["message.content"]["status"] == "complete"
    assert requirements["model_call.role"]["status"] == "not_observed"
    assert report["summary"]["required_failure_count"] == 0


def test_evidence_normalization_reads_all_event_log_segments(tmp_path: Path) -> None:
    writer = RunEventWriter(
        run_events_path(tmp_path),
        max_events_per_file=1,
        max_bytes_per_file=1024 * 1024,
    )
    writer.log_event(
        "agent.message.published",
        source="Coder",
        content="first",
        case_id="case-1",
    )
    writer.log_event(
        "agent.message.published",
        source="Coder",
        content="second",
        case_id="case-1",
    )

    report = normalize_run_evidence(tmp_path, write=False)
    events = list(iter_normalized_events(tmp_path))
    source = next(item for item in report["sources"] if item["source"] == "run_events")
    assert [event["content"] for event in events] == ["first", "second"]
    assert source["segment_count"] == 2
    assert source["artifacts"] == [
        "events/run_events.jsonl",
        "events/run_events.000001.jsonl",
    ]


def test_eval_studio_exposes_materialized_evidence(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "runs/benchmarks/model/team/task"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")
    writer = RunEventWriter(run_events_path(run_dir))
    writer.log_event("trial.started", event_id="trial", case_id="case-1", dataset_index=0)
    normalize_run_evidence(run_dir)

    client = TestClient(create_app(tmp_path))
    row = client.get("/api/runs").json()[0]
    assert row["has_evidence"] is True
    assert row["evidence_coverage"]["total_events"] == 1

    evidence = client.get(f"/api/runs/{row['id']}/evidence").json()
    coverage = client.get(f"/api/runs/{row['id']}/evidence-coverage").json()
    assert evidence["events"][0]["event_type"] == "trial.started"
    assert coverage["summary"]["total_events"] == 1
