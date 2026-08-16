from __future__ import annotations

import json
from pathlib import Path

import pytest
from lychee_mas.eval.evidence import (
    EVIDENCE_COVERAGE_FILENAME,
    EVIDENCE_EVENTS_FILENAME,
    iter_normalized_events,
    normalize_run_evidence,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _runtime_fields(**values):
    return {
        "schema_version": 3,
        "timestamp_unix_s": 1.0,
        "timestamp_utc": "2026-08-16T00:00:01+00:00",
        "case_id": "case-1",
        "sample_index": 0,
        **values,
    }


def test_normalize_run_evidence_materializes_events_and_coverage(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "spans.jsonl",
        [
            _runtime_fields(span_id="run", span_type="run_start"),
            _runtime_fields(span_id="case", span_type="case_start"),
            _runtime_fields(
                span_id="model-start",
                span_type="model_call_start",
                role="Coder",
                model_call_index=1,
            ),
            _runtime_fields(
                span_id="model-end",
                parent_span_id="model-start",
                span_type="model_call_end",
                role="Coder",
                model_call_index=1,
                input_total_positions=120,
                output_total_tokens=30,
                output_reasoning_tokens=20,
                output_answer_tokens=10,
                provider_response_payload={"large": "payload"},
            ),
            _runtime_fields(
                span_id="tool-start",
                span_type="tool_execution_start",
                tool_call_id="code-1",
                source="ComputerTerminal",
            ),
            _runtime_fields(
                span_id="tool-end",
                parent_span_id="tool-start",
                span_type="tool_execution_end",
                tool_call_id="code-1",
                source="ComputerTerminal",
                exit_code=0,
            ),
            _runtime_fields(
                span_id="message-summary",
                span_type="autogen_message",
                source="Coder",
            ),
        ],
    )
    _write_jsonl(
        tmp_path / "group_chat.jsonl",
        [
            _runtime_fields(
                schema_version=1,
                event_id="chat-start",
                event_type="group_chat_start",
                participants=["Coder", "ComputerTerminal"],
                context_visibility="shared",
            ),
            _runtime_fields(
                schema_version=1,
                event_id="chat-message-1",
                event_type="message",
                source="Coder",
                content="Run the code.",
                tool_requests=[
                    {
                        "source": "Coder",
                        "tool_call_id": "call-1",
                        "tool_name": "python",
                        "arguments": "{}",
                    }
                ],
            ),
            _runtime_fields(
                schema_version=1,
                event_id="chat-message-2",
                event_type="message",
                source="ComputerTerminal",
                content="The result is 4.",
                tool_executions=[
                    {
                        "source": "ComputerTerminal",
                        "tool_call_id": "call-1",
                        "tool_name": "python",
                        "is_error": False,
                        "output": "4",
                    }
                ],
            ),
            _runtime_fields(
                schema_version=1,
                event_id="chat-end",
                event_type="group_chat_end",
            ),
        ],
    )
    _write_jsonl(
        tmp_path / "predictions.jsonl",
        [{"case_id": "case-1", "sample_index": 0, "final_answer": "4"}],
    )
    _write_jsonl(
        tmp_path / "outputs.jsonl",
        [
            {
                "case_id": "case-1",
                "sample_index": 0,
                "final_answer": "4",
                "score": 1.0,
            }
        ],
    )

    report = normalize_run_evidence(tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / EVIDENCE_EVENTS_FILENAME).read_text().splitlines()
    ]
    requirements = {row["requirement"]: row for row in report["requirements"]}

    assert report["overall_status"] == "complete"
    assert report["summary"]["capabilities_observed"] == {
        "case_lifecycle": True,
        "model_calls": True,
        "agent_messages": True,
        "tool_executions": True,
        "predictions": True,
        "official_evaluations": True,
    }
    assert [event["event_type"] for event in events].count("agent.message") == 2
    assert any(event["event_type"] == "tool_call.request" for event in events)
    embedded_execution = next(
        event
        for event in events
        if event["event_type"] == "tool_execution.end"
        and event["correlation_id"] == "call-1"
    )
    assert embedded_execution["parent_event_id"].endswith(":tool_request:0")
    assert embedded_execution["content"] == "4"
    model_end = next(event for event in events if event["event_id"] == "spans:model-end")
    assert "provider_response_payload" not in model_end["attributes"]
    assert "provider_response_payload" in model_end["provenance"]["omitted_fields"]
    assert requirements["message.visibility"]["status"] == "complete"
    assert requirements["model_call.reasoning_tokens"]["status"] == "complete"
    assert requirements["evaluation.score"]["status"] == "complete"
    assert (tmp_path / EVIDENCE_COVERAGE_FILENAME).is_file()


def test_invalid_source_record_marks_coverage_partial(tmp_path: Path) -> None:
    (tmp_path / "spans.jsonl").write_text(
        json.dumps(_runtime_fields(span_id="run", span_type="run_start"))
        + "\n{not-json}\n",
        encoding="utf-8",
    )

    report = normalize_run_evidence(tmp_path, write=False)

    assert report["overall_status"] == "partial"
    assert report["summary"]["invalid_source_records"] == 1
    spans = next(item for item in report["sources"] if item["source"] == "spans")
    assert spans["invalid_record_count"] == 1
    assert not (tmp_path / EVIDENCE_EVENTS_FILENAME).exists()


def test_span_message_is_fallback_when_group_chat_is_missing(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "spans.jsonl",
        [
            _runtime_fields(
                span_id="message-summary",
                span_type="autogen_message",
                source="Coder",
                content="fallback content",
            )
        ],
    )

    events = list(iter_normalized_events(tmp_path))

    assert len(events) == 1
    assert events[0]["event_type"] == "agent.message"
    assert events[0]["provenance"]["source_kind"] == "spans"


def test_eval_studio_exposes_materialized_evidence(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lychee_mas.eval.studio.api import create_app

    run_dir = tmp_path / "runs/benchmarks/model/team/task"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")
    _write_jsonl(
        run_dir / "spans.jsonl",
        [_runtime_fields(span_id="case", span_type="case_start")],
    )
    normalize_run_evidence(run_dir)

    client = TestClient(create_app(tmp_path))
    row = client.get("/api/runs").json()[0]
    assert row["has_evidence"] is True
    assert row["evidence_coverage"]["total_events"] == 1

    evidence = client.get(f"/api/runs/{row['id']}/evidence").json()
    coverage = client.get(f"/api/runs/{row['id']}/evidence-coverage").json()
    assert evidence["events"][0]["event_type"] == "case.start"
    assert coverage["summary"]["total_events"] == 1
