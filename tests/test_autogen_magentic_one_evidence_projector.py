from __future__ import annotations

import json
from pathlib import Path

from lychee_mas.eval.evaluation.evaluators import (
    METRIC_OBSERVATIONS_FILENAME,
    evaluate_run_metrics,
)
from lychee_mas.eval.evaluation.evidence import (
    EVIDENCE_EVENTS_FILENAME,
    normalize_run_evidence,
)
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def _ledger(*, stalled: bool, next_speaker: str = "Coder") -> str:
    return json.dumps(
        {
            "is_request_satisfied": {"reason": "work remains", "answer": False},
            "is_progress_being_made": {
                "reason": "progress state",
                "answer": not stalled,
            },
            "is_in_loop": {"reason": "loop state", "answer": stalled},
            "next_speaker": {"reason": "delegate work", "answer": next_speaker},
            "instruction_or_question": {"reason": "next action", "answer": "continue"},
        }
    )


def _model_call(
    writer: RunEventWriter,
    *,
    common: dict[str, object],
    index: int,
    content: str,
) -> str:
    started = writer.log_event(
        "model_call.started",
        role="Orchestrator",
        sender="MagenticOneOrchestrator",
        controller=True,
        model_call_index=index,
        **common,
    )
    return writer.log_event(
        "model_call.completed",
        operation_id=started,
        role="Orchestrator",
        sender="MagenticOneOrchestrator",
        controller=True,
        model_call_index=index,
        final_content=content,
        input_total_positions=10,
        output_total_tokens=2,
        model_latency_s=0.1,
        **common,
    )


def test_autogen_magentic_one_ledger_is_projected_with_official_stall_semantics(
    tmp_path: Path,
) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    common: dict[str, object] = {
        "case_id": "case-1",
        "dataset_index": 0,
        "trial_index": 0,
        "attempt": 1,
    }
    trial_started = writer.log_event("trial.started", **common)
    writer.log_event(
        "group_chat.started",
        group_chat_class="MagenticOneGroupChat",
        group_chat_config={"type": "magentic_one", "max_stalls": 3},
        participants=["Coder", "WebSurfer"],
        member_nodes=["Coder", "WebSurfer"],
        operation_bindings=[
            {"operation": operation, "node_id": "Orchestrator", "options": {}}
            for operation in (
                "plan",
                "monitor_progress",
                "detect_stall",
                "delegate",
                "replan",
                "aggregate",
            )
        ],
        **common,
    )
    _model_call(writer, common=common, index=1, content="initial facts")
    _model_call(writer, common=common, index=2, content="initial plan")

    # Official AutoGen decrements the stall counter by one on progress instead
    # of resetting it. This sequence therefore reaches the threshold on step 5.
    for index, stalled in enumerate((True, True, False, True, True), start=3):
        _model_call(writer, common=common, index=index, content=_ledger(stalled=stalled))
    facts_update = _model_call(writer, common=common, index=8, content="updated facts")
    plan_update = _model_call(writer, common=common, index=9, content="updated plan")
    writer.log_event("group_chat.completed", **common)
    trial_event = writer.record_trial(
        {
            **common,
            "operation_id": trial_started,
            "final_output": "answer",
            "case_wall_time_s": 1.0,
        }
    )
    writer.record_evaluation(
        {
            **common,
            "parent_event_id": trial_event,
            "trial_event_id": trial_event,
            "score": 1.0,
        }
    )

    coverage = normalize_run_evidence(tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / EVIDENCE_EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    operations = [
        event
        for event in events
        if event["event_type"] == "coordination.operation.completed"
    ]
    counts: dict[str, int] = {}
    for event in operations:
        operation = event["attributes"]["operation"]
        counts[operation] = counts.get(operation, 0) + 1

    assert counts == {
        "aggregate": 1,
        "delegate": 4,
        "detect_stall": 5,
        "monitor_progress": 5,
        "plan": 1,
        "replan": 1,
    }
    replan = next(event for event in operations if event["attributes"]["operation"] == "replan")
    assert replan["attributes"]["source_event_ids"][-2:] == [facts_update, plan_update]
    assert replan["provenance"]["source_kind"] == "derived_projection"
    assert replan["provenance"]["projector_id"] == "autogen_magentic_one_ledger"
    assert coverage["projections"][0]["derived_event_count"] == 17

    evaluate_run_metrics(tmp_path)
    observations = [
        json.loads(line)
        for line in (tmp_path / METRIC_OBSERVATIONS_FILENAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in observations}
    assert by_metric["coordination.coordination_operation_count"]["value"] == 17
    assert by_metric["coordination.replan_count"]["value"] == 1
    assert by_metric["coordination.stall_rate"]["value"] == 0.8


def test_invalid_progress_ledger_candidate_is_not_projected(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    common: dict[str, object] = {
        "case_id": "case-1",
        "dataset_index": 0,
        "trial_index": 0,
        "attempt": 1,
    }
    writer.log_event(
        "group_chat.started",
        group_chat_class="MagenticOneGroupChat",
        group_chat_config={"type": "magentic_one", "max_stalls": 3},
        participants=["Coder"],
        operation_bindings=[],
        **common,
    )
    _model_call(writer, common=common, index=1, content="initial facts")
    _model_call(writer, common=common, index=2, content="initial plan")
    _model_call(
        writer,
        common=common,
        index=3,
        content='{"is_request_satisfied": {"answer": false}}',
    )

    report = normalize_run_evidence(tmp_path, write=False)

    assert report["projections"][0]["accepted_progress_ledgers"] == 0
    assert report["projections"][0]["rejected_progress_ledger_candidates"] == 1


def test_cancelled_group_chat_closes_projector_state(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    common: dict[str, object] = {
        "case_id": "case-1",
        "dataset_index": 0,
        "trial_index": 0,
        "attempt": 1,
    }
    writer.log_event(
        "group_chat.started",
        group_chat_class="MagenticOneGroupChat",
        group_chat_config={"type": "magentic_one", "max_stalls": 3},
        participants=["Coder"],
        operation_bindings=[],
        **common,
    )
    writer.log_event("group_chat.cancelled", **common)

    report = normalize_run_evidence(tmp_path, write=False)

    assert report["projections"][0]["eligible_group_chats"] == 1
    assert report["projections"][0]["active_group_chats_at_end"] == 0
