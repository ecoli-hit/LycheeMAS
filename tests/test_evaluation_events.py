from __future__ import annotations

from lychee_mas.eval.evaluation.events import write_evaluation_events
from lychee_mas.eval.evaluation.integrity import audit_run_event_integrity
from lychee_mas.eval.evaluation.projections import build_result_projection
from lychee_mas.runtime.events.store import RunEventWriter, iter_run_events, run_events_path


def _completed_trial(tmp_path) -> dict:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial_id = writer.log_event(
        "trial.started",
        case_id="case-1",
        dataset_index=0,
        trial_index=0,
    )
    terminal_id = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_index": 0,
            "operation_id": trial_id,
            "final_output": "4",
        }
    )
    return {
        "case_id": "case-1",
        "dataset_index": 0,
        "trial_index": 0,
        "trial_event_id": terminal_id,
        "task": "demo",
        "benchmark_id": "demo",
        "scorer_kind": "exact",
        "gold": "4",
        "score": 1.0,
        "score_details": {"score": 1.0},
        "is_correct": True,
        "correct": 1.0,
    }


def test_unchanged_evaluation_is_not_appended_twice(tmp_path) -> None:
    trial = _completed_trial(tmp_path)

    assert write_evaluation_events(tmp_path, [trial]) == 1
    assert write_evaluation_events(tmp_path, [trial]) == 0
    events = list(
        iter_run_events(
            tmp_path,
            event_types=("evaluation.completed", "evaluation.failed"),
        )
    )
    assert len(events) == 1


def test_changed_evaluation_is_appended_and_projection_uses_latest(tmp_path) -> None:
    trial = _completed_trial(tmp_path)
    write_evaluation_events(tmp_path, [trial])

    changed = {
        **trial,
        "score": 0.0,
        "score_details": {"score": 0.0},
        "is_correct": False,
        "correct": 0.0,
    }
    assert write_evaluation_events(tmp_path, [changed]) == 1
    assert build_result_projection(tmp_path)[0]["score"] == 0.0


def test_run_event_integrity_detects_cross_trial_operation_scope(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial_a = writer.log_event(
        "trial.started", case_id="case-a", dataset_index=0, trial_index=0, worker_id=0
    )
    model_call = writer.log_event(
        "model_call.started", case_id="case-a", trial_index=0, worker_id=0
    )
    writer.record_trial(
        {
            "case_id": "case-a",
            "dataset_index": 0,
            "trial_index": 0,
            "worker_id": 0,
            "operation_id": trial_a,
            "status": "failed",
        }
    )
    writer.log_event(
        "model_call.completed",
        operation_id=model_call,
        case_id="case-b",
        trial_index=0,
        worker_id=0,
    )

    report = audit_run_event_integrity(tmp_path)

    assert report["status"] == "invalid"
    assert report["issue_counts"]["operation_scope_mismatches"] == 1
    assert report["issue_counts"]["late_operation_terminals"] == 1


def test_run_event_integrity_rejects_a_missing_event_log(tmp_path) -> None:
    report = audit_run_event_integrity(tmp_path / "missing-run")

    assert report["status"] == "invalid"
    assert report["total_events"] == 0
    assert report["issue_counts"]["missing_event_log"] == 1


def test_run_event_integrity_accepts_scoring_after_trial_terminal(tmp_path) -> None:
    trial = _completed_trial(tmp_path)
    write_evaluation_events(tmp_path, [trial])

    report = audit_run_event_integrity(tmp_path)

    assert report["status"] == "valid"
    assert not any(report["issue_counts"].values())


def test_run_event_integrity_treats_runtime_cancellation_as_terminal(tmp_path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    trial = writer.log_event("trial.started", case_id="case-cancelled", trial_index=0)
    runtime = writer.log_event(
        "runtime.started", case_id="case-cancelled", trial_index=0
    )
    group_chat = writer.log_event(
        "group_chat.started", case_id="case-cancelled", trial_index=0
    )
    writer.log_event(
        "group_chat.cancelled",
        operation_id=group_chat,
        case_id="case-cancelled",
        trial_index=0,
    )
    writer.log_event(
        "runtime.cancelled",
        operation_id=runtime,
        case_id="case-cancelled",
        trial_index=0,
    )
    writer.record_trial(
        {
            "case_id": "case-cancelled",
            "trial_index": 0,
            "operation_id": trial,
            "status": "failed",
        }
    )

    report = audit_run_event_integrity(tmp_path)

    assert report["status"] == "valid"
    assert not any(report["issue_counts"].values())
