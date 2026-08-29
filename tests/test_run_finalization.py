from __future__ import annotations

import json

from lychee_mas.eval.runner.run_finalization import finalize_run_segment


class _EventWriter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log_event(self, event_type: str, **payload: object) -> None:
        self.events.append((event_type, payload))


def test_completed_run_is_terminal_and_rejects_new_trials(tmp_path) -> None:
    run_status_path = tmp_path / "run_status.json"
    writer = _EventWriter()
    run_status = {
        "successful_trials": 1,
        "failed_trials": 0,
        "accepting_new_trials": True,
    }

    settlement = finalize_run_segment(
        run_status=run_status,
        run_status_path=run_status_path,
        run_control_path=tmp_path / "run_control.json",
        event_writer=writer,
        operation_id="run-operation",
        trial_records=[],
        completed_trials={},
        expected_cases=1,
        trials_per_case=1,
        segment_truncated=False,
        graceful_stop_requested=False,
        previous_elapsed_s=0.0,
        segment_started_at_unix_s=0.0,
        vllm_metrics_summary=None,
        out_dir=tmp_path,
    )

    persisted = json.loads(run_status_path.read_text(encoding="utf-8"))
    assert settlement["status"] == "complete"
    assert persisted["status"] == "complete"
    assert persisted["accepting_new_trials"] is False
    assert writer.events[0][0] == "run.completed"
