from __future__ import annotations

import json
from pathlib import Path

from lychee_mas.eval.evaluation.events import failed_evaluations, write_evaluation_events
from lychee_mas.eval.experiments.finalization import ExperimentFinalizer
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


class _Registry:
    def __init__(self, instance: dict) -> None:
        self.instance = dict(instance)
        self.update_count = 0

    def get_instance(self, instance_id: str) -> dict:
        assert instance_id == self.instance["id"]
        return dict(self.instance)

    def update_instance(self, instance_id: str, **changes) -> dict:
        assert instance_id == self.instance["id"]
        self.update_count += 1
        self.instance.update(changes)
        return dict(self.instance)


def _terminal_run(tmp_path: Path, *, status: str = "complete") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": status, "completed_distinct_cases": 3}),
        encoding="utf-8",
    )
    return run_dir


def test_finalizer_materializes_evaluation_and_updates_instance(tmp_path: Path) -> None:
    run_dir = _terminal_run(tmp_path)
    registry = _Registry(
        {"id": "experiment-1", "status": "running", "queue": {}, "run_dir": str(run_dir)}
    )
    evaluated: list[Path] = []

    def evaluate(path: Path, *, write: bool) -> dict:
        assert write is True
        evaluated.append(path)
        (path / "metrics.json").write_text("{}", encoding="utf-8")
        return {"metric_evaluation": {"summary": {"observations": 7}}}

    finalizer = ExperimentFinalizer(registry, evaluator=evaluate)
    result = finalizer.finalize(
        "experiment-1",
        {"status": "completed", "return_code": 0, "run_dir": str(run_dir)},
        case_completion_target=3,
        finished_at_utc="2026-08-22T00:00:00Z",
    )

    assert evaluated == [run_dir]
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "completed"
    receipt = json.loads((run_dir / "run_finalization.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "finalized"
    assert receipt["evaluation"]["summary"]["observations"] == 7
    assert finalizer.needs_reconciliation(result) is False


def test_finalizer_preserves_run_terminal_state_when_evaluation_fails(
    tmp_path: Path,
) -> None:
    run_dir = _terminal_run(tmp_path, status="complete_with_errors")
    registry = _Registry(
        {"id": "experiment-2", "status": "running", "queue": {}, "run_dir": str(run_dir)}
    )

    def fail_evaluation(_path: Path, *, write: bool) -> dict:
        raise ValueError("invalid evidence")

    result = ExperimentFinalizer(registry, evaluator=fail_evaluation).finalize(
        "experiment-2",
        {"status": "completed", "return_code": 0, "run_dir": str(run_dir)},
        case_completion_target=None,
        finished_at_utc="2026-08-22T00:00:00Z",
    )

    assert result["status"] == "completed"
    assert result["evaluation_status"] == "failed"
    assert result["evaluation_error"] == "ValueError: invalid evidence"
    receipt = json.loads((run_dir / "run_finalization.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "finalized"
    assert receipt["evaluation"]["status"] == "failed"


def test_finalizer_does_not_hide_failed_benchmark_evaluation(tmp_path: Path) -> None:
    run_dir = _terminal_run(tmp_path, status="paused")
    writer = RunEventWriter(run_events_path(run_dir))
    started = writer.log_event(
        "trial.started", case_id="case-1", dataset_index=0, trial_index=0
    )
    terminal = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_index": 0,
            "operation_id": started,
            "final_output": "answer",
        }
    )
    write_evaluation_events(
        run_dir,
        failed_evaluations(
            [
                {
                    "case_id": "case-1",
                    "dataset_index": 0,
                    "trial_index": 0,
                    "trial_event_id": terminal,
                }
            ],
            ValueError("judge unavailable"),
        ),
    )
    registry = _Registry(
        {"id": "experiment-judge", "status": "running", "queue": {}, "run_dir": str(run_dir)}
    )

    result = ExperimentFinalizer(
        registry,
        evaluator=lambda *_args, **_kwargs: {"metric_evaluation": {"summary": {}}},
    ).finalize(
        "experiment-judge",
        {"status": "completed", "return_code": 0, "run_dir": str(run_dir)},
        case_completion_target=None,
        finished_at_utc="2026-08-22T00:00:00Z",
    )

    assert result["evaluation_status"] == "failed"
    assert "failed=1" in result["evaluation_error"]


def test_incomplete_receipt_is_reconciled_after_restart(tmp_path: Path) -> None:
    run_dir = _terminal_run(tmp_path, status="paused")
    (run_dir / "run_finalization.json").write_text(
        json.dumps({"status": "finalizing"}), encoding="utf-8"
    )
    registry = _Registry(
        {"id": "experiment-3", "status": "paused", "queue": {}, "run_dir": str(run_dir)}
    )
    finalizer = ExperimentFinalizer(
        registry,
        evaluator=lambda *_args, **_kwargs: {"metric_evaluation": None},
    )

    assert finalizer.needs_reconciliation(registry.instance) is True
    finalizer.finalize(
        "experiment-3",
        {"status": "completed", "return_code": 0, "run_dir": str(run_dir)},
        case_completion_target=None,
        finished_at_utc="2026-08-22T00:00:00Z",
    )
    assert finalizer.needs_reconciliation(registry.instance) is False
