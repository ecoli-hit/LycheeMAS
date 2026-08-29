from __future__ import annotations

import json
from pathlib import Path

import pytest
from lychee_mas.eval.evaluation.studies import StudyRun, aggregate_study
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trial(case: str, seed: int, score: float) -> dict:
    return {
        "benchmark_id": "demo",
        "task": "demo_task",
        "case_id": case,
        "dataset_index": int(case[-1]),
        "trial_seed": seed,
        "score": score,
        "is_correct": score == 1.0,
    }


def _observation(case: str, value: float) -> dict:
    return {
        "schema_version": 1,
        "metric_id": "efficiency.output_tokens",
        "status": "measured",
        "value": value,
        "unit": "tokens",
        "level": "trial",
        "case_id": case,
        "dataset_index": int(case[-1]),
        "trial_index": 0,
        "reason": None,
        "evidence_event_ids": [f"event-{case}"],
        "evaluator_fingerprint": "a" * 64,
        "attributes": {},
    }


def _run(
    root: Path,
    name: str,
    trials: list[dict],
    *,
    expected: int,
    status: str = "complete",
    error_case: tuple[str, int] | None = None,
) -> Path:
    path = root / name
    path.mkdir()
    writer = RunEventWriter(run_events_path(path))
    for record in trials:
        trial = dict(record)
        score = trial.pop("score")
        is_correct = trial.pop("is_correct")
        started_id = writer.log_event(
            "trial.started",
            case_id=record["case_id"],
            dataset_index=record["dataset_index"],
            trial_seed=record["trial_seed"],
            benchmark_id=record["benchmark_id"],
            task=record["task"],
        )
        terminal_id = writer.record_trial(
            {**trial, "operation_id": started_id, "final_output": str(score)}
        )
        writer.record_evaluation(
            {
                "case_id": record["case_id"],
                "dataset_index": record["dataset_index"],
                "trial_event_id": terminal_id,
                "score": score,
                "is_correct": is_correct,
                "correct": score,
            }
        )
    _write_json(
        path / "run_status.json",
        {
            "status": status,
            "task": "demo_task",
            "expected_trials": expected,
            "base_seed": 42,
        },
    )
    _write_json(
        path / "metrics.json",
        {
            "benchmark_id": "demo",
            "task": "demo_task",
            "model": "demo-model",
            "team": "demo-team",
            "method": "none",
        },
    )
    if error_case:
        case_id, trial_seed = error_case
        started_id = writer.log_event(
            "trial.started",
            case_id=case_id,
            dataset_index=int(case_id[-1]),
            trial_index=0,
            trial_seed=trial_seed,
            benchmark_id="demo",
            task="demo_task",
        )
        writer.record_trial(
            {
                "case_id": case_id,
                "dataset_index": int(case_id[-1]),
                "trial_index": 0,
                "trial_seed": trial_seed,
                "benchmark_id": "demo",
                "task": "demo_task",
                "operation_id": started_id,
                "status": "failed",
                "final_output": "",
                "error_type": "RuntimeError",
                "error_message": "failed",
            }
        )
    return path


def test_study_aggregator_pairs_failures_without_dropping_denominator(tmp_path: Path) -> None:
    run_a = _run(
        tmp_path,
        "run-a",
        [_trial("case1", 1, 1.0), _trial("case2", 2, 0.0)],
        expected=3,
        status="complete_with_errors",
        error_case=("case3", 3),
    )
    run_b = _run(
        tmp_path,
        "run-b",
        [
            _trial("case1", 1, 0.0),
            _trial("case2", 2, 0.0),
            _trial("case3", 3, 1.0),
        ],
        expected=3,
    )
    _write_jsonl(run_a / "metric_observations.jsonl", [_observation("case1", 20)])
    _write_jsonl(run_b / "metric_observations.jsonl", [_observation("case1", 30)])
    output = tmp_path / "study"

    report = aggregate_study(
        "paired-demo",
        [StudyRun("system-a", run_a), StudyRun("system-b", run_b)],
        output,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    systems = {row["system_id"]: row for row in report["systems"]}
    pair = report["pairwise"][0]
    assert systems["system-a"]["trial_count"] == 3
    assert systems["system-a"]["score_mean_strict"] == pytest.approx(1 / 3)
    assert systems["system-b"]["score_mean_strict"] == pytest.approx(1 / 3)
    assert pair["paired_count"] == 3
    assert pair["pairing_complete"] is True
    assert (pair["wins_a"], pair["wins_b"], pair["ties"]) == (1, 1, 1)
    assert report["summary"]["incomplete_run_count"] == 1
    assert report["summary"]["num_metric_observations"] == 2
    metric_summaries = {row["system_id"]: row for row in report["metric_summaries"]}
    assert metric_summaries["system-a"]["measured_mean"] == 20
    assert metric_summaries["system-b"]["measured_mean"] == 30
    assert (output / "study_trials.jsonl").is_file()
    assert (output / "study_trials.csv").is_file()
    assert (output / "study_report.json").is_file()
    assert (output / "study_metric_observations.jsonl").is_file()
    assert (output / "study_metric_summary.csv").is_file()


def test_study_aggregator_reports_missing_pairs(tmp_path: Path) -> None:
    run_a = _run(
        tmp_path,
        "run-a",
        [_trial("case1", 1, 1.0), _trial("case2", 2, 0.0)],
        expected=2,
    )
    run_b = _run(
        tmp_path,
        "run-b",
        [_trial("case1", 1, 0.0)],
        expected=2,
    )

    report = aggregate_study(
        "missing-demo",
        [StudyRun("system-a", run_a), StudyRun("system-b", run_b)],
        tmp_path / "study",
        bootstrap_samples=20,
    )

    pair = report["pairwise"][0]
    assert pair["union_pair_count"] == 2
    assert pair["paired_count"] == 1
    assert pair["missing_from_b_count"] == 1
    assert pair["pairing_complete"] is False
    assert report["summary"]["pairing_complete"] is False
