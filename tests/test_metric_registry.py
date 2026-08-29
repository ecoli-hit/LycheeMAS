from __future__ import annotations

from pathlib import Path

import pytest
from lychee_mas.eval.evaluation.evaluators import evaluate_run_metrics
from lychee_mas.eval.evaluation.evidence import normalize_run_evidence
from lychee_mas.eval.evaluation.metrics_registry import (
    METRIC_APPLICABILITY_FILENAME,
    MetricContract,
    MetricObservation,
    MetricRegistry,
    assess_metric_applicability,
    evaluate_run_metric_applicability,
)
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def _coverage(
    *,
    capabilities: dict[str, bool] | None = None,
    requirements: dict[str, str] | None = None,
    total_events: int = 8,
) -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-08-16T00:00:00+00:00",
        "summary": {
            "total_events": total_events,
            "capabilities_observed": capabilities
            or {
                "case_lifecycle": True,
                "model_calls": True,
                "agent_messages": True,
                "tool_executions": False,
                "predictions": True,
                "official_evaluations": True,
            },
        },
        "requirements": [
            {
                "requirement": key,
                "status": status,
                "coverage_ratio": 1.0 if status == "complete" else 0.0,
                "eligible_events": 1,
                "present_events": 1 if status == "complete" else 0,
            }
            for key, status in (
                requirements
                or {
                    "evaluation.score": "complete",
                    "prediction.final_answer": "complete",
                    "message.actor": "complete",
                    "message.content": "complete",
                    "message.visibility": "complete",
                    "model_call.role": "complete",
                    "model_call.input_tokens": "complete",
                    "model_call.output_tokens": "complete",
                    "model_call.latency": "complete",
                    "model_call.reasoning_tokens": "complete",
                    "model_call.answer_tokens": "complete",
                    "tool_execution.correlation_id": "not_observed",
                    "tool_execution.parent_link": "not_observed",
                    "tool_execution.status": "not_observed",
                }
            ).items()
        ],
    }


def test_builtin_metric_registry_contains_one_current_contract_per_metric() -> None:
    registry = MetricRegistry()
    descriptor = registry.descriptor()

    assert descriptor["num_contracts"] == 28
    assert descriptor["categories"] == ["coordination", "efficiency", "reliability", "task"]
    assert len(descriptor["registry_fingerprint"]) == 64
    contract = registry.get("task.official_score")
    assert contract.validation_status == "validated"
    assert len(contract.fingerprint) == 64
    assert "version" not in contract.to_dict()
    assert (
        registry.get("coordination.message_repetition_rate").description
        == "Surface-level repetition across the textual projection of delivered agent "
        "messages, including multimodal content blocks."
    )


def test_metric_contract_rejects_unknown_fields() -> None:
    payload = MetricRegistry().get("task.official_score").to_dict(include_fingerprint=False)
    payload["surprise"] = True

    with pytest.raises(ValueError, match="unsupported fields"):
        MetricContract.from_mapping(payload)


def test_metric_observation_distinguishes_values_from_na_states() -> None:
    measured = MetricObservation(
        metric_id="task.official_score",
        status="measured",
        value=1.0,
        unit="benchmark_score",
        level="case",
        evidence_event_ids=("outputs:case-1:0:1",),
        evaluator_fingerprint="a" * 64,
    )
    missing = MetricObservation(
        metric_id="coordination.message_repetition_rate",
        status="missing_evidence",
        unit="ratio",
        level="case",
        reason="message.content is missing",
    )

    assert measured.to_dict()["value"] == 1.0
    assert "metric_version" not in measured.to_dict()
    assert missing.to_dict()["value"] is None
    with pytest.raises(ValueError, match="must not contain a value"):
        MetricObservation(
            metric_id="coordination.message_repetition_rate",
            status="not_applicable",
            value=0.0,
            unit="ratio",
            level="case",
            reason="no message channel",
        )


def test_applicability_separates_measurable_missing_and_not_applicable() -> None:
    coverage = _coverage()
    report = assess_metric_applicability(coverage)
    by_id = {item["metric_id"]: item for item in report["assessments"]}

    assert by_id["task.official_score"]["status"] == "measurable"
    assert by_id["efficiency.input_tokens"]["status"] == "measurable"
    assert by_id["reliability.tool_error_rate"]["status"] == "not_applicable"

    coverage["requirements"] = [
        {
            **row,
            "status": "missing",
            "coverage_ratio": 0.0,
            "present_events": 0,
        }
        if row["requirement"] == "message.content"
        else row
        for row in coverage["requirements"]
    ]
    report = assess_metric_applicability(coverage)
    by_id = {item["metric_id"]: item for item in report["assessments"]}
    assert by_id["coordination.message_repetition_rate"]["status"] == "missing_evidence"


def test_run_applicability_is_materialized_from_real_evidence(tmp_path: Path) -> None:
    writer = RunEventWriter(run_events_path(tmp_path))
    call_id = writer.log_event(
        "model_call.started", case_id="case-1", dataset_index=0, role="Solver"
    )
    writer.log_event(
        "model_call.completed",
        operation_id=call_id,
        case_id="case-1",
        dataset_index=0,
        role="Solver",
        input_total_positions=10,
        output_total_tokens=2,
        model_latency_s=0.5,
    )
    trial_id = writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    terminal_id = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "operation_id": trial_id,
            "final_output": "4",
        }
    )
    writer.record_evaluation(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_event_id": terminal_id,
            "score": 1.0,
        }
    )
    coverage = normalize_run_evidence(tmp_path)

    report = evaluate_run_metric_applicability(
        tmp_path,
        coverage_report=coverage,
        write=True,
    )

    assert report["summary"]["total_contracts"] == 28
    assert (tmp_path / METRIC_APPLICABILITY_FILENAME).is_file()


def test_eval_studio_exposes_metric_contracts_and_run_gate(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from apps.eval.server.app import create_app
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "runs/benchmarks/model/team/task"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")
    writer = RunEventWriter(run_events_path(run_dir))
    trial_id = writer.log_event("trial.started", case_id="case-1", dataset_index=0)
    terminal_id = writer.record_trial(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "operation_id": trial_id,
            "final_output": "4",
        }
    )
    writer.record_evaluation(
        {
            "case_id": "case-1",
            "dataset_index": 0,
            "trial_event_id": terminal_id,
            "score": 1.0,
        }
    )
    coverage = normalize_run_evidence(run_dir)
    evaluate_run_metric_applicability(run_dir, coverage_report=coverage)
    evaluate_run_metrics(run_dir)

    client = TestClient(create_app(tmp_path))
    contracts = client.get("/api/metric-contracts").json()
    row = client.get("/api/runs").json()[0]
    gate = client.get(f"/api/runs/{row['id']}/metric-applicability").json()
    observations = client.get(f"/api/runs/{row['id']}/metric-observations").json()
    trials = client.get(f"/api/runs/{row['id']}/metric-trials").json()
    evaluation = client.get(f"/api/runs/{row['id']}/metric-evaluation").json()
    analyzed = client.post(f"/api/runs/{row['id']}/analyze").json()
    profiles = client.get("/api/evaluation-profiles").json()

    assert contracts["num_contracts"] == 28
    assert row["has_metric_applicability"] is True
    assert row["has_metric_observations"] is True
    assert row["metric_applicability"]["total_contracts"] == 28
    assert gate["registry_fingerprint"] == contracts["registry_fingerprint"]
    assert observations["total_lines"] == 28
    assert trials["total_trials"] == 1
    assert trials["total_observations"] == 28
    assert len(trials["trials"][0]["observations"]) == 28
    assert evaluation["profile"]["profile_id"] == "core"
    assert analyzed["metric_evaluation"]["summary"]["observation_count"] == 28
    assert profiles["num_profiles"] == 1
    assert profiles["profiles"][0]["profile_id"] == "core"
