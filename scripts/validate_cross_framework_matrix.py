#!/usr/bin/env python3
"""Validate inference, official scoring, evidence, and metrics for one framework matrix."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .setup_cross_framework_matrix import (
        BENCHMARK_ORDER,
        FRAMEWORKS,
        TOPOLOGIES,
        _instance_id,
        _selected_recipes,
        _team_instance_id,
    )
except ImportError:  # pragma: no cover - direct script execution
    from setup_cross_framework_matrix import (
        BENCHMARK_ORDER,
        FRAMEWORKS,
        TOPOLOGIES,
        _instance_id,
        _selected_recipes,
        _team_instance_id,
    )


def _get_json(base_url: str, path: str) -> Any:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _selected(
    raw: str,
    allowed: tuple[str, ...],
    option: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = set(values) - set(allowed)
    if (not values and not allow_empty) or unknown:
        raise ValueError(f"{option} contains unsupported values: {sorted(unknown)}")
    return values


def _validate_instance(
    instance: dict[str, Any],
    *,
    expected_team_instance_id: str,
    target: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if str(instance.get("team_instance_id") or "") != expected_team_instance_id:
        errors.append("TeamInstance binding does not match the matrix identity")
    if str(instance.get("configuration_status") or "") != "current":
        errors.append("configuration_status is not current")
    progress = dict(instance.get("progress") or {})
    completed = int(progress.get("completed_distinct_cases") or 0)
    successful = int(progress.get("successful_trials") or 0)
    failed = int(progress.get("failed_trials") or 0)
    if completed < target:
        errors.append(f"only {completed}/{target} distinct Cases completed")
    if successful < target:
        errors.append(f"only {successful}/{target} Trials completed successfully")
    if failed:
        errors.append(f"{failed} runtime-failed Trials were recorded")

    run_dir_text = str(instance.get("run_dir") or "")
    run_dir = Path(run_dir_text) if run_dir_text else None
    if run_dir is None or not run_dir.is_dir():
        errors.append("run_dir is missing")
        return {
            "status": "failed",
            "errors": errors,
            "run_dir": run_dir_text or None,
            "completed_cases": completed,
        }

    try:
        evaluation = _read_json(run_dir / "evaluation_status.json")
        coverage = _read_json(run_dir / "evidence_coverage.json")
        metrics = _read_json(run_dir / "metric_evaluation.json")
    except ValueError as exc:
        errors.append(str(exc))
        return {
            "status": "failed",
            "errors": errors,
            "run_dir": str(run_dir),
            "completed_cases": completed,
        }

    evaluated = int(evaluation.get("evaluated_trials") or 0)
    if evaluated < target:
        errors.append(f"official scorer evaluated only {evaluated}/{target} Trials")
    if str(coverage.get("overall_status") or "") != "complete":
        errors.append("normalized evidence coverage is not complete")
    metric_summary = dict(metrics.get("summary") or {})
    metric_count = int(metric_summary.get("metric_count") or 0)
    observations = int(metric_summary.get("observation_count") or 0)
    evaluator_errors = int(metric_summary.get("evaluator_error_count") or 0)
    if metric_count < 1:
        errors.append("metric profile contains no metrics")
    if observations < target * metric_count:
        errors.append(
            f"only {observations}/{target * metric_count} metric observations were emitted"
        )
    if evaluator_errors:
        errors.append(f"metric evaluator reported {evaluator_errors} errors")
    metric_rows = {
        str(item.get("metric_id") or ""): item
        for item in metrics.get("metrics") or []
    }
    result_contract_metric = metric_rows.get("reliability.result_contract_valid")
    if result_contract_metric is None:
        errors.append("result contract validity metric is missing")
    else:
        measured_contracts = int(
            (result_contract_metric.get("status_counts") or {}).get("measured") or 0
        )
        if measured_contracts < target:
            errors.append(
                f"result contract validity measured only {measured_contracts}/{target} Trials"
            )
        contract_valid_mean = result_contract_metric.get("measured_mean")
        if contract_valid_mean is None or float(contract_valid_mean) < 1.0:
            errors.append(
                "one or more Trials did not produce a valid result-contract submission"
            )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "run_dir": str(run_dir),
        "completed_cases": completed,
        "successful_trials": successful,
        "failed_trials": failed,
        "officially_evaluated_trials": evaluated,
        "metric_count": metric_count,
        "metric_observation_count": observations,
        "result_contract_valid_mean": (
            result_contract_metric.get("measured_mean")
            if result_contract_metric is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_ORDER))
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    parser.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    parser.add_argument(
        "--include-gaia-magentic-one",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.target < 1:
        raise SystemExit("--target must be positive")
    try:
        benchmarks = _selected(args.benchmarks, BENCHMARK_ORDER, "--benchmarks")
        topologies = _selected(
            args.topologies,
            TOPOLOGIES,
            "--topologies",
            allow_empty=args.include_gaia_magentic_one,
        )
        frameworks = _selected(args.frameworks, FRAMEWORKS, "--frameworks")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    instances = {
        str(item["id"]): item
        for item in _get_json(args.base_url, "/api/experiment-instances")
    }
    rows: list[dict[str, Any]] = []
    recipes = _selected_recipes(
        benchmarks,
        topologies,
        include_gaia_magentic_one=args.include_gaia_magentic_one,
    )
    for benchmark, topology in recipes:
        for framework in frameworks:
                instance_id = _instance_id(
                    benchmark, topology, framework, args.matrix_id
                )
                instance = instances.get(instance_id)
                if instance is None:
                    result = {"status": "failed", "errors": ["instance is missing"]}
                else:
                    result = _validate_instance(
                        instance,
                        expected_team_instance_id=_team_instance_id(
                            benchmark, topology, framework, args.matrix_id
                        ),
                        target=args.target,
                    )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "topology": topology,
                        "framework": framework,
                        "experiment_instance_id": instance_id,
                        **result,
                    }
                )

    failures = [item for item in rows if item["status"] != "passed"]
    completed_trials = sum(int(item.get("completed_cases") or 0) for item in rows)
    successful_trials = sum(
        int(item.get("successful_trials") or 0) for item in rows
    )
    failed_trials = sum(int(item.get("failed_trials") or 0) for item in rows)
    report = {
        "matrix_id": args.matrix_id,
        "target": args.target,
        "expected_combinations": len(rows),
        "passed_combinations": len(rows) - len(failures),
        "failed_combinations": len(failures),
        "expected_trials": len(rows) * args.target,
        "completed_trials": completed_trials,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "status": "passed" if not failures else "failed",
        "combinations": rows,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
