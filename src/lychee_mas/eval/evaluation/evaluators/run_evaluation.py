"""Materialize evidence, applicability, and metric evaluation for an existing run."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import run_events_path

from ..evidence import normalize_run_evidence
from ..metrics_registry import MetricRegistry, evaluate_run_metric_applicability
from .metric_evaluator import evaluate_run_metrics
from .profiles import EvaluationProfileRegistry


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_run_evaluation(
    run_dir: str | os.PathLike,
    *,
    profile_id: str = "core",
    write: bool = True,
    evaluate_metrics: bool = True,
    metric_registry: MetricRegistry | None = None,
    profile_registry: EvaluationProfileRegistry | None = None,
) -> dict[str, Any]:
    """Generate the complete offline evaluation layer without rerunning inference."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not run_events_path(root).is_file():
        raise FileNotFoundError(f"Run EventLog not found below {root / 'events'}")

    metric_registry = metric_registry or MetricRegistry()
    profile_registry = profile_registry or EvaluationProfileRegistry()
    coverage = normalize_run_evidence(root, write=write)
    applicability = evaluate_run_metric_applicability(
        root,
        coverage_report=coverage,
        registry=metric_registry,
        write=write,
    )
    evaluation = None
    if evaluate_metrics:
        evaluation = evaluate_run_metrics(
            root,
            profile_id=profile_id,
            metric_registry=metric_registry,
            profile_registry=profile_registry,
            write=write,
        )

    if write:
        metrics_path = root / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            metrics = {}
        if not isinstance(metrics, dict):
            metrics = {}
        metrics.update(
            {
                "evidence_coverage": coverage["summary"],
                "evidence_artifacts": coverage["artifacts"],
                "metric_applicability": applicability["summary"],
                "metric_registry_fingerprint": applicability["registry_fingerprint"],
                "metric_artifacts": applicability["artifacts"],
            }
        )
        if evaluation is not None:
            metrics.update(
                {
                    "metric_evaluation": evaluation["summary"],
                    "evaluation_profile": evaluation["profile"],
                    "metric_observation_artifacts": evaluation["artifacts"],
                }
            )
        _atomic_write_json(metrics_path, metrics)

    return {
        "run_dir": str(root),
        "evidence_coverage": coverage,
        "metric_applicability": applicability,
        "metric_evaluation": evaluation,
    }
