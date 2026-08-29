"""Run browsing and offline-evaluation use cases for every Eval interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..evaluation.evaluators import (
    EvaluationProfileRegistry,
    evaluate_run_metrics,
    materialize_run_evaluation,
)
from ..evaluation.evidence import normalize_run_evidence
from ..evaluation.metrics_registry import MetricRegistry, evaluate_run_metric_applicability
from .ports import ExperimentRepositoryPort
from .read_models.catalog import EvalCatalog
from .read_models.run_events import (
    read_execution_trace,
    read_jsonl_events,
    read_metric_trial_groups,
    read_result_projection,
    read_run_events,
)


class RunApplicationService:
    """Expose one stable query/command surface for persisted run artifacts."""

    def __init__(
        self,
        *,
        catalog: EvalCatalog,
        experiments: ExperimentRepositoryPort,
        metrics: MetricRegistry,
        evaluation_profiles: EvaluationProfileRegistry,
    ) -> None:
        self.catalog = catalog
        self.experiments = experiments
        self.metrics = metrics
        self.evaluation_profiles = evaluation_profiles

    def registered_roots(self) -> list[str | os.PathLike[Any]]:
        values: list[str | os.PathLike[Any]] = [str(self.catalog.runs_root)]
        candidates = [
            *(
                (spec.get("environment") or {}).get("runs_root")
                for spec in self.experiments.specs()
            ),
            *(instance.get("run_dir") for instance in self.experiments.instances()),
        ]
        for value in candidates:
            text = str(value or "").strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if not path.is_absolute():
                path = self.catalog.repo_root / path
            resolved = str(path.resolve())
            if resolved not in values:
                values.append(resolved)
        return values

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.catalog.runs(limit=limit, roots=self.registered_roots())

    def list_runs_page(
        self,
        *,
        query: str = "",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Search and page compact Run cards without returning full artifacts."""

        rows = self.catalog.runs(limit=100000, roots=self.registered_roots())
        normalized = str(query or "").strip().casefold()
        if normalized:
            rows = [
                row
                for row in rows
                if normalized
                in " ".join(
                    [
                        str(row.get("id") or ""),
                        str(row.get("relative_path") or ""),
                        str(row.get("run_dir") or ""),
                        str((row.get("status") or {}).get("task") or ""),
                        str((row.get("status") or {}).get("kind") or ""),
                        str((row.get("status") or {}).get("status") or ""),
                    ]
                ).casefold()
            ]
        start = max(0, int(offset))
        page_size = max(1, int(limit))
        return {
            "items": rows[start : start + page_size],
            "offset": start,
            "limit": page_size,
            "total": len(rows),
            "query": query,
        }

    def metric_contracts(self) -> dict[str, Any]:
        return self.metrics.descriptor()

    def evaluation_profile_contracts(self) -> dict[str, Any]:
        return self.evaluation_profiles.descriptor()

    def run_dir(self, run_id: str) -> Path:
        return self.catalog.run_dir(run_id, roots=self.registered_roots())

    def events(
        self,
        run_id: str,
        *,
        start_line: int,
        limit: int,
        include_total: bool = True,
    ) -> dict[str, Any]:
        return read_run_events(
            self.run_dir(run_id),
            start_line=start_line,
            limit=limit,
            include_total=include_total,
        )

    def execution_trace(
        self,
        run_id: str,
        *,
        case_id: str | None,
        trial_index: int | None,
        filters: tuple[str, ...],
        start_node: int,
        limit: int,
    ) -> dict[str, Any]:
        unknown = set(filters) - {"messages", "model_calls", "tools", "errors"}
        if unknown:
            raise ValueError(f"unknown trace filters: {sorted(unknown)}")
        return read_execution_trace(
            self.run_dir(run_id),
            case_id=case_id,
            trial_index=trial_index,
            filters=filters,
            start_node=start_node,
            limit=limit,
        )

    def results(
        self,
        run_id: str,
        *,
        start_trial: int,
        limit: int,
    ) -> dict[str, Any]:
        return read_result_projection(
            self.run_dir(run_id),
            start_trial=start_trial,
            limit=limit,
        )

    def evidence(self, run_id: str, *, start_line: int, limit: int) -> dict[str, Any]:
        return read_jsonl_events(
            self.run_dir(run_id) / "evidence.jsonl",
            start_line=start_line,
            limit=limit,
        )

    def metric_observations(
        self,
        run_id: str,
        *,
        start_line: int,
        limit: int,
    ) -> dict[str, Any]:
        return read_jsonl_events(
            self.run_dir(run_id) / "metric_observations.jsonl",
            start_line=start_line,
            limit=limit,
        )

    def metric_trials(
        self,
        run_id: str,
        *,
        start_trial: int,
        limit: int,
    ) -> dict[str, Any]:
        return read_metric_trial_groups(
            self.run_dir(run_id) / "metric_observations.jsonl",
            start_trial=start_trial,
            limit=limit,
        )

    def evidence_coverage(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        cached = self._read_mapping(run_dir / "evidence_coverage.json")
        return cached or normalize_run_evidence(run_dir, write=False)

    def metric_applicability(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        cached = self._read_mapping(run_dir / "metric_applicability.json")
        if cached:
            return cached
        coverage = normalize_run_evidence(run_dir, write=False)
        return evaluate_run_metric_applicability(
            run_dir,
            coverage_report=coverage,
            registry=self.metrics,
            write=False,
        )

    def metric_evaluation(self, run_id: str, *, profile_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        cached = self._read_mapping(run_dir / "metric_evaluation.json")
        if cached and cached.get("profile", {}).get("profile_id") == profile_id:
            return cached
        return evaluate_run_metrics(
            run_dir,
            profile_id=profile_id,
            metric_registry=self.metrics,
            profile_registry=self.evaluation_profiles,
            write=False,
        )

    def analyze(self, run_id: str, *, profile_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        status = self._read_mapping(run_dir / "run_status.json") or {}
        if str(status.get("status") or "") in {"starting", "running", "queued"}:
            raise ValueError("cannot analyze a run while it is still active")
        return materialize_run_evaluation(
            run_dir,
            profile_id=profile_id,
            write=True,
            metric_registry=self.metrics,
            profile_registry=self.evaluation_profiles,
        )

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
