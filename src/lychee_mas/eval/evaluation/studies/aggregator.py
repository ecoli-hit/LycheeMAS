"""Strict case/run/study aggregation for benchmark comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StudyRun:
    """One run directory assigned to an explicit comparison system."""

    system_id: str
    run_dir: str | os.PathLike

    def __post_init__(self) -> None:
        if not self.system_id.strip():
            raise ValueError("StudyRun system_id cannot be empty")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def _number(record: dict[str, Any], *keys: str) -> float | int | None:
    value = _first(record, *keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _replicate_index(record: dict[str, Any]) -> int:
    return int(_first(record, "trial_index", "replicate_index", default=0) or 0)


def _pair_identity(record: dict[str, Any]) -> tuple[str, str, str, int, int | None]:
    return (
        str(record.get("benchmark_id") or "unknown"),
        str(record.get("task") or "unknown"),
        str(record.get("case_id") or "unknown"),
        _replicate_index(record),
        int(record["trial_seed"]) if record.get("trial_seed") is not None else None,
    )


def _pair_key(identity: tuple[str, str, str, int, int | None]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _trial_row(system_id: str, run_id: str, run_dir: Path, record: dict[str, Any]) -> dict:
    score = _number(record, "score", "correct")
    explicit_correct = _first(record, "is_correct", "correct")
    is_correct = explicit_correct if isinstance(explicit_correct, bool) else None
    identity = _pair_identity(record)
    status = str(record.get("trial_status") or record.get("status") or "completed")
    if score is None and status == "completed":
        status = "missing_score"
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "system_id": system_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pair_key": _pair_key(identity),
        "benchmark_id": identity[0],
        "task": identity[1],
        "case_id": identity[2],
        "dataset_index": record.get("dataset_index"),
        "replicate_index": identity[3],
        "base_seed": record.get("base_seed"),
        "trial_seed": identity[4],
        "status": status,
        "official_score": score,
        "is_correct": is_correct,
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "model_calls": _number(record, "model_call_count"),
        "messages": _number(record, "message_count"),
        "tool_calls": _number(record, "tool_call_count"),
        "tool_errors": _number(record, "tool_error_count"),
        "input_tokens": _number(record, "input_tokens"),
        "output_tokens": _number(record, "output_tokens"),
        "model_latency_s": _number(record, "model_latency_s"),
        "trial_wall_time_s": _number(record, "trial_wall_time_s"),
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return round(statistics.fmean(cleaned), 8) if cleaned else None


def _percentile(values: Iterable[float | int | None], probability: float) -> float | None:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return round(cleaned[0], 8)
    position = (len(cleaned) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    value = cleaned[lower] + (cleaned[upper] - cleaned[lower]) * (position - lower)
    return round(value, 8)


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    margin /= denominator
    return [round(max(0.0, center - margin), 8), round(min(1.0, center + margin), 8)]


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1 or samples <= 0:
        value = round(values[0], 8)
        return [value, value]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [
        round(means[int(0.025 * (samples - 1))], 8),
        round(means[int(0.975 * (samples - 1))], 8),
    ]


def _strict_score(row: dict[str, Any]) -> float:
    score = row.get("official_score")
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0


def _system_summary(system_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    binary = all(row.get("is_correct") is not None for row in completed) and bool(completed)
    successes = sum(1 for row in rows if row.get("is_correct") is True)
    denominator = len(rows)
    return {
        "system_id": system_id,
        "trial_count": denominator,
        "completed_trial_count": len(completed),
        "error_or_unscored_count": denominator - len(completed),
        "score_mean_strict": _mean(_strict_score(row) for row in rows),
        "score_mean_scored_only": _mean(row.get("official_score") for row in completed),
        "binary_accuracy_strict": round(successes / denominator, 8)
        if binary and denominator
        else None,
        "binary_accuracy_wilson_95": _wilson_interval(successes, denominator)
        if binary
        else None,
        "input_tokens_mean_per_completed_trial": _mean(
            row.get("input_tokens") for row in completed
        ),
        "output_tokens_mean_per_completed_trial": _mean(
            row.get("output_tokens") for row in completed
        ),
        "trial_wall_time_s_mean_per_completed_trial": _mean(
            row.get("trial_wall_time_s") for row in completed
        ),
        "trial_wall_time_s_p50": _percentile(
            (row.get("trial_wall_time_s") for row in completed), 0.5
        ),
        "trial_wall_time_s_p95": _percentile(
            (row.get("trial_wall_time_s") for row in completed), 0.95
        ),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


def _metric_rows(
    study_id: str,
    system_id: str,
    run_id: str,
    run_dir: Path,
    observations: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = {
        (
            str(row.get("case_id") or "unknown"),
            row.get("dataset_index"),
            int(row.get("replicate_index") or 0),
        ): row
        for row in trial_rows
    }
    rows = []
    for observation in observations:
        identity = (
            str(observation.get("case_id") or "unknown"),
            observation.get("dataset_index"),
            int(observation.get("trial_index") or 0),
        )
        case = cases.get(identity, {})
        rows.append(
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "study_id": study_id,
                "system_id": system_id,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "pair_key": case.get("pair_key"),
                "benchmark_id": case.get("benchmark_id") or defaults.get("benchmark_id"),
                "task": case.get("task") or defaults.get("task"),
                "case_id": identity[0],
                "dataset_index": identity[1],
                "replicate_index": identity[2],
                "trial_seed": case.get("trial_seed"),
                "metric_id": observation.get("metric_id"),
                "status": observation.get("status"),
                "value": observation.get("value"),
                "unit": observation.get("unit"),
                "level": observation.get("level"),
                "reason": observation.get("reason"),
                "evaluator_fingerprint": observation.get("evaluator_fingerprint"),
                "evidence_event_ids": observation.get("evidence_event_ids") or [],
                "attributes": observation.get("attributes") or {},
            }
        )
    return rows


def _metric_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("system_id") or "unknown"),
                str(row.get("metric_id") or "unknown"),
            )
        ].append(row)
    summaries = []
    for (system_id, metric_id), values in sorted(grouped.items()):
        measured = [
            float(row["value"])
            for row in values
            if row.get("status") == "measured"
            and isinstance(row.get("value"), (int, float))
            and not isinstance(row.get("value"), bool)
        ]
        fingerprints = sorted(
            {
                str(row["evaluator_fingerprint"])
                for row in values
                if row.get("evaluator_fingerprint")
            }
        )
        summaries.append(
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "system_id": system_id,
                "metric_id": metric_id,
                "unit": values[0].get("unit"),
                "level": values[0].get("level"),
                "observation_count": len(values),
                "status_counts": dict(
                    sorted(Counter(str(row.get("status") or "unknown") for row in values).items())
                ),
                "measured_mean": _mean(measured),
                "measured_p50": _percentile(measured, 0.5),
                "measured_p95": _percentile(measured, 0.95),
                "evaluator_fingerprints": fingerprints,
                "mixed_evaluator_fingerprints": len(fingerprints) > 1,
            }
        )
    return summaries


def _pairwise(
    by_system: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    reports = []
    for pair_index, (left, right) in enumerate(combinations(sorted(by_system), 2)):
        left_rows = by_system[left]
        right_rows = by_system[right]
        all_keys = sorted(set(left_rows) | set(right_rows))
        paired_keys = [key for key in all_keys if key in left_rows and key in right_rows]
        differences = [
            _strict_score(left_rows[key]) - _strict_score(right_rows[key])
            for key in paired_keys
        ]
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        ties = sum(value == 0 for value in differences)
        reports.append(
            {
                "system_a": left,
                "system_b": right,
                "union_pair_count": len(all_keys),
                "paired_count": len(paired_keys),
                "missing_from_a_count": sum(key not in left_rows for key in all_keys),
                "missing_from_b_count": sum(key not in right_rows for key in all_keys),
                "pairing_complete": len(paired_keys) == len(all_keys),
                "wins_a": wins,
                "wins_b": losses,
                "ties": ties,
                "mean_score_difference_a_minus_b": _mean(differences),
                "paired_bootstrap_95": _bootstrap_mean_ci(
                    differences,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + pair_index,
                ),
                "a_correct_b_wrong": sum(
                    bool(left_rows[key].get("is_correct"))
                    and not bool(right_rows[key].get("is_correct"))
                    for key in paired_keys
                ),
                "a_wrong_b_correct": sum(
                    not bool(left_rows[key].get("is_correct"))
                    and bool(right_rows[key].get("is_correct"))
                    for key in paired_keys
                ),
            }
        )
    return reports


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def aggregate_study(
    study_id: str,
    runs: Iterable[StudyRun],
    output_dir: str | os.PathLike,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate explicit systems without silently dropping failures or missing pairs."""

    if not study_id.strip():
        raise ValueError("study_id cannot be empty")
    run_inputs = list(runs)
    if not run_inputs:
        raise ValueError("at least one StudyRun is required")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for run_number, run in enumerate(run_inputs, start=1):
        run_dir = Path(run.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        from lychee_mas.eval.evaluation.projections import build_result_projection

        outputs = build_result_projection(run_dir)
        status = _read_json(run_dir / "run_status.json")
        metrics = _read_json(run_dir / "metrics.json")
        run_id = f"run-{run_number:03d}-{run_dir.name}"
        defaults = {
            "benchmark_id": metrics.get("benchmark_id") or status.get("benchmark_id"),
            "task": metrics.get("task") or status.get("task"),
            "base_seed": status.get("base_seed"),
        }
        rows = [
            _trial_row(run.system_id, run_id, run_dir, {**defaults, **record})
            for record in outputs
        ]
        trial_rows.extend(rows)
        observations = _read_jsonl(run_dir / "metric_observations.jsonl")
        run_metric_rows = _metric_rows(
            study_id,
            run.system_id,
            run_id,
            run_dir,
            observations,
            rows,
            defaults,
        )
        metric_rows.extend(run_metric_rows)
        expected = int(status.get("expected_trials") or len(rows))
        run_rows.append(
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "study_id": study_id,
                "system_id": run.system_id,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "run_status": status.get("status") or "unknown",
                "benchmark_id": defaults["benchmark_id"],
                "task": defaults["task"],
                "model": metrics.get("model") or status.get("model"),
                "team": metrics.get("team") or status.get("team"),
                "method": metrics.get("method") or status.get("method"),
                "base_seed": status.get("base_seed"),
                "expected_trials": expected,
                "identified_trial_rows": len(rows),
                "unidentified_missing_trials": max(0, expected - len(rows)),
                "metric_observation_count": len(run_metric_rows),
                "has_metric_observations": bool(observations),
                "is_complete": expected == len(rows) and status.get("status") == "complete",
                **_system_summary(run.system_id, rows),
            }
        )

    seen: dict[tuple[str, str], str] = {}
    by_system: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trial_rows:
        key = (row["system_id"], row["pair_key"])
        if key in seen:
            raise ValueError(
                f"duplicate paired Trial for system={key[0]!r} pair_key={key[1]} "
                f"in {seen[key]} and {row['run_dir']}"
            )
        seen[key] = row["run_dir"]
        by_system[row["system_id"]][row["pair_key"]] = row

    system_rows = [
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": study_id,
            **_system_summary(system, list(rows.values())),
        }
        for system, rows in sorted(by_system.items())
    ]
    pairwise_rows = _pairwise(
        by_system,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    metric_summary_rows = _metric_summaries(metric_rows)
    incomplete_runs = sum(not row["is_complete"] for row in run_rows)
    incomplete_pairs = sum(not row["pairing_complete"] for row in pairwise_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": study_id,
        "generated_at_utc": generated_at,
        "output_dir": str(output_root),
        "summary": {
            "num_systems": len(by_system),
            "num_runs": len(run_rows),
            "num_trial_rows": len(trial_rows),
            "num_metric_observations": len(metric_rows),
            "num_system_metric_summaries": len(metric_summary_rows),
            "incomplete_run_count": incomplete_runs,
            "incomplete_pairwise_comparison_count": incomplete_pairs,
            "pairing_complete": incomplete_runs == 0 and incomplete_pairs == 0,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "systems": system_rows,
        "runs": run_rows,
        "pairwise": pairwise_rows,
        "metric_summaries": metric_summary_rows,
        "artifacts": {
            "trial_jsonl": "study_trials.jsonl",
            "trial_csv": "study_trials.csv",
            "run_jsonl": "study_runs.jsonl",
            "run_csv": "study_runs.csv",
            "system_json": "study_systems.json",
            "pairwise_json": "study_pairwise.json",
            "pairwise_csv": "study_pairwise.csv",
            "metric_observation_jsonl": "study_metric_observations.jsonl",
            "metric_observation_csv": "study_metric_observations.csv",
            "metric_summary_json": "study_metric_summary.json",
            "metric_summary_csv": "study_metric_summary.csv",
            "report": "study_report.json",
        },
    }
    _write_jsonl(output_root / "study_trials.jsonl", trial_rows)
    _write_csv(output_root / "study_trials.csv", trial_rows)
    _write_jsonl(output_root / "study_runs.jsonl", run_rows)
    _write_csv(output_root / "study_runs.csv", run_rows)
    _write_json(output_root / "study_systems.json", system_rows)
    _write_json(output_root / "study_pairwise.json", pairwise_rows)
    _write_csv(output_root / "study_pairwise.csv", pairwise_rows)
    _write_jsonl(output_root / "study_metric_observations.jsonl", metric_rows)
    _write_csv(output_root / "study_metric_observations.csv", metric_rows)
    _write_json(output_root / "study_metric_summary.json", metric_summary_rows)
    _write_csv(output_root / "study_metric_summary.csv", metric_summary_rows)
    _write_json(output_root / "study_report.json", report)
    return report
