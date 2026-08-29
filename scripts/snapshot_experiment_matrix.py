#!/usr/bin/env python3
"""Create a read-only partial metric snapshot for ExperimentInstances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lychee_mas.eval.benchmarks import get_benchmark  # noqa: E402
from lychee_mas.eval.evaluation import metrics as M  # noqa: E402
from lychee_mas.eval.evaluation.projections import build_result_projection  # noqa: E402
from lychee_mas.runtime.events.store import event_payload, iter_run_events  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_id(item: dict[str, Any], index: int) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("safe_task_id") or index)


def _topology(instance_id: str) -> str:
    for name in ("independent", "sequential", "centralized", "decentralized"):
        if f"-{name}-" in instance_id:
            return name
    return "unknown"


def _latest_swe_report(run_dir: Path) -> dict[str, Any]:
    report_dir = run_dir / "official_evaluation" / "swe_bench_verified"
    reports = [path for path in report_dir.glob("*.json") if path.is_file()]
    return _read_json(max(reports, key=lambda path: path.stat().st_mtime)) if reports else {}


def _finite(value: Any) -> float | int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return value if math.isfinite(float(value)) else None


def _message_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        parts = [_message_text(item) for item in value]
        retained = [part for part in parts if part]
        return "\n".join(retained) if retained else None
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return _message_text(value[key])
    return None


def _normalized_message(value: Any) -> str | None:
    text = _message_text(value)
    if text is None:
        return None
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _coordination_snapshot(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Evaluate coordination metrics over the same terminal Trials as the snapshot."""

    trial_keys = {
        (str(record.get("case_id") or ""), int(record.get("trial_index") or 0))
        for record in records
        if record.get("case_id") is not None
    }
    grouped: dict[tuple[str, int], list[tuple[str, Any]]] = defaultdict(list)
    for event in iter_run_events(run_dir, event_types=("agent.message.published",)):
        key = (str(event.get("case_id") or ""), int(event.get("trial_index") or 0))
        if key not in trial_keys:
            continue
        payload = event_payload(event)
        actor = str(payload.get("source") or payload.get("actor") or "").strip()
        if actor.casefold() in {"user", "system"}:
            continue
        grouped[key].append((actor, payload.get("content")))

    repetitions: list[float] = []
    balances: list[float] = []
    active_roles: list[float] = []
    message_counts: list[float] = []
    for key in trial_keys:
        messages = grouped.get(key, [])
        if not messages:
            continue
        message_counts.append(float(len(messages)))
        normalized = [
            value
            for _, content in messages
            if (value := _normalized_message(content)) is not None
        ]
        if normalized:
            repetitions.append((len(normalized) - len(set(normalized))) / len(normalized))
        if any(not actor for actor, _ in messages):
            continue
        counts = Counter(actor for actor, _ in messages)
        active_roles.append(float(len(counts)))
        if len(counts) >= 2:
            total = sum(counts.values())
            entropy = -sum(
                (count / total) * math.log(count / total) for count in counts.values()
            )
            balances.append(entropy / math.log(len(counts)))

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 8) if values else None

    return {
        "message_repetition_rate": mean(repetitions),
        "role_participation_balance": mean(balances),
        "active_role_count": mean(active_roles),
        "agent_message_count": mean(message_counts),
    }


def _existing_metric_row(
    *,
    metrics: dict[str, Any],
    benchmark_id: str,
    instance_id: str,
    run_status: str,
    terminal_case_count: int,
    runtime_failed_cases: int,
    run_dir: Path,
) -> dict[str, Any] | None:
    if int(metrics.get("num_distinct_cases") or 0) != terminal_case_count:
        return None
    equivalent = ((metrics.get("costing") or {}).get("api_equivalent_cost") or {})
    totals = equivalent.get("totals_by_currency") or {}
    cost = totals.get("CNY")
    if cost is None and len(totals) == 1:
        cost = next(iter(totals.values()))
    scored_cases = int(metrics.get("num_distinct_cases") or 0)
    return {
        "benchmark": benchmark_id,
        "topology": _topology(instance_id),
        "instance_id": instance_id,
        "run_status": run_status,
        "observed_terminal_cases": terminal_case_count,
        "scored_cases": scored_cases,
        "runtime_failed_cases": runtime_failed_cases,
        "pending_official_cases": 0,
        "scoring_failures": 0,
        "correct_cases": metrics.get("num_correct"),
        "accuracy": _finite(metrics.get("accuracy")),
        "input_tokens_per_scored_case": _finite(
            metrics.get("mean_input_text_tokens_per_case")
        ),
        "output_tokens_per_scored_case": _finite(
            metrics.get("mean_output_text_tokens_per_case")
        ),
        "model_calls_per_scored_case": _finite(metrics.get("mean_model_calls_per_case")),
        "messages_per_scored_case": _finite(metrics.get("mean_messages_per_case")),
        "tool_calls_per_scored_case": _finite(metrics.get("mean_tool_calls_per_case")),
        "tool_errors_per_scored_case": _finite(metrics.get("mean_tool_errors_per_case")),
        "model_latency_s_per_scored_case": _finite(
            metrics.get("mean_model_latency_s_per_case")
        ),
        "trial_wall_s_per_scored_case": _finite(
            metrics.get("mean_aggregate_trial_wall_time_s_per_case")
        ),
        "api_equivalent_cost_per_scored_case": (
            round(float(cost) / scored_cases, 8) if cost is not None and scored_cases else None
        ),
        "api_equivalent_cost_snapshot": round(float(cost), 8) if cost is not None else None,
        "run_dir": str(run_dir),
    }


def _gold_index(benchmark, task: str, records: list[dict[str, Any]]):
    max_index = max((int(record.get("dataset_index") or 0) for record in records), default=0)
    data = benchmark.load(task, n=max_index + 1)
    return data, {_case_id(item, index): item for index, item in enumerate(data)}


def _score_unscored_records(
    benchmark,
    task: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not records:
        return [], 0
    data, by_id = _gold_index(benchmark, task, records)
    scored: list[dict[str, Any]] = []
    failures = 0
    for record in records:
        item = by_id.get(str(record.get("case_id")))
        if item is None:
            index = int(record.get("dataset_index") or -1)
            item = data[index] if 0 <= index < len(data) else None
        if item is None:
            failures += 1
            continue
        kind = record.get("scorer_kind") or item.get("kind")
        try:
            details = benchmark.score(
                record.get("prediction") or "",
                {**item, "kind": kind},
                record=record,
            )
        except Exception:
            failures += 1
            continue
        score = float(details.get("score", 0.0))
        scored.append(
            {
                **record,
                "score": score,
                "is_correct": score == 1.0,
                "score_details": details,
                "evaluation_status": "snapshot_scored",
            }
        )
    return scored, failures


def _snapshot_instance(
    instance_id: str,
    *,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> dict[str, Any]:
    instance_path = ROOT / "configs/eval_studio/experiments/instances" / f"{instance_id}.json"
    launch_dir = ROOT / "runs/eval_studio/launches" / instance_id
    instance = _read_json(instance_path)
    job = _read_json(launch_dir / "job.json")
    run_dir = Path(job["run_dir"])
    status = _read_json(run_dir / "run_status.json")
    projection = build_result_projection(run_dir)
    terminal = [
        record
        for record in projection
        if record.get("trial_status") in {"completed", "failed"}
    ]
    completed = [record for record in terminal if record.get("trial_status") == "completed"]
    failed = [record for record in terminal if record.get("trial_status") == "failed"]
    first = (completed or failed)[0]
    task = str(status.get("task") or first.get("task") or "")
    benchmark = get_benchmark(task)
    terminal_case_count = len({str(record.get("case_id")) for record in terminal})
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and str(status.get("status") or "") not in {
        "running",
        "draining",
    }:
        existing = _existing_metric_row(
            metrics=_read_json(metrics_path),
            benchmark_id=benchmark.id,
            instance_id=instance_id,
            run_status=str(status.get("status") or instance.get("status")),
            terminal_case_count=terminal_case_count,
            runtime_failed_cases=len({str(record.get("case_id")) for record in failed}),
            run_dir=run_dir,
        )
        if existing is not None:
            coordination = _coordination_snapshot(run_dir, terminal)
            existing.update(coordination)
            existing["messages_per_scored_case"] = coordination["agent_message_count"]
            return existing

    quality: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for record in completed:
        if record.get("evaluation_status") == "completed" and record.get("score") is not None:
            score = float(record["score"])
            quality.append({**record, "score": score, "is_correct": score == 1.0})
        else:
            unscored.append(record)

    pending_official = 0
    scoring_failures = 0
    if benchmark.id == "swe_bench_verified":
        report = _latest_swe_report(run_dir)
        submitted = set(report.get("submitted_ids") or [])
        resolved = set(report.get("resolved_ids") or [])
        already_scored = {str(record.get("case_id")) for record in quality}
        for record in unscored:
            case_id = str(record.get("case_id"))
            if case_id in already_scored:
                continue
            if case_id not in submitted:
                pending_official += 1
                continue
            quality.append(
                {
                    **record,
                    "score": 1.0 if case_id in resolved else 0.0,
                    "is_correct": case_id in resolved,
                    "evaluation_status": "official_report_reused",
                }
            )
    else:
        newly_scored, scoring_failures = _score_unscored_records(
            benchmark,
            task,
            unscored,
        )
        quality.extend(newly_scored)

    quality.extend(
        {
            **record,
            "score": 0.0,
            "is_correct": False,
            "evaluation_status": "runtime_failed",
        }
        for record in failed
    )
    run_info = {
        "model": "Qwen3.6-27B",
        "backend_provider": "vllm",
        "method": "none",
        "team": instance_id,
        "task": task,
        "benchmark_id": benchmark.id,
        "scorer_kind": first.get("scorer_kind") or task,
        "probe": "mas",
        "summary_scope": "partial_snapshot",
    }
    aggregate = M.aggregate_trials(quality, run_info=run_info) if quality else {}
    coordination = _coordination_snapshot(run_dir, quality)
    scored_cases = int(aggregate.get("num_distinct_cases") or 0)
    input_tokens = sum(
        int(record.get("input_text_tokens") or record.get("input_tokens") or 0)
        for record in quality
    )
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in quality)
    api_cost = None
    if input_price_per_million is not None and output_price_per_million is not None:
        api_cost = (
            input_tokens * input_price_per_million
            + output_tokens * output_price_per_million
        ) / 1_000_000
    return {
        "benchmark": benchmark.id,
        "topology": _topology(instance_id),
        "instance_id": instance_id,
        "run_status": status.get("status") or instance.get("status"),
        "observed_terminal_cases": terminal_case_count,
        "scored_cases": scored_cases,
        "runtime_failed_cases": len({str(record.get("case_id")) for record in failed}),
        "pending_official_cases": pending_official,
        "scoring_failures": scoring_failures,
        "correct_cases": aggregate.get("num_correct"),
        "accuracy": _finite(aggregate.get("accuracy")),
        "input_tokens_per_scored_case": _finite(
            aggregate.get("mean_input_text_tokens_per_case")
        ),
        "output_tokens_per_scored_case": _finite(
            aggregate.get("mean_output_text_tokens_per_case")
        ),
        "model_calls_per_scored_case": _finite(aggregate.get("mean_model_calls_per_case")),
        "messages_per_scored_case": coordination["agent_message_count"],
        **coordination,
        "tool_calls_per_scored_case": _finite(aggregate.get("mean_tool_calls_per_case")),
        "tool_errors_per_scored_case": _finite(aggregate.get("mean_tool_errors_per_case")),
        "model_latency_s_per_scored_case": _finite(
            aggregate.get("mean_model_latency_s_per_case")
        ),
        "trial_wall_s_per_scored_case": _finite(
            aggregate.get("mean_aggregate_trial_wall_time_s_per_case")
        ),
        "api_equivalent_cost_per_scored_case": (
            round(api_cost / scored_cases, 8) if api_cost is not None and scored_cases else None
        ),
        "api_equivalent_cost_snapshot": round(api_cost, 8) if api_cost is not None else None,
        "run_dir": str(run_dir),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Partial Experiment Matrix",
        "",
        (
            "> Unfinished Cases are excluded. SWE-bench quality includes only Cases "
            "already scored by the official harness."
        ),
        "",
        (
            "| Benchmark | Topology | Status | Terminal | Scored | Pending official | Acc. "
            "| Rep. | Balance | Roles | Msgs/case | Input/case | Output/case | "
            "Calls/case | Tools/case | "
            "Model latency/case (s) | Wall/case (s) | API-eq cost/case |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["benchmark"]),
                    str(row["topology"]),
                    str(row["run_status"]),
                    str(row["observed_terminal_cases"]),
                    str(row["scored_cases"]),
                    str(row["pending_official_cases"]),
                    fmt(row["accuracy"], 3),
                    fmt(row["message_repetition_rate"], 4),
                    fmt(row["role_participation_balance"], 4),
                    fmt(row["active_role_count"]),
                    fmt(row["agent_message_count"]),
                    fmt(row["input_tokens_per_scored_case"], 1),
                    fmt(row["output_tokens_per_scored_case"], 1),
                    fmt(row["model_calls_per_scored_case"]),
                    fmt(row["tool_calls_per_scored_case"]),
                    fmt(row["model_latency_s_per_scored_case"], 1),
                    fmt(row["trial_wall_s_per_scored_case"], 1),
                    fmt(row["api_equivalent_cost_per_scored_case"], 4),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    args = parser.parse_args()

    rows = [
        _snapshot_instance(
            instance_id,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
        for instance_id in args.instance_id
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "scope": (
                    "terminal Trials observed at snapshot time; unfinished Cases excluded; "
                    "SWE quality includes only official-harness-scored Cases"
                ),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = _markdown(rows)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(output_dir)
    print(markdown)


if __name__ == "__main__":
    main()
