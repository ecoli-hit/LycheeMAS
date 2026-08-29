"""Analyze or score an existing LycheeMAS benchmark run without inference."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas.eval.benchmarks import (  # noqa: E402
    BENCHMARKS,
    Benchmark,
    BenchmarkEvaluationError,
    EvaluationContext,
    get_benchmark,
)
from lychee_mas.eval.evaluation import metrics as M  # noqa: E402
from lychee_mas.eval.evaluation.events import (  # noqa: E402
    failed_evaluations,
    write_evaluation_events,
)


def _get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_config(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


def _case_id(item: dict, index: int) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("safe_task_id") or index)


def _run_info(snapshot: dict[str, Any], trials: list[dict], args) -> dict[str, Any]:
    resolved = snapshot.get("resolved") if isinstance(snapshot, dict) else {}
    resolved = resolved if isinstance(resolved, dict) else {}
    cfg = snapshot.get("config") if isinstance(snapshot, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    first = trials[0] if trials else {}
    return {
        "model": resolved.get("model")
        or resolved.get("model_tag")
        or _get(cfg, "backend.model_tag"),
        "backend_provider": resolved.get("backend_provider") or _get(cfg, "backend.provider"),
        "method": resolved.get("method") or first.get("method") or _get(cfg, "router.method"),
        "team": resolved.get("team") or _get(cfg, "run.team"),
        "task": args.task or resolved.get("task") or first.get("task") or _get(cfg, "run.task"),
        "probe": "mas",
        "memory": resolved.get("memory") or _get(cfg, "memory.name") or "cdm",
        "router": resolved.get("router"),
        "scorer_kind": args.kind or resolved.get("scorer_kind") or first.get("scorer_kind"),
        "benchmark_id": resolved.get("benchmark_id") or first.get("benchmark_id"),
        "start_index": args.start_index
        if args.start_index is not None
        else resolved.get("start_index", 0),
        "expected_num_cases": resolved.get("n"),
        "requested_trials_per_case": resolved.get("trials_per_case", 1),
    }


def _load_gold_items(
    benchmark: Benchmark,
    task: str,
    start_index: int,
    trials: list[dict],
) -> tuple[list[dict], dict[str, dict], dict[int, dict]]:
    indices = [
        int(trial["dataset_index"])
        for trial in trials
        if trial.get("dataset_index") is not None
    ]
    count = len({str(trial.get("case_id")) for trial in trials})
    load_count = max(indices) + 1 if indices else start_index + count
    data = benchmark.load(task, n=load_count)
    by_index = {index: item for index, item in enumerate(data)}
    by_id = {_case_id(item, index): item for index, item in by_index.items()}
    selected = (
        [by_index[index] for index in sorted(set(indices)) if index in by_index]
        if indices
        else data[start_index : start_index + count]
    )
    return selected, by_id, by_index


def _score_trials(
    benchmark: Benchmark,
    trials: list[dict],
    gold_data: list[dict],
    gold_by_id: dict[str, dict],
    gold_by_index: dict[int, dict],
    *,
    kind: str | None,
) -> list[dict]:
    evaluated = []
    for offset, trial in enumerate(trials):
        item = gold_by_id.get(str(trial.get("case_id")))
        if item is None and trial.get("dataset_index") is not None:
            item = gold_by_index.get(int(trial["dataset_index"]))
        if item is None and offset < len(gold_data):
            item = gold_data[offset]
        if item is None:
            exc = RuntimeError(f"Cannot find gold item for Trial case_id={trial.get('case_id')!r}")
            from lychee_mas.runtime.events.store import exception_record

            evaluated.append({**trial, "evaluation_status": "failed", **exception_record(exc)})
            continue
        scorer_kind = kind or trial.get("scorer_kind") or item.get("kind")
        try:
            details = benchmark.score(
                trial.get("prediction", ""),
                {**item, "kind": scorer_kind},
                record=trial,
            )
        except Exception as exc:
            from lychee_mas.runtime.events.store import exception_record

            evaluated.append(
                {
                    **trial,
                    "scorer_kind": scorer_kind,
                    "benchmark_id": benchmark.id,
                    "evaluation_status": "failed",
                    **exception_record(exc),
                }
            )
            continue
        score = float(details.get("score", 0.0))
        scored_trial = dict(trial)
        scored_trial.update(
            {
                "scorer_kind": scorer_kind,
                "benchmark_id": benchmark.id,
                "gold": item.get("gold"),
                "score": score,
                "score_details": details,
                "is_correct": bool(score == 1.0) if M.is_binary_scorer(scorer_kind) else None,
                "correct": score,
                "evaluation_status": "completed",
            }
        )
        evaluated.append(scored_trial)
    return evaluated


def _score_runtime_failed_trials(
    benchmark: Benchmark,
    trials: list[dict],
    gold_data: list[dict],
    gold_by_id: dict[str, dict],
    gold_by_index: dict[int, dict],
    *,
    kind: str | None,
) -> list[dict]:
    """Count terminal runtime failures as zero without invoking the scorer."""

    evaluated = []
    for offset, trial in enumerate(trials):
        item = gold_by_id.get(str(trial.get("case_id")))
        if item is None and trial.get("dataset_index") is not None:
            item = gold_by_index.get(int(trial["dataset_index"]))
        if item is None and offset < len(gold_data):
            item = gold_data[offset]
        scorer_kind = kind or trial.get("scorer_kind") or (item or {}).get("kind")
        evaluated.append(
            {
                **trial,
                "scorer_kind": scorer_kind,
                "benchmark_id": benchmark.id,
                "gold": (item or {}).get("gold"),
                "score": 0.0,
                "score_details": {
                    "score": 0.0,
                    "reason": "trial_runtime_failed",
                    "error_type": trial.get("error_type"),
                    "error_message": trial.get("error_message"),
                },
                "is_correct": False if M.is_binary_scorer(scorer_kind) else None,
                "correct": 0.0,
                "evaluation_status": "completed",
            }
        )
    return evaluated


def _with_partial_flags(metrics: dict, run_info: dict, trial_count: int) -> dict:
    expected_n = run_info.get("expected_num_cases")
    if isinstance(expected_n, int):
        metrics["expected_num_cases"] = expected_n
        actual_cases = int(metrics.get("num_distinct_cases", trial_count) or 0)
        requested_k = int(run_info.get("requested_trials_per_case", 1) or 1)
        metrics["expected_num_trials"] = expected_n * requested_k
        metrics["is_partial"] = (
            actual_cases < expected_n
            or trial_count < expected_n * requested_k
            or not bool(metrics.get("is_sampling_complete", requested_k == 1))
        )
    return metrics


def _attach_vllm_service_metrics(metrics: dict, run_dir: str) -> dict:
    path = os.path.join(run_dir, "vllm_metrics_summary.json")
    if not os.path.isfile(path):
        metrics["vllm_service_metrics"] = None
        return metrics
    try:
        with open(path, encoding="utf-8") as handle:
            metrics["vllm_service_metrics"] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        metrics["vllm_service_metrics"] = {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return metrics


def _attach_costing(metrics: dict, trials: list[dict], run_dir: str) -> dict:
    from lychee_mas.eval.pricing.costing import attach_cost_metrics

    return attach_cost_metrics(metrics, trials, run_dir)


def _attach_evidence_coverage(
    metrics: dict[str, Any],
    run_dir: str,
    *,
    write: bool,
    evaluation_profile: str,
    evaluate_metrics: bool,
) -> dict[str, Any]:
    from lychee_mas.eval.evaluation.evaluators import materialize_run_evaluation

    result = materialize_run_evaluation(
        run_dir,
        profile_id=evaluation_profile,
        write=write,
        evaluate_metrics=evaluate_metrics,
    )
    report = result["evidence_coverage"]
    metrics["evidence_coverage"] = report["summary"]
    metrics["evidence_artifacts"] = report["artifacts"]
    applicability = result["metric_applicability"]
    metrics["metric_applicability"] = applicability["summary"]
    metrics["metric_registry_fingerprint"] = applicability["registry_fingerprint"]
    metrics["metric_artifacts"] = applicability["artifacts"]
    if evaluate_metrics:
        evaluation = result["metric_evaluation"]
        assert evaluation is not None
        metrics["metric_evaluation"] = evaluation["summary"]
        metrics["evaluation_profile"] = evaluation["profile"]
        metrics["metric_observation_artifacts"] = evaluation["artifacts"]
    return metrics


def _should_audit_contamination(task: str | None, mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if not task:
        return False
    try:
        return get_benchmark(task).contamination_audit_default
    except KeyError:
        return False


def _attach_contamination_audit(
    metrics: dict,
    trials: list[dict],
    gold_by_id: dict[str, dict],
    *,
    task: str | None,
    run_dir: str,
    mode: str,
    write: bool,
) -> dict:
    if not _should_audit_contamination(task, mode):
        return metrics
    from lychee_mas.eval.evaluation.contamination import (
        attach_audit_to_metrics,
        attach_audit_to_trials,
        audit_run,
    )

    audits, summary = audit_run(
        run_dir,
        trials=trials,
        gold_by_id=gold_by_id,
        write=write,
    )
    attach_audit_to_trials(trials, audits)
    return attach_audit_to_metrics(metrics, trials, summary)


def _resolve_run_dir(args) -> str | None:
    return args.run_dir_opt or args.run_dir


@contextmanager
def _analysis_lock(run_dir: str):
    """Keep manual, incremental, and final scorers from racing on one Run."""

    path = Path(run_dir) / ".evaluation.lock"
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_analysis_status(run_dir: str, value: dict[str, Any]) -> None:
    path = Path(run_dir) / "evaluation_status.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _analyze_run(args, run_dir: str, config_path: str) -> dict:
    from lychee_mas.eval.evaluation.integrity import audit_run_event_integrity
    from lychee_mas.eval.evaluation.projections import build_result_projection
    from lychee_mas.runtime.events.store import run_events_path

    trials = build_result_projection(run_dir)
    terminal_trials = [
        trial for trial in trials if trial["trial_status"] in {"completed", "failed"}
    ]
    if not terminal_trials:
        raise SystemExit(f"No terminal Trial found in {run_events_path(run_dir)}")
    completed_trials = [
        trial for trial in terminal_trials if trial["trial_status"] == "completed"
    ]
    runtime_failed_trials = [
        trial for trial in terminal_trials if trial["trial_status"] == "failed"
    ]
    snapshot = _load_config(config_path)
    run_info = _run_info(snapshot, terminal_trials, args)
    task = run_info.get("task")
    kind = run_info.get("scorer_kind")
    if not task:
        raise SystemExit("Cannot infer task; pass --task")
    benchmark = get_benchmark(task)
    run_info["benchmark_id"] = benchmark.id
    start_index = int(run_info.get("start_index") or 0)

    # pass@K：一个 Case 可对应多个 Trial，按去重 Case 数加载 gold。
    gold_data, gold_by_id, gold_by_index = _load_gold_items(
        benchmark, task, start_index, terminal_trials
    )
    failed_pending = [
        trial
        for trial in runtime_failed_trials
        if trial.get("evaluation_status") in {"not_scored", "failed"}
        or trial.get("evaluation_trial_event_id") != trial.get("trial_event_id")
    ]
    failed_evaluated = _score_runtime_failed_trials(
        benchmark,
        failed_pending,
        gold_data,
        gold_by_id,
        gold_by_index,
        kind=kind,
    )
    if failed_evaluated and not args.no_write:
        write_evaluation_events(run_dir, failed_evaluated)
    in_memory_evaluated = list(failed_evaluated) if args.no_write else []
    pending_trials = completed_trials
    if args.incremental:
        pending_trials = [
            trial
            for trial in completed_trials
            if trial.get("evaluation_status") in {"not_scored", "failed"}
            or trial.get("evaluation_trial_event_id") != trial.get("trial_event_id")
        ]
    if pending_trials:
        try:
            benchmark.prepare_evaluation(
                pending_trials,
                gold_by_id,
                EvaluationContext(
                    run_dir=Path(run_dir),
                    run_info=run_info,
                    options=benchmark.evaluation_options(args),
                    external_evaluator=args.external_evaluator,
                ),
            )
        except Exception as exc:
            if not args.no_write:
                write_evaluation_events(run_dir, failed_evaluations(pending_trials, exc))
            if isinstance(exc, BenchmarkEvaluationError):
                raise SystemExit(str(exc)) from exc
            raise
        evaluated = _score_trials(
            benchmark,
            pending_trials,
            gold_data,
            gold_by_id,
            gold_by_index,
            kind=kind,
        )
        if not args.no_write:
            write_evaluation_events(run_dir, evaluated)
        else:
            in_memory_evaluated.extend(evaluated)

    # Rebuild after appending Evaluation Events so partial and final metrics use
    # the same canonical projection and include every Trial scored so far.
    refreshed = build_result_projection(run_dir)
    if in_memory_evaluated:
        evaluated_by_key = {
            (str(trial.get("case_id") or ""), int(trial.get("trial_index") or 0)): trial
            for trial in in_memory_evaluated
        }
        for trial in refreshed:
            evaluated_trial = evaluated_by_key.get(
                (str(trial.get("case_id") or ""), int(trial.get("trial_index") or 0))
            )
            if evaluated_trial is not None:
                trial.update(
                    evaluation_status="completed",
                    score=evaluated_trial["score"],
                    correct=evaluated_trial["correct"],
                    gold=evaluated_trial.get("gold"),
                    score_details=evaluated_trial["score_details"],
                )
    scored = [
        trial
        for trial in refreshed
        if trial.get("trial_status") in {"completed", "failed"}
        and trial.get("evaluation_status") == "completed"
    ]
    if not scored:
        raise SystemExit("No successfully evaluated Trial is available yet")
    metrics = _with_partial_flags(
        M.aggregate_trials(scored, run_info=run_info),
        run_info,
        len(scored),
    )
    failed_evaluation_rows = [
        trial
        for trial in refreshed
        if trial.get("trial_status") in {"completed", "failed"}
        and trial.get("evaluation_status") == "failed"
    ]
    metrics["benchmark_evaluation_failed_trials"] = len(failed_evaluation_rows)
    metrics["benchmark_evaluation_coverage"] = round(
        len(scored) / max(1, len(terminal_trials)), 6
    )
    metrics["run_event_integrity"] = audit_run_event_integrity(run_dir)
    metrics = _attach_vllm_service_metrics(metrics, run_dir)
    metrics = _attach_costing(metrics, scored, run_dir)
    metrics = _attach_contamination_audit(
        metrics,
        scored,
        gold_by_id,
        task=task,
        run_dir=run_dir,
        mode=args.contamination_audit,
        write=not args.no_write,
    )
    if not args.skip_evidence_normalization:
        metrics = _attach_evidence_coverage(
            metrics,
            run_dir,
            write=not args.no_write,
            evaluation_profile=args.evaluation_profile,
            evaluate_metrics=not args.skip_metric_evaluation,
        )
    if not args.no_write:
        M.write_metrics(run_dir, metrics)
        _write_analysis_status(
            run_dir,
            {
                "status": "partial" if metrics.get("is_partial") else "complete",
                "incremental": bool(args.incremental),
                "terminal_trials": len(terminal_trials),
                "completed_trials": len(completed_trials),
                "runtime_failed_trials": len(runtime_failed_trials),
                "evaluated_trials": len(scored),
                "failed_evaluation_trials": len(failed_evaluation_rows),
                "newly_evaluated_trials": len(pending_trials) + len(failed_pending),
            },
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score completed Trials from one benchmark Run without inference."
    )
    parser.add_argument(
        "run_dir", nargs="?", help="Run directory containing events/run_events.jsonl"
    )
    parser.add_argument("--run-dir", dest="run_dir_opt", default=None)
    parser.add_argument("--config", default=None, help="Explicit config.yaml/config.json path")
    parser.add_argument("--task", default=None, help="Override task used to load gold data")
    parser.add_argument("--kind", default=None, help="Override scorer kind")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Override dataset start index for gold alignment",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute and print analysis without appending evaluation events or artifacts",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Score only completed Trials that do not yet have a matching Evaluation Event",
    )
    parser.add_argument(
        "--contamination-audit",
        choices=("auto", "on", "off"),
        default="auto",
        help="污染审计模式；auto 仅对 GAIA 启用，on 强制启用，off 关闭",
    )
    parser.add_argument(
        "--external-evaluator",
        choices=("auto", "run", "skip"),
        default="auto",
        help="HLE/SWE-bench 官方外部评测器策略；auto 与 run 都执行所需评测器",
    )
    parser.add_argument(
        "--skip-evidence-normalization",
        action="store_true",
        help="Skip evidence normalization, metric applicability, and metric evaluation",
    )
    parser.add_argument(
        "--evaluation-profile",
        default="core",
        help="Evaluation Profile ID used for MetricObservation generation",
    )
    parser.add_argument(
        "--skip-metric-evaluation",
        action="store_true",
        help="Generate evidence and applicability but skip metric_observations.jsonl",
    )
    BENCHMARKS.add_analysis_arguments(parser)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args)
    if not run_dir:
        raise SystemExit("需要提供 run_dir")
    run_dir = os.path.abspath(run_dir)
    config_path = args.config or os.path.join(run_dir, "config.yaml")

    with _analysis_lock(run_dir):
        metrics = _analyze_run(args, run_dir, config_path)

    passk = metrics.get("pass_at_k")
    passk_str = ""
    if passk:
        requested_k = int(passk["requested_trials_per_case"])
        if passk.get("is_sampling_complete") and f"pass@{requested_k}" in passk:
            passk_str = (
                f" | pass@1={passk['pass@1']} pass@{requested_k}={passk[f'pass@{requested_k}']}"
            )
        else:
            passk_str = (
                f" | pass@1={passk['pass@1']} partial_observed_any_correct="
                f"{passk.get('observed_any_correct_rate')}"
            )
    bestk = metrics.get("best_of_k")
    if bestk:
        passk_str = (
            f" | mean_score={bestk['mean_score_per_trial']} "
            f"mean_best_of_k={bestk['mean_best_of_k_score']} "
            f"complete={bestk['is_sampling_complete']}"
        )
    print(
        f"[analyze] cases={metrics['num_cases']} score_mean={metrics['score_mean']} "
        f"model_calls/case={metrics['mean_model_calls_per_case']} "
        f"tool_calls/case={metrics['mean_tool_calls_per_case']}{passk_str} -> {run_dir}",
        flush=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
