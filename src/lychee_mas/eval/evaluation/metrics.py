"""评分 + 结果落盘（迁移自 benchs/metrics.py + math_parsing_util.py）。

评分类型（kind）：
  - "mc":    选择题；预测里出现 gold 选项字母 或 gold 文本即算对。用于 medqa/arc/openbookqa。
  - "exact": 数值/短答规整后精确匹配。用于 gsm8k。
  - "aime":  数学答案等价比较（数值 + sympy 符号），支持整数/分式/根式/带符号式子。用于 aime。
  - "f1":    token 级 F1（对 gold 列表取最大），MemGAS 口径。用于 locomo/longmemeval。
  - "human_eval": HumanEval 代码单测。
  - "gaia": GAIA 官方风格短答案规整。
  - "mas_audit": AFTraj safe/unsafe + decisive step/agent audit。
  - "mas_failure_taxonomy": MAST taxonomy label F1。
  - "mas_deviation": Open Agent Traces deviation detection。
  - "mas_instruction_decay" / "mas_tracer_durability" /
    "mas_consensus_pollution" / "mas_context_leakage": AgentCollabBench metrics。
  - "bbeh": BBEH 官方确定性答案抽取与 fuzzy exact match。
  - "hle": HLE 官方 judge 结果与置信度校准。
  - "swe_bench_verified": SWE-bench 官方 Docker harness resolved 状态。
  - "workbench": WorkBench 官方最终数据库状态与 harmful side-effect 指标。

CLAUDE.md §9：所有跑分必须同时报性能与成本。

⚠️ 惰性导入：`score_aime` 用到的 math_parsing_util
（依赖 sympy/regex/latex2sympy2-extended/word2number）与写 config 用到的 yaml
都在函数内部 import，保证本模块在无这些库时也可被 import（黄金法则 4）。
"""

from __future__ import annotations

import json
import math
import os
import re
import string
from collections import Counter
from typing import Any, List, Optional


def _benchmark_for_kind(kind: str):
    from ..benchmarks import BENCHMARKS

    for benchmark in BENCHMARKS.all():
        if kind in benchmark.score_handlers:
            return benchmark
    raise KeyError(f"unknown scorer kind {kind!r}")


def _binary_scorers() -> set[str]:
    from ..benchmarks import BENCHMARKS

    return {kind for benchmark in BENCHMARKS.all() for kind in benchmark.binary_kinds}


BINARY_SCORERS = _binary_scorers()


def is_binary_scorer(kind: str | None) -> bool:
    return bool(kind and kind in BINARY_SCORERS)


# ---- 规整 / 评分 ----
def _norm(s: str) -> str:
    # 统一小写、去标点、去冠词(a/an/the)、压空白——SQuAD 式规整，使匹配更鲁棒
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _last_number(s: str) -> Optional[str]:
    # 抽取字符串里最后一个数字（去千分位逗号）；数学题答案通常在末尾
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def score_exact(pred: str, gold: str) -> float:
    g = _last_number(gold) or _norm(gold)
    p = _last_number(pred)
    if p is not None and g is not None and p == g:
        return 1.0  # 数值匹配优先
    return 1.0 if _norm(gold) and _norm(gold) in _norm(pred) else 0.0  # 否则退化为子串包含


def score_mc(pred: str, gold_text: str, gold_letter: Optional[str]) -> float:
    raw_pred = str(pred).casefold()
    if gold_letter:
        # 在预测开头 40 字符内匹配独立的选项字母，如 "A" / "(A)" / "answer: b"
        if re.search(rf"\b{re.escape(gold_letter.casefold())}\b", raw_pred[:40]):
            return 1.0
    np_ = _norm(pred)
    return 1.0 if _norm(gold_text) and _norm(gold_text) in np_ else 0.0  # 或选项文本出现在预测里


def score_f1(pred: str, golds: List[str]) -> float:
    # token 级 F1，对多个 gold 取最大值（locomo 一题可有多个可接受答案）
    pt = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)  # 词频交集 = 命中词数
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_aime(pred: str, gold: str) -> float:
    """AIME/数学评分：两边各自归一化后做数学等价比较（数值 + sympy 符号）。

    math_parsing_util 在此惰性导入（它依赖 sympy/regex/latex2sympy2-extended/word2number）。
    """
    # 数学答案抽取/等价比较工具（math_parsing_util 里函数名无 math_ 前缀）
    from ..benchmarks.math_parsing import extract_answer, math_equal, strip_answer_string

    def _math_norm(s: str) -> str:
        r"""归一化数学答案。

        含 \boxed/"answer is" 标记时用 extract_answer 抽取（它能正确取出 \boxed{} 里的表达式）；
        否则只做 strip_answer_string 归一化。aime 提取器已剥掉 \boxed，pred 多是裸表达式，故走
        strip 分支。
        """
        s = str(s)
        return extract_answer(s) if "boxed" in s else strip_answer_string(s)

    p = _math_norm(pred)
    g = _math_norm(gold)
    return 1.0 if math_equal(p, g) else 0.0


def score_gaia(pred: str, gold: str) -> float:
    from ..benchmarks.gaia import gaia_question_scorer

    return 1.0 if gaia_question_scorer(str(pred), str(gold)) else 0.0


def score_human_eval(pred: str, gold) -> float:
    from ..benchmarks.human_eval import evaluate_answer

    if not isinstance(gold, dict) or "test" not in gold or "entry_point" not in gold:
        return 0.0
    return 1.0 if evaluate_answer(pred, gold).get("success") else 0.0


def score_bbeh(pred: str, gold: str) -> float:
    from ..benchmarks.bbeh import evaluate_correctness

    return 1.0 if evaluate_correctness(str(pred), str(gold)) else 0.0


def score_mas_audit(pred: str, gold) -> float:
    from ..benchmarks.aftraj import score_audit

    return score_audit(pred, gold if isinstance(gold, dict) else {})


def score_mas_failure_taxonomy(pred: str, gold) -> float:
    from ..benchmarks.mast_data import score_taxonomy

    return score_taxonomy(pred, gold if isinstance(gold, dict) else {"labels": gold})


def score_mas_deviation(pred: str, gold) -> float:
    from ..benchmarks.open_agent_traces import score_deviation

    return score_deviation(pred, gold if isinstance(gold, dict) else {})


def score_agent_collab(pred: str, gold, metric: str | None = None) -> float:
    from ..benchmarks.agent_collab import score_agent_collab as _score_agent_collab

    return _score_agent_collab(
        pred, gold if isinstance(gold, dict) else {"expected": gold}, metric=metric
    )


def score(kind: str, pred: str, gold) -> float:
    return float(score_details(kind, pred, gold).get("score", 0.0))


def score_details(kind: str, pred: str, gold, record: dict | None = None) -> dict:
    """Compatibility entry point for callers that only have a scorer kind."""

    benchmark = _benchmark_for_kind(kind)
    return benchmark.score(pred, {"kind": kind, "gold": gold}, record=record)


# ---- run-level 汇总 ----
def _trial_number(trial: dict[str, Any], *names: str, default=0):
    for name in names:
        value = trial.get(name)
        if value is not None:
            return value
    return default


def _trial_count(trial: dict[str, Any], *names: str, list_field: str | None = None) -> int:
    value = _trial_number(trial, *names, default=None)
    if value is not None:
        return int(value)
    if list_field:
        items = trial.get(list_field)
        if isinstance(items, list):
            return len(items)
    return 0


def aggregate_trials(trials: List[dict], run_info: Optional[dict] = None) -> dict:
    """Aggregate joined Trial and Evaluation projections into Run metrics."""
    run_info = run_info or {}
    n = len(trials)
    by_case_records: dict[str, list[dict]] = {}
    for index, trial in enumerate(trials):
        case_id = str(trial.get("case_id") or f"__missing_case_id_{index}")
        by_case_records.setdefault(case_id, []).append(trial)
    distinct_cases = len(by_case_records)
    first = trials[0] if trials else {}
    kind = run_info.get("scorer_kind") or first.get("scorer_kind") or first.get("kind")
    task = run_info.get("task") or first.get("task")
    method = run_info.get("method") or first.get("method")

    score_sum = 0.0
    num_correct = 0
    model_call_count_sum = 0
    message_count_sum = 0
    input_positions_sum = 0
    text_input_tokens_sum = 0
    latent_prefix_positions_sum = 0
    output_tokens_sum = 0
    generation_latency_sum = 0.0
    trial_wall_time_sum = 0.0
    tool_call_count_sum = 0
    tool_error_count_sum = 0
    tool_request_count_sum = 0
    tool_execution_count_sum = 0
    tool_agent_error_count_sum = 0
    timing_fields = (
        "client_rate_limiter_wait_s",
        "client_http_request_latency_s",
        "client_response_postprocess_latency_s",
        "client_model_call_wall_time_s",
        "provider_request_queue_latency_s",
        "provider_scheduled_to_first_token_s",
        "provider_generation_latency_s",
        "provider_mean_inter_token_latency_s",
        "provider_output_tokens_per_second",
    )
    timing_sums: dict[str, float] = {field: 0.0 for field in timing_fields}
    timing_counts: dict[str, int] = {field: 0 for field in timing_fields}

    for trial in trials:
        score_value = float(trial.get("score", trial.get("correct", 0.0)) or 0.0)
        score_sum += score_value
        if is_binary_scorer(kind):
            is_correct = trial.get("is_correct")
            num_correct += int(bool(is_correct) if is_correct is not None else score_value == 1.0)

        model_call_count_sum += _trial_count(
            trial, "model_call_count", list_field="model_calls"
        )
        message_count_sum += _trial_count(
            trial, "message_count", list_field="messages"
        )
        input_positions_sum += int(
            _trial_number(trial, "input_tokens")
        )
        text_input_tokens_sum += int(
            _trial_number(trial, "input_text_tokens", "input_tokens")
        )
        latent_prefix_positions_sum += int(
            _trial_number(trial, "input_latent_positions")
        )
        output_tokens_sum += int(
            _trial_number(trial, "output_tokens")
        )
        generation_latency_sum += float(
            _trial_number(trial, "model_latency_s")
        )
        trial_wall_time_sum += float(_trial_number(trial, "trial_wall_time_s"))
        tool_call_count_sum += _trial_count(
            trial, "tool_call_count", list_field="tool_calls"
        )
        tool_error_count_sum += int(_trial_number(trial, "tool_error_count"))
        tool_request_count_sum += _trial_count(
            trial, "tool_call_count", list_field="tool_requests"
        )
        tool_execution_count_sum += _trial_count(
            trial,
            "tool_execution_count",
            list_field="tool_executions",
        )
        tool_agent_error_count_sum += _trial_count(
            trial,
            "tool_agent_error_count",
            list_field="tool_agent_errors",
        )
        for field in timing_fields:
            available = int(trial.get(f"{field}_available_calls") or 0)
            value_sum = trial.get(f"{field}_sum")
            if available and isinstance(value_sum, (int, float)):
                timing_counts[field] += available
                timing_sums[field] += float(value_sum)

    score_mean = round(score_sum / n, 4) if n else None
    case_divisor = distinct_cases or 1
    metrics = {
        "schema_version": 2,
        "model": run_info.get("model"),
        "backend_provider": run_info.get("backend_provider"),
        "method": method,
        "team": run_info.get("team"),
        "task": task,
        "benchmark_id": run_info.get("benchmark_id") or first.get("benchmark_id"),
        "probe": run_info.get("probe", "mas"),
        "memory": run_info.get("memory"),
        "router": run_info.get("router"),
        "summary_scope": run_info.get("summary_scope", "run"),
        "aggregation_scope": "trial_and_distinct_case_aggregates",
        "num_trials": n,
        "trial_count": n,
        "num_distinct_cases": distinct_cases,
        "num_cases": distinct_cases,
        "case_count": distinct_cases,
        "scorer_kind": kind,
        "mean_score": score_mean,
        "score_mean": score_mean,
        "accuracy": score_mean if is_binary_scorer(kind) else None,
        "num_correct_trials": num_correct if is_binary_scorer(kind) else None,
        "num_correct": num_correct if is_binary_scorer(kind) else None,
        "mean_model_calls_per_trial": round(model_call_count_sum / n, 2) if n else 0,
        "mean_messages_per_trial": round(message_count_sum / n, 2) if n else 0,
        "mean_input_total_positions_per_trial": round(input_positions_sum / n, 1) if n else 0,
        "mean_input_text_tokens_per_trial": round(text_input_tokens_sum / n, 1) if n else 0,
        "mean_input_latent_positions_per_trial": round(
            latent_prefix_positions_sum / n, 1
        ) if n else 0,
        "mean_output_text_tokens_per_trial": round(output_tokens_sum / n, 1) if n else 0,
        "mean_model_latency_s_per_trial": round(generation_latency_sum / n, 3) if n else 0,
        "mean_model_calls_per_case": round(model_call_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_messages_per_case": round(message_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_input_total_positions_per_case": round(input_positions_sum / case_divisor, 1)
        if distinct_cases else 0,
        "mean_input_text_tokens_per_case": round(text_input_tokens_sum / case_divisor, 1)
        if distinct_cases else 0,
        "mean_input_latent_positions_per_case": round(
            latent_prefix_positions_sum / case_divisor, 1
        ) if distinct_cases else 0,
        "mean_output_text_tokens_per_case": round(output_tokens_sum / case_divisor, 1)
        if distinct_cases else 0,
        "mean_model_latency_s_per_case": round(generation_latency_sum / case_divisor, 3)
        if distinct_cases else 0,
        "mean_model_latency_s_per_call": round(generation_latency_sum / model_call_count_sum, 3)
        if model_call_count_sum
        else 0,
        "mean_trial_wall_time_s": round(trial_wall_time_sum / n, 3) if n else 0,
        "mean_aggregate_trial_wall_time_s_per_case": round(
            trial_wall_time_sum / case_divisor, 3
        ) if distinct_cases else 0,
        "mean_tool_calls_per_case": round(tool_call_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_tool_errors_per_case": round(tool_error_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_tool_requests_per_case": round(tool_request_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_tool_executions_per_case": round(tool_execution_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "mean_tool_agent_errors_per_case": round(tool_agent_error_count_sum / case_divisor, 2)
        if distinct_cases else 0,
        "total_model_calls": model_call_count_sum,
        "total_messages": message_count_sum,
        "total_tool_calls": tool_call_count_sum,
        "total_tool_errors": tool_error_count_sum,
        "total_tool_requests": tool_request_count_sum,
        "total_tool_executions": tool_execution_count_sum,
        "total_tool_agent_errors": tool_agent_error_count_sum,
        "total_input_total_positions": input_positions_sum,
        "total_input_text_tokens": text_input_tokens_sum,
        "total_input_latent_positions": latent_prefix_positions_sum,
        "total_output_text_tokens": output_tokens_sum,
        "total_model_latency_s": round(generation_latency_sum, 3),
        "total_trial_wall_time_s": round(trial_wall_time_sum, 3),
        "tool_call_count": tool_call_count_sum,
        "tool_error_count": tool_error_count_sum,
        "tool_request_count": tool_request_count_sum,
        "tool_execution_count": tool_execution_count_sum,
        "tool_agent_error_count": tool_agent_error_count_sum,
    }
    for field in timing_fields:
        count = timing_counts[field]
        value_sum = timing_sums[field]
        metrics[f"{field}_available_calls"] = count
        metrics[f"{field}_mean_per_available_call"] = (
            round(value_sum / count, 6) if count else None
        )
        if field != "provider_output_tokens_per_second":
            metrics[f"{field}_total"] = round(value_sum, 6) if count else None

    # ---- 多采样：二值 scorer 报标准 pass@k；连续 scorer 报 best-of-k score ----
    if distinct_cases and n > distinct_cases:
        scores_by_case = {
            case_id: [
                float(record.get("score", record.get("correct", 0.0)) or 0.0)
                for record in records
            ]
            for case_id, records in by_case_records.items()
        }
        requested_values = [
            int(record.get("trials_per_case", 0) or 0)
            for record in trials
            if int(record.get("trials_per_case", 0) or 0) > 0
        ]
        requested_k = max(requested_values, default=max(len(v) for v in scores_by_case.values()))
        trial_counts = {case_id: len(values) for case_id, values in scores_by_case.items()}
        complete = all(count == requested_k for count in trial_counts.values())
        sampling = {
            "num_distinct_cases": distinct_cases,
            "requested_trials_per_case": requested_k,
            "completed_trials_per_case": trial_counts,
            "mean_completed_trials_per_case": round(n / distinct_cases, 2),
            "min_completed_trials_per_case": min(trial_counts.values()),
            "max_completed_trials_per_case": max(trial_counts.values()),
            "is_sampling_complete": complete,
        }
        metrics["sampling"] = sampling
        metrics.update({
            "requested_trials_per_case": requested_k,
            "trials_per_case": sampling["mean_completed_trials_per_case"],
            "max_trials_per_case": sampling["max_completed_trials_per_case"],
            "is_sampling_complete": complete,
        })

        if is_binary_scorer(kind):
            passk_block: dict[str, Any] = {
                **sampling,
                "pass@1": round(
                    sum(sum(values) / len(values) for values in scores_by_case.values())
                    / distinct_cases,
                    4,
                ),
            }
            if complete:
                for k in range(1, requested_k + 1):
                    estimates = []
                    for values in scores_by_case.values():
                        trial_n = len(values)
                        correct_n = sum(1 for value in values if value == 1.0)
                        miss_probability = (
                            math.comb(trial_n - correct_n, k) / math.comb(trial_n, k)
                            if trial_n - correct_n >= k
                            else 0.0
                        )
                        estimates.append(1.0 - miss_probability)
                    passk_block[f"pass@{k}"] = round(sum(estimates) / len(estimates), 4)
                metrics.update({
                    key: value for key, value in passk_block.items() if key.startswith("pass@")
                })
            else:
                passk_block["observed_any_correct_rate"] = round(
                    sum(1.0 if any(value == 1.0 for value in values) else 0.0
                        for values in scores_by_case.values()) / distinct_cases,
                    4,
                )
            metrics["pass_at_k"] = passk_block
        else:
            best_block = {
                **sampling,
                "mean_score_per_trial": score_mean,
                "mean_best_of_k_score": round(
                    sum(max(values) for values in scores_by_case.values()) / distinct_cases,
                    4,
                ),
            }
            metrics["best_of_k"] = best_block
            metrics["mean_best_of_k_score"] = best_block["mean_best_of_k_score"]
    if task:
        from ..benchmarks import get_benchmark

        metrics = get_benchmark(str(task)).aggregate(trials, metrics)
    return metrics


# ---- 落盘 ----
def result_dir(
    model: str,
    method: str,
    task: str,
    root: Optional[str] = None,
    *,
    team: Optional[str] = None,
) -> str:
    # Benchmark runner 使用 root/model/team/method/task；旧调用未传 team 时保持四层布局。
    if root is None:
        from ..benchmarks.common import runs_root

        root = runs_root()
    d = os.path.join(root, model, *([team] if team else []), method, task)
    os.makedirs(d, exist_ok=True)
    return d


def write_metrics(out_dir: str, metrics: dict) -> None:
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)  # Pareto 主图的数据源


def write_config(out_dir: str, config: dict) -> None:
    _dump_config(os.path.join(out_dir, "config.yaml"), config)




def _dump_config(path: str, config: dict) -> None:
    """落配置快照：优先 yaml（可读），无 yaml 时退回 JSON（保证离线也能落盘）。"""
    try:
        import yaml  # 惰性导入

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
