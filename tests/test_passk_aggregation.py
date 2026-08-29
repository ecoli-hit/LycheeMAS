"""pass@K aggregation tests for independent Trials."""

from lychee_mas.eval.evaluation.metrics import aggregate_trials


def _trial(case_id, score, trial_index, trials_per_case, kind="exact"):
    return {
        "case_id": case_id, "score": score,
        "is_correct": score == 1.0, "scorer_kind": kind,
        "trial_index": trial_index, "trials_per_case": trials_per_case,
    }


def test_passk_binary_multi_sample():
    # case A: [1,0,0] -> mean=1/3, best=1 ; case B: [0,0,0] -> mean=0, best=0
    trials = [
        _trial("A", 1.0, 0, 3), _trial("A", 0.0, 1, 3), _trial("A", 0.0, 2, 3),
        _trial("B", 0.0, 0, 3), _trial("B", 0.0, 1, 3), _trial("B", 0.0, 2, 3),
    ]
    m = aggregate_trials(trials, run_info={"scorer_kind": "exact", "task": "aime_2024"})
    pk = m["pass_at_k"]
    assert pk["num_distinct_cases"] == 2
    assert pk["max_completed_trials_per_case"] == 3
    assert pk["mean_completed_trials_per_case"] == 3.0
    assert pk["is_sampling_complete"] is True
    assert pk["pass@1"] == round((1 / 3 + 0) / 2, 4)   # 0.1667
    assert pk["pass@3"] == 0.5                          # (1 + 0) / 2
    # pass_at_k 字段同时展开到顶层，便于旧读取器直接取
    assert m["pass@1"] == pk["pass@1"] and m["pass@3"] == pk["pass@3"]
    assert m["num_trials"] == 6
    assert m["num_distinct_cases"] == 2
    assert m["num_cases"] == 2


def test_passk_absent_for_single_sample():
    # 每个 case 恰好 1 份 -> 不写 pass_at_k，其余指标与旧格式一致
    trials = [
        {"case_id": "A", "score": 1.0, "is_correct": True, "scorer_kind": "exact"},
        {"case_id": "B", "score": 0.0, "is_correct": False, "scorer_kind": "exact"},
    ]
    m = aggregate_trials(trials, run_info={"scorer_kind": "exact"})
    assert "pass_at_k" not in m
    assert "pass@1" not in m
    assert m["score_mean"] == 0.5
    assert m["num_cases"] == 2


def test_passk_continuous_scorer_best_of_k():
    # f1 等连续打分：pass@K 取 best-of-K 均值，pass@1 取均值的均值
    trials = [
        _trial("A", 0.4, 0, 2, kind="f1"), _trial("A", 0.9, 1, 2, kind="f1"),
        _trial("B", 0.2, 0, 2, kind="f1"), _trial("B", 0.0, 1, 2, kind="f1"),
    ]
    m = aggregate_trials(trials, run_info={"scorer_kind": "f1"})
    assert "pass_at_k" not in m
    best = m["best_of_k"]
    assert best["mean_score_per_trial"] == 0.375
    assert best["mean_best_of_k_score"] == round((0.9 + 0.2) / 2, 4)


def test_passk_uneven_trials_per_case():
    # 断点续跑可能导致每个 case 采样数不齐：按各自 K 归一，max_samples 取最大
    trials = [
        _trial("A", 1.0, 0, 3), _trial("A", 0.0, 1, 3),
        _trial("B", 0.0, 0, 3), _trial("B", 0.0, 1, 3), _trial("B", 1.0, 2, 3),
    ]
    m = aggregate_trials(trials, run_info={"scorer_kind": "exact"})
    pk = m["pass_at_k"]
    assert pk["num_distinct_cases"] == 2
    assert pk["max_completed_trials_per_case"] == 3
    assert pk["is_sampling_complete"] is False
    assert pk["pass@1"] == round((0.5 + 1 / 3) / 2, 4)   # (A:1/2, B:1/3) 均值
    assert "pass@3" not in pk
    assert pk["observed_any_correct_rate"] == 1.0
