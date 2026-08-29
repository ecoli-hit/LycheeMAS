"""Contracts for the public prepare/load/score benchmark API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.eval import benchmarks
from lychee_mas.eval.benchmarks import (
    BENCHMARKS,
    STANDARD_SOURCE_PROVIDERS,
    Benchmark,
    EvaluationContext,
    get_benchmark,
    hle,
    swe_bench_verified,
)
from lychee_mas.eval.benchmarks.task_config import TASK_CONFIG


def _record(*, task: str = "toy", kind: str = "exact") -> dict:
    return {
        "task": task,
        "kind": kind,
        "question": "What is 1 + 1?",
        "gold": "2",
        "context": None,
        "metadata": {"task_id": "toy-1"},
    }


def _toy_benchmark(loader) -> Benchmark:
    return Benchmark(
        benchmark_id="toy",
        full_prepare_target="toy",
        prepare_handlers={"toy": lambda force=False, source=None: "/tmp/toy"},
        loaders={"toy": loader},
        scorer_kinds={"toy": "exact"},
        score_handlers={
            "exact": lambda prediction, gold, record: {"score": 1.0 if prediction == gold else 0.0}
        },
        binary_kinds=("exact",),
    )


def test_benchmark_load_normalizes_cases_and_score_is_direct():
    benchmark = _toy_benchmark(lambda n=None: [_record()])

    case = benchmark.load(n=1)[0]

    assert case["task"] == "toy"
    assert case["kind"] == "exact"
    assert case["metadata"]["benchmark_id"] == "toy"
    assert benchmark.score("2", case)["score"] == 1.0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"task": "toy", "kind": "exact", "question": "q"}, "without gold"),
        (_record(task="other"), "returned task='other'"),
        (_record(kind="f1"), "returned kind='f1'"),
    ],
)
def test_benchmark_rejects_loader_contract_drift(record: dict, message: str):
    benchmark = _toy_benchmark(lambda n=None: [record])

    with pytest.raises(ValueError, match=message):
        benchmark.load(n=1)


def test_public_views_are_derived_from_the_benchmark_registry():
    assert set(benchmarks.LOADERS) == set(BENCHMARKS.tasks())
    assert set(benchmarks.PREPARERS) == set(BENCHMARKS.prepare_targets())
    assert set(benchmarks.FULL_PREPARE_TARGETS) == {
        benchmark.full_prepare_target for benchmark in BENCHMARKS.all()
    }
    assert TASK_CONFIG == {
        task: {"extractor": benchmark.extractor_names[task]}
        for benchmark in BENCHMARKS.all()
        for task in benchmark.runnable_tasks
    }


def test_registry_agrees_with_public_structure():
    structure = {row["benchmark_source"]: row for row in benchmarks.BENCHMARK_STRUCTURE}

    for benchmark in BENCHMARKS.all():
        row = structure[benchmark.name]
        assert row["full_prepare_target"] == benchmark.full_prepare_target
        assert set(row["prepare_targets"]) == set(benchmark.accepted_prepare_targets)
        assert set(row["runnable_tasks"]) == set(benchmark.runnable_tasks)
        assert set(row["kinds"]) == set(benchmark.scorer_kinds.values())
        assert set(benchmark.scorer_kinds.values()) <= set(benchmark.score_handlers)


def test_benchmark_owns_sources_and_honors_provider_env(monkeypatch):
    benchmark = BENCHMARKS.get("gsm8k")

    monkeypatch.setenv("LYCHEE_GSM8K_HF_ID", "organization/private-gsm8k")

    assert benchmark.provider_ids("huggingface") == [
        "organization/private-gsm8k",
        "openai/gsm8k",
        "gsm8k",
    ]
    assert benchmark.source_backend_order("huggingface") == ["huggingface"]
    assert benchmark.sources["huggingface"]["default_ids"] == [
        "openai/gsm8k",
        "gsm8k",
    ]


def test_registered_benchmarks_share_three_standard_source_providers():
    for benchmark in BENCHMARKS.all():
        assert set(STANDARD_SOURCE_PROVIDERS) <= set(benchmark.sources)

    assert BENCHMARKS.get("bbeh").source_backend_order("auto") == ["github"]
    assert BENCHMARKS.get("human_eval").source_backend_order("auto") == [
        "modelscope",
        "huggingface",
        "github",
    ]
    with pytest.raises(ValueError, match="has no registered github source"):
        BENCHMARKS.get("gsm8k").source_backend_order("github")


def test_benchmark_descriptor_owns_runtime_scoring_and_capability_contracts():
    gaia = BENCHMARKS.get("gaia").descriptor()
    bbeh = BENCHMARKS.get("bbeh").descriptor()
    swe_bench = BENCHMARKS.get("swe_bench_verified").descriptor()
    workbench = BENCHMARKS.get("workbench").descriptor()

    assert gaia["runtime_defaults"]["docker_image"] == "lychee-agbench-gaia:local"
    assert gaia["network_defaults"]["access"] == "required"
    assert gaia["capabilities"]["browser_required"] is True
    assert gaia["task_contracts"]["gaia_validation"]["kind"] == "gaia"
    assert bbeh["task_contracts"]["bbeh"]["scoring"]["profiles"]["official"]["parameters"] == {
        "evaluator": "pinned_official_deterministic"
    }
    assert bbeh["runtime_defaults"]["max_rounds"] == 1
    assert swe_bench["runtime_defaults"]["max_rounds"] == 50
    assert workbench["runtime_defaults"]["max_rounds"] == 1
    assert workbench["capabilities"]["required"] == ["text_generation", "tool_calls"]


def test_benchmark_declares_scorer_and_answer_extractor():
    benchmark = get_benchmark("aime_2024")
    messages = [{"type": "TextMessage", "content": r"APPROVE: \boxed{42}"}]

    assert benchmark.scorer_kinds["aime_2024"] == "aime"
    assert benchmark.extract_messages("aime_2024", messages) == "42"


def test_default_extractor_accepts_structured_autogen_content():
    benchmark = get_benchmark("swe_bench_verified")
    messages = [
        {
            "type": "ToolCallSummaryMessage",
            "content": [
                {"type": "TextMessage", "text": "patch prepared"},
                {"tool": "runtime:python_code", "status": "completed"},
            ],
        }
    ]

    extracted = benchmark.extract_messages("swe_bench_verified", messages)

    assert "patch prepared" in extracted
    assert "runtime:python_code" in extracted


def test_analysis_arguments_are_registered_by_the_owning_benchmarks():
    parser = argparse.ArgumentParser()
    BENCHMARKS.add_analysis_arguments(parser)

    args = parser.parse_args([])

    assert args.swebench_max_workers == 4
    assert args.swebench_timeout == 1800
    assert args.hle_judge_workers == 8
    assert args.hle_judge_auth_mode == "env"


def test_hle_official_judge_is_owned_by_hle_benchmark(monkeypatch, tmp_path):
    called = {}

    def fake_prepare(predictions, gold_by_id, run_dir, **options):
        called.update({"gold": gold_by_id, "run_dir": run_dir, "options": options})
        predictions[0]["hle_judge_response"] = {"correct": "yes", "confidence": 90}

    monkeypatch.setattr(hle, "prepare_judge_responses", fake_prepare)
    benchmark = get_benchmark("hle")
    predictions = [{"case_id": "hle-1", "final_answer": "answer"}]
    gold = {"hle-1": {"question": "question", "gold": "answer"}}

    benchmark.prepare_evaluation(
        predictions,
        gold,
        EvaluationContext(
            run_dir=tmp_path,
            run_info={"model": "judge-target"},
            options={
                "model": "judge",
                "base_url": "http://judge.test/v1",
                "auth_mode": "env",
                "api_key_env": "JUDGE_KEY",
                "api_key": None,
                "workers": 2,
                "timeout": 30.0,
                "max_tokens": 512,
            },
        ),
    )

    assert called["run_dir"] == tmp_path
    assert called["options"]["model"] == "judge"
    assert called["options"]["auth_mode"] == "env"
    assert predictions[0]["hle_judge_response"]["correct"] == "yes"


def test_swebench_official_harness_is_owned_by_swebench(monkeypatch, tmp_path):
    called = {}

    def fake_attach(predictions, run_dir, **options):
        called.update({"run_dir": run_dir, "options": options})
        predictions[0]["swebench_harness"] = {"status": "completed", "resolved": True}

    monkeypatch.setattr(swe_bench_verified, "attach_official_harness_results", fake_attach)
    benchmark = get_benchmark("swe_bench_verified")
    predictions = [{"case_id": "repo__issue", "final_answer": "diff --git ..."}]

    benchmark.prepare_evaluation(
        predictions,
        {},
        EvaluationContext(
            run_dir=tmp_path,
            run_info={"model": "model-under-test"},
            options={"max_workers": 3, "timeout": 120},
        ),
    )

    assert called["run_dir"] == tmp_path
    assert called["options"]["model_name"] == "model-under-test"
    assert called["options"]["max_workers"] == 3
    assert called["options"]["timeout"] == 120
    assert called["options"]["image_refs_by_case"] == {}
    assert isinstance(
        called["options"]["image_cache"],
        swe_bench_verified.ProjectDockerImageCache,
    )
    assert predictions[0]["swebench_harness"]["resolved"] is True


def test_swebench_harness_uses_prepared_parquet_without_remote_dataset(monkeypatch, tmp_path):
    prepared = tmp_path / "prepared"
    dataset = prepared / "dataset/test-00000-of-00001.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.touch()
    (prepared / "harness").mkdir()
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    predictions = tmp_path / "predictions.jsonl"
    predictions.touch()
    called = {}

    def fake_run(command, **kwargs):
        called["command"] = command
        called["cwd"] = kwargs["cwd"]
        (report_dir / "model.run-local.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(swe_bench_verified, "ensure_source", lambda: prepared)
    monkeypatch.setattr(swe_bench_verified.subprocess, "run", fake_run)
    monkeypatch.setattr(swe_bench_verified, "_remove_harness_containers", lambda _run_id: [])

    swe_bench_verified.run_official_harness(
        predictions,
        report_dir,
        run_id="run-local",
    )

    dataset_index = called["command"].index("--dataset_name") + 1
    assert called["command"][dataset_index] == str(dataset)
    assert called["command"][dataset_index] != swe_bench_verified.dataset_id()
    predictions_index = called["command"].index("--predictions_path") + 1
    assert Path(called["command"][predictions_index]).is_absolute()
    assert called["cwd"] == report_dir.resolve()


def test_swebench_harness_always_cleans_its_run_containers(monkeypatch, tmp_path):
    prepared = tmp_path / "prepared"
    dataset = prepared / "dataset/test-00000-of-00001.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.touch()
    (prepared / "harness").mkdir()
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    predictions = tmp_path / "predictions.jsonl"
    predictions.touch()
    cleaned = []

    monkeypatch.setattr(swe_bench_verified, "ensure_source", lambda: prepared)
    monkeypatch.setattr(
        swe_bench_verified.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("harness failed")),
    )
    monkeypatch.setattr(
        swe_bench_verified,
        "_remove_harness_containers",
        lambda run_id: cleaned.append(run_id) or [],
    )

    with pytest.raises(RuntimeError, match="harness failed"):
        swe_bench_verified.run_official_harness(
            predictions,
            report_dir,
            run_id="unique-run-id",
        )

    assert cleaned == ["unique-run-id"]


def test_swebench_official_harness_batches_images_for_bounded_cleanup(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(predictions_path, _report_dir, **options):
        rows = [
            json.loads(line)
            for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        ]
        calls.append({"rows": rows, "options": options})
        return {
            "resolved_ids": [row["instance_id"] for row in rows],
            "infra_failure_ids": [],
            "incomplete_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "failure_reasons": {},
        }

    monkeypatch.setattr(swe_bench_verified, "run_official_harness", fake_run)
    predictions = [
        {"case_id": f"case-{index}", "prediction": f"patch-{index}"}
        for index in range(5)
    ]
    cache = SimpleNamespace(policy=SimpleNamespace(max_images=2))

    swe_bench_verified.attach_official_harness_results(
        predictions,
        tmp_path,
        model_name="test-model",
        max_workers=4,
        timeout=120,
        image_refs_by_case={f"case-{index}": f"image-{index}" for index in range(5)},
        image_cache=cache,
    )

    assert [len(call["rows"]) for call in calls] == [2, 2, 1]
    assert [call["options"]["image_refs"] for call in calls] == [
        ["image-0", "image-1"],
        ["image-2", "image-3"],
        ["image-4"],
    ]
    assert all(item["swebench_harness"]["resolved"] for item in predictions)
    aggregate = list(
        (tmp_path / "official_evaluation/swe_bench_verified").glob("aggregate.*.json")
    )
    assert len(aggregate) == 1
    assert json.loads(aggregate[0].read_text(encoding="utf-8"))["parts"] == 3


def test_humaneval_prediction_collection_is_owned_by_humaneval():
    benchmark = get_benchmark("human_eval")
    messages = [
        SimpleNamespace(source="Coder", content="```python\ndef add(a, b): return a + b\n```"),
        SimpleNamespace(source="ComputerTerminal", content="Tests passed"),
    ]

    prediction = benchmark.collect_prediction(
        query=None,
        messages=messages,
        workspace=None,
        default_text="Tests passed",
    )

    assert prediction.startswith("```python")
