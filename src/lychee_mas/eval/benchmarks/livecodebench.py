"""LiveCodeBench release-v6 code-generation adapter.

Inference remains framework-neutral. Scoring is delegated to a pinned checkout
of the official LiveCodeBench checker so that private tests, timeout behavior,
and pass/fail semantics do not silently drift into a LycheeMAS reimplementation.
"""

from __future__ import annotations

import base64
import json
import pickle
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import Benchmark, BenchmarkEvaluationError, EvaluationContext
from .common import (
    clone_git_source,
    conversion_only,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
    raw_source_matches,
    remove_path,
)
from .registry import register_benchmark

DATASET_ID = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_VERSION = "release_v6"
DATA_FILENAME = "test6.jsonl"
EVALUATOR_REPO = "https://github.com/LiveCodeBench/LiveCodeBench.git"
EVALUATOR_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
EXPECTED_CASES = 175

SOURCES: dict[str, Any] = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {
        "env": "LYCHEE_LIVECODEBENCH_HF_ID",
        "default_ids": [DATASET_ID],
        "revision": DATASET_REVISION,
    },
    "github": {"env": None, "default_ids": []},
    "other_defaults": [
        {
            "provider": "github",
            "id": EVALUATOR_REPO,
            "revision": EVALUATOR_REVISION,
            "selectable": False,
            "purpose": "pinned official prompt extraction and code-generation checker",
        }
    ],
    "fallback_files": [],
}


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("livecodebench")


def _prepared_file(root: str | Path | None = None) -> Path:
    return source_dir(root) / DATA_FILENAME


def _contract_path(root: str | Path | None = None) -> Path:
    return source_dir(root) / "adapter_contract.json"


def _raw_dataset() -> Path:
    return raw_source_dir("livecodebench", "huggingface", DATASET_ID)


def _evaluator_checkout() -> Path:
    return raw_source_dir("livecodebench", "github", EVALUATOR_REPO)


def _contract() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "release_version": RELEASE_VERSION,
        "data_filename": DATA_FILENAME,
        "expected_cases": EXPECTED_CASES,
        "evaluator_repository": EVALUATOR_REPO,
        "evaluator_revision": EVALUATOR_REVISION,
    }


def _ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        contract = json.loads(_contract_path(path.parent).read_text(encoding="utf-8"))
        if contract != _contract():
            return False
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    required = {
        "question_content",
        "question_id",
        "contest_date",
        "public_test_cases",
        "private_test_cases",
        "metadata",
    }
    return len(rows) == EXPECTED_CASES and required.issubset(rows[0])


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "huggingface"}:
        raise ValueError("LiveCodeBench release-v6 data is registered through Hugging Face")
    destination = _prepared_file(root)
    evaluator = _evaluator_checkout()
    if _ready(destination) and evaluator.is_dir() and not force_download:
        result = subprocess.run(
            ["git", "-C", str(evaluator), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == EVALUATOR_REVISION:
            return source_dir(root)

    raw = _raw_dataset()
    raw_ready = raw_source_matches(
        raw,
        provider="huggingface",
        source_id=DATASET_ID,
        revision=DATASET_REVISION,
    ) and (raw / DATA_FILENAME).is_file()
    if conversion_only() and not raw_ready:
        raise FileNotFoundError(
            f"LiveCodeBench raw source does not match {DATASET_REVISION}: {raw}"
        )
    if not conversion_only() and (force_download or not raw_ready):
        from huggingface_hub import hf_hub_download

        if force_download:
            remove_path(raw)
        log_download_source(
            "livecodebench",
            "huggingface",
            DATASET_ID,
            raw,
            revision=DATASET_REVISION,
        )
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=DATA_FILENAME,
            revision=DATASET_REVISION,
            local_dir=str(raw),
        )

    clone_git_source(
        "livecodebench",
        EVALUATOR_REPO,
        evaluator,
        revision=EVALUATOR_REVISION,
        force=force_download,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw / DATA_FILENAME, destination)
    _contract_path(root).write_text(
        json.dumps(_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not _ready(destination):
        raise RuntimeError("prepared LiveCodeBench release-v6 data is incomplete")
    return source_dir(root)


def _decode_tests(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        # This is the official LiveCodeBench encoding for private tests. The
        # pickle is accepted only from the pinned dataset snapshot above.
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))
    if not isinstance(decoded, list):
        raise ValueError("LiveCodeBench tests must decode to a list")
    return [dict(item) for item in decoded]


def _input_output(row: Mapping[str, Any]) -> str:
    public = _decode_tests(str(row["public_test_cases"]))
    private = _decode_tests(str(row["private_test_cases"]))
    tests = [*public, *private]
    metadata = json.loads(str(row.get("metadata") or "{}"))
    return json.dumps(
        {
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
            "fn_name": metadata.get("func_name"),
        },
        ensure_ascii=False,
    )


def _question(row: Mapping[str, Any]) -> str:
    content = str(row["question_content"])
    starter = str(row.get("starter_code") or "")
    if starter:
        format_text = (
            "Use the following starter code and return the complete solution in one "
            f"Python code block.\n```python\n{starter}\n```"
        )
    else:
        format_text = (
            "Read input from stdin and write output to stdout. Return the complete "
            "program in one Python code block. Do not hard-code sample tests."
        )
    return (
        "You are solving the LiveCodeBench code-generation scenario.\n\n"
        f"### Question\n{content}\n\n### Output contract\n{format_text}"
    )


def load_livecodebench(n: Optional[int] = None) -> list[dict[str, Any]]:
    path = ensure_source() / DATA_FILENAME
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "task": "livecodebench",
                    "kind": "livecodebench",
                    "question": _question(row),
                    "gold": {
                        "question_id": str(row["question_id"]),
                        "input_output": _input_output(row),
                    },
                    "context": None,
                    "metadata": {
                        "task_id": str(row["question_id"]),
                        "question_id": str(row["question_id"]),
                        "question_title": row.get("question_title"),
                        "platform": row.get("platform"),
                        "difficulty": row.get("difficulty"),
                        "contest_id": row.get("contest_id"),
                        "contest_date": row.get("contest_date"),
                        "release_version": RELEASE_VERSION,
                        "official_dataset_revision": DATASET_REVISION,
                        "official_evaluator_revision": EVALUATOR_REVISION,
                    },
                }
            )
            if n is not None and len(records) >= n:
                break
    return records


def _load_cache(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[(str(row["case_id"]), int(row.get("trial_index", 0)))] = row
    return cache


def attach_official_results(
    predictions: list[dict[str, Any]],
    gold_by_id: Mapping[str, dict[str, Any]],
    run_dir: Path,
    *,
    workers: int,
    timeout: int,
) -> None:
    cache_path = run_dir / "official_evaluation" / "livecodebench_results.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_cache(cache_path)
    missing = []
    for prediction in predictions:
        key = (str(prediction.get("case_id") or ""), int(prediction.get("trial_index", 0)))
        if key in cached:
            continue
        gold = dict((gold_by_id[key[0]].get("gold") or {}))
        missing.append(
            {
                "case_id": key[0],
                "trial_index": key[1],
                "question_id": str(gold["question_id"]),
                "input_output": str(gold["input_output"]),
                "prediction": str(prediction.get("prediction") or ""),
            }
        )
    if missing:
        temporary_dir = cache_path.parent / ".livecodebench-eval"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        payload_path = temporary_dir / "input.json"
        output_path = temporary_dir / "output.json"
        payload_path.write_text(
            json.dumps(
                {"rows": missing, "workers": workers, "timeout": timeout},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "lychee_mas.eval.benchmarks.livecodebench_official",
                "--evaluator-repo",
                str(_evaluator_checkout()),
                "--input",
                str(payload_path),
                "--output",
                str(output_path),
            ],
            check=True,
            timeout=max(600, len(missing) * 180),
        )
        evaluated = json.loads(output_path.read_text(encoding="utf-8"))["results"]
        with cache_path.open("a", encoding="utf-8") as handle:
            for row in evaluated:
                row["dataset_revision"] = DATASET_REVISION
                row["evaluator_revision"] = EVALUATOR_REVISION
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                cached[(str(row["case_id"]), int(row["trial_index"]))] = row
        remove_path(temporary_dir)

    for prediction in predictions:
        key = (str(prediction.get("case_id") or ""), int(prediction.get("trial_index", 0)))
        prediction["livecodebench_official_result"] = cached[key]


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(_prediction: str, _gold: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record.get("livecodebench_official_result") or {})
    if not result:
        raise BenchmarkEvaluationError("LiveCodeBench official checker result is missing")
    return {
        "score": 1.0 if result.get("passed") else 0.0,
        "passed": bool(result.get("passed")),
        "question_id": result.get("question_id"),
        "test_results": result.get("test_results"),
        "evaluator_metadata": result.get("evaluator_metadata"),
        "official_dataset_revision": DATASET_REVISION,
        "official_evaluator_revision": EVALUATOR_REVISION,
    }


class LiveCodeBenchBenchmark(Benchmark):
    def add_analysis_arguments(self, parser: Any) -> None:
        parser.add_argument("--livecodebench-eval-workers", type=int, default=4)
        parser.add_argument("--livecodebench-test-timeout", type=int, default=6)

    def evaluation_options(self, args: Any) -> dict[str, Any]:
        return {
            "workers": args.livecodebench_eval_workers,
            "timeout": args.livecodebench_test_timeout,
        }

    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        if context.external_evaluator == "skip":
            raise BenchmarkEvaluationError(
                "LiveCodeBench requires the pinned official code-generation checker"
            )
        attach_official_results(predictions, gold_by_id, context.run_dir, **context.options)

    def aggregate(
        self,
        samples: Sequence[Mapping[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        metrics["livecodebench_release_version"] = RELEASE_VERSION
        metrics["livecodebench_expected_cases"] = EXPECTED_CASES
        metrics["livecodebench_dataset_revision"] = DATASET_REVISION
        metrics["livecodebench_evaluator_revision"] = EVALUATOR_REVISION
        return metrics


BENCHMARK = register_benchmark(
    LiveCodeBenchBenchmark(
        benchmark_id="livecodebench",
        name="LiveCodeBench Code Generation Lite",
        category="code_generation",
        sources=SOURCES,
        full_prepare_target="livecodebench",
        prepare_handlers={"livecodebench": _prepare},
        loaders={"livecodebench": load_livecodebench},
        scorer_kinds={"livecodebench": "livecodebench"},
        score_handlers={"livecodebench": _score},
        binary_kinds=("livecodebench",),
        evaluation_mode="batch_final",
        capabilities={"required": ["text_generation", "code_execution"]},
        runtime_defaults={
            "docker_image": "lychee-python-sandbox:local",
            "max_new_tokens": 32768,
            "max_turns": 12,
            "trials_per_case": 1,
        },
        scoring_profiles={
            "livecodebench": {
                "default_profile": "official_release_v6",
                "profiles": {
                    "official_release_v6": {
                        "scorer_id": "livecodebench",
                        "parameters": {
                            "release_version": RELEASE_VERSION,
                            "dataset": "code_generation_lite",
                            "test_timeout_seconds": 6,
                            "official_pass_at_k": True,
                        },
                    }
                },
            }
        },
        contamination_audit_default=True,
    )
)
