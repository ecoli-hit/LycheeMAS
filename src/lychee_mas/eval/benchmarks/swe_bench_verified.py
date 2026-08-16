"""SWE-bench Verified loader, workspace preparation, and harness adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import (
    Benchmark,
    BenchmarkEvaluationError,
    CaseMaterialization,
    EvaluationContext,
    resolve_provider_ids,
)
from .common import (
    clone_git_source,
    conversion_only,
    load_local_dataset_split,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
    raw_source_matches,
    remove_path,
)
from .registry import register_benchmark

SOURCES = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {
        "env": "LYCHEE_SWEBENCH_VERIFIED_HF_ID",
        "default_ids": ["SWE-bench/SWE-bench_Verified"],
        "revision": "03e151cf5560b1af6a4363c6a9d766deaaea6b56",
    },
    "github": {
        "env": None,
        "default_ids": ["https://github.com/SWE-bench/SWE-bench.git"],
        "revision": "128cbd1a5759694874e6bd56624cb2fd6fb079e2",
        "source_type": "git",
        "purpose": "required official Docker evaluation harness",
    },
    "other_defaults": [],
    "fallback_files": [],
}

DATASET_ID = str(SOURCES["huggingface"]["default_ids"][0])
DATASET_REVISION = str(SOURCES["huggingface"]["revision"])
HARNESS_REPO = resolve_provider_ids(SOURCES, "github")[0]
HARNESS_REVISION = str(SOURCES["github"]["revision"])


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("swe_bench_verified")


def dataset_id() -> str:
    return resolve_provider_ids(SOURCES, "huggingface")[0]


def _raw_dataset() -> Path:
    return raw_source_dir("swe_bench_verified", "huggingface", dataset_id())


def _raw_harness() -> Path:
    return raw_source_dir("swe_bench_verified", "github", "SWE-bench/SWE-bench")


def _prepared_dataset(root: str | Path | None = None) -> Path:
    return source_dir(root) / "dataset" / "test-00000-of-00001.parquet"


def _contract_path(root: Path) -> Path:
    return root / "adapter_contract.json"


def _ready(root: Path) -> bool:
    if not (
        (root / "dataset" / "test-00000-of-00001.parquet").is_file()
        and (root / "harness" / "swebench" / "harness" / "run_evaluation.py").is_file()
        and _contract_path(root).is_file()
    ):
        return False
    try:
        contract = json.loads(_contract_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return contract == {
        "dataset_id": dataset_id(),
        "dataset_revision": DATASET_REVISION,
        "harness_revision": HARNESS_REVISION,
    }


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "huggingface", "github"}:
        raise ValueError("SWE-bench Verified uses HuggingFace data and the official GitHub harness")
    prepared = source_dir(root)
    if _ready(prepared) and not force_download:
        return prepared

    raw_dataset = _raw_dataset()
    resolved_dataset_id = dataset_id()
    raw_dataset_ready = raw_source_matches(
        raw_dataset,
        provider="huggingface",
        source_id=resolved_dataset_id,
        revision=DATASET_REVISION,
    )
    if conversion_only() and not raw_dataset_ready:
        raise FileNotFoundError(
            "SWE-bench Verified Raw data does not match pinned revision "
            f"{DATASET_REVISION}: {raw_dataset}"
        )
    if not conversion_only() and (force_download or not raw_dataset_ready):
        from huggingface_hub import snapshot_download

        log_download_source(
            "swe_bench_verified",
            "huggingface",
            resolved_dataset_id,
            raw_dataset,
            revision=DATASET_REVISION,
        )
        try:
            snapshot_download(
                repo_id=resolved_dataset_id,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=str(raw_dataset),
            )
        except Exception:
            (raw_dataset / ".lychee_source.json").unlink(missing_ok=True)
            raise
    if not raw_dataset.exists():
        raise FileNotFoundError(f"SWE-bench Verified raw dataset is missing: {raw_dataset}")

    raw_harness = clone_git_source(
        "swe_bench_verified",
        HARNESS_REPO,
        _raw_harness(),
        revision=HARNESS_REVISION,
        force=force_download,
    )
    dataset = load_local_dataset_split(raw_dataset, "test")
    destination = _prepared_dataset(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    remove_path(temporary)
    dataset.to_parquet(str(temporary))
    os.replace(temporary, destination)

    harness_destination = prepared / "harness"
    remove_path(harness_destination)
    shutil.copytree(raw_harness, harness_destination, ignore=shutil.ignore_patterns(".git"))
    _contract_path(prepared).write_text(
        json.dumps(
            {
                "dataset_id": dataset_id(),
                "dataset_revision": DATASET_REVISION,
                "harness_revision": HARNESS_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not _ready(prepared):
        raise RuntimeError("prepared SWE-bench Verified data or official harness is incomplete")
    return prepared


def load_swe_bench_verified(n: Optional[int] = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    prepared = ensure_source()
    dataset = load_dataset(
        "parquet",
        data_files={"test": str(prepared / "dataset" / "test-00000-of-00001.parquet")},
        split="test",
    )
    records = []
    for row in dataset:
        instance_id = str(row["instance_id"])
        records.append(
            {
                "task": "swe_bench_verified",
                "kind": "swe_bench_verified",
                "question": (
                    f"Repository: {row['repo']}\n"
                    f"Base commit: {row['base_commit']}\n\n"
                    f"Issue:\n{row['problem_statement']}\n\n"
                    "Modify the repository in the workspace to resolve the issue. "
                    "Inspect and test the code as needed; the evaluator will collect your git diff."
                ),
                "gold": {"instance_id": instance_id},
                "context": None,
                "metadata": {
                    "task_id": instance_id,
                    "benchmark_adapter": "swe_bench_verified",
                    "instance_id": instance_id,
                    "repo": str(row["repo"]),
                    "base_commit": str(row["base_commit"]),
                    "version": row.get("version"),
                    "official_dataset_id": dataset_id(),
                    "official_harness_revision": HARNESS_REVISION,
                },
            }
        )
        if n is not None and len(records) >= n:
            break
    return records


def _repo_cache(repo: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", repo)
    return Path(os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT", "data/benchmarks/raw")) / (
        f"swe_bench_verified/github_repositories/{safe}"
    )


def prepare_case_workspace(metadata: dict[str, Any], workspace: Path) -> Path:
    """Create a case-local checkout without exposing hidden tests to the agent."""

    repo = str(metadata["repo"])
    commit = str(metadata["base_commit"])
    cache = _repo_cache(repo).resolve()
    lock_path = cache.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (cache / ".git").is_dir():
            remove_path(cache)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{repo}.git",
                    str(cache),
                ],
                check=True,
            )
        exists = (
            subprocess.run(
                ["git", "-C", str(cache), "cat-file", "-e", f"{commit}^{{commit}}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if not exists:
            subprocess.run(
                ["git", "-C", str(cache), "fetch", "--depth", "1", "origin", commit],
                check=True,
            )
        # A shared clone cannot lazily fetch blobs promised only by its source
        # repository. Materialize the exact task tree in the locked cache first.
        subprocess.run(
            ["git", "-C", str(cache), "checkout", "--detach", "--force", commit],
            check=True,
        )
        subprocess.run(["git", "-C", str(cache), "reset", "--hard", commit], check=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    checkout = workspace / "repo"
    remove_path(checkout)
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(cache), str(checkout)], check=True
    )
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if status:
        preview = "\n".join(status.splitlines()[:20])
        remove_path(checkout)
        raise RuntimeError(
            f"SWE-bench case checkout is incomplete or dirty before agent execution:\n{preview}"
        )
    return checkout


def collect_model_patch(workspace: Path) -> str:
    repo = workspace / "repo"
    if not (repo / ".git").is_dir():
        return ""
    subprocess.run(
        ["git", "-C", str(repo), "add", "-N", "."],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def write_harness_predictions(
    predictions: list[dict[str, Any]], destination: Path, *, model_name: str
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(
                json.dumps(
                    {
                        "instance_id": str(record["case_id"]),
                        "model_patch": str(record.get("final_answer") or ""),
                        "model_name_or_path": model_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destination


def run_official_harness(
    predictions_path: Path,
    report_dir: Path,
    *,
    run_id: str,
    max_workers: int = 4,
    timeout: int = 1800,
) -> dict[str, Any]:
    prepared = ensure_source()
    harness = prepared / "harness"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(harness) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_id(),
        "--split",
        "test",
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
        "--report_dir",
        str(report_dir),
    ]
    subprocess.run(command, cwd=report_dir, env=env, check=True)
    reports = sorted(report_dir.glob(f"*.{run_id}.json"))
    if not reports:
        raise RuntimeError("SWE-bench harness completed without a run report")
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def attach_official_harness_results(
    predictions: list[dict[str, Any]],
    run_dir: Path,
    *,
    model_name: str,
    max_workers: int,
    timeout: int,
) -> None:
    """Run the official harness and attach one normalized result per prediction."""

    evaluation_dir = run_dir / "official_evaluation" / "swe_bench_verified"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    official_predictions = write_harness_predictions(
        predictions,
        evaluation_dir / "predictions.jsonl",
        model_name=model_name,
    )
    raw_run_id = f"lychee-{run_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_run_id).strip("-")
    report = run_official_harness(
        official_predictions,
        evaluation_dir,
        run_id=run_id,
        max_workers=max_workers,
        timeout=timeout,
    )
    resolved = set(report.get("resolved_ids") or [])
    infra = set(report.get("infra_failure_ids") or [])
    incomplete = set(report.get("incomplete_ids") or [])
    empty = set(report.get("empty_patch_ids") or [])
    errors = set(report.get("error_ids") or [])
    reasons = dict(report.get("failure_reasons") or {})
    for prediction in predictions:
        case_id = str(prediction.get("case_id") or "")
        status = (
            "incomplete"
            if case_id in incomplete
            else "empty_patch"
            if case_id in empty
            else "error"
            if case_id in errors
            else "completed"
        )
        prediction["swebench_harness"] = {
            "status": status,
            "resolved": case_id in resolved,
            "infrastructure_failure": case_id in infra,
            "failure_reason": reasons.get(case_id),
            "run_id": run_id,
        }


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(_prediction: str, _gold, record: Mapping[str, Any]) -> dict:
    harness = record.get("swebench_harness")
    if not isinstance(harness, dict):
        return {"score": 0.0, "harness_status": "missing"}
    resolved = bool(harness.get("resolved"))
    return {
        "score": 1.0 if resolved else 0.0,
        "harness_status": str(harness.get("status") or "completed"),
        "resolved": resolved,
        "infrastructure_failure": bool(harness.get("infrastructure_failure", False)),
        "failure_reason": harness.get("failure_reason"),
    }


class SWEBenchVerifiedBenchmark(Benchmark):
    def materialize_case(
        self,
        query: Any,
        workspace: Path,
        task_text: str,
    ) -> CaseMaterialization:
        checkout = prepare_case_workspace(query.meta, workspace)
        return CaseMaterialization(
            task_text=f"{task_text}\n\nThe repository checkout is available at repo/.",
            visible_paths=[str(checkout)],
            details={"repository_checkout": str(checkout)},
        )

    def collect_prediction(
        self,
        *,
        query: Any,
        messages: Sequence[Any],
        workspace: Path | None,
        default_text: str,
    ) -> str:
        del query, messages, default_text
        if workspace is None:
            raise RuntimeError("SWE-bench prediction collection requires a workspace")
        return collect_model_patch(workspace)

    def add_analysis_arguments(self, parser: Any) -> None:
        parser.add_argument("--swebench-max-workers", type=int, default=4)
        parser.add_argument("--swebench-timeout", type=int, default=1800)

    def evaluation_options(self, args: Any) -> dict[str, Any]:
        return {
            "max_workers": args.swebench_max_workers,
            "timeout": args.swebench_timeout,
        }

    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        del gold_by_id
        if context.external_evaluator == "skip":
            raise BenchmarkEvaluationError(
                "SWE-bench Verified requires the official Docker harness."
            )
        attach_official_harness_results(
            predictions,
            context.run_dir,
            model_name=str(context.run_info.get("model") or "lychee-mas"),
            **context.options,
        )


BENCHMARK = register_benchmark(
    SWEBenchVerifiedBenchmark(
        benchmark_id="swe_bench_verified",
        name="SWE-bench Verified",
        category="software_engineering",
        sources=SOURCES,
        full_prepare_target="swe_bench_verified",
        prepare_handlers={"swe_bench_verified": _prepare},
        loaders={"swe_bench_verified": load_swe_bench_verified},
        scorer_kinds={"swe_bench_verified": "swe_bench_verified"},
        score_handlers={"swe_bench_verified": _score},
        binary_kinds=("swe_bench_verified",),
        capabilities={"required": ["code_generation", "file_access", "code_execution"]},
        sandbox_profiles={
            "generation": {
                "label": "SWE-bench generation workspace",
                "docker_image": "lychee-python-sandbox:local",
                "description": (
                    "The agent edits a case-local repository; final scoring uses the "
                    "separate official SWE-bench Docker harness."
                ),
            }
        },
        runtime_defaults={
            "code_executor": "docker",
            "code_timeout": 60,
            "docker_image": "lychee-python-sandbox:local",
            "max_new_tokens": 32768,
            "max_turns": 20,
        },
        network_defaults={"access": "required", "mode": "direct"},
        scoring_profiles={
            "swe_bench_verified": {
                "default_profile": "official",
                "profiles": {
                    "official": {
                        "scorer_id": "swe_bench_verified",
                        "parameters": {
                            "evaluator": "official_docker_harness",
                            "prediction_field": "model_patch",
                        },
                    }
                },
            }
        },
    )
)
