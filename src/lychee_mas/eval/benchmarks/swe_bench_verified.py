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

from lychee_mas.eval.infrastructure.docker_image_cache import (
    DockerImageCachePolicy,
    ProjectDockerImageCache,
    swebench_image_usage_from_runs,
)

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

SOURCES: dict[str, Any] = {
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
                    "official_image": str(row["image"]),
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
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "fetch",
            "--no-tags",
            "--depth",
            "1",
            str(cache),
            commit,
        ],
        check=True,
    )
    alternates = checkout / ".git" / "objects" / "info" / "alternates"
    if alternates.exists():
        remove_path(checkout)
        raise RuntimeError("SWE-bench case checkout unexpectedly depends on a Git alternates store")
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", f"https://github.com/{repo}.git"],
        check=True,
    )
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
        capture_output=True,
        check=True,
    )
    # A valid binary patch can contain arbitrary bytes. Preserve the patch
    # instead of failing the whole Trial during Python's implicit UTF-8 decode.
    return result.stdout.decode("utf-8", errors="surrogateescape")


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
                        "model_patch": str(record.get("prediction") or ""),
                        "model_name_or_path": model_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destination


def _remove_harness_containers(run_id: str) -> list[str]:
    """Remove only SWE-bench containers created for one unique harness Run."""

    listed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={run_id}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        return []
    container_ids = [item for item in listed.stdout.splitlines() if item]
    if not container_ids:
        return []
    removed = subprocess.run(
        ["docker", "rm", "-f", *container_ids],
        text=True,
        capture_output=True,
        check=False,
    )
    return removed.stdout.splitlines() if removed.returncode == 0 else []


def run_official_harness(
    predictions_path: Path,
    report_dir: Path,
    *,
    run_id: str,
    max_workers: int = 4,
    timeout: int = 1800,
    image_refs: Sequence[str] = (),
    image_cache: ProjectDockerImageCache | None = None,
) -> dict[str, Any]:
    prepared = ensure_source().resolve()
    harness = prepared / "harness"
    predictions_path = predictions_path.resolve()
    report_dir = report_dir.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(harness) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(_prepared_dataset(prepared).resolve()),
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
    lease = None
    try:
        if image_cache is None:
            try:
                subprocess.run(command, cwd=report_dir, env=env, check=True)
            finally:
                _remove_harness_containers(run_id)
        else:
            with image_cache.lease(image_refs, owner=run_id) as active_lease:
                lease = active_lease
                try:
                    subprocess.run(command, cwd=report_dir, env=env, check=True)
                finally:
                    _remove_harness_containers(run_id)
    finally:
        if lease is not None:
            (report_dir / f"docker_image_cache.{run_id}.json").write_text(
                json.dumps(
                    {
                        "lease_id": lease.lease_id,
                        "owner": lease.owner,
                        "images": list(lease.images),
                        "reports": lease.reports,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
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
    image_refs_by_case: Mapping[str, str],
    image_cache: ProjectDockerImageCache,
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
    base_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_run_id).strip("-")
    configured_batch_size = image_cache.policy.max_images
    batch_size = configured_batch_size if configured_batch_size > 0 else max_workers
    batch_size = max(1, batch_size)
    reports: list[dict[str, Any]] = []
    run_id_by_case: dict[str, str] = {}
    for batch_index, start in enumerate(range(0, len(predictions), batch_size), start=1):
        batch = predictions[start : start + batch_size]
        run_id = f"{base_run_id}-part-{batch_index:04d}"
        batch_path = (
            official_predictions
            if len(predictions) <= batch_size
            else write_harness_predictions(
                batch,
                evaluation_dir / f"predictions.part-{batch_index:04d}.jsonl",
                model_name=model_name,
            )
        )
        case_ids = [str(item.get("case_id") or "") for item in batch]
        image_refs = [image_refs_by_case[item] for item in case_ids if item in image_refs_by_case]
        reports.append(
            run_official_harness(
                batch_path,
                evaluation_dir,
                run_id=run_id,
                max_workers=max_workers,
                timeout=timeout,
                image_refs=image_refs,
                image_cache=image_cache,
            )
        )
        run_id_by_case.update({case_id: run_id for case_id in case_ids})

    def _merged_ids(key: str) -> set[str]:
        return {str(item) for report in reports for item in report.get(key) or []}

    resolved = _merged_ids("resolved_ids")
    infra = _merged_ids("infra_failure_ids")
    incomplete = _merged_ids("incomplete_ids")
    empty = _merged_ids("empty_patch_ids")
    errors = _merged_ids("error_ids")
    reasons = {
        str(case_id): reason
        for report in reports
        for case_id, reason in dict(report.get("failure_reasons") or {}).items()
    }
    (evaluation_dir / f"aggregate.{base_run_id}.json").write_text(
        json.dumps(
            {
                "run_id": base_run_id,
                "parts": len(reports),
                "batch_size": batch_size,
                "resolved_ids": sorted(resolved),
                "infra_failure_ids": sorted(infra),
                "incomplete_ids": sorted(incomplete),
                "empty_patch_ids": sorted(empty),
                "error_ids": sorted(errors),
                "failure_reasons": reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
            "run_id": run_id_by_case.get(case_id, base_run_id),
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
        parser.add_argument(
            "--swebench-image-cache-mode",
            choices=("bounded", "keep", "remove_after_use"),
            default="bounded",
        )
        parser.add_argument("--swebench-image-cache-max-images", type=int, default=24)
        parser.add_argument("--swebench-image-cache-min-free-gib", type=float, default=12.0)
        parser.add_argument("--swebench-image-cache-target-free-gib", type=float, default=20.0)

    def evaluation_options(self, args: Any) -> dict[str, Any]:
        return {
            "max_workers": args.swebench_max_workers,
            "timeout": args.swebench_timeout,
            "image_cache_mode": args.swebench_image_cache_mode,
            "image_cache_max_images": args.swebench_image_cache_max_images,
            "image_cache_min_free_gib": args.swebench_image_cache_min_free_gib,
            "image_cache_target_free_gib": args.swebench_image_cache_target_free_gib,
        }

    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        if context.external_evaluator == "skip":
            raise BenchmarkEvaluationError(
                "SWE-bench Verified requires the official Docker harness."
            )
        policy = DockerImageCachePolicy(
            mode=str(context.options.get("image_cache_mode", "bounded")),
            max_images=int(context.options.get("image_cache_max_images", 24)),
            min_free_gib=float(context.options.get("image_cache_min_free_gib", 12.0)),
            target_free_gib=float(context.options.get("image_cache_target_free_gib", 20.0)),
        )
        image_cache = ProjectDockerImageCache(policy=policy)
        if not image_cache.manifest_path.exists():
            runs_root = Path(
                os.environ.get("LYCHEE_BENCHMARK_RUNS_ROOT", "runs/benchmarks")
            )
            image_cache.adopt(
                swebench_image_usage_from_runs(
                    dataset_path=_prepared_dataset(),
                    runs_root=runs_root,
                )
            )
        image_refs_by_case = {}
        for prediction in predictions:
            case_id = str(prediction.get("case_id") or "")
            item = gold_by_id.get(case_id) or {}
            metadata = dict(item.get("metadata") or {})
            image = str(metadata.get("official_image") or "").strip()
            if image:
                image_refs_by_case[case_id] = image
        attach_official_harness_results(
            predictions,
            context.run_dir,
            model_name=str(context.run_info.get("model") or "lychee-mas"),
            max_workers=int(context.options.get("max_workers", 4)),
            timeout=int(context.options.get("timeout", 1800)),
            image_refs_by_case=image_refs_by_case,
            image_cache=image_cache,
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
        evaluation_mode="batch_final",
        result_kind="patch",
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
            "max_rounds": 50,
            "max_turns": 100,
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
