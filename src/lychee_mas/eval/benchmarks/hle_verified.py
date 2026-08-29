"""HLE-Verified Gold adapter with the HLE official response/judge contract."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import EvaluationContext
from .common import (
    conversion_only,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
    raw_source_matches,
    remove_path,
)
from .hle import SYSTEM_PROMPT, HLEBenchmark, judge_score_details, prepare_judge_responses
from .registry import register_benchmark

DATASET_ID = "skylenage-ai/HLE-Verified"
DATASET_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
OFFICIAL_REPOSITORY = "https://github.com/SKYLENAGE-AI/HLE-Verified"
SUBSET = "Gold subset"
EXPECTED_CASES = 668

SOURCES: dict[str, Any] = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {
        "env": "LYCHEE_HLE_VERIFIED_HF_ID",
        "default_ids": [DATASET_ID],
        "revision": DATASET_REVISION,
    },
    "github": {"env": None, "default_ids": []},
    "other_defaults": [
        {
            "provider": "github",
            "id": OFFICIAL_REPOSITORY,
            "revision": "main",
            "selectable": False,
            "purpose": "official schema, verification protocol, and reporting guidance",
        },
        {
            "provider": "modelscope",
            "id": "lmms-lab/HLE-Verified",
            "selectable": False,
            "purpose": (
                "text-only mirror; images were removed, so it is documented but not "
                "used by the multimodal benchmark adapter"
            ),
        },
    ],
    "fallback_files": [],
}


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("hle_verified")


def _prepared_file(root: str | Path | None = None) -> Path:
    return source_dir(root) / "gold.parquet"


def _contract_path(root: str | Path | None = None) -> Path:
    return source_dir(root) / "adapter_contract.json"


def _raw_dataset() -> Path:
    return raw_source_dir("hle_verified", "huggingface", DATASET_ID)


def _contract() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "subset": SUBSET,
        "expected_cases": EXPECTED_CASES,
        "solver_response_contract": "cais_hle_official",
        "judge_contract": "cais_hle_official_structured_judge",
    }


def _ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        contract = json.loads(_contract_path(path.parent).read_text(encoding="utf-8"))
        if contract != _contract():
            return False
        import pyarrow.parquet as parquet

        metadata = parquet.read_metadata(path)
        schema_names = set(parquet.read_schema(path).names)
    except Exception:
        return False
    return metadata.num_rows == EXPECTED_CASES and {
        "id",
        "Verified_Classes",
        "question",
        "answer",
        "json",
    }.issubset(schema_names)


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "huggingface"}:
        raise ValueError("HLE-Verified multimodal data is registered through Hugging Face")
    destination = _prepared_file(root)
    if _ready(destination) and not force_download:
        return source_dir(root)

    raw = _raw_dataset()
    raw_ready = raw_source_matches(
        raw,
        provider="huggingface",
        source_id=DATASET_ID,
        revision=DATASET_REVISION,
    ) and bool(list((raw / "data").glob("Gold_subset.part*.parquet")))
    if conversion_only() and not raw_ready:
        raise FileNotFoundError(
            f"HLE-Verified raw source does not match {DATASET_REVISION}: {raw}"
        )
    if not conversion_only() and (force_download or not raw_ready):
        from huggingface_hub import snapshot_download

        if force_download:
            remove_path(raw)
        log_download_source(
            "hle_verified",
            "huggingface",
            DATASET_ID,
            raw,
            revision=DATASET_REVISION,
        )
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=str(raw),
            allow_patterns=["data/Gold_subset.part*.parquet"],
        )

    files = sorted((raw / "data").glob("Gold_subset.part*.parquet"))
    if not files:
        raise FileNotFoundError(f"HLE-Verified Gold parquet files are missing under {raw}")
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"test": [str(path) for path in files]},
        split="test",
    )
    if len(dataset) != EXPECTED_CASES:
        raise RuntimeError(
            f"HLE-Verified Gold expected {EXPECTED_CASES} cases, found {len(dataset)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    remove_path(temporary)
    dataset.to_parquet(str(temporary))
    os.replace(temporary, destination)
    metadata = raw / ".lychee_source.json"
    if metadata.is_file():
        shutil.copy2(metadata, destination.parent / ".lychee_source.json")
    _contract_path(root).write_text(
        json.dumps(_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not _ready(destination):
        raise RuntimeError("prepared HLE-Verified Gold parquet is incomplete")
    return source_dir(root)


def load_hle_verified(n: Optional[int] = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    source = ensure_source()
    dataset = load_dataset(
        "parquet",
        data_files={"test": str(source / "gold.parquet")},
        split="test",
    )
    records: list[dict[str, Any]] = []
    for row in dataset:
        payload = json.loads(str(row.get("json") or "{}"))
        image = str(payload.get("image") or "") or None
        question = str(row["question"])
        records.append(
            {
                "task": "hle_verified",
                "kind": "hle_verified",
                "question": question,
                "gold": str(row["answer"]),
                "context": None,
                "metadata": {
                    "task_id": str(row["id"]),
                    "verified_subset": str(row["Verified_Classes"]),
                    "category": row.get("category"),
                    "raw_subject": row.get("raw_subject"),
                    "answer_type": payload.get("answer_type"),
                    "rationale": payload.get("rationale"),
                    "verification": payload.get("verify_meta_info"),
                    "official_dataset_revision": DATASET_REVISION,
                    "image": image,
                    "system_prompt": SYSTEM_PROMPT,
                    "multimodal_content": [
                        {"type": "text", "text": question},
                        *(
                            [{"type": "image_url", "image_url": {"url": image}}]
                            if image
                            else []
                        ),
                    ],
                },
            }
        )
        if n is not None and len(records) >= n:
            break
    return records


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(_prediction: str, _gold: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("hle_judge_error"):
        from .base import BenchmarkEvaluationError

        error = record["hle_judge_error"]
        raise BenchmarkEvaluationError(
            str(error.get("error_message") if isinstance(error, Mapping) else error)
        )
    details = judge_score_details(record.get("hle_judge_response"))
    details["verified_subset"] = SUBSET
    details["official_dataset_revision"] = DATASET_REVISION
    return details


class HLEVerifiedBenchmark(HLEBenchmark):
    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        missing = [item for item in predictions if not item.get("hle_judge_response")]
        if not missing:
            return
        if context.external_evaluator == "skip":
            from .base import BenchmarkEvaluationError

            raise BenchmarkEvaluationError(
                "HLE-Verified requires official-format judge responses"
            )
        prepare_judge_responses(missing, gold_by_id, context.run_dir, **context.options)

    def aggregate(
        self,
        samples: Sequence[Mapping[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = super().aggregate(samples, metrics)
        metrics["hle_verified_subset"] = SUBSET
        metrics["hle_verified_expected_cases"] = EXPECTED_CASES
        metrics["hle_verified_dataset_revision"] = DATASET_REVISION
        return metrics


BENCHMARK = register_benchmark(
    HLEVerifiedBenchmark(
        benchmark_id="hle_verified",
        name="HLE-Verified Gold",
        category="expert_knowledge_multimodal",
        sources=SOURCES,
        full_prepare_target="hle_verified",
        prepare_handlers={"hle_verified": _prepare},
        loaders={"hle_verified": load_hle_verified},
        scorer_kinds={"hle_verified": "hle_verified"},
        score_handlers={"hle_verified": _score},
        binary_kinds=("hle_verified",),
        evaluation_mode="batch_final",
        evaluation_requirements={
            "config_key": "hle_judge",
            "required_when_external_evaluator": ["auto", "run"],
            "required_fields": ["model", "base_url"],
        },
        capabilities={
            "required": ["text_generation", "vision"],
            "attachments_required": True,
        },
        runtime_defaults={"max_new_tokens": 32768, "max_turns": 12},
        scoring_profiles={
            "hle_verified": {
                "default_profile": "official_gold",
                "profiles": {
                    "official_gold": {
                        "scorer_id": "hle_verified",
                        "parameters": {
                            "subset": SUBSET,
                            "evaluator": "cais_hle_official_structured_llm_judge",
                            "reports_calibration_error": True,
                        },
                    }
                },
            }
        },
    )
)
