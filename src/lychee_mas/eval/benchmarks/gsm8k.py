"""GSM8K loader and data preparation."""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

from .base import Benchmark, resolve_provider_ids, resolve_source_backend_order
from .common import (
    load_parquet,
    parquet_ready,
    prepared_root,
    save_hf_dataset,
    save_modelscope_dataset,
)
from .registry import register_benchmark

SOURCES = {
    "modelscope": {
        "env": "LYCHEE_GSM8K_MODELSCOPE_ID",
        "default_ids": ["AI-ModelScope/gsm8k", "modelscope/gsm8k"],
    },
    "huggingface": {
        "env": "LYCHEE_GSM8K_HF_ID",
        "default_ids": ["openai/gsm8k", "gsm8k"],
    },
    "github": {"env": None, "default_ids": []},
    "other_defaults": [],
    "fallback_files": [],
}

EXPECTED_SPLIT_ROWS = {"train": 7473, "test": 1319}


def _download_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = resolve_provider_ids(SOURCES, "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_modelscope_dataset(
                        dataset_id,
                        "main",
                        split,
                        out_dir,
                        benchmark="gsm8k",
                        raw_subset_dir="main",
                    )
                    _require_valid_split(split, dataset_id)
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope GSM8K download failed; " + " | ".join(errors))


def _download_huggingface() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = resolve_provider_ids(SOURCES, "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_hf_dataset(
                        dataset_id,
                        "main",
                        split,
                        out_dir,
                        benchmark="gsm8k",
                        raw_subset_dir="main",
                    )
                    _require_valid_split(split, dataset_id)
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("HuggingFace GSM8K download failed; " + " | ".join(errors))


def prepare_gsm8k(force: bool = False, source: Optional[str] = None) -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    if not force and _has_ready_cache():
        return out_dir
    if force and os.path.isdir(os.path.join(prepared_root(), "gsm8k")):
        shutil.rmtree(os.path.join(prepared_root(), "gsm8k"))
    errors: list[str] = []
    for backend in resolve_source_backend_order("gsm8k", SOURCES, source):
        try:
            return _download_modelscope() if backend == "modelscope" else _download_huggingface()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare GSM8K; " + " | ".join(errors))


def _has_ready_cache() -> bool:
    return _split_ready("train") and _split_ready("test")


def _split_ready(split: str) -> bool:
    expected_rows = EXPECTED_SPLIT_ROWS[split]
    if not parquet_ready(
        "gsm8k/main",
        split,
        required_columns=("question", "answer"),
        min_rows=expected_rows,
    ):
        return False
    try:
        dataset = load_parquet("gsm8k/main", split)
        questions = [str(value).strip() for value in dataset["question"]]
        return len(dataset) == expected_rows and len(set(questions)) == expected_rows
    except Exception:
        return False


def _require_valid_split(split: str, source_id: str) -> None:
    if not _split_ready(split):
        raise RuntimeError(
            f"GSM8K source {source_id!r} did not produce the official main/{split} "
            f"split ({EXPECTED_SPLIT_ROWS[split]} unique questions)"
        )


def load_gsm8k(n: Optional[int] = None) -> List[Dict]:
    prepare_gsm8k()
    out = []
    for it in load_parquet("gsm8k/main", "test"):
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            {
                "task": "gsm8k",
                "kind": "exact",
                "question": it["question"].strip() + "\nGive the final numeric answer.",
                "gold": gold,
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out


def _score(prediction: str, gold, _record) -> dict:
    from ..evaluation.metrics import score_exact

    expected = gold[0] if isinstance(gold, (list, tuple)) else gold
    return {"score": score_exact(prediction, str(expected))}


BENCHMARK = register_benchmark(
    Benchmark(
        benchmark_id="gsm8k",
        name="GSM8K",
        category="math",
        sources=SOURCES,
        full_prepare_target="gsm8k",
        prepare_handlers={"gsm8k": prepare_gsm8k},
        loaders={"gsm8k": load_gsm8k},
        scorer_kinds={"gsm8k": "exact"},
        score_handlers={"exact": _score},
        binary_kinds=("exact",),
    )
)
