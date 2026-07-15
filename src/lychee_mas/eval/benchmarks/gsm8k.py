"""GSM8K loader and data preparation."""
from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

from .common import load_parquet, parquet_ready, prepared_root, save_hf_dataset, save_modelscope_dataset
from .source_catalog import provider_ids, source_backend_order


def _download_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = provider_ids("gsm8k", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_modelscope_dataset(dataset_id, "main", split, out_dir, benchmark="gsm8k")
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope GSM8K download failed; " + " | ".join(errors))


def _download_huggingface() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = provider_ids("gsm8k", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_hf_dataset(dataset_id, "main", split, out_dir, benchmark="gsm8k")
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
    for backend in source_backend_order("gsm8k", source):
        try:
            return _download_modelscope() if backend == "modelscope" else _download_huggingface()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare GSM8K; " + " | ".join(errors))


def _has_ready_cache() -> bool:
    return _split_ready("train") and _split_ready("test")


def _split_ready(split: str) -> bool:
    return parquet_ready("gsm8k/main", split, required_columns=("question", "answer"))


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
