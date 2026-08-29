"""Stable case and dataset identities shared by Run planning and Trial execution."""

from __future__ import annotations

from typing import Any


def case_id_from_item(item: dict[str, Any], dataset_index: int) -> str:
    """Return the public benchmark case identity with a dataset-index fallback."""

    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("safe_task_id") or dataset_index)


def dataset_index_from_item(item: dict[str, Any], fallback: int) -> int:
    """Resolve the original dataset index preserved by benchmark sampling."""

    from lychee_mas.eval.benchmarks.sampling import dataset_index

    return dataset_index(item, fallback)
