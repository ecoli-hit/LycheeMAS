"""Deterministic Case selection for full and subset runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

CASE_SELECTION_METHODS = {"head", "uniform", "stratified"}
_DEFAULT_STRATA_FIELDS = ("level", "task", "category", "domain", "repo", "subset")


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def dataset_index(record: dict[str, Any], fallback: int) -> int:
    metadata = _metadata(record)
    return int(metadata.get("_lychee_dataset_index", fallback))


def _value(record: dict[str, Any], field: str) -> str | None:
    metadata = _metadata(record)
    value = metadata.get(field, record.get(field))
    return None if value in (None, "") else str(value)


def _uniform_indices(indices: Sequence[int], count: int) -> list[int]:
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    return [indices[offset * (len(indices) - 1) // (count - 1)] for offset in range(count)]


def select_cases(
    records: Sequence[dict[str, Any]],
    *,
    count: int | None,
    start_index: int = 0,
    method: str = "head",
    strata_field: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select Cases without losing their original dataset indices."""

    selection_method = str(method or "head").lower()
    if selection_method not in CASE_SELECTION_METHODS:
        raise ValueError(f"unknown case selection method {selection_method!r}")
    start = max(0, int(start_index))
    candidates = list(range(start, len(records)))
    requested = len(candidates) if count is None else max(0, int(count))
    requested = min(requested, len(candidates))
    resolved_strata_field = strata_field
    if selection_method == "head":
        selected_indices = candidates[:requested]
    elif selection_method == "uniform":
        selected_indices = _uniform_indices(candidates, requested)
    else:
        if not resolved_strata_field:
            resolved_strata_field = next(
                (
                    field
                    for field in _DEFAULT_STRATA_FIELDS
                    if any(_value(records[index], field) is not None for index in candidates)
                ),
                None,
            )
        if not resolved_strata_field:
            selected_indices = _uniform_indices(candidates, requested)
            selection_method = "uniform"
        else:
            groups: dict[str, list[int]] = defaultdict(list)
            for index in candidates:
                groups[_value(records[index], resolved_strata_field) or "<missing>"].append(index)
            ordered_groups = sorted(groups.items())
            if count is None:
                selected_indices = []
                offset = 0
                while len(selected_indices) < requested:
                    for _key, values in ordered_groups:
                        if offset < len(values):
                            selected_indices.append(values[offset])
                    offset += 1
                allocations = {}
            else:
                allocations = {key: 0 for key, _values in ordered_groups}
            if count is not None and requested < len(ordered_groups):
                chosen_positions = _uniform_indices(
                    list(range(len(ordered_groups))), requested
                )
                for position in chosen_positions:
                    allocations[ordered_groups[position][0]] = 1
            elif count is not None:
                for key, _values in ordered_groups:
                    allocations[key] = 1
                remaining = requested - len(ordered_groups)
                while remaining:
                    progressed = False
                    for key, values in ordered_groups:
                        if allocations[key] < len(values):
                            allocations[key] += 1
                            remaining -= 1
                            progressed = True
                            if remaining == 0:
                                break
                    if not progressed:
                        break
            if count is not None:
                selected_indices = []
                for key, values in ordered_groups:
                    selected_indices.extend(_uniform_indices(values, allocations[key]))
    selected: list[dict[str, Any]] = []
    for index in selected_indices:
        record = dict(records[index])
        metadata = dict(record.get("metadata") or {})
        metadata["_lychee_dataset_index"] = index
        record["metadata"] = metadata
        selected.append(record)
    return selected, {
        "method": selection_method,
        "requested_count": count,
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "start_index": start,
        "strata_field": resolved_strata_field,
        "selected_dataset_indices": selected_indices,
    }
