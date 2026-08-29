"""Tests for deterministic, representative benchmark Case selection."""

from lychee_mas.eval.benchmarks.sampling import dataset_index, select_cases


def _records(count: int) -> list[dict]:
    return [
        {"question": f"q-{index}", "metadata": {"level": (index % 3) + 1}}
        for index in range(count)
    ]


def test_uniform_selection_spans_the_candidate_range() -> None:
    selected, details = select_cases(_records(100), count=5, method="uniform")

    assert details["selected_dataset_indices"] == [0, 24, 49, 74, 99]
    assert [dataset_index(item, -1) for item in selected] == [0, 24, 49, 74, 99]


def test_stratified_selection_covers_each_detected_level() -> None:
    selected, details = select_cases(_records(12), count=6, method="stratified")

    assert details["strata_field"] == "level"
    assert {item["metadata"]["level"] for item in selected} == {1, 2, 3}
    assert details["selected_dataset_indices"] == [0, 9, 1, 10, 2, 11]


def test_stratified_selection_spreads_across_more_groups_than_cases() -> None:
    records = [
        {"question": f"q-{index}", "metadata": {"category": f"g-{index}"}}
        for index in range(10)
    ]

    _selected, details = select_cases(
        records,
        count=3,
        method="stratified",
        strata_field="category",
    )

    assert details["selected_dataset_indices"] == [0, 4, 9]


def test_full_selection_preserves_all_records_after_start_index() -> None:
    selected, details = select_cases(_records(8), count=None, start_index=2, method="head")

    assert len(selected) == 6
    assert details["selected_dataset_indices"] == [2, 3, 4, 5, 6, 7]


def test_full_stratified_selection_interleaves_groups_for_staged_runs() -> None:
    selected, details = select_cases(_records(8), count=None, method="stratified")

    assert len(selected) == 8
    assert details["selected_dataset_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert [item["metadata"]["level"] for item in selected[:3]] == [1, 2, 3]
