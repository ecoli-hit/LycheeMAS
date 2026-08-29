from __future__ import annotations

import json
from pathlib import Path

from lychee_mas.eval.application.read_models.catalog import EvalCatalog
from lychee_mas.eval.application.runs import RunApplicationService
from lychee_mas.runtime.events.store import RunEventWriter, run_events_path


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "task": "gsm8k",
                "expected_trials": 10,
                "successful_trials": 9,
                "large_internal_state": "x" * 10000,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.9,
                "score_mean": 0.9,
                "per_case_details": ["x" * 1000],
            }
        ),
        encoding="utf-8",
    )
    RunEventWriter(run_events_path(run_dir), run_id="run-1")


def test_run_catalog_returns_card_summaries_instead_of_full_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "gsm8k" / "run-1"
    _write_run(run_dir)
    catalog = EvalCatalog(tmp_path, runs_root=runs_root)

    rows = catalog.runs()

    assert len(rows) == 1
    assert rows[0]["status"]["successful_trials"] == 9
    assert "large_internal_state" not in rows[0]["status"]
    assert rows[0]["metrics"]["accuracy"] == 0.9
    assert "per_case_details" not in rows[0]["metrics"]


def test_run_catalog_eliminates_nested_recursive_scan_roots(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    nested = runs_root / "benchmark" / "run-1"
    external = tmp_path / "external-runs"
    catalog = EvalCatalog(tmp_path, runs_root=runs_root)

    roots = catalog._run_roots([nested, external, runs_root])

    assert set(roots) == {runs_root.resolve(), external.resolve()}
    assert nested.resolve() not in roots


def test_run_catalog_short_cache_is_returned_as_an_independent_value(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "gsm8k" / "run-1")
    catalog = EvalCatalog(tmp_path, runs_root=runs_root)

    first = catalog.runs()
    first[0]["status"]["status"] = "mutated-by-caller"
    second = catalog.runs()

    assert second[0]["status"]["status"] == "completed"


def test_run_application_pages_and_searches_compact_cards(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "gsm8k" / "run-1")
    _write_run(runs_root / "gaia" / "run-2")
    service = RunApplicationService(
        catalog=EvalCatalog(tmp_path, runs_root=runs_root),
        experiments=type(
            "Experiments", (), {"specs": lambda self: [], "instances": lambda self: []}
        )(),
        metrics=object(),
        evaluation_profiles=object(),
    )

    page = service.list_runs_page(query="gaia", offset=0, limit=1)

    assert page["total"] == 1
    assert len(page["items"]) == 1
    assert "gaia" in page["items"][0]["relative_path"]
