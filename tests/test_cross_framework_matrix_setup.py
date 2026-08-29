import pytest
import scripts.setup_cross_framework_matrix as matrix
from scripts.setup_cross_framework_matrix import (
    BENCHMARK_ORDER,
    BENCHMARKS,
    FRAMEWORKS,
    MAGENTIC_ONE_RECIPE,
    TOPOLOGIES,
    _case_target_reached,
    _experiment_spec,
    _priority,
    _selected_recipes,
    _smoke_requires_rebuild,
    _write_recipe_audit,
)
from scripts.validate_cross_framework_matrix import _validate_instance


def test_matrix_setup_reads_nested_case_target_progress() -> None:
    assert _case_target_reached({"status": "paused", "progress": {"remaining_to_case_target": 0}})
    assert not _case_target_reached(
        {"status": "paused", "progress": {"remaining_to_case_target": 1}}
    )


def test_incomplete_terminal_smoke_is_rebuilt() -> None:
    assert _smoke_requires_rebuild(
        {
            "status": "stopped",
            "progress": {"completed_distinct_cases": 0, "failed_trials": 0},
        },
        target=5,
    )
    assert not _smoke_requires_rebuild(
        {
            "status": "completed",
            "progress": {"completed_distinct_cases": 5, "failed_trials": 0},
        },
        target=5,
    )
    assert _smoke_requires_rebuild(
        {
            "status": "completed",
            "progress": {"completed_distinct_cases": 5, "failed_trials": 1},
        },
        target=5,
    )
    assert not _smoke_requires_rebuild(
        {"status": "queued", "launch_dir": None, "progress": None},
        target=5,
    )
    assert _smoke_requires_rebuild(
        {
            "status": "queued",
            "launch_dir": "/tmp/old-launch",
            "progress": {"completed_distinct_cases": 0, "failed_trials": 0},
        },
        target=5,
    )


def test_swe_matrix_generation_preserves_coherent_runtime_budget() -> None:
    plan = BENCHMARKS["swe-bench-verified"]

    assert plan.max_turns == 14
    assert plan.max_model_calls == 448
    assert plan.max_case_wall_time_s == 10_800


def test_livecodebench_matrix_bounds_agentic_tool_loops() -> None:
    plan = BENCHMARKS["livecodebench"]

    assert plan.max_turns == 16
    assert plan.max_model_calls == 96
    assert plan.max_case_wall_time_s == 3600


def test_complete_matrix_contains_75_framework_recipes() -> None:
    recipes = _selected_recipes(
        BENCHMARK_ORDER,
        TOPOLOGIES,
        include_gaia_magentic_one=True,
    )

    assert len(recipes) == 25
    assert len(recipes) * len(FRAMEWORKS) == 75
    assert recipes[-1] == ("gaia", MAGENTIC_ONE_RECIPE)


def test_new_benchmark_gate_contains_24_framework_recipes() -> None:
    recipes = _selected_recipes(
        ("livecodebench", "hle-verified"),
        TOPOLOGIES,
        include_gaia_magentic_one=False,
    )

    assert len(recipes) == 8
    assert len(recipes) * len(FRAMEWORKS) == 24


def test_matrix_can_select_only_gaia_magentic_one() -> None:
    recipes = _selected_recipes(
        ("gaia",),
        (),
        include_gaia_magentic_one=True,
    )

    assert recipes == (("gaia", MAGENTIC_ONE_RECIPE),)


def test_recipe_audit_rejects_ineligible_controlled_binding(
    tmp_path,
    monkeypatch,
) -> None:
    team_spec = {
        "schema_version": 13,
        "id": "gaia",
        "metadata": {
            "provenance": {
                "track": "controlled_portability",
                "evidence_level": "paper_inspired",
                "sources": ["https://example.test/source"],
            }
        },
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        matrix,
        "_request",
        lambda *_args, **_kwargs: {
            "mapping_level": "approximated",
            "controlled_comparison_eligible": False,
            "semantic_deltas": ["unsupported edge policy"],
        },
    )

    with pytest.raises(RuntimeError, match="ineligible framework bindings"):
        _write_recipe_audit(
            base_url="http://example.test",
            matrix_id="audit-test",
            team_specs={"gaia": team_spec},
            recipes=(("gaia", MAGENTIC_ONE_RECIPE),),
            frameworks=("langgraph",),
        )

    audit = tmp_path / "runs/eval_studio/audit-test-recipe-audit.json"
    assert audit.is_file()


def test_gaia_magentic_one_uses_native_team_and_highest_priority() -> None:
    spec = _experiment_spec("gaia", MAGENTIC_ONE_RECIPE)

    assert spec["team_spec_id"] == "gaia"
    assert _priority("gaia", MAGENTIC_ONE_RECIPE) == 0
    assert all(
        _priority(benchmark, topology) > 0
        for benchmark in BENCHMARK_ORDER
        for topology in TOPOLOGIES
    )


def test_gaia_magentic_one_is_a_controlled_semantic_mapping() -> None:
    import json
    from pathlib import Path

    value = json.loads(
        Path("configs/eval_studio/teams/specs/gaia.json").read_text(encoding="utf-8")
    )

    assert value["metadata"]["provenance"]["track"] == "controlled_portability"
    assert value["metadata"]["provenance"]["evidence_level"] == "paper_inspired"
    assert len(value["metadata"]["provenance"]["sources"]) >= 2


def test_matrix_validation_rejects_an_invalid_result_contract(tmp_path) -> None:
    import json

    (tmp_path / "evaluation_status.json").write_text(
        json.dumps({"evaluated_trials": 2}), encoding="utf-8"
    )
    (tmp_path / "evidence_coverage.json").write_text(
        json.dumps({"overall_status": "complete"}), encoding="utf-8"
    )
    (tmp_path / "metric_evaluation.json").write_text(
        json.dumps(
            {
                "summary": {
                    "metric_count": 1,
                    "observation_count": 2,
                    "evaluator_error_count": 0,
                },
                "metrics": [
                    {
                        "metric_id": "reliability.result_contract_valid",
                        "status_counts": {"measured": 2},
                        "measured_mean": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _validate_instance(
        {
            "team_instance_id": "team-instance",
            "configuration_status": "current",
            "progress": {
                "completed_distinct_cases": 2,
                "successful_trials": 2,
                "failed_trials": 0,
            },
            "run_dir": str(tmp_path),
        },
        expected_team_instance_id="team-instance",
        target=2,
    )

    assert result["status"] == "failed"
    assert any("valid result-contract" in error for error in result["errors"])
