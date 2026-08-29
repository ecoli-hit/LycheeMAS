from __future__ import annotations

import time
from pathlib import Path

from lychee_mas.eval.experiments.registry import ExperimentRegistry
from lychee_mas.eval.scheduling.manager import ExperimentQueueManager


def test_deployment_capacity_expands_only_when_healthy_load_is_saturated(
    tmp_path: Path,
) -> None:
    manager = ExperimentQueueManager(ExperimentRegistry(tmp_path), None, None, None)
    manager._active_allocations = {
        "dynamic-run": {
            "instance_id": "dynamic-run",
            "trial_concurrency_mode": "auto",
            "max_trial_slots": 16,
            "remaining_work": 16,
            "in_flight_trial_slots": 2,
            "deployment_capacities": {"shared-vllm": 16},
            "deployment_admission_policies": {
                "shared-vllm": {
                    "mode": "auto",
                    "minimum": 2,
                    "initial": 8,
                    "maximum": 16,
                    "increase_step": 2,
                    "decrease_factor": 0.75,
                    "healthy_polls_required": 1,
                }
            },
            "run_dir": str(tmp_path / "run"),
        }
    }
    manager._deployment_health["shared-vllm"] = {
        "effective_capacity": 8,
        "hard_capacity": 16,
    }
    snapshots = iter(
        [
            {
                "status": "ok",
                "timestamp_unix_s": 1,
                "requests_running": 2,
                "requests_waiting": 0,
                "gpu_cache_usage_fraction": 0.2,
            },
            {
                "status": "ok",
                "timestamp_unix_s": 2,
                "requests_running": 8,
                "requests_waiting": 0,
                "gpu_cache_usage_fraction": 0.4,
            },
        ]
    )
    manager._deployment_health_monitor.snapshot_reader = (
        lambda _run_dir, _deployment_id: next(snapshots)
    )

    manager._update_deployment_health()
    assert manager._deployment_health["shared-vllm"]["effective_capacity"] == 8
    assert manager._deployment_health["shared-vllm"]["capacity_phase"] == "under_observed"

    manager._active_allocations["dynamic-run"]["in_flight_trial_slots"] = 8
    manager._update_deployment_health()
    assert manager._deployment_health["shared-vllm"]["effective_capacity"] == 10
    assert manager._deployment_health["shared-vllm"]["capacity_phase"] == "probing"


def test_idle_stale_vllm_pressure_does_not_block_fixed_group_forever(
    tmp_path: Path,
) -> None:
    manager = ExperimentQueueManager(ExperimentRegistry(tmp_path), None, None, None)
    manager._pending_allocations = {
        "fixed-run": {
            "instance_id": "fixed-run",
            "trial_concurrency_mode": "fixed",
            "max_trial_slots": 5,
            "remaining_work": 5,
            "in_flight_trial_slots": 0,
            "deployment_capacities": {"shared-vllm": 10},
            "deployment_admission_policies": {
                "shared-vllm": {
                    "mode": "auto",
                    "minimum": 2,
                    "initial": 5,
                    "maximum": 10,
                }
            },
            "run_dir": str(tmp_path / "run"),
        }
    }
    manager._deployment_health["shared-vllm"] = {
        "effective_capacity": 3,
        "vllm_admission_blocked": True,
        "gpu_admission_blocked": False,
    }
    manager._deployment_health_monitor.snapshot_reader = lambda _run_dir, _deployment_id: {
        "status": "ok",
        "timestamp_unix_s": time.time() - 120,
        "requests_running": 0,
        "requests_waiting": 0,
        "gpu_cache_usage_fraction": 0,
    }

    manager._update_deployment_health()

    health = manager._deployment_health["shared-vllm"]
    assert health["capacity_phase"] == "metrics_stale"
    assert health["vllm_admission_blocked"] is False


def test_experiment_pressure_projection_reuses_deployment_gpu_snapshot(
    tmp_path: Path,
) -> None:
    manager = ExperimentQueueManager(ExperimentRegistry(tmp_path), None, None, None)
    expected = {"level": "normal", "pressure_score": 0.25, "devices": [{"index": 6}]}
    manager._deployment_health_monitor._latest_gpu_pressure = expected

    pressure = manager._pressure_snapshot_locked({})

    assert pressure["gpu"] == expected
