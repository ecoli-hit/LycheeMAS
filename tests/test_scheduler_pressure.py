from __future__ import annotations

from types import SimpleNamespace

import pytest
from lychee_mas.eval.scheduling.pressure import (
    sample_gpu_pressure,
    summarize_api_pressure,
    summarize_vllm_pressure,
)


def test_gpu_pressure_uses_memory_temperature_and_power_as_safety_signals() -> None:
    pressure = sample_gpu_pressure(
        {6, 7},
        command_runner=lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "6, 100, 70000, 81920, 72, 340, 400\n"
                "7, 98, 79000, 81920, 86, 398, 400\n"
            )
        ),
    )

    assert pressure["level"] == "high"
    assert pressure["max_utilization_percent"] == 100
    assert pressure["max_memory_allocation_fraction"] == pytest.approx(79000 / 81920)
    assert pressure["max_temperature_c"] == 86
    assert pressure["max_power_limit_fraction"] == pytest.approx(398 / 400)
    assert pressure["pressure_score"] > 0.9


def test_resource_pressure_scores_are_isolated_by_domain() -> None:
    vllm = summarize_vllm_pressure(
        [
            (
                "vllm-instance",
                {
                    "used": 31,
                    "capacity": 32,
                    "health_feedback": {
                        "pressure_score": 0.25,
                        "requests_running": 28,
                        "requests_waiting": 0,
                        "gpu_cache_usage_fraction": 0.4,
                        "queue_time_window_mean_s": 0.2,
                        "time_to_first_token_window_mean_s": 0.8,
                        "gpu_pressure_level": "high",
                        "gpu_memory_allocation_fraction": 0.99,
                    },
                },
                {},
            )
        ]
    )
    assert vllm["pressure_score"] == pytest.approx(0.25)
    assert "used" not in vllm
    assert "capacity" not in vllm

    api = summarize_api_pressure(
        [("api-instance", {"associated_running_trials": 3}, {})]
    )
    assert api["associated_running_trials"] == 3
    assert api["pressure_score"] is None
    assert api["provider_quota_available"] is False


def test_vllm_pressure_marks_missing_and_stale_telemetry_explicitly() -> None:
    pending = summarize_vllm_pressure([("vllm", {}, {})])
    assert pending["level"] == "unknown"
    assert pending["telemetry_available"] is False

    stale = summarize_vllm_pressure(
        [
            (
                "vllm",
                {
                    "health_feedback": {
                        "pressure_score": 0.1,
                        "requests_running": 1,
                        "sample_age_s": 45,
                    }
                },
                {},
            )
        ]
    )
    assert stale["level"] == "unknown"
    assert stale["telemetry_available"] is True
    assert stale["telemetry_stale"] is True
