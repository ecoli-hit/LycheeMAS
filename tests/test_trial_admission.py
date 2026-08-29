from __future__ import annotations

from lychee_mas.eval.scheduling.admission import rebalance_trial_admissions


def _instance(instance_id: str, *, priority: int = 100, status: str = "queued") -> dict:
    return {
        "id": instance_id,
        "status": status,
        "configuration_status": "current",
        "queue": {"priority": priority},
        "created_at_utc": f"2026-08-20T00:00:{priority:02d}Z",
    }


def _allocation(instance_id: str, *, mode: str, maximum: int, remaining: int) -> dict:
    return {
        "instance_id": instance_id,
        "trial_concurrency_mode": mode,
        "max_trial_slots": maximum,
        "remaining_work": remaining,
        "current_trial_slots": 0,
        "in_flight_trial_slots": 0,
        "segment_started_trials": 0,
        "admission_total_issued": 0,
        "runner_admission_total_observed": 0,
        "trial_admission_protocol": True,
        "deployment_capacities": {"model": 64},
    }


def _tick(
    *,
    active: dict[str, dict] | None = None,
    pending: dict[str, dict] | None = None,
    instances: dict[str, dict] | None = None,
    pressure: bool = False,
    hard_limit: int = 64,
):
    return rebalance_trial_admissions(
        active_allocations=active or {},
        pending_allocations=pending or {},
        instances=instances or {},
        scheduler_accepting_pending=True,
        max_parallel_instances=16,
        max_running_trials=hard_limit,
        now_monotonic=10.0,
        last_admission_monotonic=0.0,
        sample_interval_s=2.0,
        max_new_trials_per_tick=4,
        allocation_pressure_block_reason=lambda _: "overloaded" if pressure else None,
        running_pool_pressure_block_reason=lambda *_: "overloaded" if pressure else None,
    )


def test_pending_fixed_instance_participates_in_normal_priority_admission() -> None:
    pending = {"fixed": _allocation("fixed", mode="fixed", maximum=5, remaining=5)}

    decision = _tick(
        pending=pending,
        instances={"fixed": _instance("fixed", priority=10)},
    )

    assert decision.admitted_count == 4
    assert pending["fixed"]["current_trial_slots"] == 4
    assert pending["fixed"]["admission_wait_reason"] == "next_admission_tick"


def test_pending_fixed_instance_does_not_bypass_first_admission_pressure() -> None:
    pending = {"fixed": _allocation("fixed", mode="fixed", maximum=5, remaining=5)}

    decision = _tick(
        pending=pending,
        instances={"fixed": _instance("fixed", priority=10)},
        pressure=True,
    )

    assert decision.admitted_count == 0
    assert pending["fixed"]["current_trial_slots"] == 0
    assert pending["fixed"]["admission_wait_reason"] == "pressure_paused"


def test_active_fixed_instance_is_replenished_before_pressure_gate() -> None:
    allocation = _allocation("fixed", mode="fixed", maximum=5, remaining=5)
    allocation.update(
        current_trial_slots=2,
        in_flight_trial_slots=2,
        segment_started_trials=2,
        admission_total_issued=2,
        launch_dir="/tmp/fixed",
    )

    decision = _tick(
        active={"fixed": allocation},
        instances={"fixed": _instance("fixed", priority=10, status="running")},
        pressure=True,
    )

    assert decision.admitted_count == 3
    assert allocation["current_trial_slots"] == 5
    assert decision.updates == (("fixed", "/tmp/fixed", 5, True, 5),)


def test_unobserved_runner_admission_is_retried_until_acknowledged() -> None:
    allocation = _allocation("retry", mode="auto", maximum=3, remaining=3)
    allocation.update(
        current_trial_slots=3,
        in_flight_trial_slots=1,
        segment_started_trials=1,
        admission_total_issued=3,
        runner_admission_total_observed=1,
        launch_dir="/tmp/retry",
    )

    first = _tick(
        active={"retry": allocation},
        instances={"retry": _instance("retry", priority=10, status="running")},
    )

    assert first.admitted_count == 0
    assert first.updates == (("retry", "/tmp/retry", 3, True, 3),)

    allocation["runner_admission_total_observed"] = 3
    acknowledged = _tick(
        active={"retry": allocation},
        instances={"retry": _instance("retry", priority=10, status="running")},
    )

    assert acknowledged.updates == ()


def test_general_admission_follows_priority_then_enqueue_order() -> None:
    pending = {
        "low": _allocation("low", mode="auto", maximum=4, remaining=4),
        "high": _allocation("high", mode="auto", maximum=4, remaining=4),
    }
    instances = {
        "low": _instance("low", priority=50),
        "high": _instance("high", priority=10),
    }

    _tick(pending=pending, instances=instances)

    assert pending["high"]["current_trial_slots"] == 4
    assert pending["low"]["current_trial_slots"] == 0
    assert pending["low"]["admission_wait_reason"] == "next_admission_tick"


def test_global_trial_hard_limit_is_a_safety_cap() -> None:
    active = _allocation("active", mode="auto", maximum=4, remaining=4)
    active.update(
        current_trial_slots=3,
        in_flight_trial_slots=3,
        segment_started_trials=3,
        admission_total_issued=3,
    )
    pending = {"next": _allocation("next", mode="auto", maximum=4, remaining=4)}

    decision = _tick(
        active={"active": active},
        pending=pending,
        instances={
            "active": _instance("active", priority=10, status="running"),
            "next": _instance("next", priority=20),
        },
        hard_limit=3,
    )

    assert decision.admitted_count == 0
    assert pending["next"]["admission_wait_reason"] == "running_trial_hard_limit"
