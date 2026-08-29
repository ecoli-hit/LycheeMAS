"""Pure Trial-admission policy for the Eval Studio scheduler.

The scheduler manager owns threads, processes, files, and locks.  This module
owns only the deterministic state transition from queue/pool observations to
new Trial permits.  Keeping that boundary explicit makes priority, fixed
concurrency, dynamic concurrency, and pressure behavior independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Allocation = dict[str, Any]
PressureCheck = Callable[[Allocation], str | None]
PoolPressureCheck = Callable[[list[Allocation], set[str]], str | None]
ControlUpdate = tuple[str, str, int, bool, int]


@dataclass(frozen=True)
class AdmissionDecision:
    """Observable result of one scheduler admission tick."""

    updates: tuple[ControlUpdate, ...]
    admitted_count: int
    admitted_at_monotonic: float | None


def _priority_key(
    allocation: Allocation,
    instances: dict[str, dict[str, Any]],
) -> tuple[int, str, str]:
    instance = instances.get(str(allocation["instance_id"]), {})
    return (
        int((instance.get("queue") or {}).get("priority", 100)),
        str(instance.get("queued_at_utc") or instance.get("created_at_utc") or ""),
        str(allocation["instance_id"]),
    )


def trial_demand(allocation: Allocation) -> int:
    """Return the bounded Trial demand visible to the central scheduler."""

    observed = (
        None
        if allocation.get("concurrency_authority") == "scheduler"
        else allocation.get("observed_trial_slot_demand")
    )
    configured = int(allocation.get("max_trial_slots") or 1)
    remaining_value = allocation.get("remaining_work")
    remaining = configured if remaining_value is None else int(remaining_value)
    return max(
        0,
        min(
            configured,
            remaining,
            int(observed) if observed is not None else configured,
        ),
    )


def rebalance_trial_admissions(
    *,
    active_allocations: dict[str, Allocation],
    pending_allocations: dict[str, Allocation],
    instances: dict[str, dict[str, Any]],
    scheduler_accepting_pending: bool,
    max_parallel_instances: int,
    max_running_trials: int,
    now_monotonic: float,
    last_admission_monotonic: float,
    sample_interval_s: float,
    max_new_trials_per_tick: int,
    allocation_pressure_block_reason: PressureCheck,
    running_pool_pressure_block_reason: PoolPressureCheck,
) -> AdmissionDecision:
    """Apply one deterministic two-phase admission tick.

    Phase 1 restores fixed-concurrency ExperimentInstances already in the
    running pool.  Phase 2 walks the persistent priority queue one Trial at a
    time when current resource pressure permits it.  Pending fixed instances
    participate in phase 2; once their first Trial is admitted and their runner
    starts, later ticks maintain their fixed concurrency through phase 1.
    """

    if not active_allocations and not pending_allocations:
        return AdmissionDecision((), 0, None)

    active = list(active_allocations.values())
    active_ids = {str(item["instance_id"]) for item in active}
    pending_limit = (
        max(0, int(max_parallel_instances) - len(active))
        if scheduler_accepting_pending
        else 0
    )
    pending = sorted(
        (
            item
            for instance_id, item in pending_allocations.items()
            if instance_id not in active_ids
            and instances.get(instance_id, {}).get("status") == "queued"
            and instances.get(instance_id, {}).get("configuration_status") == "current"
        ),
        key=lambda item: _priority_key(item, instances),
    )[:pending_limit]
    selected_pending_ids = {str(item["instance_id"]) for item in pending}
    for instance_id, allocation in pending_allocations.items():
        if instance_id not in selected_pending_ids:
            allocation["current_trial_slots"] = 0

    ordered = sorted(
        [*active, *pending],
        key=lambda item: _priority_key(item, instances),
    )
    if not ordered:
        return AdmissionDecision((), 0, None)

    previous = {
        str(item["instance_id"]): int(item.get("current_trial_slots") or 0)
        for item in ordered
    }
    previous_admissions = {
        str(item["instance_id"]): int(item.get("admission_total_issued") or 0)
        for item in ordered
    }

    total_occupied = 0
    for allocation in ordered:
        allocation.pop("admission_wait_reason", None)
        allocation.pop("required_gang_trial_slots", None)
        target = trial_demand(allocation)
        instance_id = str(allocation["instance_id"])
        running = (
            max(0, int(allocation.get("in_flight_trial_slots") or 0))
            if instance_id in active_ids
            else 0
        )
        started = max(0, int(allocation.get("segment_started_trials") or 0))
        issued = max(started, int(allocation.get("admission_total_issued") or 0))
        supports_admission = bool(allocation.get("trial_admission_protocol", True))
        if supports_admission:
            launching = max(0, issued - started)
            occupied = min(target, running + launching)
        else:
            occupied = min(target, max(running, previous.get(instance_id, 0)))
            launching = max(0, occupied - running)
        allocation["admission_total_issued"] = issued
        allocation["current_trial_slots"] = occupied
        allocation["launching_trial_count"] = launching
        total_occupied += occupied

    global_available = max(0, int(max_running_trials) - total_occupied)
    sample_ready = now_monotonic - last_admission_monotonic >= sample_interval_s
    admitted_this_tick = 0

    def issue(allocation: Allocation, count: int) -> None:
        nonlocal admitted_this_tick, global_available
        for _ in range(max(0, count)):
            allocation["current_trial_slots"] = int(
                allocation.get("current_trial_slots") or 0
            ) + 1
            allocation["launching_trial_count"] = int(
                allocation.get("launching_trial_count") or 0
            ) + 1
            allocation["admission_total_issued"] = int(
                allocation.get("admission_total_issued") or 0
            ) + 1
            global_available -= 1
            admitted_this_tick += 1

    # Existing fixed groups are a maintenance obligation of the running pool.
    for allocation in ordered:
        instance_id = str(allocation["instance_id"])
        fixed = str(allocation.get("trial_concurrency_mode") or "auto") == "fixed"
        if instance_id not in active_ids or not fixed:
            continue
        current = int(allocation.get("current_trial_slots") or 0)
        unfilled = trial_demand(allocation) - current
        if unfilled <= 0:
            continue
        if global_available <= 0:
            allocation["admission_wait_reason"] = "running_trial_hard_limit"
            continue
        grant = min(unfilled, global_available)
        issue(allocation, grant)
        if grant < unfilled:
            allocation["admission_wait_reason"] = "running_trial_hard_limit"

    pool_pressure_block = running_pool_pressure_block_reason(ordered, active_ids)
    admission_budget = (
        min(
            max(0, int(max_new_trials_per_tick) - admitted_this_tick),
            global_available,
        )
        if sample_ready and pool_pressure_block is None
        else 0
    )

    if pool_pressure_block is not None:
        for allocation in ordered:
            if trial_demand(allocation) <= int(allocation.get("current_trial_slots") or 0):
                continue
            instance_id = str(allocation["instance_id"])
            fixed_in_pool = (
                instance_id in active_ids
                and str(allocation.get("trial_concurrency_mode") or "auto") == "fixed"
            )
            if not fixed_in_pool and not allocation.get("admission_wait_reason"):
                allocation["admission_wait_reason"] = "pressure_paused"
    else:
        for allocation in ordered:
            instance_id = str(allocation["instance_id"])
            fixed_in_pool = (
                instance_id in active_ids
                and str(allocation.get("trial_concurrency_mode") or "auto") == "fixed"
            )
            if fixed_in_pool:
                continue
            current = int(allocation.get("current_trial_slots") or 0)
            unfilled = trial_demand(allocation) - current
            if unfilled <= 0:
                continue
            if allocation_pressure_block_reason(allocation):
                allocation["admission_wait_reason"] = "pressure_paused"
                continue
            if global_available <= 0:
                allocation["admission_wait_reason"] = "running_trial_hard_limit"
                continue
            if admission_budget <= 0:
                allocation["admission_wait_reason"] = "next_admission_tick"
                continue
            grant = min(unfilled, global_available, admission_budget)
            issue(allocation, grant)
            admission_budget -= grant
            if grant < unfilled:
                allocation["admission_wait_reason"] = (
                    "running_trial_hard_limit"
                    if global_available <= 0
                    else "next_admission_tick"
                )

    updates: list[ControlUpdate] = []
    for queue_rank, allocation in enumerate(ordered, 1):
        current = int(allocation.get("current_trial_slots") or 0)
        in_flight = max(0, int(allocation.get("in_flight_trial_slots") or 0))
        allocation["launching_trial_count"] = max(0, current - in_flight)
        allocation["queue_rank"] = queue_rank
        instance_id = str(allocation["instance_id"])
        if instance_id not in active_ids:
            continue
        supports_admission = bool(allocation.get("trial_admission_protocol", True))
        desired_admission_total = int(allocation.get("admission_total_issued") or 0)
        runner_observed_total = int(
            allocation.get("runner_admission_total_observed") or 0
        )
        control_changed = (
            (
                previous_admissions.get(instance_id) != desired_admission_total
                or runner_observed_total < desired_admission_total
            )
            if supports_admission
            else previous.get(instance_id) != current
        )
        if control_changed:
            updates.append(
                (
                    instance_id,
                    str(allocation.get("launch_dir") or ""),
                    current,
                    supports_admission,
                    desired_admission_total,
                )
            )

    return AdmissionDecision(
        updates=tuple(updates),
        admitted_count=admitted_this_tick,
        admitted_at_monotonic=now_monotonic if admitted_this_tick else None,
    )
