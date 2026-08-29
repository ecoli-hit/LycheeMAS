"""Read models derived from mutable scheduler allocation state."""

from __future__ import annotations

from typing import Any


def resource_usage(
    active_allocations: dict[str, dict[str, Any]],
    deployment_health: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    usage: dict[str, dict[str, Any]] = {}
    for allocation in active_allocations.values():
        running_trials = max(0, int(allocation.get("in_flight_trial_slots") or 0))
        launching_trials = max(0, int(allocation.get("launching_trial_count") or 0))
        for deployment_id, raw_capacity in allocation.get("deployment_capacities", {}).items():
            row = usage.setdefault(
                deployment_id,
                {
                    "associated_running_trials": 0,
                    "associated_launching_trials": 0,
                    "request_concurrency_limit": int(raw_capacity),
                    "instance_ids": [],
                },
            )
            row["request_concurrency_limit"] = min(
                int(row["request_concurrency_limit"]), int(raw_capacity)
            )
            row["associated_running_trials"] += running_trials
            row["associated_launching_trials"] += launching_trials
            row["instance_ids"].append(allocation["instance_id"])
    for deployment_id, row in usage.items():
        health = deployment_health.get(deployment_id)
        if health:
            row["health_feedback"] = dict(health)
    return usage


def running_trial_pool(
    active_allocations: dict[str, dict[str, Any]],
    pending_allocations: dict[str, dict[str, Any]],
    *,
    hard_limit: int,
) -> dict[str, int]:
    """Return actual and dispatching Trial occupancy across all Runs."""

    allocations = {**pending_allocations, **active_allocations}
    running = sum(
        max(0, int(item.get("in_flight_trial_slots") or 0))
        for item in active_allocations.values()
    )
    admitted = sum(
        max(
            max(0, int(item.get("current_trial_slots") or 0)),
            max(0, int(item.get("in_flight_trial_slots") or 0)),
        )
        for item in allocations.values()
    )
    return {
        "occupied": admitted,
        "running": running,
        "launching": max(0, admitted - running),
        "hard_limit": int(hard_limit),
        "available_before_hard_limit": max(0, int(hard_limit) - admitted),
    }


def trial_queue_snapshot(
    active_allocations: dict[str, dict[str, Any]],
    pending_allocations: dict[str, dict[str, Any]],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe logical per-Experiment queues sharing one running pool."""

    instance_map = {str(item["id"]): item for item in instances}

    def priority_key(allocation: dict[str, Any]) -> tuple[int, str, str]:
        instance = instance_map.get(str(allocation["instance_id"]), {})
        return (
            int((instance.get("queue") or {}).get("priority", 100)),
            str(instance.get("queued_at_utc") or instance.get("created_at_utc") or ""),
            str(allocation["instance_id"]),
        )

    active_ids = set(active_allocations)
    allocations = {**pending_allocations, **active_allocations}
    rows: list[dict[str, Any]] = []
    for rank, allocation in enumerate(sorted(allocations.values(), key=priority_key), 1):
        instance_id = str(allocation["instance_id"])
        instance = instance_map.get(instance_id, {})
        configured = max(1, int(allocation.get("max_trial_slots") or 1))
        remaining = allocation.get("remaining_work")
        if allocation.get("concurrency_authority") == "scheduler":
            requested = min(
                configured,
                max(0, int(remaining)) if remaining is not None else configured,
            )
        else:
            observed_requested = allocation.get("observed_trial_slot_demand")
            requested = (
                configured if observed_requested is None else int(observed_requested)
            )
        running = max(0, int(allocation.get("in_flight_trial_slots") or 0))
        launching = max(0, int(allocation.get("launching_trial_count") or 0))
        remaining_count = max(0, int(remaining)) if remaining is not None else None
        waiting = (
            max(0, remaining_count - running - launching)
            if remaining_count is not None
            else None
        )
        if str(allocation.get("runner_status") or "") in {
            "draining",
            "stopped",
            "paused",
            "complete",
            "completed",
            "complete_with_errors",
            "failed",
            "failed_fast",
        }:
            waiting = 0
        rows.append(
            {
                "rank": rank,
                "instance_id": instance_id,
                "state": "active" if instance_id in active_ids else "queued",
                "runner_status": allocation.get("runner_status"),
                "concurrency_authority": allocation.get("concurrency_authority"),
                "trial_concurrency_mode": allocation.get("trial_concurrency_mode", "auto"),
                "priority": int((instance.get("queue") or {}).get("priority", 100)),
                "queued_at_utc": instance.get("queued_at_utc"),
                "max_trial_concurrency": configured,
                "target_trial_concurrency": max(0, int(requested or 0)),
                "running_trial_count": running,
                "launching_trial_count": launching,
                "waiting_trial_count": waiting,
                "active_trials": list(allocation.get("active_trials") or []),
                "admission_wait_reason": allocation.get("admission_wait_reason"),
                "remaining_trial_count": remaining_count,
            }
        )
    return rows


def trial_sequences(queues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    running_sequence: list[dict[str, Any]] = []
    launching_sequence: list[dict[str, Any]] = []
    waiting_segments: list[dict[str, Any]] = []
    for queue in queues:
        instance_id = str(queue["instance_id"])
        active_trials = list(queue.get("active_trials") or [])
        running_count = int(queue.get("running_trial_count") or 0)
        for index in range(running_count):
            detail = active_trials[index] if index < len(active_trials) else {}
            running_sequence.append(
                {
                    "instance_id": instance_id,
                    "state": "running",
                    "case_id": detail.get("case_id"),
                    "case_order": detail.get("case_order"),
                    "trial_index": detail.get("trial_index"),
                    "started_at_unix_s": detail.get("started_at_unix_s"),
                }
            )
        for _ in range(int(queue.get("launching_trial_count") or 0)):
            launching_sequence.append(
                {
                    "instance_id": instance_id,
                    "state": "launching",
                    "case_id": None,
                    "case_order": None,
                    "trial_index": None,
                }
            )
        waiting_count = queue.get("waiting_trial_count")
        if waiting_count:
            waiting_segments.append(
                {
                    "instance_id": instance_id,
                    "priority": queue.get("priority"),
                    "count": int(waiting_count),
                }
            )
    running_sequence.sort(key=lambda item: float(item.get("started_at_unix_s") or 0))
    return {
        "running_trial_sequence": [*running_sequence, *launching_sequence],
        "waiting_trial_segments": waiting_segments,
    }
