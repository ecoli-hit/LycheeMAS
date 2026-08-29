"""Deployment-owned provider and GPU health feedback for Trial admission."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from ...runtime.execution.concurrency import assess_vllm_pressure
from ..scheduling.pressure import sample_gpu_pressure


class DeploymentHealthMonitor:
    """Project allocation demand and provider telemetry into capacity feedback."""

    def __init__(self, deployment_registry=None, snapshot_reader=None) -> None:
        self.deployment_registry = deployment_registry
        self.snapshot_reader = snapshot_reader or latest_vllm_snapshot
        self._latest_gpu_pressure: dict[str, Any] = {
            "level": "unknown",
            "summary": "无活动 GPU",
            "detail": "",
        }
        self._latest_deployment_health: dict[str, dict[str, Any]] = {}

    def gpu_pressure(self) -> dict[str, Any]:
        return dict(self._latest_gpu_pressure)

    def snapshot(self) -> dict[str, Any]:
        return {
            "deployments": {
                key: dict(value) for key, value in self._latest_deployment_health.items()
            },
            "gpu": self.gpu_pressure(),
        }

    def update(
        self,
        allocations: list[dict[str, Any]],
        current_health: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        hard_capacities: dict[str, int] = {}
        policies: dict[str, dict[str, Any]] = {}
        in_flight_counts: dict[str, int] = {}
        demand_counts: dict[str, int] = {}
        latest: dict[str, dict[str, Any]] = {}
        active_gpu_ids: set[int] = set()
        deployments = self._active_deployments()

        for allocation in allocations:
            demand = max(0, int(allocation.get("max_trial_slots") or 0))
            remaining = allocation.get("remaining_work")
            if remaining is not None:
                demand = min(demand, max(0, int(remaining)))
            for deployment_id, capacity in allocation.get("deployment_capacities", {}).items():
                in_flight_counts[deployment_id] = in_flight_counts.get(deployment_id, 0) + max(
                    0,
                    int(allocation.get("in_flight_trial_slots") or 0),
                )
                demand_counts[deployment_id] = demand_counts.get(deployment_id, 0) + demand
                hard_capacities[deployment_id] = min(
                    hard_capacities.get(deployment_id, int(capacity)),
                    int(capacity),
                )
                policy = dict(
                    (allocation.get("deployment_admission_policies") or {}).get(deployment_id)
                    or {}
                )
                if policy:
                    policies[deployment_id] = policy
                snapshot = self.snapshot_reader(
                    str(allocation.get("run_dir") or ""),
                    deployment_id,
                )
                if snapshot and float(snapshot.get("timestamp_unix_s") or 0) >= float(
                    latest.get(deployment_id, {}).get("timestamp_unix_s") or 0
                ):
                    latest[deployment_id] = snapshot
                deployment = deployments.get(deployment_id) or {}
                for token in str(deployment.get("cuda_visible_devices") or "").split(","):
                    if token.strip().isdigit():
                        active_gpu_ids.add(int(token.strip()))

        gpu_pressure = sample_gpu_pressure(active_gpu_ids)
        self._latest_gpu_pressure = dict(gpu_pressure)
        next_health = {
            key: dict(value)
            for key, value in current_health.items()
            if key in hard_capacities
        }
        for deployment_id, hard_capacity in hard_capacities.items():
            next_health[deployment_id] = self._deployment_feedback(
                hard_capacity=hard_capacity,
                policy=policies.get(deployment_id, {}),
                state=next_health.get(deployment_id, {}),
                snapshot=latest.get(deployment_id),
                in_flight=in_flight_counts.get(deployment_id, 0),
                demand=demand_counts.get(deployment_id, 0),
                gpu_pressure=gpu_pressure,
            )
        self._latest_deployment_health = {
            key: dict(value) for key, value in next_health.items()
        }
        return next_health

    def _active_deployments(self) -> dict[str, dict[str, Any]]:
        if self.deployment_registry is None:
            return {}
        try:
            return {
                str(item.get("id") or ""): item
                for item in self.deployment_registry.instances(probe=False)
            }
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _deployment_feedback(
        *,
        hard_capacity: int,
        policy: dict[str, Any],
        state: dict[str, Any],
        snapshot: dict[str, Any] | None,
        in_flight: int,
        demand: int,
        gpu_pressure: dict[str, Any],
    ) -> dict[str, Any]:
        policy = {
            "mode": "auto",
            "minimum": min(4, hard_capacity),
            "initial": hard_capacity,
            "maximum": hard_capacity,
            "increase_step": 1,
            "decrease_factor": 0.75,
            "healthy_polls_required": 3,
            **policy,
        }
        state = dict(state)
        effective = max(
            int(policy["minimum"]),
            min(
                int(state.get("effective_capacity") or policy["initial"]),
                int(policy["maximum"]),
                hard_capacity,
            ),
        )
        if not snapshot:
            state.update(effective_capacity=effective, hard_capacity=hard_capacity)
            return state

        sample_timestamp = float(snapshot.get("timestamp_unix_s") or 0)
        sample_age_s = max(0.0, time.time() - sample_timestamp)
        gpu_blocked = str(gpu_pressure.get("level") or "unknown") == "high"
        if in_flight == 0 and sample_age_s > 30.0:
            state.update(
                effective_capacity=effective,
                hard_capacity=hard_capacity,
                capacity_semantics="online_admission_limit",
                capacity_phase="metrics_stale",
                reason="idle_metrics_stale",
                healthy_polls=0,
                sample_age_s=sample_age_s,
                vllm_admission_blocked=False,
                gpu_admission_blocked=gpu_blocked,
                admission_reasons=["gpu_safety_limit"] if gpu_blocked else [],
            )
            return state
        if state.get("sample_timestamp_unix_s") == snapshot.get("timestamp_unix_s"):
            state.update(effective_capacity=effective, hard_capacity=hard_capacity)
            return state

        assessment = assess_vllm_pressure(snapshot)
        vllm_overload = bool(assessment["overload"])
        healthy = bool(assessment["healthy"])
        pressure_reasons = list(assessment["reasons"])
        admission_reasons = list(pressure_reasons)
        if gpu_blocked:
            admission_reasons.append("gpu_safety_limit")
        healthy_polls = int(state.get("healthy_polls") or 0)
        observed_load = max(
            in_flight,
            int(
                float(assessment["requests_running"])
                + float(assessment["requests_waiting"])
            ),
        )
        backlog = demand > in_flight
        saturation_threshold = max(
            int(policy["minimum"]),
            int(math.ceil(effective * 0.8)),
        )
        saturated = observed_load >= saturation_threshold
        observed_safe_capacity = max(
            int(state.get("observed_safe_capacity") or 0),
            observed_load if healthy else 0,
        )
        reason = "steady"
        capacity_phase = "observing"
        if policy["mode"] == "fixed":
            effective = int(policy["maximum"])
            healthy_polls = 0
            reason = "fixed_capacity"
            capacity_phase = "fixed"
        elif vllm_overload:
            effective = max(
                in_flight,
                int(policy["minimum"]),
                int(math.floor(effective * float(policy["decrease_factor"]))),
            )
            healthy_polls = 0
            reason = "vllm_overload"
            capacity_phase = "backing_off"
        elif healthy and backlog and saturated:
            healthy_polls += 1
            capacity_phase = "probing"
            if (
                healthy_polls >= int(policy["healthy_polls_required"])
                and effective < int(policy["maximum"])
            ):
                effective = min(
                    int(policy["maximum"]),
                    effective + int(policy["increase_step"]),
                )
                healthy_polls = 0
                reason = "vllm_healthy_window"
        elif healthy and backlog:
            healthy_polls = 0
            reason = "healthy_below_admission_limit"
            capacity_phase = "under_observed"
        elif healthy:
            healthy_polls = 0
            reason = "healthy_without_backlog"
            capacity_phase = "demand_limited"
        else:
            healthy_polls = 0

        return {
            "effective_capacity": effective,
            "capacity_semantics": "online_admission_limit",
            "hard_capacity": hard_capacity,
            "observed_safe_capacity": observed_safe_capacity,
            "observed_load": observed_load,
            "demanded_trial_slots": demand,
            "capacity_phase": capacity_phase,
            "saturated_for_probe": saturated,
            "healthy_polls": healthy_polls,
            "reason": reason,
            "pressure_score": float(assessment["score"]),
            "pressure_reasons": pressure_reasons,
            "admission_reasons": admission_reasons,
            "vllm_admission_blocked": vllm_overload,
            "gpu_admission_blocked": gpu_blocked,
            "requests_waiting": float(assessment["requests_waiting"]),
            "requests_running": assessment["requests_running"],
            "gpu_cache_usage_fraction": float(assessment["gpu_cache_usage_fraction"]),
            "preemptions_window": assessment["preemptions_window"],
            "queue_time_window_mean_s": assessment["queue_time_window_mean_s"],
            "queue_time_window_p95_s": assessment["queue_time_window_p95_s"],
            "time_to_first_token_window_mean_s": assessment[
                "time_to_first_token_window_mean_s"
            ],
            "time_to_first_token_window_p95_s": assessment[
                "time_to_first_token_window_p95_s"
            ],
            "e2e_latency_window_mean_s": assessment["e2e_latency_window_mean_s"],
            "prefix_cache_hit_ratio_window": assessment["prefix_cache_hit_ratio_window"],
            "sample_timestamp_unix_s": snapshot.get("timestamp_unix_s"),
            "sample_age_s": sample_age_s,
        }


def latest_vllm_snapshot(run_dir: str, deployment_id: str) -> dict[str, Any] | None:
    """Read the newest sample for one deployment without loading the full file."""

    if not run_dir:
        return None
    path = Path(run_dir) / "vllm_metrics.jsonl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - 256 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and str(value.get("deployment_id") or "") == deployment_id:
            return value
    return None
