"""Trial-level fixed and adaptive admission control for benchmark runners."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any


def _reject_unknown_policy_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: " + ", ".join(sorted(unknown)))


def assess_vllm_pressure(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Classify recent vLLM pressure without hiding the underlying signals."""

    running = max(0.0, float(snapshot.get("requests_running") or 0))
    waiting = max(0.0, float(snapshot.get("requests_waiting") or 0))
    cache = max(0.0, float(snapshot.get("gpu_cache_usage_fraction") or 0))
    queue_s = max(0.0, float(snapshot.get("queue_time_window_mean_s") or 0))
    queue_p95_s = max(0.0, float(snapshot.get("queue_time_window_p95_s") or 0))
    ttft_s = max(0.0, float(snapshot.get("time_to_first_token_window_mean_s") or 0))
    ttft_p95_s = max(0.0, float(snapshot.get("time_to_first_token_window_p95_s") or 0))
    e2e_s = max(0.0, float(snapshot.get("e2e_latency_window_mean_s") or 0))
    preemptions = max(0.0, float(snapshot.get("preemptions_total_window") or 0))
    waiting_ratio = waiting / max(1.0, running)

    score = min(
        1.0,
        0.25 * min(1.0, waiting_ratio / 0.5)
        + 0.30 * min(1.0, cache / 0.92)
        + 0.20 * min(1.0, queue_s / 5.0)
        + 0.20 * min(1.0, ttft_s / 6.0)
        + 0.05 * min(1.0, preemptions),
    )
    overload_reasons: list[str] = []
    if snapshot.get("status") != "ok":
        overload_reasons.append("metrics_unavailable")
    if preemptions > 0:
        overload_reasons.append("preemption")
    if cache >= 0.92:
        overload_reasons.append("kv_cache")
    if waiting_ratio >= 0.5 and waiting >= 2:
        overload_reasons.append("waiting_queue")
    if queue_s >= 5.0:
        overload_reasons.append("queue_latency")
    if queue_p95_s >= 10.0:
        overload_reasons.append("queue_latency_p95")
    if ttft_s >= 6.0:
        overload_reasons.append("ttft")
    if ttft_p95_s >= 12.0:
        overload_reasons.append("ttft_p95")
    healthy = (
        snapshot.get("status") == "ok"
        and preemptions == 0
        and cache < 0.80
        and waiting_ratio < 0.10
        and queue_s < 1.5
        and (queue_p95_s == 0 or queue_p95_s < 3.0)
        and ttft_s < 3.0
        and (ttft_p95_s == 0 or ttft_p95_s < 5.0)
    )
    level = "high" if overload_reasons else "medium" if score >= 0.5 or waiting > 0 else "normal"
    return {
        "level": level,
        "score": round(score, 6),
        "overload": bool(overload_reasons),
        "healthy": healthy,
        "reasons": overload_reasons,
        "requests_running": running,
        "requests_waiting": waiting,
        "waiting_running_ratio": waiting_ratio,
        "gpu_cache_usage_fraction": cache,
        "preemptions_window": preemptions,
        "queue_time_window_mean_s": queue_s,
        "queue_time_window_p95_s": queue_p95_s or None,
        "time_to_first_token_window_mean_s": ttft_s,
        "time_to_first_token_window_p95_s": ttft_p95_s or None,
        "e2e_latency_window_mean_s": e2e_s,
        "prefix_cache_hit_ratio_window": snapshot.get("prefix_cache_hit_ratio_window"),
    }


def normalize_deployment_admission_policy(
    value: dict[str, Any] | None, *, hard_capacity: int
) -> dict[str, Any]:
    """Normalize shared-service admission control independently of vLLM startup."""

    policy = dict(value or {})
    _reject_unknown_policy_fields(
        policy,
        {
            "mode",
            "minimum",
            "initial",
            "maximum",
            "increase_step",
            "decrease_factor",
            "healthy_polls_required",
        },
        "deployment admission policy",
    )
    mode = str(policy.get("mode") or "auto")
    if mode not in {"fixed", "auto"}:
        raise ValueError("deployment admission mode must be fixed or auto")
    maximum = int(policy.get("maximum") or hard_capacity)
    minimum = int(policy.get("minimum") or min(4, maximum))
    initial = int(policy.get("initial") or maximum)
    increase_step = int(policy.get("increase_step") or 2)
    decrease_factor = float(policy.get("decrease_factor") or 0.75)
    healthy_polls_required = int(policy.get("healthy_polls_required") or 3)
    if min(hard_capacity, minimum, initial, maximum, increase_step, healthy_polls_required) < 1:
        raise ValueError("deployment admission capacities must be positive")
    if maximum > hard_capacity:
        raise ValueError("deployment admission maximum cannot exceed request max_concurrency")
    if not minimum <= initial <= maximum:
        raise ValueError("deployment admission requires minimum <= initial <= maximum")
    if not 0 < decrease_factor < 1:
        raise ValueError("deployment admission decrease_factor must be between 0 and 1")
    return {
        "mode": mode,
        "minimum": minimum,
        "initial": initial,
        "maximum": maximum,
        "increase_step": increase_step,
        "decrease_factor": decrease_factor,
        "healthy_polls_required": healthy_polls_required,
    }


def normalize_concurrency_policy(
    value: dict[str, Any] | None, *, configured_concurrency: int
) -> dict[str, Any]:
    policy = dict(value or {})
    _reject_unknown_policy_fields(
        policy,
        {
            "mode",
            "minimum",
            "initial",
            "maximum",
            "increase_step",
            "decrease_factor",
            "control_window_trials",
        },
        "trial concurrency policy",
    )
    mode = str(policy.get("mode") or "fixed")
    if mode not in {"fixed", "auto"}:
        raise ValueError("concurrency policy mode must be fixed or auto")
    if mode == "fixed":
        if configured_concurrency < 1:
            raise ValueError("configured concurrency must be positive")
        return {
            "mode": "fixed",
            "initial": configured_concurrency,
            "minimum": configured_concurrency,
            "maximum": configured_concurrency,
            "increase_step": 1,
            "decrease_factor": 0.5,
            "control_window_trials": max(1, configured_concurrency),
        }
    maximum = int(policy.get("maximum") or configured_concurrency)
    minimum = int(policy.get("minimum") or 1)
    initial = int(policy.get("initial") or (configured_concurrency if mode == "fixed" else minimum))
    increase_step = int(policy.get("increase_step") or 1)
    decrease_factor = float(policy.get("decrease_factor") or 0.5)
    window = int(policy.get("control_window_trials") or max(8, initial))
    if min(minimum, initial, maximum, increase_step, window) < 1:
        raise ValueError("concurrency policy integer values must be positive")
    if not minimum <= initial <= maximum:
        raise ValueError("concurrency policy requires minimum <= initial <= maximum")
    if not 0 < decrease_factor < 1:
        raise ValueError("concurrency policy decrease_factor must be between 0 and 1")
    return {
        "mode": mode,
        "initial": initial,
        "minimum": minimum,
        "maximum": maximum,
        "increase_step": increase_step,
        "decrease_factor": decrease_factor,
        "control_window_trials": window,
    }


class TrialConcurrencyController:
    """AIMD Trial admission controller; service capacity remains unchanged."""

    def __init__(
        self,
        policy: dict[str, Any],
        *,
        external_limit: int | None = None,
        external_authoritative: bool = False,
    ) -> None:
        self.policy = dict(policy)
        self.target = int(policy["initial"])
        self.external_limit = self._normalize_external_limit(external_limit)
        self.external_authoritative = bool(external_authoritative)
        self.active = 0
        self._condition = asyncio.Condition()
        self._window_completed = 0
        self._window_overloads = 0
        self.history = [self._event("initial", self.effective_target, self.effective_target)]

    @property
    def effective_target(self) -> int:
        """Return the concurrency admitted by the active control authority."""

        if self.external_limit is None:
            return self.target
        if self.external_authoritative:
            return max(0, min(int(self.policy["maximum"]), self.external_limit))
        return max(0, min(self.target, self.external_limit))

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.effective_target)
            self.active += 1

    async def release(self) -> None:
        async with self._condition:
            self.active = max(0, self.active - 1)
            self._condition.notify_all()

    async def observe(
        self, *, record: dict[str, Any], service_metrics: list[dict[str, Any]] | None = None
    ) -> dict[str, Any] | None:
        if self.policy["mode"] != "auto" or self.external_authoritative:
            return None
        self._window_completed += 1
        error_text = " ".join(
            str(record.get(key) or "") for key in ("error_type", "error_message")
        ).lower()
        overload = record.get("status") == "error" and any(
            marker in error_text
            for marker in ("timeout", "rate limit", "overload", "out of memory", "oom", "503")
        )
        metrics = [item for item in service_metrics or [] if item.get("status") == "ok"]
        service_pressure = [assess_vllm_pressure(item) for item in metrics]
        overload = overload or any(item["overload"] for item in service_pressure)
        self._window_overloads += int(overload)
        if self._window_completed < int(self.policy["control_window_trials"]):
            return None
        previous = self.target
        if self._window_overloads:
            self.target = max(
                int(self.policy["minimum"]),
                int(math.floor(self.target * float(self.policy["decrease_factor"]))),
            )
            reason = "overload_signal"
        else:
            self.target = min(
                int(self.policy["maximum"]),
                self.target + int(self.policy["increase_step"]),
            )
            reason = "healthy_window"
        self._window_completed = 0
        self._window_overloads = 0
        if self.target == previous:
            return None
        event = self._event(reason, previous, self.target)
        self.history.append(event)
        async with self._condition:
            self._condition.notify_all()
        return event

    async def set_external_limit(
        self,
        value: int | None,
        *,
        reason: str = "scheduler_allocation",
    ) -> dict[str, Any] | None:
        """Apply a scheduler cap without changing the runner's AIMD state."""

        normalized = self._normalize_external_limit(value)
        previous = self.effective_target
        if normalized == self.external_limit:
            return None
        self.external_limit = normalized
        current = self.effective_target
        event = self._event(reason, previous, current)
        event["adaptive_target"] = self.target
        event["external_limit"] = self.external_limit
        event["authority"] = "external" if self.external_authoritative else "runner"
        self.history.append(event)
        async with self._condition:
            self._condition.notify_all()
        return event

    @staticmethod
    def _normalize_external_limit(value: int | None) -> int | None:
        if value is None:
            return None
        normalized = int(value)
        if normalized < 0:
            raise ValueError("external concurrency limit must be non-negative or null")
        return normalized

    def _event(self, reason: str, previous: int, target: int) -> dict[str, Any]:
        return {
            "timestamp_unix_s": round(time.time(), 6),
            "reason": reason,
            "previous": previous,
            "target": target,
        }
