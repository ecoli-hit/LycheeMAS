"""Case-level fixed and adaptive admission control for benchmark runners."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any


def normalize_concurrency_policy(
    value: dict[str, Any] | None, *, configured_concurrency: int
) -> dict[str, Any]:
    policy = dict(value or {})
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
            "control_window_cases": max(1, configured_concurrency),
        }
    maximum = int(policy.get("maximum") or configured_concurrency)
    minimum = int(policy.get("minimum") or 1)
    initial = int(policy.get("initial") or (configured_concurrency if mode == "fixed" else minimum))
    increase_step = int(policy.get("increase_step") or 1)
    decrease_factor = float(policy.get("decrease_factor") or 0.5)
    window = int(policy.get("control_window_cases") or max(8, initial))
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
        "control_window_cases": window,
    }


class CaseConcurrencyController:
    """AIMD admission controller; vLLM startup capacity remains unchanged."""

    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = dict(policy)
        self.target = int(policy["initial"])
        self.active = 0
        self._condition = asyncio.Condition()
        self._window_completed = 0
        self._window_overloads = 0
        self.history = [self._event("initial", self.target, self.target)]

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.target)
            self.active += 1

    async def release(self) -> None:
        async with self._condition:
            self.active = max(0, self.active - 1)
            self._condition.notify_all()

    async def observe(
        self, *, record: dict[str, Any], service_metrics: list[dict[str, Any]] | None = None
    ) -> dict[str, Any] | None:
        if self.policy["mode"] != "auto":
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
        waiting = max((float(item.get("requests_waiting") or 0) for item in metrics), default=0)
        cache = max(
            (float(item.get("gpu_cache_usage_fraction") or 0) for item in metrics),
            default=0,
        )
        overload = overload or waiting > max(1.0, self.target * 0.25) or cache >= 0.95
        self._window_overloads += int(overload)
        if self._window_completed < int(self.policy["control_window_cases"]):
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

    def _event(self, reason: str, previous: int, target: int) -> dict[str, Any]:
        return {
            "timestamp_unix_s": round(time.time(), 6),
            "reason": reason,
            "previous": previous,
            "target": target,
        }
