"""Resource-specific pressure summaries used by Trial admission control.

This module deliberately keeps vLLM, API, and GPU signals separate.  A caller
may stop admitting work when any relevant resource is overloaded, but the
individual scores retain their own units and evidence.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

PressureRow = tuple[str, dict[str, Any], dict[str, Any]]


def summarize_vllm_pressure(rows: list[PressureRow]) -> dict[str, Any]:
    """Aggregate request, queue, TTFT, KV-cache, and preemption observations."""

    if not rows:
        return {"level": "unknown", "summary": "无活动 vLLM", "detail": ""}
    feedback = [dict(usage.get("health_feedback") or {}) for _, usage, _ in rows]
    observed_feedback = [item for item in feedback if item]
    if not observed_feedback:
        return {
            "level": "unknown",
            "summary": "vLLM telemetry pending",
            "detail": "Deployment 已关联到运行池，但尚未收到可用于准入控制的 vLLM 指标。",
            "pressure_score": None,
            "telemetry_available": False,
        }
    sample_ages = [
        float(item.get("sample_age_s") or 0.0)
        for item in observed_feedback
        if item.get("sample_age_s") is not None
    ]
    max_sample_age_s = max(sample_ages, default=0.0)
    waiting = sum(float(item.get("requests_waiting") or 0) for item in feedback)
    running = sum(float(item.get("requests_running") or 0) for item in feedback)
    cache = max(
        (float(item.get("gpu_cache_usage_fraction") or 0) for item in feedback),
        default=0.0,
    )
    pressure_score = max(
        (float(item.get("pressure_score") or 0) for item in feedback), default=0.0
    )
    queue_s = max(
        (float(item.get("queue_time_window_mean_s") or 0) for item in feedback),
        default=0.0,
    )
    queue_p95_s = max(
        (float(item.get("queue_time_window_p95_s") or 0) for item in feedback),
        default=0.0,
    )
    ttft_s = max(
        (float(item.get("time_to_first_token_window_mean_s") or 0) for item in feedback),
        default=0.0,
    )
    ttft_p95_s = max(
        (float(item.get("time_to_first_token_window_p95_s") or 0) for item in feedback),
        default=0.0,
    )
    preemptions = sum(float(item.get("preemptions_window") or 0) for item in feedback)
    reasons = sorted(
        {str(reason) for item in feedback for reason in item.get("pressure_reasons") or []}
    )
    telemetry_stale = bool(sample_ages) and max_sample_age_s > 30.0
    level = (
        "unknown"
        if telemetry_stale
        else "high"
        if reasons
        else "medium"
        if pressure_score >= 0.5 or waiting > 0
        else "normal"
    )
    return {
        "level": level,
        "summary": f"running {running:g} · waiting {waiting:g}",
        "detail": (
            f"KV {cache:.1%} · queue mean/p95 {queue_s:.2f}/{queue_p95_s:.2f}s · "
            f"TTFT mean/p95 {ttft_s:.2f}/{ttft_p95_s:.2f}s"
            + (f" · {', '.join(reasons)}" if reasons else "")
            + (f" · telemetry stale {max_sample_age_s:.1f}s" if telemetry_stale else "")
        ),
        "requests_running": running,
        "requests_waiting": waiting,
        "gpu_cache_usage_fraction": cache,
        "pressure_score": pressure_score,
        "queue_time_window_mean_s": queue_s,
        "queue_time_window_p95_s": queue_p95_s or None,
        "time_to_first_token_window_mean_s": ttft_s,
        "time_to_first_token_window_p95_s": ttft_p95_s or None,
        "preemptions_window": preemptions,
        "reasons": reasons,
        "telemetry_available": True,
        "telemetry_stale": telemetry_stale,
        "max_sample_age_s": max_sample_age_s if sample_ages else None,
    }


def summarize_api_pressure(rows: list[PressureRow]) -> dict[str, Any]:
    """Report only API pressure that the platform can observe truthfully."""

    if not rows:
        return {"level": "unknown", "summary": "无活动 API", "detail": ""}
    associated_trials = sum(
        int(usage.get("associated_running_trials") or 0) for _, usage, _ in rows
    )
    return {
        "level": "unknown",
        "summary": f"关联 {associated_trials} 个运行中 Trial",
        "detail": (
            "Trial 数不等于并发 API 请求数；Provider quota 和跨进程客户端信号"
            "当前不可统一读取，429、timeout 与 retry 写入 EventLog。"
        ),
        "associated_running_trials": associated_trials,
        "pressure_score": None,
        "provider_quota_available": False,
    }


def sample_gpu_pressure(
    gpu_ids: set[int],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Sample GPU allocation safety signals without treating utilization as overload."""

    if not gpu_ids:
        return {"level": "unknown", "summary": "无活动 GPU", "detail": ""}
    try:
        result = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )

        def optional_number(value: str) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        devices = []
        for line in result.stdout.splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) != 7 or not values[0].isdigit():
                continue
            index = int(values[0])
            if index not in gpu_ids:
                continue
            devices.append(
                {
                    "index": index,
                    "utilization_percent": float(values[1]),
                    "memory_used_mib": float(values[2]),
                    "memory_total_mib": float(values[3]),
                    "temperature_c": optional_number(values[4]),
                    "power_draw_w": optional_number(values[5]),
                    "power_limit_w": optional_number(values[6]),
                }
            )
        if not devices:
            raise ValueError("requested GPU IDs were not reported")
        max_util = max(float(item["utilization_percent"] or 0.0) for item in devices)
        max_memory = max(
            float(item["memory_used_mib"] or 0.0)
            / max(1.0, float(item["memory_total_mib"] or 0.0))
            for item in devices
        )
        temperatures = [
            float(item["temperature_c"])
            for item in devices
            if item["temperature_c"] is not None
        ]
        power_ratios = [
            float(item["power_draw_w"]) / max(1.0, float(item["power_limit_w"]))
            for item in devices
            if item["power_draw_w"] is not None and item["power_limit_w"] is not None
        ]
        max_temperature = max(temperatures, default=0.0)
        max_power = max(power_ratios, default=0.0)
        pressure_score = max(
            min(1.0, max(0.0, (max_memory - 0.85) / 0.13)),
            min(1.0, max(0.0, (max_temperature - 65.0) / 20.0)),
            min(1.0, max(0.0, (max_power - 0.75) / 0.23)),
        )
        temperature_text = f"{max_temperature:.0f}°C" if temperatures else "unavailable"
        power_text = f"{max_power:.1%}" if power_ratios else "unavailable"
        level = (
            "high"
            if max_temperature >= 85 or max_power >= 0.98 or max_memory >= 0.98
            else "medium"
            if max_temperature >= 78 or max_power >= 0.90 or max_memory >= 0.95
            else "normal"
        )
        return {
            "level": level,
            "summary": (
                f"GPU {','.join(str(item['index']) for item in devices)} "
                f"· util {max_util:.0f}%"
            ),
            "detail": (
                f"显存分配 {max_memory:.1%} · 温度 {temperature_text} · "
                f"功耗上限比 {power_text}；GPU util 是吞吐利用率，不单独判定过载"
            ),
            "max_utilization_percent": max_util,
            "max_memory_allocation_fraction": max_memory,
            "max_temperature_c": max_temperature,
            "max_power_limit_fraction": max_power,
            "pressure_score": round(pressure_score, 6),
            "devices": devices,
        }
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as exc:
        return {
            "level": "unknown",
            "summary": "GPU 压力不可用",
            "detail": f"{type(exc).__name__}: {exc}",
        }
