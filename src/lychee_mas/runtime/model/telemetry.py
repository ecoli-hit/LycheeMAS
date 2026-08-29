"""Provider-neutral model timing fields shared by framework adapters."""

from __future__ import annotations

from typing import Any


def optional_seconds(result: Any, name: str) -> float | None:
    """Read one optional duration-like field from a backend result."""

    value = getattr(result, name, None)
    return round(float(value), 6) if value is not None else None


def model_timing_fields(result: Any) -> dict[str, Any]:
    """Expose client and provider timings with explicit availability."""

    provider_metrics = dict(getattr(result, "provider_request_metrics", {}) or {})
    return {
        "client_rate_limiter_wait_s": optional_seconds(
            result, "client_rate_limiter_wait_s"
        ),
        "client_http_request_latency_s": optional_seconds(
            result, "client_http_request_latency_s"
        ),
        "client_response_postprocess_latency_s": optional_seconds(
            result, "client_response_postprocess_latency_s"
        ),
        "client_model_call_wall_time_s": optional_seconds(
            result, "client_model_call_wall_time_s"
        ),
        "provider_request_metrics_available": bool(provider_metrics),
        "provider_request_queue_latency_s": optional_seconds(
            result, "provider_request_queue_latency_s"
        ),
        "provider_scheduled_to_first_token_s": optional_seconds(
            result, "provider_scheduled_to_first_token_s"
        ),
        "provider_generation_latency_s": optional_seconds(
            result, "provider_generation_latency_s"
        ),
        "provider_mean_inter_token_latency_s": optional_seconds(
            result, "provider_mean_inter_token_latency_s"
        ),
        "provider_output_tokens_per_second": optional_seconds(
            result, "provider_output_tokens_per_second"
        ),
        "provider_request_metrics": provider_metrics,
    }


__all__ = ["model_timing_fields", "optional_seconds"]
