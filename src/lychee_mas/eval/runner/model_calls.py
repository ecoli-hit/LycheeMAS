"""Normalize provider-neutral model-call usage for Trial terminal records."""

from __future__ import annotations

from typing import Any

_DURATION_FIELDS = (
    "client_rate_limiter_wait_s",
    "client_http_request_latency_s",
    "client_response_postprocess_latency_s",
    "client_model_call_wall_time_s",
    "provider_request_queue_latency_s",
    "provider_scheduled_to_first_token_s",
    "provider_generation_latency_s",
    "provider_mean_inter_token_latency_s",
)


def _number(decision: dict[str, Any], *names: str, default: int | float = 0) -> Any:
    for name in names:
        if name in decision and decision[name] is not None:
            return decision[name]
    return default


def normalize_model_call(decision: dict[str, Any]) -> dict[str, Any]:
    """Normalize historical and current backend field names."""

    input_positions = int(_number(decision, "input_positions", "prompt_pos", default=0))
    latent_positions = int(
        _number(decision, "latent_prefix_positions", "prefix_len", default=0)
    )
    text_tokens = int(
        _number(
            decision,
            "text_input_tokens",
            default=max(0, input_positions - latent_positions),
        )
    )
    output_tokens = int(_number(decision, "output_tokens", "gen_tokens", default=0))
    latency = float(
        _number(decision, "model_generation_latency_s", "latency_s", default=0.0)
    )
    normalized = dict(decision)
    normalized.update(
        {
            "input_total_positions": input_positions,
            "input_text_tokens": text_tokens,
            "input_latent_positions": latent_positions,
            "output_text_tokens": output_tokens,
            "model_latency_s": round(latency, 3),
            "input_positions": input_positions,
            "text_input_tokens": text_tokens,
            "latent_prefix_positions": latent_positions,
            "output_tokens": output_tokens,
            "model_generation_latency_s": round(latency, 3),
        }
    )
    return normalized


def _optional_values(model_calls: list[dict[str, Any]], field: str) -> list[float]:
    return [float(call[field]) for call in model_calls if call.get(field) is not None]


def summarize_model_calls(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the provider-neutral usage projection stored with one Trial."""

    input_positions = sum(call["input_positions"] for call in model_calls)
    text_tokens = sum(call["text_input_tokens"] for call in model_calls)
    latent_positions = sum(call["latent_prefix_positions"] for call in model_calls)
    output_tokens = sum(call["output_tokens"] for call in model_calls)
    latency = round(sum(call["model_generation_latency_s"] for call in model_calls), 3)
    timing_summary: dict[str, Any] = {}
    for field in _DURATION_FIELDS:
        values = _optional_values(model_calls, field)
        timing_summary[f"{field}_available_calls"] = len(values)
        timing_summary[f"{field}_total"] = round(sum(values), 6) if values else None
        timing_summary[f"{field}_mean"] = (
            round(sum(values) / len(values), 6) if values else None
        )
    provider_tps = _optional_values(model_calls, "provider_output_tokens_per_second")
    timing_summary["provider_output_tokens_per_second_available_calls"] = len(provider_tps)
    timing_summary["provider_output_tokens_per_second_mean"] = (
        round(sum(provider_tps) / len(provider_tps), 6) if provider_tps else None
    )
    timing_summary["provider_request_metrics_available_calls"] = sum(
        bool(call.get("provider_request_metrics_available")) for call in model_calls
    )
    routing_trace = [
        {
            key: call.get(key)
            for key in (
                "role",
                "turn_index",
                "sender_role",
                "memory_channel",
                "routing_reason",
                "nl_strategy",
                "latent_strategy",
            )
        }
        for call in model_calls
    ]
    return {
        "num_model_calls": len(model_calls),
        "sum_input_total_positions": input_positions,
        "sum_input_text_tokens": text_tokens,
        "sum_input_latent_positions": latent_positions,
        "sum_output_text_tokens": output_tokens,
        "sum_model_latency_s": latency,
        "model_call_count": len(model_calls),
        "input_positions_total": input_positions,
        "text_input_tokens_total": text_tokens,
        "latent_prefix_positions_total": latent_positions,
        "output_tokens_total": output_tokens,
        "model_generation_latency_s_total": latency,
        **timing_summary,
        "model_calls": model_calls,
        "routing_trace": routing_trace,
        "cost_prompt_pos": input_positions,
        "gen_tokens": output_tokens,
        "latency_s": latency,
    }
