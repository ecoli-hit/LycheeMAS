"""Reproducible cost accounting from immutable Eval Studio run snapshots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _model_calls_from_spans(run_dir: Path) -> list[dict[str, Any]]:
    """Read every completed call, including calls from failed/retried cases."""

    return [
        {
            "deployment_instance_id": span.get("deployment_instance_id"),
            "input_text_tokens": span.get("input_text_tokens", 0),
            "input_cached_tokens": span.get("input_cached_tokens"),
            "output_text_tokens": span.get("output_text_tokens", 0),
            "output_reasoning_tokens": span.get("output_reasoning_tokens"),
            "output_answer_tokens": span.get("output_answer_tokens"),
            "invocation_policy": span.get("invocation_policy") or {},
            "case_id": span.get("case_id"),
            "k_index": span.get("k_index"),
            "attempt": span.get("attempt"),
            "model_call_index": span.get("model_call_index"),
        }
        for span in _load_jsonl(run_dir / "spans.jsonl")
        if span.get("span_type") == "model_call_end"
    ]


def _failed_model_calls_from_spans(run_dir: Path) -> list[dict[str, Any]]:
    return [
        span
        for span in _load_jsonl(run_dir / "spans.jsonl")
        if span.get("span_type") == "model_call_error"
    ]


def _configuration_snapshot(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "config_snapshot/configuration_snapshot.json")


def _run_elapsed_seconds(run_dir: Path) -> tuple[float | None, str]:
    status = _load_json(run_dir / "run_status.json")
    accumulated = status.get("accumulated_run_elapsed_before_segment_s")
    segment_started = status.get("current_segment_started_at_unix_s")
    segment_finished = status.get("finished_at_unix_s") or status.get("updated_at_unix_s")
    if accumulated is not None and segment_started is not None and segment_finished is not None:
        return (
            max(0.0, float(accumulated))
            + max(0.0, float(segment_finished) - float(segment_started)),
            "run_status.accumulated_segments",
        )
    cumulative = status.get("cumulative_run_elapsed_s")
    if cumulative is not None:
        return max(0.0, float(cumulative)), "run_status.cumulative_run_elapsed_s"
    started = status.get("started_at_unix_s")
    finished = status.get("finished_at_unix_s") or status.get("updated_at_unix_s")
    if started is None or finished is None:
        return None, "unavailable"
    return max(0.0, float(finished) - float(started)), "run_status.segment_elapsed_fallback"


def _gpu_count(deployment: dict[str, Any]) -> int:
    devices = [
        item.strip()
        for item in str(deployment.get("cuda_visible_devices") or "").split(",")
        if item.strip()
    ]
    if str(deployment.get("kind") or "") == "hf":
        # HFBackend moves one model onto one configured torch device. Merely making
        # extra GPUs visible does not allocate the model across them.
        return 1
    if devices:
        required = int(deployment.get("tensor_parallel_size") or 1) * int(
            deployment.get("data_parallel_size") or 1
        )
        return min(len(devices), max(1, required))
    return max(
        1,
        int(deployment.get("tensor_parallel_size") or 1)
        * int(deployment.get("data_parallel_size") or 1),
    )


def _token_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(int(call.get("input_text_tokens") or 0) for call in calls)
    output_tokens = sum(int(call.get("output_text_tokens") or 0) for call in calls)
    cached_values = [
        int(call["input_cached_tokens"])
        for call in calls
        if call.get("input_cached_tokens") is not None
    ]
    reasoning_values = [
        int(call["output_reasoning_tokens"])
        for call in calls
        if call.get("output_reasoning_tokens") is not None
    ]
    cached = sum(cached_values)
    reasoning = sum(reasoning_values)
    return {
        "model_call_count": len(calls),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached if len(cached_values) == len(calls) else None,
        "uncached_input_tokens": (
            max(0, input_tokens - cached) if len(cached_values) == len(calls) else None
        ),
        "output_tokens": output_tokens,
        "reasoning_output_tokens": (
            reasoning if len(reasoning_values) == len(calls) else None
        ),
        "answer_output_tokens": (
            max(0, output_tokens - reasoning)
            if len(reasoning_values) == len(calls)
            else None
        ),
        "cached_input_breakdown_available": len(cached_values) == len(calls),
        "reasoning_output_breakdown_available": len(reasoning_values) == len(calls),
    }


def _rates_for_input(
    pricing: dict[str, Any], input_tokens: int
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    tiers = list(pricing.get("rate_tiers") or [])
    if not tiers:
        return dict(pricing.get("rates") or {}), {"rate_tier": None}
    for tier in tiers:
        limit = int(tier.get("up_to_input_tokens") or 0)
        if input_tokens <= limit:
            return dict(tier.get("rates") or {}), {
                "rate_tier": {"up_to_input_tokens": limit}
            }
    return None, {
        "status": "input_exceeds_rate_tiers",
        "input_tokens": input_tokens,
        "maximum_priced_input_tokens": int(
            tiers[-1].get("up_to_input_tokens") or 0
        ),
    }


def _call_uses_thinking(call: dict[str, Any], usage: dict[str, Any]) -> bool:
    mode = str((call.get("invocation_policy") or {}).get("thinking_mode") or "")
    if mode == "enabled":
        return True
    if mode == "disabled":
        return False
    return bool(usage.get("reasoning_output_tokens"))


def _token_cost_with_rates(
    usage: dict[str, Any], rates: dict[str, Any], *, thinking: bool
) -> tuple[float | None, dict[str, Any]]:
    missing = [name for name in ("input", "output") if rates.get(name) is None]
    if missing:
        return None, {"status": "needs_rate", "missing_rates": missing}
    input_rate = float(rates["input"])
    output_rate = float(rates["output"])
    cached_value = rates.get("cached_input")
    reasoning_value = rates.get("reasoning_output")
    thinking_value = rates.get("thinking_output")
    cached_rate = input_rate if cached_value is None else float(cached_value)
    reasoning_rate = output_rate if reasoning_value is None else float(reasoning_value)
    if usage["cached_input_breakdown_available"]:
        input_cost = (
            float(usage["uncached_input_tokens"]) * input_rate
            + float(usage["cached_input_tokens"]) * cached_rate
        ) / 1_000_000
        input_assumption = (
            "provider_cached_token_breakdown_with_cached_rate"
            if cached_value is not None
            else "provider_cached_token_breakdown_without_cached_rate_"
            "all_input_charged_at_standard_rate"
        )
    else:
        input_cost = float(usage["input_tokens"]) * input_rate / 1_000_000
        input_assumption = "all_input_charged_at_standard_rate"
    if thinking and thinking_value is not None:
        output_cost = (
            float(usage["output_tokens"]) * float(thinking_value) / 1_000_000
        )
        output_assumption = "all_output_charged_at_thinking_mode_rate"
    elif usage["reasoning_output_breakdown_available"]:
        output_cost = (
            float(usage["answer_output_tokens"]) * output_rate
            + float(usage["reasoning_output_tokens"]) * reasoning_rate
        ) / 1_000_000
        output_assumption = "provider_reasoning_token_breakdown"
    else:
        output_cost = float(usage["output_tokens"]) * output_rate / 1_000_000
        output_assumption = "all_output_charged_at_standard_rate"
    return input_cost + output_cost, {
        "status": "calculated",
        "input_cost": input_cost,
        "output_cost": output_cost,
        "input_assumption": input_assumption,
        "output_assumption": output_assumption,
    }


def _token_cost(
    usage: dict[str, Any],
    pricing: dict[str, Any],
    calls: list[dict[str, Any]] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    if not calls:
        rates, tier_detail = _rates_for_input(pricing, int(usage["input_tokens"]))
        if rates is None:
            return None, tier_detail
        cost, detail = _token_cost_with_rates(
            usage,
            rates,
            thinking=_call_uses_thinking({}, usage),
        )
        return cost, {**detail, **tier_detail}

    total_input_cost = 0.0
    total_output_cost = 0.0
    tier_breakdown: dict[int, dict[str, Any]] = {}
    input_assumptions: set[str] = set()
    output_assumptions: set[str] = set()
    for call in calls:
        call_usage = _token_usage([call])
        rates, tier_detail = _rates_for_input(
            pricing, int(call_usage["input_tokens"])
        )
        if rates is None:
            return None, tier_detail
        _, detail = _token_cost_with_rates(
            call_usage,
            rates,
            thinking=_call_uses_thinking(call, call_usage),
        )
        if detail.get("status") != "calculated":
            return None, detail
        total_input_cost += float(detail["input_cost"])
        total_output_cost += float(detail["output_cost"])
        input_assumptions.add(str(detail["input_assumption"]))
        output_assumptions.add(str(detail["output_assumption"]))
        tier = tier_detail.get("rate_tier")
        if tier:
            limit = int(tier["up_to_input_tokens"])
            bucket = tier_breakdown.setdefault(
                limit,
                {
                    "up_to_input_tokens": limit,
                    "model_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                },
            )
            bucket["model_call_count"] += 1
            bucket["input_tokens"] += int(call_usage["input_tokens"])
            bucket["output_tokens"] += int(call_usage["output_tokens"])
            bucket["cost"] += float(detail["input_cost"]) + float(
                detail["output_cost"]
            )
    return total_input_cost + total_output_cost, {
        "status": "calculated",
        "input_cost": total_input_cost,
        "output_cost": total_output_cost,
        "input_assumption": ",".join(sorted(input_assumptions)),
        "output_assumption": ",".join(sorted(output_assumptions)),
        "rate_tier_breakdown": [
            {**item, "cost": round(float(item["cost"]), 8)}
            for _, item in sorted(tier_breakdown.items())
        ],
    }


def _cost_report(
    *,
    deployments: list[dict[str, Any]],
    pricing_instances: dict[str, dict[str, Any]],
    pricing_specs: dict[str, dict[str, Any]],
    calls_by_deployment: dict[str, list[dict[str, Any]]],
    all_calls: list[dict[str, Any]],
    elapsed_s: float | None,
    elapsed_source: str,
    failed_calls_by_deployment: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    line_items: list[dict[str, Any]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    for deployment in deployments:
        pricing_id = str(deployment.get("actual_pricing_instance_id") or "")
        pricing = pricing_instances.get(pricing_id)
        if not pricing:
            line_items.append(
                {
                    "deployment_instance_id": deployment.get("id"),
                    "pricing_instance_id": pricing_id or None,
                    "status": "missing_pricing_snapshot",
                    "cost": None,
                }
            )
            continue
        spec = pricing_specs.get(str(pricing.get("pricing_spec_id") or ""), {})
        basis = str(spec.get("basis") or "")
        deployment_id = str(deployment.get("id") or "")
        calls = calls_by_deployment.get(deployment_id, [])
        failed_call_count = len(failed_calls_by_deployment.get(deployment_id, []))
        item: dict[str, Any] = {
            "deployment_instance_id": deployment_id,
            "pricing_instance_id": pricing_id,
            "pricing_spec_id": pricing.get("pricing_spec_id"),
            "basis": basis,
            "currency": pricing.get("currency"),
            "rates": pricing.get("rates") or {},
            "failed_model_calls_without_usage": failed_call_count,
        }
        if basis == "token_usage":
            usage = _token_usage(calls)
            cost, detail = _token_cost(usage, pricing, calls)
            item.update(usage=usage, cost=cost, **detail)
            if cost is not None and failed_call_count:
                item.update(
                    status="calculated_lower_bound",
                    cost_is_lower_bound=True,
                    lower_bound_reason="failed provider calls have no completed usage payload",
                )
        elif basis == "allocated_gpu_time":
            rate = (pricing.get("rates") or {}).get("gpu_hour")
            pricing_metadata = dict(pricing.get("metadata") or {})
            billing_mode = spec.get("billing_mode")
            gpu_count = _gpu_count(deployment)
            gpu_hours = (elapsed_s * gpu_count / 3600.0) if elapsed_s is not None else None
            cost = float(rate) * gpu_hours if rate is not None and gpu_hours is not None else None
            item.update(
                status=(
                    "calculated"
                    if cost is not None
                    else "needs_rate" if rate is None else "missing_run_elapsed_time"
                ),
                run_elapsed_s=elapsed_s,
                run_elapsed_source=elapsed_source,
                allocated_gpu_count=gpu_count,
                allocated_gpu_hours=gpu_hours,
                gpu_hour_rate=rate,
                billing_mode=billing_mode,
                accelerator=pricing_metadata.get("accelerator"),
                quoted_node_month=pricing_metadata.get("quoted_node_month"),
                node_gpu_count=pricing_metadata.get("node_gpu_count"),
                amortization_hours_per_month=spec.get(
                    "amortization_hours_per_month"
                ),
                amortization_policy=spec.get("amortization_policy"),
                cost=cost,
                attribution="experiment_wall_time_x_allocated_gpus",
            )
        else:
            item.update(status="unsupported_pricing_basis", cost=None)
        if item.get("cost") is not None:
            totals[str(item["currency"])] += float(item["cost"])
        line_items.append(item)
    return {
        "status": (
            "calculated_lower_bound"
            if line_items
            and all(item.get("cost") is not None for item in line_items)
            and any(item.get("cost_is_lower_bound") for item in line_items)
            else "calculated"
            if line_items and all(item.get("cost") is not None for item in line_items)
            else "partial_or_unconfigured"
        ),
        "totals_by_currency": {
            currency: round(value, 8) for currency, value in sorted(totals.items())
        },
        "line_items": line_items,
        "unassigned_model_call_count": len(calls_by_deployment.get("", [])),
        "unassigned_failed_model_call_count": len(failed_calls_by_deployment.get("", [])),
        "total_model_call_count": len(all_calls),
    }


def attach_cost_metrics(
    metrics: dict[str, Any], samples: list[dict[str, Any]], run_dir: str | Path
) -> dict[str, Any]:
    """Attach actual and optional API-equivalent costs to aggregate metrics."""

    path = Path(run_dir)
    snapshot = _configuration_snapshot(path)
    deployments = list(snapshot.get("deployment_instances") or [])
    pricing_instances = {
        str(item.get("id") or ""): item for item in snapshot.get("pricing_instances") or []
    }
    pricing_specs = {
        str(item.get("id") or ""): item for item in snapshot.get("pricing_specs") or []
    }
    prediction_calls = [
        dict(call)
        for sample in samples
        for call in sample.get("model_calls") or []
        if isinstance(call, dict)
    ]
    span_calls = _model_calls_from_spans(path)
    failed_span_calls = _failed_model_calls_from_spans(path)
    all_calls = span_calls or prediction_calls
    calls_by_deployment: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in all_calls:
        calls_by_deployment[str(call.get("deployment_instance_id") or "")].append(call)
    failed_calls_by_deployment: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in failed_span_calls:
        failed_calls_by_deployment[
            str(call.get("deployment_instance_id") or "")
        ].append(call)
    if len(deployments) == 1 and calls_by_deployment.get(""):
        deployment_id = str(deployments[0].get("id") or "")
        calls_by_deployment[deployment_id].extend(calls_by_deployment.pop(""))
    if len(deployments) == 1 and failed_calls_by_deployment.get(""):
        deployment_id = str(deployments[0].get("id") or "")
        failed_calls_by_deployment[deployment_id].extend(
            failed_calls_by_deployment.pop("")
        )
    elapsed_s, elapsed_source = _run_elapsed_seconds(path)
    actual = _cost_report(
        deployments=deployments,
        pricing_instances=pricing_instances,
        pricing_specs=pricing_specs,
        calls_by_deployment=calls_by_deployment,
        all_calls=all_calls,
        elapsed_s=elapsed_s,
        elapsed_source=elapsed_source,
        failed_calls_by_deployment=failed_calls_by_deployment,
    )
    equivalent_items: list[dict[str, Any]] = []
    equivalent_totals: defaultdict[str, float] = defaultdict(float)
    for deployment in deployments:
        deployment_id = str(deployment.get("id") or "")
        equivalent_id = str(
            deployment.get("api_equivalent_pricing_instance_id") or ""
        )
        if not equivalent_id:
            continue
        pricing = pricing_instances.get(equivalent_id)
        if pricing:
            usage = _token_usage(calls_by_deployment.get(deployment_id, []))
            failed_call_count = len(failed_calls_by_deployment.get(deployment_id, []))
            cost, detail = _token_cost(
                usage,
                pricing,
                calls_by_deployment.get(deployment_id, []),
            )
            item = {
                "deployment_instance_id": deployment_id,
                "pricing_instance_id": equivalent_id,
                "pricing_spec_id": pricing.get("pricing_spec_id"),
                "currency": pricing.get("currency"),
                "rates": pricing.get("rates") or {},
                "usage": usage,
                "cost": cost,
                "failed_model_calls_without_usage": failed_call_count,
                **detail,
            }
            if cost is not None and failed_call_count:
                item.update(
                    status="calculated_lower_bound",
                    cost_is_lower_bound=True,
                    lower_bound_reason="failed model calls have no completed usage payload",
                )
            if cost is not None:
                equivalent_totals[str(pricing.get("currency"))] += float(cost)
        else:
            item = {
                "deployment_instance_id": deployment_id,
                "pricing_instance_id": equivalent_id,
                "status": "missing_pricing_snapshot",
                "cost": None,
            }
        equivalent_items.append(item)
    equivalent = {
        "status": (
            "calculated_lower_bound"
            if equivalent_items
            and all(item.get("cost") is not None for item in equivalent_items)
            and any(item.get("cost_is_lower_bound") for item in equivalent_items)
            else "calculated"
            if equivalent_items
            and all(item.get("cost") is not None for item in equivalent_items)
            else "partial_or_unconfigured"
        ),
        "totals_by_currency": {
            currency: round(value, 8)
            for currency, value in sorted(equivalent_totals.items())
        },
        "line_items": equivalent_items,
    }
    comparison: dict[str, Any] = {
        "status": "unavailable",
        "reason": "actual and API-equivalent totals require the same configured currency",
    }
    shared_currencies = set(actual["totals_by_currency"]) & set(
        equivalent["totals_by_currency"]
    )
    if "lower_bound" in str(actual["status"]) or "lower_bound" in str(
        equivalent["status"]
    ):
        comparison = {
            "status": "unavailable_due_to_incomplete_provider_usage",
            "reason": (
                "failed model calls have no completed usage payload; cost differences "
                "between lower bounds are not reliable"
            ),
        }
    elif len(shared_currencies) == 1:
        currency = next(iter(shared_currencies))
        actual_total = float(actual["totals_by_currency"][currency])
        equivalent_total = float(equivalent["totals_by_currency"][currency])
        difference = equivalent_total - actual_total
        comparison = {
            "status": "calculated",
            "currency": currency,
            "actual_cost": actual_total,
            "api_equivalent_cost": equivalent_total,
            "api_equivalent_minus_actual": round(difference, 8),
            "actual_savings_vs_api_equivalent": round(difference, 8),
            "actual_to_api_equivalent_ratio": (
                round(actual_total / equivalent_total, 8)
                if equivalent_total > 0
                else None
            ),
        }
    metrics["costing"] = {
        "schema_version": 1,
        "scope": "model_inference_only",
        "actual_cost": actual,
        "api_equivalent_cost": equivalent,
        "comparison": comparison,
        "source": "immutable_configuration_snapshot_and_model_calls",
        "excluded_cost_categories": [
            "paid_tools_and_search",
            "cpu_ram_storage_network",
            "provider_requests_without_completed_usage",
            "provider_specific_non_token_fees",
        ],
        "billing_reconciliation_required": bool(failed_span_calls) or any(
            str(item.get("kind") or "") in {"api", "vllm"}
            and item.get("managed", True) is False
            for item in deployments
        ),
        "failed_model_calls_without_usage": len(failed_span_calls),
        "model_call_source": (
            "spans.model_call_end" if span_calls else "predictions.model_calls_fallback"
        ),
    }
    return metrics
