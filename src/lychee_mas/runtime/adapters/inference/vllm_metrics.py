"""Periodic vLLM service metrics collection for benchmark runs.

The OpenAI-compatible response describes one request.  The Prometheus endpoint
describes the shared vLLM service.  Keeping those two sources separate avoids
misattributing service-wide queue pressure to an individual benchmark case.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

_SELECTED_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
}

_SELECTED_HISTOGRAMS = {
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
}

_GAUGE_ALIASES = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "gpu_cache_usage_fraction": (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
    ),
}

_COUNTER_ALIASES = {
    "preemptions_total": "vllm:num_preemptions_total",
    "request_success_total": "vllm:request_success_total",
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "generation_tokens_total": "vllm:generation_tokens_total",
    "prefix_cache_hits_total": "vllm:prefix_cache_hits_total",
    "prefix_cache_queries_total": "vllm:prefix_cache_queries_total",
}

_HISTOGRAM_ALIASES = {
    "queue_time": "vllm:request_queue_time_seconds",
    "time_to_first_token": "vllm:time_to_first_token_seconds",
    "e2e_latency": "vllm:e2e_request_latency_seconds",
    "inter_token_latency": "vllm:inter_token_latency_seconds",
}


def metrics_url(base_url: str, explicit_url: str | None = None) -> str:
    """Return the Prometheus URL for a vLLM OpenAI-compatible endpoint."""

    if explicit_url:
        return str(explicit_url).rstrip("/")
    parsed = urlsplit(str(base_url).rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/metrics", "", ""))


def _sample_base_name(name: str) -> str:
    for suffix in ("_bucket", "_count", "_sum", "_created"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_prometheus_metrics(payload: str) -> dict[str, list[dict[str, Any]]]:
    """Parse and retain the vLLM metrics used by the benchmark observer."""

    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError as exc:  # pragma: no cover - covered by benchmark extra
        raise RuntimeError(
            "prometheus-client is required for vLLM service metrics collection"
        ) from exc

    selected: dict[str, list[dict[str, Any]]] = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            base_name = _sample_base_name(sample.name)
            if sample.name not in _SELECTED_METRICS and base_name not in _SELECTED_HISTOGRAMS:
                continue
            selected.setdefault(sample.name, []).append(
                {
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                }
            )
    return selected


def _sum_samples(metrics: dict[str, list[dict[str, Any]]], name: str) -> float | None:
    samples = metrics.get(name)
    if not samples:
        return None
    return sum(float(sample["value"]) for sample in samples)


def _histogram_bucket_totals(
    metrics: dict[str, list[dict[str, Any]]], name: str
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for sample in metrics.get(f"{name}_bucket") or []:
        boundary = str((sample.get("labels") or {}).get("le") or "")
        if boundary:
            totals[boundary] = totals.get(boundary, 0.0) + float(sample["value"])
    return totals


def _flatten_snapshot(metrics: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for alias, names in _GAUGE_ALIASES.items():
        flattened[alias] = next(
            (value for name in names if (value := _sum_samples(metrics, name)) is not None),
            None,
        )
    for alias, name in _COUNTER_ALIASES.items():
        flattened[alias] = _sum_samples(metrics, name)
    for alias, name in _HISTOGRAM_ALIASES.items():
        flattened[f"{alias}_count_total"] = _sum_samples(metrics, f"{name}_count")
        flattened[f"{alias}_seconds_total"] = _sum_samples(metrics, f"{name}_sum")
        flattened[f"{alias}_bucket_totals"] = _histogram_bucket_totals(metrics, name)
    return flattened


def _window_quantile(
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    alias: str,
    quantile: float,
) -> float | None:
    current_buckets = current.get(f"{alias}_bucket_totals") or {}
    previous_buckets = previous.get(f"{alias}_bucket_totals") or {}
    if not current_buckets or not previous_buckets:
        return None
    deltas: list[tuple[float, float]] = []
    for boundary, value in current_buckets.items():
        if boundary not in previous_buckets:
            continue
        try:
            numeric_boundary = float(boundary)
        except ValueError:
            if boundary != "+Inf":
                continue
            numeric_boundary = float("inf")
        count = float(value) - float(previous_buckets[boundary])
        if count >= 0:
            deltas.append((numeric_boundary, count))
    if not deltas:
        return None
    deltas.sort(key=lambda item: item[0])
    total = deltas[-1][1]
    if total <= 0:
        return None
    threshold = total * quantile
    previous_finite_boundary: float | None = None
    for boundary, cumulative_count in deltas:
        if cumulative_count >= threshold:
            # Match Prometheus histogram_quantile semantics for the terminal
            # +Inf bucket: use the highest finite upper bound as the estimate.
            return previous_finite_boundary if boundary == float("inf") else boundary
        if boundary != float("inf"):
            previous_finite_boundary = boundary
    return None


def _window_snapshot(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    """Derive one scrape-window view from cumulative Prometheus counters."""

    if not previous:
        return {}

    def delta(field: str) -> float | None:
        left = current.get(field)
        right = previous.get(field)
        if left is None or right is None:
            return None
        value = float(left) - float(right)
        return value if value >= 0 else None

    result: dict[str, Any] = {}
    for name in _COUNTER_ALIASES:
        result[f"{name}_window"] = delta(name)
    for alias in _HISTOGRAM_ALIASES:
        count = delta(f"{alias}_count_total")
        seconds = delta(f"{alias}_seconds_total")
        result[f"{alias}_window_count"] = count
        result[f"{alias}_window_mean_s"] = (
            seconds / count if count and seconds is not None else None
        )
        result[f"{alias}_window_p95_s"] = _window_quantile(
            current,
            previous,
            alias=alias,
            quantile=0.95,
        )
    hits = result.get("prefix_cache_hits_total_window")
    queries = result.get("prefix_cache_queries_total_window")
    result["prefix_cache_hit_ratio_window"] = (
        hits / queries if hits is not None and queries and queries > 0 else None
    )
    return result


def targets_from_deployments(document: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Resolve unique vLLM service endpoints from a deployment document."""

    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for deployment in (document or {}).get("deployments", []):
        if str(deployment.get("kind")) != "vllm":
            continue
        base_url = str(deployment.get("base_url") or "").strip()
        if not base_url:
            continue
        endpoint = metrics_url(base_url, deployment.get("metrics_url"))
        if endpoint in seen:
            continue
        seen.add(endpoint)
        targets.append(
            {
                "deployment_id": str(deployment.get("id") or "vllm"),
                "model_id": str(deployment.get("model_id") or ""),
                "base_url": base_url,
                "metrics_url": endpoint,
                "trust_env": bool(deployment.get("trust_env", True)),
                "exclusive_run_attribution": bool(
                    deployment.get("metrics_exclusive_run_attribution", False)
                ),
                "api_key_env": deployment.get("api_key_env"),
                "auth_mode": str(deployment.get("auth_mode") or "none"),
            }
        )
    return targets


class VLLMMetricsMonitor:
    """Collect vLLM Prometheus snapshots in a daemon thread."""

    def __init__(
        self,
        *,
        targets: list[dict[str, Any]],
        output_path: str | Path,
        summary_path: str | Path,
        interval_s: float = 5.0,
        request_timeout_s: float = 2.0,
        append: bool = False,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("vLLM metrics interval must be positive")
        if request_timeout_s <= 0:
            raise ValueError("vLLM metrics request timeout must be positive")
        self.targets = [dict(target) for target in targets]
        self.output_path = Path(output_path)
        self.summary_path = Path(summary_path)
        self.interval_s = float(interval_s)
        self.request_timeout_s = float(request_timeout_s)
        self.append = bool(append)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._seq = 0
        if self.append and self.output_path.is_file():
            for line in self.output_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    self._records.append(record)
                    self._seq = max(self._seq, int(record.get("seq") or 0))

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.append:
            self.output_path.write_text("", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._run,
            name="lychee-vllm-metrics",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.request_timeout_s * len(self.targets) + 1.0))
        # Capture counters after the final request has completed; without this
        # close-out scrape, short runs can miss their last completion delta.
        for target in self.targets:
            self._collect(target)
        summary = self.summary()
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.summary_path)
        return summary

    def _run(self) -> None:
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            for target in self.targets:
                if self._stop.is_set():
                    break
                self._collect(target)
            remaining = self.interval_s - (time.monotonic() - cycle_started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _collect(self, target: dict[str, Any]) -> None:
        started = time.monotonic()
        record: dict[str, Any] = {
            "schema_version": 1,
            "seq": self._seq + 1,
            "timestamp_unix_s": round(time.time(), 6),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "deployment_service",
            "exclusive_run_attribution": bool(target.get("exclusive_run_attribution", False)),
            "deployment_id": target["deployment_id"],
            "model_id": target.get("model_id"),
            "metrics_url": target["metrics_url"],
        }
        try:
            headers = {"Accept": "text/plain; version=0.0.4"}
            key_name = str(target.get("api_key_env") or "")
            if target.get("auth_mode") == "env" and key_name and os.environ.get(key_name):
                headers["Authorization"] = f"Bearer {os.environ[key_name]}"
            request = Request(target["metrics_url"], headers=headers)
            opener = (
                build_opener() if target.get("trust_env", True) else build_opener(ProxyHandler({}))
            )
            with opener.open(request, timeout=self.request_timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
            metrics = parse_prometheus_metrics(body)
            flattened = _flatten_snapshot(metrics)
            previous = next(
                (
                    item
                    for item in reversed(self._records)
                    if item.get("status") == "ok"
                    and item.get("deployment_id") == target["deployment_id"]
                ),
                None,
            )
            record.update(
                status="ok",
                scrape_latency_s=round(time.monotonic() - started, 6),
                metrics=metrics,
                **flattened,
                **_window_snapshot(flattened, previous),
            )
        except Exception as exc:  # the benchmark run must survive observer failures
            record.update(
                status="error",
                scrape_latency_s=round(time.monotonic() - started, 6),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        self._seq += 1
        self._records.append(record)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        by_target: dict[str, list[dict[str, Any]]] = {}
        for record in self._records:
            by_target.setdefault(str(record.get("deployment_id") or "unknown"), []).append(record)
        return {
            "schema_version": 1,
            "scope": "deployment_service",
            "interval_s": self.interval_s,
            "sample_count": len(self._records),
            "targets": {
                target_id: self._summarize_target(records)
                for target_id, records in sorted(by_target.items())
            },
        }

    def latest_snapshots(self) -> list[dict[str, Any]]:
        """Return the latest immutable record for each monitored deployment."""

        latest: dict[str, dict[str, Any]] = {}
        for record in self._records:
            latest[str(record.get("deployment_id") or "unknown")] = dict(record)
        return list(latest.values())

    @staticmethod
    def _summarize_target(records: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [record for record in records if record.get("status") == "ok"]
        result: dict[str, Any] = {
            "sample_count": len(records),
            "successful_sample_count": len(successful),
            "error_sample_count": len(records) - len(successful),
            "exclusive_run_attribution": bool(records[-1].get("exclusive_run_attribution")),
        }
        for field in _GAUGE_ALIASES:
            values = [
                float(record[field]) for record in successful if record.get(field) is not None
            ]
            result[f"{field}_mean"] = round(sum(values) / len(values), 6) if values else None
            result[f"{field}_max"] = round(max(values), 6) if values else None

        if successful:
            first_metrics = successful[0].get("metrics") or {}
            last_metrics = successful[-1].get("metrics") or {}
            for name in sorted(_SELECTED_METRICS):
                first = _sum_samples(first_metrics, name)
                last = _sum_samples(last_metrics, name)
                if first is not None and last is not None:
                    result[f"{name}_delta"] = round(last - first, 6)
            for name in sorted(_SELECTED_HISTOGRAMS):
                first_count = _sum_samples(first_metrics, f"{name}_count")
                last_count = _sum_samples(last_metrics, f"{name}_count")
                first_sum = _sum_samples(first_metrics, f"{name}_sum")
                last_sum = _sum_samples(last_metrics, f"{name}_sum")
                if (
                    first_count is None
                    or last_count is None
                    or first_sum is None
                    or last_sum is None
                ):
                    continue
                count_delta = float(last_count) - float(first_count)
                sum_delta = float(last_sum) - float(first_sum)
                result[f"{name}_count_delta"] = int(count_delta)
                result[f"{name}_mean_seconds"] = (
                    round(sum_delta / count_delta, 6) if count_delta > 0 else None
                )
        return result
