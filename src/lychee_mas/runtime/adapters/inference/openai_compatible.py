"""OpenAI-compatible chat backend for API-only experiments.

This backend implements the small synchronous interface consumed by
``autogen_injection_client``: ``generate_chat`` returns text plus usage/latency
metadata. It is intended for none/NL-channel runs when local GPUs are busy.
Latent prefix methods require local hidden states and are therefore not
available for API providers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lychee_mas.runtime.model.token_budget import tokenized_sequence_length

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested request options without mutating deployment defaults."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class APIGenResult:
    text: str
    n_prompt_pos: int
    n_gen_tokens: int
    latency_s: float
    prefix_len: int = 0
    request_queue_latency_s: float = 0.0
    client_rate_limiter_wait_s: Optional[float] = None
    client_http_request_latency_s: Optional[float] = None
    client_response_postprocess_latency_s: Optional[float] = None
    client_model_call_wall_time_s: Optional[float] = None
    provider_request_queue_latency_s: Optional[float] = None
    provider_scheduled_to_first_token_s: Optional[float] = None
    provider_generation_latency_s: Optional[float] = None
    provider_mean_inter_token_latency_s: Optional[float] = None
    provider_output_tokens_per_second: Optional[float] = None
    provider_request_metrics: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning_content: str = ""
    reasoning_tokens: Optional[int] = None
    answer_tokens: Optional[int] = None
    reasoning_tokens_source: Optional[str] = None
    answer_tokens_source: Optional[str] = None
    cached_input_tokens: Optional[int] = None
    cached_input_tokens_source: Optional[str] = None
    cached_input_tokens_status: str = "not_reported"
    provider_message_content: Any = None
    provider_reasoning_content: Any = None
    raw_decoded_text: Optional[str] = None
    provider_request_payload: dict[str, Any] = field(default_factory=dict)
    provider_response_payload: dict[str, Any] = field(default_factory=dict)
    response_payload_origin: str = "provider"
    request_timeout_s: Optional[float] = None
    request_max_retries: int = 0
    observed_generation_tokens_per_second: Optional[float] = None
    timeout_policy: dict[str, Any] = field(default_factory=dict)


def _provider_request_metrics(response: Any) -> dict[str, Any]:
    """Normalize optional vLLM per-request response metrics.

    Newer vLLM releases expose a top-level ``metrics`` extension on the
    OpenAI-compatible response.  The OpenAI SDK preserves unknown fields in
    ``model_extra``; direct attributes are also accepted for compatibility.
    """

    raw = getattr(response, "metrics", None)
    if raw is None:
        extra = getattr(response, "model_extra", None)
        if isinstance(extra, dict):
            raw = extra.get("metrics")
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
        raw = vars(raw)
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key in (
        "queue_time_ms",
        "time_to_first_token_ms",
        "generation_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    ):
        value = raw.get(key)
        if value is None:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            normalized[key] = value
    return normalized


def _cached_input_usage(
    usage_payload: Any,
) -> tuple[Optional[int], Optional[str], str]:
    """Read provider-reported per-request prefix-cache usage without estimating it."""

    if not isinstance(usage_payload, dict):
        return None, None, "not_reported"
    detail_fields = (
        ("prompt_tokens_details", "usage.prompt_tokens_details.cached_tokens"),
        ("input_tokens_details", "usage.input_tokens_details.cached_tokens"),
    )
    for field_name, source in detail_fields:
        details = usage_payload.get(field_name)
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            return int(details["cached_tokens"]), source, "available"
    if usage_payload.get("prompt_cache_hit_tokens") is not None:
        return (
            int(usage_payload["prompt_cache_hit_tokens"]),
            "usage.prompt_cache_hit_tokens",
            "available",
        )
    return None, None, "not_reported"


def _normalize_timeout_policy(
    value: Optional[Dict[str, Any]], *, minimum_timeout_s: float
) -> dict[str, Any]:
    """Normalize the transport timeout policy used by non-streaming chat calls."""

    raw = dict(value or {})
    mode = str(raw.get("mode") or "adaptive")
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("request timeout mode must be fixed or adaptive")
    minimum_s = float(raw.get("minimum_s", minimum_timeout_s))
    maximum_s = float(raw.get("maximum_s", max(minimum_s, 1800.0)))
    base_s = float(raw.get("base_s", 30.0))
    initial_tps = float(raw.get("initial_generation_tokens_per_second", 20.0))
    observed_tps_ceiling = float(raw.get("observed_tokens_per_second_ceiling", 40.0))
    safety_factor = float(raw.get("safety_factor", 1.5))
    ewma_alpha = float(raw.get("ewma_alpha", 0.25))
    if minimum_s <= 0 or maximum_s < minimum_s:
        raise ValueError("request timeout requires 0 < minimum_s <= maximum_s")
    if base_s < 0 or initial_tps <= 0 or observed_tps_ceiling <= 0:
        raise ValueError("request timeout speed and base values must be positive")
    if safety_factor < 1:
        raise ValueError("request timeout safety_factor must be >= 1")
    if not 0 < ewma_alpha <= 1:
        raise ValueError("request timeout ewma_alpha must be in (0, 1]")
    return {
        "mode": mode,
        "minimum_s": minimum_s,
        "maximum_s": maximum_s,
        "base_s": base_s,
        "initial_generation_tokens_per_second": initial_tps,
        "observed_tokens_per_second_ceiling": observed_tps_ceiling,
        "safety_factor": safety_factor,
        "ewma_alpha": ewma_alpha,
    }


def _to_openai_content(content: Any) -> Any:
    """Map backend-neutral text/image parts to Chat Completions content parts."""

    if not isinstance(content, list):
        return content
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue
        kind = part.get("type")
        if kind in {"text", "image_url"}:
            converted.append(dict(part))
            continue
        if kind == "image":
            image = part.get("image") or part.get("url")
            to_openai = getattr(image, "to_openai_format", None)
            if callable(to_openai):
                converted.append(dict(to_openai()))
            elif isinstance(image, str):
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image, "detail": part.get("detail", "auto")},
                    }
                )
            else:
                raise TypeError("image content part requires an Image object or URL/data URI")
            continue
        raise ValueError(f"unsupported multimodal content part type: {kind!r}")
    return converted


class RequestRateLimiter:
    """Bound request concurrency and start rate for one model endpoint.

    ``host`` scope uses an advisory file lock for conservative coordination
    between separate LycheeMAS processes on the same server. The in-process
    semaphore remains necessary because POSIX locks are process-scoped.
    """

    def __init__(
        self,
        *,
        key: str,
        max_concurrency: int = 0,
        min_interval_s: float = 0.0,
        scope: str = "process",
    ):
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be non-negative")
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be non-negative")
        if scope not in {"process", "host"}:
            raise ValueError("rate limit scope must be process or host")
        self.max_concurrency = int(max_concurrency)
        self.min_interval_s = float(min_interval_s)
        self.scope = scope
        self._semaphore = (
            threading.BoundedSemaphore(self.max_concurrency) if self.max_concurrency else None
        )
        self._schedule_lock = threading.Lock()
        self._last_start = 0.0
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        self._host_lock_path = (
            Path(tempfile.gettempdir()) / "lychee_mas_rate_limits" / f"{digest}.lock"
        )

    def _wait_process_interval(self) -> None:
        if not self.min_interval_s:
            return
        with self._schedule_lock:
            wait_s = self.min_interval_s - (time.monotonic() - self._last_start)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_start = time.monotonic()

    @contextmanager
    def slot(self):
        if self._semaphore is not None:
            self._semaphore.acquire()
        host_handle = None
        try:
            if self.scope == "host":
                import fcntl

                self._host_lock_path.parent.mkdir(parents=True, exist_ok=True)
                host_handle = self._host_lock_path.open("a+", encoding="utf-8")
                fcntl.flock(host_handle.fileno(), fcntl.LOCK_EX)
                host_handle.seek(0)
                try:
                    previous_start = float(host_handle.read().strip() or 0.0)
                except ValueError:
                    previous_start = 0.0
                wait_s = self.min_interval_s - (time.time() - previous_start)
                if wait_s > 0:
                    time.sleep(wait_s)
                host_handle.seek(0)
                host_handle.truncate()
                host_handle.write(str(time.time()))
                host_handle.flush()
                os.fsync(host_handle.fileno())
            else:
                self._wait_process_interval()
            yield
        finally:
            if host_handle is not None:
                import fcntl

                fcntl.flock(host_handle.fileno(), fcntl.LOCK_UN)
                host_handle.close()
            if self._semaphore is not None:
                self._semaphore.release()


class OpenAICompatibleBackend:
    supports_concurrent_requests = True
    supports_request_seed = True
    supports_native_tools = True
    supports_request_overrides = True
    supports_extra_create_args = True
    supports_json_output = True
    supports_structured_output = True

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key: Optional[str] = None,
        auth_mode: str = "env",
        trust_env: bool = True,
        proxy_url: Optional[str] = None,
        timeout: float = 120.0,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        model_info: Optional[Dict[str, Any]] = None,
        request_limits: Optional[Dict[str, Any]] = None,
        request_timeout: Optional[Dict[str, Any]] = None,
        max_retries: int = 0,
        extra_body: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None,
        reasoning_token_accounting: Optional[str] = None,
    ):
        import httpx
        from openai import OpenAI

        if auth_mode not in {"env", "none"}:
            raise ValueError("auth_mode must be env or none")
        resolved_key = api_key or (os.environ.get(api_key_env) if auth_mode == "env" else "EMPTY")
        if auth_mode == "env" and not resolved_key:
            raise ValueError(f"API key not found; set {api_key_env} or pass api_key explicitly.")
        self.model_name = model
        self.base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        self.api_key_env = api_key_env
        self.auth_mode = auth_mode
        self.trust_env = trust_env
        self.proxy_url = str(proxy_url or "").strip() or None
        self.timeout = timeout
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.model_info = model_info or {}
        self.context_window = self.model_info.get("context_window") or self.model_info.get(
            "max_model_length"
        )
        self.request_limits = dict(request_limits or {})
        self.request_timeout = _normalize_timeout_policy(
            request_timeout, minimum_timeout_s=float(timeout)
        )
        self.max_retries = int(max_retries)
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._throughput_lock = threading.Lock()
        self._observed_generation_tps: Optional[float] = None
        self.extra_body = dict(extra_body or {})
        self.tok = None
        self.tokenizer_path = str(tokenizer_path or "").strip() or None
        self.reasoning_token_accounting = str(reasoning_token_accounting or "inconclusive")
        self._tokenizer_lock = threading.Lock()
        self._tokenizer_load_attempted = False
        self._tokenizer_load_error: Optional[str] = None
        self._tokenizer_load_latency_s: Optional[float] = None
        self._tokenizer_warmup_report: dict[str, Any] = {
            "reasoning_token_accounting": self.reasoning_token_accounting,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_warmup_status": "not_required",
            "tokenizer_warmup_latency_s": 0.0,
        }
        self.http_client = httpx.Client(
            trust_env=trust_env,
            proxy=self.proxy_url,
            timeout=float(self.request_timeout["maximum_s"]),
        )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
        )
        self.rate_limiter = RequestRateLimiter(
            key=f"{self.base_url}|{self.model_name}",
            max_concurrency=int(self.request_limits.get("max_concurrency", 0) or 0),
            min_interval_s=float(self.request_limits.get("min_interval_s", 0.0) or 0.0),
            scope=str(self.request_limits.get("scope") or "process"),
        )
        if self.reasoning_token_accounting == "client_retokenized_required":
            self.warmup_tokenizer()

    def _ensure_tokenizer(self):
        """Load the served model tokenizer once in this experiment process."""

        tokenizer_path = getattr(self, "tokenizer_path", None)
        if not tokenizer_path:
            return None
        lock = getattr(self, "_tokenizer_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._tokenizer_lock = lock
        with lock:
            if getattr(self, "tok", None) is None and not getattr(
                self, "_tokenizer_load_attempted", False
            ):
                self._tokenizer_load_attempted = True
                started = time.monotonic()
                try:
                    from transformers import AutoTokenizer

                    self.tok = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=True,
                    )
                    self._tokenizer_load_error = None
                except Exception as exc:
                    self.tok = None
                    self._tokenizer_load_error = f"{type(exc).__name__}: {exc}"
                finally:
                    self._tokenizer_load_latency_s = time.monotonic() - started
            return getattr(self, "tok", None)

    def warmup_tokenizer(self, *, force: bool = False) -> dict[str, Any]:
        """Prepare client-side output token accounting before the first case."""

        accounting = str(getattr(self, "reasoning_token_accounting", "inconclusive"))
        previous = dict(getattr(self, "_tokenizer_warmup_report", {}) or {})
        if previous.get("tokenizer_warmup_status") in {"ready", "failed", "unavailable"}:
            return previous
        if accounting != "client_retokenized_required" and not force:
            report = {
                "reasoning_token_accounting": accounting,
                "tokenizer_path": getattr(self, "tokenizer_path", None),
                "tokenizer_warmup_status": "not_required",
                "tokenizer_warmup_latency_s": 0.0,
            }
            self._tokenizer_warmup_report = report
            return dict(report)
        if not getattr(self, "tokenizer_path", None):
            report = {
                "reasoning_token_accounting": accounting,
                "tokenizer_path": None,
                "tokenizer_warmup_status": "unavailable",
                "tokenizer_warmup_latency_s": 0.0,
                "tokenizer_warmup_error": "tokenizer_path is not configured",
            }
            self._tokenizer_warmup_report = report
            return dict(report)

        started = time.monotonic()
        tokenizer = self._ensure_tokenizer()
        warmup_tokens = None
        error = getattr(self, "_tokenizer_load_error", None)
        if tokenizer is not None:
            try:
                warmup_tokens = len(tokenizer.encode("tokenizer warmup", add_special_tokens=False))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        tokenizer_load_latency = getattr(self, "_tokenizer_load_latency_s", None)
        report = {
            "reasoning_token_accounting": accounting,
            "tokenizer_path": getattr(self, "tokenizer_path", None),
            "tokenizer_warmup_status": "ready" if tokenizer is not None and not error else "failed",
            "tokenizer_warmup_latency_s": round(time.monotonic() - started, 6),
            "tokenizer_load_latency_s": (
                round(float(tokenizer_load_latency), 6)
                if tokenizer_load_latency is not None
                else None
            ),
            "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
            "tokenizer_warmup_tokens": warmup_tokens,
            "tokenizer_warmup_error": error,
        }
        self._tokenizer_warmup_report = report
        return dict(report)

    def startup_observability(self) -> dict[str, Any]:
        """Return startup-only telemetry without issuing another model request."""

        return dict(getattr(self, "_tokenizer_warmup_report", {}) or {})

    def _retokenize(self, text: str) -> Optional[int]:
        """Count separated provider text with the local served-model tokenizer."""

        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            return None
        return len(tokenizer.encode(str(text or ""), add_special_tokens=False))

    def count_chat_tokens(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[int, str]:
        """Count the provider request with the served model's chat template."""

        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            serialized = json.dumps(
                {"messages": messages, "tools": tools or []},
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            return max(1, (len(serialized) + 2) // 3), "estimated_json_chars_div_3"
        kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            token_ids = tokenizer.apply_chat_template(messages, **kwargs)
            return tokenized_sequence_length(token_ids), "tokenizer_chat_template"
        except Exception:
            serialized = json.dumps(
                {"messages": messages, "tools": tools or []},
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            return len(tokenizer.encode(serialized, add_special_tokens=False)), (
                "tokenizer_serialized_request_fallback"
            )

    def request_timeout_plan(self, max_new_tokens: int) -> dict[str, Any]:
        """Return the timeout selected for one non-streaming completion request."""

        minimum_timeout = float(getattr(self, "timeout", 120.0))
        policy = getattr(self, "request_timeout", None)
        if not isinstance(policy, dict):
            policy = _normalize_timeout_policy(None, minimum_timeout_s=minimum_timeout)
        observed = getattr(self, "_observed_generation_tps", None)
        initial_tps = float(policy["initial_generation_tokens_per_second"])
        # A timeout is a failure guard, not a throughput target. Fast recent calls must
        # never shorten the conservative cold-start deadline because shared services can
        # slow down abruptly under concurrent load. Slower observations may lengthen it.
        planning_candidates = [
            initial_tps,
            float(policy["observed_tokens_per_second_ceiling"]),
        ]
        if observed and observed > 0:
            planning_candidates.append(float(observed))
        planning_tps = min(planning_candidates)
        if policy["mode"] == "fixed":
            timeout_s = float(policy["minimum_s"])
        else:
            predicted_generation_s = max(1, int(max_new_tokens)) / planning_tps
            timeout_s = float(policy["base_s"]) + (
                predicted_generation_s * float(policy["safety_factor"])
            )
            timeout_s = max(float(policy["minimum_s"]), timeout_s)
            timeout_s = min(float(policy["maximum_s"]), timeout_s)
        return {
            "mode": str(policy["mode"]),
            "timeout_s": round(timeout_s, 3),
            "planning_generation_tokens_per_second": round(planning_tps, 3),
            "observed_generation_tokens_per_second": (
                round(float(observed), 3) if observed and observed > 0 else None
            ),
            "max_retries": int(getattr(self, "max_retries", 0)),
            "policy": dict(policy),
        }

    def _observe_generation_throughput(self, completion_tokens: int, latency_s: float) -> None:
        if completion_tokens <= 0 or latency_s <= 0:
            return
        observed = float(completion_tokens) / float(latency_s)
        lock = getattr(self, "_throughput_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._throughput_lock = lock
        with lock:
            previous = getattr(self, "_observed_generation_tps", None)
            policy = getattr(self, "request_timeout", None) or _normalize_timeout_policy(
                None, minimum_timeout_s=float(getattr(self, "timeout", 120.0))
            )
            alpha = float(policy["ewma_alpha"])
            self._observed_generation_tps = (
                observed if previous is None else (alpha * observed) + ((1 - alpha) * previous)
            )

    def generate_chat(
        self,
        messages: List[Dict],
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        json_output: Any = None,
        request_overrides: Optional[Dict[str, Any]] = None,
        extra_create_args: Optional[Dict[str, Any]] = None,
    ) -> APIGenResult:
        from lychee_mas.runtime.events.store import to_jsonable

        request_overrides = dict(request_overrides or {})
        extra_create_args = dict(extra_create_args or {})
        api_messages = []
        for message in messages:
            api_message = {
                key: message[key]
                for key in (
                    "role",
                    "content",
                    "reasoning_content",
                    "tool_calls",
                    "tool_call_id",
                    "name",
                )
                if key in message and message[key] is not None
            }
            if "content" in api_message:
                api_message["content"] = _to_openai_content(api_message["content"])
            api_messages.append(api_message)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": max_new_tokens,
        }
        if tools:
            payload["tools"] = tools
            if isinstance(tool_choice, str) and tool_choice in {"auto", "required", "none"}:
                payload["tool_choice"] = tool_choice
            elif isinstance(tool_choice, dict):
                payload["tool_choice"] = dict(tool_choice)
        do_sample = request_overrides.get("do_sample", getattr(self, "do_sample", None))
        temperature = request_overrides.get("temperature", self.temperature)
        top_p = request_overrides.get("top_p", self.top_p)
        if do_sample is False:
            # OpenAI-compatible servers do not expose Transformers' do_sample flag.
            # temperature=0 and top_p=1 are the provider-level greedy equivalent.
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0
        else:
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
        request_seed = self.seed if seed is None else seed
        if request_seed is not None:
            payload["seed"] = request_seed
        extra_body = _deep_merge(
            getattr(self, "extra_body", None) or {},
            dict(request_overrides.get("extra_body") or {}),
        )
        if extra_body:
            payload["extra_body"] = extra_body

        protected = {"messages", "tools", "stream"}
        conflicts = protected.intersection(extra_create_args)
        if conflicts:
            raise ValueError("extra_create_args cannot override " + ", ".join(sorted(conflicts)))
        payload.update(extra_create_args)

        if json_output is True:
            payload["response_format"] = {"type": "json_object"}
        elif json_output is not None and json_output is not False:
            try:
                from pydantic import BaseModel
            except ImportError as exc:  # pragma: no cover - project dependency
                raise RuntimeError("Pydantic is required for structured output") from exc
            if not isinstance(json_output, type) or not issubclass(json_output, BaseModel):
                raise ValueError("json_output must be a boolean or a Pydantic BaseModel class")
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_output.__name__,
                    "strict": True,
                    "schema": json_output.model_json_schema(),
                },
            }

        model_call_started = time.monotonic()
        queued_at = time.monotonic()
        timeout_plan = self.request_timeout_plan(max_new_tokens)
        with self.rate_limiter.slot():
            queue_latency = time.monotonic() - queued_at
            deadline = request_overrides.get("_request_deadline_monotonic_s")
            if deadline is not None:
                deadline_remaining_s = float(deadline) - time.monotonic()
                if deadline_remaining_s <= 0:
                    raise TimeoutError(
                        "Trial deadline reached while waiting for provider admission"
                    )
                timeout_plan["policy_timeout_s"] = timeout_plan["timeout_s"]
                timeout_plan["deadline_remaining_s"] = round(deadline_remaining_s, 3)
                timeout_plan["timeout_s"] = round(
                    min(float(timeout_plan["timeout_s"]), deadline_remaining_s),
                    3,
                )
                timeout_plan["capped_by_trial_deadline"] = (
                    timeout_plan["timeout_s"] < timeout_plan["policy_timeout_s"]
                )
            request_started = time.monotonic()
            request_client = self.client
            with_options = getattr(request_client, "with_options", None)
            if callable(with_options):
                request_client = with_options(
                    timeout=float(timeout_plan["timeout_s"]),
                    max_retries=int(timeout_plan["max_retries"]),
                )
            response = request_client.chat.completions.create(**payload)
            latency = time.monotonic() - request_started
        postprocess_started = time.monotonic()
        choice = response.choices[0] if response.choices else None
        provider_message_content = choice.message.content if choice and choice.message else None
        text = provider_message_content if isinstance(provider_message_content, str) else ""
        reasoning_content = ""
        if choice and choice.message:
            reasoning_content = str(
                getattr(choice.message, "reasoning_content", None)
                or getattr(choice.message, "reasoning", None)
                or ""
            )
        raw_tool_calls = []
        if choice and choice.message:
            for call in getattr(choice.message, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                if function is None:
                    continue
                raw_tool_calls.append(
                    {
                        "id": str(getattr(call, "id", "") or ""),
                        "name": str(getattr(function, "name", "") or ""),
                        "arguments": str(getattr(function, "arguments", "") or "{}"),
                    }
                )
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        request_metrics = _provider_request_metrics(response)
        provider_generation_latency_s = (
            float(request_metrics["generation_time_ms"]) / 1000.0
            if request_metrics.get("generation_time_ms") is not None
            else None
        )
        self._observe_generation_throughput(
            completion_tokens,
            provider_generation_latency_s or latency,
        )
        usage_payload = to_jsonable(usage) if usage is not None else {}
        completion_details = (
            usage_payload.get("completion_tokens_details", {})
            if isinstance(usage_payload, dict)
            else {}
        )
        (
            cached_tokens_value,
            cached_tokens_source,
            cached_tokens_status,
        ) = _cached_input_usage(usage_payload)
        reasoning_tokens_value = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        )
        reasoning_tokens: int | None = (
            int(reasoning_tokens_value) if reasoning_tokens_value is not None else None
        )
        answer_tokens: int | None
        reasoning_tokens_source: str | None
        answer_tokens_source: str | None
        if reasoning_tokens is not None:
            answer_tokens = max(0, completion_tokens - reasoning_tokens)
            reasoning_tokens_source = "provider_usage"
            answer_tokens_source = "provider_usage_derived"
        else:
            reasoning_tokens = self._retokenize(reasoning_content)
            answer_tokens = self._retokenize(text)
            reasoning_tokens_source = "client_retokenized" if reasoning_tokens is not None else None
            answer_tokens_source = "client_retokenized" if answer_tokens is not None else None
        provider_response_payload = to_jsonable(response)
        postprocess_latency = time.monotonic() - postprocess_started
        model_call_wall_time = time.monotonic() - model_call_started
        return APIGenResult(
            text or "",
            prompt_tokens,
            completion_tokens,
            latency,
            request_queue_latency_s=queue_latency,
            client_rate_limiter_wait_s=queue_latency,
            client_http_request_latency_s=latency,
            client_response_postprocess_latency_s=postprocess_latency,
            client_model_call_wall_time_s=model_call_wall_time,
            provider_request_queue_latency_s=(
                float(request_metrics["queue_time_ms"]) / 1000.0
                if request_metrics.get("queue_time_ms") is not None
                else None
            ),
            provider_scheduled_to_first_token_s=(
                float(request_metrics["time_to_first_token_ms"]) / 1000.0
                if request_metrics.get("time_to_first_token_ms") is not None
                else None
            ),
            provider_generation_latency_s=provider_generation_latency_s,
            provider_mean_inter_token_latency_s=(
                float(request_metrics["mean_itl_ms"]) / 1000.0
                if request_metrics.get("mean_itl_ms") is not None
                else None
            ),
            provider_output_tokens_per_second=(
                float(request_metrics["tokens_per_second"])
                if request_metrics.get("tokens_per_second") is not None
                else None
            ),
            provider_request_metrics=request_metrics,
            tool_calls=raw_tool_calls,
            finish_reason=str(getattr(choice, "finish_reason", None) or "stop"),
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens_source=reasoning_tokens_source,
            answer_tokens_source=answer_tokens_source,
            cached_input_tokens=(
                int(cached_tokens_value) if cached_tokens_value is not None else None
            ),
            cached_input_tokens_source=cached_tokens_source,
            cached_input_tokens_status=cached_tokens_status,
            provider_message_content=provider_message_content,
            provider_reasoning_content=reasoning_content or None,
            raw_decoded_text=None,
            provider_request_payload=to_jsonable(payload),
            provider_response_payload=provider_response_payload,
            request_timeout_s=float(timeout_plan["timeout_s"]),
            request_max_retries=int(timeout_plan["max_retries"]),
            observed_generation_tokens_per_second=(
                round(float(getattr(self, "_observed_generation_tps", 0.0)), 3)
                if getattr(self, "_observed_generation_tps", None)
                else None
            ),
            timeout_policy=dict(timeout_plan),
        )

    def generate_chat_with_prefix(self, messages: List[Dict], prefix, max_new_tokens: int = 256):
        raise NotImplementedError(
            "OpenAICompatibleBackend does not support latent prefix generation. "
            "Use method=none or method=nl_only for API runs."
        )

    def encode_hidden(self, text: str, max_tokens: int = 4096):
        raise NotImplementedError(
            "OpenAICompatibleBackend cannot expose hidden states. "
            "Use the local HF backend for latent-channel experiments."
        )
