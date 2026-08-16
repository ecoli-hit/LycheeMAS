"""Backend-neutral model-context and generation-budget policy.

The allocator has two independent jobs:

1. limit the messages visible to one model call; and
2. split the remaining model window between input, reasoning, and final output.

It deliberately contains no AutoGen, Transformers, or provider imports so the
runtime skeleton remains importable in the offline development environment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

MODEL_CONTEXT_TYPES = {"unbounded", "buffered", "token_limited"}
BUDGET_POLICY_FIELDS = (
    "max_input_tokens",
    "min_output_reserve_tokens",
    "min_thinking_reserve_tokens",
    "max_thinking_budget_tokens",
    "min_final_reserve_tokens",
    "safety_margin_tokens",
)


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _non_negative_int(value: Any, *, name: str, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def normalize_model_context_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the AutoGen model-context subset exposed by LycheeMAS."""

    policy = dict(value or {"type": "unbounded"})
    policy_type = str(policy.get("type") or "unbounded")
    if policy_type not in MODEL_CONTEXT_TYPES:
        raise ValueError(
            f"unsupported model context type {policy_type!r}; "
            f"choose {sorted(MODEL_CONTEXT_TYPES)}"
        )
    if policy_type == "unbounded":
        unknown = set(policy) - {"type"}
        if unknown:
            raise ValueError(
                "unbounded model context does not accept fields: "
                + ", ".join(sorted(unknown))
            )
        return {"type": "unbounded"}
    if policy_type == "buffered":
        unknown = set(policy) - {"type", "buffer_size"}
        if unknown:
            raise ValueError(
                "buffered model context does not accept fields: "
                + ", ".join(sorted(unknown))
            )
        size = _optional_positive_int(policy.get("buffer_size"), name="buffer_size")
        if size is None:
            raise ValueError("buffered model context requires buffer_size")
        return {"type": "buffered", "buffer_size": size}

    unknown = set(policy) - {"type", "token_limit"}
    if unknown:
        raise ValueError(
            "token_limited model context does not accept fields: "
            + ", ".join(sorted(unknown))
        )
    token_limit = _optional_positive_int(policy.get("token_limit"), name="token_limit")
    return {
        "type": "token_limited",
        **({"token_limit": token_limit} if token_limit is not None else {}),
    }


@dataclass(frozen=True)
class TokenBudgetPolicy:
    """Configured limits for one role or GroupChat controller invocation."""

    max_input_tokens: int | None
    min_output_reserve_tokens: int
    min_thinking_reserve_tokens: int
    max_thinking_budget_tokens: int | None
    min_final_reserve_tokens: int
    safety_margin_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_token_budget_policy(
    value: Mapping[str, Any] | None,
    *,
    max_new_tokens: int,
) -> TokenBudgetPolicy:
    """Validate one two-level input/output and thinking/final budget policy."""

    maximum = int(max_new_tokens)
    if maximum < 1:
        raise ValueError("max_new_tokens must be positive")
    raw = dict(value or {})
    unknown = set(raw) - set(BUDGET_POLICY_FIELDS)
    if unknown:
        raise ValueError("unknown token budget fields: " + ", ".join(sorted(unknown)))

    max_input = _optional_positive_int(raw.get("max_input_tokens"), name="max_input_tokens")
    min_output = _non_negative_int(
        raw.get("min_output_reserve_tokens"),
        name="min_output_reserve_tokens",
        default=min(maximum, 1024),
    )
    min_thinking = _non_negative_int(
        raw.get("min_thinking_reserve_tokens"),
        name="min_thinking_reserve_tokens",
    )
    max_thinking = _optional_positive_int(
        raw.get("max_thinking_budget_tokens"),
        name="max_thinking_budget_tokens",
    )
    min_final = _non_negative_int(
        raw.get("min_final_reserve_tokens"),
        name="min_final_reserve_tokens",
        default=min(min_output, 512),
    )
    safety = _non_negative_int(
        raw.get("safety_margin_tokens"),
        name="safety_margin_tokens",
        default=256,
    )

    if min_thinking + min_final > min_output:
        raise ValueError(
            "min_thinking_reserve_tokens + min_final_reserve_tokens must be "
            "<= min_output_reserve_tokens"
        )
    if min_output > maximum:
        raise ValueError("min_output_reserve_tokens must be <= max_new_tokens")
    if max_thinking is not None:
        if min_thinking > max_thinking:
            raise ValueError(
                "min_thinking_reserve_tokens must be <= max_thinking_budget_tokens"
            )
        if max_thinking + min_final > maximum:
            raise ValueError(
                "max_thinking_budget_tokens + min_final_reserve_tokens must be "
                "<= max_new_tokens"
            )

    return TokenBudgetPolicy(
        max_input_tokens=max_input,
        min_output_reserve_tokens=min_output,
        min_thinking_reserve_tokens=min_thinking,
        max_thinking_budget_tokens=max_thinking,
        min_final_reserve_tokens=min_final,
        safety_margin_tokens=safety,
    )


@dataclass(frozen=True)
class TokenBudgetAllocation:
    """Effective per-call budget after context filtering and token counting."""

    context_window_tokens: int | None
    configured_max_new_tokens: int
    effective_max_new_tokens: int
    configured_max_input_tokens: int | None
    effective_max_input_tokens: int | None
    input_tokens_before: int
    input_tokens_after: int
    configured_max_thinking_budget_tokens: int | None
    effective_max_thinking_budget_tokens: int | None
    min_output_reserve_tokens: int
    min_thinking_reserve_tokens: int
    min_final_reserve_tokens: int
    safety_margin_tokens: int
    dropped_messages: int
    prompt_truncated: bool
    token_count_method: str
    model_context_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def span_fields(self) -> dict[str, Any]:
        return {
            "context_window_tokens": self.context_window_tokens,
            "configured_max_new_tokens": self.configured_max_new_tokens,
            "effective_max_new_tokens": self.effective_max_new_tokens,
            "configured_max_input_tokens": self.configured_max_input_tokens,
            "effective_max_input_tokens": self.effective_max_input_tokens,
            "input_tokens_before_budgeting": self.input_tokens_before,
            "input_tokens_after_budgeting": self.input_tokens_after,
            "configured_max_thinking_budget_tokens": (
                self.configured_max_thinking_budget_tokens
            ),
            "effective_max_thinking_budget_tokens": (
                self.effective_max_thinking_budget_tokens
            ),
            "min_output_reserve_tokens": self.min_output_reserve_tokens,
            "min_thinking_reserve_tokens": self.min_thinking_reserve_tokens,
            "min_final_reserve_tokens": self.min_final_reserve_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "budget_dropped_messages": self.dropped_messages,
            "budget_prompt_truncated": self.prompt_truncated,
            "token_count_method": self.token_count_method,
            "model_context_policy": self.model_context_policy,
        }


@dataclass(frozen=True)
class PreparedModelRequest:
    messages: list[dict[str, Any]]
    allocation: TokenBudgetAllocation


def _json_token_estimate(messages: Sequence[Mapping[str, Any]], tools: Sequence[Any]) -> int:
    serialized = json.dumps(
        {"messages": list(messages), "tools": list(tools)},
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return max(1, (len(serialized) + 3) // 4)


def count_request_tokens(
    backend: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Any] = (),
) -> tuple[int, str]:
    """Count a serialized chat request, with an explicit estimated fallback."""

    counter = getattr(backend, "count_chat_tokens", None)
    if callable(counter):
        value = counter(list(messages), tools=list(tools))
        if isinstance(value, tuple):
            return int(value[0]), str(value[1])
        return int(value), "backend"

    tokenizer = getattr(backend, "tok", None)
    if tokenizer is not None:
        try:
            kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = list(tools)
            token_ids = tokenizer.apply_chat_template(list(messages), **kwargs)
            tool_method = "+tools" if tools else ""
            return len(token_ids), f"tokenizer_chat_template{tool_method}"
        except Exception:
            pass
    return _json_token_estimate(messages, tools), "estimated_json_chars_div_4"


def _history_groups(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[list[dict]]]:
    """Keep system messages pinned and tool-call/result pairs atomic."""

    systems: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    for raw in messages:
        message = dict(raw)
        if message.get("role") == "system":
            systems.append(message)
            continue
        if message.get("role") == "tool" and groups:
            previous = groups[-1][-1]
            if previous.get("role") == "assistant" or groups[-1][0].get("role") == "assistant":
                groups[-1].append(message)
                continue
        groups.append([message])
    return systems, groups


def _flatten_groups(systems: Sequence[dict], groups: Sequence[Sequence[dict]]) -> list[dict]:
    return [*systems, *(message for group in groups for message in group)]


def _apply_buffered_policy(
    messages: Sequence[Mapping[str, Any]], buffer_size: int
) -> tuple[list[dict[str, Any]], int]:
    original = [dict(message) for message in messages]
    result = original[-buffer_size:]
    if result and result[0].get("role") == "tool":
        result = result[1:]
    return result, max(0, len(messages) - len(result))


def _apply_token_limited_policy(
    messages: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    count: Callable[[Sequence[Mapping[str, Any]]], int],
) -> tuple[list[dict[str, Any]], int]:
    """Mirror AutoGen TokenLimitedChatCompletionContext's middle eviction."""

    current = [dict(message) for message in messages]
    dropped = 0
    while current and count(current) > limit:
        current.pop(len(current) // 2)
        dropped += 1
    if current and current[0].get("role") == "tool":
        current.pop(0)
        dropped += 1
    return current, dropped


def _trim_to_token_limit(
    messages: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    count: Callable[[Sequence[Mapping[str, Any]]], int],
) -> tuple[list[dict[str, Any]], int]:
    systems, groups = _history_groups(messages)
    dropped = 0
    current = _flatten_groups(systems, groups)
    while count(current) > limit and len(groups) > 1:
        removed = groups.pop(0)
        dropped += len(removed)
        current = _flatten_groups(systems, groups)
    if count(current) > limit:
        raise ValueError(
            "the pinned system messages and latest message exceed the effective input "
            f"budget of {limit} tokens"
        )
    return current, dropped


def prepare_model_request(
    backend: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Any] = (),
    max_new_tokens: int,
    budget_policy: Mapping[str, Any] | None = None,
    model_context_policy: Mapping[str, Any] | None = None,
) -> PreparedModelRequest:
    """Apply model context, trim overflow, and derive effective output budgets."""

    policy = normalize_token_budget_policy(
        budget_policy,
        max_new_tokens=max_new_tokens,
    )
    context_policy = normalize_model_context_policy(model_context_policy)
    original = [dict(message) for message in messages]
    before, method = count_request_tokens(backend, original, tools)
    current = original
    dropped = 0

    if context_policy["type"] == "buffered":
        current, removed = _apply_buffered_policy(current, context_policy["buffer_size"])
        dropped += removed
    elif context_policy["type"] == "token_limited" and context_policy.get("token_limit"):
        current, removed = _apply_token_limited_policy(
            current,
            limit=int(context_policy["token_limit"]),
            count=lambda values: count_request_tokens(backend, values, tools)[0],
        )
        dropped += removed

    context_window_value = getattr(backend, "context_window", None)
    context_window = (
        int(context_window_value)
        if context_window_value not in (None, "", 0, "0")
        else None
    )
    context_input_limit = None
    if context_window is not None:
        context_input_limit = (
            context_window
            - policy.min_output_reserve_tokens
            - policy.safety_margin_tokens
        )
        if context_input_limit < 1:
            raise ValueError(
                "context window is too small for min_output_reserve_tokens and "
                "safety_margin_tokens"
            )
    limits = [
        value
        for value in (policy.max_input_tokens, context_input_limit)
        if value is not None
    ]
    effective_input_limit = min(limits) if limits else None
    if effective_input_limit is not None:
        current, removed = _trim_to_token_limit(
            current,
            limit=effective_input_limit,
            count=lambda values: count_request_tokens(backend, values, tools)[0],
        )
        dropped += removed

    after, method = count_request_tokens(backend, current, tools)
    effective_output = int(max_new_tokens)
    if context_window is not None:
        remaining = context_window - after - policy.safety_margin_tokens
        if remaining < policy.min_output_reserve_tokens:
            raise ValueError(
                "input leaves fewer tokens than min_output_reserve_tokens after "
                "context trimming"
            )
        effective_output = min(effective_output, remaining)

    effective_thinking = None
    if policy.max_thinking_budget_tokens is not None:
        effective_thinking = min(
            policy.max_thinking_budget_tokens,
            max(0, effective_output - policy.min_final_reserve_tokens),
        )
        if effective_thinking < policy.min_thinking_reserve_tokens:
            raise ValueError(
                "effective thinking budget is smaller than min_thinking_reserve_tokens"
            )

    allocation = TokenBudgetAllocation(
        context_window_tokens=context_window,
        configured_max_new_tokens=int(max_new_tokens),
        effective_max_new_tokens=int(effective_output),
        configured_max_input_tokens=policy.max_input_tokens,
        effective_max_input_tokens=effective_input_limit,
        input_tokens_before=before,
        input_tokens_after=after,
        configured_max_thinking_budget_tokens=policy.max_thinking_budget_tokens,
        effective_max_thinking_budget_tokens=effective_thinking,
        min_output_reserve_tokens=policy.min_output_reserve_tokens,
        min_thinking_reserve_tokens=policy.min_thinking_reserve_tokens,
        min_final_reserve_tokens=policy.min_final_reserve_tokens,
        safety_margin_tokens=policy.safety_margin_tokens,
        dropped_messages=dropped,
        prompt_truncated=dropped > 0,
        token_count_method=method,
        model_context_policy=context_policy,
    )
    return PreparedModelRequest(messages=current, allocation=allocation)
