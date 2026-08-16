from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from lychee_mas.memory.context import RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.token_budget import (
    normalize_model_context_policy,
    normalize_token_budget_policy,
    prepare_model_request,
)


class _WordBackend:
    context_window = 100

    @staticmethod
    def count_chat_tokens(messages, tools=()):
        words = sum(len(str(message.get("content") or "").split()) for message in messages)
        return words + len(tools), "test_words"


class _Memory:
    name = "test"

    def observe(self, messages):
        return None

    def recall(self, decision, query):
        return SimpleNamespace(
            NL_Channel=None,
            Latent_Channel=None,
            NL_strategy="none",
            Latent_strategy="none",
        )


def test_two_level_budget_uses_actual_remaining_window() -> None:
    prepared = prepare_model_request(
        _WordBackend(),
        [{"role": "user", "content": " ".join(["x"] * 70)}],
        max_new_tokens=50,
        budget_policy={
            "min_output_reserve_tokens": 20,
            "min_thinking_reserve_tokens": 5,
            "max_thinking_budget_tokens": 18,
            "min_final_reserve_tokens": 7,
            "safety_margin_tokens": 5,
        },
    )
    assert prepared.allocation.effective_max_new_tokens == 25
    assert prepared.allocation.effective_max_thinking_budget_tokens == 18
    assert prepared.allocation.input_tokens_after == 70
    assert prepared.allocation.token_count_method == "test_words"


def test_budget_constraint_requires_thinking_plus_final_inside_output_reserve() -> None:
    with pytest.raises(ValueError, match="must be <= min_output_reserve_tokens"):
        normalize_token_budget_policy(
            {
                "min_output_reserve_tokens": 10,
                "min_thinking_reserve_tokens": 6,
                "min_final_reserve_tokens": 5,
            },
            max_new_tokens=32,
        )


def test_buffered_context_matches_autogen_recent_message_semantics() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "latest"},
    ]
    prepared = prepare_model_request(
        _WordBackend(),
        messages,
        max_new_tokens=20,
        budget_policy={
            "min_output_reserve_tokens": 1,
            "min_final_reserve_tokens": 0,
            "safety_margin_tokens": 0,
        },
        model_context_policy={"type": "buffered", "buffer_size": 2},
    )
    assert [message["content"] for message in prepared.messages] == ["middle", "latest"]
    assert prepared.allocation.dropped_messages == 2


def test_model_context_policies_are_strictly_validated() -> None:
    assert normalize_model_context_policy({"type": "unbounded"}) == {
        "type": "unbounded"
    }
    assert normalize_model_context_policy(
        {"type": "token_limited", "token_limit": 4096}
    ) == {"type": "token_limited", "token_limit": 4096}
    with pytest.raises(ValueError, match="requires buffer_size"):
        normalize_model_context_policy({"type": "buffered"})


def test_model_call_span_records_configured_and_effective_budgets(tmp_path) -> None:
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_plain_client
    from lychee_mas.runtime.spans import JsonlSpanLogger

    class Backend(_WordBackend):
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = True
        supports_native_tools = False
        model_info = {}

        def generate_chat(self, messages, max_new_tokens=64, request_overrides=None):
            assert max_new_tokens == 25
            assert request_overrides == {
                "extra_body": {"thinking_token_budget": 18}
            }
            return SimpleNamespace(
                text="Solver",
                reasoning_content="",
                n_prompt_pos=70,
                n_gen_tokens=2,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={},
                provider_response_payload={},
            )

    path = tmp_path / "spans.jsonl"
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=JsonlSpanLogger(str(path)),
    )
    ctx.set_case("budget-case", 0)
    client = make_plain_client(
        Backend(),
        ctx=ctx,
        max_new_tokens=50,
        request_overrides={
            "token_budget_policy": {
                "min_output_reserve_tokens": 20,
                "min_thinking_reserve_tokens": 5,
                "max_thinking_budget_tokens": 18,
                "min_final_reserve_tokens": 7,
                "safety_margin_tokens": 5,
            },
            "extra_body": {"thinking_token_budget": 18},
        },
    )
    asyncio.run(
        client.create(
            [UserMessage(content=" ".join(["x"] * 70), source="user")]
        )
    )
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert start["configured_max_new_tokens"] == 50
    assert start["effective_max_new_tokens"] == 25
    assert start["effective_max_thinking_budget_tokens"] == 18
    assert end["configured_max_new_tokens"] == 50
    assert end["effective_max_new_tokens"] == 25
