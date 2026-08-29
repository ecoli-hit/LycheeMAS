from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from lychee_mas.memory.context import RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.model.token_budget import (
    normalize_model_context_policy,
    normalize_token_budget_policy,
    prepare_model_request,
    tokenized_sequence_length,
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


def test_tokenized_sequence_length_reads_batch_encoding_input_ids() -> None:
    batch_encoding = {
        "input_ids": list(range(98_305)),
        "attention_mask": [1] * 98_305,
    }

    assert len(batch_encoding) == 2
    assert tokenized_sequence_length(batch_encoding) == 98_305


def test_batch_encoding_count_cannot_overrun_context_window() -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            count = sum(len(str(item.get("content") or "")) for item in messages)
            return {
                "input_ids": list(range(count)),
                "attention_mask": [1] * count,
            }

    backend = SimpleNamespace(context_window=100, tok=Tokenizer())
    prepared = prepare_model_request(
        backend,
        [
            {"role": "system", "content": "s" * 5},
            {"role": "user", "content": "u" * 20},
            {"role": "assistant", "content": "m" * 40},
            {"role": "assistant", "content": "l" * 30},
        ],
        max_new_tokens=40,
        budget_policy={
            "min_output_reserve_tokens": 10,
            "min_final_reserve_tokens": 0,
            "safety_margin_tokens": 5,
        },
    )

    allocation = prepared.allocation
    assert allocation.input_tokens_before == 95
    assert allocation.input_tokens_after == 55
    assert allocation.dropped_messages == 1
    assert allocation.effective_max_new_tokens == 40
    assert (
        allocation.input_tokens_after
        + allocation.effective_max_new_tokens
        + allocation.safety_margin_tokens
        <= allocation.context_window_tokens
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


def test_context_budget_preserves_user_task_and_atomic_latest_tool_exchange() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task anchor"},
        {"role": "assistant", "content": "old analysis"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "function": {"name": "read"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": " ".join(["tool-result"] * 70),
        },
    ]
    prepared = prepare_model_request(
        _WordBackend(),
        messages,
        max_new_tokens=20,
        budget_policy={
            "max_input_tokens": 75,
            "min_output_reserve_tokens": 1,
            "min_final_reserve_tokens": 0,
            "safety_margin_tokens": 0,
        },
    )

    roles = [message["role"] for message in prepared.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert prepared.messages[1]["content"] == "original task anchor"
    assert prepared.messages[-1]["tool_call_id"] == "call-1"
    assert prepared.allocation.dropped_messages == 1


def test_model_context_policies_are_strictly_validated() -> None:
    assert normalize_model_context_policy({"type": "unbounded"}) == {"type": "unbounded"}
    assert normalize_model_context_policy({"type": "token_limited", "token_limit": 4096}) == {
        "type": "token_limited",
        "token_limit": 4096,
    }
    with pytest.raises(ValueError, match="requires buffer_size"):
        normalize_model_context_policy({"type": "buffered"})


def test_model_call_event_records_configured_and_effective_budgets(tmp_path) -> None:
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter

    class Backend(_WordBackend):
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = True
        supports_native_tools = False
        model_info = {}

        def generate_chat(self, messages, max_new_tokens=64, request_overrides=None):
            assert max_new_tokens == 25
            assert request_overrides == {"extra_body": {"thinking_token_budget": 18}}
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

    path = tmp_path / "run_events.jsonl"
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=RunEventWriter(path),
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
        controller=True,
    )
    asyncio.run(client.create([UserMessage(content=" ".join(["x"] * 70), source="user")]))
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert start["payload"]["configured_max_new_tokens"] == 50
    assert start["payload"]["effective_max_new_tokens"] == 25
    assert start["payload"]["effective_max_thinking_budget_tokens"] == 18
    assert end["payload"]["configured_max_new_tokens"] == 50
    assert end["payload"]["effective_max_new_tokens"] == 25
