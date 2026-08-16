"""Offline tests for benchmark topology, team inspection, and run controls."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.eval.benchmarks.manifest import (
    verify_prepared_manifest,
    write_prepared_manifest,
)
from lychee_mas.eval.benchmarks.topology import graph_from_benchmark_record
from lychee_mas.memory.context import ModelCallBudgetExceeded, RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.concurrency import (
    CaseConcurrencyController,
    normalize_concurrency_policy,
)
from lychee_mas.runtime.seeding import (
    PREDICTION_SEED_DERIVATION,
    derive_prediction_seed,
)


class _Memory:
    def reset(self):
        pass

    def observe(self, messages):
        pass

    def recall(self, decision, query):
        return SimpleNamespace(
            NL_Channel=None,
            Latent_Channel=None,
            NL_strategy=None,
            Latent_strategy=None,
        )


def _record():
    return {
        "topology": {
            "type": "linear_chain",
            "agents": [
                {
                    "agent_id": "A1",
                    "role": "Planner",
                    "system_prompt": "Plan.",
                    "receives_from": [],
                },
                {
                    "agent_id": "A2",
                    "role": "Solver",
                    "system_prompt": "Solve.",
                    "receives_from": ["A1"],
                },
            ],
            "edges": [["A1", "A2"]],
            "speaking_order": ["A1", "A2"],
        }
    }


def test_record_topology_builds_real_graph():
    graph, dynamic = graph_from_benchmark_record(_record(), fallback_profile="reason", rounds=2)
    assert dynamic is True
    assert graph.names == ["A1", "A2"]
    assert graph.edges == {"A1": ["A2"], "A2": []}
    assert graph.meta["group_chat"] == {
        "type": "selector",
        "selector_func_factory": "topology_selector",
    }
    assert graph.meta["speaking_order"] == ["A1", "A2"]


def test_contamination_audit_separates_official_filename_risk_from_solution_hit(tmp_path):
    from lychee_mas.eval.contamination import audit_run

    case_id = "32102e3e-d12a-4209-9163-7b3a104efe5d"
    spans = [
        {
            "seq": 1,
            "span_type": "workspace_prepared",
            "case_id": case_id,
            "k_index": 0,
            "workspace": str(tmp_path / "ws_0123456789abcdef0123456789abcdef"),
            "workspace_is_new": True,
            "preexisting_entry_count": 0,
            "workspace_policy": "uuid4_isolated_per_attempt_v1",
            "attachment_filename_policy": "preserve_official_filename",
            "visible_attachment_names": [f"{case_id}.xlsx"],
        }
    ]
    chats = [
        {
            "seq": 2,
            "event_type": "message",
            "case_id": case_id,
            "k_index": 0,
            "source": "WebSurfer",
            "content": (
                "I opened https://example.test/gaia/solutions/"
                f"{case_id}/answer and found FINAL ANSWER: 42"
            ),
        }
    ]
    for name, records in (("spans.jsonl", spans), ("group_chat.jsonl", chats)):
        (tmp_path / name).write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    audits, summary = audit_run(
        tmp_path,
        samples=[{"case_id": case_id, "k_index": 0, "score": 1.0}],
        gold_by_id={case_id: {"gold": "42"}},
    )

    assert audits[0]["risk_flags"] == ["official_attachment_task_id_exposed"]
    assert "workspace_case_id_exposed" not in audits[0]["contamination_flags"]
    assert "known_solution_page" in audits[0]["contamination_flags"]
    assert "gold_like_text_in_web_evidence" in audits[0]["contamination_flags"]
    assert audits[0]["contamination_suspected"] is True
    assert summary["num_audited_cases"] == 1
    assert summary["num_suspected_cases"] == 1


def test_contamination_audit_detects_legacy_workspace_reuse_without_web_false_positive(
    tmp_path,
):
    from lychee_mas.eval.contamination import audit_run

    case_id = "c61d22de-5f6c-4958-a7f6-5e9707bd3466"
    legacy = str(tmp_path / case_id)
    spans = [
        {
            "seq": seq,
            "span_type": "workspace_prepared",
            "case_id": case_id,
            "k_index": 0,
            "workspace": legacy,
            "copied_files": [],
        }
        for seq in (1, 2)
    ]
    (tmp_path / "spans.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in spans), encoding="utf-8"
    )
    (tmp_path / "group_chat.jsonl").write_text(
        json.dumps(
            {
                "seq": 3,
                "case_id": case_id,
                "k_index": 0,
                "source": "WebSurfer",
                "content": "The ordinary page contains the number 42 in an unrelated table.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audits, summary = audit_run(
        tmp_path,
        samples=[{"case_id": case_id, "k_index": 0, "score": 0.0}],
        gold_by_id={case_id: {"gold": "42"}},
    )

    assert "workspace_case_id_exposed" in audits[0]["contamination_flags"]
    assert "workspace_reused" in audits[0]["contamination_flags"]
    assert "gold_like_text_in_web_evidence" not in audits[0]["contamination_flags"]
    assert summary["flag_case_counts"]["workspace_reused"] == 1


def test_routing_context_declares_topology_filtered_visibility():
    graph, _ = graph_from_benchmark_record(_record(), fallback_profile="reason", rounds=2)
    ctx = RoutingContext(
        task="agent_collab_idr",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    ctx.configure_graph(graph)
    assert ctx.context_visibility == "topology_filtered"
    assert ctx.graph_edges == {"A1": ["A2"], "A2": []}


def test_routing_context_enforces_one_hard_model_call_budget_per_case():
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        max_model_calls_per_case=2,
    )

    assert ctx.reserve_model_call("Orchestrator", controller=True) == 1
    assert ctx.reserve_model_call("Coder") == 2
    with pytest.raises(ModelCallBudgetExceeded, match="Maximum model calls per case 2"):
        ctx.reserve_model_call("WebSurfer")
    assert ctx.model_calls_started == 2
    assert ctx.model_call_budget_exhausted is True

    ctx.reset(preserve_model_call_budget=True)
    assert ctx.model_calls_started == 2
    ctx.reset()
    assert ctx.model_calls_started == 0
    assert ctx.model_call_budget_exhausted is False


def test_adaptive_case_concurrency_uses_aimd_without_changing_service_capacity():
    policy = normalize_concurrency_policy(
        {
            "mode": "auto",
            "initial": 4,
            "minimum": 2,
            "maximum": 8,
            "increase_step": 2,
            "decrease_factor": 0.5,
            "control_window_cases": 2,
        },
        configured_concurrency=4,
    )
    controller = CaseConcurrencyController(policy)

    async def exercise():
        assert await controller.observe(record={"status": "completed"}) is None
        increased = await controller.observe(record={"status": "completed"})
        assert increased == {
            "timestamp_unix_s": increased["timestamp_unix_s"],
            "reason": "healthy_window",
            "previous": 4,
            "target": 6,
        }
        assert (
            await controller.observe(record={"status": "error", "error_type": "APITimeoutError"})
            is None
        )
        decreased = await controller.observe(record={"status": "completed"})
        assert decreased["reason"] == "overload_signal"
        assert decreased["previous"] == 6
        assert decreased["target"] == 3

    asyncio.run(exercise())
    assert policy["maximum"] == 8
    assert [event["target"] for event in controller.history] == [4, 6, 3]


def test_costing_uses_deployment_pricing_snapshots_and_provider_token_details(
    tmp_path: Path,
):
    from lychee_mas.eval.costing import attach_cost_metrics

    snapshot_dir = tmp_path / "config_snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "configuration_snapshot.json").write_text(
        json.dumps(
            {
                "deployment_instances": [
                    {
                        "id": "instance-qwen",
                        "kind": "vllm",
                        "cuda_visible_devices": "4,5",
                        "tensor_parallel_size": 2,
                        "data_parallel_size": 1,
                        "actual_pricing_instance_id": "local-a800",
                        "api_equivalent_pricing_instance_id": "dashscope-qwen",
                    }
                ],
                "pricing_specs": [
                    {
                        "id": "gpu",
                        "basis": "allocated_gpu_time",
                        "billing_mode": "monthly_node_amortized",
                        "amortization_hours_per_month": 730,
                        "amortization_policy": "allocated_gpu_share",
                    },
                    {"id": "tokens", "basis": "token_usage"},
                ],
                "pricing_instances": [
                    {
                        "id": "local-a800",
                        "pricing_spec_id": "gpu",
                        "currency": "CNY",
                        "rates": {"gpu_hour": 5.0},
                        "metadata": {
                            "accelerator": "NVIDIA A800",
                            "quoted_node_month": 29200.0,
                            "node_gpu_count": 8,
                        },
                    },
                    {
                        "id": "dashscope-qwen",
                        "pricing_spec_id": "tokens",
                        "currency": "CNY",
                        "rates": {
                            "input": 2.0,
                            "cached_input": 1.0,
                            "output": 4.0,
                            "reasoning_output": 6.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_status.json").write_text(
        json.dumps({"started_at_unix_s": 100.0, "finished_at_unix_s": 3700.0}),
        encoding="utf-8",
    )
    samples = [
        {
            "model_calls": [
                {
                    "deployment_instance_id": "instance-qwen",
                    "input_text_tokens": 100,
                    "input_cached_tokens": 40,
                    "output_text_tokens": 50,
                    "output_reasoning_tokens": 20,
                }
            ]
        }
    ]

    costing = attach_cost_metrics({}, samples, tmp_path)["costing"]

    actual = costing["actual_cost"]
    assert actual["totals_by_currency"] == {"CNY": 10.0}
    assert actual["line_items"][0]["allocated_gpu_hours"] == 2.0
    assert actual["line_items"][0]["billing_mode"] == "monthly_node_amortized"
    assert actual["line_items"][0]["quoted_node_month"] == 29200.0
    equivalent = costing["api_equivalent_cost"]
    assert equivalent["totals_by_currency"]["CNY"] == pytest.approx(0.0004)
    assert equivalent["line_items"][0]["usage"]["cached_input_tokens"] == 40
    assert equivalent["line_items"][0]["usage"]["reasoning_output_tokens"] == 20
    assert costing["model_call_source"] == "predictions.model_calls_fallback"

    (tmp_path / "spans.jsonl").write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "span_type": "model_call_end",
                    "deployment_instance_id": "instance-qwen",
                    "input_text_tokens": 200,
                    "input_cached_tokens": 100,
                    "output_text_tokens": 80,
                    "output_reasoning_tokens": 30,
                },
                {
                    "span_type": "model_call_error",
                    "deployment_instance_id": "instance-qwen",
                    "error_type": "APITimeoutError",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    span_costing = attach_cost_metrics({}, samples, tmp_path)["costing"]
    assert span_costing["model_call_source"] == "spans.model_call_end"
    assert span_costing["api_equivalent_cost"]["line_items"][0]["usage"]["input_tokens"] == 200
    assert span_costing["failed_model_calls_without_usage"] == 1
    assert span_costing["api_equivalent_cost"]["status"] == "calculated_lower_bound"
    assert (
        span_costing["api_equivalent_cost"]["line_items"][0]["failed_model_calls_without_usage"]
        == 1
    )


def test_costing_preserves_zero_optional_rates_and_resume_elapsed_segments(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.costing import _run_elapsed_seconds, _token_cost

    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 60,
        "output_tokens": 50,
        "reasoning_output_tokens": 20,
        "answer_output_tokens": 30,
        "cached_input_breakdown_available": True,
        "reasoning_output_breakdown_available": True,
    }
    cost, detail = _token_cost(
        usage,
        {
            "rates": {
                "input": 2.0,
                "cached_input": 0.0,
                "output": 4.0,
                "reasoning_output": 0.0,
            }
        },
    )
    assert cost == pytest.approx((60 * 2.0 + 30 * 4.0) / 1_000_000)
    assert detail["input_cost"] == pytest.approx(120 / 1_000_000)
    assert detail["output_cost"] == pytest.approx(120 / 1_000_000)
    assert detail["input_assumption"] == "provider_cached_token_breakdown_with_cached_rate"

    no_cached_rate_cost, no_cached_rate_detail = _token_cost(
        usage,
        {
            "rates": {
                "input": 2.0,
                "cached_input": None,
                "output": 4.0,
            }
        },
    )
    assert no_cached_rate_cost == pytest.approx((100 * 2.0 + 50 * 4.0) / 1_000_000)
    assert (
        no_cached_rate_detail["input_assumption"]
        == "provider_cached_token_breakdown_without_cached_rate_"
        "all_input_charged_at_standard_rate"
    )

    (tmp_path / "run_status.json").write_text(
        json.dumps(
            {
                "accumulated_run_elapsed_before_segment_s": 120.0,
                "current_segment_started_at_unix_s": 1000.0,
                "finished_at_unix_s": 1060.0,
                "cumulative_run_elapsed_s": 120.0,
            }
        ),
        encoding="utf-8",
    )
    elapsed, source = _run_elapsed_seconds(tmp_path)
    assert elapsed == 180.0
    assert source == "run_status.accumulated_segments"


def test_costing_selects_rate_tier_per_call_and_prices_thinking_mode_output():
    from lychee_mas.eval.costing import _token_cost, _token_usage

    calls = [
        {
            "input_text_tokens": 100_000,
            "input_cached_tokens": 20_000,
            "output_text_tokens": 10,
            "output_reasoning_tokens": 0,
            "invocation_policy": {"thinking_mode": "disabled"},
        },
        {
            "input_text_tokens": 200_000,
            "input_cached_tokens": 0,
            "output_text_tokens": 10,
            "output_reasoning_tokens": 5,
            "invocation_policy": {"thinking_mode": "enabled"},
        },
    ]
    pricing = {
        "rate_tiers": [
            {
                "up_to_input_tokens": 128_000,
                "rates": {
                    "input": 0.8,
                    "cached_input": 0.16,
                    "output": 2.0,
                    "thinking_output": 8.0,
                },
            },
            {
                "up_to_input_tokens": 256_000,
                "rates": {
                    "input": 2.4,
                    "cached_input": 0.48,
                    "output": 20.0,
                    "thinking_output": 24.0,
                },
            },
        ]
    }

    cost, detail = _token_cost(_token_usage(calls), pricing, calls)

    expected = (80_000 * 0.8 + 20_000 * 0.16 + 10 * 2.0 + 200_000 * 2.4 + 10 * 24.0) / 1_000_000
    assert cost == pytest.approx(expected)
    assert detail["output_assumption"] == (
        "all_output_charged_at_thinking_mode_rate,provider_reasoning_token_breakdown"
    )
    assert [item["model_call_count"] for item in detail["rate_tier_breakdown"]] == [
        1,
        1,
    ]


def test_effective_max_turns_uses_rounds_only_as_round_robin_fallback():
    from lychee_mas.runtime.backends.autogen_runtime import _effective_max_turns

    assert _effective_max_turns("round_robin", 3, max_rounds=4, max_turns=None) == 12
    assert _effective_max_turns("selector", 3, max_rounds=99, max_turns=17) == 17
    assert _effective_max_turns("magentic_one", 4, max_rounds=99, max_turns=None) == 20


def test_model_call_budget_exception_becomes_group_chat_stop_result():
    from autogen_agentchat.messages import TextMessage
    from lychee_mas.layers.construct.templates import RoleProfileTeamBuilder
    from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime

    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        max_model_calls_per_case=1,
    )

    class GroupChat:
        async def run_stream(self, *, task):
            yield TextMessage(content=f"partial: {task}", source="Coder")
            ctx.model_call_budget_exhausted = True
            raise RuntimeError("wrapped ModelCallBudgetExceeded")

    graph = RoleProfileTeamBuilder(team="single", rounds=1).build()
    runtime = AutoGenRuntime(
        backend=object(),
        ctx=ctx,
        max_model_calls_per_case=1,
    )
    result, requests, executions, errors = asyncio.run(
        runtime._run_group_chat_stream(GroupChat(), "question", graph)
    )

    assert [message.content for message in result.messages] == ["partial: question"]
    assert result.stop_reason == "Maximum model calls per case 1 reached."
    assert requests == executions == errors == []


def test_topology_visibility_uses_official_message_filter_agent():
    from autogen_agentchat.messages import TextMessage
    from lychee_mas.runtime.backends.autogen_runtime import _apply_official_message_filter

    node = SimpleNamespace(name="A2", meta={"receives_from": ["A1"]})
    wrapped = _apply_official_message_filter(
        SimpleNamespace(description="solver"),
        node,
        "topology_filtered",
    )
    messages = [
        TextMessage(content="task", source="user"),
        TextMessage(content="plan", source="A1"),
        TextMessage(content="private", source="A3"),
        TextMessage(content="prior", source="A2"),
    ]
    assert type(wrapped).__name__ == "MessageFilterAgent"
    assert [message.source for message in wrapped._apply_filter(messages)] == [
        "user",
        "A1",
        "A2",
    ]


def test_non_dynamic_graph_does_not_filter_messages():
    graph, dynamic = graph_from_benchmark_record({}, fallback_profile="reason", rounds=2)
    assert dynamic is False
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    ctx.configure_graph(graph)
    assert ctx.context_visibility == "shared"


def test_static_role_profile_declares_one_explicit_group_chat():
    from lychee_mas.layers.construct.templates import RoleProfileTeamBuilder
    from lychee_mas.runtime.backends.autogen_runtime import _group_chat_config

    graph = RoleProfileTeamBuilder(team="reason", rounds=2).build()
    assert graph.meta["team"] == "reason"
    assert _group_chat_config(graph) == {"type": "round_robin"}
    assert graph.meta["termination"]["conditions"] == [{"type": "text_mention", "text": "APPROVE"}]


def test_tool_metrics_use_typed_autogen_events():
    from autogen_core import FunctionCall
    from autogen_core.models import FunctionExecutionResult
    from lychee_mas.runtime.backends.autogen_runtime import _structured_tool_records

    requests, executions = _structured_tool_records(
        SimpleNamespace(
            type="ToolCallRequestEvent",
            source="Solver",
            content=[FunctionCall(id="call-1", name="lookup", arguments='{"q":"x"}')],
        ),
        3,
    )
    assert requests[0]["tool_name"] == "lookup"
    assert executions == []

    requests, executions = _structured_tool_records(
        SimpleNamespace(
            type="ToolCallExecutionEvent",
            source="Solver",
            content=[
                FunctionExecutionResult(
                    call_id="call-1",
                    name="lookup",
                    content="timeout",
                    is_error=True,
                )
            ],
        ),
        4,
    )
    assert requests == []
    assert executions[0]["is_error"] is True

    requests, executions = _structured_tool_records(
        SimpleNamespace(type="TextMessage", source="ComputerTerminal", content="error"),
        5,
    )
    assert requests == []
    assert executions == []


def test_docker_container_proxy_executes_as_host_user():
    from lychee_mas.runtime.backends.autogen_runtime import _DockerContainerUserProxy

    class Container:
        status = "running"

        def __init__(self):
            self.calls = []

        def exec_run(self, command, *args, **kwargs):
            self.calls.append((command, args, kwargs))
            return "ok"

    container = Container()
    proxy = _DockerContainerUserProxy(container, "1000:1000")

    assert proxy.status == "running"
    assert proxy.exec_run(["python", "solution.py"]) == "ok"
    assert container.calls[-1][2]["user"] == "1000:1000"

    proxy.exec_run(["id"], user="2000:2000")
    assert container.calls[-1][2]["user"] == "2000:2000"


def test_manifest_detects_missing_prepared_file(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    prepared_root = tmp_path / "prepared"
    raw_source = raw_root / "gsm8k" / "modelscope" / "AI-ModelScope--gsm8k"
    prepared = prepared_root / "gsm8k" / "main"
    raw_source.mkdir(parents=True)
    prepared.mkdir(parents=True)
    (raw_source / "train.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    data_file = prepared / "train.jsonl"
    data_file.write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", str(raw_root))
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", str(prepared_root))

    manifest = write_prepared_manifest("gsm8k", prepared, source_mode="auto", hash_files=True)
    assert verify_prepared_manifest(manifest)["status"] == "ready"
    data_file.unlink()
    result = verify_prepared_manifest(manifest)
    assert result["status"] == "invalid"
    assert result["missing_files"] == ["train.jsonl"]
    assert result["raw_sources_available"]


def test_network_backend_calls_are_concurrent_and_receive_request_seed():
    from lychee_mas.runtime.backends.autogen_injection_client import _call_backend

    class Backend:
        supports_concurrent_requests = True
        supports_request_seed = True

        def generate_chat(self, messages, *, seed=None):
            time.sleep(0.08)
            return seed

    async def run_calls():
        started = time.perf_counter()
        values = await asyncio.gather(
            _call_backend(Backend(), "generate_chat", [], request_seed=11),
            _call_backend(Backend(), "generate_chat", [], request_seed=12),
        )
        return values, time.perf_counter() - started

    values, elapsed = asyncio.run(run_calls())
    assert values == [11, 12]
    assert elapsed < 0.14


def test_prediction_seed_is_stable_by_case_identity_and_k_index():
    common = {
        "base_seed": 42,
        "benchmark_id": "gaia",
        "task": "gaia_validation",
        "case_id": "case-001",
    }
    first = derive_prediction_seed(**common, k_index=0)

    assert first == derive_prediction_seed(**common, k_index=0)
    assert 0 <= first < 2**31
    assert first != derive_prediction_seed(**common, k_index=1)
    assert first != derive_prediction_seed(**{**common, "case_id": "case-002"}, k_index=0)
    assert first != derive_prediction_seed(**{**common, "base_seed": 43}, k_index=0)
    assert PREDICTION_SEED_DERIVATION == "sha256_case_k_31bit_v1"


def test_local_hf_request_seed_resets_the_serialized_torch_rng():
    import torch
    from lychee_mas.runtime.backends.hf_backend import HFBackend

    backend = HFBackend.__new__(HFBackend)
    backend._set_request_seed(12345)
    first = torch.rand(4)
    backend._set_request_seed(12345)
    second = torch.rand(4)

    assert HFBackend.supports_request_seed is True
    assert torch.equal(first, second)


def test_routing_context_records_prediction_seed_on_spans_and_group_chat(tmp_path):
    from lychee_mas.runtime.spans import JsonlGroupChatLogger, JsonlSpanLogger

    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=JsonlSpanLogger(str(tmp_path / "spans.jsonl")),
        group_chat_logger=JsonlGroupChatLogger(str(tmp_path / "group_chat.jsonl")),
    )
    ctx.base_seed = 42
    ctx.prediction_seed = 123456
    ctx.generation_seed = 123456
    ctx.seed_derivation = PREDICTION_SEED_DERIVATION
    ctx.set_case("case-001", 9)
    ctx.current_k_index = 2
    ctx.log_span("case_start")
    ctx.log_group_chat("message", source="Coder", content="answer")

    span = json.loads((tmp_path / "spans.jsonl").read_text().strip())
    message = json.loads((tmp_path / "group_chat.jsonl").read_text().strip())
    for record in (span, message):
        assert record["base_seed"] == 42
        assert record["prediction_seed"] == 123456
        assert record["seed_derivation"] == PREDICTION_SEED_DERIVATION


def test_backend_call_links_autogen_cancellation_token():
    from autogen_core import CancellationToken
    from lychee_mas.runtime.backends.autogen_injection_client import _call_backend

    class Backend:
        runs_in_worker_thread = True
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_cancellation_token = True

        def generate_chat(self, messages, *, cancellation_token=None):
            while not cancellation_token.is_cancelled():
                time.sleep(0.005)
            return "stopped"

    async def cancel_call():
        token = CancellationToken()
        task = asyncio.create_task(
            _call_backend(
                Backend(),
                "generate_chat",
                [],
                cancellation_token=token,
            )
        )
        await asyncio.sleep(0.02)
        token.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(cancel_call()) is True


def test_shared_span_logger_keeps_worker_case_context_isolated(tmp_path):
    from lychee_mas.runtime.spans import JsonlSpanLogger

    logger = JsonlSpanLogger(str(tmp_path / "spans.jsonl"))
    contexts = [
        RoutingContext(
            task="gsm8k",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            span_logger=logger,
        )
        for _ in range(2)
    ]
    for index, ctx in enumerate(contexts):
        ctx.set_case(f"case-{index}", index)
        ctx.log_span("case_start")
    logger.log("run_end")

    rows = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in rows[:2]] == ["case-0", "case-1"]
    assert rows[-1]["case_id"] is None


def test_shared_group_chat_logger_keeps_worker_case_context_isolated(tmp_path):
    from lychee_mas.runtime.spans import JsonlGroupChatLogger

    logger = JsonlGroupChatLogger(str(tmp_path / "group_chat.jsonl"))
    contexts = [
        RoutingContext(
            task="gaia_validation",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            group_chat_logger=logger,
        )
        for _ in range(2)
    ]
    for index, ctx in enumerate(contexts):
        ctx.set_case(f"case-{index}", index)
        ctx.current_k_index = index + 2
        ctx.current_attempt = 1
        ctx.log_group_chat("message", source="Coder", content=f"answer-{index}")
    logger.log("run_marker")

    rows = [json.loads(line) for line in (tmp_path / "group_chat.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in rows[:2]] == ["case-0", "case-1"]
    assert [row["k_index"] for row in rows[:2]] == [2, 3]
    assert rows[-1]["case_id"] is None


def test_jsonl_loggers_continue_sequence_when_resuming(tmp_path):
    from lychee_mas.runtime.spans import JsonlGroupChatLogger, JsonlSpanLogger

    for logger_type, filename, event_name in (
        (JsonlSpanLogger, "spans.jsonl", "run_start"),
        (JsonlGroupChatLogger, "group_chat.jsonl", "group_chat_start"),
    ):
        path = tmp_path / filename
        logger_type(str(path)).log(event_name, case_id="case-0", attempt=0)
        logger_type(str(path), append=True).log(event_name, case_id="case-0", attempt=1)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["seq"] for row in rows] == [1, 2]
        assert [row["attempt"] for row in rows] == [0, 1]


def test_run_status_json_is_replaced_atomically(tmp_path):
    namespace = runpy.run_path("scripts/run_mas.py")
    target = tmp_path / "run_status.json"
    namespace["_atomic_write_json"](str(target), {"status": "running"})
    namespace["_atomic_write_json"](str(target), {"status": "completed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "completed"}
    assert list(tmp_path.glob("run_status.json.tmp.*")) == []


def test_run_mas_resolves_prefix_length_from_current_and_legacy_configs():
    namespace = runpy.run_path("scripts/run_mas.py")
    resolve = namespace["_resolve_prefix_length"]

    assert resolve({"router": {"P": 0}, "memory": {"P": 16}}, None) == 0
    assert resolve({"memory": {"P": 12}}, None) == 12
    assert resolve({"router": {"P": 4}, "memory": {"P": 12}}, 8) == 8


def test_openai_backend_preserves_native_tool_protocol():
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend

    captured = {}

    class _Slot:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Limiter:
        def slot(self):
            return _Slot()

    function = SimpleNamespace(name="visit_url", arguments='{"url":"https://example.com"}')
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call_1", function=function)],
        reasoning="I should call the URL tool.",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=7,
            prompt_cache_hit_tokens=4,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )

    def create(**payload):
        captured.update(payload)
        return response

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "model"
    backend.do_sample = True
    backend.temperature = 0.0
    backend.top_p = 1.0
    backend.seed = 0
    backend.extra_body = {
        "top_k": 10,
        "chat_template_kwargs": {"preserve_thinking": True},
    }
    backend.rate_limiter = _Limiter()
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = backend.generate_chat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "previous",
                        "type": "function",
                        "function": {"name": "visit_url", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "previous", "content": "done"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "visit_url",
                    "description": "Visit a URL",
                    "parameters": {"type": "object"},
                },
            }
        ],
        request_overrides={
            "temperature": 0.6,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        },
        json_output=True,
    )

    assert captured["messages"][0]["tool_calls"][0]["id"] == "previous"
    assert captured["messages"][1]["tool_call_id"] == "previous"
    assert captured["tools"][0]["function"]["name"] == "visit_url"
    assert captured["tool_choice"] == "auto"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.6
    assert captured["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {
            "preserve_thinking": True,
            "enable_thinking": True,
        },
    }
    assert result.tool_calls == [
        {
            "id": "call_1",
            "name": "visit_url",
            "arguments": '{"url":"https://example.com"}',
        }
    ]
    assert result.finish_reason == "tool_calls"
    assert result.reasoning_content == "I should call the URL tool."
    assert result.reasoning_tokens == 5
    assert result.answer_tokens == 2
    assert result.cached_input_tokens == 4
    assert result.cached_input_tokens_source == "usage.prompt_cache_hit_tokens"
    assert result.cached_input_tokens_status == "available"
    assert result.provider_message_content is None
    assert result.provider_reasoning_content == "I should call the URL tool."
    assert result.raw_decoded_text is None
    assert result.provider_request_payload["messages"][0]["tool_calls"][0]["id"] == "previous"
    assert result.provider_response_payload["choices"][0]["message"]["reasoning"] == (
        "I should call the URL tool."
    )
    assert result.response_payload_origin == "provider"


def test_cached_input_usage_supports_openai_and_responses_detail_shapes():
    from lychee_mas.runtime.backends.openai_api_backend import _cached_input_usage

    assert _cached_input_usage({"prompt_tokens_details": {"cached_tokens": 64}}) == (
        64,
        "usage.prompt_tokens_details.cached_tokens",
        "available",
    )
    assert _cached_input_usage({"input_tokens_details": {"cached_tokens": 32}}) == (
        32,
        "usage.input_tokens_details.cached_tokens",
        "available",
    )
    assert _cached_input_usage({"prompt_tokens": 10}) == (
        None,
        None,
        "not_reported",
    )


def test_local_hf_marks_cross_request_cache_reporting_as_unsupported():
    from lychee_mas.runtime.backends.hf_backend import GenResult

    result = GenResult(text="ok", n_prompt_pos=3, n_gen_tokens=1, latency_s=0.1)
    assert result.cached_input_tokens is None
    assert result.cached_input_tokens_source is None
    assert result.cached_input_tokens_status == "not_supported"


def test_openai_backend_maps_do_sample_false_to_provider_greedy_parameters():
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend

    captured = {}

    class _Slot:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="4", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "model"
    backend.do_sample = False
    backend.temperature = 0.7
    backend.top_p = 0.8
    backend.seed = 0
    backend.extra_body = {}
    backend.rate_limiter = SimpleNamespace(slot=lambda: _Slot())
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.update(payload) or response
            )
        )
    )

    result = backend.generate_chat([{"role": "user", "content": "2+2"}])

    assert result.text == "4"
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 1.0


def test_openai_backend_normalizes_vllm_per_request_metrics():
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend

    class _Slot:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="4", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        model_extra={
            "metrics": {
                "queue_time_ms": 12.5,
                "time_to_first_token_ms": 20,
                "generation_time_ms": 40,
                "mean_itl_ms": 5,
                "tokens_per_second": 25,
            }
        },
    )
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "model"
    backend.do_sample = False
    backend.temperature = None
    backend.top_p = None
    backend.seed = None
    backend.extra_body = {}
    backend.rate_limiter = SimpleNamespace(slot=lambda: _Slot())
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_payload: response))
    )

    result = backend.generate_chat([{"role": "user", "content": "2+2"}])

    assert result.provider_request_queue_latency_s == pytest.approx(0.0125)
    assert result.provider_scheduled_to_first_token_s == pytest.approx(0.02)
    assert result.provider_generation_latency_s == pytest.approx(0.04)
    assert result.provider_mean_inter_token_latency_s == pytest.approx(0.005)
    assert result.provider_output_tokens_per_second == pytest.approx(25)
    assert result.client_rate_limiter_wait_s is not None
    assert result.client_http_request_latency_s is not None
    assert result.client_response_postprocess_latency_s is not None
    assert result.client_model_call_wall_time_s >= result.client_http_request_latency_s


def test_vllm_prometheus_parser_keeps_selected_service_metrics():
    from lychee_mas.runtime.vllm_metrics import parse_prometheus_metrics

    parsed = parse_prometheus_metrics(
        """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen"} 2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen"} 3
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{model_name="Qwen",le="1.0"} 4
vllm:request_queue_time_seconds_count{model_name="Qwen"} 4
vllm:request_queue_time_seconds_sum{model_name="Qwen"} 1.5
# TYPE unrelated gauge
unrelated 99
"""
    )

    assert parsed["vllm:num_requests_running"][0]["value"] == 2
    assert parsed["vllm:num_requests_waiting"][0]["value"] == 3
    assert parsed["vllm:request_queue_time_seconds_count"][0]["value"] == 4
    assert "unrelated" not in parsed


def test_local_sandbox_source_fingerprint_and_labels(monkeypatch: pytest.MonkeyPatch):
    from lychee_mas.runtime import docker_sandbox

    spec = docker_sandbox.LOCAL_SANDBOX_SPECS["agbench_gaia"]
    fingerprint = docker_sandbox.source_fingerprint(".", spec)
    inspected = [
        {
            "Id": "sha256:image",
            "Config": {
                "Labels": {
                    docker_sandbox.PROFILE_LABEL: "agbench_gaia",
                    docker_sandbox.SCHEMA_LABEL: docker_sandbox.SANDBOX_SCHEMA_VERSION,
                    docker_sandbox.FINGERPRINT_LABEL: fingerprint,
                }
            },
        }
    ]

    monkeypatch.setattr(
        docker_sandbox.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(inspected),
            stderr="",
        ),
    )

    report = docker_sandbox.verify_local_sandbox(
        "lychee-agbench-gaia:local",
        repo_root=".",
        run_capability_check=False,
    )
    assert report["profile"] == "agbench_gaia"
    assert report["fingerprint"] == fingerprint


def test_local_sandbox_rejects_stale_fingerprint(monkeypatch: pytest.MonkeyPatch):
    from lychee_mas.runtime import docker_sandbox

    inspected = [
        {
            "Id": "sha256:stale",
            "Config": {
                "Labels": {
                    docker_sandbox.PROFILE_LABEL: "agbench_gaia",
                    docker_sandbox.SCHEMA_LABEL: docker_sandbox.SANDBOX_SCHEMA_VERSION,
                    docker_sandbox.FINGERPRINT_LABEL: "old",
                }
            },
        }
    ]
    monkeypatch.setattr(
        docker_sandbox.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(inspected),
            stderr="",
        ),
    )

    with pytest.raises(docker_sandbox.SandboxVerificationError, match="stale"):
        docker_sandbox.verify_local_sandbox(
            "lychee-agbench-gaia:local",
            repo_root=".",
            run_capability_check=False,
        )


def test_gaia_sandbox_profile_uses_only_agbench_layers():
    from lychee_mas.runtime import docker_sandbox
    from scripts.prepare_benchmarks import _docker_targets_for_prepare_targets

    base = docker_sandbox.LOCAL_SANDBOX_SPECS["agbench_base"]
    aligned = docker_sandbox.LOCAL_SANDBOX_SPECS["agbench_gaia"]

    assert base["image"] == "lychee-agbench-base:local"
    assert base["deps"] == []
    assert aligned["image"] == "lychee-agbench-gaia:local"
    assert aligned["deps"] == ["agbench_base"]
    assert docker_sandbox.DOCKER_BUILD_ORDER.index("agbench_base") < (
        docker_sandbox.DOCKER_BUILD_ORDER.index("agbench_gaia")
    )
    assert _docker_targets_for_prepare_targets(["gaia_validation"], {}) == {"agbench_gaia"}


def test_agbench_base_and_gaia_requirements_are_auditable():
    def requirement_lines(path: Path) -> set[str]:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    official_gaia = requirement_lines(
        Path(
            "src/autogen/python/packages/agbench/benchmarks/GAIA/"
            "Templates/MagenticOne/requirements.txt"
        )
    )
    project_gaia = requirement_lines(Path("docker/agbench-gaia.requirements.txt"))
    assert project_gaia == official_gaia

    base_capabilities = json.loads(
        Path("docker/agbench-base-capabilities.json").read_text(encoding="utf-8")
    )
    gaia_capabilities = json.loads(
        Path("docker/agbench-gaia-capabilities.json").read_text(encoding="utf-8")
    )
    assert gaia_capabilities["extends"] == "agbench_base"
    assert set(base_capabilities["commands"]) <= set(gaia_capabilities["commands"])
    assert set(base_capabilities["python_modules"]) <= set(gaia_capabilities["python_modules"])


def test_openai_backend_adapts_timeout_to_output_budget_and_observed_speed():
    from lychee_mas.runtime.backends.openai_api_backend import (
        OpenAICompatibleBackend,
        _normalize_timeout_policy,
    )

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.timeout = 120.0
    backend.max_retries = 0
    backend.request_timeout = _normalize_timeout_policy(
        {
            "mode": "adaptive",
            "maximum_s": 1800,
            "initial_generation_tokens_per_second": 20,
            "observed_tokens_per_second_ceiling": 40,
            "safety_factor": 1.5,
            "base_s": 30,
        },
        minimum_timeout_s=120,
    )
    backend._throughput_lock = threading.Lock()
    backend._observed_generation_tps = None

    initial = backend.request_timeout_plan(16384)
    assert initial["timeout_s"] == pytest.approx(1258.8)
    assert initial["planning_generation_tokens_per_second"] == 20
    assert initial["max_retries"] == 0

    backend._observe_generation_throughput(4800, 100)
    observed = backend.request_timeout_plan(16384)
    assert observed["observed_generation_tokens_per_second"] == 48
    assert observed["planning_generation_tokens_per_second"] == 40
    assert observed["timeout_s"] == pytest.approx(644.4)


def test_openai_backend_prewarms_tokenizer_for_client_retokenization(
    monkeypatch: pytest.MonkeyPatch,
):
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text in {"tokenizer warmup", "answer"}
            assert add_special_tokens is False
            return [1, 2, 3]

    fake_auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: (
            FakeTokenizer()
            if path == "/models/Qwen3.6-27B" and kwargs["trust_remote_code"]
            else None
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=fake_auto_tokenizer),
    )
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.tok = None
    backend.tokenizer_path = "/models/Qwen3.6-27B"
    backend.reasoning_token_accounting = "client_retokenized_required"
    backend._tokenizer_lock = threading.Lock()
    backend._tokenizer_load_attempted = False
    backend._tokenizer_load_error = None
    backend._tokenizer_load_latency_s = None
    backend._tokenizer_warmup_report = {}

    report = backend.warmup_tokenizer()

    assert report["tokenizer_warmup_status"] == "ready"
    assert report["tokenizer_warmup_tokens"] == 3
    assert report["tokenizer_class"] == "FakeTokenizer"
    assert backend._retokenize("answer") == 3


def test_openai_backend_does_not_warm_when_provider_reports_reasoning_usage():
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.tokenizer_path = "/models/Qwen3.6-27B"
    backend.reasoning_token_accounting = "provider_usage"
    backend._tokenizer_warmup_report = {}

    report = backend.warmup_tokenizer()

    assert report["tokenizer_warmup_status"] == "not_required"


def test_openai_backend_supports_structured_output_and_extra_create_args():
    from lychee_mas.runtime.backends.openai_api_backend import OpenAICompatibleBackend
    from pydantic import BaseModel

    class Verdict(BaseModel):
        accepted: bool

    class _Slot:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"accepted":true}', tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3),
    )
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "model"
    backend.temperature = None
    backend.top_p = None
    backend.seed = None
    backend.extra_body = {}
    backend.rate_limiter = SimpleNamespace(slot=lambda: _Slot())
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.update(payload) or response
            )
        )
    )

    backend.generate_chat(
        [{"role": "user", "content": "judge"}],
        json_output=Verdict,
        extra_create_args={"frequency_penalty": 0.2},
    )

    assert captured["frequency_penalty"] == 0.2
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["schema"]["required"] == ["accepted"]


def test_openai_backend_maps_canonical_image_parts():
    from lychee_mas.runtime.backends.openai_api_backend import _to_openai_content

    data_uri = "data:image/png;base64,AAAA"
    converted = _to_openai_content(
        [
            {"type": "text", "text": "What is shown?"},
            {"type": "image", "url": data_uri},
        ]
    )
    assert converted == [
        {"type": "text", "text": "What is shown?"},
        {
            "type": "image_url",
            "image_url": {"url": data_uri, "detail": "auto"},
        },
    ]


def test_autogen_chat_conversion_preserves_reasoning_as_reasoning_content():
    from autogen_core import FunctionCall
    from autogen_core.models import AssistantMessage
    from lychee_mas.runtime.backends.autogen_injection_client import _to_chat

    text_message = AssistantMessage(
        content="Final answer",
        thought="Private reasoning",
        source="Solver",
    )
    tool_message = AssistantMessage(
        content=[FunctionCall(id="call-1", name="lookup", arguments='{"q":"x"}')],
        thought="Need a lookup",
        source="Researcher",
    )

    converted = _to_chat([text_message, tool_message])

    assert converted[0]["content"] == "Final answer"
    assert converted[0]["reasoning_content"] == "Private reasoning"
    assert converted[1]["content"] is None
    assert converted[1]["reasoning_content"] == "Need a lookup"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "lookup"


def test_autogen_chat_conversion_preserves_multimodal_content_parts():
    from autogen_core import Image
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import _to_chat
    from PIL import Image as PILImage

    image = Image.from_pil(PILImage.new("RGB", (3, 2), color="white"))
    converted = _to_chat([UserMessage(content=["Read this image", image], source="user")])

    assert converted[0]["content"][0] == {"type": "text", "text": "Read this image"}
    image_part = converted[0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_qwen_reasoning_output_is_split_without_changing_plain_output():
    from lychee_mas.runtime.backends.hf_backend import _split_reasoning_output

    reasoning, final = _split_reasoning_output("<think>\nwork\n</think>\n\n42")
    assert reasoning == "work"
    assert final == "42"
    assert _split_reasoning_output("plain answer") == ("", "plain answer")
    assert _split_reasoning_output("<think>unfinished") == ("unfinished", "")


def test_hf_multimodal_mapping_keeps_image_order():
    from autogen_core import Image
    from lychee_mas.runtime.backends.hf_backend import _to_hf_multimodal_messages
    from PIL import Image as PILImage

    image = Image.from_pil(PILImage.new("RGB", (4, 3), color="black"))
    messages, images = _to_hf_multimodal_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    image.to_openai_format(),
                    {"type": "text", "text": "Last"},
                ],
            }
        ]
    )
    assert messages[0]["content"] == [
        {"type": "text", "text": "First"},
        {"type": "image"},
        {"type": "text", "text": "Last"},
    ]
    assert len(images) == 1
    assert images[0].size == (4, 3)


def test_selector_control_call_records_span_and_returns_autogen_thought(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_plain_client
    from lychee_mas.runtime.spans import JsonlSpanLogger

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = True
        model_info = {}
        tok = None

        def generate_chat(self, messages, max_new_tokens=64, request_overrides=None):
            assert request_overrides == {"extra_body": {"thinking_token_budget": 4}}
            return SimpleNamespace(
                text="Solver",
                reasoning_content="The solver should speak next.",
                provider_message_content=None,
                provider_reasoning_content=None,
                raw_decoded_text="<think>The solver should speak next.</think>Solver",
                n_prompt_pos=10,
                n_gen_tokens=6,
                reasoning_tokens=4,
                answer_tokens=2,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={"answer": "Solver"},
            )

    logger = JsonlSpanLogger(str(tmp_path / "spans.jsonl"))
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=logger,
    )
    ctx.trace_detail_level = "full"
    ctx.set_case("case-1", 0)
    client = make_plain_client(
        Backend(),
        ctx=ctx,
        max_new_tokens=64,
        request_overrides={"extra_body": {"thinking_token_budget": 4}},
    )
    result = asyncio.run(client.create([UserMessage(content="Who should speak?", source="user")]))

    assert result.content == "Solver"
    assert result.thought == "The solver should speak next."
    assert ctx.decisions[0]["controller"] is True
    spans = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert [span["span_type"] for span in spans] == ["model_call_start", "model_call_end"]
    assert spans[0]["autogen_model_messages"][0]["content"] == "Who should speak?"
    assert spans[0]["role_visible_messages"] == spans[0]["autogen_model_messages"]
    assert spans[0]["backend_messages"] == spans[0]["autogen_model_messages"]
    assert spans[0]["thinking_budget_requested"] == 4
    assert spans[0]["thinking_budget_parameter"] == "thinking_token_budget"
    assert spans[0]["thinking_budget_enforced_by"] == "vllm"
    assert spans[-1]["provider_response_payload"] == {"answer": "Solver"}
    assert spans[-1]["output_total_tokens"] == 6
    assert spans[-1]["output_reasoning_tokens"] == 4
    assert spans[-1]["output_answer_tokens"] == 2
    assert "parsed_output" not in spans[-1]
    assert "raw_decoded_text" not in spans[-1]


def test_model_client_usage_is_cumulative_but_create_result_usage_is_per_call():
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_plain_client

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = False
        supports_extra_create_args = False
        model_info = {}
        tok = None

        def __init__(self):
            self.calls = 0

        def generate_chat(self, messages, max_new_tokens=64):
            self.calls += 1
            return SimpleNamespace(
                text="Solver",
                reasoning_content="",
                n_prompt_pos=self.calls,
                n_gen_tokens=2,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={},
                provider_response_payload={},
            )

    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    client = make_plain_client(Backend(), ctx=ctx, max_new_tokens=64)

    async def call_twice():
        first = await client.create([UserMessage(content="one", source="user")])
        second = await client.create([UserMessage(content="two", source="user")])
        return first, second

    first, second = asyncio.run(call_twice())
    assert (first.usage.prompt_tokens, first.usage.completion_tokens) == (1, 2)
    assert (second.usage.prompt_tokens, second.usage.completion_tokens) == (2, 2)
    assert (client.actual_usage().prompt_tokens, client.actual_usage().completion_tokens) == (3, 4)
    assert client.total_usage() == client.actual_usage()


def test_group_chat_rejects_fake_model_streaming():
    from lychee_mas.runtime.group_chat import normalize_group_chat_config

    try:
        normalize_group_chat_config({"type": "selector", "model_client_streaming": True})
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("streaming=true must fail validation")


def test_injection_client_does_not_rewrite_magentic_one_ledger(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_injection_client
    from lychee_mas.runtime.spans import JsonlSpanLogger

    shorthand = '{"reason":"Search the web next.","answer":false}'

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = False
        supports_native_tools = False
        supports_json_output = True
        model_info = {}
        tok = None

        def generate_chat(self, messages, max_new_tokens=64, json_output=None):
            assert json_output is True
            return SimpleNamespace(
                text=shorthand,
                reasoning_content="",
                provider_message_content=shorthand,
                provider_reasoning_content=None,
                raw_decoded_text=None,
                n_prompt_pos=12,
                n_gen_tokens=8,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                prefix_len=0,
                provider_request_payload={"messages": messages},
                provider_response_payload={"content": shorthand},
            )

    logger = JsonlSpanLogger(str(tmp_path / "spans.jsonl"))
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=logger,
    )
    ctx.set_case("case-ledger", 0)
    client = make_injection_client(Backend(), "Orchestrator", ctx, max_new_tokens=64)
    result = asyncio.run(
        client.create(
            [UserMessage(content="Return the progress ledger.", source="user")],
            json_output=True,
        )
    )

    assert result.content == shorthand
    spans = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert spans[-1]["provider_response_payload"] == {"content": shorthand}
    assert "parsed_final_content" not in spans[-1]
    assert "json_output_repaired" not in spans[-1]
    assert "json_output_normalized" not in spans[-1]


def test_compact_model_call_start_omits_only_message_triplet(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_plain_client
    from lychee_mas.runtime.spans import JsonlSpanLogger

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = False
        model_info = {}
        tok = None

        def generate_chat(self, messages, max_new_tokens=64):
            return SimpleNamespace(
                text="Solver",
                reasoning_content="",
                n_prompt_pos=3,
                n_gen_tokens=1,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={"answer": "Solver"},
            )

    logger = JsonlSpanLogger(str(tmp_path / "spans.jsonl"))
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=logger,
    )
    ctx.trace_detail_level = "compact"
    ctx.set_case("case-compact", 0)
    client = make_plain_client(Backend(), ctx=ctx, max_new_tokens=64)
    asyncio.run(client.create([UserMessage(content="Pick a speaker", source="user")]))

    start, end = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert start["trace_detail_level"] == "compact"
    for key in ("autogen_model_messages", "role_visible_messages", "backend_messages"):
        assert key not in start
    assert end["provider_request_payload"]["messages"][0]["content"] == "Pick a speaker"
    assert end["provider_response_payload"] == {"answer": "Solver"}


def test_compact_and_full_span_schemas_differ_only_at_model_call_start_triplet(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.backends.autogen_injection_client import make_plain_client
    from lychee_mas.runtime.spans import JsonlSpanLogger

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = False
        model_info = {}
        tok = None

        def generate_chat(self, messages, max_new_tokens=64):
            return SimpleNamespace(
                text="Solver",
                reasoning_content="",
                n_prompt_pos=3,
                n_gen_tokens=1,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={"answer": "Solver"},
            )

    def run_mode(mode: str):
        path = tmp_path / mode / "spans.jsonl"
        logger = JsonlSpanLogger(str(path), run_fields={"trace_detail_level": mode})
        ctx = RoutingContext(
            task="gsm8k",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            span_logger=logger,
        )
        ctx.trace_detail_level = mode
        ctx.set_case("same-case", 0)
        client = make_plain_client(Backend(), ctx=ctx, max_new_tokens=64)
        asyncio.run(client.create([UserMessage(content="Pick a speaker", source="user")]))
        return [json.loads(line) for line in path.read_text().splitlines()]

    compact, full = run_mode("compact"), run_mode("full")
    triplet = {"autogen_model_messages", "role_visible_messages", "backend_messages"}
    assert set(full[0]) - set(compact[0]) == triplet
    assert set(compact[0]) - set(full[0]) == set()
    assert set(compact[1]) == set(full[1])
    assert compact[1]["provider_request_payload"] == full[1]["provider_request_payload"]
    assert compact[1]["provider_response_payload"] == full[1]["provider_response_payload"]


def test_group_chat_file_keeps_full_event_while_span_keeps_summary(tmp_path):
    from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime
    from lychee_mas.runtime.spans import JsonlGroupChatLogger, JsonlSpanLogger

    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        span_logger=JsonlSpanLogger(str(tmp_path / "spans.jsonl")),
        group_chat_logger=JsonlGroupChatLogger(str(tmp_path / "group_chat.jsonl")),
    )
    ctx.set_case("case-chat", 7)
    ctx.current_k_index = 2
    ctx.current_attempt = 1
    runtime = AutoGenRuntime(backend=object(), ctx=ctx)
    runtime._log_autogen_event(
        SimpleNamespace(
            source="Coder",
            type="TextMessage",
            content="full group-chat message",
            models_usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            metadata={"tag": "official-event"},
        ),
        3,
    )

    group_event = json.loads((tmp_path / "group_chat.jsonl").read_text().strip())
    span = json.loads((tmp_path / "spans.jsonl").read_text().strip())
    assert group_event["content"] == "full group-chat message"
    assert group_event["metadata"] == {"tag": "official-event"}
    assert group_event["k_index"] == 2
    assert group_event["attempt"] == 1
    assert span["span_type"] == "autogen_message"
    assert span["prompt_tokens"] == 11
    assert "content" not in span


def test_terminal_source_policy_matches_official_examples():
    from lychee_mas.layers.construct.templates import team_to_agentspecs

    gaia_agents = team_to_agentspecs("gaia")
    human_terminal = next(
        node for node in team_to_agentspecs("human_eval") if node.name == "ComputerTerminal"
    )
    gaia_terminal = next(node for node in gaia_agents if node.name == "ComputerTerminal")
    assert [node.name for node in gaia_agents] == [
        "Coder",
        "ComputerTerminal",
        "FileSurfer",
        "WebSurfer",
    ]
    assert human_terminal.meta["sources"] == ["Coder"]
    assert "sources" not in gaia_terminal.meta


def test_answer_extractor_ignores_autogen_thought_events():
    from lychee_mas.eval.task_config import extractor_for_task

    messages = [
        SimpleNamespace(
            type="TextMessage",
            content="Solution complete.\nAPPROVE: \\boxed{70000}",
        ),
        SimpleNamespace(
            type="ThoughtEvent",
            content="APPROVE: 70000\nThis private reasoning must not become the answer.",
        ),
    ]

    assert extractor_for_task("gsm8k")(messages) == r"\boxed{70000}"
