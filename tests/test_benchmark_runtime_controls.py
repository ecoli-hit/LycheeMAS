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
from lychee_mas.memory.context import (
    ModelCallBudgetExceeded,
    RoutingContext,
    TrialDeadlineExceeded,
)
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.events.store import run_events_path
from lychee_mas.runtime.execution.concurrency import (
    TrialConcurrencyController,
    assess_vllm_pressure,
    normalize_concurrency_policy,
    normalize_deployment_admission_policy,
)
from lychee_mas.runtime.execution.seeding import (
    TRIAL_SEED_DERIVATION,
    derive_trial_seed,
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


def _team_graph(team_id: str, *, rounds: int = 1):
    from lychee_mas.eval.teams.compiler import load_team_spec

    path = (
        Path(__file__).resolve().parents[1] / "configs/eval_studio/teams/specs" / f"{team_id}.json"
    )
    return load_team_spec(path, rounds=rounds)[0]


def test_contamination_audit_separates_official_filename_risk_from_solution_hit(tmp_path):
    from lychee_mas.eval.evaluation.contamination import audit_run
    from lychee_mas.runtime.events.store import RunEventWriter

    case_id = "32102e3e-d12a-4209-9163-7b3a104efe5d"
    events = [
        {
            "seq": 1,
            "event_type": "workspace.prepared",
            "case_id": case_id,
            "trial_index": 0,
            "workspace": str(tmp_path / "ws_0123456789abcdef0123456789abcdef"),
            "workspace_is_new": True,
            "preexisting_entry_count": 0,
            "workspace_policy": "uuid4_isolated_per_attempt_v1",
            "attachment_filename_policy": "preserve_official_filename",
            "visible_attachment_names": [f"{case_id}.xlsx"],
        }
    ]
    messages = [
        {
            "seq": 2,
            "event_type": "agent.message.published",
            "case_id": case_id,
            "trial_index": 0,
            "source": "WebSurfer",
            "content": (
                "I opened https://example.test/gaia/solutions/"
                f"{case_id}/answer and found FINAL ANSWER: 42"
            ),
        }
    ]
    writer = RunEventWriter(run_events_path(tmp_path))
    for record in events:
        value = dict(record)
        writer.log_event(value.pop("event_type"), **value)
    for record in messages:
        value = dict(record)
        writer.log_event(value.pop("event_type"), **value)

    audits, summary = audit_run(
        tmp_path,
        trials=[{"case_id": case_id, "trial_index": 0, "score": 1.0}],
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
    from lychee_mas.eval.evaluation.contamination import audit_run
    from lychee_mas.runtime.events.store import RunEventWriter

    case_id = "c61d22de-5f6c-4958-a7f6-5e9707bd3466"
    legacy = str(tmp_path / case_id)
    events = [
        {
            "seq": seq,
            "event_type": "workspace.prepared",
            "case_id": case_id,
            "trial_index": 0,
            "workspace": legacy,
            "copied_files": [],
        }
        for seq in (1, 2)
    ]
    writer = RunEventWriter(run_events_path(tmp_path))
    for record in events:
        value = dict(record)
        writer.log_event(value.pop("event_type"), **value)
    writer.log_event(
        "agent.message.published",
        case_id=case_id,
        trial_index=0,
        source="WebSurfer",
        content="The ordinary page contains the number 42 in an unrelated table.",
    )

    audits, summary = audit_run(
        tmp_path,
        trials=[{"case_id": case_id, "trial_index": 0, "score": 0.0}],
        gold_by_id={case_id: {"gold": "42"}},
    )

    assert "workspace_case_id_exposed" in audits[0]["contamination_flags"]
    assert "workspace_reused" in audits[0]["contamination_flags"]
    assert "gold_like_text_in_web_evidence" not in audits[0]["contamination_flags"]
    assert summary["flag_case_counts"]["workspace_reused"] == 1


def test_contamination_audit_does_not_treat_current_run_workspace_as_prior_run(tmp_path):
    from lychee_mas.eval.evaluation.contamination import audit_run
    from lychee_mas.runtime.events.store import RunEventWriter, run_events_path

    case_id = "case-current-workspace"
    workspace = tmp_path / "runs" / "lychee_tool_workspaces" / "ws_current"
    writer = RunEventWriter(run_events_path(tmp_path))
    writer.log_event(
        "workspace.prepared",
        case_id=case_id,
        trial_index=0,
        workspace=str(workspace),
        workspace_is_new=True,
    )
    writer.log_event(
        "tool_execution.failed",
        case_id=case_id,
        trial_index=0,
        tool_name="read_text_file",
        arguments={"path": str(workspace / "missing.txt")},
        output=f"No such file: {workspace / 'missing.txt'}",
        is_error=True,
    )

    audits, summary = audit_run(
        tmp_path,
        trials=[{"case_id": case_id, "trial_index": 0, "score": 0.0}],
        gold_by_id={case_id: {"gold": "answer"}},
    )

    assert "prior_run_artifact_access" not in audits[0]["contamination_flags"]
    assert summary["flag_case_counts"].get("prior_run_artifact_access", 0) == 0


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


def test_routing_context_rejects_model_call_after_trial_deadline():
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        case_deadline_monotonic_s=0.0,
    )

    with pytest.raises(TrialDeadlineExceeded, match="Trial deadline reached"):
        ctx.reserve_model_call("Researcher")

    assert ctx.model_calls_started == 0
    assert ctx.case_deadline_exceeded is True


def test_adaptive_trial_concurrency_uses_aimd_without_changing_service_capacity():
    policy = normalize_concurrency_policy(
        {
            "mode": "auto",
            "initial": 4,
            "minimum": 2,
            "maximum": 8,
            "increase_step": 2,
            "decrease_factor": 0.5,
            "control_window_trials": 2,
        },
        configured_concurrency=4,
    )
    controller = TrialConcurrencyController(policy)

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


def test_trial_concurrency_combines_local_aimd_with_scheduler_allocation():
    policy = normalize_concurrency_policy(
        {
            "mode": "auto",
            "initial": 8,
            "minimum": 2,
            "maximum": 16,
            "control_window_trials": 4,
        },
        configured_concurrency=8,
    )
    controller = TrialConcurrencyController(policy, external_limit=3)

    async def exercise():
        assert controller.target == 8
        assert controller.effective_target == 3
        increased = await controller.set_external_limit(6)
        assert increased["previous"] == 3
        assert increased["target"] == 6
        assert increased["adaptive_target"] == 8
        reduced = await controller.set_external_limit(2)
        assert reduced["target"] == 2
        assert controller.target == 8

    asyncio.run(exercise())


def test_trial_concurrency_zero_allocation_pauses_new_trials_without_resetting_aimd():
    policy = normalize_concurrency_policy(
        {
            "mode": "auto",
            "initial": 8,
            "minimum": 2,
            "maximum": 16,
            "control_window_trials": 4,
        },
        configured_concurrency=8,
    )
    controller = TrialConcurrencyController(policy, external_limit=4)

    async def exercise():
        paused = await controller.set_external_limit(0)
        assert paused["previous"] == 4
        assert paused["target"] == 0
        assert controller.target == 8
        assert controller.effective_target == 0

        acquire = asyncio.create_task(controller.acquire())
        await asyncio.sleep(0)
        assert not acquire.done()

        resumed = await controller.set_external_limit(3)
        assert resumed["previous"] == 0
        assert resumed["target"] == 3
        await asyncio.wait_for(acquire, timeout=1)
        assert controller.active == 1
        await controller.release()

    asyncio.run(exercise())


def test_scheduler_authority_disables_duplicate_runner_aimd() -> None:
    policy = normalize_concurrency_policy(
        {
            "mode": "auto",
            "initial": 2,
            "minimum": 1,
            "maximum": 16,
            "control_window_trials": 1,
        },
        configured_concurrency=2,
    )
    controller = TrialConcurrencyController(
        policy,
        external_limit=6,
        external_authoritative=True,
    )

    async def exercise() -> None:
        assert controller.effective_target == 6
        assert (
            await controller.observe(record={"status": "error", "error_type": "APITimeoutError"})
            is None
        )
        assert controller.target == 2
        raised = await controller.set_external_limit(9)
        assert raised["previous"] == 6
        assert raised["target"] == 9
        assert raised["authority"] == "external"
        assert controller.effective_target == 9

    asyncio.run(exercise())


def test_deployment_admission_policy_and_vllm_pressure_are_explicit():
    policy = normalize_deployment_admission_policy(
        {
            "mode": "auto",
            "minimum": 8,
            "initial": 32,
            "maximum": 48,
            "increase_step": 2,
        },
        hard_capacity=48,
    )
    assert policy["initial"] == 32
    assert policy["maximum"] == 48

    healthy = assess_vllm_pressure(
        {
            "status": "ok",
            "requests_running": 24,
            "requests_waiting": 0,
            "gpu_cache_usage_fraction": 0.55,
            "queue_time_window_mean_s": 0.4,
            "time_to_first_token_window_mean_s": 1.2,
            "preemptions_total_window": 0,
        },
    )
    assert healthy["healthy"] is True
    assert healthy["overload"] is False

    overloaded = assess_vllm_pressure(
        {
            "status": "ok",
            "requests_running": 32,
            "requests_waiting": 18,
            "gpu_cache_usage_fraction": 0.93,
            "queue_time_window_mean_s": 6.0,
            "time_to_first_token_window_mean_s": 7.0,
            "preemptions_total_window": 1,
        },
    )
    assert overloaded["overload"] is True
    assert set(overloaded["reasons"]) >= {
        "preemption",
        "kv_cache",
        "waiting_queue",
        "queue_latency",
        "ttft",
    }


def test_costing_uses_deployment_pricing_snapshots_and_provider_token_details(
    tmp_path: Path,
):
    from lychee_mas.eval.pricing.costing import attach_cost_metrics

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
    from lychee_mas.runtime.events.store import RunEventWriter

    writer = RunEventWriter(run_events_path(tmp_path))
    operation_id = writer.log_event("model_call.started", deployment_instance_id="instance-qwen")
    writer.log_event(
        "model_call.completed",
        operation_id=operation_id,
        deployment_instance_id="instance-qwen",
        input_text_tokens=100,
        input_cached_tokens=40,
        output_text_tokens=50,
        output_reasoning_tokens=20,
    )

    costing = attach_cost_metrics({}, [], tmp_path)["costing"]

    actual = costing["actual_cost"]
    assert actual["totals_by_currency"] == {"CNY": 10.0}
    assert actual["line_items"][0]["allocated_gpu_hours"] == 2.0
    assert actual["line_items"][0]["billing_mode"] == "monthly_node_amortized"
    assert actual["line_items"][0]["quoted_node_month"] == 29200.0
    equivalent = costing["api_equivalent_cost"]
    assert equivalent["totals_by_currency"]["CNY"] == pytest.approx(0.0004)
    assert equivalent["line_items"][0]["usage"]["cached_input_tokens"] == 40
    assert equivalent["line_items"][0]["usage"]["reasoning_output_tokens"] == 20
    assert costing["model_call_source"] == "run_events.model_call.completed"

    (tmp_path / "vllm_metrics_summary.json").write_text(
        json.dumps(
            {
                "scope": "deployment_service",
                "targets": {
                    "instance-qwen": {"exclusive_run_attribution": False}
                },
            }
        ),
        encoding="utf-8",
    )
    shared_costing = attach_cost_metrics({}, [], tmp_path)["costing"]
    shared_actual = shared_costing["actual_cost"]
    assert shared_actual["status"] == "calculated_upper_bound"
    assert shared_actual["line_items"][0]["cost_is_upper_bound"] is True
    assert (
        shared_costing["comparison"]["status"]
        == "unavailable_due_to_shared_service_attribution"
    )

    writer = RunEventWriter(run_events_path(tmp_path))
    operation_id = writer.log_event("model_call.started", deployment_instance_id="instance-qwen")
    writer.log_event(
        "model_call.completed",
        operation_id=operation_id,
        deployment_instance_id="instance-qwen",
        input_text_tokens=200,
        input_cached_tokens=100,
        output_text_tokens=80,
        output_reasoning_tokens=30,
    )
    failed_operation_id = writer.log_event(
        "model_call.started", deployment_instance_id="instance-qwen"
    )
    writer.log_event(
        "model_call.failed",
        operation_id=failed_operation_id,
        deployment_instance_id="instance-qwen",
        error_type="APITimeoutError",
    )
    event_costing = attach_cost_metrics({}, [], tmp_path)["costing"]
    assert event_costing["model_call_source"] == "run_events.model_call.completed"
    assert event_costing["api_equivalent_cost"]["line_items"][0]["usage"]["input_tokens"] == 200
    assert event_costing["failed_model_calls_without_usage"] == 1
    assert event_costing["api_equivalent_cost"]["status"] == "calculated_lower_bound"
    assert (
        event_costing["api_equivalent_cost"]["line_items"][0]["failed_model_calls_without_usage"]
        == 1
    )


def test_costing_preserves_zero_optional_rates_and_resume_elapsed_segments(
    tmp_path: Path,
) -> None:
    from lychee_mas.eval.pricing.costing import _run_elapsed_seconds, _token_cost

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
    from lychee_mas.eval.pricing.costing import _token_cost, _token_usage

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
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import _effective_max_turns

    assert _effective_max_turns("round_robin", 3, max_rounds=4, max_turns=None) == 12
    assert _effective_max_turns("selector", 3, max_rounds=99, max_turns=17) == 17
    assert _effective_max_turns("magentic_one", 4, max_rounds=99, max_turns=None) == 20


def test_model_call_budget_exception_becomes_group_chat_stop_result():
    from autogen_agentchat.messages import TextMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        max_model_calls_per_case=1,
    )

    class GroupChat:
        async def run_stream(self, *, task, cancellation_token):
            assert cancellation_token.is_cancelled() is False
            yield TextMessage(content=f"partial: {task}", source="Coder")
            ctx.model_call_budget_exhausted = True
            raise RuntimeError("wrapped ModelCallBudgetExceeded")

    graph = _team_graph("single")
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


def test_autogen_group_chat_cancellation_propagates_to_official_token():
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    received = {}

    class GroupChat:
        async def run_stream(self, *, task, cancellation_token):
            received["task"] = task
            received["token"] = cancellation_token
            pending = asyncio.get_running_loop().create_future()
            cancellation_token.link_future(pending)
            await pending
            if False:
                yield None

    graph = _team_graph("single")
    runtime = AutoGenRuntime(backend=object(), ctx=ctx)

    async def cancel_stream():
        task = asyncio.create_task(runtime._run_group_chat_stream(GroupChat(), "question", graph))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_stream())

    assert received["task"] == "question"
    assert received["token"].is_cancelled() is True


def test_observed_code_executor_blocks_consecutive_identical_code():
    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock, CodeResult
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        _ObservedCodeExecutor,
    )

    class Wrapped:
        def __init__(self):
            self.calls = 0

        async def execute_code_blocks(self, code_blocks, cancellation_token):
            self.calls += 1
            return CodeResult(exit_code=0, output="first result")

    class Context:
        def __init__(self):
            self.events = []

        def log_event(self, event_type, **payload):
            self.events.append((event_type, payload))
            return f"event-{len(self.events)}"

    async def execute_twice():
        wrapped = Wrapped()
        context = Context()
        records = []
        executor = _ObservedCodeExecutor(wrapped, context, records)
        blocks = [CodeBlock(code="print('same')", language="python")]
        first = await executor.execute_code_blocks(blocks, CancellationToken())
        second = await executor.execute_code_blocks(blocks, CancellationToken())
        return wrapped, records, first, second

    wrapped, records, first, second = asyncio.run(execute_twice())

    assert wrapped.calls == 1
    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Blocked a consecutive identical code execution" in second.output
    assert records[-1]["repeated_identical_call_blocked"] is True


def test_observed_code_executor_bounds_the_result_delivered_to_group_chat():
    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock, CodeResult
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        _ObservedCodeExecutor,
    )

    raw_output = "large-line\n" * 10_000

    class Wrapped:
        async def execute_code_blocks(self, code_blocks, cancellation_token):
            return CodeResult(exit_code=0, output=raw_output)

    class Context:
        def __init__(self):
            self.events = []

        def log_event(self, event_type, **payload):
            self.events.append((event_type, payload))
            return f"event-{len(self.events)}"

    async def execute_once():
        context = Context()
        records = []
        executor = _ObservedCodeExecutor(
            Wrapped(),
            context,
            records,
            max_inline_tool_result_chars=2000,
        )
        result = await executor.execute_code_blocks(
            [CodeBlock(code="print('large')", language="python")],
            CancellationToken(),
        )
        return context, records, result

    context, records, result = asyncio.run(execute_once())

    assert len(result.output) <= 2000
    assert "tool output truncated for model context" in result.output
    assert records[0]["output"] == raw_output
    assert records[0]["delivered_output"] == result.output
    assert records[0]["output_truncated_for_model_context"] is True
    assert context.events[-1][1]["output"] == raw_output


def test_autogen_cleanup_has_a_hard_grace_period():
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    class Context:
        def __init__(self):
            self.events = []

        def log_event(self, event_type, **payload):
            self.events.append((event_type, payload))

    async def never_finishes():
        await asyncio.Event().wait()

    async def run_cleanup():
        context = Context()
        runtime = AutoGenRuntime(backend=object(), ctx=context)
        result = await runtime._run_cleanup_steps(
            [("stuck", never_finishes)],
            operation_id="runtime-1",
            timeout_s=0.01,
        )
        await asyncio.sleep(0)
        return context, result

    context, result = asyncio.run(run_cleanup())

    assert result == {"stuck": False}
    assert context.events[-1][0] == "runtime.cleanup_timed_out"


def test_autogen_cleanup_cancels_child_tasks_when_parent_is_cancelled():
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    class Context:
        def log_event(self, _event_type, **_payload):
            return None

    async def run_cleanup():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def never_finishes():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        runtime = AutoGenRuntime(backend=object(), ctx=Context())
        cleanup = asyncio.create_task(
            runtime._run_cleanup_steps(
                [("stuck", never_finishes)],
                operation_id="runtime-1",
            )
        )
        await started.wait()
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        await asyncio.sleep(0)
        assert stopped.is_set()

    asyncio.run(run_cleanup())


def test_autogen_cleanup_treats_an_already_closed_browser_as_idempotent():
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    class Context:
        def __init__(self):
            self.events = []

        def log_event(self, event_type, **payload):
            self.events.append((event_type, payload))

    async def already_closed():
        raise RuntimeError("BrowserContext.close: Target page, context or browser has been closed")

    async def run_cleanup():
        context = Context()
        runtime = AutoGenRuntime(backend=object(), ctx=context)
        result = await runtime._run_cleanup_steps(
            [("agent:WebSurfer", already_closed)],
            operation_id="runtime-1",
        )
        return context, result

    context, result = asyncio.run(run_cleanup())

    assert result == {"agent:WebSurfer": True}
    assert not context.events


def test_autogen_runtime_guard_closes_scope_when_cancelled_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime

    class Context:
        def __init__(self):
            self.events = []
            self.current_runtime_event_id = None

        def log_event(self, event_type, **payload):
            self.events.append((event_type, payload))

    context = Context()
    runtime = AutoGenRuntime(backend=object(), ctx=context)

    async def cancel_during_cleanup(_team, _query):
        context.current_runtime_event_id = "runtime-1"
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime, "_run_impl", cancel_during_cleanup)

    async def run_once():
        with pytest.raises(asyncio.CancelledError):
            await runtime.run(object(), object())

    asyncio.run(run_once())

    assert context.events == [
        (
            "runtime.cancelled",
            {
                "operation_id": "runtime-1",
                "runtime_elapsed_s": pytest.approx(0.0, abs=0.05),
            },
        )
    ]
    assert context.current_runtime_event_id is None


def test_topology_visibility_uses_official_message_filter_agent():
    from autogen_agentchat.messages import TextMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        _apply_official_message_filter,
    )

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


def test_team_spec_declares_one_explicit_group_chat():
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import _group_chat_config

    graph = _team_graph("reason", rounds=2)
    assert graph.meta["team"] == "reason"
    assert _group_chat_config(graph) == {"type": "round_robin"}
    assert graph.meta["lifecycle"]["termination"] == {"condition": "result_submitted"}
    assert graph.meta["result_contract"]["conditions"] == [{"type": "result_submitted"}]
    assert graph.meta["result_contract"]["submitters"] == ["verifier"]


def test_tool_metrics_use_typed_autogen_events():
    from autogen_core import FunctionCall
    from autogen_core.models import FunctionExecutionResult
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import _structured_tool_records

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
    from lychee_mas.runtime.adapters.infrastructure.docker import DockerContainerUserProxy

    class Container:
        status = "running"

        def __init__(self):
            self.calls = []

        def exec_run(self, command, *args, **kwargs):
            self.calls.append((command, args, kwargs))
            return "ok"

    container = Container()
    proxy = DockerContainerUserProxy(container, "1000:1000")

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
    from lychee_mas.runtime.adapters.frameworks.autogen.client import _call_backend

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


def test_trial_seed_is_stable_by_case_identity_and_trial_index():
    common = {
        "base_seed": 42,
        "benchmark_id": "gaia",
        "task": "gaia_validation",
        "case_id": "case-001",
    }
    first = derive_trial_seed(**common, trial_index=0)

    assert first == derive_trial_seed(**common, trial_index=0)
    assert 0 <= first < 2**31
    assert first != derive_trial_seed(**common, trial_index=1)
    assert first != derive_trial_seed(**{**common, "case_id": "case-002"}, trial_index=0)
    assert first != derive_trial_seed(**{**common, "base_seed": 43}, trial_index=0)
    assert TRIAL_SEED_DERIVATION == "sha256_case_trial_31bit_v1"


def test_local_hf_request_seed_resets_the_serialized_torch_rng():
    import torch
    from lychee_mas.runtime.adapters.inference.hf import HFBackend

    backend = HFBackend.__new__(HFBackend)
    backend._set_request_seed(12345)
    first = torch.rand(4)
    backend._set_request_seed(12345)
    second = torch.rand(4)

    assert HFBackend.supports_request_seed is True
    assert torch.equal(first, second)


def test_routing_context_records_trial_seed_on_all_events(tmp_path):
    from lychee_mas.runtime.events.store import RunEventWriter, event_payload

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.base_seed = 42
    ctx.trial_seed = 123456
    ctx.generation_seed = 123456
    ctx.seed_derivation = TRIAL_SEED_DERIVATION
    ctx.set_case("case-001", 9)
    ctx.current_trial_index = 2
    ctx.log_event("trial.started")
    ctx.log_event("agent.message.published", source="Coder", content="answer")

    trial_event, message = [
        json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()
    ]
    for record in (trial_event, message):
        payload = event_payload(record)
        assert payload["base_seed"] == 42
        assert payload["trial_seed"] == 123456
        assert payload["seed_derivation"] == TRIAL_SEED_DERIVATION


def test_backend_call_links_autogen_cancellation_token():
    from autogen_core import CancellationToken
    from lychee_mas.runtime.adapters.frameworks.autogen.client import _call_backend

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


def test_shared_event_writer_keeps_worker_case_context_isolated(tmp_path):
    from lychee_mas.runtime.events.store import RunEventWriter

    writer = RunEventWriter(run_events_path(tmp_path))
    contexts = [
        RoutingContext(
            task="gsm8k",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            event_writer=writer,
        )
        for _ in range(2)
    ]
    for index, ctx in enumerate(contexts):
        ctx.set_case(f"case-{index}", index)
        ctx.log_event("trial.started")
    writer.log_event("run.marker")

    rows = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    assert [row["case_id"] for row in rows[:2]] == ["case-0", "case-1"]
    assert rows[-1]["case_id"] is None


def test_shared_event_writer_keeps_trial_context_isolated(tmp_path):
    from lychee_mas.runtime.events.store import RunEventWriter

    writer = RunEventWriter(run_events_path(tmp_path))
    contexts = [
        RoutingContext(
            task="gaia_validation",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            event_writer=writer,
        )
        for _ in range(2)
    ]
    for index, ctx in enumerate(contexts):
        ctx.set_case(f"case-{index}", index)
        ctx.current_trial_index = index + 2
        ctx.current_attempt = 1
        ctx.log_event("agent.message.published", source="Coder", content=f"answer-{index}")
    writer.log_event("run.marker")

    rows = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    assert [row["case_id"] for row in rows[:2]] == ["case-0", "case-1"]
    assert [row["trial_index"] for row in rows[:2]] == [2, 3]
    assert rows[-1]["case_id"] is None


def test_run_event_writer_continues_global_sequence_when_resuming(tmp_path):
    from lychee_mas.runtime.events.store import RunEventWriter

    path = run_events_path(tmp_path)
    RunEventWriter(path).log_event("run.started", case_id="case-0", attempt=0)
    RunEventWriter(path, append=True).log_event("group_chat.started", case_id="case-0", attempt=1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == [1, 2]
    assert [row["attempt"] for row in rows] == [0, 1]
    assert [row["event_type"] for row in rows] == ["run.started", "group_chat.started"]
    assert all("event_family" not in row for row in rows)


def test_analysis_counts_terminal_runtime_failure_as_zero():
    namespace = runpy.run_path("scripts/analyze_benchmark_run.py")

    class _Benchmark:
        id = "toy"

    evaluated = namespace["_score_runtime_failed_trials"](
        _Benchmark(),
        [
            {
                "case_id": "case-1",
                "dataset_index": 0,
                "trial_index": 0,
                "scorer_kind": "exact",
                "error_type": "RuntimeError",
                "error_message": "runtime crashed",
            }
        ],
        [{"gold": "answer", "kind": "exact"}],
        {"case-1": {"gold": "answer", "kind": "exact"}},
        {0: {"gold": "answer", "kind": "exact"}},
        kind="exact",
    )

    assert evaluated[0]["score"] == 0.0
    assert evaluated[0]["correct"] == 0.0
    assert evaluated[0]["evaluation_status"] == "completed"
    assert evaluated[0]["score_details"]["reason"] == "trial_runtime_failed"


def test_openai_backend_preserves_native_tool_protocol():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

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
    from lychee_mas.runtime.adapters.inference.openai_compatible import _cached_input_usage

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
    from lychee_mas.runtime.adapters.inference.hf import GenResult

    result = GenResult(text="ok", n_prompt_pos=3, n_gen_tokens=1, latency_s=0.1)
    assert result.cached_input_tokens is None
    assert result.cached_input_tokens_source is None
    assert result.cached_input_tokens_status == "not_supported"


def test_openai_backend_maps_do_sample_false_to_provider_greedy_parameters():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

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
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

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
    from lychee_mas.runtime.adapters.inference.vllm_metrics import parse_prometheus_metrics

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


def test_vllm_window_snapshot_reports_histogram_p95():
    from lychee_mas.runtime.adapters.inference.vllm_metrics import _window_snapshot

    previous = {
        "queue_time_count_total": 10.0,
        "queue_time_seconds_total": 5.0,
        "queue_time_bucket_totals": {"1.0": 8.0, "5.0": 10.0, "+Inf": 10.0},
    }
    current = {
        "queue_time_count_total": 20.0,
        "queue_time_seconds_total": 20.0,
        "queue_time_bucket_totals": {"1.0": 13.0, "5.0": 19.0, "+Inf": 20.0},
    }

    snapshot = _window_snapshot(current, previous)

    assert snapshot["queue_time_window_mean_s"] == pytest.approx(1.5)
    assert snapshot["queue_time_window_p95_s"] == pytest.approx(5.0)


def test_local_sandbox_source_fingerprint_and_labels(monkeypatch: pytest.MonkeyPatch):
    from lychee_mas.runtime.adapters.infrastructure import docker as docker_sandbox

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
    from lychee_mas.runtime.adapters.infrastructure import docker as docker_sandbox

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
    from lychee_mas.runtime.adapters.infrastructure import docker as docker_sandbox

    base = docker_sandbox.LOCAL_SANDBOX_SPECS["agbench_base"]
    aligned = docker_sandbox.LOCAL_SANDBOX_SPECS["agbench_gaia"]

    assert base["image"] == "lychee-agbench-base:local"
    assert base["deps"] == []
    assert aligned["image"] == "lychee-agbench-gaia:local"
    assert aligned["deps"] == ["agbench_base"]
    assert docker_sandbox.DOCKER_BUILD_ORDER.index("agbench_base") < (
        docker_sandbox.DOCKER_BUILD_ORDER.index("agbench_gaia")
    )
    assert docker_sandbox.docker_targets_for_prepare_targets(["gaia_validation"], {}) == {
        "agbench_gaia"
    }


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
    from lychee_mas.runtime.adapters.inference.openai_compatible import (
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
    assert observed["planning_generation_tokens_per_second"] == 20
    assert observed["timeout_s"] == pytest.approx(1258.8)

    backend._observed_generation_tps = 10
    slowed = backend.request_timeout_plan(16384)
    assert slowed["planning_generation_tokens_per_second"] == 10
    assert slowed["timeout_s"] == pytest.approx(1800)


def test_openai_backend_caps_request_timeout_by_trial_deadline():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

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

    class _Client:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_payload: response)
            )

        def with_options(self, *, timeout, max_retries):
            captured.update(timeout=timeout, max_retries=max_retries)
            return self

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "model"
    backend.do_sample = False
    backend.temperature = None
    backend.top_p = None
    backend.seed = None
    backend.extra_body = {}
    backend.rate_limiter = SimpleNamespace(slot=lambda: _Slot())
    backend.client = _Client()
    backend.request_timeout_plan = lambda _max_tokens: {
        "mode": "adaptive",
        "timeout_s": 1200.0,
        "max_retries": 0,
        "policy": {},
    }

    result = backend.generate_chat(
        [{"role": "user", "content": "2+2"}],
        request_overrides={"_request_deadline_monotonic_s": time.monotonic() + 3.0},
    )

    assert 0.1 <= captured["timeout"] <= 3.0
    assert result.request_timeout_s == pytest.approx(captured["timeout"], abs=0.01)
    assert result.timeout_policy["capped_by_trial_deadline"] is True


def test_openai_backend_does_not_send_request_after_trial_deadline():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

    class _Completions:
        def create(self, **_kwargs):
            raise AssertionError("provider request must not be sent after the Trial deadline")

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    backend.model_name = "fake-model"
    backend.request_options = {}
    backend.temperature = None
    backend.top_p = None
    backend.seed = None
    backend.rate_limiter = SimpleNamespace(
        slot=lambda: __import__("contextlib").nullcontext()
    )
    backend.request_timeout_plan = lambda _max_new_tokens: {
        "timeout_s": 60.0,
        "max_retries": 0,
    }

    with pytest.raises(TimeoutError, match="Trial deadline reached"):
        backend.generate_chat(
            [{"role": "user", "content": "hello"}],
            max_new_tokens=64,
            request_overrides={"_request_deadline_monotonic_s": time.monotonic() - 1.0},
        )


def test_openai_backend_prewarms_tokenizer_for_client_retokenization(
    monkeypatch: pytest.MonkeyPatch,
):
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text in {"tokenizer warmup", "answer"}
            assert add_special_tokens is False
            return [1, 2, 3]

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "hello"}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return [1, 2, 3, 4, 5]

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
    assert backend.count_chat_tokens([{"role": "user", "content": "hello"}]) == (
        5,
        "tokenizer_chat_template",
    )


def test_openai_backend_does_not_warm_when_provider_reports_reasoning_usage():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.tokenizer_path = "/models/Qwen3.6-27B"
    backend.reasoning_token_accounting = "provider_usage"
    backend._tokenizer_warmup_report = {}

    report = backend.warmup_tokenizer()

    assert report["tokenizer_warmup_status"] == "not_required"


def test_openai_backend_supports_structured_output_and_extra_create_args():
    from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend
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
    from lychee_mas.runtime.adapters.inference.openai_compatible import _to_openai_content

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
    from lychee_mas.runtime.adapters.frameworks.autogen.client import _to_chat

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
    from lychee_mas.runtime.adapters.frameworks.autogen.client import _to_chat
    from PIL import Image as PILImage

    image = Image.from_pil(PILImage.new("RGB", (3, 2), color="white"))
    converted = _to_chat([UserMessage(content=["Read this image", image], source="user")])

    assert converted[0]["content"][0] == {"type": "text", "text": "Read this image"}
    image_part = converted[0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_qwen_reasoning_output_is_split_without_changing_plain_output():
    from lychee_mas.runtime.adapters.inference.hf import _split_reasoning_output

    reasoning, final = _split_reasoning_output("<think>\nwork\n</think>\n\n42")
    assert reasoning == "work"
    assert final == "42"
    assert _split_reasoning_output("plain answer") == ("", "plain answer")
    assert _split_reasoning_output("<think>unfinished") == ("unfinished", "")


def test_hf_multimodal_mapping_keeps_image_order():
    from autogen_core import Image
    from lychee_mas.runtime.adapters.inference.hf import _to_hf_multimodal_messages
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
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter, event_payload

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

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-1", 0)
    client = make_plain_client(
        Backend(),
        ctx=ctx,
        max_new_tokens=64,
        request_overrides={"extra_body": {"thinking_token_budget": 4}},
        controller=True,
    )
    result = asyncio.run(client.create([UserMessage(content="Who should speak?", source="user")]))

    assert result.content == "Solver"
    assert result.thought == "The solver should speak next."
    assert ctx.decisions[0]["controller"] is True
    events = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "model_call.started",
        "model_call.completed",
    ]
    start, completed = map(event_payload, events)
    assert start["autogen_model_messages"][0]["content"] == "Who should speak?"
    assert start["role_visible_messages"] == start["autogen_model_messages"]
    assert start["backend_messages"] == start["autogen_model_messages"]
    assert start["thinking_budget_requested"] == 4
    assert start["thinking_budget_parameter"] == "thinking_token_budget"
    assert start["thinking_budget_enforced_by"] == "vllm"
    assert completed["provider_response_payload"] == {"answer": "Solver"}
    assert completed["output_total_tokens"] == 6
    assert completed["output_reasoning_tokens"] == 4
    assert completed["output_answer_tokens"] == 2
    assert "parsed_output" not in completed
    assert "raw_decoded_text" not in completed


def test_model_client_usage_is_cumulative_but_create_result_usage_is_per_call():
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client

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
    assert [item["controller"] for item in ctx.decisions] == [False, False]


def test_autogen_client_recovers_reasoning_only_response_as_a_real_model_call(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter, event_payload

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
            if self.calls == 1:
                text = ""
                reasoning = "I have solved it but omitted the deliverable."
            else:
                assert "Return the final deliverable now" in messages[-1]["content"]
                text = "final answer"
                reasoning = ""
            return SimpleNamespace(
                text=text,
                reasoning_content=reasoning,
                n_prompt_pos=10,
                n_gen_tokens=4,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={},
            )

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="livecodebench",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-recovery", 0)
    backend = Backend()
    client = make_plain_client(backend, role="Reviewer", ctx=ctx, max_new_tokens=64)

    result = asyncio.run(client.create([UserMessage(content="Review it", source="user")]))

    assert result.content == "final answer"
    assert backend.calls == 2
    events = [json.loads(line) for line in run_events_path(tmp_path).read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "model_call.started",
        "model_call.completed",
        "model_call.empty_final_recovery_requested",
        "model_call.started",
        "model_call.completed",
    ]
    recovery = event_payload(events[2])
    assert recovery["recovery_attempt"] == 1
    assert recovery["max_recovery_attempts"] == 3


def test_group_chat_rejects_fake_model_streaming():
    from lychee_mas.runtime.coordination.group_chat import normalize_group_chat_config

    try:
        normalize_group_chat_config({"type": "selector", "model_client_streaming": True})
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("streaming=true must fail validation")


def test_provider_finish_reason_is_mapped_to_autogen_contract():
    from lychee_mas.runtime.adapters.frameworks.autogen.client import _autogen_finish_reason

    assert _autogen_finish_reason("tool_calls", has_function_calls=True) == "function_calls"
    assert _autogen_finish_reason("tool_calls", has_function_calls=False) == "unknown"
    assert _autogen_finish_reason("stop") == "stop"
    assert _autogen_finish_reason("provider_specific_reason") == "unknown"


def test_plain_control_client_does_not_rewrite_magentic_one_ledger(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter, event_payload

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

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-ledger", 0)
    client = make_plain_client(
        Backend(), role="Orchestrator", ctx=ctx, max_new_tokens=64, controller=True
    )
    result = asyncio.run(
        client.create(
            [UserMessage(content="Return the progress ledger.", source="user")],
            json_output=True,
        )
    )

    assert result.content == shorthand
    events = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    completed = event_payload(events[-1])
    assert completed["provider_response_payload"] == {"content": shorthand}
    assert "parsed_final_content" not in completed
    assert "json_output_repaired" not in completed
    assert "json_output_normalized" not in completed


def test_tool_request_is_logged_once_and_execution_links_to_it(tmp_path):
    from autogen_core.models import FunctionExecutionResult, UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_injection_client
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime
    from lychee_mas.runtime.events.store import RunEventWriter

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = False
        supports_native_tools = True
        supports_json_output = True
        model_info = {}
        tok = None

        def generate_chat(self, messages, max_new_tokens=64, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "visit_url"
            return SimpleNamespace(
                text="",
                reasoning_content="",
                n_prompt_pos=12,
                n_gen_tokens=4,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="tool_calls",
                prefix_len=0,
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "visit_url",
                        "arguments": '{"url":"https://example.com"}',
                    }
                ],
                provider_request_payload={"messages": messages},
                provider_response_payload={},
            )

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-tools", 0)
    client = make_injection_client(Backend(), "WebSurfer", ctx, max_new_tokens=64)
    result = asyncio.run(
        client.create(
            [UserMessage(content="Open the page", source="user")],
            tools=[
                {
                    "name": "visit_url",
                    "description": "Visit a URL",
                    "parameters": {"type": "object"},
                }
            ],
        )
    )

    runtime = AutoGenRuntime.__new__(AutoGenRuntime)
    runtime.ctx = ctx
    runtime.web_proxy_url = None
    runtime._log_autogen_event(
        SimpleNamespace(
            type="ToolCallRequestEvent",
            source="WebSurfer",
            content=result.content,
            models_usage=None,
            metadata={},
        ),
        1,
    )
    runtime._log_autogen_event(
        SimpleNamespace(
            type="ToolCallExecutionEvent",
            source="WebSurfer",
            content=[
                FunctionExecutionResult(
                    call_id="call-1",
                    name="visit_url",
                    content="opened",
                    is_error=False,
                )
            ],
            models_usage=None,
            metadata={},
        ),
        2,
    )

    events = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    requests = [event for event in events if event["event_type"] == "tool_call.requested"]
    executions = [event for event in events if event["event_type"] == "tool_execution.observed"]
    assert len(requests) == 1
    assert requests[0]["parent_event_id"] == next(
        event["event_id"] for event in events if event["event_type"] == "model_call.completed"
    )
    assert executions[0]["parent_event_id"] == requests[0]["event_id"]
    assert executions[0]["correlation_id"] == requests[0]["correlation_id"] == "call-1"
    assert executions[0]["payload"]["duration_s"] >= 0


def test_model_call_start_always_preserves_message_triplet(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter

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

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-lossless", 0)
    client = make_plain_client(Backend(), ctx=ctx, max_new_tokens=64, controller=True)
    asyncio.run(client.create([UserMessage(content="Pick a speaker", source="user")]))

    start, end = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    start_payload = start["payload"]
    end_payload = end["payload"]
    for key in ("autogen_model_messages", "role_visible_messages", "backend_messages"):
        assert key in start_payload
    assert end_payload["provider_request_payload"]["messages"][0]["content"] == "Pick a speaker"
    assert end_payload["provider_response_payload"] == {"answer": "Solver"}


def test_plain_control_client_projects_selector_decision_events(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import make_plain_client
    from lychee_mas.runtime.events.store import RunEventWriter, iter_run_events

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

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="bbeh",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-selector", 0)
    selected: list[str] = []
    client = make_plain_client(
        Backend(),
        role="Selector",
        ctx=ctx,
        max_new_tokens=64,
        controller=True,
        controller_operation="select_next",
        controller_candidates=("Solver", "Verifier"),
        controller_decision_callback=selected.append,
    )

    asyncio.run(client.create([UserMessage(content="Choose the next node", source="user")]))

    events = list(
        iter_run_events(tmp_path, event_types=("coordination.operation.completed",))
    )
    assert len(events) == 1
    assert events[0]["case_id"] == "case-selector"
    assert {
        key: events[0]["payload"][key]
        for key in (
            "operation",
            "actor",
            "framework",
            "decision_source",
            "raw_decision",
            "selected_node",
            "valid",
        )
    } == {
        "operation": "select_next",
        "actor": "Selector",
        "framework": "autogen",
        "decision_source": "model_client",
        "raw_decision": "Solver",
        "selected_node": "Solver",
        "valid": True,
    }
    assert selected == ["Solver"]


def test_autogen_clients_forward_trial_deadline_to_backend(tmp_path):
    from autogen_core.models import UserMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.client import (
        make_injection_client,
        make_plain_client,
    )
    from lychee_mas.runtime.events.store import RunEventWriter

    class Backend:
        supports_concurrent_requests = False
        supports_request_seed = False
        supports_request_overrides = True
        supports_native_tools = False
        model_info = {}
        tok = None

        def __init__(self) -> None:
            self.overrides: list[dict] = []

        def generate_chat(self, messages, max_new_tokens=64, request_overrides=None):
            self.overrides.append(dict(request_overrides or {}))
            return SimpleNamespace(
                text="done",
                reasoning_content="",
                n_prompt_pos=3,
                n_gen_tokens=1,
                latency_s=0.01,
                request_queue_latency_s=0.0,
                finish_reason="stop",
                prefix_len=0,
                provider_request_payload={"messages": messages},
                provider_response_payload={"content": "done"},
            )

    for index, make_client in enumerate((make_injection_client, make_plain_client)):
        run_dir = tmp_path / str(index)
        backend = Backend()
        ctx = RoutingContext(
            task="bbeh",
            router=fixed_channel_router("none"),
            memory=_Memory(),
            event_writer=RunEventWriter(run_events_path(run_dir)),
        )
        ctx.set_case(f"deadline-{index}", 0)
        ctx.case_deadline_monotonic_s = time.monotonic() + 10.0
        client = make_client(backend, role="Solver", ctx=ctx, max_new_tokens=64)

        asyncio.run(client.create([UserMessage(content="answer", source="user")]))

        assert backend.overrides
        assert backend.overrides[0]["_request_deadline_monotonic_s"] == pytest.approx(
            ctx.case_deadline_monotonic_s
        )


def test_event_writer_rejects_legacy_non_dotted_event_names(tmp_path):
    from lychee_mas.runtime.events.store import RunEventWriter

    writer = RunEventWriter(run_events_path(tmp_path))
    with pytest.raises(ValueError, match="invalid dotted event_type"):
        writer.log_event("model_call_start")


def test_agent_message_event_keeps_full_message_without_duplicate_record(tmp_path):
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime
    from lychee_mas.runtime.events.store import RunEventWriter

    writer = RunEventWriter(run_events_path(tmp_path))
    ctx = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=writer,
    )
    ctx.set_case("case-chat", 7)
    ctx.current_trial_index = 2
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

    events = [json.loads(line) for line in (run_events_path(tmp_path)).read_text().splitlines()]
    assert len(events) == 1
    group_event = events[0]
    payload = group_event["payload"]
    assert payload["content"] == "full group-chat message"
    assert payload["metadata"] == {"tag": "official-event"}
    assert group_event["trial_index"] == 2
    assert group_event["attempt"] == 1
    assert "event_family" not in group_event
    assert group_event["event_type"] == "agent.message.published"
    assert payload["prompt_tokens"] == 11


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
    from lychee_mas.eval.benchmarks.task_config import extractor_for_task

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


def test_magentic_one_retries_ledger_with_wrong_fence_language():
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import BaseChatMessage, TextMessage
    from autogen_agentchat.teams import MagenticOneGroupChat
    from autogen_core import CancellationToken
    from autogen_ext.models.replay import ReplayChatCompletionClient

    class EchoAgent(BaseChatAgent):
        @property
        def produced_message_types(self):
            return (TextMessage,)

        async def on_messages(
            self,
            messages: list[BaseChatMessage],
            cancellation_token: CancellationToken,
        ) -> Response:
            return Response(chat_message=TextMessage(content="done", source=self.name))

        async def on_reset(self, cancellation_token: CancellationToken) -> None:
            return None

    completed_ledger = json.dumps(
        {
            "is_request_satisfied": {"answer": True, "reason": "done"},
            "is_progress_being_made": {"answer": True, "reason": "done"},
            "is_in_loop": {"answer": False, "reason": "done"},
            "instruction_or_question": {"answer": "Task completed", "reason": "done"},
            "next_speaker": {"answer": "agent", "reason": "done"},
        }
    )
    model_client = ReplayChatCompletionClient(
        chat_completions=[
            "No facts",
            "No plan",
            f"```python\n{completed_ledger}\n```",
            completed_ledger,
            "final answer",
        ]
    )
    team = MagenticOneGroupChat(
        participants=[EchoAgent("agent", description="echo agent")],
        model_client=model_client,
    )

    result = asyncio.run(team.run(task="Finish the task"))

    assert result.messages[-1].to_text() == "final answer"
    assert result.stop_reason == "done"


def test_autogen_web_surfer_recovers_transient_start_page_reset() -> None:
    from lychee_mas.runtime.adapters.frameworks.autogen.web_surfer import (
        resilient_multimodal_web_surfer_class,
    )

    class ResetNavigationError(Exception):
        pass

    class FakePage:
        def __init__(self) -> None:
            self.visited: list[str] = []

        async def goto(self, url: str, **_kwargs) -> None:
            self.visited.append(url)

    class FailingWebSurfer:
        def __init__(self) -> None:
            self._page = FakePage()
            self._chat_history = ["stale"]
            self._last_download = "file"
            self._prior_metadata_hash = "hash"

        async def on_reset(self, _cancellation_token) -> None:
            self._chat_history.clear()
            raise ResetNavigationError("start page timed out")

    recovered: list[str] = []
    web_surfer_class = resilient_multimodal_web_surfer_class(
        base_class=FailingWebSurfer,
        recoverable_error_type=ResetNavigationError,
    )
    agent = web_surfer_class(
        on_reset_recovered=lambda exc: recovered.append(str(exc)),
    )

    asyncio.run(agent.on_reset(None))

    assert agent._page.visited == ["about:blank"]
    assert agent._chat_history == []
    assert agent._last_download is None
    assert agent._prior_metadata_hash is None
    assert recovered == ["start page timed out"]


def test_swe_bench_case_checkout_is_independent_of_repository_cache(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from lychee_mas.eval.benchmarks.swe_bench_verified import prepare_case_workspace

    raw_root = tmp_path / "raw"
    cache = raw_root / "swe_bench_verified/github_repositories/example--project"
    cache.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(cache)], check=True)
    subprocess.run(["git", "-C", str(cache), "config", "user.name", "Lychee Test"], check=True)
    subprocess.run(
        ["git", "-C", str(cache), "config", "user.email", "lychee-test@example.invalid"],
        check=True,
    )
    (cache / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(cache), "add", "module.py"], check=True)
    subprocess.run(["git", "-C", str(cache), "commit", "-q", "-m", "base"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(cache), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", str(raw_root))

    checkout = prepare_case_workspace(
        {"repo": "example/project", "base_commit": commit},
        tmp_path / "workspace",
    )

    assert not (checkout / ".git/objects/info/alternates").exists()
    assert (checkout / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    shutil.rmtree(cache)
    subprocess.run(["git", "-C", str(checkout), "status", "--porcelain"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True,
    )


def test_swe_bench_patch_collection_preserves_non_utf8_diff_bytes(tmp_path):
    import subprocess

    from lychee_mas.eval.benchmarks.swe_bench_verified import collect_model_patch

    repo = tmp_path / "workspace" / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Lychee Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "lychee-test@example.invalid"],
        check=True,
    )
    target = repo / "payload.dat"
    target.write_bytes(b"original\n")
    subprocess.run(["git", "-C", str(repo), "add", "payload.dat"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
    target.write_bytes(b"changed:\xac\n")

    patch = collect_model_patch(tmp_path / "workspace")

    assert patch.startswith("diff --git")
    assert b"changed:\xac" in patch.encode("utf-8", errors="surrogateescape")
