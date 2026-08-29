from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from lychee_mas.core.types import Answer, Message, Trajectory
from lychee_mas.eval.runner.backend_assembly import BackendAssemblyRequest, assemble_backend
from lychee_mas.eval.runner.cli import build_run_parser
from lychee_mas.eval.runner.configuration import (
    get_config_value,
    optional_positive_limit,
    resolve_prefix_length,
)
from lychee_mas.eval.runner.model_calls import normalize_model_call, summarize_model_calls
from lychee_mas.eval.runner.run_artifacts import model_tag
from lychee_mas.eval.runner.run_state import (
    atomic_write_json,
    graceful_stop_request,
    scheduler_trial_admission_total,
    scheduler_trial_concurrency,
)
from lychee_mas.eval.runner.trial_executor import TrialExecutionSpec, execute_trial
from lychee_mas.eval.runner.trial_pool import TrialPoolRequest, execute_trial_pool
from lychee_mas.memory.context import RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.execution.concurrency import normalize_concurrency_policy


class _Memory:
    def __init__(self) -> None:
        self.seeded: list[str] = []

    def reset(self) -> None:
        pass

    def seed(self, value: str) -> None:
        self.seeded.append(value)

    def observe(self, messages) -> None:
        pass

    def recall(self, decision, query):
        return SimpleNamespace(
            NL_Channel=None,
            Latent_Channel=None,
            NL_strategy=None,
            Latent_strategy=None,
        )


def _trial_spec(
    *, max_attempts: int = 1, max_case_wall_time_s: float | None = None
) -> TrialExecutionSpec:
    return TrialExecutionSpec(
        case_order=1,
        num_cases=1,
        item={"question": "2 + 2?", "context": "Use arithmetic."},
        dataset_index=7,
        trial_index=0,
        trials_per_case=1,
        task="gsm8k",
        kind="gsm8k",
        benchmark_id="gsm8k",
        method="none",
        profile="reason",
        max_rounds=1,
        max_case_wall_time_s=max_case_wall_time_s,
        max_attempts_per_trial=max_attempts,
    )


def _routing_context(memory: _Memory) -> RoutingContext:
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=memory,
    )
    context.base_seed = 42
    return context


def _team_graph() -> SimpleNamespace:
    """Return the minimum explicit TeamSpec graph required by a Trial."""

    return SimpleNamespace(meta={"dynamic_topology": False}, nodes=[], edges={})


def _backend_request(**overrides) -> BackendAssemblyRequest:
    values = {
        "backend_provider": "unknown",
        "method": "none",
        "model_tag": "model-a",
        "api_model": None,
        "model_path": None,
        "device": "cpu",
        "enable_thinking": False,
        "dtype_name": "float32",
        "vision": False,
        "deployment_document": None,
        "graph": None,
        "generation": {},
        "api_options": {},
    }
    values.update(overrides)
    return BackendAssemblyRequest(**values)


def test_configuration_resolution_has_one_public_contract() -> None:
    config = {"router": {"P": 4}, "memory": {"P": 12}, "run": {"task": "gsm8k"}}

    assert get_config_value(config, "run.task") == "gsm8k"
    assert get_config_value(config, "run.missing", "fallback") == "fallback"
    assert resolve_prefix_length(config, None) == 4
    assert resolve_prefix_length(config, 8) == 8
    assert resolve_prefix_length({"memory": {"P": 12}}, None) == 12


def test_run_cli_parser_exposes_framework_trial_and_pressure_controls() -> None:
    args = build_run_parser().parse_args(
        [
            "--config",
            "run.yaml",
            "--runtime",
            "langgraph",
            "--trials-per-case",
            "3",
            "--trial-concurrency",
            "8",
            "--collect-vllm-metrics",
        ]
    )

    assert args.config == "run.yaml"
    assert args.runtime == "langgraph"
    assert args.trials_per_case == 3
    assert args.trial_concurrency == 8
    assert args.collect_vllm_metrics is True


def test_model_tag_prefers_explicit_identity_then_truthful_model_source() -> None:
    args = SimpleNamespace(model_tag="paper-label", api_model=None, model_path=None)
    assert model_tag(args, {}) == "paper-label"

    args = SimpleNamespace(model_tag=None, api_model=None, model_path="/models/Qwen3.5-9B")
    assert model_tag(args, {}, model_path=args.model_path) == "Qwen3.5-9B"

    args = SimpleNamespace(model_tag=None, api_model="qwen-plus", model_path=None)
    assert model_tag(args, {}, backend_provider="api", api_model=args.api_model, suffix="-api") == (
        "qwen-plus-api"
    )


def test_backend_assembly_rejects_incomplete_api_and_unknown_provider_early() -> None:
    with pytest.raises(SystemExit, match="requires backend.model"):
        assemble_backend(_backend_request(backend_provider="api"))
    with pytest.raises(SystemExit, match="unknown backend.provider"):
        assemble_backend(_backend_request())


@pytest.mark.parametrize("value", [None, "unlimited", "unbounded"])
def test_optional_positive_limit_accepts_unbounded_values(value) -> None:
    assert optional_positive_limit(value, name="limit") is None


def test_optional_positive_limit_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        optional_positive_limit(0, name="limit")


def test_run_state_files_are_atomic_and_tolerate_partial_input(tmp_path) -> None:
    target = tmp_path / "run_status.json"
    atomic_write_json(target, {"status": "running"})
    atomic_write_json(target, {"status": "completed"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "completed"}
    assert list(tmp_path.glob("run_status.json.tmp.*")) == []


def test_scheduler_control_file_projection(tmp_path) -> None:
    target = tmp_path / "run_segment.json"
    atomic_write_json(
        target,
        {
            "scheduler_trial_concurrency": 7,
            "scheduler_trial_admission_total": 19,
            "action": "stop_after_active_trials",
        },
    )

    assert scheduler_trial_concurrency(target) == 7
    assert scheduler_trial_admission_total(target) == 19
    assert graceful_stop_request(target)["action"] == "stop_after_active_trials"


def test_model_call_usage_projection_normalizes_provider_fields() -> None:
    call = normalize_model_call(
        {
            "prompt_pos": 12,
            "prefix_len": 2,
            "gen_tokens": 5,
            "latency_s": 0.25,
            "role": "Solver",
        }
    )
    summary = summarize_model_calls([call])

    assert call["text_input_tokens"] == 10
    assert summary["input_positions_total"] == 12
    assert summary["latent_prefix_positions_total"] == 2
    assert summary["output_tokens_total"] == 5
    assert summary["routing_trace"] == [
        {
            "role": "Solver",
            "turn_index": None,
            "sender_role": None,
            "memory_channel": None,
            "routing_reason": None,
            "nl_strategy": None,
            "latent_strategy": None,
        }
    ]


def test_trial_executor_projects_one_successful_terminal_record() -> None:
    memory = _Memory()
    context = _routing_context(memory)
    trajectory = Trajectory(
        messages=[Message(sender="Solver", content="4")],
        final_answer=Answer(content="4", source="Solver"),
        meta={
            "decisions": [
                {
                    "role": "Solver",
                    "prompt_pos": 10,
                    "gen_tokens": 2,
                    "latency_s": 0.1,
                }
            ],
            "case_wall_time_s": 0.2,
        },
    )

    class _Runtime:
        async def run(self, graph, query):
            assert query.id == "7"
            return trajectory

    result = asyncio.run(
        execute_trial(
            spec=_trial_spec(),
            memory=memory,
            context=context,
            runtime=_Runtime(),
            base_graph=_team_graph(),
        )
    )

    assert result["exception"] is None
    assert result["retry_count"] == 0
    assert result["record"]["case_id"] == "7"
    assert result["record"]["final_output"] == "4"
    assert result["record"]["input_positions_total"] == 10
    assert result["record"]["output_tokens_total"] == 2
    assert memory.seeded == ["Use arithmetic."]


def test_trial_executor_retries_infrastructure_failure_without_new_trial() -> None:
    memory = _Memory()
    context = _routing_context(memory)

    class _Runtime:
        calls = 0

        async def run(self, graph, query):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    runtime = _Runtime()
    result = asyncio.run(
        execute_trial(
            spec=_trial_spec(max_attempts=2),
            memory=memory,
            context=context,
            runtime=runtime,
            base_graph=_team_graph(),
        )
    )

    assert runtime.calls == 2
    assert result["retry_count"] == 1
    assert isinstance(result["exception"], RuntimeError)
    assert result["record"]["status"] == "failed"
    assert result["record"]["error_type"] == "RuntimeError"
    assert result["record"]["trial_index"] == 0


def test_trial_executor_sets_one_monotonic_deadline_for_the_whole_trial() -> None:
    memory = _Memory()
    context = _routing_context(memory)

    class _Runtime:
        async def run(self, graph, query):
            assert context.case_deadline_monotonic_s is not None
            assert context.case_deadline_monotonic_s > time.monotonic()
            return Trajectory(
                messages=[Message(sender="Solver", content="4")],
                final_answer=Answer(content="4", source="Solver"),
                meta={"decisions": [], "case_wall_time_s": 0.01},
            )

    result = asyncio.run(
        execute_trial(
            spec=_trial_spec(max_case_wall_time_s=1.0),
            memory=memory,
            context=context,
            runtime=_Runtime(),
            base_graph=_team_graph(),
        )
    )

    assert result["exception"] is None
    assert context.case_deadline_exceeded is False


def test_trial_pool_owns_runner_concurrency_even_when_no_work_remains(tmp_path) -> None:
    run_status_path = tmp_path / "run_status.json"
    request = TrialPoolRequest(
        work_items=[],
        initial_worker=(None, None, None, None),
        worker_factory=lambda: (None, None, None, None),
        worker_capacity=4,
        run_status={"active_cases": []},
        run_status_path=str(run_status_path),
        run_control_path=str(tmp_path / "run_control.json"),
        run_event_id="run-1",
        event_writer=SimpleNamespace(),
        concurrency_policy=normalize_concurrency_policy({}, configured_concurrency=1),
        initial_scheduler_trial_concurrency=None,
        initial_scheduler_trial_admission_total=None,
        scheduler_controls_concurrency=False,
        data_count=0,
        start_index=0,
        trials_per_case=1,
        base_seed=0,
        benchmark_id="gsm8k",
        task="gsm8k",
        kind="gsm8k",
        method="none",
        profile="single",
        max_rounds=1,
        max_case_wall_time_s=None,
        max_attempts_per_trial=1,
        on_trial_error="continue",
        successful_trials_by_case={},
        trial_records=[],
        base_graph=_team_graph(),
    )

    result = asyncio.run(execute_trial_pool(request))

    assert result.graceful_stop_requested is False
    assert result.effective_worker_count == 1
    assert json.loads(run_status_path.read_text(encoding="utf-8"))[
        "effective_trial_concurrency"
    ] == 1


def test_trial_pool_recycles_runtime_after_failed_trial(tmp_path, monkeypatch) -> None:
    """A failed Trial must not share mutable Runtime state with the next Trial."""

    from lychee_mas.eval.runner import trial_pool as trial_pool_module

    class _EventWriter:
        def __init__(self) -> None:
            self.trials: list[dict] = []

        def record_trial(self, record) -> None:
            self.trials.append(dict(record))

        def log_event(self, *_args, **_kwargs) -> None:
            return None

    event_writer = _EventWriter()
    initial_context = SimpleNamespace(
        identity="initial",
        event_writer=event_writer,
        current_run_event_id="run-1",
        current_trial_event_id="trial-initial",
        current_worker_id=None,
    )
    replacement_contexts: list[SimpleNamespace] = []

    def worker_factory():
        context = SimpleNamespace(
            identity=f"replacement-{len(replacement_contexts) + 1}",
            event_writer=None,
            current_run_event_id=None,
            current_trial_event_id="trial-replacement",
            current_worker_id=None,
        )
        replacement_contexts.append(context)
        return object(), object(), context, object()

    seen_contexts: list[str] = []

    async def fake_execute_trial(*, spec, context, **_kwargs):
        seen_contexts.append(context.identity)
        failed = len(seen_contexts) == 1
        return {
            "record": {
                "case_id": str(spec.item["case_id"]),
                "trial_index": spec.trial_index,
                "status": "failed" if failed else "completed",
                "error_type": "TimeoutError" if failed else None,
                "error_message": "timed out" if failed else None,
            },
            "retry_count": 0,
            "exception": TimeoutError("timed out") if failed else None,
        }

    monkeypatch.setattr(trial_pool_module, "execute_trial", fake_execute_trial)
    run_status_path = tmp_path / "run_status.json"
    request = TrialPoolRequest(
        work_items=[
            (0, {"case_id": "case-a", "question": "A"}, 0),
            (1, {"case_id": "case-b", "question": "B"}, 0),
        ],
        initial_worker=(object(), object(), initial_context, object()),
        worker_factory=worker_factory,
        worker_capacity=1,
        run_status={
            "active_cases": [],
            "retry_attempt_count": 0,
            "failed_trials": 0,
            "successful_trials": 0,
        },
        run_status_path=str(run_status_path),
        run_control_path=str(tmp_path / "run_control.json"),
        run_event_id="run-1",
        event_writer=event_writer,
        concurrency_policy=normalize_concurrency_policy({}, configured_concurrency=1),
        initial_scheduler_trial_concurrency=None,
        initial_scheduler_trial_admission_total=None,
        scheduler_controls_concurrency=False,
        data_count=2,
        start_index=0,
        trials_per_case=1,
        base_seed=0,
        benchmark_id="gsm8k",
        task="gsm8k",
        kind="gsm8k",
        method="none",
        profile="single",
        max_rounds=1,
        max_case_wall_time_s=1.0,
        max_attempts_per_trial=1,
        on_trial_error="continue",
        successful_trials_by_case={},
        trial_records=[],
        base_graph=_team_graph(),
    )

    asyncio.run(execute_trial_pool(request))

    assert seen_contexts == ["initial", "replacement-1"]
    assert len(replacement_contexts) == 1
    assert replacement_contexts[0].event_writer is event_writer
    assert replacement_contexts[0].current_run_event_id == "run-1"
    assert [record["case_id"] for record in event_writer.trials] == ["case-a", "case-b"]
