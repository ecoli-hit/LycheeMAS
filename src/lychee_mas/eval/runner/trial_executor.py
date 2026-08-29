"""Execute one isolated benchmark Trial and project its terminal record."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .model_calls import normalize_model_call, summarize_model_calls
from .trial_identity import case_id_from_item


@dataclass(frozen=True, slots=True)
class TrialExecutionSpec:
    """Immutable inputs that identify and constrain one Trial execution."""

    case_order: int
    num_cases: int
    item: dict[str, Any]
    dataset_index: int
    trial_index: int
    trials_per_case: int
    task: str
    kind: str
    benchmark_id: str
    method: str
    profile: str
    max_rounds: int
    max_case_wall_time_s: float | None
    max_attempts_per_trial: int


def _trial_tag(spec: TrialExecutionSpec) -> str:
    if spec.trials_per_case <= 1:
        return ""
    return f" trial={spec.trial_index + 1}/{spec.trials_per_case}"


def _tool_projection(trajectory) -> dict[str, Any]:
    tool_calls = list(trajectory.meta.get("tool_calls", []))
    return {
        "tool_calls": tool_calls,
        "tool_requests": list(trajectory.meta.get("tool_requests", [])),
        "tool_executions": list(trajectory.meta.get("tool_executions", [])),
        "tool_agent_errors": list(trajectory.meta.get("tool_agent_errors", [])),
        "tool_call_count": int(trajectory.meta.get("tool_call_count", len(tool_calls)) or 0),
        "tool_error_count": int(trajectory.meta.get("tool_error_count", 0) or 0),
        "tool_request_count": int(trajectory.meta.get("tool_request_count", 0) or 0),
        "tool_execution_count": int(trajectory.meta.get("tool_execution_count", 0) or 0),
        "tool_agent_error_count": int(trajectory.meta.get("tool_agent_error_count", 0) or 0),
    }


def _failed_record(
    *,
    spec: TrialExecutionSpec,
    case_id: str,
    base_seed: int,
    trial_seed: int,
    error: dict[str, Any],
    model_calls: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    from lychee_mas.runtime.execution.seeding import TRIAL_SEED_DERIVATION

    usage = summarize_model_calls(model_calls)
    return {
        "case_id": case_id,
        "dataset_index": spec.dataset_index,
        "base_seed": base_seed,
        "trial_seed": trial_seed,
        "seed_derivation": TRIAL_SEED_DERIVATION,
        "trial_index": spec.trial_index,
        "trials_per_case": spec.trials_per_case,
        "task": spec.task,
        "method": spec.method,
        "scorer_kind": spec.kind,
        "benchmark_id": spec.benchmark_id,
        "question": spec.item["question"][:500],
        "final_output": "",
        "status": "failed",
        "error_type": error["error_type"],
        "error_message": error["error_message"],
        "traceback": error["traceback"],
        **usage,
        "num_messages": 0,
        "case_wall_time_s": round(elapsed_s, 3),
        "num_tool_calls": 0,
        "num_tool_errors": 0,
        "num_tool_requests": 0,
        "num_tool_executions": 0,
        "num_tool_agent_errors": 0,
        "message_count": 0,
        "tool_call_count": 0,
        "tool_error_count": 0,
        "tool_request_count": 0,
        "tool_execution_count": 0,
        "tool_agent_error_count": 0,
        "tool_calls": [],
        "tool_requests": [],
        "tool_executions": [],
        "tool_agent_errors": [],
        "n_messages": 0,
    }


def _completed_record(
    *,
    spec: TrialExecutionSpec,
    case_id: str,
    base_seed: int,
    trial_seed: int,
    dynamic_topology: bool,
    case_graph,
    trajectory,
    elapsed_s: float,
) -> dict[str, Any]:
    from lychee_mas.runtime.execution.seeding import TRIAL_SEED_DERIVATION

    final_output = trajectory.final_answer.content if trajectory.final_answer else ""
    model_calls = [
        normalize_model_call(value) for value in trajectory.meta.get("decisions", [])
    ]
    usage = summarize_model_calls(model_calls)
    tools = _tool_projection(trajectory)
    message_count = len(trajectory.messages)
    wall_time = float(trajectory.meta.get("case_wall_time_s", 0.0) or 0.0)
    record = {
        "case_id": case_id,
        "dataset_index": spec.dataset_index,
        "base_seed": base_seed,
        "trial_seed": trial_seed,
        "seed_derivation": TRIAL_SEED_DERIVATION,
        "trial_index": spec.trial_index,
        "trials_per_case": spec.trials_per_case,
        "task": spec.task,
        "method": spec.method,
        "scorer_kind": spec.kind,
        "benchmark_id": spec.benchmark_id,
        "question": spec.item["question"][:500],
        "final_output": final_output,
        "stop_reason": trajectory.meta.get("stop_reason"),
        "max_model_calls_per_case": trajectory.meta.get("max_model_calls_per_case"),
        "model_calls_started": trajectory.meta.get("model_calls_started", len(model_calls)),
        "model_call_budget_exhausted": bool(
            trajectory.meta.get("model_call_budget_exhausted", False)
        ),
        **usage,
        "num_messages": message_count,
        "case_wall_time_s": round(wall_time or elapsed_s, 3),
        "num_tool_calls": tools["tool_call_count"],
        "num_tool_errors": tools["tool_error_count"],
        "num_tool_requests": tools["tool_request_count"],
        "num_tool_executions": tools["tool_execution_count"],
        "num_tool_agent_errors": tools["tool_agent_error_count"],
        "message_count": message_count,
        **tools,
        "workspace": trajectory.meta.get("workspace"),
        "workspace_info": trajectory.meta.get("workspace_info", {}),
        "copied_files": trajectory.meta.get("copied_files", []),
        "dynamic_topology": dynamic_topology,
        "topology": {
            "type": case_graph.meta.get("topology_type"),
            "roles": [node.name for node in case_graph.nodes],
            "edges": case_graph.edges,
            "speaking_order": case_graph.meta.get("speaking_order"),
        }
        if dynamic_topology
        else None,
        "n_messages": message_count,
    }
    return record


async def execute_trial(
    *,
    spec: TrialExecutionSpec,
    memory,
    context,
    runtime,
    base_graph,
) -> dict[str, Any]:
    """Run one Trial against its explicit TeamSpec graph.

    Benchmark records provide task data only. Team structure is compiled once
    from the required TeamSpec and must never be inferred from dataset fields.
    """

    from lychee_mas.core.types import TaskQuery
    from lychee_mas.runtime.events.store import exception_record
    from lychee_mas.runtime.execution.seeding import TRIAL_SEED_DERIVATION, derive_trial_seed

    item = spec.item
    case_id = case_id_from_item(item, spec.dataset_index)
    trial_tag = _trial_tag(spec)
    raw_metadata = item.get("metadata")
    metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    case_graph = base_graph
    dynamic_topology = bool(base_graph.meta.get("dynamic_topology"))
    runtime_query_id = (
        case_id if spec.trials_per_case == 1 else f"{case_id}__trial{spec.trial_index}"
    )
    query = TaskQuery(
        question=item["question"],
        context=item.get("context"),
        gold=None,
        id=runtime_query_id,
        meta={**metadata, "kind": spec.kind, "dynamic_topology": dynamic_topology},
    )
    print(f"  [{spec.case_order}/{spec.num_cases}{trial_tag}] start case_id={case_id}", flush=True)
    started_monotonic_s = time.monotonic()
    context.case_deadline_monotonic_s = (
        None
        if spec.max_case_wall_time_s is None
        else started_monotonic_s + spec.max_case_wall_time_s
    )
    context.case_deadline_exceeded = False
    context.set_case(case_id, spec.dataset_index)
    context.current_trial_index = spec.trial_index
    context.current_attempt = None
    context.set_trial_event(None)
    base_seed = int(getattr(context, "base_seed", 0))
    trial_seed = derive_trial_seed(
        base_seed=base_seed,
        benchmark_id=spec.benchmark_id,
        task=spec.task,
        case_id=case_id,
        trial_index=spec.trial_index,
    )
    context.trial_seed = trial_seed
    context.seed_derivation = TRIAL_SEED_DERIVATION
    context.generation_seed = trial_seed
    trial_event_id = context.log_event(
        "trial.started",
        case_order=spec.case_order,
        num_cases=spec.num_cases,
        trial_index=spec.trial_index,
        question=item["question"],
        has_context=bool(item.get("context")),
        scorer_kind=spec.kind,
        benchmark_id=spec.benchmark_id,
        task=spec.task,
        method=spec.method,
        trials_per_case=spec.trials_per_case,
        dynamic_topology=dynamic_topology,
        topology_type=case_graph.meta.get("topology_type"),
        topology_roles=[node.name for node in case_graph.nodes],
        topology_edges=case_graph.edges,
        speaking_order=case_graph.meta.get("speaking_order"),
        base_seed=base_seed,
        trial_seed=trial_seed,
        seed_derivation=TRIAL_SEED_DERIVATION,
        max_case_wall_time_s=spec.max_case_wall_time_s,
    )
    context.set_trial_event(trial_event_id)

    trajectory = None
    error = None
    last_exception = None
    retry_count = 0
    for attempt in range(spec.max_attempts_per_trial):
        context.reset(preserve_model_call_budget=attempt > 0)
        context.set_case(case_id, spec.dataset_index)
        context.current_trial_index = spec.trial_index
        context.current_attempt = attempt + 1
        attempt_event_id = context.log_event(
            "attempt.started",
            parent_event_id=trial_event_id,
            attempt=attempt + 1,
            max_attempts=spec.max_attempts_per_trial,
            trial_index=spec.trial_index,
        )
        context.current_attempt_event_id = attempt_event_id
        try:
            if item.get("context") and hasattr(memory, "seed"):
                memory.seed(item["context"])
            if spec.max_case_wall_time_s is None:
                trajectory = await runtime.run(case_graph, query)
            else:
                deadline = context.case_deadline_monotonic_s
                assert deadline is not None
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"Trial exceeded max_case_wall_time_s={spec.max_case_wall_time_s:g}"
                    )
                try:
                    trajectory = await asyncio.wait_for(
                        runtime.run(case_graph, query), timeout=remaining_s
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"Trial exceeded max_case_wall_time_s={spec.max_case_wall_time_s:g}"
                    ) from exc
            context.log_event(
                "attempt.completed",
                parent_event_id=trial_event_id,
                operation_id=attempt_event_id,
                attempt=attempt + 1,
                max_attempts=spec.max_attempts_per_trial,
            )
            break
        except Exception as exc:
            last_exception = exc
            error = exception_record(exc)
            will_retry = attempt + 1 < spec.max_attempts_per_trial
            context.log_event(
                "attempt.failed",
                parent_event_id=trial_event_id,
                operation_id=attempt_event_id,
                attempt=attempt + 1,
                max_attempts=spec.max_attempts_per_trial,
                will_retry=will_retry,
                trial_index=spec.trial_index,
                **error,
            )
            if will_retry:
                retry_count += 1
                print(
                    f"  [{spec.case_order}/{spec.num_cases}{trial_tag}] retry "
                    f"attempt={attempt + 2}/{spec.max_attempts_per_trial} "
                    f"after {error['error_type']}: {error['error_message']}",
                    flush=True,
                )
        finally:
            context.current_attempt_event_id = None

    elapsed_s = time.monotonic() - started_monotonic_s
    if trajectory is None:
        assert error is not None and last_exception is not None
        model_calls = [normalize_model_call(value) for value in context.decisions]
        record = _failed_record(
            spec=spec,
            case_id=case_id,
            base_seed=base_seed,
            trial_seed=trial_seed,
            error=error,
            model_calls=model_calls,
            elapsed_s=elapsed_s,
        )
        print(
            f"  [{spec.case_order}/{spec.num_cases}{trial_tag}] error case_id={case_id} "
            f"{error['error_type']}: {error['error_message']}",
            flush=True,
        )
        return {"record": record, "retry_count": retry_count, "exception": last_exception}

    record = _completed_record(
        spec=spec,
        case_id=case_id,
        base_seed=base_seed,
        trial_seed=trial_seed,
        dynamic_topology=dynamic_topology,
        case_graph=case_graph,
        trajectory=trajectory,
        elapsed_s=elapsed_s,
    )
    print(
        f"  [{spec.case_order}/{spec.num_cases}{trial_tag}] msgs={record['message_count']} "
        f"input_pos={record['input_positions_total']} ans={record['final_output'][:60]!r}",
        flush=True,
    )
    return {"record": record, "retry_count": retry_count, "exception": None}
