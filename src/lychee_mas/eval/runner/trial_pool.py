"""Execute admitted Trial work with Runner-local concurrency and stop control."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from .run_state import (
    atomic_write_json,
    graceful_stop_request,
    scheduler_trial_admission_total,
    scheduler_trial_concurrency,
)
from .trial_executor import TrialExecutionSpec, execute_trial
from .trial_identity import case_id_from_item, dataset_index_from_item

STOP_AFTER_ACTIVE_TRIALS = "stop_after_active_trials"


@dataclass(slots=True)
class TrialPoolRequest:
    """Mutable Run state and immutable policy required by one Trial pool."""

    work_items: list[tuple[int, dict[str, Any], int]]
    initial_worker: tuple[Any, Any, Any, Any]
    worker_factory: Callable[[], tuple[Any, Any, Any, Any]]
    worker_capacity: int
    run_status: dict[str, Any]
    run_status_path: str
    run_control_path: str
    run_event_id: str
    event_writer: Any
    concurrency_policy: dict[str, Any]
    initial_scheduler_trial_concurrency: int | None
    initial_scheduler_trial_admission_total: int | None
    scheduler_controls_concurrency: bool
    data_count: int
    start_index: int
    trials_per_case: int
    base_seed: int
    benchmark_id: str
    task: str
    kind: str
    method: str
    profile: str
    max_rounds: int
    max_case_wall_time_s: float | None
    max_attempts_per_trial: int
    on_trial_error: str
    successful_trials_by_case: dict[str, set[int]]
    trial_records: list[dict[str, Any]]
    base_graph: Any
    vllm_metrics_monitor: Any | None = None


@dataclass(frozen=True, slots=True)
class TrialPoolResult:
    """Terminal state produced by a normally drained Trial pool."""

    graceful_stop_requested: bool
    effective_worker_count: int


async def execute_trial_pool(request: TrialPoolRequest) -> TrialPoolResult:
    """Run work items while honoring Scheduler admissions and graceful stop."""

    from lychee_mas.runtime.execution.concurrency import TrialConcurrencyController
    from lychee_mas.runtime.execution.seeding import derive_trial_seed

    workers = [request.initial_worker]
    for _ in range(1, min(request.worker_capacity, max(1, len(request.work_items)))):
        worker = request.worker_factory()
        worker[2].event_writer = request.event_writer
        worker[2].current_run_event_id = request.run_event_id
        workers.append(worker)
    request.run_status["effective_trial_concurrency"] = len(workers)
    atomic_write_json(request.run_status_path, request.run_status)

    queue: asyncio.Queue = asyncio.Queue()
    for work_item in request.work_items:
        queue.put_nowait(work_item)
    for _ in workers:
        queue.put_nowait(None)

    output_lock = asyncio.Lock()
    graceful_stop = asyncio.Event()
    controller = TrialConcurrencyController(
        request.concurrency_policy,
        external_limit=request.initial_scheduler_trial_concurrency,
        external_authoritative=request.scheduler_controls_concurrency,
    )
    request.run_status["current_trial_concurrency"] = controller.effective_target
    request.run_status["adaptive_trial_concurrency_target"] = (
        None if request.scheduler_controls_concurrency else controller.target
    )
    request.run_status["concurrency_history"] = list(controller.history)
    admission_total = int(request.initial_scheduler_trial_admission_total or 0)
    consumed_admissions = 0
    admission_condition = asyncio.Condition()

    async def observe_graceful_stop() -> bool:
        stop = graceful_stop_request(request.run_control_path)
        if not stop:
            return False
        if graceful_stop.is_set():
            return True
        async with output_lock:
            if graceful_stop.is_set():
                return True
            graceful_stop.set()
            status = request.run_status
            status["status"] = "draining"
            status["accepting_new_trials"] = False
            status["stop_requested_at_utc"] = stop.get("requested_at_utc")
            status["stop_requested_by"] = stop.get("requested_by")
            status["stop_reason"] = STOP_AFTER_ACTIVE_TRIALS
            status["active_trials_at_stop_request"] = len(status.get("active_cases") or [])
            status["updated_at_unix_s"] = round(time.time(), 6)
            atomic_write_json(request.run_status_path, status)
            request.event_writer.log_event(
                "run.stop_requested",
                parent_event_id=request.run_event_id,
                action=STOP_AFTER_ACTIVE_TRIALS,
                requested_at_utc=stop.get("requested_at_utc"),
                requested_by=stop.get("requested_by"),
                active_trials=status["active_trials_at_stop_request"],
            )
            print(
                "[control] graceful stop requested; no new Trial will start, "
                f"active_trials={status['active_trials_at_stop_request']}",
                flush=True,
            )
        return True

    async def observe_scheduler_allocation() -> dict[str, Any] | None:
        nonlocal admission_total

        requested = scheduler_trial_concurrency(
            request.run_control_path,
            fallback=request.initial_scheduler_trial_concurrency,
        )
        requested_total = scheduler_trial_admission_total(
            request.run_control_path,
            fallback=request.initial_scheduler_trial_admission_total,
        )
        event = await controller.set_external_limit(requested)
        if requested_total is not None and int(requested_total) > admission_total:
            admission_total = int(requested_total)
            admission_changed = True
            async with admission_condition:
                admission_condition.notify_all()
        else:
            admission_changed = False
        if event is None and not admission_changed:
            return None

        async with output_lock:
            status = request.run_status
            status["scheduler_trial_concurrency"] = requested
            status["scheduler_trial_admission_total"] = admission_total
            status["current_trial_concurrency"] = controller.effective_target
            status["adaptive_trial_concurrency_target"] = (
                None if request.scheduler_controls_concurrency else controller.target
            )
            status["concurrency_history"] = list(controller.history)
            status["updated_at_unix_s"] = round(time.time(), 6)
            atomic_write_json(request.run_status_path, status)
            if event is not None:
                request.event_writer.log_event(
                    "concurrency.changed",
                    parent_event_id=request.run_event_id,
                    reason=event["reason"],
                    previous=event["previous"],
                    target=event["target"],
                    adaptive_target=event.get("adaptive_target"),
                    external_limit=event.get("external_limit"),
                    policy=request.concurrency_policy,
                )
                print(
                    "[concurrency] "
                    f"reason={event['reason']} target={event['previous']}->{event['target']} "
                    f"external_limit={event.get('external_limit')}",
                    flush=True,
                )
            if admission_changed:
                request.event_writer.log_event(
                    "trial.admission_updated",
                    parent_event_id=request.run_event_id,
                    admission_total=admission_total,
                    consumed_total=consumed_admissions,
                )
        return event

    async def control_loop() -> None:
        while not graceful_stop.is_set():
            await observe_scheduler_allocation()
            if await observe_graceful_stop():
                return
            await asyncio.sleep(1.0)

    async def acquire_trial_admission() -> bool:
        nonlocal consumed_admissions

        if not request.scheduler_controls_concurrency:
            return True
        while not graceful_stop.is_set():
            if await observe_graceful_stop():
                return False
            async with admission_condition:
                if consumed_admissions < admission_total:
                    consumed_admissions += 1
                    async with output_lock:
                        status = request.run_status
                        status["segment_started_trials"] = consumed_admissions
                        status["scheduler_trial_admission_total"] = admission_total
                        status["updated_at_unix_s"] = round(time.time(), 6)
                        atomic_write_json(request.run_status_path, status)
                    return True
                try:
                    await asyncio.wait_for(admission_condition.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        return False

    async def acquire_trial_capacity() -> bool:
        while not graceful_stop.is_set():
            if await observe_graceful_stop():
                return False
            try:
                await asyncio.wait_for(controller.acquire(), timeout=1.0)
                if await acquire_trial_admission():
                    return True
                await controller.release()
                return False
            except asyncio.TimeoutError:
                continue
        return False

    async def worker_loop(worker_id: int, worker) -> None:
        memory, _router, context, runtime = worker
        while True:
            if await observe_graceful_stop():
                return
            work_item = await queue.get()
            if work_item is None:
                queue.task_done()
                return
            context.current_worker_id = worker_id
            if not await acquire_trial_capacity():
                queue.task_done()
                return

            case_offset, item, trial_index = work_item
            dataset_index = dataset_index_from_item(item, request.start_index + case_offset)
            case_id = case_id_from_item(item, dataset_index)
            async with output_lock:
                active_cases = {
                    int(active["worker_id"]): active
                    for active in request.run_status.get("active_cases") or []
                }
                active_cases[worker_id] = {
                    "worker_id": worker_id,
                    "case_order": case_offset + 1,
                    "num_cases": request.data_count,
                    "case_id": case_id,
                    "trial_index": trial_index,
                    "trial_seed": derive_trial_seed(
                        base_seed=request.base_seed,
                        benchmark_id=request.benchmark_id,
                        task=request.task,
                        case_id=case_id,
                        trial_index=trial_index,
                    ),
                    "trials_per_case": request.trials_per_case,
                    "started_at_unix_s": round(time.time(), 6),
                }
                request.run_status["active_cases"] = list(active_cases.values())
                request.run_status["updated_at_unix_s"] = round(time.time(), 6)
                atomic_write_json(request.run_status_path, request.run_status)

            try:
                result = await execute_trial(
                    spec=TrialExecutionSpec(
                        case_order=case_offset + 1,
                        num_cases=request.data_count,
                        item=item,
                        dataset_index=dataset_index,
                        trial_index=trial_index,
                        trials_per_case=request.trials_per_case,
                        task=request.task,
                        kind=request.kind,
                        benchmark_id=request.benchmark_id,
                        method=request.method,
                        profile=request.profile,
                        max_rounds=request.max_rounds,
                        max_case_wall_time_s=request.max_case_wall_time_s,
                        max_attempts_per_trial=request.max_attempts_per_trial,
                    ),
                    memory=memory,
                    context=context,
                    runtime=runtime,
                    base_graph=request.base_graph,
                )
            finally:
                await controller.release()

            record = result["record"]
            concurrency_event = await controller.observe(
                record=record,
                service_metrics=(
                    request.vllm_metrics_monitor.latest_snapshots()
                    if request.vllm_metrics_monitor is not None
                    else []
                ),
            )
            async with output_lock:
                status = request.run_status
                status["active_cases"] = [
                    active
                    for active in status.get("active_cases") or []
                    if int(active.get("worker_id", -1)) != worker_id
                ]
                request.trial_records.append(record)
                request.event_writer.record_trial(
                    {
                        **record,
                        "operation_id": context.current_trial_event_id,
                        "parent_event_id": context.current_run_event_id,
                    }
                )
                status["retry_attempt_count"] += int(result["retry_count"])
                if record.get("status") == "failed":
                    status["failed_trials"] += 1
                    status["last_error"] = {
                        "case_id": record["case_id"],
                        "trial_index": record.get("trial_index", 0),
                        "error_type": record.get("error_type"),
                        "error_message": record.get("error_message"),
                    }
                else:
                    status["successful_trials"] += 1
                    request.successful_trials_by_case.setdefault(record["case_id"], set()).add(
                        int(record.get("trial_index", 0) or 0)
                    )
                completed_cases = sum(
                    len(indices) >= request.trials_per_case
                    for indices in request.successful_trials_by_case.values()
                )
                status["completed_distinct_cases"] = completed_cases
                status["remaining_distinct_cases"] = max(
                    0, request.data_count - completed_cases
                )
                status["last_completed_case"] = {
                    "worker_id": worker_id,
                    "case_order": case_offset + 1,
                    "num_cases": request.data_count,
                    "case_id": record["case_id"],
                    "trial_index": record.get("trial_index", 0),
                    "status": record.get("status", "ok"),
                    "score": record.get("score"),
                    "finished_at_unix_s": round(time.time(), 6),
                }
                status["current_trial_concurrency"] = controller.effective_target
                status["adaptive_trial_concurrency_target"] = (
                    None if request.scheduler_controls_concurrency else controller.target
                )
                if concurrency_event is not None:
                    status["concurrency_history"] = list(controller.history)
                    context.log_event(
                        "concurrency.changed",
                        parent_event_id=context.current_run_event_id,
                        reason=concurrency_event["reason"],
                        previous=concurrency_event["previous"],
                        target=concurrency_event["target"],
                        policy=request.concurrency_policy,
                    )
                    print(
                        "[concurrency] "
                        f"reason={concurrency_event['reason']} "
                        f"target={concurrency_event['previous']}->{concurrency_event['target']}",
                        flush=True,
                    )
                status["updated_at_unix_s"] = round(time.time(), 6)
                if record.get("status") == "failed" and request.on_trial_error == "fail-fast":
                    status["status"] = "failed_fast"
                atomic_write_json(request.run_status_path, status)
            queue.task_done()
            if record.get("status") == "failed" and request.on_trial_error == "fail-fast":
                raise result["exception"]
            if record.get("status") == "failed":
                # A cancelled asyncio.to_thread() call cannot terminate its underlying
                # thread. Never reuse mutable Runtime/Context state after a failed Trial:
                # a late provider response must remain attached to the old Trial instead
                # of being attributed to the next case handled by this worker slot.
                memory, _router, context, runtime = request.worker_factory()
                context.event_writer = request.event_writer
                context.current_run_event_id = request.run_event_id
                context.current_worker_id = worker_id

    control_task = asyncio.create_task(control_loop(), name="run-control-loop")
    try:
        await asyncio.gather(
            *(worker_loop(worker_id, worker) for worker_id, worker in enumerate(workers))
        )
    finally:
        control_task.cancel()
        with suppress(asyncio.CancelledError):
            await control_task
    return TrialPoolResult(
        graceful_stop_requested=graceful_stop.is_set(),
        effective_worker_count=len(workers),
    )
