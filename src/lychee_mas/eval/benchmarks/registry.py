"""In-memory lookup for registered benchmark implementations."""

from __future__ import annotations

from typing import Any, Mapping

from .base import Benchmark


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._benchmarks: dict[str, Benchmark] = {}
        self._tasks: dict[str, Benchmark] = {}
        self._prepare_targets: dict[str, Benchmark] = {}

    def register(self, benchmark: Benchmark) -> Benchmark:
        if benchmark.id in self._benchmarks:
            raise ValueError(f"duplicate benchmark {benchmark.id!r}")
        for task in benchmark.runnable_tasks:
            if task in self._tasks:
                raise ValueError(f"duplicate runnable benchmark task {task!r}")
        for target in benchmark.accepted_prepare_targets:
            if target in self._prepare_targets:
                raise ValueError(f"duplicate benchmark prepare target {target!r}")
        self._benchmarks[benchmark.id] = benchmark
        self._tasks.update({task: benchmark for task in benchmark.runnable_tasks})
        self._prepare_targets.update(
            {target: benchmark for target in benchmark.accepted_prepare_targets}
        )
        return benchmark

    def get(self, benchmark_id: str) -> Benchmark:
        try:
            return self._benchmarks[benchmark_id]
        except KeyError as exc:
            available = ", ".join(self._benchmarks)
            raise KeyError(f"unknown benchmark {benchmark_id!r}; available: {available}") from exc

    def for_task(self, task: str) -> Benchmark:
        try:
            return self._tasks[task]
        except KeyError as exc:
            available = ", ".join(self._tasks)
            raise KeyError(f"unknown benchmark task {task!r}; available: {available}") from exc

    def for_prepare_target(self, target: str) -> Benchmark:
        try:
            return self._prepare_targets[target]
        except KeyError as exc:
            available = ", ".join(self._prepare_targets)
            raise KeyError(
                f"unknown benchmark prepare target {target!r}; available: {available}"
            ) from exc

    def for_query(self, task: str, metadata: Mapping[str, Any] | None) -> Benchmark:
        metadata = metadata or {}
        benchmark_id = str(metadata.get("benchmark_id") or "")
        return self.get(benchmark_id) if benchmark_id else self.for_task(task)

    def all(self) -> tuple[Benchmark, ...]:
        return tuple(self._benchmarks.values())

    def tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def prepare_targets(self) -> tuple[str, ...]:
        return tuple(self._prepare_targets)

    def add_analysis_arguments(self, parser: Any) -> None:
        for benchmark in self.all():
            benchmark.add_analysis_arguments(parser)


BENCHMARKS = BenchmarkRegistry()


def register_benchmark(benchmark: Benchmark) -> Benchmark:
    return BENCHMARKS.register(benchmark)


def get_benchmark(name: str) -> Benchmark:
    """Return a benchmark by benchmark ID or runnable task."""

    try:
        return BENCHMARKS.get(name)
    except KeyError:
        return BENCHMARKS.for_task(name)
