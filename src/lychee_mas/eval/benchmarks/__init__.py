"""Benchmark registry for LycheeMAS evaluation.

Each loader returns records shaped as ``{task, kind, question, gold, context}``.
The entry points stay intentionally small: benchmark-specific loading and data
preparation live in sibling modules, while this file only exposes public
``load``/``prepare`` registries and registers benchmark classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.registry import REGISTRY

# Importing each module registers its BENCHMARK object. Registration is lazy
# with respect to datasets and heavyweight evaluation dependencies.
from . import (  # noqa: E402,F401
    aftraj,
    agent_collab,
    aime_2024,
    bbeh,
    choice_qa,
    gaia,
    gsm8k,
    hle,
    hle_verified,
    human_eval,
    livecodebench,
    locomo10,
    mast_data,
    open_agent_traces,
    swe_bench_verified,
    workbench,
)
from .base import (
    STANDARD_SOURCE_PROVIDERS,
    Benchmark,
    BenchmarkCase,
    BenchmarkEvaluationError,
    CaseMaterialization,
    EvaluationContext,
    ToolBundle,
)
from .common import DATA_BACKENDS, prepared_root, processed_root, raw_root
from .registry import BENCHMARKS, get_benchmark

RAW = raw_root()
PREPARED = prepared_root()
PROCESSED = processed_root()

REASONING_POLE = ("gsm8k", "aime_2024", "bbeh", "hle", "hle_verified")
FACT_POLE = ("medqa", "openbookqa", "arc_easy")
MEMORY_TASKS = ("locomo10",)
CODE_TASKS = ("human_eval", "livecodebench")
SOFTWARE_ENGINEERING_TASKS = ("swe_bench_verified",)
WORKPLACE_TOOL_TASKS = ("workbench",)
TOOL_TASKS = (
    "gaia_validation",
    "gaia_validation_level_1",
    "gaia_validation_level_2",
    "gaia_validation_level_3",
)
MAS_DIAGNOSTIC_TASKS = (
    "aftraj_audit",
    "aftraj_audit_test",
    "mast_failure",
    "open_agent_traces",
)
MAS_COLLAB_TASKS = (
    "agent_collab_idr",
    "agent_collab_rtd",
    "agent_collab_cpr",
    "agent_collab_clc",
)

FULL_PREPARE_TARGETS = tuple(benchmark.full_prepare_target for benchmark in BENCHMARKS.all())

BENCHMARK_STRUCTURE: list[dict[str, Any]] = [
    {
        "benchmark_source": "GSM8K",
        "full_prepare_target": "gsm8k",
        "prepare_targets": ["gsm8k"],
        "runnable_tasks": ["gsm8k"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "AIME 2024",
        "full_prepare_target": "aime_2024",
        "prepare_targets": ["aime_2024"],
        "runnable_tasks": ["aime_2024"],
        "kinds": ["aime"],
    },
    {
        "benchmark_source": "ARC-Easy",
        "full_prepare_target": "arc_easy",
        "prepare_targets": ["arc_easy"],
        "runnable_tasks": ["arc_easy"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "OpenBookQA",
        "full_prepare_target": "openbookqa",
        "prepare_targets": ["openbookqa"],
        "runnable_tasks": ["openbookqa"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "MedQA",
        "full_prepare_target": "medqa",
        "prepare_targets": ["medqa"],
        "runnable_tasks": ["medqa"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "LoCoMo10",
        "full_prepare_target": "locomo10",
        "prepare_targets": ["locomo10"],
        "runnable_tasks": ["locomo10"],
        "kinds": ["f1"],
    },
    {
        "benchmark_source": "HumanEval",
        "full_prepare_target": "human_eval",
        "prepare_targets": ["human_eval"],
        "runnable_tasks": ["human_eval"],
        "kinds": ["human_eval"],
    },
    {
        "benchmark_source": "GAIA",
        "full_prepare_target": "gaia",
        "prepare_targets": [
            "gaia",
            "gaia_validation",
            "gaia_validation_level_1",
            "gaia_validation_level_2",
            "gaia_validation_level_3",
        ],
        "runnable_tasks": [
            "gaia_validation",
            "gaia_validation_level_1",
            "gaia_validation_level_2",
            "gaia_validation_level_3",
        ],
        "kinds": ["gaia"],
    },
    {
        "benchmark_source": "AFTraj-2K",
        "full_prepare_target": "aftraj",
        "prepare_targets": ["aftraj", "aftraj_audit", "aftraj_audit_test"],
        "runnable_tasks": ["aftraj_audit", "aftraj_audit_test"],
        "kinds": ["mas_audit"],
    },
    {
        "benchmark_source": "AgentCollabBench",
        "full_prepare_target": "agent_collab",
        "prepare_targets": [
            "agent_collab",
            "agent_collab_idr",
            "agent_collab_rtd",
            "agent_collab_cpr",
            "agent_collab_clc",
        ],
        "runnable_tasks": [
            "agent_collab_idr",
            "agent_collab_rtd",
            "agent_collab_cpr",
            "agent_collab_clc",
        ],
        "kinds": [
            "mas_instruction_decay",
            "mas_tracer_durability",
            "mas_consensus_pollution",
            "mas_context_leakage",
        ],
    },
    {
        "benchmark_source": "MAST-Data",
        "full_prepare_target": "mast_data",
        "prepare_targets": ["mast_data", "mast_failure"],
        "runnable_tasks": ["mast_failure"],
        "kinds": ["mas_failure_taxonomy"],
    },
    {
        "benchmark_source": "Open Agent Traces",
        "full_prepare_target": "open_agent_traces",
        "prepare_targets": ["open_agent_traces"],
        "runnable_tasks": ["open_agent_traces"],
        "kinds": ["mas_deviation"],
    },
    {
        "benchmark_source": "BIG-Bench Extra Hard",
        "full_prepare_target": "bbeh",
        "prepare_targets": ["bbeh"],
        "runnable_tasks": ["bbeh"],
        "kinds": ["bbeh"],
    },
    {
        "benchmark_source": "Humanity's Last Exam",
        "full_prepare_target": "hle",
        "prepare_targets": ["hle"],
        "runnable_tasks": ["hle"],
        "kinds": ["hle"],
    },
    {
        "benchmark_source": "HLE-Verified Gold",
        "full_prepare_target": "hle_verified",
        "prepare_targets": ["hle_verified"],
        "runnable_tasks": ["hle_verified"],
        "kinds": ["hle_verified"],
    },
    {
        "benchmark_source": "LiveCodeBench Code Generation Lite",
        "full_prepare_target": "livecodebench",
        "prepare_targets": ["livecodebench"],
        "runnable_tasks": ["livecodebench"],
        "kinds": ["livecodebench"],
    },
    {
        "benchmark_source": "SWE-bench Verified",
        "full_prepare_target": "swe_bench_verified",
        "prepare_targets": ["swe_bench_verified"],
        "runnable_tasks": ["swe_bench_verified"],
        "kinds": ["swe_bench_verified"],
    },
    {
        "benchmark_source": "WorkBench Revisited",
        "full_prepare_target": "workbench",
        "prepare_targets": ["workbench"],
        "runnable_tasks": ["workbench"],
        "kinds": ["workbench"],
    },
]


BENCHMARK_MAPPINGS: dict[str, list[dict[str, Any]]] = {
    "GSM8K": [
        {"prepare_target": "gsm8k", "runnable_tasks": ["gsm8k"], "kinds": ["exact"]},
    ],
    "AIME 2024": [
        {"prepare_target": "aime_2024", "runnable_tasks": ["aime_2024"], "kinds": ["aime"]},
    ],
    "ARC-Easy": [
        {"prepare_target": "arc_easy", "runnable_tasks": ["arc_easy"], "kinds": ["mc"]},
    ],
    "OpenBookQA": [
        {"prepare_target": "openbookqa", "runnable_tasks": ["openbookqa"], "kinds": ["mc"]},
    ],
    "MedQA": [
        {"prepare_target": "medqa", "runnable_tasks": ["medqa"], "kinds": ["mc"]},
    ],
    "LoCoMo10": [
        {"prepare_target": "locomo10", "runnable_tasks": ["locomo10"], "kinds": ["f1"]},
    ],
    "HumanEval": [
        {"prepare_target": "human_eval", "runnable_tasks": ["human_eval"], "kinds": ["human_eval"]},
    ],
    "GAIA": [
        {
            "prepare_target": "gaia",
            "runnable_tasks": [],
            "kinds": [],
            "note": "full dataset prepare only",
        },
        {
            "prepare_target": "gaia_validation",
            "runnable_tasks": ["gaia_validation"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_1",
            "runnable_tasks": ["gaia_validation_level_1"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_2",
            "runnable_tasks": ["gaia_validation_level_2"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_3",
            "runnable_tasks": ["gaia_validation_level_3"],
            "kinds": ["gaia"],
        },
    ],
    "AFTraj-2K": [
        {
            "prepare_target": "aftraj",
            "runnable_tasks": ["aftraj_audit", "aftraj_audit_test"],
            "kinds": ["mas_audit"],
        },
        {
            "prepare_target": "aftraj_audit",
            "runnable_tasks": ["aftraj_audit"],
            "kinds": ["mas_audit"],
        },
        {
            "prepare_target": "aftraj_audit_test",
            "runnable_tasks": ["aftraj_audit_test"],
            "kinds": ["mas_audit"],
        },
    ],
    "AgentCollabBench": [
        {
            "prepare_target": "agent_collab",
            "runnable_tasks": [
                "agent_collab_idr",
                "agent_collab_rtd",
                "agent_collab_cpr",
                "agent_collab_clc",
            ],
            "kinds": [
                "mas_instruction_decay",
                "mas_tracer_durability",
                "mas_consensus_pollution",
                "mas_context_leakage",
            ],
        },
        {
            "prepare_target": "agent_collab_idr",
            "runnable_tasks": ["agent_collab_idr"],
            "kinds": ["mas_instruction_decay"],
        },
        {
            "prepare_target": "agent_collab_rtd",
            "runnable_tasks": ["agent_collab_rtd"],
            "kinds": ["mas_tracer_durability"],
        },
        {
            "prepare_target": "agent_collab_cpr",
            "runnable_tasks": ["agent_collab_cpr"],
            "kinds": ["mas_consensus_pollution"],
        },
        {
            "prepare_target": "agent_collab_clc",
            "runnable_tasks": ["agent_collab_clc"],
            "kinds": ["mas_context_leakage"],
        },
    ],
    "MAST-Data": [
        {
            "prepare_target": "mast_data",
            "runnable_tasks": ["mast_failure"],
            "kinds": ["mas_failure_taxonomy"],
        },
        {
            "prepare_target": "mast_failure",
            "runnable_tasks": ["mast_failure"],
            "kinds": ["mas_failure_taxonomy"],
        },
    ],
    "Open Agent Traces": [
        {
            "prepare_target": "open_agent_traces",
            "runnable_tasks": ["open_agent_traces"],
            "kinds": ["mas_deviation"],
        },
    ],
    "BIG-Bench Extra Hard": [
        {"prepare_target": "bbeh", "runnable_tasks": ["bbeh"], "kinds": ["bbeh"]},
    ],
    "Humanity's Last Exam": [
        {"prepare_target": "hle", "runnable_tasks": ["hle"], "kinds": ["hle"]},
    ],
    "HLE-Verified Gold": [
        {
            "prepare_target": "hle_verified",
            "runnable_tasks": ["hle_verified"],
            "kinds": ["hle_verified"],
        },
    ],
    "LiveCodeBench Code Generation Lite": [
        {
            "prepare_target": "livecodebench",
            "runnable_tasks": ["livecodebench"],
            "kinds": ["livecodebench"],
        },
    ],
    "SWE-bench Verified": [
        {
            "prepare_target": "swe_bench_verified",
            "runnable_tasks": ["swe_bench_verified"],
            "kinds": ["swe_bench_verified"],
        },
    ],
    "WorkBench Revisited": [
        {
            "prepare_target": "workbench",
            "runnable_tasks": ["workbench"],
            "kinds": ["workbench"],
        },
    ],
}


for _row in BENCHMARK_STRUCTURE:
    _row["mappings"] = BENCHMARK_MAPPINGS[_row["benchmark_source"]]


def _prepare_handler(target: str):
    def run(force: bool = False, source: Optional[str] = None) -> str:
        return BENCHMARKS.for_prepare_target(target).prepare(target, force=force, source=source)

    return run


SOURCE_PREPARERS = {
    target: _prepare_handler(target)
    for benchmark in BENCHMARKS.all()
    for target in benchmark.direct_prepare_targets
}
PREPARE_ALIASES = {
    alias: resolved
    for benchmark in BENCHMARKS.all()
    for alias, resolved in benchmark.prepare_aliases.items()
}
PREPARERS = {target: _prepare_handler(target) for target in BENCHMARKS.prepare_targets()}


def prepare(task: str, force: bool = False, source: Optional[str] = None) -> str:
    try:
        return PREPARERS[task](force, source)
    except KeyError as exc:
        available = ", ".join(sorted(PREPARERS))
        raise KeyError(f"No preparer for benchmark {task!r}; available: {available}") from exc


LOADERS = {
    task: (lambda n=None, task=task: BENCHMARKS.for_task(task).load(task, n=n))
    for task in BENCHMARKS.tasks()
}


def load(task: str, n: Optional[int] = None) -> List[Dict]:
    return get_benchmark(task).load(task, n=n)


def score(task: str, prediction: str, case: dict, *, record: dict | None = None) -> dict:
    """Score one prediction with the benchmark that owns ``task``."""

    return get_benchmark(task).score(prediction, case, record=record)


class _Benchmark:
    """Uniform registry adapter: construction is lazy and does not touch data."""

    task: str = ""

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def load(self, n: Optional[int] = None) -> List[Dict]:
        return load(self.task, n=self.n if n is None else n)


def _register_benchmarks() -> None:
    for _task in LOADERS:
        cls = type(f"Benchmark_{_task}", (_Benchmark,), {"task": _task, "name": _task})
        REGISTRY.register("benchmark", _task)(cls)


_register_benchmarks()

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "BenchmarkCase",
    "BenchmarkEvaluationError",
    "CaseMaterialization",
    "EvaluationContext",
    "STANDARD_SOURCE_PROVIDERS",
    "ToolBundle",
    "get_benchmark",
    "load",
    "prepare",
    "score",
    "PREPARERS",
    "SOURCE_PREPARERS",
    "PREPARE_ALIASES",
    "LOADERS",
    "FULL_PREPARE_TARGETS",
    "BENCHMARK_STRUCTURE",
    "BENCHMARK_MAPPINGS",
    "DATA_BACKENDS",
    "RAW",
    "PREPARED",
    "PROCESSED",
    "REASONING_POLE",
    "FACT_POLE",
    "MEMORY_TASKS",
    "CODE_TASKS",
    "SOFTWARE_ENGINEERING_TASKS",
    "WORKPLACE_TOOL_TASKS",
    "TOOL_TASKS",
    "MAS_DIAGNOSTIC_TASKS",
    "MAS_COLLAB_TASKS",
]
