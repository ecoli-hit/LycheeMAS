"""Reusable services behind the benchmark run CLI."""

from .backend_assembly import BackendAssembly, BackendAssemblyRequest, assemble_backend
from .cli import build_run_parser
from .configuration import (
    apply_benchmark_contract,
    as_bool,
    get_config_value,
    load_yaml,
    optional_positive_limit,
    resolve_prefix_length,
)
from .run_artifacts import model_tag, prepare_trials_for_resume
from .run_state import (
    atomic_write_json,
    graceful_stop_request,
    read_json,
    read_jsonl,
    run_status_elapsed_seconds,
    scheduler_trial_admission_total,
    scheduler_trial_concurrency,
)
from .team_inspection import print_team_structure
from .trial_executor import TrialExecutionSpec, execute_trial
from .trial_identity import case_id_from_item, dataset_index_from_item
from .trial_pool import TrialPoolRequest, TrialPoolResult, execute_trial_pool

__all__ = [
    "apply_benchmark_contract",
    "assemble_backend",
    "as_bool",
    "atomic_write_json",
    "BackendAssembly",
    "BackendAssemblyRequest",
    "build_run_parser",
    "get_config_value",
    "graceful_stop_request",
    "load_yaml",
    "model_tag",
    "optional_positive_limit",
    "read_json",
    "read_jsonl",
    "prepare_trials_for_resume",
    "print_team_structure",
    "resolve_prefix_length",
    "run_status_elapsed_seconds",
    "scheduler_trial_admission_total",
    "scheduler_trial_concurrency",
    "TrialExecutionSpec",
    "TrialPoolRequest",
    "TrialPoolResult",
    "case_id_from_item",
    "dataset_index_from_item",
    "execute_trial",
    "execute_trial_pool",
]
