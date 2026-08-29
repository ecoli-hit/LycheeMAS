#!/usr/bin/env python3
"""Materialize controlled benchmark/team/framework experiment matrices."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lychee_mas.eval.teams.contracts import model_resource_requirements


@dataclass(frozen=True)
class BenchmarkPlan:
    spec_id: str
    task: str
    instance_id: str
    strata_field: str
    max_turns: int
    max_model_calls: int
    max_case_wall_time_s: int
    evidence_candidate_topology: str
    evidence_rationale: str
    full_case_count: int
    scoring_profile: str = "official"
    docker_image: str | None = None
    proxy: bool = False


BENCHMARKS = {
    "bbeh": BenchmarkPlan(
        spec_id="bbeh",
        task="bbeh",
        instance_id="bbeh-scan-8591e166",
        strata_field="bbeh_task",
        max_turns=12,
        max_model_calls=32,
        # BBEH multi-agent recipes may spend several long reasoning calls behind
        # a shared serving queue. Keep one framework-neutral wall-time budget so
        # timeout policy does not become an unreported framework variable.
        max_case_wall_time_s=5400,
        evidence_candidate_topology="sequential",
        evidence_rationale=(
            "EvoMAS directly evaluates BBEH with single, debate, majority-vote, "
            "peer-review, CROTO, and SMOA pools. This three-node sequential team is "
            "a reduced controlled adaptation of the staged solve-review pattern, "
            "not an EvoMAS reproduction."
        ),
        full_case_count=4520,
    ),
    "gaia": BenchmarkPlan(
        spec_id="gaia",
        task="gaia_validation",
        instance_id="gaia-scan-a218b61c",
        strata_field="level",
        max_turns=20,
        max_model_calls=128,
        max_case_wall_time_s=5400,
        evidence_candidate_topology="centralized",
        evidence_rationale=(
            "Magentic-One, CORAL, COLA and Anemoi provide GAIA evidence for "
            "central or semi-central orchestration; this portable team is inspired "
            "by those systems rather than a reproduction."
        ),
        full_case_count=165,
        docker_image="lychee-agbench-gaia:local",
        proxy=True,
    ),
    "swe-bench-verified": BenchmarkPlan(
        spec_id="swe_bench_verified",
        task="swe_bench_verified",
        instance_id="swe_bench_verified-scan-38388f5e",
        strata_field="repo",
        # A v23 trace reached only the 15th selector decision at the three-hour
        # Trial boundary. Keep the legal GroupChat path inside that wall budget
        # instead of advertising turns that cannot normally complete.
        max_turns=14,
        # Fourteen activations may each use a bounded 30-step tool loop. Reserve
        # additional calls for model-based selection and final submission.
        max_model_calls=448,
        max_case_wall_time_s=10800,
        evidence_candidate_topology="centralized",
        evidence_rationale=(
            "BOAD and related software-engineering agents motivate orchestrated "
            "localization, editing and validation roles; this is a controlled adaptation."
        ),
        full_case_count=500,
        docker_image="lychee-python-sandbox:local",
        proxy=True,
    ),
    "workbench": BenchmarkPlan(
        spec_id="workbench",
        task="workbench",
        instance_id="workbench-scan-2cdb49c0",
        strata_field="domain",
        max_turns=16,
        max_model_calls=64,
        # WorkBench can involve several long tool-backed turns under a shared
        # serving queue. Use the same framework-neutral Trial wall budget as the
        # other long-horizon controlled recipes.
        max_case_wall_time_s=5400,
        evidence_candidate_topology="decentralized",
        evidence_rationale=(
            "EvoMAS publishes WorkBench single/debate/majority-vote/peer-review/CROTO "
            "pools, while the Nature agent-scaling study directly compares SAS and "
            "four MAS architectures and observes its strongest WorkBench gain for a "
            "peer-coordination condition. This handoff team remains a controlled "
            "adaptation, not either paper's reproduction."
        ),
        full_case_count=690,
    ),
    "livecodebench": BenchmarkPlan(
        spec_id="livecodebench",
        task="livecodebench",
        instance_id="",
        strata_field="difficulty",
        # Fourteen turns can stop immediately after Implementer hands control to
        # Reviewer, before the approved submitter gets its final activation.
        max_turns=16,
        max_model_calls=96,
        max_case_wall_time_s=3600,
        evidence_candidate_topology="sequential",
        evidence_rationale=(
            "LiveCodeBench is officially a code-generation benchmark rather than a "
            "prescribed MAS recipe. The four teams are controlled LycheeMAS "
            "adaptations used to compare framework and coordination effects while "
            "retaining the pinned official evaluator."
        ),
        full_case_count=175,
        scoring_profile="official_release_v6",
        docker_image="lychee-python-sandbox:local",
        proxy=True,
    ),
    "hle-verified": BenchmarkPlan(
        spec_id="hle_verified",
        task="hle_verified",
        instance_id="",
        strata_field="category",
        max_turns=16,
        max_model_calls=96,
        max_case_wall_time_s=3600,
        evidence_candidate_topology="centralized",
        evidence_rationale=(
            "HLE-Verified defines a difficult multimodal QA and judge contract but "
            "does not prescribe one MAS topology. The four teams are controlled "
            "LycheeMAS adaptations over the public Gold subset."
        ),
        full_case_count=668,
        scoring_profile="official_gold",
    ),
}
TOPOLOGIES = ("independent", "sequential", "centralized", "decentralized")
MAGENTIC_ONE_RECIPE = "magentic-one"
FRAMEWORKS = ("autogen", "langgraph", "crewai")
DEPLOYMENT_INSTANCE_ID = "qwen35-9b-vllm-gpu47-cap128"
BENCHMARK_ORDER = tuple(BENCHMARKS)

METHOD_SEMANTICS = {
    # This legacy matrix key denotes one reasoning locus. It must not be confused
    # with the parallel multi-agent "Independent" ensemble used by some papers.
    "independent": "single_agent_baseline",
    "sequential": "fixed_order_pipeline",
    "centralized": "model_routed_central_control",
    "decentralized": "peer_handoff_without_persistent_controller",
    "magentic-one": "task_and_progress_ledger_orchestration",
}

FRAMEWORK_EXECUTION_CONTRACTS = {
    "autogen": {
        "runtime_class": "AutoGenRuntime",
        "native_kernel": "AutoGen AgentChat GroupChat/Swarm/GraphFlow",
        "execution_owner": (
            "AutoGen advances its native team loop; LycheeMAS supplies the model "
            "client, portable resources, EventLog, result projection, and scorer."
        ),
    },
    "langgraph": {
        "runtime_class": "LangGraphRuntime",
        "native_kernel": "LangGraph StateGraph",
        "execution_owner": (
            "LangGraph advances the compiled graph; LycheeMAS implements graph "
            "nodes, routing functions, model/tool calls, EventLog, and scoring."
        ),
    },
    "crewai": {
        "runtime_class": "CrewAIRuntime",
        "native_kernel": "CrewAI Crew/Process or Flow",
        "execution_owner": (
            "CrewAI advances native Crew task lifecycles and Flow execution; "
            "LycheeMAS injects compiled operation handlers and portable services. "
            "The Binding Report records whether the resulting mapping is exact, "
            "composed, or approximated."
        ),
    },
}

KNOWN_OFFICIAL_RECIPES = {
    ("gaia", "autogen"): (
        "AutoGen Magentic-One / AGBench GAIA is represented by the additional "
        "native-replication recipe; the four topology recipes remain the controlled "
        "cross-framework portability track."
    ),
}


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
    return json.loads(text) if text else None


def _resolve_benchmark_instance_id(base_url: str, plan: BenchmarkPlan) -> str:
    """Resolve the current prepared BenchmarkInstance for one BenchmarkSpec."""

    if plan.instance_id:
        return plan.instance_id
    payload = _request(base_url, "GET", "/api/benchmark-instances")
    candidates = [
        item
        for item in payload
        if str(item.get("benchmark_spec_id") or "") == plan.spec_id
        and bool(item.get("available", True))
    ]
    if not candidates:
        raise RuntimeError(
            f"no ready BenchmarkInstance found for BenchmarkSpec {plan.spec_id!r}; "
            "scan the prepared benchmark root in Resource Center first"
        )
    candidates.sort(
        key=lambda item: (
            str(item.get("checked_at_utc") or item.get("created_at_utc") or ""),
            str(item.get("id") or ""),
        ),
        reverse=True,
    )
    return str(candidates[0]["id"])


def _case_target_reached(payload: dict[str, Any]) -> bool:
    """Return whether an ExperimentInstance progress response reached its target."""

    progress = payload.get("progress")
    if not isinstance(progress, dict):
        progress = payload
    remaining = progress.get("remaining_to_case_target")
    return remaining is not None and int(remaining) <= 0


def _smoke_requires_rebuild(payload: dict[str, Any], *, target: int) -> bool:
    """Reject terminal smoke Instances that did not complete a clean target."""

    status = str(payload.get("status") or "")
    if status in {"running", "starting", "draining"}:
        return False
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        progress = {}
    if status == "queued" and not payload.get("launch_dir") and not progress:
        return False
    return bool(
        int(progress.get("failed_trials") or 0) > 0
        or int(progress.get("completed_distinct_cases") or 0) < int(target)
    )


def _priority(benchmark: str, topology: str) -> int:
    if benchmark == "gaia" and topology == MAGENTIC_ONE_RECIPE:
        return 0
    benchmark_offset = BENCHMARK_ORDER.index(benchmark)
    evidence_candidate = BENCHMARKS[benchmark].evidence_candidate_topology
    if topology == evidence_candidate:
        return 10 + benchmark_offset
    if topology == "independent":
        return 20 + benchmark_offset
    remaining = [item for item in TOPOLOGIES if item not in {evidence_candidate, "independent"}]
    return 30 + benchmark_offset * 2 + remaining.index(topology)


def _experiment_spec(benchmark: str, topology: str) -> dict[str, Any]:
    plan = BENCHMARKS[benchmark]
    independent = topology == "independent"
    team_spec_id = "gaia" if topology == MAGENTIC_ONE_RECIPE else f"{benchmark}-{topology}"
    proxy_targets = {
        "downloads": False,
        "web_surfer": plan.proxy,
        "code_executor": plan.proxy,
        "model_backend": False,
    }
    return {
        "schema_version": 3,
        "id": f"{benchmark}-{topology}-qwen35",
        "benchmark": {
            "benchmark_spec_id": plan.spec_id,
            "runnable_task": plan.task,
            "cases": "all",
            "start_index": 0,
            "case_selection": "stratified",
            "case_strata_field": plan.strata_field,
            "scoring_profile": plan.scoring_profile,
        },
        "team_spec_id": team_spec_id,
        "runtime": {
            "method": "none",
            "trials_per_case": 1,
            "max_rounds": 4,
            "max_turns": 1 if independent else plan.max_turns,
            "max_model_calls_per_case": plan.max_model_calls,
            "max_case_wall_time_s": plan.max_case_wall_time_s,
            "max_attempts_per_trial": 1,
            "on_trial_error": "continue",
            "max_new_tokens": 32768,
            "max_input_tokens": None,
            "min_output_reserve_tokens": 2048,
            "min_thinking_reserve_tokens": 0,
            # Keep the system-comparison matrix controlled: every model-backed node,
            # including framework controllers, receives the same bounded thinking budget.
            "max_thinking_budget_tokens": 8192,
            "min_final_reserve_tokens": 1024,
            "safety_margin_tokens": 256,
            "trial_concurrency": 32,
            "concurrency_policy": {
                "mode": "auto",
                "initial": 8,
                "minimum": 1,
                "maximum": 32,
                "increase_step": 2,
                "decrease_factor": 0.5,
                "control_window_trials": 4,
            },
            "seed": 0,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "code_executor": "docker",
            "code_timeout": 60,
            "work_root": "runs/lychee_tool_workspaces",
            "web_headless": True,
            "save_screenshots": False,
            "docker_image": plan.docker_image,
            "trace_model_calls": True,
        },
        "observability": {
            "collect_vllm_metrics": True,
            "vllm_metrics_interval_s": 5.0,
            "event_log_max_events_per_file": 10000,
            "event_log_max_mib_per_file": 64.0,
        },
        "evaluation": {
            "external_evaluator": "run",
            "profile_id": "core",
            **(
                {
                    "hle_judge": {
                        "model": "Qwen3.5-9B",
                        "base_url": "http://127.0.0.1:6200/v1",
                        "auth_mode": "none",
                        "api_key_env": "VLLM_API_KEY",
                        "workers": 8,
                        "timeout_s": 3600.0,
                        "max_tokens": 4096,
                        "thinking_mode": "disabled",
                        "output_mode": "local_json_object",
                        "max_attempts": 3,
                    }
                }
                if benchmark == "hle-verified"
                else {}
            ),
        },
        "network": {
            "mode": "proxy" if plan.proxy else "direct",
            "proxy_url": "http://127.0.0.1:7897" if plan.proxy else "",
            "no_proxy": "127.0.0.1,localhost,::1",
            "targets": proxy_targets,
            "docker_bridge_host": "172.17.0.1",
            "container_proxy_port": 17897,
            # Bing is also the default search surface used by WebSurfer and is
            # reachable on the target server. A Google-only probe can reject a
            # usable DIRECT/Mihomo path before the benchmark starts.
            "probe_url": (
                "https://www.bing.com/" if plan.proxy else "https://www.google.com/generate_204"
            ),
        },
        "environment": {
            "repo_root": "/data/bk/LycheeMAS",
            "python": "/data/bk/LycheeMAS/.venv/bin/python",
            "conda_sh": "",
            "conda_env": "",
            "raw_root": "/data/bk/LycheeMAS/data/benchmarks/raw",
            "prepared_root": "/data/bk/LycheeMAS/data/benchmarks/prepared",
            "models_root": "/data/bk/LycheeMAS/models",
            "runs_root": "/data/bk/LycheeMAS/runs/benchmarks",
        },
        "notes": (
            "Controlled portability track, not an official framework or paper recipe. "
            "Full benchmark scope; staged execution is controlled only by the "
            "ExperimentInstance Case target. Evidence-candidate rationale: "
            + plan.evidence_rationale
        ),
    }


def _instance_id(
    benchmark: str,
    topology: str,
    framework: str,
    matrix_id: str,
) -> str:
    return f"{benchmark}-{topology}-qwen35-{framework}-{matrix_id}-run"


def _team_instance_id(
    benchmark: str,
    topology: str,
    framework: str,
    matrix_id: str,
) -> str:
    """Identify the immutable TeamSpec/framework/deployment binding for one matrix."""

    return f"{benchmark}-{topology}-qwen35-{framework}-{matrix_id}"


def _launch_conflicts(instance: dict[str, Any]) -> bool:
    """Return whether a canonical launch directory belongs to an older Run."""

    launch_dir = Path(
        str(instance.get("launch_dir") or "")
        or f"runs/eval_studio/launches/{instance['id']}"
    )
    job_path = launch_dir / "job.json"
    if not job_path.is_file():
        return False
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return str(job.get("run_dir") or "") != str(instance.get("run_dir") or "")


def _archive_launch(instance: dict[str, Any]) -> Path | None:
    """Preserve an immutable old launch before reusing its stable matrix ID."""

    launch_dir = Path(
        str(instance.get("launch_dir") or "")
        or f"runs/eval_studio/launches/{instance['id']}"
    )
    if not launch_dir.exists():
        return None
    archive_root = Path("runs/eval_studio/archived_launches")
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archive_root / f"{instance['id']}-{stamp}"
    launch_dir.rename(destination)
    return destination


def _validate_registry_ids(
    recipes: tuple[tuple[str, str], ...],
    frameworks: tuple[str, ...],
    matrix_id: str,
) -> None:
    """Fail before mutating registries when a generated ID exceeds the contract."""

    invalid: list[str] = []
    for benchmark, topology in recipes:
        for framework in frameworks:
            for value in (
                _team_instance_id(benchmark, topology, framework, matrix_id),
                _instance_id(benchmark, topology, framework, matrix_id),
            ):
                if len(value) > 64:
                    invalid.append(value)
    if invalid:
        raise SystemExit(
            "generated registry IDs exceed the 64-character limit; choose a shorter "
            "--matrix-id:\n" + "\n".join(f"  {value}" for value in invalid)
        )


def _selected_recipes(
    benchmarks: tuple[str, ...],
    topologies: tuple[str, ...],
    *,
    include_gaia_magentic_one: bool,
) -> tuple[tuple[str, str], ...]:
    recipes = [(benchmark, topology) for benchmark in benchmarks for topology in topologies]
    if include_gaia_magentic_one and "gaia" in benchmarks:
        recipes.append(("gaia", MAGENTIC_ONE_RECIPE))
    return tuple(recipes)


def _team_instance_payload(
    team_spec: dict[str, Any],
    *,
    benchmark: str,
    topology: str,
    framework: str,
    matrix_id: str,
) -> dict[str, Any]:
    bindings = []
    for requirement in model_resource_requirements(team_spec):
        bindings.append(
            {
                "node_id": str(requirement["node_id"]),
                "requirement": "model_inference",
                "resource_instance_type": "DeploymentInstance",
                "resource_instance_id": DEPLOYMENT_INSTANCE_ID,
            }
        )
    return {
        "schema_version": 5,
        "id": _team_instance_id(benchmark, topology, framework, matrix_id),
        "team_spec_id": str(team_spec["id"]),
        "creation_source": "cross_framework_matrix",
        "runtime_framework": framework,
        "framework_options": {},
        "resource_bindings": bindings,
    }


def _write_recipe_audit(
    *,
    base_url: str,
    matrix_id: str,
    team_specs: dict[str, dict[str, Any]],
    recipes: tuple[tuple[str, str], ...],
    frameworks: tuple[str, ...],
) -> Path:
    """Persist the evidence and adapter mapping behind every controlled recipe."""

    rows: list[dict[str, Any]] = []
    for benchmark, topology in recipes:
        team_spec_id = "gaia" if topology == MAGENTIC_ONE_RECIPE else f"{benchmark}-{topology}"
        team_spec = team_specs[team_spec_id]
        provenance = dict((team_spec.get("metadata") or {}).get("provenance") or {})
        expected_track = "controlled_portability"
        if provenance.get("track") != expected_track:
            raise RuntimeError(
                f"TeamSpec {team_spec_id!r} is not registered for the {expected_track} track"
            )
        if provenance.get("evidence_level") not in {
            "paper_reproduction",
            "paper_inspired",
            "hypothesis",
        }:
            raise RuntimeError(f"TeamSpec {team_spec_id!r} has no valid evidence_level")
        if not provenance.get("sources"):
            raise RuntimeError(f"TeamSpec {team_spec_id!r} has no evidence source")
        for framework in frameworks:
            binding_report = _request(
                base_url,
                "POST",
                "/api/team-binding-report",
                {
                    "runtime_framework": framework,
                    "team_spec": team_spec,
                },
            )
            controlled_eligible = bool(binding_report.get("controlled_comparison_eligible"))
            rows.append(
                {
                    "benchmark": benchmark,
                    "matrix_topology_key": topology,
                    "method_semantics": METHOD_SEMANTICS[topology],
                    "runtime_framework": framework,
                    "recipe_origin": (
                        "magentic_one_semantic_mapping"
                        if topology == MAGENTIC_ONE_RECIPE
                        else "lychee_controlled_adaptation"
                    ),
                    "claim_scope": (
                        "controlled_framework_comparison"
                        if controlled_eligible
                        else "operational_portability_only"
                    ),
                    "reproduction_claim": "none",
                    "framework_execution": FRAMEWORK_EXECUTION_CONTRACTS[framework],
                    "known_official_recipe": KNOWN_OFFICIAL_RECIPES.get((benchmark, framework)),
                    "team_spec_id": team_spec_id,
                    "team_spec_schema_version": team_spec.get("schema_version"),
                    "provenance": provenance,
                    "binding_report": binding_report,
                }
            )
    audit = {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "track": "controlled_portability",
        "evidence_tracks": {
            "native_replication": "framework or benchmark owner recipe, unchanged",
            "literature_replication": "paper recipe frozen at a cited revision",
            "controlled_portability": "one LycheeMAS TeamSpec compiled across adapters",
        },
        "terminology_note": (
            "The legacy matrix key 'independent' denotes the one-node single-agent "
            "baseline. It is not the parallel multi-agent Independent architecture "
            "used by some literature."
        ),
        "combinations": rows,
    }
    path = Path("runs/eval_studio") / f"{matrix_id}-recipe-audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ineligible = [
        row for row in rows if not bool(row["binding_report"].get("controlled_comparison_eligible"))
    ]
    if ineligible:
        failures = "\n".join(
            "  "
            f"{row['benchmark']}:{row['matrix_topology_key']}:{row['runtime_framework']} "
            f"level={row['binding_report'].get('mapping_level')} "
            f"deltas={row['binding_report'].get('semantic_deltas') or []}"
            for row in ineligible
        )
        raise RuntimeError(
            "controlled matrix contains ineligible framework bindings; no registry "
            f"objects were created. Audit: {path}\n{failures}"
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--phase", choices=("smoke", "formal"), default="smoke")
    parser.add_argument(
        "--smoke-cases",
        type=int,
        default=6,
        help=(
            "Cumulative smoke Case target. The default is six so GAIA's "
            "stratified selector samples two Cases from each validation level."
        ),
    )
    parser.add_argument(
        "--formal-cases",
        default="20",
        help=(
            "Cumulative formal Case target. Use an integer for staged execution or "
            "'all' to use each selected benchmark's full registered Case count."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARK_ORDER),
        help="Comma-separated benchmark keys; defaults to the complete matrix.",
    )
    parser.add_argument(
        "--include-gaia-magentic-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GAIA Magentic-One semantics for all selected frameworks.",
    )
    parser.add_argument(
        "--topologies",
        default=",".join(TOPOLOGIES),
        help=(
            "Comma-separated topology keys; defaults to the complete matrix. "
            "Pass an empty value with --include-gaia-magentic-one to run only the "
            "GAIA Magentic-One recipe."
        ),
    )
    parser.add_argument(
        "--frameworks",
        default=",".join(FRAMEWORKS),
        help="Comma-separated framework keys; defaults to the complete matrix.",
    )
    parser.add_argument(
        "--matrix-id",
        required=True,
        help=(
            "Stable identifier for one smoke-to-formal matrix. Smoke and formal phases "
            "must use the same value so completed smoke trials are resumed, not rerun."
        ),
    )
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--start-scheduler", action="store_true")
    parser.add_argument("--max-parallel-instances", type=int, default=16)
    args = parser.parse_args()
    if args.phase == "smoke":
        fixed_target: int | None = args.smoke_cases
    elif str(args.formal_cases).strip().lower() == "all":
        fixed_target = None
    else:
        try:
            fixed_target = int(args.formal_cases)
        except ValueError as exc:
            raise SystemExit("--formal-cases must be a positive integer or 'all'") from exc
    if fixed_target is not None and fixed_target < 1:
        raise SystemExit("Case target must be positive")
    matrix_id = str(args.matrix_id).strip().lower().replace("_", "-")
    if not matrix_id or not all(char.isalnum() or char == "-" for char in matrix_id):
        raise SystemExit("--matrix-id must contain only letters, digits, and hyphens")
    selected_benchmarks = tuple(
        item.strip() for item in str(args.benchmarks).split(",") if item.strip()
    )
    unknown_benchmarks = set(selected_benchmarks) - set(BENCHMARKS)
    if not selected_benchmarks or unknown_benchmarks:
        raise SystemExit(
            "--benchmarks must select one or more of "
            f"{','.join(BENCHMARK_ORDER)}; unknown={sorted(unknown_benchmarks)}"
        )
    selected_topologies = tuple(
        item.strip() for item in str(args.topologies).split(",") if item.strip()
    )
    unknown_topologies = set(selected_topologies) - set(TOPOLOGIES)
    magentic_one_selected = bool(args.include_gaia_magentic_one and "gaia" in selected_benchmarks)
    if unknown_topologies or (not selected_topologies and not magentic_one_selected):
        raise SystemExit(
            "--topologies must select one or more of "
            f"{','.join(TOPOLOGIES)}, unless GAIA Magentic-One is included; "
            f"unknown={sorted(unknown_topologies)}"
        )
    selected_frameworks = tuple(
        item.strip() for item in str(args.frameworks).split(",") if item.strip()
    )
    unknown_frameworks = set(selected_frameworks) - set(FRAMEWORKS)
    if not selected_frameworks or unknown_frameworks:
        raise SystemExit(
            "--frameworks must select one or more of "
            f"{','.join(FRAMEWORKS)}; unknown={sorted(unknown_frameworks)}"
        )
    recipes = _selected_recipes(
        selected_benchmarks,
        selected_topologies,
        include_gaia_magentic_one=args.include_gaia_magentic_one,
    )
    _validate_registry_ids(recipes, selected_frameworks, matrix_id)
    benchmark_instance_ids = {
        benchmark: _resolve_benchmark_instance_id(args.base_url, BENCHMARKS[benchmark])
        for benchmark in selected_benchmarks
    }

    team_specs = {
        str(item["id"]): item for item in _request(args.base_url, "GET", "/api/team-specs")["specs"]
    }
    recipe_audit_path = _write_recipe_audit(
        base_url=args.base_url,
        matrix_id=matrix_id,
        team_specs=team_specs,
        recipes=recipes,
        frameworks=selected_frameworks,
    )
    team_instances = {
        str(item["id"]): item
        for item in _request(args.base_url, "GET", "/api/team-instances")["instances"]
    }
    stale_team_instance_ids = {
        _team_instance_id(benchmark, topology, framework, matrix_id)
        for benchmark, topology in recipes
        for framework in selected_frameworks
        if (
            (current := team_instances.get(
                _team_instance_id(benchmark, topology, framework, matrix_id)
            ))
            and current.get("configuration_status") != "current"
        )
    }
    if stale_team_instance_ids:
        referenced_experiments = _request(
            args.base_url, "GET", "/api/experiment-instances"
        )
        for current in referenced_experiments:
            if str(current.get("team_instance_id") or "") not in stale_team_instance_ids:
                continue
            status = str(current.get("status") or "")
            if status == "queued":
                current = _request(
                    args.base_url,
                    "POST",
                    f"/api/experiment-instances/{current['id']}/dequeue",
                    {},
                )
                status = str(current.get("status") or "")
            if status in {"running", "starting", "draining"}:
                raise RuntimeError(
                    f"stale TeamInstance {current.get('team_instance_id')!r} is referenced "
                    f"by active ExperimentInstance {current.get('id')!r}; stop or dequeue "
                    "that experiment before rebuilding the matrix"
                )
            _request(
                args.base_url,
                "DELETE",
                f"/api/experiment-instances/{current['id']}",
            )
            _archive_launch(current)
    rebound_team_instances = 0
    created_team_instances = 0
    for benchmark, topology in recipes:
        team_spec_id = "gaia" if topology == MAGENTIC_ONE_RECIPE else f"{benchmark}-{topology}"
        team_spec = team_specs[team_spec_id]
        for framework in selected_frameworks:
            team_instance_id = _team_instance_id(
                benchmark,
                topology,
                framework,
                matrix_id,
            )
            current = team_instances.get(team_instance_id)
            if current and current.get("configuration_status") == "current":
                continue
            if current:
                _request(
                    args.base_url,
                    "DELETE",
                    f"/api/team-instances/{team_instance_id}",
                )
                rebound_team_instances += 1
            else:
                created_team_instances += 1
            _request(
                args.base_url,
                "PUT",
                f"/api/team-instances/{team_instance_id}",
                _team_instance_payload(
                    team_spec,
                    benchmark=benchmark,
                    topology=topology,
                    framework=framework,
                    matrix_id=matrix_id,
                ),
            )

    # Persist every ExperimentSpec before reading Instance status. Otherwise a
    # just-updated Spec can leave the in-memory Instance snapshot falsely marked
    # current and the matrix setup would skip the required re-instantiation.
    for benchmark, topology in recipes:
        spec = _experiment_spec(benchmark, topology)
        _request(
            args.base_url,
            "PUT",
            f"/api/experiment-specs/{spec['id']}",
            spec,
        )

    existing = {
        str(item["id"]): item
        for item in _request(args.base_url, "GET", "/api/experiment-instances")
    }
    created = 0
    updated = 0
    enqueued = 0
    for benchmark, topology in recipes:
        plan = BENCHMARKS[benchmark]
        target = fixed_target if fixed_target is not None else plan.full_case_count
        spec = _experiment_spec(benchmark, topology)
        for framework in selected_frameworks:
            instance_id = _instance_id(
                benchmark,
                topology,
                framework,
                matrix_id,
            )
            current = existing.get(instance_id)
            expected_team_instance_id = _team_instance_id(
                benchmark,
                topology,
                framework,
                matrix_id,
            )
            binding_changed = bool(
                current and str(current.get("team_instance_id") or "") != expected_team_instance_id
            )
            failed_smoke = bool(
                current
                and args.phase == "smoke"
                and _smoke_requires_rebuild(current, target=int(target))
            )
            launch_conflict = bool(current and _launch_conflicts(current))
            needs_rebuild = bool(
                current
                and (
                    current.get("configuration_status") != "current"
                    or binding_changed
                    or failed_smoke
                    or launch_conflict
                )
            )
            if current and needs_rebuild and str(current.get("status") or "") == "queued":
                current = _request(
                    args.base_url,
                    "POST",
                    f"/api/experiment-instances/{instance_id}/dequeue",
                    {},
                )
            if current and needs_rebuild:
                status = str(current.get("status") or "")
                if status in {"queued", "running", "starting", "draining"}:
                    raise RuntimeError(
                        f"stale ExperimentInstance {instance_id!r} is {status}; "
                        "stop or dequeue it before rebuilding the matrix"
                    )
                _request(
                    args.base_url,
                    "DELETE",
                    f"/api/experiment-instances/{instance_id}",
                )
                _archive_launch(current)
                existing.pop(instance_id, None)
            controls = {
                "next_segment_case_limit": target,
                "case_completion_target": target,
                "priority": _priority(benchmark, topology),
            }
            if instance_id not in existing:
                try:
                    _request(
                        args.base_url,
                        "POST",
                        "/api/experiment-instances",
                        {
                            "id": instance_id,
                            "experiment_spec_id": spec["id"],
                            "benchmark_instance_id": benchmark_instance_ids[benchmark],
                            "team_instance_id": expected_team_instance_id,
                            "launcher": {"type": "subprocess"},
                            **controls,
                        },
                    )
                    created += 1
                except RuntimeError as exc:
                    if "already exists" not in str(exc):
                        raise
                    # A concurrently running matrix setup may have created the
                    # immutable Instance after this process read the registry.
                    _request(
                        args.base_url,
                        "PATCH",
                        f"/api/experiment-instances/{instance_id}",
                        controls,
                    )
                    updated += 1
            else:
                _request(
                    args.base_url,
                    "PATCH",
                    f"/api/experiment-instances/{instance_id}",
                    controls,
                )
                updated += 1
            if args.enqueue:
                current = _request(
                    args.base_url,
                    "GET",
                    f"/api/experiment-instances/{instance_id}/progress?tail=0",
                )
                status = str(current.get("status") or "")
                target_reached = _case_target_reached(current)
                if not target_reached and status not in {
                    "queued",
                    "running",
                    "starting",
                    "draining",
                }:
                    try:
                        _request(
                            args.base_url,
                            "POST",
                            f"/api/experiment-instances/{instance_id}/enqueue",
                            {},
                        )
                    except RuntimeError as exc:
                        if "already reached its cumulative Case target" not in str(exc):
                            raise
                    else:
                        enqueued += 1

    if args.start_scheduler:
        _request(
            args.base_url,
            "POST",
            "/api/experiment-queue/start",
            {"max_parallel_instances": args.max_parallel_instances},
        )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "matrix_id": matrix_id,
                "case_completion_target": (
                    fixed_target if fixed_target is not None else "benchmark_full"
                ),
                "benchmark_case_targets": {
                    benchmark: (
                        fixed_target
                        if fixed_target is not None
                        else BENCHMARKS[benchmark].full_case_count
                    )
                    for benchmark in selected_benchmarks
                },
                "benchmarks": list(selected_benchmarks),
                "topologies": list(selected_topologies),
                "frameworks": list(selected_frameworks),
                "recipes": [f"{benchmark}:{topology}" for benchmark, topology in recipes],
                "specs": len(recipes),
                "instances": len(recipes) * len(selected_frameworks),
                "created": created,
                "updated": updated,
                "created_team_instances": created_team_instances,
                "rebound_team_instances": rebound_team_instances,
                "enqueued": enqueued,
                "scheduler_started": args.start_scheduler,
                "recipe_audit": str(recipe_audit_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
