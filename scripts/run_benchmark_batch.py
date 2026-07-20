"""Run reusable LycheeMAS benchmark matrices.

This script is only an orchestration helper. It first delegates inference to
``scripts/run_mas.py`` and then delegates scoring to
``scripts/analyze_benchmark_run.py --score-predictions``.

``--size smoke`` and ``--size full`` use the same task defaults, prompts,
scorers, runtime, tools, and dependencies. They only change default sample
count, result root, and terminal label.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TASKS = (
    "aftraj_audit_test",
    "aftraj_audit",
    "agent_collab_idr",
    "agent_collab_rtd",
    "agent_collab_cpr",
    "agent_collab_clc",
    "aime_2024",
    "arc_easy",
    "gaia_validation",
    "gaia_validation_level_1",
    "gaia_validation_level_2",
    "gaia_validation_level_3",
    "gsm8k",
    "human_eval",
    "locomo10",
    "mast_failure",
    "medqa",
    "open_agent_traces",
    "openbookqa",
)

SIZE_DEFAULTS = {
    "smoke": {"n": "10", "runs_root": "runs/benchmarks/smokes"},
    "full": {"n": "all", "runs_root": "runs/benchmarks/full"},
}


def _split_csv(raw: str | None, default: tuple[str, ...]) -> list[str]:
    if not raw:
        return list(default)
    tasks: list[str] = []
    for chunk in raw.split(","):
        name = chunk.strip()
        if name:
            tasks.append(name)
    return tasks


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool, keep_going: bool,
         label: str) -> bool:
    print(f"\n[{label}] " + " ".join(cmd), flush=True)
    if dry_run:
        return True
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        return True
    except subprocess.CalledProcessError:
        if keep_going:
            print(f"[{label}] command failed; continuing because --keep-going is set", flush=True)
            return False
        raise


def _base_env(raw_root: str | None, prepared_root: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    if raw_root:
        env["LYCHEE_BENCHMARK_RAW_ROOT"] = raw_root
    if prepared_root:
        env["LYCHEE_BENCHMARK_PREPARED_ROOT"] = prepared_root
    return env


def _config_for_task(task: str, backend: str) -> str:
    if task == "human_eval":
        return "configs/benchmarks/human_eval.yaml"
    if task.startswith("gaia_validation"):
        return "configs/benchmarks/gaia.yaml"
    if backend == "api":
        return "configs/benchmarks/api.yaml"
    return "configs/benchmarks/local_hf.yaml"


def _cmd(args, *, task: str, backend: str, runs_root: str) -> list[str]:
    cmd = [
        args.python,
        "scripts/run_mas.py",
        "--config",
        _config_for_task(task, backend),
        "--runtime",
        "autogen",
        "--backend",
        backend,
        "--task",
        task,
        "--n",
        str(args.n),
        "--code-executor",
        args.code_executor,
        "--code-timeout",
        str(args.code_timeout),
        "--runs-root",
        runs_root,
    ]
    if args.max_rounds is not None:
        cmd += ["--max-rounds", str(args.max_rounds)]
    if args.max_turns is not None:
        cmd += ["--max-turns", str(args.max_turns)]
    if args.team:
        cmd += ["--team", args.team]
    if args.docker_image:
        cmd += ["--docker-image", args.docker_image]
    if args.resume:
        cmd += ["--resume"]
    if args.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    if backend == "api":
        cmd += ["--method", args.api_method]
        if args.api_model:
            cmd += ["--api-model", args.api_model]
        if args.api_base_url:
            cmd += ["--api-base-url", args.api_base_url]
    else:
        cmd += ["--method", args.local_method, "--P", str(args.P)]
        if args.model_path:
            cmd += ["--model-path", args.model_path]
        if args.model_tag:
            cmd += ["--model-tag", args.model_tag]
        if args.device:
            cmd += ["--device", args.device]
    return cmd


def _latest_run_dir(runs_root: str, task: str) -> Path:
    candidates = list(Path(runs_root).glob(f"**/{task}/predictions.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No predictions.jsonl found for task={task!r} under {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def _score_latest(args, *, env: dict[str, str], task: str, runs_root: str, label: str) -> bool:
    if args.dry_run:
        print(
            f"\n[{label}] {args.python} scripts/analyze_benchmark_run.py "
            f"<latest {task} run under {runs_root}> --score-predictions",
            flush=True,
        )
        return True
    try:
        run_dir = _latest_run_dir(runs_root, task)
    except FileNotFoundError as exc:
        if args.keep_going:
            print(f"[{label}] {exc}; continuing because --keep-going is set", flush=True)
            return False
        raise
    return _run(
        [args.python, "scripts/analyze_benchmark_run.py", str(run_dir), "--score-predictions"],
        env=env,
        dry_run=False,
        keep_going=args.keep_going,
        label=label,
    )


def build_parser(*, default_size: str = "smoke") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LycheeMAS benchmark matrices")
    parser.add_argument(
        "--size",
        choices=("smoke", "full"),
        default=default_size,
        help="Run size. smoke defaults to --n 10; full defaults to --n all.",
    )
    parser.add_argument(
        "--mode",
        choices=("api", "local-hf", "all"),
        default="api",
        help="Which backend matrix to run.",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated benchmark runnable tasks")
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--prepared-root", default="data/benchmarks/prepared")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Result root. Defaults to runs/benchmarks/smokes or runs/benchmarks/full by --size.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--n",
        default=None,
        help="Samples per task. Defaults to 10 for smoke and all for full; all/0 means full run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to run_mas.py for each task")
    parser.add_argument("--skip-score", action="store_true",
                        help="Only run inference; do not call analyze_benchmark_run.py --score-predictions")

    parser.add_argument("--team", default=None, help="Optional explicit team override")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-method", choices=("none", "nl_only"), default="none")

    parser.add_argument(
        "--model-path",
        default=os.environ.get("LYCHEE_HF_MODEL"),
        help="Local HF model path. Defaults to LYCHEE_HF_MODEL when set.",
    )
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-method", choices=("none", "nl_only", "latent_only", "both"),
                        default="none")
    parser.add_argument("--P", type=int, default=0)

    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override per-task YAML run.max_new_tokens; omit to use 8192/16384 config defaults.",
    )
    parser.add_argument("--code-executor", choices=("docker", "local"), default="docker")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--code-timeout", type=int, default=120)
    return parser


def _apply_size_defaults(args) -> None:
    defaults = SIZE_DEFAULTS[args.size]
    if args.n is None:
        args.n = defaults["n"]
    if args.runs_root is None:
        args.runs_root = defaults["runs_root"]


def run_matrix(args) -> None:
    _apply_size_defaults(args)
    env = _base_env(args.raw_root, args.prepared_root)
    tasks = _split_csv(args.tasks, DEFAULT_TASKS)
    runs_root = args.runs_root
    label = args.size

    if args.mode in {"api", "all"}:
        for task in tasks:
            task_runs_root = str(Path(runs_root) / "api")
            cmd = _cmd(args, task=task, backend="api", runs_root=task_runs_root)
            ok = _run(cmd, env=env, dry_run=args.dry_run, keep_going=args.keep_going,
                      label=label)
            if ok and not args.skip_score:
                _score_latest(args, env=env, task=task, runs_root=task_runs_root, label=label)

    if args.mode in {"local-hf", "all"}:
        for task in tasks:
            task_runs_root = str(Path(runs_root) / "local_hf")
            cmd = _cmd(args, task=task, backend="hf", runs_root=task_runs_root)
            ok = _run(cmd, env=env, dry_run=args.dry_run, keep_going=args.keep_going,
                      label=label)
            if ok and not args.skip_score:
                _score_latest(args, env=env, task=task, runs_root=task_runs_root, label=label)


def main(default_size: str = "smoke") -> None:
    parser = build_parser(default_size=default_size)
    args = parser.parse_args()
    run_matrix(args)


if __name__ == "__main__":
    main()
