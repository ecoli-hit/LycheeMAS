"""Run-local artifacts, resume indexing, and startup diagnostics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .configuration import get_config_value
from .trial_identity import case_id_from_item, dataset_index_from_item


class TeeStream:
    """Mirror one terminal stream to a durable Run-local console log."""

    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, text):
        self.primary.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self.primary, name)


def start_console_log(out_dir: str, *, append: bool = False) -> str:
    """Start terminal mirroring and return the console-log path."""

    path = os.path.join(out_dir, "console_log.txt")
    log_file = open(path, "a" if append else "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    if append:
        print("\n" + "=" * 88, flush=True)
        print(f"[resume] appending to existing console_log={path}", flush=True)
    print(f"[log] console_log={path}", flush=True)
    return path


def prepare_trials_for_resume(
    run_dir: str,
    data: list[dict[str, Any]],
    *,
    start_index: int,
    resume: bool,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    """Index successful terminal events by exact ``(case_id, trial_index)``."""

    if not resume:
        return {}, {
            "enabled": False,
            "existing_trials": 0,
            "kept_successful_trials": 0,
        }

    from lychee_mas.runtime.events.store import (
        event_data,
        migrate_legacy_run_events,
        read_trial_records,
    )

    migration = migrate_legacy_run_events(run_dir)
    existing = read_trial_records(run_dir)
    valid_case_ids = {
        case_id_from_item(item, dataset_index_from_item(item, start_index + offset))
        for offset, item in enumerate(data)
    }
    completed: dict[str, set[int]] = {}
    ignored = 0
    for event in existing:
        record = event_data(event)
        case_id = str(record.get("case_id") or "")
        status = "failed" if record.get("event_type") == "trial.failed" else "completed"
        trial_index = int(record.get("trial_index", 0) or 0)
        if (
            case_id not in valid_case_ids
            or status == "failed"
            or trial_index in completed.get(case_id, set())
        ):
            ignored += 1
            continue
        completed.setdefault(case_id, set()).add(trial_index)
    kept = sum(len(values) for values in completed.values())
    return completed, {
        "enabled": True,
        "existing_trials": len(existing),
        "kept_successful_trials": kept,
        "ignored_trial_events": ignored,
        "skipped_trial_count": kept,
        "event_log_migration": migration,
    }


def print_resolved_config(
    snapshot: dict[str, Any],
    *,
    out_dir: str,
    run_events_path: str,
    resume_info: dict[str, Any],
) -> None:
    """Print the exact frozen parameters written for this Run."""

    resolved = dict(snapshot.get("resolved") or {})
    resolved["out_dir"] = out_dir
    resolved["run_events_path"] = run_events_path
    resolved["resume"] = resume_info
    print("[config] resolved parameters:", flush=True)
    print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def model_tag(
    args,
    config: dict[str, Any],
    *,
    backend_provider: str | None = None,
    api_model: str | None = None,
    model_path: str | None = None,
    suffix: str = "",
) -> str:
    """Resolve a truthful, stable model label for result directories."""

    if args.model_tag:
        return args.model_tag
    if args.api_model and api_model:
        return f"{api_model}{suffix}"
    if args.model_path and model_path:
        return os.path.basename(os.path.normpath(model_path))
    configured = get_config_value(config, "backend.model_tag")
    if configured:
        return str(configured)
    if backend_provider == "api" and api_model:
        return f"{api_model}{suffix}"
    if model_path:
        return os.path.basename(os.path.normpath(model_path))
    return api_model or "model"


def require_local_docker_image(image: str, *, repo_root: str | Path) -> dict | None:
    """Fail before inference when a project-owned sandbox is stale or incomplete."""

    if not image.endswith(":local"):
        return None
    from lychee_mas.runtime.adapters.infrastructure.docker import (
        SandboxVerificationError,
        verify_local_sandbox,
    )

    try:
        report = verify_local_sandbox(image, repo_root=str(repo_root))
    except SandboxVerificationError as exc:
        raise SystemExit(
            f"Docker sandbox {image!r} is not ready: {exc}. "
            "Rebuild and verify it with: python scripts/prepare_benchmarks.py "
            "--tasks human_eval,gaia --docker-images always"
        ) from exc
    print(
        f"[sandbox] image={image} profile={report.get('profile')} "
        f"fingerprint={str(report.get('fingerprint') or '')[:12]} verified=true",
        flush=True,
    )
    return report
