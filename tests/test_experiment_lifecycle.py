from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from lychee_mas.eval.experiments.lifecycle import (
    finished_instance_projection,
    read_run_status,
)
from lychee_mas.eval.scheduling.jobs import JobManager


def test_read_run_status_uses_job_run_directory(tmp_path) -> None:
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "paused", "completed_distinct_cases": 3}),
        encoding="utf-8",
    )

    assert read_run_status({"run_dir": str(tmp_path)})["status"] == "paused"
    assert read_run_status({}) == {}


def test_finished_instance_requeues_a_paused_segment_when_queue_stays_enabled() -> None:
    projection = finished_instance_projection(
        {"queue": {"enabled": True}},
        {"status": "completed", "return_code": 0},
        {"status": "paused", "completed_distinct_cases": 3},
        case_completion_target=10,
        finished_at_utc="2026-08-21T00:00:00Z",
    )

    assert projection["status"] == "queued"
    assert projection["queue"]["resume_from_status"] == "paused"
    assert projection["job_status"] == "completed"


def test_finished_instance_disables_queue_after_reaching_case_target() -> None:
    projection = finished_instance_projection(
        {"queue": {"enabled": True}},
        {"status": "completed", "return_code": 0},
        {"status": "paused", "completed_distinct_cases": 10},
        case_completion_target=10,
        finished_at_utc="2026-08-21T00:00:00Z",
    )

    assert projection["status"] == "paused"
    assert projection["queue"]["enabled"] is False
    assert projection["paused_at_utc"] == "2026-08-21T00:00:00Z"


def test_finished_instance_preserves_terminal_failure_as_failed() -> None:
    projection = finished_instance_projection(
        {"queue": {"enabled": False}},
        {"status": "failed", "return_code": 2},
        {},
        case_completion_target=None,
        finished_at_utc="2026-08-21T00:00:00Z",
    )

    assert projection["status"] == "failed"
    assert projection["return_code"] == 2


def test_run_failure_overrides_a_successful_launcher_exit() -> None:
    projection = finished_instance_projection(
        {"queue": {"enabled": False}},
        {"status": "completed", "return_code": 0},
        {"status": "failed"},
        case_completion_target=None,
        finished_at_utc="2026-08-21T00:00:00Z",
    )

    assert projection["status"] == "failed"
    assert projection["run_status"] == "failed"
    assert projection["lifecycle_consistent"] is False
    assert "Run failed" in projection["lifecycle_conflict"]


def test_read_run_status_uses_the_instance_fallback_directory(tmp_path) -> None:
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "complete_with_errors"}), encoding="utf-8"
    )

    assert read_run_status({}, fallback_run_dir=tmp_path)["status"] == "complete_with_errors"


def test_job_stop_persists_terminal_state_before_slow_sandbox_cleanup(
    tmp_path, monkeypatch
) -> None:
    launch_dir = tmp_path / "launch"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "active_cases": ["case-1"]}),
        encoding="utf-8",
    )
    manager = JobManager(SimpleNamespace(repo_root=tmp_path))
    manager._write_state(
        launch_dir,
        {
            "schema_version": 2,
            "launch_id": "slow-cleanup",
            "status": "running",
            "run_dir": str(run_dir),
            "pid": None,
            "tmux_target": None,
        },
    )
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()

    def slow_cleanup(state):
        cleanup_started.set()
        release_cleanup.wait(timeout=5)
        return {"status": "completed", "workspace_count": 0, "removed": []}

    monkeypatch.setattr(manager, "_cleanup_launch_docker_sandboxes", slow_cleanup)
    started = time.monotonic()
    state = manager.stop(launch_dir)

    assert time.monotonic() - started < 1.0
    assert state["status"] == "stopped"
    assert state["sandbox_cleanup"] == {"status": "pending"}
    assert manager.status(launch_dir)["status"] == "stopped"
    assert read_run_status({"run_dir": str(run_dir)})["status"] == "stopped"
    assert cleanup_started.wait(timeout=1)

    release_cleanup.set()
    for _ in range(100):
        if manager.status(launch_dir).get("sandbox_cleanup", {}).get("status") == "completed":
            break
        time.sleep(0.01)
    assert manager.status(launch_dir)["sandbox_cleanup"]["status"] == "completed"
