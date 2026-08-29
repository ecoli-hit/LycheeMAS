from __future__ import annotations

import threading
from pathlib import Path

from lychee_mas.eval.scheduling.run_supervisor import RunSupervisor


def test_run_supervisor_emits_terminal_and_release_callbacks_once() -> None:
    finished = threading.Event()
    released = threading.Event()
    terminal_states: list[tuple[str, dict]] = []
    released_ids: list[str] = []

    class _Jobs:
        def status(self, launch_dir: Path) -> dict:
            return {"status": "completed", "launch_dir": str(launch_dir)}

    supervisor = RunSupervisor(
        _Jobs(),
        on_finished=lambda instance_id, state: (
            terminal_states.append((instance_id, state)),
            finished.set(),
        ),
        on_released=lambda instance_id: (released_ids.append(instance_id), released.set()),
        poll_interval_s=0.01,
    )

    assert supervisor.start("run-1", Path("/tmp/run-1")) is True
    assert finished.wait(1.0)
    assert released.wait(1.0)
    assert terminal_states == [
        ("run-1", {"status": "completed", "launch_dir": "/tmp/run-1"})
    ]
    assert released_ids == ["run-1"]
    assert supervisor.monitored_instance_ids() == []


def test_run_supervisor_rejects_duplicate_monitor_attachment() -> None:
    may_finish = threading.Event()
    released = threading.Event()

    class _Jobs:
        def status(self, launch_dir: Path) -> dict:
            if may_finish.is_set():
                return {"status": "completed"}
            return {"status": "running"}

    supervisor = RunSupervisor(
        _Jobs(),
        on_finished=lambda *_args: None,
        on_released=lambda *_args: released.set(),
        poll_interval_s=0.01,
    )
    launch_dir = Path("/tmp/run-2")

    assert supervisor.start("run-2", launch_dir) is True
    assert supervisor.start("run-2", launch_dir) is False
    may_finish.set()
    assert released.wait(1.0)
