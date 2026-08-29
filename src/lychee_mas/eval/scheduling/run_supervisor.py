"""Supervise already-launched Run processes independently of queue policy."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class RunSupervisor:
    """Poll launch state and emit exactly one terminal/release callback per Run."""

    def __init__(
        self,
        jobs,
        *,
        on_finished: Callable[[str, dict[str, Any]], Any],
        on_released: Callable[[str], Any],
        poll_interval_s: float = 2.0,
    ) -> None:
        self.jobs = jobs
        self.on_finished = on_finished
        self.on_released = on_released
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        self._lock = threading.RLock()
        self._monitored_instance_ids: set[str] = set()

    def monitored_instance_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._monitored_instance_ids)

    def start(self, instance_id: str, launch_dir: Path) -> bool:
        """Start one daemon monitor; return False when it is already attached."""

        with self._lock:
            if instance_id in self._monitored_instance_ids:
                return False
            self._monitored_instance_ids.add(instance_id)

        def monitor() -> None:
            try:
                self._monitor(instance_id, launch_dir)
            finally:
                with self._lock:
                    self._monitored_instance_ids.discard(instance_id)
                self.on_released(instance_id)

        threading.Thread(
            target=monitor,
            name=f"eval-studio-monitor-{instance_id}",
            daemon=True,
        ).start()
        return True

    def _monitor(self, instance_id: str, launch_dir: Path) -> None:
        while True:
            state = self.jobs.status(launch_dir)
            status = str(state.get("status") or "")
            if status not in {"starting", "running"}:
                self.on_finished(instance_id, state)
                return
            time.sleep(self.poll_interval_s)
