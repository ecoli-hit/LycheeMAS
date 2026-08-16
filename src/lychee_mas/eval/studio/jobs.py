"""Process and tmux job management for Eval Studio launches."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution import CompiledPlan, ExecutionPlanCompiler

_TMUX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PERCENT = re.compile(r"(?:percent=(\d{1,3})|(\d{1,3})(?:\.\d+)?%)")
_CASE_PROGRESS = re.compile(r"\[(\d+)/(\d+)(?:[^\]]*)\]")
_CASE_START = re.compile(
    r"\[(\d+)/(\d+)(?:[^\]]*)\]\s+start\s+case_id=([^\s]+)"
)
_MODEL_ACTIVITY = re.compile(
    r"\[model:([^\]]+)\]\s+(start|done)\s+turn=(\d+)(?:\s+sender=([^\s]+))?"
)
_ACTIVE_JOB_STATUSES = {"starting", "running"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, compiler: ExecutionPlanCompiler) -> None:
        self.compiler = compiler
        self.launch_root = compiler.repo_root / "runs/eval_studio/launches"
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    @staticmethod
    def tmux_sessions() -> list[dict[str, str]]:
        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}\t#{session_windows}"],
                text=True,
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []
        rows = []
        for line in result.stdout.splitlines():
            name, _, windows = line.partition("\t")
            if name:
                rows.append({"name": name, "windows": windows})
        return rows

    def launch(self, plan: CompiledPlan, mode: str | None = None) -> dict[str, Any]:
        if (plan.launch_dir / "job.json").is_file():
            existing = self.status(plan.launch_dir)
            qualifier = "active " if self.is_active_state(existing) else ""
            raise ValueError(
                f"{qualifier}launch {plan.launch_id!r} already exists; "
                "create a new ExperimentInstance ID"
            )
        self.compiler.materialize(plan)
        execution = plan.project["execution"]
        mode = mode or execution.get("mode", "command_only")
        self.validate(plan, mode=mode)
        state = {
            "schema_version": 2,
            "launch_id": plan.launch_id,
            "mode": mode,
            "status": "prepared" if mode == "command_only" else "starting",
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "launch_dir": str(plan.launch_dir),
            "run_dir": str(plan.run_dir),
            "pid": None,
            "tmux_target": None,
        }
        if mode == "command_only":
            self._write_state(plan.launch_dir, state)
            return state

        script = str(self._write_managed_script(plan))
        exit_code_path = plan.launch_dir / "exit_code"
        exit_code_path.unlink(missing_ok=True)
        repo_root = str(plan.project["environment"]["repo_root"])
        if mode == "subprocess":
            process = subprocess.Popen(
                ["bash", script],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self._lock:
                self._processes[plan.launch_id] = process
            state.update(status="running", **self._process_identity(process.pid))
        elif mode in {"new_tmux_session", "existing_tmux_session"}:
            session = str(execution.get("tmux_session") or "lychee-eval")
            window = str(execution.get("tmux_window") or plan.project["name"])
            self._validate_tmux_name(session)
            self._validate_tmux_name(window)
            if mode == "new_tmux_session":
                command = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session,
                    "-c",
                    repo_root,
                    "bash",
                    script,
                ]
                target = session
            else:
                command = [
                    "tmux",
                    "new-window",
                    "-t",
                    session,
                    "-n",
                    window,
                    "-c",
                    repo_root,
                    "bash",
                    script,
                ]
                target = f"{session}:{window}"
            subprocess.run(command, check=True)
            state.update(status="running", tmux_target=target)
        else:
            raise ValueError(
                "execution mode must be command_only, subprocess, new_tmux_session, "
                "or existing_tmux_session"
            )
        state["updated_at_utc"] = _utc_now()
        self._write_state(plan.launch_dir, state)
        if mode == "subprocess":
            self._watch(plan.launch_id, plan.launch_dir, process)
        return state

    def validate(self, plan: CompiledPlan, mode: str | None = None) -> dict[str, Any]:
        """Validate a launch without starting a process or tmux window."""

        execution = plan.project["execution"]
        mode = str(mode or execution.get("mode") or "command_only")
        if mode not in {
            "command_only",
            "subprocess",
            "new_tmux_session",
            "existing_tmux_session",
        }:
            raise ValueError(
                "execution mode must be command_only, subprocess, new_tmux_session, "
                "or existing_tmux_session"
            )

        python = str(plan.project["environment"].get("python") or "python")
        if mode != "command_only" and os.path.sep in python and not Path(python).is_file():
            raise FileNotFoundError(
                f"所选 Python 可执行文件不存在：{python}。"
                "请在“运行环境”选择可用环境并点击“当前实验环境”。"
            )

        result: dict[str, Any] = {"mode": mode, "python": python}
        if mode not in {"new_tmux_session", "existing_tmux_session"}:
            return result
        if shutil.which("tmux") is None:
            raise FileNotFoundError("所选 Launcher 需要 tmux，但当前服务器找不到 tmux 命令。")

        session = str(execution.get("tmux_session") or "lychee-eval")
        window = str(execution.get("tmux_window") or plan.project["name"])
        self._validate_tmux_name(session)
        self._validate_tmux_name(window)
        session_exists = (
            subprocess.run(
                ["tmux", "has-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        if mode == "existing_tmux_session" and not session_exists:
            raise ValueError(f"tmux session 不存在：{session}")
        if mode == "new_tmux_session" and session_exists:
            raise ValueError(f"tmux session 已存在，不能重复新建：{session}")
        result["tmux_session"] = session
        result["tmux_window"] = window
        return result

    @staticmethod
    def _write_managed_script(plan: CompiledPlan) -> Path:
        """Wrap every launcher with durable logging and an observable exit marker."""

        path = plan.launch_dir / "managed_run.sh"
        run_all = plan.launch_dir / "run_all.sh"
        log = plan.launch_dir / "launch.log"
        exit_code = plan.launch_dir / "exit_code"
        temporary_exit = plan.launch_dir / "exit_code.tmp"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set +e\n"
            f"bash {shlex.quote(str(run_all))} >> {shlex.quote(str(log))} 2>&1\n"
            "code=$?\n"
            f"printf '%s\\n' \"$code\" > {shlex.quote(str(temporary_exit))}\n"
            f"mv {shlex.quote(str(temporary_exit))} {shlex.quote(str(exit_code))}\n"
            "exit \"$code\"\n",
            encoding="utf-8",
        )
        path.chmod(0o750)
        return path

    def launch_utility(
        self,
        *,
        job_id: str,
        command: list[str],
        cwd: Path,
        job_root: Path,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start a non-shell resource job such as dataset/model preparation."""
        if not _TMUX_NAME.fullmatch(job_id):
            raise ValueError("invalid utility job id")
        launch_dir = (job_root / job_id).resolve()
        launch_dir.mkdir(parents=True, exist_ok=False)
        log = (launch_dir / "launch.log").open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        with self._lock:
            self._processes[job_id] = process
        state = {
            "schema_version": 2,
            "launch_id": job_id,
            "mode": "subprocess",
            "status": "running",
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "launch_dir": str(launch_dir),
            "run_dir": None,
            **self._process_identity(process.pid),
            "tmux_target": None,
            "command": command,
        }
        self._write_state(launch_dir, state)
        self._watch(job_id, launch_dir, process)
        return state

    def _watch(self, job_id: str, launch_dir: Path, process: subprocess.Popen) -> None:
        def wait() -> None:
            return_code = process.wait()
            state = self._read_state(launch_dir)
            if state:
                state.update(return_code=return_code, updated_at_utc=_utc_now())
                if state.get("status") != "stopped":
                    state["status"] = "completed" if return_code == 0 else "failed"
                self._write_state(launch_dir, state)
            with self._lock:
                self._processes.pop(job_id, None)

        threading.Thread(target=wait, name=f"studio-job-{job_id}", daemon=True).start()

    def status(self, launch_dir: Path) -> dict[str, Any]:
        state = self._read_state(launch_dir)
        if not state:
            raise FileNotFoundError(launch_dir / "job.json")
        launch_id = str(state["launch_id"])
        if state.get("status") == "stopped":
            return state
        exit_code_path = launch_dir / "exit_code"
        if exit_code_path.is_file():
            try:
                return_code = int(exit_code_path.read_text(encoding="utf-8").strip())
            except ValueError:
                return_code = 1
            state.update(
                status="completed" if return_code == 0 else "failed",
                return_code=return_code,
                updated_at_utc=_utc_now(),
            )
            self._write_state(launch_dir, state)
            return state
        with self._lock:
            process = self._processes.get(launch_id)
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                state["status"] = "running"
            else:
                state["status"] = "completed" if return_code == 0 else "failed"
                state["return_code"] = return_code
                with self._lock:
                    self._processes.pop(launch_id, None)
        elif state.get("tmux_target"):
            target = str(state["tmux_target"])
            exists = self._tmux_target_exists(target)
            if not exists and state.get("status") == "running":
                state["status"] = "failed"
                state["failure_reason"] = "tmux target closed without an exit marker"
        elif state.get("pid") and state.get("status") in _ACTIVE_JOB_STATUSES:
            alive, reason = self._process_identity_matches(state)
            if alive:
                state["status"] = "running"
                state["recovered_after_server_restart"] = True
            else:
                state["status"] = "failed"
                state["failure_reason"] = reason or "process exited without an exit marker"
        state["updated_at_utc"] = _utc_now()
        self._write_state(launch_dir, state)
        return state

    def progress(self, launch_dir: Path, *, tail: int = 80) -> dict[str, Any]:
        state = self.status(launch_dir)
        log_path = launch_dir / "launch.log"
        text = self._tail_text(log_path)
        lines = [line for line in text.replace("\r", "\n").splitlines() if line.strip()]
        percentages = []
        for line in lines:
            for match in _PERCENT.finditer(line):
                value = int(match.group(1) or match.group(2))
                if 0 <= value <= 100:
                    percentages.append(value)
        percent = max(percentages, default=0)
        case_progress = [
            (int(match.group(1)), int(match.group(2)))
            for line in lines
            if "start case_id=" not in line
            for match in _CASE_PROGRESS.finditer(line)
            if int(match.group(2)) > 0
        ]
        run_status_path = (
            Path(str(state["run_dir"])) / "run_status.json"
            if state.get("run_dir")
            else None
        )
        run_status = self._read_json(run_status_path) if run_status_path else {}
        expected = int(run_status.get("expected_predictions") or 0)
        successful = int(run_status.get("successful_predictions") or 0)
        errors = int(run_status.get("error_predictions") or 0)
        completed = successful + errors
        if expected:
            percent = min(100, int(completed * 100 / expected))
        elif case_progress:
            completed, total = max(case_progress, key=lambda value: value[0] / value[1])
            percent = max(percent, min(100, int(completed * 100 / total)))
        if state.get("status") == "completed":
            percent = 100
        current_cases = list(run_status.get("active_cases") or [])
        run_is_active = str(run_status.get("status") or "") in _ACTIVE_JOB_STATUSES
        job_is_active = str(state.get("status") or "") in _ACTIVE_JOB_STATUSES
        if not current_cases and run_is_active and job_is_active:
            current_cases = self._active_cases_from_log(lines)
        activity = self._latest_model_activity(lines)
        elapsed_s = self._run_elapsed_seconds(run_status)
        return {
            "state": state,
            "percent": percent,
            "message": lines[-1] if lines else state.get("status", "pending"),
            "lines": lines[-tail:],
            "log_path": str(log_path),
            "run_status_path": str(run_status_path) if run_status_path else None,
            "run_status": str(run_status.get("status") or ""),
            "expected_predictions": expected or None,
            "completed_predictions": completed if expected else None,
            "successful_predictions": successful if expected else None,
            "error_predictions": errors if expected else None,
            "skipped_predictions": int(run_status.get("skipped_predictions") or 0)
            if expected
            else None,
            "remaining_predictions": max(0, expected - completed) if expected else None,
            "case_concurrency": run_status.get("effective_case_concurrency"),
            "current_cases": current_cases,
            "current_activity": activity,
            "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "last_completed_case": run_status.get("last_completed_case"),
        }

    @staticmethod
    def _tail_text(path: Path, max_bytes: int = 256 * 1024) -> str:
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    @staticmethod
    def _read_json(path: Path | None) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _active_cases_from_log(lines: list[str]) -> list[dict[str, Any]]:
        started: dict[int, dict[str, Any]] = {}
        completed: set[int] = set()
        for line in lines:
            start = _CASE_START.search(line)
            if start:
                order = int(start.group(1))
                started[order] = {
                    "case_order": order,
                    "num_cases": int(start.group(2)),
                    "case_id": start.group(3),
                }
                completed.discard(order)
                continue
            match = _CASE_PROGRESS.search(line)
            if match:
                completed.add(int(match.group(1)))
        return [value for order, value in started.items() if order not in completed]

    @staticmethod
    def _latest_model_activity(lines: list[str]) -> dict[str, Any] | None:
        for line in reversed(lines):
            match = _MODEL_ACTIVITY.search(line)
            if match:
                return {
                    "role": match.group(1),
                    "phase": match.group(2),
                    "turn": int(match.group(3)),
                    "sender": match.group(4),
                }
        return None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _run_elapsed_seconds(cls, status: dict[str, Any]) -> float | None:
        """Use the same cumulative resume-segment semantics as cost accounting."""
        accumulated = cls._float_or_none(
            status.get("accumulated_run_elapsed_before_segment_s")
        )
        segment_started = cls._float_or_none(
            status.get("current_segment_started_at_unix_s")
        )
        segment_finished = cls._float_or_none(
            status.get("finished_at_unix_s") or status.get("updated_at_unix_s")
        )
        if (
            str(status.get("status") or "") in _ACTIVE_JOB_STATUSES
            and status.get("finished_at_unix_s") is None
        ):
            segment_finished = time.time()
        if accumulated is not None and segment_started is not None:
            return accumulated + max(
                0.0, (segment_finished or time.time()) - segment_started
            )
        cumulative = cls._float_or_none(status.get("cumulative_run_elapsed_s"))
        if cumulative is not None:
            return cumulative
        started = cls._float_or_none(status.get("started_at_unix_s"))
        if started is None:
            return None
        return max(0.0, (segment_finished or time.time()) - started)

    def stop(self, launch_dir: Path) -> dict[str, Any]:
        state = self.status(launch_dir)
        launch_id = str(state["launch_id"])
        signaled_group: int | None = None
        with self._lock:
            process = self._processes.get(launch_id)
        if process is not None and process.poll() is None:
            signaled_group = os.getpgid(process.pid)
            os.killpg(signaled_group, signal.SIGTERM)
        elif state.get("pid") and state.get("status") in _ACTIVE_JOB_STATUSES:
            alive, reason = self._process_identity_matches(state)
            if not alive:
                raise ValueError(
                    f"refusing to stop launch {launch_id!r}: "
                    f"{reason or 'process identity mismatch'}"
                )
            signaled_group = int(state.get("process_group_id") or state["pid"])
            os.killpg(signaled_group, signal.SIGTERM)
        target = state.get("tmux_target")
        if target:
            if ":" in str(target):
                subprocess.run(["tmux", "kill-window", "-t", str(target)], check=False)
            else:
                subprocess.run(["tmux", "kill-session", "-t", str(target)], check=False)
        if signaled_group is not None:
            deadline = time.monotonic() + 5.0
            while self._process_group_exists(signaled_group) and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._process_group_exists(signaled_group):
                os.killpg(signaled_group, signal.SIGKILL)
        state.update(status="stopped", updated_at_utc=_utc_now())
        self._write_state(launch_dir, state)
        self._mark_run_stopped(state)
        return state

    @staticmethod
    def _mark_run_stopped(state: dict[str, Any]) -> None:
        run_dir = str(state.get("run_dir") or "").strip()
        if not run_dir:
            return
        path = Path(run_dir) / "run_status.json"
        status = JobManager._read_json(path)
        if not status or status.get("status") not in {"starting", "running"}:
            return
        now = time.time()
        accumulated = status.get("accumulated_run_elapsed_before_segment_s")
        segment_started = status.get("current_segment_started_at_unix_s")
        if accumulated is not None and segment_started is not None:
            cumulative = max(0.0, float(accumulated)) + max(
                0.0, now - float(segment_started)
            )
        else:
            started = status.get("started_at_unix_s")
            cumulative = (
                max(0.0, now - float(started))
                if started is not None
                else float(status.get("cumulative_run_elapsed_s") or 0.0)
            )
        status.update(
            status="stopped",
            interrupted_active_cases=list(status.get("active_cases") or []),
            active_cases=[],
            updated_at_unix_s=now,
            finished_at_unix_s=now,
            cumulative_run_elapsed_s=round(cumulative, 6),
        )
        temporary_path: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".run_status.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    def launches(self) -> list[dict[str, Any]]:
        """Return every persisted experiment launch with refreshed process state."""

        if not self.launch_root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.launch_root.iterdir()):
            if not path.is_dir() or not (path / "job.json").is_file():
                continue
            try:
                rows.append(self.status(path))
            except (OSError, TypeError, ValueError):
                continue
        return rows

    @staticmethod
    def is_active_state(state: dict[str, Any]) -> bool:
        return str(state.get("status") or "") in _ACTIVE_JOB_STATUSES

    @staticmethod
    def experiment_instance_snapshot(launch_dir: Path) -> dict[str, Any]:
        candidates = [
            launch_dir / "config_snapshot/experiment_instance.json",
            launch_dir / "config_snapshot/configuration_snapshot.json",
            launch_dir / "project.json",
        ]
        for path in candidates:
            value = JobManager._read_json(path)
            if not value:
                continue
            snapshot = value.get("experiment_instance")
            if snapshot is None:
                snapshot = (value.get("configuration_snapshot") or {}).get(
                    "experiment_instance"
                )
            if isinstance(snapshot, dict) and snapshot.get("id"):
                return dict(snapshot)
        return {}

    @staticmethod
    def _process_identity(pid: int) -> dict[str, Any]:
        identity: dict[str, Any] = {"pid": pid, "process_group_id": os.getpgid(pid)}
        start_ticks = JobManager._proc_start_time_ticks(pid)
        if start_ticks is not None:
            identity["pid_start_time_ticks"] = start_ticks
        return identity

    @staticmethod
    def _proc_start_time_ticks(pid: int) -> int | None:
        path = Path(f"/proc/{pid}/stat")
        if not path.is_file():
            return None
        try:
            fields = path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            if not fields or fields[0] == "Z":
                return None
            return int(fields[19])
        except (IndexError, OSError, ValueError):
            return None

    @staticmethod
    def _process_identity_matches(state: dict[str, Any]) -> tuple[bool, str | None]:
        try:
            pid = int(state["pid"])
            os.kill(pid, 0)
            actual_group = os.getpgid(pid)
        except (KeyError, OSError, TypeError, ValueError):
            return False, "recorded process is no longer running"
        expected_group = state.get("process_group_id")
        if expected_group is not None and actual_group != int(expected_group):
            return False, "recorded process group no longer matches"
        expected_start = state.get("pid_start_time_ticks")
        actual_start = JobManager._proc_start_time_ticks(pid)
        if actual_start is None:
            return False, "recorded process is no longer running"
        if expected_start is not None and actual_start != int(expected_start):
            return False, "recorded PID has been reused by another process"
        if expected_start is None:
            launch_dir = str(state.get("launch_dir") or "")
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                return False, "cannot validate legacy process command"
            if launch_dir and launch_dir not in command:
                return False, "legacy process command does not match the launch directory"
        return True, None

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _tmux_target_exists(target: str) -> bool:
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", target], capture_output=True, check=False
            ).returncode
            == 0
        )

    @staticmethod
    def _validate_tmux_name(value: str) -> None:
        if not _TMUX_NAME.fullmatch(value):
            raise ValueError("tmux names may contain only letters, numbers, '.', '_' and '-'")

    @staticmethod
    def _write_state(launch_dir: Path, state: dict[str, Any]) -> None:
        path = launch_dir / "job.json"
        launch_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=launch_dir,
                prefix=".job.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    @staticmethod
    def _read_state(launch_dir: Path) -> dict[str, Any]:
        path = launch_dir / "job.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
