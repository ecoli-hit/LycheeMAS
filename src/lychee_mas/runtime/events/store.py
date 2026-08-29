"""Lossless append-only event journal for one benchmark run."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

RUN_EVENT_SCHEMA_VERSION = 2
RUN_EVENTS_DIRNAME = "events"
RUN_EVENTS_FILENAME = "run_events.jsonl"
RUN_PAYLOADS_DIRNAME = "payloads"
RUNTIME_WORKSPACES_FILENAME = "runtime_workspaces.jsonl"
DEFAULT_EVENT_LOG_MAX_EVENTS_PER_FILE = 10_000
DEFAULT_EVENT_LOG_MAX_BYTES_PER_FILE = 64 * 1024 * 1024
DEFAULT_EVENT_PAYLOAD_INLINE_BYTES = 64 * 1024
_EVENT_LOG_MAX_EVENTS_ENV = "LYCHEE_EVENT_LOG_MAX_EVENTS_PER_FILE"
_EVENT_LOG_MAX_BYTES_ENV = "LYCHEE_EVENT_LOG_MAX_BYTES_PER_FILE"
_EVENT_PAYLOAD_INLINE_BYTES_ENV = "LYCHEE_EVENT_PAYLOAD_INLINE_BYTES"
_SEGMENT_INDEX_WIDTH = 6
_PAYLOAD_REF_KEY = "$payload_ref"


@dataclass(frozen=True)
class EventTypeSpec:
    """Runtime-checkable metadata for one event type or event namespace."""

    pattern: str
    category: str
    required_payload_fields: tuple[str, ...] = ()
    terminal: bool = False
    description: str = ""

    @property
    def is_prefix(self) -> bool:
        return self.pattern.endswith(".*")

    def matches(self, event_type: str) -> bool:
        if self.is_prefix:
            return event_type.startswith(self.pattern[:-1])
        return event_type == self.pattern


class EventTypeRegistry:
    """Declare event namespaces and validate payload contracts before writing."""

    def __init__(self) -> None:
        self._specs: list[EventTypeSpec] = []

    def register(
        self,
        pattern: str,
        *,
        category: str,
        required_payload_fields: Iterable[str] = (),
        terminal: bool = False,
        description: str = "",
    ) -> None:
        if any(spec.pattern == pattern for spec in self._specs):
            raise ValueError(f"duplicate event type registration {pattern!r}")
        self._specs.append(
            EventTypeSpec(
                pattern=pattern,
                category=category,
                required_payload_fields=tuple(required_payload_fields),
                terminal=terminal,
                description=description,
            )
        )

    def resolve(self, event_type: str) -> EventTypeSpec:
        exact = [spec for spec in self._specs if not spec.is_prefix and spec.matches(event_type)]
        if exact:
            return exact[0]
        prefixes = [spec for spec in self._specs if spec.is_prefix and spec.matches(event_type)]
        if prefixes:
            return max(prefixes, key=lambda spec: len(spec.pattern))
        raise ValueError(f"unregistered event_type {event_type!r}")

    def validate(self, event_type: str, payload: Mapping[str, Any]) -> EventTypeSpec:
        if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", event_type):
            raise ValueError(f"invalid dotted event_type {event_type!r}")
        spec = self.resolve(event_type)
        missing = [name for name in spec.required_payload_fields if name not in payload]
        if missing:
            raise ValueError(f"{event_type} payload is missing required fields: {missing}")
        return spec

    def describe(self) -> list[dict[str, Any]]:
        return [
            {**asdict(spec), "operation_phase": _operation_phase(spec.pattern)}
            for spec in self._specs
        ]


EVENT_TYPES = EventTypeRegistry()
for _pattern, _category in (
    ("run.*", "run_lifecycle"),
    ("trial.*", "trial_lifecycle"),
    ("attempt.*", "attempt_lifecycle"),
    ("runtime.*", "runtime_operation"),
    ("backend.*", "backend_operation"),
    ("group_chat.*", "group_chat_operation"),
    ("model_call.*", "model_operation"),
    ("tool_call.*", "tool_protocol"),
    ("tool_execution.*", "tool_operation"),
    ("code_executor.*", "tool_operation"),
    ("workspace.*", "workspace_operation"),
    ("web.*", "web_operation"),
    ("agent.*", "agent_message"),
    ("node_invocation.*", "team_graph_execution"),
    ("control_edge.*", "team_graph_control"),
    ("data_edge.*", "team_graph_data"),
    ("data_item.*", "team_graph_data"),
    ("memory.*", "team_graph_memory"),
    ("result.*", "team_graph_result"),
    ("scheduler_cycle.*", "scheduler_operation"),
    ("result_contract.*", "result_contract"),
    ("concurrency.*", "runtime_control"),
):
    EVENT_TYPES.register(_pattern, category=_category)
for _event_type, _category, _description in (
    ("run.started", "run_lifecycle", "A benchmark Run began."),
    ("run.resumed", "run_lifecycle", "An existing Run journal was resumed."),
    (
        "run.stop_requested",
        "run_lifecycle",
        "A graceful stop was requested; no new Trial may begin.",
    ),
    (
        "run.stopped",
        "run_lifecycle",
        "A Run stopped after its in-flight Trials reached terminal states.",
    ),
    (
        "run.paused",
        "run_lifecycle",
        "A Run reached its configured execution-segment Case limit.",
    ),
    ("run.completed", "run_lifecycle", "A Run reached its terminal completed state."),
    ("run.failed", "run_lifecycle", "A Run stopped because an uncaught error escaped."),
    ("trial.started", "trial_lifecycle", "One independent execution of a Case began."),
    (
        "trial.interrupted",
        "trial_lifecycle",
        "A previous process ended before this Trial operation emitted a result.",
    ),
    ("trial.skipped", "trial_lifecycle", "Resume logic skipped an already completed Trial."),
    ("attempt.started", "attempt_lifecycle", "One infrastructure Attempt began."),
    ("attempt.completed", "attempt_lifecycle", "An Attempt completed successfully."),
    ("attempt.failed", "attempt_lifecycle", "An Attempt failed and may be retried."),
    ("backend.probed", "backend_operation", "Backend startup capabilities were probed."),
    ("concurrency.changed", "runtime_control", "The adaptive Case concurrency target changed."),
    ("runtime.started", "runtime_operation", "The MAS runtime for one Attempt began."),
    ("runtime.cancelled", "runtime_operation", "The MAS runtime was cancelled."),
    ("runtime.completed", "runtime_operation", "The MAS runtime completed."),
    ("runtime.failed", "runtime_operation", "The MAS runtime failed."),
    ("workspace.prepared", "workspace_operation", "A per-Trial tool workspace was prepared."),
    ("code_executor.started", "tool_operation", "The code executor startup operation began."),
    ("code_executor.ready", "tool_operation", "The code executor became ready."),
    ("code_executor.stopped", "tool_operation", "The code executor stopped."),
    ("web.context_startup.started", "web_operation", "A browser context startup began."),
    ("web.context_startup.completed", "web_operation", "A browser context became ready."),
    ("web.context_startup.failed", "web_operation", "A browser context failed to start."),
    (
        "web.reset.recovered",
        "web_operation",
        "A transient browser reset failure recovered on a neutral blank page.",
    ),
    (
        "group_chat.selected",
        "group_chat_operation",
        "A RuntimeAdapter selected its framework-native coordination implementation.",
    ),
    ("group_chat.started", "group_chat_operation", "A framework-native team execution began."),
    ("group_chat.cancelled", "group_chat_operation", "A group-chat execution was cancelled."),
    (
        "group_chat.completed",
        "group_chat_operation",
        "A framework-native team execution reached a stop condition.",
    ),
    ("group_chat.failed", "group_chat_operation", "A framework-native team execution failed."),
    (
        "coordination.operation.completed",
        "coordination_operation",
        "A declared TeamSpec coordination operation produced an observable result.",
    ),
    (
        "coordination.protocol_violation",
        "coordination_operation",
        "A framework-native coordinator output violated its declared protocol.",
    ),
    (
        "framework.output_rejected",
        "runtime_operation",
        "A framework rejected an otherwise faithfully preserved model output.",
    ),
    (
        "result_contract.validated",
        "result_contract",
        "A benchmark-aware result projection satisfied its availability contract.",
    ),
    (
        "result_contract.failed",
        "result_contract",
        "A benchmark-aware result projection did not satisfy its availability contract.",
    ),
    (
        "agent.message.published",
        "agent_message",
        "A runtime participant published a message or typed event.",
    ),
    (
        "node_invocation.started",
        "team_graph_execution",
        "One activation of an ExecutableNode or StoreNode began.",
    ),
    (
        "node_invocation.completed",
        "team_graph_execution",
        "One Node activation completed and emitted its declared outputs.",
    ),
    (
        "node_invocation.failed",
        "team_graph_execution",
        "One Node activation failed before producing a valid NodeOutput.",
    ),
    (
        "control_edge.selected",
        "team_graph_control",
        "A declared ControlEdge selected the next Node activation.",
    ),
    (
        "control_edge.rejected",
        "team_graph_control",
        "A requested ControlEdge was not eligible under its guard or contract.",
    ),
    (
        "data_edge.transferred",
        "team_graph_data",
        "A declared DataEdge transferred one or more typed DataItems.",
    ),
    (
        "data_item.created",
        "team_graph_data",
        "A Node output created one typed DataItem.",
    ),
    (
        "data_item.stored",
        "team_graph_data",
        "A StoreNode accepted one typed DataItem.",
    ),
    (
        "data_item.claimed",
        "team_graph_data",
        "A consumer reserved one DataItem for an upcoming Node activation.",
    ),
    (
        "data_item.consumed",
        "team_graph_data",
        "A Node activation consumed a previously stored DataItem.",
    ),
    (
        "data_item.released",
        "team_graph_data",
        "A failed or cancelled activation released a claimed DataItem.",
    ),
    (
        "data_item.expired",
        "team_graph_data",
        "A DataItem expired under its declared lifetime policy.",
    ),
    (
        "memory.initialized",
        "team_graph_memory",
        "One explicit TeamSpec memory state was initialized for a Trial.",
    ),
    (
        "memory.recalled",
        "team_graph_memory",
        "A Node activation recalled readable TeamSpec memory records.",
    ),
    (
        "memory.written",
        "team_graph_memory",
        "A completed Node output was written to an eligible TeamSpec memory state.",
    ),
    (
        "memory.evicted",
        "team_graph_memory",
        "A TeamSpec memory record was removed by the declared retention policy.",
    ),
    (
        "result.submitted",
        "team_graph_result",
        "The ResultStore committed the value exposed to benchmark projection.",
    ),
    ("model_call.started", "model_operation", "A real backend model request began."),
    ("model_call.completed", "model_operation", "A backend model request completed."),
    ("model_call.failed", "model_operation", "A backend model request failed."),
    (
        "model_call.budget_exhausted",
        "runtime_control",
        "A request was blocked by the per-Trial call limit.",
    ),
    (
        "model_call.budget_stopped",
        "runtime_control",
        "GroupChat stopped after call-budget exhaustion.",
    ),
    ("tool_call.requested", "tool_protocol", "A model or runtime event requested a tool call."),
    (
        "tool_loop.limit_reached",
        "runtime_control",
        "A framework-neutral tool loop reached its configured iteration limit.",
    ),
    ("tool_execution.started", "tool_operation", "An observed tool execution began."),
    ("tool_execution.completed", "tool_operation", "An observed tool execution completed."),
    ("tool_execution.failed", "tool_operation", "An observed tool execution failed."),
    (
        "tool_execution.observed",
        "tool_operation",
        "A runtime exposed an already-completed tool result.",
    ),
):
    EVENT_TYPES.register(
        _event_type,
        category=_category,
        terminal=_event_type.endswith(
            (
                ".completed",
                ".failed",
                ".stopped",
                ".paused",
                ".interrupted",
                ".cancelled",
            )
        ),
        description=_description,
    )
EVENT_TYPES.register(
    "trial.completed",
    category="trial_result",
    required_payload_fields=("final_output",),
    terminal=True,
    description="A Trial produced its terminal Prediction.",
)
EVENT_TYPES.register(
    "trial.failed",
    category="trial_result",
    terminal=True,
    description="A Trial exhausted its Attempts without a Prediction.",
)
EVENT_TYPES.register(
    "evaluation.completed",
    category="evaluation_result",
    required_payload_fields=("score", "trial_event_id"),
    terminal=True,
    description="A benchmark scorer evaluated one Trial.",
)
EVENT_TYPES.register(
    "evaluation.failed",
    category="evaluation_result",
    required_payload_fields=("trial_event_id",),
    terminal=True,
    description="A benchmark scorer failed for one Trial.",
)


_WRITABLE_ENVELOPE_FIELDS = frozenset(
    {
        "event_id",
        "worker_id",
        "case_id",
        "dataset_index",
        "trial_index",
        "attempt",
        "operation_id",
        "parent_event_id",
        "correlation_id",
        "payload",
    }
)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {str(key): to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json"))
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def exception_record(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _operation_phase(event_type: str) -> str:
    suffix = event_type.rsplit(".", 1)[-1]
    if suffix == "started":
        return "started"
    if suffix in {"completed", "failed", "stopped", "interrupted", "cancelled"}:
        return "terminal"
    return "point"


def _positive_limit(value: int | str | None, *, default: int, name: str) -> int:
    resolved = default if value is None else int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def event_segment_path(base_path: str | os.PathLike, segment_index: int) -> Path:
    """Return the physical file for one logical EventLog segment."""

    base = Path(base_path)
    if segment_index < 0:
        raise ValueError("segment_index must be non-negative")
    if segment_index == 0:
        return base
    return base.with_name(f"{base.stem}.{segment_index:0{_SEGMENT_INDEX_WIDTH}d}{base.suffix}")


def event_segment_paths(base_path: str | os.PathLike) -> list[Path]:
    """List all physical segments for one logical EventLog in sequence order."""

    base = Path(base_path)
    indexed: dict[int, Path] = {}
    if base.is_file():
        indexed[0] = base
    pattern = re.compile(
        rf"^{re.escape(base.stem)}\.(\d{{{_SEGMENT_INDEX_WIDTH}}})"
        rf"{re.escape(base.suffix)}$"
    )
    if base.parent.is_dir():
        for candidate in base.parent.glob(f"{base.stem}.*{base.suffix}"):
            match = pattern.fullmatch(candidate.name)
            if match:
                index = int(match.group(1))
                if index > 0:
                    indexed[index] = candidate
    return [indexed[index] for index in sorted(indexed)]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_store_root(base_path: Path) -> tuple[Path, Path]:
    run_dir = (
        base_path.parent.parent if base_path.parent.name == RUN_EVENTS_DIRNAME else base_path.parent
    )
    return run_dir, run_dir / RUN_PAYLOADS_DIRNAME / "sha256"


def _is_payload_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {_PAYLOAD_REF_KEY}


def _store_payload_value(value: Any, *, payload_root: Path) -> dict[str, Any]:
    encoded = _canonical_json_bytes(value)
    digest = hashlib.sha256(encoded).hexdigest()
    relative_path = Path(RUN_PAYLOADS_DIRNAME) / "sha256" / digest[:2] / f"{digest}.json.gz"
    target = payload_root / digest[:2] / f"{digest}.json.gz"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return {
        _PAYLOAD_REF_KEY: {
            "sha256": digest,
            "size_bytes": len(encoded),
            "encoding": "json+gzip",
            "path": relative_path.as_posix(),
        }
    }


def _externalize_payload_value(
    value: Any,
    *,
    payload_root: Path,
    inline_bytes: int,
    externalize_self: bool = True,
) -> Any:
    if _is_payload_ref(value):
        return value
    externalized: Any
    if isinstance(value, dict):
        externalized = {
            str(key): _externalize_payload_value(
                item,
                payload_root=payload_root,
                inline_bytes=inline_bytes,
                externalize_self=True,
            )
            for key, item in value.items()
        }
    elif isinstance(value, list):
        externalized = [
            _externalize_payload_value(
                item,
                payload_root=payload_root,
                inline_bytes=inline_bytes,
                externalize_self=True,
            )
            for item in value
        ]
    else:
        externalized = value
    if not externalize_self or len(_canonical_json_bytes(externalized)) <= inline_bytes:
        return externalized
    return _store_payload_value(externalized, payload_root=payload_root)


def materialize_payload_value(value: Any, *, run_dir: str | os.PathLike) -> Any:
    """Recursively resolve Run-local content-addressed payload references."""

    root = Path(run_dir).expanduser().resolve()
    if _is_payload_ref(value):
        reference = value[_PAYLOAD_REF_KEY]
        if not isinstance(reference, Mapping):
            raise ValueError("invalid RunEvent payload reference")
        relative = Path(str(reference.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid RunEvent payload path {relative!s}")
        path = root / relative
        try:
            encoded = gzip.decompress(path.read_bytes())
        except OSError as exc:
            raise ValueError(f"cannot read RunEvent payload {path}: {exc}") from exc
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != str(reference.get("sha256") or ""):
            raise ValueError(f"RunEvent payload hash mismatch for {path}")
        if len(encoded) != int(reference.get("size_bytes") or -1):
            raise ValueError(f"RunEvent payload size mismatch for {path}")
        return materialize_payload_value(json.loads(encoded), run_dir=root)
    if isinstance(value, dict):
        return {
            str(key): materialize_payload_value(item, run_dir=root) for key, item in value.items()
        }
    if isinstance(value, list):
        return [materialize_payload_value(item, run_dir=root) for item in value]
    return value


class RunEventWriter:
    """Write canonical RunEvents with a stable envelope and validated payload."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        run_id: str | None = None,
        experiment_instance_id: str | None = None,
        worker_id: int | None = None,
        append: bool = False,
        max_events_per_file: int | None = None,
        max_bytes_per_file: int | None = None,
        payload_inline_bytes: int | None = None,
    ) -> None:
        self.path = str(path)
        self.base_path = Path(self.path)
        self.process_lock_path = self.base_path.parent / f".{self.base_path.name}.lock"
        self.state_path = self.base_path.parent / f".{self.base_path.name}.state.json"
        self.run_dir, self.payload_root = _payload_store_root(self.base_path)
        self.workspace_manifest_path = self.run_dir / RUNTIME_WORKSPACES_FILENAME
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.max_events_per_file = _positive_limit(
            max_events_per_file
            if max_events_per_file is not None
            else os.environ.get(_EVENT_LOG_MAX_EVENTS_ENV),
            default=DEFAULT_EVENT_LOG_MAX_EVENTS_PER_FILE,
            name="max_events_per_file",
        )
        self.max_bytes_per_file = _positive_limit(
            max_bytes_per_file
            if max_bytes_per_file is not None
            else os.environ.get(_EVENT_LOG_MAX_BYTES_ENV),
            default=DEFAULT_EVENT_LOG_MAX_BYTES_PER_FILE,
            name="max_bytes_per_file",
        )
        self.payload_inline_bytes = _positive_limit(
            payload_inline_bytes
            if payload_inline_bytes is not None
            else os.environ.get(_EVENT_PAYLOAD_INLINE_BYTES_ENV),
            default=DEFAULT_EVENT_PAYLOAD_INLINE_BYTES,
            name="payload_inline_bytes",
        )
        self.default_worker_id = worker_id
        with self._process_lock():
            if append:
                existing = self._read_shared_state() or self._existing_identity()
                if event_segment_paths(self.base_path) and existing is None:
                    raise ValueError(f"cannot append incompatible RunEvent journal {self.path}")
            else:
                existing = None
                for segment in event_segment_paths(self.base_path):
                    segment.unlink()
                self.base_path.write_text("", encoding="utf-8")
                self.state_path.unlink(missing_ok=True)
                self.workspace_manifest_path.unlink(missing_ok=True)
            self.run_id = run_id or (existing or {}).get("run_id") or f"run_{uuid.uuid4().hex}"
            self.experiment_instance_id = experiment_instance_id or (existing or {}).get(
                "experiment_instance_id"
            )
            self._apply_shared_state(existing or {})
            self._write_shared_state()

    @property
    def current_path(self) -> Path:
        return event_segment_path(self.base_path, self.segment_index)

    def _process_lock(self):
        """Serialize EventLog sequence allocation across inference and evaluator processes."""

        class _Lock:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.handle = None

            def __enter__(self):
                self.handle = self.path.open("a+b")
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
                return self

            def __exit__(self, exc_type, exc, traceback_value):
                assert self.handle is not None
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()

        return _Lock(self.process_lock_path)

    def _read_shared_state(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(value, dict) or not value.get("run_id"):
            return None
        return value

    def _apply_shared_state(self, value: Mapping[str, Any]) -> None:
        state_run_id = str(value.get("run_id") or "")
        if state_run_id and state_run_id != self.run_id:
            raise ValueError(f"RunEvent run_id changed in shared writer state: {self.path}")
        self.seq = int(value.get("seq") or 0)
        self.segment_index = int(value.get("segment_index") or 0)
        self.segment_event_count = int(value.get("segment_event_count") or 0)
        self.segment_size_bytes = int(value.get("segment_size_bytes") or 0)
        if not self.experiment_instance_id:
            self.experiment_instance_id = value.get("experiment_instance_id")

    def _write_shared_state(self) -> None:
        value = {
            "run_id": self.run_id,
            "experiment_instance_id": self.experiment_instance_id,
            "seq": self.seq,
            "segment_index": self.segment_index,
            "segment_event_count": self.segment_event_count,
            "segment_size_bytes": self.segment_size_bytes,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _existing_identity(self) -> dict[str, Any] | None:
        paths = event_segment_paths(self.base_path)
        if not paths or paths[0] != self.base_path or self.base_path.stat().st_size == 0:
            return None
        first: dict[str, Any] | None = None
        last_seq = 0
        last_count = 0
        expected_run_id: str | None = None
        for segment_index, path in enumerate(paths):
            if path != event_segment_path(self.base_path, segment_index):
                return None
            segment_count = 0
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    if not isinstance(record, dict) or record.get("schema_version") != 2:
                        return None
                    record_run_id = str(record.get("run_id") or "")
                    if expected_run_id is None:
                        expected_run_id = record_run_id
                    if not record_run_id or record_run_id != expected_run_id:
                        return None
                    first = first or record
                    last_seq = max(last_seq, int(record.get("seq") or 0))
                    segment_count += 1
            last_count = segment_count
        if first is None or not first.get("run_id"):
            return None
        return {
            "run_id": first["run_id"],
            "experiment_instance_id": first.get("experiment_instance_id"),
            "seq": last_seq,
            "segment_index": len(paths) - 1,
            "segment_event_count": last_count,
            "segment_size_bytes": paths[-1].stat().st_size,
        }

    def _rotate_before(self, encoded_size: int) -> None:
        if self.segment_event_count <= 0:
            return
        event_limit_reached = self.segment_event_count >= self.max_events_per_file
        byte_limit_reached = self.segment_size_bytes + encoded_size > self.max_bytes_per_file
        if not event_limit_reached and not byte_limit_reached:
            return
        self.segment_index += 1
        self.segment_event_count = 0
        self.segment_size_bytes = 0
        self.current_path.write_text("", encoding="utf-8")

    def _record_workspace_ownership(self, payload: Mapping[str, Any]) -> None:
        workspace = str(payload.get("workspace") or "").strip()
        if not workspace:
            return
        self.workspace_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": self.run_id,
            "workspace": workspace,
            "case_id": payload.get("case_id"),
            "trial_index": payload.get("trial_index"),
            "attempt": payload.get("attempt"),
        }
        with self.workspace_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def emit(
        self,
        event_type: str,
        *,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        worker_id: int | None = None,
        case_id: str | None = None,
        dataset_index: int | None = None,
        trial_index: int | None = None,
        attempt: int | None = None,
        operation_id: str | None = None,
        parent_event_id: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        payload_value = dict(payload or {})
        EVENT_TYPES.validate(event_type, payload_value)
        event_id = event_id or uuid.uuid4().hex
        phase = _operation_phase(event_type)
        if phase == "started" and operation_id is None:
            operation_id = event_id
        now = time.time()
        with self._lock, self._process_lock():
            shared = self._read_shared_state()
            if shared is None:
                shared = self._existing_identity() or {}
            self._apply_shared_state(shared)
            self.seq += 1
            record = {
                "schema_version": RUN_EVENT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "seq": self.seq,
                "event_id": event_id,
                "event_type": event_type,
                "timestamp_unix_s": round(now, 6),
                "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "experiment_instance_id": self.experiment_instance_id,
                "worker_id": self.default_worker_id if worker_id is None else worker_id,
                "case_id": case_id,
                "dataset_index": dataset_index,
                "trial_index": 0 if trial_index is None and case_id is not None else trial_index,
                "attempt": attempt,
                "operation_id": operation_id,
                "parent_event_id": parent_event_id,
                "correlation_id": correlation_id,
                "payload": _externalize_payload_value(
                    to_jsonable(payload_value),
                    payload_root=self.payload_root,
                    inline_bytes=self.payload_inline_bytes,
                    externalize_self=False,
                ),
            }
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            self._rotate_before(len(encoded))
            with self.current_path.open("ab") as handle:
                handle.write(encoded)
            self.segment_event_count += 1
            self.segment_size_bytes += len(encoded)
            if event_type == "workspace.prepared":
                self._record_workspace_ownership(
                    {
                        **payload_value,
                        "case_id": case_id,
                        "trial_index": trial_index,
                        "attempt": attempt,
                    }
                )
            self._write_shared_state()
        return event_id

    def log_event(self, event_type: str, **fields: Any) -> str:
        """Convenience API that separates envelope fields from the payload."""

        envelope = {
            key: fields.pop(key) for key in tuple(fields) if key in _WRITABLE_ENVELOPE_FIELDS
        }
        explicit_payload = fields.pop("payload", None)
        if explicit_payload is not None:
            fields = {**dict(explicit_payload), **fields}
        return self.emit(event_type, payload=fields, **envelope)

    def record_trial(self, record: Mapping[str, Any]) -> str:
        value = dict(record)
        status = str(value.pop("status", "completed"))
        event_type = "trial.failed" if status in {"error", "failed"} else "trial.completed"
        envelope = {
            key: value.pop(key)
            for key in (
                "event_id",
                "worker_id",
                "case_id",
                "dataset_index",
                "trial_index",
                "attempt",
                "operation_id",
                "parent_event_id",
                "correlation_id",
            )
            if key in value
        }
        retained: dict[str, Any] = {
            key: value[key]
            for key in (
                "task",
                "method",
                "scorer_kind",
                "benchmark_id",
                "base_seed",
                "trial_seed",
                "seed_derivation",
                "trials_per_case",
                "final_output",
                "stop_reason",
                "error_type",
                "error_message",
                "traceback",
                "case_wall_time_s",
                "model_calls_started",
                "message_count",
                "tool_call_count",
                "tool_request_count",
                "tool_execution_count",
                "tool_error_count",
                "tool_agent_error_count",
            )
            if key in value
        }
        if not envelope.get("operation_id"):
            raise ValueError("Trial terminal events require the trial.started operation_id")
        return self.emit(event_type, payload=retained, **envelope)

    def record_evaluation(self, record: Mapping[str, Any]) -> str:
        value = dict(record)
        status = str(value.pop("evaluation_status", "completed"))
        event_type = (
            "evaluation.failed" if status in {"error", "failed"} else "evaluation.completed"
        )
        value.pop("final_output", None)
        envelope = {
            key: value.pop(key)
            for key in (
                "event_id",
                "worker_id",
                "case_id",
                "dataset_index",
                "trial_index",
                "attempt",
                "operation_id",
                "parent_event_id",
                "correlation_id",
            )
            if key in value
        }
        if not envelope.get("parent_event_id") and value.get("trial_event_id"):
            envelope["parent_event_id"] = value["trial_event_id"]
        return self.emit(event_type, payload=value, **envelope)


def run_events_path(run_dir: str | os.PathLike) -> Path:
    return Path(run_dir).expanduser().resolve() / RUN_EVENTS_DIRNAME / RUN_EVENTS_FILENAME


def run_event_paths(run_dir: str | os.PathLike) -> list[Path]:
    """List every physical segment belonging to a Run's logical EventLog."""

    return event_segment_paths(run_events_path(run_dir))


def migrate_legacy_run_events(run_dir: str | os.PathLike) -> dict[str, Any]:
    """Move pre-directory EventLog segments into the canonical ``events/`` folder.

    Early schema-v2 runs wrote ``run_events*.jsonl`` directly below the Run
    directory. Resume must append to the same logical EventLog, so migration is
    deliberately strict: legacy and canonical segments may not coexist.
    """

    root = Path(run_dir).expanduser().resolve()
    legacy_base = root / RUN_EVENTS_FILENAME
    canonical_base = run_events_path(root)
    legacy_paths = event_segment_paths(legacy_base)
    canonical_paths = event_segment_paths(canonical_base)
    if not legacy_paths:
        return {
            "migrated": False,
            "segment_count": len(canonical_paths),
            "event_log_dir": str(canonical_base.parent),
        }
    if canonical_paths:
        raise ValueError(
            "legacy and canonical RunEvent segments both exist; refusing to merge "
            f"{root} automatically"
        )

    canonical_base.parent.mkdir(parents=True, exist_ok=True)
    targets = [canonical_base.parent / path.name for path in legacy_paths]
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        raise FileExistsError(f"RunEvent migration target already exists: {conflicts[0]}")

    moved: list[tuple[Path, Path]] = []
    try:
        for source, target in zip(legacy_paths, targets, strict=True):
            os.replace(source, target)
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                os.replace(target, source)
        raise
    return {
        "migrated": True,
        "segment_count": len(targets),
        "event_log_dir": str(canonical_base.parent),
        "paths": [str(path) for path in targets],
    }


def iter_run_events(
    run_dir: str | os.PathLike,
    *,
    event_types: Iterable[str] | None = None,
    event_prefixes: Iterable[str] | None = None,
    materialize_payloads: bool = True,
) -> Iterator[dict[str, Any]]:
    type_filter = set(event_types or ())
    prefixes = tuple(event_prefixes or ())
    paths = run_event_paths(run_dir)
    if not paths:
        return
    expected_run_id: str | None = None
    previous_seq = 0
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid RunEvent at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"RunEvent must be an object at {path}:{line_number}")
                if value.get("schema_version") != RUN_EVENT_SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported RunEvent schema at {path}:{line_number}; "
                        f"expected {RUN_EVENT_SCHEMA_VERSION}, "
                        f"got {value.get('schema_version')!r}"
                    )
                run_id = str(value.get("run_id") or "")
                if expected_run_id is None:
                    expected_run_id = run_id
                if not run_id or run_id != expected_run_id:
                    raise ValueError(f"RunEvent run_id changed at {path}:{line_number}")
                seq = int(value.get("seq") or 0)
                if seq <= previous_seq:
                    raise ValueError(
                        f"RunEvent seq is not strictly increasing at {path}:{line_number}"
                    )
                previous_seq = seq
                event_type = str(value.get("event_type") or "")
                if type_filter and event_type not in type_filter:
                    continue
                if prefixes and not event_type.startswith(prefixes):
                    continue
                if materialize_payloads:
                    value["payload"] = materialize_payload_value(
                        value.get("payload"), run_dir=Path(run_dir).expanduser().resolve()
                    )
                yield value


def event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def event_data(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return domain data with identity fields, without journal bookkeeping."""

    data = {
        key: event.get(key)
        for key in (
            "run_id",
            "event_id",
            "event_type",
            "case_id",
            "dataset_index",
            "trial_index",
            "attempt",
            "operation_id",
            "parent_event_id",
            "correlation_id",
        )
        if event.get(key) is not None
    }
    data.update(event_payload(event))
    return data


def _latest_terminal_records(
    run_dir: str | os.PathLike,
    event_types: Iterable[str],
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for event in iter_run_events(run_dir, event_types=event_types):
        case_id = str(event.get("case_id") or "")
        if not case_id:
            continue
        key = (case_id, int(event.get("trial_index") or 0))
        latest[key] = event
    return sorted(
        latest.values(),
        key=lambda item: (
            int(item.get("dataset_index") or 0),
            int(item.get("trial_index") or 0),
        ),
    )


def read_trial_records(run_dir: str | os.PathLike) -> list[dict[str, Any]]:
    return _latest_terminal_records(run_dir, ("trial.completed", "trial.failed"))


def read_unfinished_trial_operations(run_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """Return Trial starts that have no result or interruption for their operation."""

    open_operations: dict[str, dict[str, Any]] = {}
    for event in iter_run_events(
        run_dir,
        event_types=(
            "trial.started",
            "trial.completed",
            "trial.failed",
            "trial.interrupted",
        ),
    ):
        operation_id = str(event.get("operation_id") or "")
        if not operation_id:
            continue
        if event.get("event_type") == "trial.started":
            open_operations[operation_id] = event
        else:
            open_operations.pop(operation_id, None)
    return sorted(open_operations.values(), key=lambda item: int(item.get("seq") or 0))


def read_evaluation_records(run_dir: str | os.PathLike) -> list[dict[str, Any]]:
    return _latest_terminal_records(run_dir, ("evaluation.completed", "evaluation.failed"))
