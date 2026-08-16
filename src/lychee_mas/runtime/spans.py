"""Incremental JSONL span/event logging for benchmark runs."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {str(k): to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
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


class _JsonlAppendLogger:
    def __init__(
        self, path: str, *, run_fields: dict[str, Any] | None = None, append: bool = False
    ) -> None:
        self.path = path
        self.run_fields = run_fields or {}
        self.seq = self._last_seq(path) if append else 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not append:
            open(path, "w", encoding="utf-8").close()

    def _append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")

    def _last_seq(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        last = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line).get("seq", 0)
                except Exception:
                    continue
                try:
                    last = max(last, int(value))
                except Exception:
                    continue
        return last


class JsonlSpanLogger(_JsonlAppendLogger):
    """Append-only runtime span logger."""

    def log(self, span_type: str, **fields: Any) -> str:
        span_id = str(fields.pop("span_id", None) or uuid.uuid4().hex[:12])
        now = time.time()
        with self._lock:
            self.seq += 1
            record = {
                "schema_version": 3,
                "seq": self.seq,
                "span_id": span_id,
                "parent_span_id": fields.pop("parent_span_id", None),
                "span_type": span_type,
                "timestamp_unix_s": round(now, 6),
                "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                **self.run_fields,
                "case_id": fields.pop("case_id", None),
                "sample_index": fields.pop("sample_index", None),
                **fields,
            }
            self._append(record)
        return span_id

class JsonlGroupChatLogger(_JsonlAppendLogger):
    """Append-only AutoGen GroupChat transcript/event logger."""

    def log(self, event_type: str, **fields: Any) -> str:
        event_id = str(fields.pop("event_id", None) or uuid.uuid4().hex[:12])
        now = time.time()
        with self._lock:
            self.seq += 1
            record = {
                "schema_version": 1,
                "seq": self.seq,
                "event_id": event_id,
                "event_type": event_type,
                "timestamp_unix_s": round(now, 6),
                "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                **self.run_fields,
                "case_id": fields.pop("case_id", None),
                "sample_index": fields.pop("sample_index", None),
                **fields,
            }
            self._append(record)
        return event_id
