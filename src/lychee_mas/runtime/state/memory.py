"""Trial-scoped execution of explicit TeamSpec memory state.

This service owns portable memory semantics. Framework-native memory APIs are
adapters around this contract; they must not silently change access, retrieval,
retention, injection, or write timing.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from lychee_mas.runtime.contracts.memory import memory_states


def _estimated_tokens(text: str) -> int:
    """Return a conservative tokenizer-independent retention estimate."""

    return max(1, math.ceil(len(text) / 4)) if text else 0


@dataclass(frozen=True)
class MemoryRecall:
    """Rendered memory fragments for one Node activation."""

    node_id: str
    model_context: tuple[str, ...] = ()
    node_input: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    item_count: int = 0

    @property
    def empty(self) -> bool:
        return self.item_count == 0


@dataclass
class _MemoryRecord:
    id: str
    content: str
    writer_node_id: str
    created_monotonic_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def event_value(self) -> dict[str, Any]:
        return {
            "memory_record_id": self.id,
            "content": self.content,
            "writer_node_id": self.writer_node_id,
            "created_monotonic_s": self.created_monotonic_s,
            "estimated_tokens": _estimated_tokens(self.content),
            "metadata": dict(self.metadata),
        }


class TeamMemoryRuntime:
    """Execute the portable subset of TeamSpec memory for one Trial.

    The first implementation deliberately supports only chronological,
    Trial-scoped memory. Semantic/hybrid retrieval requires an explicit vector
    provider binding; run/persistent lifetime requires a durable state owner.
    Refusing unsupported semantics is safer than silently substituting a
    different algorithm in one framework.
    """

    def __init__(
        self,
        team_spec: Mapping[str, Any],
        *,
        log_event: Callable[..., str | None] | None = None,
    ) -> None:
        self._log_event = log_event
        self._states = {
            str(item["id"]): dict(item) for item in memory_states(dict(team_spec))
        }
        self._records: dict[str, list[_MemoryRecord]] = {
            memory_id: [] for memory_id in self._states
        }
        self._written_invocations: set[tuple[str, str, str]] = set()
        self._recall_cache: dict[tuple[str, str], MemoryRecall] = {}
        self._validate_portable_support()
        self._load_initial_values()
        for memory_id, state in self._states.items():
            self._emit(
                "memory.initialized",
                memory_id=memory_id,
                lifetime=state.get("lifetime"),
                retrieval_mode=((state.get("memory") or {}).get("retrieval") or {}).get(
                    "mode"
                ),
                reader_node_ids=list(state.get("readers") or []),
                writer_node_ids=list(state.get("writers") or []),
                initial_item_count=len(self._records[memory_id]),
            )

    @property
    def enabled(self) -> bool:
        return bool(self._states)

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(self._states)

    def _validate_portable_support(self) -> None:
        for memory_id, state in self._states.items():
            lifetime = str(state.get("lifetime") or "trial")
            if lifetime != "trial":
                raise ValueError(
                    f"Memory {memory_id!r} lifetime={lifetime!r} requires a durable "
                    "MemoryInstance binding; the portable runtime currently supports "
                    "lifetime='trial' only"
                )
            mode = str(
                (((state.get("memory") or {}).get("retrieval") or {}).get("mode"))
                or "chronological"
            )
            if mode != "chronological":
                raise ValueError(
                    f"Memory {memory_id!r} retrieval.mode={mode!r} requires an explicit "
                    "semantic index binding; the portable runtime currently supports "
                    "mode='chronological' only"
                )

    def _load_initial_values(self) -> None:
        for memory_id, state in self._states.items():
            initial = state.get("initial")
            values = (
                initial
                if isinstance(initial, list)
                else ([] if initial in ({}, None) else [initial])
            )
            for value in values:
                if isinstance(value, Mapping):
                    content = str(value.get("content") or value.get("text") or "")
                    metadata = {
                        str(key): item
                        for key, item in value.items()
                        if key not in {"content", "text"}
                    }
                else:
                    content = str(value)
                    metadata = {}
                if content:
                    self._records[memory_id].append(
                        _MemoryRecord(
                            id=uuid.uuid4().hex,
                            content=content,
                            writer_node_id="team_spec.initial",
                            created_monotonic_s=time.monotonic(),
                            metadata=metadata,
                        )
                    )
            self._enforce_retention(memory_id)

    def recall(
        self,
        node_id: str,
        *,
        task: str,
        node_input: str,
        invocation_id: str | None = None,
    ) -> MemoryRecall:
        """Recall every memory state explicitly readable by ``node_id``."""

        cache_key = (node_id, str(invocation_id or ""))
        if invocation_id is not None and cache_key in self._recall_cache:
            return self._recall_cache[cache_key]

        model_context: list[str] = []
        input_fragments: list[str] = []
        recalled_memory_ids: list[str] = []
        total_items = 0
        for memory_id, state in self._states.items():
            if node_id not in set(map(str, state.get("readers") or [])):
                continue
            policy = dict(state.get("memory") or {})
            retrieval = dict(policy.get("retrieval") or {})
            top_k = int(retrieval.get("top_k") or 8)
            records = self._records[memory_id][-top_k:]
            if not records:
                continue
            rendered_items = "\n".join(
                f"{index}. {record.content}" for index, record in enumerate(records, start=1)
            )
            template = str(
                (policy.get("injection") or {}).get("template")
                or "Relevant memory from {memory_id}:\n{items}"
            )
            rendered = template.format(
                memory_id=memory_id,
                items=rendered_items,
                task=task,
                node_input=node_input,
            )
            target = str(
                (policy.get("injection") or {}).get("target") or "model_context"
            )
            (model_context if target == "model_context" else input_fragments).append(rendered)
            recalled_memory_ids.append(memory_id)
            total_items += len(records)
            self._emit(
                "memory.recalled",
                memory_id=memory_id,
                node_id=node_id,
                invocation_id=invocation_id,
                retrieval_mode="chronological",
                query_source=retrieval.get("query_source"),
                injection_target=target,
                item_count=len(records),
                record_ids=[record.id for record in records],
                rendered_content=rendered,
            )
        result = MemoryRecall(
            node_id=node_id,
            model_context=tuple(model_context),
            node_input=tuple(input_fragments),
            memory_ids=tuple(recalled_memory_ids),
            item_count=total_items,
        )
        if invocation_id is not None:
            self._recall_cache[cache_key] = result
        return result

    def write_node_output(
        self,
        node_id: str,
        content: str,
        *,
        invocation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        """Write one completed Node output to all eligible memory states."""

        content = str(content or "")
        if not content.strip():
            return ()
        written: list[str] = []
        for memory_id, state in self._states.items():
            if node_id not in set(map(str, state.get("writers") or [])):
                continue
            write = dict((state.get("memory") or {}).get("write") or {})
            if str(write.get("policy") or "node_output") != "node_output":
                continue
            dedupe_key = (memory_id, node_id, invocation_id)
            if dedupe_key in self._written_invocations:
                continue
            self._written_invocations.add(dedupe_key)
            record = _MemoryRecord(
                id=uuid.uuid4().hex,
                content=content,
                writer_node_id=node_id,
                created_monotonic_s=time.monotonic(),
                metadata=dict(metadata or {}),
            )
            self._records[memory_id].append(record)
            self._emit(
                "memory.written",
                memory_id=memory_id,
                node_id=node_id,
                invocation_id=invocation_id,
                write_policy="node_output",
                write_source=write.get("source") or "source.output",
                record=record.event_value(),
            )
            self._enforce_retention(memory_id)
            written.append(memory_id)
        return tuple(written)

    def _enforce_retention(self, memory_id: str) -> None:
        state = self._states[memory_id]
        retention = dict(state.get("retention") or {})
        max_items = retention.get("max_items")
        max_tokens = retention.get("max_tokens")
        overflow = str(retention.get("overflow") or "drop_oldest")
        records = self._records[memory_id]

        def exceeds() -> bool:
            return bool(
                (max_items is not None and len(records) > int(max_items))
                or (
                    max_tokens is not None
                    and sum(_estimated_tokens(item.content) for item in records)
                    > int(max_tokens)
                )
            )

        while exceeds():
            if overflow != "drop_oldest":
                raise RuntimeError(
                    f"Memory {memory_id!r} exceeded retention and overflow={overflow!r}"
                )
            evicted = records.pop(0)
            self._emit(
                "memory.evicted",
                memory_id=memory_id,
                reason="retention_limit",
                record=evicted.event_value(),
            )

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            memory_id: [record.event_value() for record in records]
            for memory_id, records in self._records.items()
        }

    def _emit(self, event_type: str, **fields: Any) -> str | None:
        return self._log_event(event_type, **fields) if self._log_event else None


__all__ = ["MemoryRecall", "TeamMemoryRuntime"]
