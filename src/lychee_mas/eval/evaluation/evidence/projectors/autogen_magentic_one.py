"""Project AutoGen Magentic-One ledger calls into neutral coordination evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

PROJECTOR_ID = "autogen_magentic_one_ledger"
PROJECTOR_VERSION = 1
MAPPING_VERSION = "official_magentic_one_state_machine_v1"

_PROGRESS_KEYS = (
    "is_request_satisfied",
    "is_progress_being_made",
    "is_in_loop",
    "instruction_or_question",
    "next_speaker",
)
_FENCED_JSON_RE = re.compile(r"```(?:\s*([\w+\-]+))?\n([\s\S]*?)```")


def _trial_key(event: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(event.get("case_id") or ""),
        int(event.get("trial_index") or 0),
        int(event.get("attempt") or 0),
    )


def _attributes(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes")
    return value if isinstance(value, dict) else {}


def _progress_ledger(content: Any, participants: set[str]) -> dict[str, Any] | None:
    if not isinstance(content, str) or '"is_request_satisfied"' not in content:
        return None
    matches = _FENCED_JSON_RE.findall(content)
    try:
        if not matches:
            values = [json.loads(content)]
        else:
            values = []
            for language, candidate in matches:
                if language and language.strip().lower() != "json":
                    return None
                values.append(json.loads(candidate))
    except (TypeError, ValueError):
        return None
    if len(values) != 1 or not isinstance(values[0], dict):
        return None
    ledger = values[0]
    for key in _PROGRESS_KEYS:
        entry = ledger.get(key)
        if not isinstance(entry, dict) or "answer" not in entry or "reason" not in entry:
            return None
    if not bool(ledger["is_request_satisfied"]["answer"]):
        if ledger["next_speaker"]["answer"] not in participants:
            return None
    return ledger


def _operation_bindings(attributes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in attributes.get("operation_bindings") or []:
        if not isinstance(raw, dict):
            continue
        operation = str(raw.get("operation") or "").strip()
        if operation:
            result[operation] = dict(raw)
    return result


@dataclass
class _LedgerState:
    group_chat_event_id: str
    participants: set[str]
    controller_node_id: str
    max_stalls: int
    operation_bindings: dict[str, dict[str, Any]]
    initialization_call_ids: list[str] = field(default_factory=list)
    plan_emitted: bool = False
    consecutive_stalls: int = 0
    replans: int = 0
    pending_replan_source_ids: list[str] = field(default_factory=list)
    pending_replan_updates: int = 0


class AutoGenMagenticOneLedgerProjector:
    """Recover neutral operations from losslessly recorded official ledger calls."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, int, int], _LedgerState] = {}
        self._report: dict[str, Any] = {
            "projector_id": PROJECTOR_ID,
            "projector_version": PROJECTOR_VERSION,
            "mapping_version": MAPPING_VERSION,
            "eligible_group_chats": 0,
            "accepted_progress_ledgers": 0,
            "rejected_progress_ledger_candidates": 0,
            "derived_event_count": 0,
            "derived_event_type_counts": {},
        }

    def observe(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = str(event.get("event_type") or "")
        key = _trial_key(event)
        if event_type == "group_chat.started":
            return self._start_group_chat(key, event)
        state = self._states.get(key)
        if state is None:
            return []
        if event_type == "model_call.completed":
            return self._model_call_completed(state, event)
        if event_type == "group_chat.completed":
            derived = [
                self._derived_event(
                    source=event,
                    state=state,
                    operation="aggregate",
                    source_event_ids=[str(event.get("event_id") or "")],
                    projection_basis="official_group_chat_completed",
                    state_snapshot=self._state_snapshot(state),
                )
            ]
            self._states.pop(key, None)
            return self._record(derived)
        if event_type in {"group_chat.failed", "group_chat.cancelled"}:
            self._states.pop(key, None)
        return []

    def report(self) -> dict[str, Any]:
        return {
            **self._report,
            "active_group_chats_at_end": len(self._states),
        }

    def _start_group_chat(
        self,
        key: tuple[str, int, int],
        event: dict[str, Any],
    ) -> list[dict[str, Any]]:
        attributes = _attributes(event)
        config = attributes.get("group_chat_config")
        config = config if isinstance(config, dict) else {}
        group_chat_class = str(attributes.get("group_chat_class") or "")
        if group_chat_class != "MagenticOneGroupChat" and config.get("type") != "magentic_one":
            return []
        bindings = _operation_bindings(attributes)
        controller = next(
            (
                str(binding.get("node_id"))
                for operation in ("monitor_progress", "delegate", "plan")
                if (binding := bindings.get(operation)) and binding.get("node_id")
            ),
            "Orchestrator",
        )
        participants = {
            str(item)
            for item in attributes.get("participants") or attributes.get("member_nodes") or []
            if str(item).strip()
        }
        self._states[key] = _LedgerState(
            group_chat_event_id=str(event.get("event_id") or ""),
            participants=participants,
            controller_node_id=controller,
            max_stalls=max(1, int(config.get("max_stalls") or 3)),
            operation_bindings=bindings,
        )
        self._report["eligible_group_chats"] += 1
        return []

    def _model_call_completed(
        self,
        state: _LedgerState,
        event: dict[str, Any],
    ) -> list[dict[str, Any]]:
        attributes = _attributes(event)
        if attributes.get("controller") is not True:
            return []
        actor = str(event.get("actor") or "")
        sender = str(attributes.get("sender") or "")
        if actor != state.controller_node_id and sender != "MagenticOneOrchestrator":
            return []

        source_event_id = str(event.get("event_id") or "")
        content = attributes.get("final_content")
        ledger = _progress_ledger(content, state.participants)
        if ledger is None:
            if isinstance(content, str) and '"is_request_satisfied"' in content:
                self._report["rejected_progress_ledger_candidates"] += 1
                return []
            return self._non_ledger_controller_call(state, event, source_event_id)

        self._report["accepted_progress_ledgers"] += 1
        derived: list[dict[str, Any]] = []
        if not state.plan_emitted:
            derived.append(
                self._plan_event(
                    state,
                    event,
                    [*state.initialization_call_ids, source_event_id],
                    "first_accepted_progress_ledger_after_task_ledger_initialization",
                )
            )

        derived.append(
            self._derived_event(
                source=event,
                state=state,
                operation="monitor_progress",
                source_event_ids=[source_event_id],
                progress_ledger=ledger,
                state_snapshot=self._state_snapshot(state),
            )
        )
        satisfied = bool(ledger["is_request_satisfied"]["answer"])
        if satisfied:
            return self._record(derived)

        making_progress = bool(ledger["is_progress_being_made"]["answer"])
        in_loop = bool(ledger["is_in_loop"]["answer"])
        stalled = (not making_progress) or in_loop
        if stalled:
            state.consecutive_stalls += 1
        else:
            # This deliberately mirrors AutoGen's official decrement semantics.
            state.consecutive_stalls = max(0, state.consecutive_stalls - 1)
        derived.append(
            self._derived_event(
                source=event,
                state=state,
                operation="detect_stall",
                source_event_ids=[source_event_id],
                stalled=stalled,
                in_loop=in_loop,
                making_progress=making_progress,
                consecutive_stalls=state.consecutive_stalls,
                state_snapshot=self._state_snapshot(state),
            )
        )
        if state.consecutive_stalls >= state.max_stalls:
            state.pending_replan_source_ids = [source_event_id]
            state.pending_replan_updates = 2
            return self._record(derived)

        derived.append(
            self._derived_event(
                source=event,
                state=state,
                operation="delegate",
                source_event_ids=[source_event_id],
                next_node=ledger["next_speaker"]["answer"],
                instruction=ledger["instruction_or_question"]["answer"],
                request_satisfied=False,
                state_snapshot=self._state_snapshot(state),
            )
        )
        return self._record(derived)

    def _non_ledger_controller_call(
        self,
        state: _LedgerState,
        event: dict[str, Any],
        source_event_id: str,
    ) -> list[dict[str, Any]]:
        if state.pending_replan_updates:
            state.pending_replan_source_ids.append(source_event_id)
            state.pending_replan_updates -= 1
            if state.pending_replan_updates:
                return []
            state.replans += 1
            derived = [
                self._derived_event(
                    source=event,
                    state=state,
                    operation="replan",
                    source_event_ids=list(state.pending_replan_source_ids),
                    projection_basis="official_facts_update_and_plan_update_completed",
                    consecutive_stalls=state.consecutive_stalls,
                    replans=state.replans,
                    state_snapshot=self._state_snapshot(state),
                )
            ]
            state.pending_replan_source_ids.clear()
            return self._record(derived)

        if not state.plan_emitted:
            state.initialization_call_ids.append(source_event_id)
            if len(state.initialization_call_ids) >= 2:
                return self._record(
                    [
                        self._plan_event(
                            state,
                            event,
                            list(state.initialization_call_ids),
                            "official_initial_facts_and_plan_calls_completed",
                        )
                    ]
                )
        return []

    def _plan_event(
        self,
        state: _LedgerState,
        source: dict[str, Any],
        source_event_ids: list[str],
        projection_basis: str,
    ) -> dict[str, Any]:
        state.plan_emitted = True
        return self._derived_event(
            source=source,
            state=state,
            operation="plan",
            source_event_ids=source_event_ids,
            projection_basis=projection_basis,
            state_snapshot=self._state_snapshot(state),
        )

    @staticmethod
    def _state_snapshot(state: _LedgerState) -> dict[str, Any]:
        return {
            "consecutive_stalls": state.consecutive_stalls,
            "replans": state.replans,
        }

    def _derived_event(
        self,
        *,
        source: dict[str, Any],
        state: _LedgerState,
        operation: str,
        source_event_ids: list[str],
        **values: Any,
    ) -> dict[str, Any]:
        source_ids = [item for item in source_event_ids if item]
        identity = ":".join((PROJECTOR_ID, *source_ids, operation))
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        source_attributes = _attributes(source)
        binding = state.operation_bindings.get(operation) or {}
        operation_options = binding.get("options")
        operation_options = dict(operation_options) if isinstance(operation_options, dict) else {}
        attributes = {
            "operation": operation,
            "framework": "autogen",
            "operation_options": operation_options,
            "derived": True,
            "projector_id": PROJECTOR_ID,
            "projector_version": PROJECTOR_VERSION,
            "mapping_version": MAPPING_VERSION,
            "source_event_ids": source_ids,
            "operation_id": event_id,
            "source_artifact": source_attributes.get("source_artifact"),
            "source_line": source_attributes.get("source_line"),
            **values,
        }
        return {
            "schema_version": source.get("schema_version"),
            "run_id": source.get("run_id"),
            "event_id": event_id,
            "event_type": "coordination.operation.completed",
            "timestamp_unix_s": source.get("timestamp_unix_s"),
            "timestamp_utc": source.get("timestamp_utc"),
            "case_id": source.get("case_id"),
            "dataset_index": source.get("dataset_index"),
            "trial_index": source.get("trial_index"),
            "attempt": source.get("attempt"),
            "actor": state.controller_node_id,
            "receivers": [],
            "visible_to": [],
            "operation_id": event_id,
            "parent_event_id": source.get("event_id"),
            "correlation_id": state.group_chat_event_id,
            "status": "completed",
            "content": None,
            "attributes": attributes,
            "provenance": {
                "source_kind": "derived_projection",
                "source_artifact": source_attributes.get("source_artifact"),
                "source_line": source_attributes.get("source_line"),
                "source_record_id": source.get("event_id"),
                "source_event_ids": source_ids,
                "source_schema_version": source.get("schema_version"),
                "projector_id": PROJECTOR_ID,
                "projector_version": PROJECTOR_VERSION,
                "mapping_version": MAPPING_VERSION,
                "omitted_fields": [],
            },
        }

    def _record(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts = self._report["derived_event_type_counts"]
        for event in events:
            operation = str(_attributes(event).get("operation") or "unknown")
            counts[operation] = int(counts.get(operation) or 0) + 1
            self._report["derived_event_count"] += 1
        return events


__all__ = [
    "AutoGenMagenticOneLedgerProjector",
    "MAPPING_VERSION",
    "PROJECTOR_ID",
    "PROJECTOR_VERSION",
]
