"""Deterministic semantic oracle for canonical TeamSpec v14 documents.

The reference interpreter is not a fourth MAS framework. It executes the
observable Node, Relation, SharedState, and Lifecycle contracts with supplied
Node handlers. Framework adapters use the same fixtures to prove that their
native execution preserves the portable contract.
"""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from lychee_mas.core.types import Message
from lychee_mas.eval.teams.contracts import normalize_team_spec_document
from lychee_mas.runtime.contracts.node import NodeInput, NodeOutput
from lychee_mas.runtime.state.memory import TeamMemoryRuntime

NodeHandler = Callable[[NodeInput], NodeOutput | Awaitable[NodeOutput]]
EventSink = Callable[..., Any]


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {} and value != ()


@dataclass
class ReferenceDataItem:
    """One immutable value written to a v14 SharedState."""

    id: str
    semantic_type: str
    value: Any
    source_node: str
    source_path: str
    sequence: int
    consumed_by: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ReferenceExecutionResult:
    status: str
    steps: int
    result: Any
    shared_state: dict[str, tuple[ReferenceDataItem, ...]]
    node_outputs: tuple[NodeOutput, ...]
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Activation:
    priority: int
    sequence: int
    node_id: str
    control_signal: str
    relation_id: str | None
    transfers: tuple[dict[str, Any], ...]


class TeamSpecReferenceInterpreter:
    """Execute normalized TeamSpec v14 semantics without a model framework."""

    def __init__(
        self,
        team_spec: Mapping[str, Any],
        handlers: Mapping[str, NodeHandler],
        *,
        event_sink: EventSink | None = None,
        max_steps: int = 100,
        run_id: str = "reference-run",
    ) -> None:
        self.team_spec = normalize_team_spec_document(dict(team_spec))
        self.handlers = dict(handlers)
        self.event_sink = event_sink
        lifecycle_limit = (self.team_spec["lifecycle"].get("limits") or {}).get(
            "max_node_calls"
        )
        requested_limit = max(1, int(max_steps))
        self.max_steps = (
            min(requested_limit, int(lifecycle_limit))
            if lifecycle_limit
            else requested_limit
        )
        self.run_id = str(run_id)
        self.nodes = {str(node["id"]): dict(node) for node in self.team_spec["nodes"]}
        self.relations = [dict(item) for item in self.team_spec["relations"]]
        self.state_specs = {
            str(item["id"]): dict(item) for item in self.team_spec["shared_state"]
        }
        self._events: list[dict[str, Any]] = []
        self._event_sequence = 0
        self._invocation_sequence = 0
        self._data_sequence = 0
        self._ready_sequence = 0
        self._ready: deque[_Activation] = deque()
        self._outputs: list[NodeOutput] = []
        self._latest_output: dict[str, NodeOutput] = {}
        self._state_items: dict[str, list[ReferenceDataItem]] = {
            state_id: [] for state_id in self.state_specs
        }
        self._relation_uses: dict[str, int] = {}
        self._submitted: dict[str, Any] = {}
        self._result: Any = None
        self._cancelled = False
        self._memory: TeamMemoryRuntime | None = None

    async def run(self, task: str) -> ReferenceExecutionResult:
        self._reset()
        self._emit("run.started", team_spec_id=self.team_spec["id"])
        self._initialize_shared_state()
        self._memory = TeamMemoryRuntime(self.team_spec, log_event=self._emit)
        self._schedule_entries()

        steps = 0
        while self._ready and not self._is_complete() and not self._cancelled:
            if steps >= self.max_steps:
                self._emit("run.terminated", reason="max_steps", steps=steps)
                return self._execution_result("terminated", steps)
            activation = self._pop_ready()
            await self._invoke(activation, task=task)
            steps += 1

        if self._is_complete():
            self._emit("run.completed", steps=steps)
            return self._execution_result("completed", steps)
        if self._cancelled:
            self._emit("run.terminated", reason="cancelled", steps=steps)
            return self._execution_result("terminated", steps)

        deadlock = str(self.team_spec["lifecycle"]["failure"]["deadlock"])
        if deadlock == "submit_best_effort" and self._outputs:
            self._result = self._portable_output(self._outputs[-1])
            if _present(self._result):
                self._emit(
                    "result.submitted",
                    result=self._result,
                    submitter_node_id=self._outputs[-1].node_id,
                    best_effort=True,
                )
                self._emit("run.completed", steps=steps, best_effort=True)
                return self._execution_result("completed", steps)
        self._emit("run.terminated", reason="deadlock", steps=steps)
        return self._execution_result("terminated", steps)

    def _reset(self) -> None:
        self._events.clear()
        self._event_sequence = 0
        self._invocation_sequence = 0
        self._data_sequence = 0
        self._ready_sequence = 0
        self._ready.clear()
        self._outputs.clear()
        self._latest_output.clear()
        self._relation_uses.clear()
        self._submitted.clear()
        self._result = None
        self._cancelled = False
        self._memory = None
        for items in self._state_items.values():
            items.clear()

    def _initialize_shared_state(self) -> None:
        for state_id, state in self.state_specs.items():
            if state["kind"] == "memory":
                continue
            initial = state.get("initial")
            if isinstance(initial, list):
                values = initial
            elif isinstance(initial, dict):
                values = [initial] if initial else []
            else:
                values = [] if initial is None else [initial]
            for value in values:
                self._write_state(
                    state_id,
                    value,
                    source_node="team_spec.initial",
                    source_path="initial",
                    operation="initialize",
                )

    def _schedule_entries(self) -> None:
        for entry in self.team_spec["lifecycle"]["entry"]:
            self._schedule(
                node_id=str(entry["node"]),
                control_signal="trial_started",
                relation_id=None,
                transfers=tuple(dict(item) for item in entry.get("inputs") or []),
                priority=0,
            )

    async def _invoke(self, activation: _Activation, *, task: str) -> None:
        node_id = activation.node_id
        handler = self.handlers.get(node_id)
        if handler is None:
            raise ValueError(f"reference execution has no handler for Node {node_id!r}")
        invocation_id = self._next_invocation_id(node_id)
        materialized = self._materialize_inputs(activation, task=task)
        memory_recall = (
            self._memory.recall(
                node_id,
                task=str(materialized["task"]),
                node_input=str(materialized["data_by_port"]),
                invocation_id=invocation_id,
            )
            if self._memory
            else None
        )
        if memory_recall and memory_recall.model_context:
            materialized["state"]["team_memory_model_context"] = list(
                memory_recall.model_context
            )
        if memory_recall and memory_recall.node_input:
            materialized["data_by_port"]["team_memory"] = list(
                memory_recall.node_input
            )

        self._emit(
            "node_invocation.started",
            node_id=node_id,
            node_kind=str(self.nodes[node_id]["kind"]),
            invocation_id=invocation_id,
            control_signal=activation.control_signal,
            relation_id=activation.relation_id,
        )
        node_input = NodeInput(
            node_id=node_id,
            invocation_id=invocation_id,
            task=str(materialized["task"]),
            control_signal=activation.control_signal,
            messages=tuple(materialized["messages"]),
            state=dict(materialized["state"]),
            artifacts=tuple(materialized["artifacts"]),
            data_by_port=dict(materialized["data_by_port"]),
            metadata={
                "runtime": "team_spec_v14_reference_interpreter",
                "relation_id": activation.relation_id,
                "memory_ids": list(memory_recall.memory_ids) if memory_recall else [],
            },
        )
        try:
            value = handler(node_input)
            output = await value if inspect.isawaitable(value) else value
            if not isinstance(output, NodeOutput):
                raise TypeError(f"handler for {node_id!r} must return NodeOutput")
            if output.node_id != node_id or output.invocation_id != invocation_id:
                raise ValueError(
                    f"handler for {node_id!r} returned a mismatched NodeOutput"
                )
        except BaseException as exc:
            self._emit(
                "node_invocation.failed",
                node_id=node_id,
                node_kind=str(self.nodes[node_id]["kind"]),
                invocation_id=invocation_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            self._route_relations(node_id, trigger="failed", output=None)
            if self.team_spec["lifecycle"]["failure"]["unhandled"] != "continue":
                raise
            return

        self._outputs.append(output)
        self._latest_output[node_id] = output
        self._write_declared_state(node_id, output)
        if self._memory:
            self._memory.write_node_output(
                node_id,
                str(self._portable_output(output) or ""),
                invocation_id=invocation_id,
                metadata={"status": output.status},
            )
        self._emit(
            "node_invocation.completed",
            node_id=node_id,
            node_kind=str(self.nodes[node_id]["kind"]),
            invocation_id=invocation_id,
            status=output.status,
            emitted_control_signals=list(output.control_signals),
            emitted_data_ports=sorted(output.data_by_port),
        )
        if output.status == "failed":
            self._route_relations(node_id, trigger="failed", output=output)
            return

        submitted = self._collect_result(node_id, output)
        triggers = ["completed"]
        if _present(self._portable_output(output)):
            triggers.append("output_emitted")
        triggers.extend(str(item) for item in output.control_signals)
        if submitted:
            triggers.append("result_submitted")
        for trigger in dict.fromkeys(triggers):
            self._route_relations(node_id, trigger=trigger, output=output)

    def _materialize_inputs(self, activation: _Activation, *, task: str) -> dict[str, Any]:
        task_value: Any = task
        messages: list[Message] = []
        state: dict[str, Any] = {}
        artifacts: list[str] = []
        data_by_port: dict[str, Any] = {}
        source_node = None
        if activation.relation_id:
            relation = next(
                item
                for item in self.relations
                if str(item["id"]) == activation.relation_id
            )
            source_node = str(relation["from"])

        for transfer in activation.transfers:
            value = self._resolve_transfer_source(
                str(transfer["source"]), task=task, source_node=source_node
            )
            value = self._apply_view(value, transfer, source_node=source_node)
            if transfer.get("required") and not _present(value):
                raise ValueError(
                    f"Node {activation.node_id!r} is missing required source "
                    f"{transfer['source']!r}"
                )
            target = str(transfer["target"])
            if target == "task" and _present(value):
                task_value = value
            elif target == "messages":
                messages.extend(self._as_messages(value))
            elif target == "state":
                state_key = str(transfer["source"]).split(".", 1)[-1]
                state[state_key] = value
            elif target == "artifacts":
                values = value if isinstance(value, list) else [value]
                artifacts.extend(str(item) for item in values if item is not None)
            else:
                data_by_port[target] = value
            self._emit(
                "data_edge.transferred",
                relation_id=activation.relation_id,
                source=transfer["source"],
                target={"node": activation.node_id, "input": target},
                view=transfer.get("view"),
                required=bool(transfer.get("required")),
            )
        return {
            "task": task_value,
            "messages": messages,
            "state": state,
            "artifacts": artifacts,
            "data_by_port": data_by_port,
        }

    def _resolve_transfer_source(
        self, source: str, *, task: str, source_node: str | None
    ) -> Any:
        if source == "trial.task":
            return task
        if source.startswith("shared."):
            state_id = source.split(".", 1)[1]
            if state_id not in self._state_items:
                raise ValueError(f"unknown SharedState source {source!r}")
            return list(self._state_items[state_id])
        if not source.startswith("source."):
            raise ValueError(f"unsupported DataTransfer source {source!r}")
        if not source_node:
            raise ValueError(f"{source!r} requires a source Node")
        output = self._latest_output.get(source_node)
        if output is None:
            return None
        path = source.split(".")[1:]
        if path == ["output"]:
            return self._portable_output(output)
        if path == ["result"]:
            return output.result
        if path == ["message"]:
            return output.message
        if path == ["artifacts"]:
            return list(output.artifacts)
        if path and path[0] == "state":
            return self._walk(output.state_updates, path[1:])
        if path and path[0] in {"data", "port"}:
            return self._walk(output.data_by_port, path[1:])
        return self._walk(output.data_by_port, path)

    @staticmethod
    def _walk(value: Any, path: list[str]) -> Any:
        current = value
        for item in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(item)
        return current

    def _apply_view(
        self,
        value: Any,
        transfer: Mapping[str, Any],
        *,
        source_node: str | None,
    ) -> Any:
        items = value
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], ReferenceDataItem)
        ):
            records = list(value)
            if str(transfer.get("view") or "all") == "from_source" and source_node:
                records = [item for item in records if item.source_node == source_node]
            items = [item.value for item in records]
        view = str(transfer.get("view") or "all")
        if not isinstance(items, list):
            return items
        if view == "latest":
            return items[-1] if items else None
        if view == "latest_n":
            return items[-int(transfer.get("latest_n") or 1) :]
        if view == "summary":
            return "\n".join(map(str, items))
        return items

    @staticmethod
    def _as_messages(value: Any) -> list[Message]:
        values = value if isinstance(value, list) else ([] if value is None else [value])
        result: list[Message] = []
        for item in values:
            if isinstance(item, Message):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(
                    Message(
                        sender=str(
                            item.get("sender") or item.get("source") or "shared_state"
                        ),
                        content=str(
                            item.get("content") or item.get("text") or item
                        ),
                    )
                )
            else:
                result.append(Message(sender="shared_state", content=str(item)))
        return result

    def _write_declared_state(self, node_id: str, output: NodeOutput) -> None:
        for state_id, state in self.state_specs.items():
            if state["kind"] == "memory" or node_id not in state["writers"]:
                continue
            value: Any = None
            source_path = "source.output"
            if state["kind"] == "message_channel":
                value = output.message or self._portable_output(output)
            elif state["kind"] == "structured_state":
                value = output.state_updates.get(state_id, output.state_updates)
                source_path = "source.state"
            elif state["kind"] == "artifact_store":
                value = (
                    list(output.artifacts)
                    if output.artifacts
                    else output.data_by_port.get(state_id)
                )
                source_path = "source.artifacts"
            if _present(value):
                self._write_state(
                    state_id,
                    value,
                    source_node=node_id,
                    source_path=source_path,
                    operation=str(state["update"]),
                )

    def _write_state(
        self,
        state_id: str,
        value: Any,
        *,
        source_node: str,
        source_path: str,
        operation: str,
    ) -> None:
        state = self.state_specs[state_id]
        self._data_sequence += 1
        item = ReferenceDataItem(
            id=f"data-{self._data_sequence:06d}",
            semantic_type=str(state["kind"]),
            value=value,
            source_node=source_node,
            source_path=source_path,
            sequence=self._data_sequence,
        )
        self._emit(
            "data_item.created",
            data_item_id=item.id,
            semantic_type=item.semantic_type,
            source_node=source_node,
            source_path=source_path,
        )
        if operation in {"initialize", "replace", "commit"}:
            self._state_items[state_id] = [item]
        elif operation == "merge":
            previous = (
                self._state_items[state_id][-1].value
                if self._state_items[state_id]
                else {}
            )
            merged = dict(previous) if isinstance(previous, Mapping) else {}
            if isinstance(value, Mapping):
                merged.update(value)
            item.value = merged
            self._state_items[state_id] = [item]
        else:
            self._state_items[state_id].append(item)
        self._enforce_retention(state_id)
        self._emit(
            "data_item.stored",
            data_item_id=item.id,
            shared_state_id=state_id,
            operation=operation,
        )

    def _enforce_retention(self, state_id: str) -> None:
        state = self.state_specs[state_id]
        retention = dict(state.get("retention") or {})
        max_items = retention.get("max_items")
        items = self._state_items[state_id]
        if max_items is None or len(items) <= int(max_items):
            return
        overflow = str(retention.get("overflow") or "drop_oldest")
        if overflow == "drop_oldest":
            while len(items) > int(max_items):
                expired = items.pop(0)
                self._emit(
                    "data_item.expired",
                    data_item_id=expired.id,
                    shared_state_id=state_id,
                    reason="max_items",
                )
            return
        raise RuntimeError(f"SharedState {state_id!r} exceeded retention.max_items")

    def _collect_result(self, node_id: str, output: NodeOutput) -> bool:
        collected = False
        result_contract = self.team_spec["lifecycle"]["result"]
        for submission in result_contract["submissions"]:
            if str(submission["from"]) != node_id:
                continue
            value = self._submission_value(str(submission["source"]), output)
            if not _present(value):
                continue
            key = str(submission["key"])
            self._submitted[key] = value
            collected = True
            self._emit(
                "result.submitted",
                result=value,
                result_key=key,
                submitter_node_id=node_id,
                source=submission["source"],
            )
        mode = str(result_contract["mode"])
        if mode == "first_valid" and self._submitted:
            self._result = next(iter(self._submitted.values()))
        elif mode == "all" and len(self._submitted) == len(
            result_contract["submissions"]
        ):
            self._result = dict(self._submitted)
        elif mode == "aggregate" and self._submitted:
            self._result = dict(self._submitted)
        return collected

    def _submission_value(self, source: str, output: NodeOutput) -> Any:
        if source == "source.output":
            return self._portable_output(output)
        if source == "source.result":
            return output.result
        if source == "source.message":
            return output.message.content if output.message else None
        if source.startswith("source.state."):
            return self._walk(output.state_updates, source.split(".")[2:])
        if source.startswith("source.data."):
            return self._walk(output.data_by_port, source.split(".")[2:])
        return None

    @staticmethod
    def _portable_output(output: NodeOutput | None) -> Any:
        if output is None:
            return None
        if output.message is not None:
            return output.message.content
        if "response" in output.data_by_port:
            return output.data_by_port["response"]
        if output.result is not None:
            return output.result
        if len(output.data_by_port) == 1:
            return next(iter(output.data_by_port.values()))
        return None

    def _route_relations(
        self,
        node_id: str,
        *,
        trigger: str,
        output: NodeOutput | None,
    ) -> None:
        candidates = [
            relation
            for relation in self.relations
            if str(relation["from"]) == node_id
            and relation.get("control")
            and str(relation["control"].get("trigger") or "completed")
            in {trigger, "always"}
            and self._relation_available(relation)
            and self._condition_allows(relation, output=output)
        ]
        by_action: dict[str, list[dict[str, Any]]] = {}
        for relation in candidates:
            by_action.setdefault(str(relation["control"]["action"]), []).append(relation)
        metadata = output.metadata if output else {}
        selected_node = str(metadata.get("selected_node_id") or "")
        selected_relation = str(metadata.get("selected_relation_id") or "")
        for action, relations in by_action.items():
            ordered = sorted(
                relations,
                key=lambda item: (
                    int(item["control"].get("priority") or 0),
                    str(item["id"]),
                ),
            )
            if action in {"select_next", "delegate", "handoff"}:
                if selected_relation:
                    ordered = [
                        item for item in ordered if str(item["id"]) == selected_relation
                    ]
                elif selected_node:
                    ordered = [
                        item for item in ordered if str(item["to"]) == selected_node
                    ]
                elif len(ordered) > 1:
                    raise ValueError(
                        f"Node {node_id!r} must select one {action!r} Relation"
                    )
                else:
                    ordered = ordered[:1]
            for relation in ordered:
                if action == "finish":
                    continue
                if action == "cancel":
                    self._cancelled = True
                    continue
                self._select_relation(relation)

    def _relation_available(self, relation: Mapping[str, Any]) -> bool:
        maximum = (relation.get("control") or {}).get("max_uses")
        return maximum is None or self._relation_uses.get(
            str(relation["id"]), 0
        ) < int(maximum)

    def _condition_allows(
        self, relation: Mapping[str, Any], *, output: NodeOutput | None
    ) -> bool:
        condition = dict((relation.get("control") or {}).get("condition") or {})
        if not condition:
            return True
        source = str(condition["source"])
        if source == "source.output":
            value = self._portable_output(output)
        elif source.startswith("source.state."):
            value = self._walk(
                output.state_updates if output else {}, source.split(".")[2:]
            )
        elif source.startswith("shared."):
            state_id = source.split(".", 1)[1]
            values = self._state_items.get(state_id, [])
            value = values[-1].value if values else None
        else:
            value = None
        operator = str(condition["operator"])
        expected = condition.get("value")
        operations = {
            "eq": lambda: value == expected,
            "ne": lambda: value != expected,
            "in": lambda: value in expected if expected is not None else False,
            "not_in": lambda: value not in expected if expected is not None else True,
            "exists": lambda: value is not None,
            "truthy": lambda: bool(value),
            "falsy": lambda: not bool(value),
            "lt": lambda: value is not None and value < expected,
            "lte": lambda: value is not None and value <= expected,
            "gt": lambda: value is not None and value > expected,
            "gte": lambda: value is not None and value >= expected,
        }
        return bool(operations[operator]())

    def _select_relation(self, relation: Mapping[str, Any]) -> None:
        relation_id = str(relation["id"])
        self._relation_uses[relation_id] = self._relation_uses.get(relation_id, 0) + 1
        control = dict(relation["control"])
        self._emit(
            "control_edge.selected",
            relation_id=relation_id,
            source_node_id=str(relation["from"]),
            target_node_id=str(relation["to"]),
            action=str(control["action"]),
            priority=int(control.get("priority") or 0),
            use_count=self._relation_uses[relation_id],
        )
        self._schedule(
            node_id=str(relation["to"]),
            control_signal=str(control["action"]),
            relation_id=relation_id,
            transfers=tuple(
                dict(item)
                for item in (relation.get("data") or {}).get("transfers") or []
            ),
            priority=int(control.get("priority") or 0),
        )

    def _schedule(
        self,
        *,
        node_id: str,
        control_signal: str,
        relation_id: str | None,
        transfers: tuple[dict[str, Any], ...],
        priority: int,
    ) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"activation references unknown Node {node_id!r}")
        self._ready_sequence += 1
        self._ready.append(
            _Activation(
                priority=priority,
                sequence=self._ready_sequence,
                node_id=node_id,
                control_signal=control_signal,
                relation_id=relation_id,
                transfers=transfers,
            )
        )

    def _pop_ready(self) -> _Activation:
        selected = min(self._ready, key=lambda item: (item.priority, item.sequence))
        self._ready.remove(selected)
        return selected

    def _is_complete(self) -> bool:
        return self._result is not None

    def _next_invocation_id(self, node_id: str) -> str:
        self._invocation_sequence += 1
        return f"{node_id}-{self._invocation_sequence:06d}"

    def _emit(self, event_type: str, **payload: Any) -> str:
        self._event_sequence += 1
        event = {
            "schema_version": 2,
            "run_id": self.run_id,
            "sequence": self._event_sequence,
            "event_type": event_type,
            "payload": payload,
        }
        self._events.append(event)
        if self.event_sink is not None:
            self.event_sink(event_type, **payload)
        return f"reference-event-{self._event_sequence:06d}"

    def _execution_result(
        self, status: str, steps: int
    ) -> ReferenceExecutionResult:
        return ReferenceExecutionResult(
            status=status,
            steps=steps,
            result=self._result,
            shared_state={
                state_id: tuple(items)
                for state_id, items in self._state_items.items()
            },
            node_outputs=tuple(self._outputs),
            events=tuple(self._events),
        )


__all__ = [
    "ReferenceDataItem",
    "ReferenceExecutionResult",
    "TeamSpecReferenceInterpreter",
]
