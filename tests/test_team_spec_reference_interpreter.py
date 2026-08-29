from __future__ import annotations

import asyncio
from typing import Any

from lychee_mas.core.types import Message
from lychee_mas.eval.teams.contracts import normalize_team_spec_document
from lychee_mas.runtime.contracts.node import NodeOutput
from lychee_mas.runtime.reference import TeamSpecReferenceInterpreter


def _transfer(source: str, target: str, *, required: bool = False) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "required": required,
        "view": "all",
        "latest_n": None,
        "filter": None,
        "transform": None,
        "schema": {},
    }


def _node(
    node_id: str,
    *,
    kind: str = "model_agent",
    operations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "name": node_id,
        "kind": kind,
        "behavior": {
            "type": "code_executor" if kind == "tool_executor" else "assistant",
            "options": {},
        },
        "purpose": node_id,
        "instructions": "",
        "capabilities": ["reason"] if kind == "model_agent" else ["execute_code"],
        "operations": operations or [],
        "tools": [],
        "context": {"type": "unbounded"},
        "limits": {"max_tool_iterations": 1},
    }


def _relation(
    relation_id: str,
    source: str,
    target: str,
    *,
    action: str = "activate",
    transfers: list[dict[str, Any]] | None = None,
    max_uses: int | None = None,
) -> dict[str, Any]:
    return {
        "id": relation_id,
        "from": source,
        "to": target,
        "control": {
            "trigger": "completed",
            "action": action,
            "condition": None,
            "priority": 0,
            "on_failure": "fail_trial",
            "max_uses": max_uses,
        },
        "data": {"transfers": transfers or [_transfer("trial.task", "task", required=True)]},
    }


def _state(
    state_id: str,
    *,
    kind: str,
    readers: list[str],
    writers: list[str],
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "id": state_id,
        "kind": kind,
        "description": "",
        "readers": readers,
        "writers": writers,
        "update": "append",
        "lifetime": "trial",
        "initial": [],
        "schema": {},
        "retention": {
            "max_items": None,
            "max_tokens": None,
            "overflow": "drop_oldest",
        },
    }
    if memory is not None:
        value["memory"] = memory
    return value


def _spec(
    *,
    nodes: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entry: str,
    submitter: str,
    shared_state: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return normalize_team_spec_document(
        {
            "schema_version": 14,
            "id": "reference-team",
            "metadata": {
                "name": "reference-team",
                "description": "",
                "tags": [],
                "provenance": {
                    "track": "custom",
                    "evidence_level": "hypothesis",
                    "sources": [],
                    "notes": "",
                },
            },
            "nodes": nodes,
            "relations": relations,
            "shared_state": shared_state or [],
            "lifecycle": {
                "entry": [
                    {
                        "node": entry,
                        "inputs": [_transfer("trial.task", "task", required=True)],
                    }
                ],
                "result": {
                    "submissions": [
                        {
                            "from": submitter,
                            "source": "source.output",
                            "key": "final_answer",
                        }
                    ],
                    "mode": "first_valid",
                    "schema": {},
                },
                "termination": {"condition": "result_submitted"},
                "failure": {
                    "unhandled": "fail_trial",
                    "deadlock": "fail_trial",
                },
                "limits": {
                    "max_turns": None,
                    "max_node_calls": None,
                    "max_stalls": None,
                    "timeout_seconds": None,
                },
            },
        }
    )


def test_reference_interpreter_executes_v14_result_contract() -> None:
    spec = _spec(nodes=[_node("Solver")], relations=[], entry="Solver", submitter="Solver")

    def solve(node_input):
        assert node_input.task == "2+2"
        return NodeOutput(
            node_id="Solver",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="4",
        )

    result = asyncio.run(TeamSpecReferenceInterpreter(spec, {"Solver": solve}).run("2+2"))

    assert result.status == "completed"
    assert result.result == "4"
    event_types = [event["event_type"] for event in result.events]
    assert event_types[:2] == ["run.started", "data_edge.transferred"]
    assert event_types[-2:] == ["result.submitted", "run.completed"]


def test_reference_interpreter_uses_explicit_v14_control_selection() -> None:
    selector = _node(
        "Selector",
        operations=[
            {
                "type": "select_next",
                "options": {
                    "mode": "deterministic",
                    "max_attempts": 1,
                    "allow_repeat": False,
                    "finish": {"allowed": False, "requires_result": True},
                },
            }
        ],
    )
    spec = _spec(
        nodes=[selector, _node("Worker")],
        relations=[
            _relation(
                "select-worker",
                "Selector",
                "Worker",
                action="select_next",
            )
        ],
        entry="Selector",
        submitter="Worker",
    )

    def select(node_input):
        return NodeOutput(
            node_id="Selector",
            invocation_id=node_input.invocation_id,
            status="completed",
            metadata={"selected_node_id": "Worker"},
        )

    def work(node_input):
        assert node_input.task == "task"
        return NodeOutput(
            node_id="Worker",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="done",
        )

    result = asyncio.run(
        TeamSpecReferenceInterpreter(
            spec, {"Selector": select, "Worker": work}
        ).run("task")
    )

    assert result.result == "done"
    selected = [
        event["payload"]["relation_id"]
        for event in result.events
        if event["event_type"] == "control_edge.selected"
    ]
    assert selected == ["select-worker"]


def test_reference_interpreter_materializes_shared_message_channel() -> None:
    conversation = _state(
        "Conversation",
        kind="message_channel",
        readers=["Consumer"],
        writers=["Producer"],
    )
    spec = _spec(
        nodes=[_node("Producer"), _node("Consumer")],
        relations=[
            _relation(
                "producer-consumer",
                "Producer",
                "Consumer",
                transfers=[
                    _transfer("trial.task", "task", required=True),
                    _transfer("shared.Conversation", "messages"),
                ],
            )
        ],
        entry="Producer",
        submitter="Consumer",
        shared_state=[conversation],
    )

    def produce(node_input):
        return NodeOutput(
            node_id="Producer",
            invocation_id=node_input.invocation_id,
            status="completed",
            message=Message(sender="Producer", content="evidence"),
        )

    def consume(node_input):
        assert [message.content for message in node_input.messages] == ["evidence"]
        return NodeOutput(
            node_id="Consumer",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="verified",
        )

    result = asyncio.run(
        TeamSpecReferenceInterpreter(
            spec, {"Producer": produce, "Consumer": consume}
        ).run("question")
    )

    assert result.result == "verified"
    assert result.shared_state["Conversation"][0].source_node == "Producer"


def test_reference_interpreter_injects_explicit_trial_memory() -> None:
    memory = _state(
        "TeamMemory",
        kind="memory",
        readers=["Consumer"],
        writers=["Producer"],
        memory={
            "retrieval": {
                "mode": "chronological",
                "top_k": 4,
                "score_threshold": None,
                "query_source": "task_and_node_input",
            },
            "write": {"policy": "node_output", "source": "source.output"},
            "injection": {
                "target": "model_context",
                "template": "Memory {memory_id}:\n{items}",
            },
        },
    )
    spec = _spec(
        nodes=[_node("Producer"), _node("Consumer")],
        relations=[_relation("next", "Producer", "Consumer")],
        entry="Producer",
        submitter="Consumer",
        shared_state=[memory],
    )

    def produce(node_input):
        return NodeOutput(
            node_id="Producer",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="remember this",
        )

    def consume(node_input):
        assert "remember this" in node_input.state["team_memory_model_context"][0]
        return NodeOutput(
            node_id="Consumer",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="recalled",
        )

    result = asyncio.run(
        TeamSpecReferenceInterpreter(
            spec, {"Producer": produce, "Consumer": consume}
        ).run("task")
    )

    assert result.result == "recalled"
    assert any(
        event["event_type"] == "memory.recalled" for event in result.events
    )


def test_reference_interpreter_honors_relation_max_uses() -> None:
    spec = _spec(
        nodes=[_node("Worker")],
        relations=[_relation("loop-once", "Worker", "Worker", max_uses=1)],
        entry="Worker",
        submitter="Worker",
    )
    calls = 0

    def work(node_input):
        nonlocal calls
        calls += 1
        return NodeOutput(
            node_id="Worker",
            invocation_id=node_input.invocation_id,
            status="completed",
            result="done" if calls == 2 else None,
        )

    result = asyncio.run(TeamSpecReferenceInterpreter(spec, {"Worker": work}).run("task"))

    assert result.status == "completed"
    assert result.result == "done"
    assert calls == 2
