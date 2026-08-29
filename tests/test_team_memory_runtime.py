from __future__ import annotations

import pytest
from lychee_mas.runtime.events.store import EVENT_TYPES
from lychee_mas.runtime.state import TeamMemoryRuntime


def _team_spec(*, mode: str = "chronological", max_items: int | None = 2) -> dict:
    return {
        "shared_state": [
            {
                "id": "WorkingMemory",
                "kind": "memory",
                "readers": ["Solver"],
                "writers": ["Researcher"],
                "lifetime": "trial",
                "initial": ["initial fact"],
                "retention": {
                    "max_items": max_items,
                    "max_tokens": None,
                    "overflow": "drop_oldest",
                },
                "memory": {
                    "retrieval": {
                        "mode": mode,
                        "top_k": 8,
                        "score_threshold": None,
                        "query_source": "task_and_node_input",
                    },
                    "write": {"policy": "node_output", "source": "source.output"},
                    "injection": {
                        "target": "model_context",
                        "template": "Memory {memory_id}:\n{items}",
                    },
                },
            }
        ]
    }


def test_trial_memory_enforces_access_and_retention() -> None:
    events: list[tuple[str, dict]] = []

    def log_event(event_type: str, **fields):
        events.append((event_type, fields))
        return f"event-{len(events)}"

    memory = TeamMemoryRuntime(_team_spec(), log_event=log_event)
    assert memory.write_node_output(
        "Solver", "not writable", invocation_id="solver-1"
    ) == ()
    assert memory.write_node_output(
        "Researcher", "first finding", invocation_id="researcher-1"
    ) == ("WorkingMemory",)
    assert memory.write_node_output(
        "Researcher", "second finding", invocation_id="researcher-2"
    ) == ("WorkingMemory",)

    recall = memory.recall(
        "Solver", task="question", node_input="candidate", invocation_id="solver-1"
    )
    assert recall.item_count == 2
    assert "initial fact" not in recall.model_context[0]
    assert "first finding" in recall.model_context[0]
    assert "second finding" in recall.model_context[0]
    assert memory.recall(
        "Researcher", task="question", node_input="", invocation_id="researcher-3"
    ).empty
    assert any(event_type == "memory.evicted" for event_type, _ in events)


def test_memory_write_is_idempotent_per_node_invocation() -> None:
    memory = TeamMemoryRuntime(_team_spec(max_items=None))
    assert memory.write_node_output("Researcher", "fact", invocation_id="same")
    assert not memory.write_node_output("Researcher", "fact", invocation_id="same")
    assert len(memory.snapshot()["WorkingMemory"]) == 2


def test_unsupported_semantic_memory_fails_instead_of_silent_fallback() -> None:
    with pytest.raises(ValueError, match="semantic index binding"):
        TeamMemoryRuntime(_team_spec(mode="semantic"))


def test_memory_event_namespace_is_registered() -> None:
    assert EVENT_TYPES.resolve("memory.recalled").category == "team_graph_memory"
