"""Contract tests for framework-neutral TeamSpec RuntimeAdapters."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.core.types import TaskQuery
from lychee_mas.eval.teams.compiler import load_team_spec
from lychee_mas.eval.teams.contracts import normalize_team_spec_document
from lychee_mas.memory.context import RoutingContext
from lychee_mas.memory.routing.static import fixed_channel_router
from lychee_mas.runtime.adapters.frameworks.autogen.coordination import (
    termination_condition as autogen_termination_condition,
)
from lychee_mas.runtime.adapters.frameworks.bindings import compile_node_bindings
from lychee_mas.runtime.coordination.compiler import (
    compile_coordination,
    coordination_topology,
    message_policies,
    message_sources,
    operation_options,
    parse_next_node_decision,
    runtime_support,
)
from lychee_mas.runtime.coordination.ledger_orchestration import (
    LedgerOrchestrationProtocol,
    parse_progress_ledger,
)
from lychee_mas.runtime.events.store import RunEventWriter, iter_run_events, run_events_path
from lychee_mas.runtime.model.gateway import GatewayReply, ModelGateway
from lychee_mas.runtime.results.contract import (
    result_messages,
    result_source,
    result_submitter_aliases,
    validate_result_contract,
)

_BENCHMARKS = ("gaia", "bbeh", "swe-bench-verified", "workbench")
_TOPOLOGIES = ("independent", "sequential", "centralized", "decentralized")
_FRAMEWORKS = ("autogen", "langgraph", "crewai")


def test_autogen_result_submitted_condition_stops_only_on_approved_submitter() -> None:
    from autogen_agentchat.messages import TextMessage

    events: list[tuple[str, dict]] = []
    team = SimpleNamespace(
        meta={
            "result_contract": {
                "submitters": ["Reviewer"],
                "conditions": [{"type": "result_submitted"}],
            }
        }
    )
    ctx = SimpleNamespace(log_event=lambda name, **fields: events.append((name, fields)))
    condition = autogen_termination_condition(team, ctx)

    assert condition is not None
    assert asyncio.run(condition([TextMessage(content="draft", source="Implementer")])) is None
    stop = asyncio.run(
        condition([TextMessage(content="```python\nprint(1)\n```", source="Reviewer")])
    )

    assert stop is not None
    assert stop.source == "ResultSubmittedTermination"
    assert events == [
        (
            "result.submission_detected",
            {
                "source": "Reviewer",
                "message_type": "TextMessage",
                "submitters": ["Reviewer"],
                "origin": "autogen_termination_adapter",
            },
        )
    ]


def test_autogen_multimodal_payload_accepts_gif_data_uri() -> None:
    import base64
    from io import BytesIO

    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import AutoGenRuntime
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="GIF")
    query = TaskQuery(
        question="Inspect the image.",
        meta={
            "multimodal_content": [
                {"type": "text", "text": "Inspect the image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/gif;base64,"
                        + base64.b64encode(buffer.getvalue()).decode()
                    },
                },
            ]
        },
    )

    payload = AutoGenRuntime._build_task_payload(query, query.question)

    assert len(payload.content) == 2
    assert payload.content[0] == "Inspect the image."
    assert type(payload.content[1]).__name__ == "Image"


@pytest.mark.parametrize("benchmark", _BENCHMARKS)
@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_cross_framework_matrix_team_specs_compile(
    benchmark: str,
    topology: str,
) -> None:
    path = Path(f"configs/eval_studio/teams/specs/{benchmark}-{topology}.json")
    graph, document = load_team_spec(path, rounds=1)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    coordination = graph.meta["coordination_ir"]
    assert coordination_topology(coordination) == topology
    assert graph.nodes
    assert [node.name for node in graph.nodes] == coordination["ordered_members"]
    assert "pattern" not in coordination
    assert set(graph.meta["adapter_plans"]) == set(_FRAMEWORKS)
    provenance = document["metadata"]["provenance"]
    assert provenance["track"] == "controlled_portability"
    assert provenance["sources"]
    assert set(document) == {
        "schema_version",
        "id",
        "metadata",
        "nodes",
        "relations",
        "shared_state",
        "lifecycle",
    }
    assert compile_coordination(document) == coordination
    assert document["lifecycle"]["result"]["submissions"]
    assert document["shared_state"]
    if len(coordination["members"]) > 1:
        assert any(
            relation.get("data")
            and any(
                transfer["target"] == "messages"
                for transfer in relation["data"]["transfers"]
            )
            for relation in document["relations"]
        )
    assert persisted["schema_version"] == 14
    assert all("agent_type" not in node for node in persisted["nodes"])
    assert all(
        {"id", "from", "to"} <= set(relation)
        and bool(relation.get("control") or relation.get("data"))
        for relation in persisted["relations"]
    )


def test_gaia_node_bindings_are_compiled_from_semantics_not_names() -> None:
    path = Path("configs/eval_studio/teams/specs/gaia.json")
    document = normalize_team_spec_document(json.loads(path.read_text(encoding="utf-8")))
    coordination = compile_coordination(document)
    original = compile_node_bindings("autogen", document, coordination)

    renamed = deepcopy(document)
    for index, node in enumerate(renamed["nodes"]):
        node["name"] = f"Display Name {index}"
    renamed_coordination = compile_coordination(renamed)

    assert compile_node_bindings("autogen", renamed, renamed_coordination) == original
    assert {
        item["node_id"]: item["specialization"] for item in original
    } == {
        "Coder": "magentic_one_coder",
        "ComputerTerminal": "code_executor",
        "FileSurfer": "file_surfer",
        "WebSurfer": "multimodal_web_surfer",
        "Orchestrator": "magentic_one_orchestrator",
    }


def test_team_spec_rejects_framework_shaped_agent_type() -> None:
    path = Path("configs/eval_studio/teams/specs/gaia.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    model_node = next(
        node
        for node in document["nodes"]
        if node.get("kind") == "model_agent"
    )
    model_node["agent_type"] = "coder"

    with pytest.raises(ValueError, match="unsupported fields: agent_type"):
        normalize_team_spec_document(document)


def test_control_relations_never_grant_implicit_message_visibility() -> None:
    graph, document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    without_messages = deepcopy(document)
    for state in without_messages["shared_state"]:
        if state["kind"] == "message_channel":
            state["readers"] = []
            state["writers"] = []
    for relation in without_messages["relations"]:
        if relation.get("data"):
            relation["data"]["transfers"] = [
                transfer
                for transfer in relation["data"]["transfers"]
                if transfer["target"] != "messages"
            ]

    coordination = graph.meta["coordination_ir"]
    sources = message_sources(without_messages, coordination["members"])

    assert sources == {member: [] for member in coordination["members"]}


def test_autogen_composes_bounded_handoff_from_explicit_team_contract() -> None:
    from autogen_agentchat.messages import TextMessage
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        bounded_handoff_selector,
    )

    graph, _ = load_team_spec(
        "configs/eval_studio/teams/specs/swe-bench-verified-decentralized.json",
        rounds=1,
    )
    plan = graph.meta["adapter_plans"]["autogen"]

    assert plan["mapping_level"] == "composed"
    assert plan["config"]["type"] == "selector"
    assert plan["config"]["selector_func_factory"] == "bounded_handoff_selector"

    selector = bounded_handoff_selector(graph)
    messages = [TextMessage(content="task", source="user")]
    assert selector(messages) == "Investigator"
    messages.append(TextMessage(content="root cause", source="Investigator"))
    assert selector(messages) == "Implementer"
    messages.append(TextMessage(content="patch", source="Implementer"))
    assert selector(messages) == "Reviewer"
    messages.append(TextMessage(content="gap", source="Reviewer"))
    assert selector(messages) == "Investigator"


def test_autogen_magentic_one_renders_framework_neutral_result_prompt() -> None:
    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        _render_magentic_one_final_answer_prompt,
    )

    rendered = _render_magentic_one_final_answer_prompt(
        "Task: {task}\nEvidence: {history}\nReturn the shortest answer.",
        "How many?",
    )

    assert "Task: How many?" in rendered
    assert "complete team transcript" in rendered
    assert "{history}" not in rendered


def test_executor_message_sources_come_from_data_edges_not_control_edges() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia.json",
        rounds=1,
    )
    terminal = next(node for node in graph.nodes if node.name == "ComputerTerminal")

    assert "Coder" in terminal.meta["sources"]
    assert "Orchestrator" in terminal.meta["control_predecessors"]


def test_message_policies_preserve_direction_and_history() -> None:
    members = ["A", "B", "C"]
    document = {
        "relations": [
            {
                "id": "A-to-B",
                "from": "A",
                "to": "B",
                "data": {
                    "transfers": [
                        {
                            "source": "source.output",
                            "target": "messages",
                            "view": "latest",
                        }
                    ]
                },
            },
        ],
        "shared_state": [],
    }

    policies = message_policies(document, members)

    assert policies["A"] == {}
    assert policies["B"] == {
        "A": {
            "delivery": "immediate",
            "history": "latest",
            "content_types": ["multimodal", "text", "tool_call", "tool_result"],
        }
    }
    assert policies["C"] == {}


@pytest.mark.parametrize("framework", _FRAMEWORKS)
@pytest.mark.parametrize("benchmark", _BENCHMARKS)
@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_cross_framework_matrix_has_explicit_binding_report(
    framework: str,
    benchmark: str,
    topology: str,
) -> None:
    graph, document = load_team_spec(
        f"configs/eval_studio/teams/specs/{benchmark}-{topology}.json",
        rounds=1,
    )

    report = runtime_support(framework, graph.meta["coordination_ir"], document)

    assert report["supported"] is True
    assert report["framework"] == framework
    assert report["adapter_strategy"] == graph.meta["adapter_plans"][framework]["strategy"]
    assert report["implementation"]
    assert report["mapping_level"] in {"exact", "composed", "approximated"}
    assert isinstance(report["semantic_deltas"], list)


def test_binding_report_marks_unexecuted_relation_semantics() -> None:
    graph, document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    transfer = next(
        transfer
        for relation in document["relations"]
        for transfer in (relation.get("data") or {}).get("transfers") or []
        if transfer["target"] == "state"
    )
    transfer["filter"] = {"content_type": "artifact"}

    report = runtime_support("autogen", graph.meta["coordination_ir"], document)

    assert report["mapping_level"] == "approximated"
    assert report["controlled_comparison_eligible"] is False
    assert any("DataTransfer filters" in item for item in report["semantic_deltas"])


def test_binding_report_marks_unsupported_execution_kind() -> None:
    graph, document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-independent.json",
        rounds=1,
    )
    member_id = graph.meta["coordination_ir"]["members"][0]
    member = next(node for node in document["nodes"] if node["id"] == member_id)
    member["kind"] = "remote"

    report = runtime_support("langgraph", graph.meta["coordination_ir"], document)

    assert report["mapping_level"] == "approximated"
    assert report["controlled_comparison_eligible"] is False
    assert any("remote" in item for item in report["semantic_deltas"])


def test_binding_report_accepts_composed_coordination_operation() -> None:
    graph, document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia-centralized.json",
        rounds=1,
    )

    report = runtime_support("crewai", graph.meta["coordination_ir"], document)

    assert report["mapping_level"] == "composed"
    assert report["controlled_comparison_eligible"] is True
    assert report["semantic_deltas"] == []


def test_coordination_operations_are_declared_not_inferred_from_capabilities() -> None:
    path = Path("configs/eval_studio/teams/specs/gaia.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    orchestrator = next(node for node in document["nodes"] if node["id"] == "Orchestrator")

    assert {item["type"] for item in orchestrator["operations"]} == {
        "plan",
        "delegate",
        "monitor_progress",
        "detect_stall",
        "replan",
        "aggregate",
    }
    orchestrator["operations"] = []
    normalized = normalize_team_spec_document(document)
    assert not compile_coordination(normalized)["operations"]


@pytest.mark.parametrize("framework", ["langgraph", "crewai"])
def test_binding_report_accepts_composed_full_orchestration(framework: str) -> None:
    graph, document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia.json",
        rounds=1,
    )

    report = runtime_support(framework, graph.meta["coordination_ir"], document)

    assert report["mapping_level"] == "composed"
    assert report["controlled_comparison_eligible"] is True
    assert report["semantic_deltas"] == []


def _progress_ledger(*, next_speaker: str = "Coder") -> str:
    return json.dumps(
        {
            "is_request_satisfied": {"reason": "work remains", "answer": False},
            "is_in_loop": {"reason": "new work", "answer": False},
            "is_progress_being_made": {"reason": "yes", "answer": True},
            "next_speaker": {"reason": "code is needed", "answer": next_speaker},
            "instruction_or_question": {
                "reason": "delegate one bounded step",
                "answer": "Implement the calculation.",
            },
        }
    )


def test_progress_ledger_parser_requires_one_exact_unmodified_json_object() -> None:
    decision = parse_progress_ledger(
        _progress_ledger(),
        candidates=["Coder", "WebSurfer"],
    )

    assert decision.next_node == "Coder"
    assert decision.making_progress is True
    with pytest.raises(json.JSONDecodeError):
        parse_progress_ledger(
            f"```json\n{_progress_ledger()}\n```",
            candidates=["Coder"],
        )
    with pytest.raises(ValueError, match="unknown Node"):
        parse_progress_ledger(
            _progress_ledger(next_speaker="Unknown"),
            candidates=["Coder"],
        )
    malformed = json.loads(_progress_ledger())
    malformed["instruction_or_question"] = "Implement the calculation."
    with pytest.raises(ValueError, match="instruction_or_question"):
        parse_progress_ledger(json.dumps(malformed), candidates=["Coder"])


def test_ledger_protocol_retries_without_repairing_model_output() -> None:
    class SequenceGateway:
        def __init__(self) -> None:
            self.replies = ["not-json", _progress_ledger()]
            self.messages: list[list[dict]] = []
            self.json_output: list[object] = []

        async def invoke(self, _role, messages, **kwargs):
            self.messages.append([dict(item) for item in messages])
            self.json_output.append(kwargs.get("json_output"))
            return GatewayReply(content=self.replies.pop(0))

    gateway = SequenceGateway()
    protocol = LedgerOrchestrationProtocol(
        gateway=gateway,  # type: ignore[arg-type]
        controller_node_id="Orchestrator",
        task="Solve the task",
        candidates=["Coder"],
        team_description="Coder writes code",
        parse_attempts=2,
    )

    decision = asyncio.run(protocol.decide([]))

    assert decision.next_node == "Coder"
    assert len(gateway.messages) == 2
    assert all(
        isinstance(schema, type) and schema.__name__ == "_ProgressLedgerSchema"
        for schema in gateway.json_output
    )
    assert gateway.messages[1][1] == {"role": "assistant", "content": "not-json"}
    assert "do not use markdown fences" in gateway.messages[1][2]["content"].lower()
    assert (
        '"instruction_or_question":{"reason":"...","answer":"..."}'
        in (gateway.messages[1][2]["content"])
    )


def test_ledger_protocol_emits_declared_operation_evidence() -> None:
    class Gateway:
        async def invoke(self, _role, messages, **_kwargs):
            prompt = str(messages[-1]["content"])
            if "Assess progress" in prompt:
                return GatewayReply(content=_progress_ledger())
            if "Produce a concise executable plan" in prompt:
                return GatewayReply(content="Use Coder, then aggregate.")
            return GatewayReply(content="Known facts")

    events: list[tuple[str, dict]] = []

    def log_event(event_type: str, **payload):
        events.append((event_type, payload))

    protocol = LedgerOrchestrationProtocol(
        gateway=Gateway(),  # type: ignore[arg-type]
        controller_node_id="Orchestrator",
        task="Solve the task",
        candidates=["Coder"],
        team_description="Coder writes code",
        operation_options={
            "plan": {"mode": "model"},
            "delegate": {"mode": "model"},
            "monitor_progress": {"mode": "model"},
            "detect_stall": {"mode": "model", "window": 3},
        },
        event_logger=log_event,
        framework="langgraph",
    )

    asyncio.run(protocol.initialize())
    asyncio.run(protocol.decide([]))

    assert [payload["operation"] for _, payload in events] == [
        "plan",
        "monitor_progress",
        "detect_stall",
        "delegate",
    ]
    assert {event_type for event_type, _ in events} == {"coordination.operation.completed"}
    assert events[-1][1]["next_node"] == "Coder"


def test_result_contract_only_accepts_configured_submitter_messages() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    messages = [
        SimpleNamespace(source="Analyst", content="draft"),
        SimpleNamespace(source="Solver", content="candidate"),
        SimpleNamespace(source="Verifier", content="The final answer is: 42"),
    ]

    assert result_submitter_aliases(graph) >= {"verifier"}
    assert [item.content for item in result_messages(graph, messages)] == [
        "The final answer is: 42"
    ]
    assert result_source(graph, messages) == "Verifier"


def test_magentic_one_framework_source_maps_to_abstract_orchestrator() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia.json",
        rounds=1,
    )
    message = SimpleNamespace(
        source="MagenticOneOrchestrator",
        content="The final answer is: 17",
    )

    assert graph.meta["operation_nodes"]["aggregate"]["id"] == "Orchestrator"
    assert "magenticoneorchestrator" in result_submitter_aliases(graph)
    assert result_messages(graph, [message]) == [message]
    assert result_source(graph, [message]) == "MagenticOneOrchestrator"


def test_result_contract_does_not_substitute_an_unapproved_role() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    messages = [SimpleNamespace(source="Solver", content="candidate")]

    assert result_messages(graph, messages) == []
    assert result_source(graph, messages) is None


def test_text_result_contract_requires_an_approved_non_empty_submission() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )

    invalid = validate_result_contract(
        graph,
        [SimpleNamespace(source="Solver", content="42")],
        result_kind="text",
        final_content="",
    )
    valid = validate_result_contract(
        graph,
        [SimpleNamespace(source="Verifier", content="The final answer is: 42")],
        result_kind="text",
        final_content="42",
    )
    json_empty = validate_result_contract(
        graph,
        [SimpleNamespace(source="Verifier", content='""')],
        result_kind="text",
        final_content='""',
    )

    assert invalid.valid is False
    assert json_empty.valid is False
    assert json_empty.payload_empty is True
    assert valid.valid is True
    assert valid.source == "Verifier"


def test_action_and_patch_contracts_do_not_require_text_messages() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/workbench-independent.json",
        rounds=1,
    )

    action = validate_result_contract(
        graph,
        [],
        result_kind="action",
        final_content="",
        tool_requests=[],
    )
    patch = validate_result_contract(
        graph,
        [],
        result_kind="patch",
        final_content="",
        workspace="/tmp/workspace",
    )

    assert action.valid is True
    assert patch.valid is True


class _Memory:
    def reset(self) -> None:
        pass


class _Backend:
    model_name = "fake-model"
    tok = None
    context_window_tokens = 4096

    def generate_chat(self, messages, **_kwargs):
        choosing = any(
            "Choose the next participant" in str(message.get("content")) for message in messages
        )
        text = "FINISH" if choosing else "The answer is 42. TERMINATE"
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=[],
        )


class _NeverSatisfiedLedgerBackend(_Backend):
    def generate_chat(self, messages, **_kwargs):
        prompt = str(messages[-1].get("content") or "")
        if "Assess progress and choose the next Node" in prompt:
            text = _progress_ledger(next_speaker="Coder")
        elif (
            "Provide the final answer to the original request" in prompt
            or "shortest final answer string" in prompt
        ):
            text = "42"
        elif "Produce a concise executable plan" in prompt:
            text = "Ask Coder to solve the task, then aggregate the result."
        elif "Analyze the request before delegating work" in prompt:
            text = "GIVEN OR VERIFIED FACTS: the task asks for 40 + 2."
        else:
            text = "Coder found that 40 + 2 = 42."
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=[],
        )


@pytest.mark.parametrize(
    ("dependency", "runtime_module", "runtime_class"),
    [
        (
            "langgraph",
            "lychee_mas.runtime.adapters.frameworks.langgraph.runtime",
            "LangGraphRuntime",
        ),
        ("crewai", "lychee_mas.runtime.adapters.frameworks.crewai.runtime", "CrewAIRuntime"),
    ],
)
def test_portable_sequential_team_runs_through_framework_adapter(
    dependency: str,
    runtime_module: str,
    runtime_class: str,
) -> None:
    pytest.importorskip(dependency)
    module = __import__(runtime_module, fromlist=[runtime_class])
    runtime_type = getattr(module, runtime_class)
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    runtime = runtime_type(
        backend=_Backend(),
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        model_id="fake-model",
        memory_method="none",
    )

    trajectory = asyncio.run(
        runtime.run(
            graph,
            TaskQuery(id="case-1", question="What is 40 + 2?", gold="42"),
        )
    )

    assert [message.sender for message in trajectory.messages] == [
        node.name for node in graph.nodes
    ]
    assert trajectory.final_answer.content == "The answer is 42. TERMINATE"
    assert trajectory.meta["framework"] == dependency
    assert trajectory.meta["framework_implementation"]
    assert context.model_calls_started == len(graph.nodes)
    assert len(trajectory.meta["decisions"]) == len(graph.nodes)
    assert sum(item["input_positions"] for item in context.decisions) == len(graph.nodes) * 20


@pytest.mark.parametrize(
    ("dependency", "runtime_module", "runtime_class"),
    [
        (
            "langgraph",
            "lychee_mas.runtime.adapters.frameworks.langgraph.runtime",
            "LangGraphRuntime",
        ),
        ("crewai", "lychee_mas.runtime.adapters.frameworks.crewai.runtime", "CrewAIRuntime"),
    ],
)
def test_composed_orchestration_aggregates_when_participant_turn_limit_is_reached(
    dependency: str,
    runtime_module: str,
    runtime_class: str,
) -> None:
    pytest.importorskip(dependency)
    module = __import__(runtime_module, fromlist=[runtime_class])
    runtime_type = getattr(module, runtime_class)
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia.json",
        rounds=4,
    )
    graph.nodes = [node for node in graph.nodes if node.name == "Coder"]
    graph.edges = {"Coder": []}
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    runtime = runtime_type(
        backend=_NeverSatisfiedLedgerBackend(),
        ctx=context,
        max_new_tokens=128,
        max_rounds=4,
        max_turns=1,
        max_model_calls_per_case=16,
        model_id="fake-model",
        code_executor="local",
        memory_method="none",
    )

    trajectory = asyncio.run(
        runtime.run(
            graph,
            TaskQuery(id="case-ledger-limit", question="What is 40 + 2?", gold="42"),
        )
    )

    assert trajectory.final_answer.content == "42"
    assert trajectory.final_answer.source == "Orchestrator"
    assert trajectory.meta["result_contract"]["valid"] is True
    assert context.model_calls_started == 5


def test_langgraph_turn_limit_does_not_depend_on_last_participant_round_count() -> None:
    pytest.importorskip("langgraph")
    from lychee_mas.runtime.adapters.frameworks.langgraph.runtime import LangGraphRuntime

    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia.json",
        rounds=1,
    )
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    runtime = LangGraphRuntime(
        backend=_Backend(),
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        max_turns=5,
        model_id="fake-model",
        memory_method="none",
    )

    # ``rounds`` used to increase only when the final participant in the Node
    # list ran, so it was not a valid framework-neutral termination counter.
    assert runtime._should_stop(graph, {"steps": 2, "rounds": 99}, 5) is False
    assert runtime._should_stop(graph, {"steps": 5, "rounds": 0}, 5) is True


class _ExplicitHandoffBackend(_Backend):
    def __init__(self) -> None:
        self._turn = 0

    def generate_chat(self, messages, **_kwargs):
        self._turn += 1
        if self._turn == 1:
            text = "Analysis complete. HANDOFF: Solver"
        elif self._turn == 2:
            text = "Candidate complete. HANDOFF: Verifier"
        else:
            text = "The final answer is: 42 TERMINATE"
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=[],
        )


class _ImplicitHandoffBackend(_Backend):
    def generate_chat(self, messages, **_kwargs):
        text = "The current step is complete."
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=[],
        )


def test_handoff_operation_fallback_is_distinct_from_relation_failure_policy() -> None:
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/livecodebench-decentralized.json",
        rounds=1,
    )
    coordination = graph.meta["coordination_ir"]

    assert operation_options(
        coordination,
        "handoff",
        node_id="Implementer",
    )["fallback"] == "next_priority"
    implementer_relation = next(
        relation
        for relation in coordination["handoff_relations"]
        if relation["source"] == "Implementer"
    )
    assert implementer_relation["contract"]["on_failure"] == "fail_trial"


@pytest.mark.parametrize(
    ("dependency", "runtime_module", "runtime_class"),
    [
        (
            "langgraph",
            "lychee_mas.runtime.adapters.frameworks.langgraph.runtime",
            "LangGraphRuntime",
        ),
        ("crewai", "lychee_mas.runtime.adapters.frameworks.crewai.runtime", "CrewAIRuntime"),
    ],
)
def test_handoff_operation_uses_next_priority_when_model_omits_choice(
    dependency: str,
    runtime_module: str,
    runtime_class: str,
) -> None:
    pytest.importorskip(dependency)
    module = __import__(runtime_module, fromlist=[runtime_class])
    runtime_type = getattr(module, runtime_class)
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/livecodebench-decentralized.json",
        rounds=1,
    )
    context = RoutingContext(
        task="livecodebench",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    backend = _ImplicitHandoffBackend()
    runtime = runtime_type(
        backend=backend,
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        max_turns=3,
        model_id="fake-model",
        memory_method="none",
    )
    runtime._current_tool_requests = []
    runtime._current_tool_executions = []
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework=dependency,
    )

    messages = asyncio.run(
        runtime._execute_team(
            team=graph,
            task_text="Write a program.",
            query=TaskQuery(id="implicit-handoff", question="Write a program."),
            gateway=gateway,
            tools_by_role={node.name: {} for node in graph.nodes},
        )
    )

    assert [message.sender for message in messages] == [
        "Investigator",
        "Implementer",
        "Reviewer",
    ]


def test_crewai_handoff_follows_explicit_team_relation() -> None:
    pytest.importorskip("crewai")
    from lychee_mas.runtime.adapters.frameworks.crewai.runtime import CrewAIRuntime

    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-decentralized.json",
        rounds=1,
    )
    context = RoutingContext(
        task="bbeh",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    backend = _ExplicitHandoffBackend()
    runtime = CrewAIRuntime(
        backend=backend,
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        max_turns=3,
        model_id="fake-model",
        memory_method="none",
    )
    runtime._current_tool_requests = []
    runtime._current_tool_executions = []
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    messages = asyncio.run(
        runtime._flow_schedule(
            graph,
            "What is 40 + 2?",
            gateway,
            {node.name: {} for node in graph.nodes},
            "handoff",
        )
    )

    assert [message.sender for message in messages] == ["Analyst", "Solver", "Verifier"]
    assert messages[-1].content == "The final answer is: 42 TERMINATE"


def test_crewai_preserves_empty_model_output_for_shared_result_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("crewai")
    from crewai import Crew
    from lychee_mas.runtime.adapters.frameworks.crewai.runtime import CrewAIRuntime

    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/swe-bench-verified-independent.json",
        rounds=1,
    )
    context = RoutingContext(
        task="swe_bench_verified",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=RunEventWriter(run_events_path(tmp_path)),
    )
    context.set_case("django__django-10097", 0)
    backend = _Backend()
    runtime = CrewAIRuntime(
        backend=backend,
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        model_id="fake-model",
        memory_method="none",
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    async def reject_empty_output(_crew: Crew):
        raise ValueError("Invalid response from LLM call - None or empty.")

    monkeypatch.setattr(Crew, "kickoff_async", reject_empty_output)

    messages = asyncio.run(
        runtime._run_crew(
            graph,
            "Fix the repository.",
            gateway,
            {node.name: {} for node in graph.nodes},
            "single_task_crew",
        )
    )

    assert messages == []
    rejected = list(iter_run_events(tmp_path, event_types=("framework.output_rejected",)))
    assert len(rejected) == 1
    assert rejected[0]["payload"]["framework"] == "crewai"
    assert rejected[0]["payload"]["projection"] == "empty"


class _SelectorRetryBackend(_Backend):
    def __init__(self) -> None:
        self.selector_replies = iter(
            [
                "I choose Solver because it should reason next.",
                "Solver",
                "FINISH",
                "Verifier",
                "FINISH",
            ]
        )
        self.selector_messages: list[list[dict]] = []

    def generate_chat(self, messages, **_kwargs):
        choosing = any(
            "Choose the next participant" in str(message.get("content"))
            or "permitted participant" in str(message.get("content"))
            for message in messages
        )
        if choosing:
            self.selector_messages.append(list(messages))
            text = next(self.selector_replies)
        else:
            text = "The answer is 42."
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=[],
        )


@pytest.mark.parametrize(
    ("dependency", "runtime_module", "runtime_class"),
    [
        (
            "langgraph",
            "lychee_mas.runtime.adapters.frameworks.langgraph.runtime",
            "LangGraphRuntime",
        ),
        ("crewai", "lychee_mas.runtime.adapters.frameworks.crewai.runtime", "CrewAIRuntime"),
    ],
)
def test_model_selector_does_not_offer_finish_before_result(
    dependency: str,
    runtime_module: str,
    runtime_class: str,
    tmp_path: Path,
) -> None:
    pytest.importorskip(dependency)
    module = __import__(runtime_module, fromlist=[runtime_class])
    runtime_type = getattr(module, runtime_class)
    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-centralized.json",
        rounds=1,
    )
    backend = _SelectorRetryBackend()
    context = RoutingContext(
        task="bbeh",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=RunEventWriter(run_events_path(tmp_path)),
    )
    context.set_case("selector-contract", 0)
    runtime = runtime_type(
        backend=backend,
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        max_turns=4,
        model_id="fake-model",
        memory_method="none",
    )

    asyncio.run(
        runtime.run(
            graph,
            TaskQuery(id="selector-contract", question="What is 40 + 2?", gold="42"),
        )
    )

    first_prompt = backend.selector_messages[0][0]["content"]
    assert "FINISH is not permitted until an approved result submitter" in first_prompt
    assert "or FINISH if the task is complete" not in first_prompt
    operations = list(
        iter_run_events(tmp_path, event_types=("coordination.operation.completed",))
    )
    assert operations
    assert {event["payload"]["operation"] for event in operations} == {"select_next"}


def test_langgraph_selector_reprompts_after_invalid_model_decision() -> None:
    pytest.importorskip("langgraph")
    from lychee_mas.runtime.adapters.frameworks.langgraph.runtime import LangGraphRuntime

    graph, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-centralized.json",
        rounds=1,
    )
    backend = _SelectorRetryBackend()
    context = RoutingContext(
        task="bbeh",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    runtime = LangGraphRuntime(
        backend=backend,
        ctx=context,
        max_new_tokens=128,
        max_rounds=1,
        max_turns=4,
        model_id="fake-model",
        memory_method="none",
    )

    trajectory = asyncio.run(
        runtime.run(
            graph,
            TaskQuery(id="selector-retry", question="What is 40 + 2?", gold="42"),
        )
    )

    assert [message.sender for message in trajectory.messages] == ["Solver", "Verifier"]
    assert len(backend.selector_messages) == 4
    assert backend.selector_messages[1][-1]["content"].startswith(
        "That response is not one of the permitted values"
    )
    assert "FINISH is not permitted" in backend.selector_messages[3][-1]["content"]


def test_next_node_decision_accepts_only_exact_or_unambiguous_prefix() -> None:
    candidates = ["Analyst", "Solver", "Verifier"]

    assert parse_next_node_decision(" Solver ", candidates=candidates) == "Solver"
    assert parse_next_node_decision("Solver: 372.00", candidates=candidates) == "Solver"
    assert (
        parse_next_node_decision(
            "\n\nSolver\n\nI will explain why this participant is next.",
            candidates=candidates,
        )
        == "Solver"
    )
    assert parse_next_node_decision("FINISH", candidates=candidates) == "FINISH"
    assert (
        parse_next_node_decision(
            "I choose Solver because it should reason next.", candidates=candidates
        )
        is None
    )


class _ToolBackend(_Backend):
    def __init__(self) -> None:
        self.calls = 0
        self.tool_names: list[str] = []

    def generate_chat(self, messages, **kwargs):
        self.calls += 1
        self.tool_names = [item["function"]["name"] for item in (kwargs.get("tools") or [])]
        if self.calls == 1:
            text = ""
            tool_calls = [
                {
                    "id": "call-double",
                    "name": "double",
                    "arguments": '{"value": 21}',
                }
            ]
        else:
            text = "The tool returned 42."
            tool_calls = []
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=tool_calls,
        )


def test_model_gateway_records_tool_loop_and_usage() -> None:
    backend = _ToolBackend()
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="langgraph",
    )

    reply = asyncio.run(
        gateway.invoke(
            "Solver",
            [{"role": "user", "content": "Double 21."}],
            tools={"double": lambda value: value * 2},
            max_tool_iterations=2,
        )
    )

    assert reply.content == "The tool returned 42."
    assert reply.prompt_tokens == 40
    assert reply.completion_tokens == 16
    assert reply.tool_requests[0]["tool_name"] == "double"
    assert reply.tool_executions[0]["output"] == "42"
    assert len(context.decisions) == 2
    assert backend.tool_names == ["double"]


def test_model_gateway_keeps_event_scope_when_shared_context_changes(tmp_path: Path) -> None:
    class BlockingBackend(_Backend):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def generate_chat(self, messages, **kwargs):
            self.started.set()
            assert self.release.wait(timeout=2)
            return super().generate_chat(messages, **kwargs)

    backend = BlockingBackend()
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
        event_writer=RunEventWriter(run_events_path(tmp_path)),
    )
    context.set_case("case-a", 1)
    context.current_trial_index = 0
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    async def invoke_while_context_changes() -> None:
        task = asyncio.create_task(
            gateway.invoke("Solver", [{"role": "user", "content": "2+2"}])
        )
        assert await asyncio.to_thread(backend.started.wait, 2)
        context.set_case("case-b", 2)
        backend.release.set()
        await task

    asyncio.run(invoke_while_context_changes())

    events = list(
        iter_run_events(
            tmp_path,
            event_types=("model_call.started", "model_call.completed"),
        )
    )
    assert [event["case_id"] for event in events] == ["case-a", "case-a"]
    assert [event["dataset_index"] for event in events] == [1, 1]
    assert events[0]["operation_id"] == events[1]["operation_id"]


def test_model_gateway_rejects_missing_tool_arguments_before_invocation() -> None:
    class MissingArgumentBackend(_Backend):
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, messages, **_kwargs):
            self.calls += 1
            tool_calls = (
                [{"id": "call-python", "name": "python_code", "arguments": "{}"}]
                if self.calls == 1
                else []
            )
            return SimpleNamespace(
                text="" if tool_calls else "I could not execute malformed code.",
                reasoning_content="",
                n_prompt_pos=20,
                n_gen_tokens=8,
                latency_s=0.001,
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={},
                tool_calls=tool_calls,
            )

    invoked = False

    def python_code(code: str) -> str:
        nonlocal invoked
        invoked = True
        return code

    context = RoutingContext(
        task="livecodebench",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=MissingArgumentBackend(),
        max_new_tokens=128,
        framework="langgraph",
    )

    reply = asyncio.run(
        gateway.invoke(
            "Coder",
            [{"role": "user", "content": "Write code."}],
            tools={"python_code": python_code},
            max_tool_iterations=1,
        )
    )

    assert invoked is False
    assert reply.tool_executions[0]["failure_kind"] == "signature_mismatch"
    assert reply.tool_executions[0]["invalid_tool_arguments"] is True
    assert "missing a required argument: 'code'" in reply.tool_executions[0][
        "error_message"
    ]


def test_model_gateway_records_all_provider_timing_fields() -> None:
    class TimingBackend(_Backend):
        def generate_chat(self, messages, **_kwargs):
            return SimpleNamespace(
                text="done",
                reasoning_content="",
                n_prompt_pos=20,
                n_gen_tokens=8,
                latency_s=0.5,
                client_rate_limiter_wait_s=0.1,
                client_http_request_latency_s=0.4,
                client_response_postprocess_latency_s=0.05,
                client_model_call_wall_time_s=0.55,
                provider_request_queue_latency_s=0.02,
                provider_scheduled_to_first_token_s=0.03,
                provider_generation_latency_s=0.35,
                provider_mean_inter_token_latency_s=0.01,
                provider_output_tokens_per_second=40.0,
                provider_request_metrics={"queue_time_ms": 20.0},
                finish_reason="stop",
                provider_request_payload={"messages": messages},
                provider_response_payload={"content": "done"},
                tool_calls=[],
            )

    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=TimingBackend(),
        max_new_tokens=128,
        framework="langgraph",
    )

    asyncio.run(gateway.invoke("Solver", [{"role": "user", "content": "Solve it."}]))

    decision = context.decisions[-1]
    assert decision["client_rate_limiter_wait_s"] == 0.1
    assert decision["client_http_request_latency_s"] == 0.4
    assert decision["client_response_postprocess_latency_s"] == 0.05
    assert decision["client_model_call_wall_time_s"] == 0.55
    assert decision["provider_mean_inter_token_latency_s"] == 0.01
    assert decision["provider_request_metrics_available"] is True
    assert decision["provider_request_metrics"] == {"queue_time_ms": 20.0}


def test_model_gateway_resolves_coordination_calls_by_node_binding() -> None:
    role_backend = SimpleNamespace(model_name="role-model")
    legacy_backend = SimpleNamespace(model_name="legacy-control-model")
    context = RoutingContext(
        task="gaia_validation",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    context.deployment_of_role["Orchestrator"] = "orchestrator-deployment"
    gateway = ModelGateway(
        ctx=context,
        backend=legacy_backend,
        backend_resolver=lambda deployment_id: (
            role_backend if deployment_id == "orchestrator-deployment" else legacy_backend
        ),
        model_resolver=lambda deployment_id: f"model:{deployment_id}",
        max_new_tokens=128,
        max_new_tokens_resolver=lambda role, fallback: 512 if role == "Orchestrator" else fallback,
        invocation_overrides_resolver=lambda role: {"temperature": 0.2, "role": role},
        control_deployment_id="legacy-control-deployment",
        control_max_new_tokens=64,
        control_invocation_overrides={"temperature": 0.0},
        framework="langgraph",
    )

    backend, deployment_id, model_id, maximum, overrides = gateway._binding(
        "Orchestrator",
        controller=True,
    )

    assert backend is role_backend
    assert deployment_id == "orchestrator-deployment"
    assert model_id == "model:orchestrator-deployment"
    assert maximum == 512
    assert overrides == {"temperature": 0.2, "role": "Orchestrator"}


class _LargeToolOutputBackend(_Backend):
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict]] = []

    def generate_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages.append(list(messages))
        tool_calls = (
            [{"id": "call-large", "name": "large", "arguments": "{}"}] if self.calls == 1 else []
        )
        return SimpleNamespace(
            text="" if tool_calls else "done",
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={},
            tool_calls=tool_calls,
        )


def test_model_gateway_preserves_full_tool_output_but_bounds_model_copy() -> None:
    backend = _LargeToolOutputBackend()
    context = RoutingContext(
        task="swe_bench_verified",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        max_inline_tool_result_chars=1000,
        framework="langgraph",
    )
    full_output = "A" * 2500

    reply = asyncio.run(
        gateway.invoke(
            "Developer",
            [{"role": "user", "content": "Inspect it."}],
            tools={"large": lambda: full_output},
            max_tool_iterations=2,
        )
    )

    execution = reply.tool_executions[0]
    assert execution["output"] == full_output
    assert execution["output_chars"] == 2500
    assert execution["output_truncated_for_model_context"] is True
    assert len(execution["delivered_output"]) <= 1000
    delivered = backend.messages[1][-1]["content"]
    assert delivered == execution["delivered_output"]
    assert "full output is preserved in the EventLog" in delivered


class _ExhaustedToolBackend(_Backend):
    def __init__(self) -> None:
        self.calls = 0
        self.tool_counts: list[int] = []
        self.final_messages: list[dict] = []

    def generate_chat(self, messages, **kwargs):
        self.calls += 1
        self.tool_counts.append(len(kwargs.get("tools") or []))
        if kwargs.get("tools"):
            text = ""
            tool_calls = [
                {
                    "id": f"call-{self.calls}",
                    "name": "double",
                    "arguments": '{"value": 21}',
                }
            ]
        else:
            self.final_messages = list(messages)
            text = "Best final response: 42."
            tool_calls = []
        return SimpleNamespace(
            text=text,
            reasoning_content="",
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={"content": text},
            tool_calls=tool_calls,
        )


def test_model_gateway_forces_textual_finalization_after_tool_limit() -> None:
    backend = _ExhaustedToolBackend()
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    reply = asyncio.run(
        gateway.invoke(
            "Developer",
            [{"role": "user", "content": "Solve the task."}],
            tools={"double": lambda value: value * 2},
            max_tool_iterations=2,
        )
    )

    assert reply.content == "Best final response: 42."
    assert backend.tool_counts == [1, 1, 0]
    assert len(reply.tool_requests) == 2
    assert reply.prompt_tokens == 60
    assert reply.completion_tokens == 24
    assert backend.final_messages[-1]["role"] == "user"


class _ReasoningOnlyThenFinalBackend(_Backend):
    def __init__(self, *, final_call: int = 2) -> None:
        self.calls = 0
        self.final_call = final_call
        self.messages: list[list[dict]] = []

    def generate_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages.append(list(messages))
        final = "Final deliverable." if self.calls >= self.final_call else ""
        reasoning = "I have finished the analysis." if not final else ""
        return SimpleNamespace(
            text=final,
            reasoning_content=reasoning,
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.001,
            finish_reason="stop",
            provider_request_payload={"messages": messages},
            provider_response_payload={
                "content": final,
                "reasoning_content": reasoning,
            },
            tool_calls=[],
        )


def test_model_gateway_recovers_reasoning_only_response_once() -> None:
    backend = _ReasoningOnlyThenFinalBackend()
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    reply = asyncio.run(gateway.invoke("Solver", [{"role": "user", "content": "Solve it."}]))

    assert reply.content == "Final deliverable."
    assert backend.calls == 2
    assert backend.messages[-1][-1]["content"].startswith(
        "Your previous response contained reasoning"
    )
    assert reply.prompt_tokens == 40
    assert reply.completion_tokens == 16


def test_model_gateway_retries_multiple_reasoning_only_responses() -> None:
    backend = _ReasoningOnlyThenFinalBackend(final_call=4)
    context = RoutingContext(
        task="gsm8k",
        router=fixed_channel_router("none"),
        memory=_Memory(),
    )
    gateway = ModelGateway(
        ctx=context,
        backend=backend,
        max_new_tokens=128,
        framework="crewai",
    )

    reply = asyncio.run(gateway.invoke("Solver", [{"role": "user", "content": "Solve it."}]))

    assert reply.content == "Final deliverable."
    assert backend.calls == 4
    assert reply.prompt_tokens == 80
    assert reply.completion_tokens == 32


def test_magentic_one_protocol_diagnostic_identifies_invalid_next_speaker():
    from types import SimpleNamespace

    from lychee_mas.runtime.adapters.frameworks.autogen.runtime import (
        _magentic_one_protocol_diagnostic,
    )

    ctx = SimpleNamespace(
        last_controller_exchange={
            "model_call_index": 7,
            "output": json.dumps(
                {
                    "is_request_satisfied": {"answer": False, "reason": "unfinished"},
                    "is_progress_being_made": {"answer": True, "reason": "progress"},
                    "is_in_loop": {"answer": False, "reason": "not looping"},
                    "instruction_or_question": {"answer": "continue", "reason": "work"},
                    "next_speaker": {"answer": "User", "reason": "ask user"},
                }
            ),
        }
    )

    diagnostic = _magentic_one_protocol_diagnostic(ctx, ["Coder", "WebSurfer"])

    assert diagnostic is not None
    assert diagnostic["violation"] == "invalid_next_speaker"
    assert diagnostic["next_speaker"] == "User"
    assert diagnostic["model_call_index"] == 7


def test_observed_file_surfer_preserves_official_output_and_reports_request():
    import asyncio

    from lychee_mas.runtime.adapters.frameworks.autogen.file_surfer import (
        observed_file_surfer_class,
    )

    request_ids: set[str] = set()
    observed: list[tuple[str, object]] = []

    class FakeFileSurfer:
        def __init__(self, *args, **kwargs):
            pass

        async def _generate_reply(self, *args, **kwargs):
            request_ids.add("call-1")
            return False, "official file output"

    observed_class = observed_file_surfer_class(base_class=FakeFileSurfer)
    agent = observed_class(
        request_ids=lambda: set(request_ids),
        on_tool_completed=lambda call_id, output: observed.append((call_id, output)),
    )

    output = asyncio.run(agent._generate_reply())

    assert output == (False, "official file output")
    assert observed == [("call-1", output)]
