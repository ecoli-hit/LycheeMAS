from __future__ import annotations

from types import SimpleNamespace

import pytest
from lychee_mas.core.types import TaskQuery
from lychee_mas.eval.teams.compiler import load_team_spec
from lychee_mas.runtime.conformance import evaluate_runtime_conformance
from lychee_mas.runtime.coordination.compiler import message_sources, runtime_support
from lychee_mas.runtime.results.projection import project_result


class _Benchmark:
    def __init__(self, result_kind: str) -> None:
        self.result_kind = result_kind

    @staticmethod
    def extract_messages(_task, messages) -> str:
        return str(messages[-1].content) if messages else ""

    def collect_prediction(self, *, messages, workspace, default_text, **_kwargs) -> str:
        if self.result_kind == "patch":
            return "diff --git a/a.py b/a.py" if workspace else ""
        if self.result_kind == "action":
            return "action-trace"
        return default_text


@pytest.mark.parametrize(
    ("result_kind", "messages", "workspace", "tool_requests", "expected"),
    [
        (
            "text",
            [SimpleNamespace(source="Verifier", content="The final answer is: 42")],
            None,
            None,
            "The final answer is: 42",
        ),
        ("action", [], None, [{"id": "action-1", "tool_name": "click"}], "action-trace"),
        ("patch", [], "/tmp/workspace", None, "diff --git a/a.py b/a.py"),
    ],
)
@pytest.mark.parametrize("framework", ["autogen", "langgraph", "crewai"])
def test_result_contract_projection_is_end_to_end_for_all_result_kinds(
    framework,
    result_kind,
    messages,
    workspace,
    tool_requests,
    expected,
) -> None:
    spec_name = "bbeh-sequential" if result_kind == "text" else "workbench-independent"
    team, _document = load_team_spec(
        f"configs/eval_studio/teams/specs/{spec_name}.json",
        rounds=1,
    )

    projected = project_result(
        benchmark=_Benchmark(result_kind),
        task="contract-test",
        team=team,
        query=TaskQuery(id="case-1", question="task", gold=None),
        messages=messages,
        workspace=workspace,
        tool_requests=tool_requests,
    )

    assert projected.content == expected
    assert projected.validation.kind == result_kind
    assert projected.validation.valid is True
    report = evaluate_runtime_conformance(
        framework=framework,
        team=team,
        result_validation=projected.validation,
        tool_requests=tool_requests,
        tool_executions=tool_requests,
    )
    assert report.status == "conformant"


@pytest.mark.parametrize("framework", ["autogen", "langgraph", "crewai"])
def test_conformance_report_uses_the_same_invariants_for_every_framework(
    framework: str,
) -> None:
    team, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    projected = project_result(
        benchmark=_Benchmark("text"),
        task="bbeh",
        team=team,
        query=TaskQuery(id="case-1", question="task", gold=None),
        messages=[SimpleNamespace(source="Verifier", content="42")],
        workspace=None,
    )

    report = evaluate_runtime_conformance(
        framework=framework,
        team=team,
        result_validation=projected.validation,
        tool_requests=[{"tool_call_id": "call-1"}],
        tool_executions=[{"tool_call_id": "call-1", "is_error": True}],
    )

    assert report.status == "conformant"
    assert report.controlled_comparison_eligible is True
    assert {check.id for check in report.checks} == {
        "framework_binding_supported",
        "portable_semantics_preserved",
        "result_contract_valid",
        "tool_trace_complete",
    }


def test_missing_tool_execution_is_a_conformance_failure() -> None:
    team, _document = load_team_spec(
        "configs/eval_studio/teams/specs/bbeh-sequential.json",
        rounds=1,
    )
    projected = project_result(
        benchmark=_Benchmark("text"),
        task="bbeh",
        team=team,
        query=TaskQuery(id="case-1", question="task", gold=None),
        messages=[SimpleNamespace(source="Verifier", content="42")],
        workspace=None,
    )

    report = evaluate_runtime_conformance(
        framework="autogen",
        team=team,
        result_validation=projected.validation,
        tool_requests=[{"tool_call_id": "missing"}],
        tool_executions=[],
    )

    assert report.status == "failed"
    assert report.controlled_comparison_eligible is False
    tool_check = next(
        check for check in report.checks if check.id == "tool_trace_complete"
    )
    assert tool_check.passed is False


@pytest.mark.parametrize("framework", ["autogen", "langgraph", "crewai"])
def test_control_message_and_data_contracts_share_one_portable_source_of_truth(
    framework: str,
) -> None:
    team, document = load_team_spec(
        "configs/eval_studio/teams/specs/gaia-centralized.json",
        rounds=1,
    )
    coordination = team.meta["coordination_ir"]
    members = coordination["members"]
    expected_messages = message_sources(document, members)

    assert {
        relation["id"] for relation in coordination["control_relations"]
    } == {
        relation["id"]
        for relation in document["relations"]
        if relation.get("control")
    }
    names_by_id = {str(node.meta["node_id"]): node.name for node in team.nodes}
    for node in team.nodes:
        node_id = str(node.meta["node_id"])
        assert set(node.meta["receives_from"]) == {
            names_by_id[source]
            for source in expected_messages[node_id]
            if source in names_by_id
        }
        expected_data_ids = {
            relation["id"]
            for relation in document["relations"]
            if relation.get("data")
            and node_id in {relation["from"], relation["to"]}
        }
        assert {relation["id"] for relation in node.meta["resource_access"]} == expected_data_ids

    support = runtime_support(framework, coordination, document)
    assert support["supported"] is True
    assert support["mapping_level"] in {"exact", "composed", "approximated"}
