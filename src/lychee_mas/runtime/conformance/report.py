"""Post-run conformance report for one framework adapter execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lychee_mas.runtime.adapters.frameworks.bindings import build_framework_binding_report
from lychee_mas.runtime.contracts.runtime import MASGraph
from lychee_mas.runtime.results.contract import ResultContractValidation


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    id: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RuntimeConformanceReport:
    """Observed Trial invariants, kept separate from benchmark correctness."""

    framework: str
    status: str
    mapping_level: str
    controlled_comparison_eligible: bool
    observation_coverage: dict[str, str]
    checks: tuple[ConformanceCheck, ...]
    semantic_deltas: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_record_id(value: dict[str, Any]) -> str:
    return str(
        value.get("tool_call_id")
        or value.get("call_id")
        or value.get("id")
        or ""
    )


def evaluate_runtime_conformance(
    *,
    framework: str,
    team: MASGraph,
    result_validation: ResultContractValidation,
    tool_requests: list[dict[str, Any]] | None = None,
    tool_executions: list[dict[str, Any]] | None = None,
    observation_coverage: dict[str, str] | None = None,
) -> RuntimeConformanceReport:
    """Evaluate portable structural invariants after one adapter run.

    The report does not compare stochastic model text and does not score the
    benchmark answer. It checks only the Team/Runtime boundary promised by every
    adapter.
    """

    support = build_framework_binding_report(
        framework,
        dict(team.meta.get("coordination_ir") or {}),
        dict(team.meta.get("team_spec") or {}),
    )
    requests = list(tool_requests or [])
    executions = list(tool_executions or [])
    request_ids = {_tool_record_id(item) for item in requests if _tool_record_id(item)}
    execution_ids = {
        _tool_record_id(item) for item in executions if _tool_record_id(item)
    }
    if request_ids:
        missing_executions = sorted(request_ids - execution_ids)
        tool_trace_complete = not missing_executions
        tool_detail = (
            "every identified tool request has a terminal execution record"
            if tool_trace_complete
            else "missing terminal execution records for: " + ", ".join(missing_executions)
        )
    else:
        tool_trace_complete = len(executions) >= len(requests)
        tool_detail = (
            "tool trace cardinality is complete"
            if tool_trace_complete
            else f"observed {len(requests)} requests but {len(executions)} executions"
        )

    checks = (
        ConformanceCheck(
            id="framework_binding_supported",
            passed=bool(support.get("supported")),
            detail=str(support.get("implementation") or support.get("reason") or ""),
        ),
        ConformanceCheck(
            id="portable_semantics_preserved",
            passed=bool(support.get("controlled_comparison_eligible")),
            detail=(
                f"mapping_level={support.get('mapping_level')}"
                if support.get("controlled_comparison_eligible")
                else "; ".join(support.get("semantic_deltas") or [])
            ),
        ),
        ConformanceCheck(
            id="result_contract_valid",
            passed=bool(result_validation.valid),
            detail=result_validation.reason,
        ),
        ConformanceCheck(
            id="tool_trace_complete",
            passed=tool_trace_complete,
            detail=tool_detail,
        ),
    )
    required_checks = [
        check for check in checks if check.id != "portable_semantics_preserved"
    ]
    status = "conformant" if all(check.passed for check in required_checks) else "failed"
    comparison_eligible = status == "conformant" and bool(
        support.get("controlled_comparison_eligible")
    )
    return RuntimeConformanceReport(
        framework=framework,
        status=status,
        mapping_level=str(support.get("mapping_level") or "unsupported"),
        controlled_comparison_eligible=comparison_eligible,
        observation_coverage=dict(observation_coverage or {}),
        checks=checks,
        semantic_deltas=tuple(map(str, support.get("semantic_deltas") or [])),
    )


__all__ = [
    "ConformanceCheck",
    "RuntimeConformanceReport",
    "evaluate_runtime_conformance",
]
