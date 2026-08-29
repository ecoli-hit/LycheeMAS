"""Framework-neutral projection of native framework messages into one result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lychee_mas.core.types import TaskQuery
from lychee_mas.runtime.contracts.runtime import MASGraph

from .contract import (
    ResultContractValidation,
    result_messages,
    result_source,
    validate_result_contract,
)


@dataclass(frozen=True)
class ProjectedResult:
    """One benchmark-owned result plus the Team result-contract evidence."""

    content: str
    source: str | None
    eligible_messages: tuple[Any, ...]
    validation: ResultContractValidation


def project_result(
    *,
    benchmark: Any,
    task: str,
    team: MASGraph,
    query: TaskQuery,
    messages: list[Any],
    workspace: str | Path | None,
    default_text: str | None = None,
    tool_requests: list[Any] | None = None,
) -> ProjectedResult:
    """Apply one result selection, collection, and validation path to every Runtime."""

    eligible = result_messages(team, messages)
    fallback = (
        default_text
        if default_text is not None
        else benchmark.extract_messages(task, eligible)
    )
    content = benchmark.collect_prediction(
        query=query,
        messages=messages,
        workspace=workspace,
        default_text=fallback,
    )
    validation = validate_result_contract(
        team,
        messages,
        result_kind=benchmark.result_kind,
        final_content=content,
        workspace=workspace,
        tool_requests=tool_requests,
    )
    return ProjectedResult(
        content=content,
        source=result_source(team, messages),
        eligible_messages=tuple(eligible),
        validation=validation,
    )


__all__ = ["ProjectedResult", "project_result"]
