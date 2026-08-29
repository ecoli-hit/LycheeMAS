"""Runtime helpers for applying the framework-neutral Team result contract."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lychee_mas.runtime.contracts.runtime import MASGraph


def _message_source(message: Any) -> str:
    return str(
        getattr(message, "sender", "")
        or getattr(message, "source", "")
        or getattr(message, "name", "")
    )


def result_submitter_aliases(team: MASGraph) -> set[str]:
    """Resolve TeamSpec submitter Node IDs to framework-visible source names."""

    submitters = {
        str(item)
        for item in (team.meta.get("result_contract") or {}).get("submitters") or []
    }
    if not submitters:
        return set()
    aliases = set(submitters)
    for node in team.nodes:
        node_id = str(node.meta.get("node_id") or node.id)
        if node_id in submitters:
            aliases.update({str(node.id), str(node.name), str(node.role)})
    operation_nodes = team.meta.get("operation_nodes") or {}
    for operation, node in operation_nodes.items():
        if not node or str(node.get("id") or "") not in submitters:
            continue
        aliases.add(str(node["id"]))
        if operation == "aggregate" and team.meta.get("control_node_id") == node.get("id"):
            # AutoGen's MagenticOneGroupChat publishes the bound Orchestrator
            # under this framework-owned source name.
            aliases.add("MagenticOneOrchestrator")
    return {item.casefold() for item in aliases if item}


def result_messages(team: MASGraph, messages: Iterable[Any]) -> list[Any]:
    """Return only messages eligible to supply the Team's text result.

    An empty ``result_contract.submitters`` list means that every participant message
    is eligible.  A configured submitter contract is strict: if no matching
    message was published, the result candidates are empty rather than silently
    substituting another role's output.
    """

    rows = list(messages)
    aliases = result_submitter_aliases(team)
    if not aliases:
        return rows
    return [row for row in rows if _message_source(row).casefold() in aliases]


def result_source(team: MASGraph, messages: Iterable[Any]) -> str | None:
    eligible = result_messages(team, messages)
    return _message_source(eligible[-1]) if eligible else None


def _text_payload_empty(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if not (text.startswith('"') and text.endswith('"')):
        return False
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(decoded, str) and not decoded.strip()


@dataclass(frozen=True)
class ResultContractValidation:
    """One benchmark-aware validation of a Team's terminal result projection."""

    kind: str
    valid: bool
    reason: str
    source: str | None
    submitters: tuple[str, ...]
    eligible_message_count: int
    payload_empty: bool
    collector: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_result_contract(
    team: MASGraph,
    messages: Iterable[Any],
    *,
    result_kind: str,
    final_content: str,
    workspace: str | Path | None = None,
    tool_requests: Iterable[Any] | None = None,
) -> ResultContractValidation:
    """Validate projection availability without confusing it with correctness.

    Text results require a non-empty message from an approved submitter. Action
    and patch benchmarks are collected from tool events or the workspace, so an
    empty textual chat result is not itself a contract failure. Their official
    scorers remain responsible for deciding whether an empty action list or patch
    is correct.
    """

    rows = list(messages)
    eligible = result_messages(team, rows)
    source = result_source(team, rows)
    submitters = tuple(
        str(item)
        for item in (team.meta.get("result_contract") or {}).get("submitters") or []
    )
    kind = str(result_kind).strip().lower()
    payload_empty = _text_payload_empty(final_content)

    if kind == "text":
        if not eligible:
            valid = False
            reason = "no message was published by an approved result submitter"
        elif payload_empty:
            valid = False
            reason = "the approved submitter projection is empty"
        else:
            valid = True
            reason = "non-empty text was projected from an approved submitter"
        collector = "approved_submitter_message"
    elif kind == "patch":
        valid = workspace is not None
        reason = (
            "workspace patch collector completed; official harness judges patch quality"
            if valid
            else "patch result requires a materialized workspace"
        )
        collector = "workspace_patch"
    elif kind == "action":
        valid = tool_requests is not None
        reason = (
            "tool-action collector completed; official state scorer judges action quality"
            if valid
            else "action result requires an observed tool-request collection"
        )
        collector = "tool_action_trace"
    else:
        raise ValueError(f"unsupported result contract kind {kind!r}")

    return ResultContractValidation(
        kind=kind,
        valid=valid,
        reason=reason,
        source=source,
        submitters=submitters,
        eligible_message_count=len(eligible),
        payload_empty=payload_empty,
        collector=collector,
    )


__all__ = [
    "ResultContractValidation",
    "result_messages",
    "result_source",
    "result_submitter_aliases",
    "validate_result_contract",
]
