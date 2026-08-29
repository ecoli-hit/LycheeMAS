"""Framework-neutral input and output contracts for one Node activation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from lychee_mas.core.types import Message


@dataclass(frozen=True)
class NodeInput:
    """Materialized values presented to one ExecutableNode activation.

    TeamSpec v14 DataTransfers and SharedState access rules are resolved into
    this value before a framework adapter invokes the concrete Node.
    """

    node_id: str
    invocation_id: str
    task: str
    control_signal: str = "activate"
    messages: tuple[Message, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    data_by_port: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeOutput:
    """Materialized values emitted by one ExecutableNode activation."""

    node_id: str
    invocation_id: str
    status: Literal["completed", "failed"]
    message: Message | None = None
    control_signals: tuple[str, ...] = ()
    state_updates: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    result: Any = None
    data_by_port: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["NodeInput", "NodeOutput"]
