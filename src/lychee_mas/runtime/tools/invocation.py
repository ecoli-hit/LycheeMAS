"""Validation boundary for model-generated tool invocations."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable


class ToolInvocationError(ValueError):
    """A model produced a tool request that cannot be invoked as declared."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def bind_tool_arguments(
    *,
    tool_name: str,
    arguments_text: str,
    function: Callable | None,
) -> dict[str, Any]:
    """Parse and bind one tool call before user code is entered."""

    try:
        arguments = json.loads(arguments_text or "{}")
    except json.JSONDecodeError as exc:
        raise ToolInvocationError(
            f"invalid arguments for tool {tool_name!r}: arguments are not valid JSON",
            reason="invalid_json",
        ) from exc
    if not isinstance(arguments, dict):
        raise ToolInvocationError(
            f"invalid arguments for tool {tool_name!r}: expected a JSON object",
            reason="non_object",
        )
    if function is None:
        raise ToolInvocationError(
            f"unknown tool {tool_name!r}",
            reason="unknown_tool",
        )
    try:
        inspect.signature(function).bind(**arguments)
    except TypeError as exc:
        raise ToolInvocationError(
            f"invalid arguments for tool {tool_name!r}: {exc}",
            reason="signature_mismatch",
        ) from exc
    return arguments


__all__ = ["ToolInvocationError", "bind_tool_arguments"]
