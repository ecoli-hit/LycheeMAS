"""MAS framework adapters implementing the common Runtime contract."""

from __future__ import annotations

from typing import Any

__all__ = ["AutoGenRuntime", "CrewAIRuntime", "LangGraphRuntime", "MockRuntime"]


def __getattr__(name: str) -> Any:
    if name == "AutoGenRuntime":
        from .autogen.runtime import AutoGenRuntime

        return AutoGenRuntime
    if name == "CrewAIRuntime":
        from .crewai.runtime import CrewAIRuntime

        return CrewAIRuntime
    if name == "LangGraphRuntime":
        from .langgraph.runtime import LangGraphRuntime

        return LangGraphRuntime
    if name == "MockRuntime":
        from .mock import MockRuntime

        return MockRuntime
    raise AttributeError(name)
