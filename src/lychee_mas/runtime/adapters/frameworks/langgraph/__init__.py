"""LangGraph adapter package with lazy runtime imports."""

from __future__ import annotations

from typing import Any

__all__ = ["LangGraphRuntime"]


def __getattr__(name: str) -> Any:
    if name == "LangGraphRuntime":
        from .runtime import LangGraphRuntime

        return LangGraphRuntime
    raise AttributeError(name)
