"""CrewAI adapter package with lazy runtime imports."""

from __future__ import annotations

from typing import Any

__all__ = ["CrewAIRuntime"]


def __getattr__(name: str) -> Any:
    if name == "CrewAIRuntime":
        from .runtime import CrewAIRuntime

        return CrewAIRuntime
    raise AttributeError(name)
