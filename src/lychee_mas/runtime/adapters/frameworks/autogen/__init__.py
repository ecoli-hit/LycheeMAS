"""AutoGen adapter package with lazy runtime imports."""

from __future__ import annotations

from typing import Any

__all__ = ["AutoGenRuntime"]


def __getattr__(name: str) -> Any:
    if name == "AutoGenRuntime":
        from .runtime import AutoGenRuntime

        return AutoGenRuntime
    raise AttributeError(name)
