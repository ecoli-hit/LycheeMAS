"""Framework-neutral runtime contracts and concrete adapters.

Importing this package registers lightweight framework and inference adapters;
heavy third-party dependencies remain lazily imported by each adapter.
"""
from __future__ import annotations

from . import adapters  # noqa: F401  触发 runtime/model_client 注册
from .contracts.runtime import BaseRuntime, MASGraph, MASTeam, Runtime

__all__ = ["Runtime", "BaseRuntime", "MASGraph", "MASTeam"]
