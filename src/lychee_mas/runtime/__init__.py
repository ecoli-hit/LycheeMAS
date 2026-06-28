"""runtime：运行时抽象（隔离 AutoGen，预留 MAF 迁移，CLAUDE.md §8）。

- base.py     Runtime 协议（run/intercept）+ MASGraph/MASTeam 轻量容器
- backends/   mock（离线默认）/ autogen（生产）+ HF/vLLM model_client

import 本包会触发所有后端注册（不触发 torch/autogen，惰性导入约定）。
"""
from __future__ import annotations

from . import backends  # noqa: F401  触发 runtime/model_client 注册
from .base import BaseRuntime, MASGraph, MASTeam, Runtime

__all__ = ["Runtime", "BaseRuntime", "MASGraph", "MASTeam"]
