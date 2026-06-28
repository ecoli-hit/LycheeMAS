"""Runtime 后端：mock（离线默认）+ autogen（生产）+ HF/vLLM model_client。

唯一允许 `import autogen_*` 的位置就是本目录下的 `autogen_*.py`（CLAUDE.md 黄金法则 1）。
所有后端的 autogen/torch 导入都已惰性化到方法内部，因此 import 本包（触发注册）不需要这些重依赖。

注册触发：
  runtime/mock, runtime/autogen
  model_client/injection, model_client/vllm
"""
from __future__ import annotations

from . import (
    autogen_injection_client,  # noqa: F401  -> model_client/injection
    autogen_runtime,  # noqa: F401  -> runtime/autogen
    mock_runtime,  # noqa: F401  -> runtime/mock
    vllm_client,  # noqa: F401  -> model_client/vllm
)
from .mock_runtime import MockRuntime

__all__ = ["MockRuntime"]
