"""网络剪枝与优化（Prune）。

- GraphPruner   graph_pruner/agentprune（已实现，AgentPrune 时空掩码）；
                agentdropout / agentdropout_v2（桩）
- VocabAdapter  vocab_adapter/agentvocab（桩）

import 本包触发上述组件注册（不触发 torch/autogen）。
"""
from __future__ import annotations

from .base import GraphPruner, VocabAdapter
from .pruners import AgentDropout, AgentDropoutV2, AgentPrunePruner
from .vocab import AgentVocab

__all__ = [
    "GraphPruner",
    "VocabAdapter",
    "AgentDropout",
    "AgentDropoutV2",
    "AgentPrunePruner",
    "AgentVocab",
]
