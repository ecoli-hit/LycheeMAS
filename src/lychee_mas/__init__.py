"""LycheeMAS —— 运行时可插拔的多智能体系统研究框架（架构设计见 docs/DESIGN.md）。

把整个 MAS 统一表示为带时序/记忆状态的有向图 G=(V,E,W,T,M)；每层是对 G（或其执行轨迹 τ）的一次变换：
  Construct / Prune / Memory / Processing / Attribute-and-Train，
外加插件系统（pre_run / post_run / optimizer 三接缝）承载运行前/运行后/离线优化。

设计四原则：可插拔可消融（registry + config）、Runtime 抽象隔离执行引擎（autogen / langgraph
双后端）、性能-成本联合度量、可复现。

import 本包会触发所有可插拔组件的注册（runtime/memory/plugins/aggregator/...），但**不触发**
torch / transformers / autogen / langgraph（重依赖全部惰性导入）。因此 `import lychee_mas` 与
`REGISTRY.snapshot()` 在纯离线、无重依赖的环境也能成功。
"""
from __future__ import annotations

__version__ = "0.1.0"

# 依次 import 各子包以触发组件注册（side-effect import；放进 __all__ 以避免被判为未使用）。
# 顶层研究子包（与 layers 同级）：memory(CDM)、trace(归因/信用+TraceStore)、train(RL/提示优化)。
# attribute_train 已拆成 trace+train；stores 并入 memory(MemoryStore)/trace(TraceStore) 后删除。
from . import eval, layers, memory, plugins, runtime, trace, train
from .core.registry import REGISTRY  # 暴露注册表
from .pipeline import Orchestrator

__all__ = ["__version__", "REGISTRY", "Orchestrator", "layers", "memory", "plugins",
           "trace", "train", "runtime", "eval"]
