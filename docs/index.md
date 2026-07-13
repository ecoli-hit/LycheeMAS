# LycheeMAS

**基于 [AutoGen](https://github.com/microsoft/autogen) 的多智能体系统（MAS）研究框架。** 核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），每个阶段都是对 G（或其执行轨迹 τ）的一次变换。这样所有阶段共享同一套类型与运行时，组件**可插拔、可消融**。

!!! tip "本站是什么"
    这是 LycheeMAS 的开发文档站：叙述性**指南**（安装 / 快速上手 / 架构 / 核心概念 / 组件开发）+ 从代码 docstring 自动生成的 **[API Reference](reference/lychee_mas/)**（每个模块一页，始终与源码同步）。

# 整体架构

| 阶段             | 模块                             | 职责                                           | 状态                                           |
| ---------------- | -------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 构建 Construct   | `lychee_mas.layers.construct`  | 团队组建 + 静态/动态拓扑                       | static 已实现<br />AgentInit 已实现（待merge） |
| 剪枝 Prune       | `lychee_mas.layers.prune`      | 网络剪枝 + 模型级词表降本                      | AgentDropout<br />MASPO<br />AgentVocab        |
| 处理 Processing  | `lychee_mas.layers.processing` | 决定跑几次 MAS + 如何归约（serial / parallel） | Serial<br />Parallel （开发中）                |
| 记忆 Memory      | `lychee_mas.memory`            | 运行时多表征记忆管理（**CDM 主线**）     | **双通道记忆轨迹 已实现（待优化）**     |
| 归因 Trace       | `lychee_mas.trace`            | 错误归因/信用（trace）   | 开发中                                         |
| 训练 Train       | `lychee_mas.train`             | MASPO，AgentInit，Agentic RL                   | 开发中                                         |
| 核心组件 Core   | `lychee_mas.core`              | 公共类型 + 组件注册表 REGISTRY                 | 完成（迭代中）                                 |
| 运行组件 Runtime | `lychee_mas.runtime`           | 隔离 AutoGen 的运行时抽象 + 后端               | 完成（迭代中）                                 |
| 测试 eval        | `lychee_mas.eval`              | 核心测试组件                                   | 开发中                                         |

```
src/lychee_mas/
├── __init__.py            暴露 __version__ 与 REGISTRY；import 时触发所有组件注册（不触发重依赖）
├── core/
│   ├── registry.py        Registry + REGISTRY + CATEGORIES（含 memory_router）
│   └── types.py           AgentSpec/Message/Answer/Trajectory/TaskQuery/Budget/BudgetUnit（纯 dataclass）
├── runtime/
│   ├── base.py            Runtime 协议（run/intercept）+ MASGraph/MASTeam 轻量容器 + BaseRuntime
│   └── backends/
│       ├── mock_runtime.py            runtime/mock（离线确定性，纯标准库）—— 测试/CI/示例默认
│       ├── autogen_runtime.py         runtime/autogen（封装 SelectorGroupChat；autogen 惰性导入）
│       ├── autogen_injection_client.py model_client/injection（注入+路由的 ChatCompletionClient 工厂）
│       ├── hf_backend.py              HFBackend（生成 + latent 注入；torch/transformers 惰性导入）
│       └── vllm_client.py             model_client/vllm（桩）
├── memory/            ★ CDM（顶层包）：manager 接缝 + router 接缝 + 通道（NL/Latent）+ context.py（RoutingContext）+ store.py（MemoryStore）
├── trace/             ★ 归因/信用（读侧，顶层包）：FailureAttributor + CreditAssigner（桩）+ store.py（TraceStore：消息级落点 + 决策日志）
├── train/             ★ 训练（写侧，顶层包）：Trainer + trainer/maspo（桩）；RL 库放 extra [train]
├── layers/
│   ├── construct/      AgentSelector + TopologyGenerator；templates.py（Role/TEAMS）；topology_generator/static
│   ├── prune/          GraphPruner + VocabAdapter（桩）
│   └── processing/     决定跑几次 MAS：serial/（processor/serial 跑 1 次）+ parallel/（processor/parallel 并发 K 次 + aggregator 聚合：self_consistency 可跑 / dynamicagg 桩）
├── pipeline.py            Orchestrator.run（端到端编排，按 config 从 REGISTRY 取组件）
└── eval/
    ├── benchmarks/        数据 loaders + benchmark/<task> 注册（惰性加载，不在 import 读盘）
    ├── metrics.py         score（exact/aime/mc/f1）+ result_dir/write_results（math/yaml 惰性导入）
    ├── math_parsing_util.py  Qwen2.5-Math 借用的数学解析（逐字保留；heavy 依赖，仅 score_aime 内惰性 import）
    └── task_config.py     每个 task 的默认队伍 + 答案提取策略
```

## 四个设计原则

- **可插拔可消融**：每个算法 = 注册一个类（`@REGISTRY.register(category, name)`）+ 由 config/CLI 选择，换单一组件即一组对照实验，**不改编排器**。
- **Runtime 抽象隔离 AutoGen**：业务层只依赖 `runtime.Runtime` 协议；AutoGen 调用全部封装在 `runtime/backends/autogen_*.py`。
- **性能-成本联合度量**：评测同时报 accuracy / token / latency。
- **可复现**：固定随机种子；落 config 快照 + git SHA 到 `runs/`。

## 当前研究主线 = 记忆层 CDM

**CDM**（`lychee_mas.memory`）= 双通道记忆（自然语言 `nl` + 隐空间 `latent`）+ 运行时动态通道选择。隐空间通道两种物化策略：`soft_token`（免训练自压缩）与 `c2c`（训练好的 Cache-to-Cache 逐层 KV 融合）。详见 **[CDM 双通道记忆](concepts/cdm-memory.md)**。

## 从这里开始

<div class="grid cards" markdown>
