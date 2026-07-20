# LycheeMAS

**基于 [AutoGen](https://github.com/microsoft/autogen) 的多智能体系统（MAS）研究框架。** 核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），每个阶段都是对 G（或其执行轨迹 τ）的一次变换。这样所有阶段共享同一套类型与运行时，组件**可插拔、可消融**。

!!! tip "本站是什么"
    这是 LycheeMAS 的开发文档站：叙述性**指南**（安装 / 快速上手 / 架构 / 核心概念 / 组件开发）+ 从代码 docstring 自动生成的 **[API Reference](reference/lychee_mas/)**（每个模块一页，始终与源码同步）。

# 整体架构

| 阶段             | 模块                             | 职责                                           | 状态                                           |
| ---------------- | -------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 构建 Construct   | `lychee_mas.layers.construct`  | 团队组建 + 静态/动态拓扑                       | static 已实现<br />AgentInit 已实现（需 `.[construct]`） |
| 剪枝 Prune       | `lychee_mas.layers.prune`      | 网络剪枝 + 模型级词表降本                      | AgentDropout<br />MASPO<br />AgentVocab        |
| 处理 Processing  | `lychee_mas.layers.processing` | 决定跑几次 MAS + 如何归约（serial / parallel） | Serial<br />Parallel （开发中）                |
| 记忆 Memory      | `lychee_mas.memory`            | 运行时多表征记忆管理（**CDM 主线**）     | **双通道记忆轨迹 已实现（待优化）**     |
| 归因 Trace       | `lychee_mas.trace`            | 错误归因/信用（trace）   | 开发中                                         |
| 训练 Train       | `lychee_mas.train`             | MASPO，Agentic RL                              | 开发中                                         |
| 核心组件 Core   | `lychee_mas.core`              | 公共类型 + 组件注册表 REGISTRY                 | 完成（迭代中）                                 |
| 运行组件 Runtime | `lychee_mas.runtime`           | 隔离 AutoGen 的运行时抽象 + 后端               | 完成（迭代中）                                 |
| 测试 eval        | `lychee_mas.eval`              | 核心测试组件                                   | 开发中                                         |

## 源码结构（`src/lychee_mas/`）

> 框架代码全貌（每个模块的类/函数详情见 **[API Reference](reference/lychee_mas/)**，一模块一页、与源码同步）。

```text
src/lychee_mas/                       # ★ 框架本体（import 即触发全部组件注册，不触发 torch/autogen）
├── core/                             # 公共基座（零重依赖）
│   ├── types.py                      #   统一图抽象类型：AgentSpec/Message/Answer/Trajectory/TaskQuery/Budget
│   └── registry.py                   #   Registry + REGISTRY + CATEGORIES（@register / create / list / snapshot）
├── runtime/                          # 运行时抽象（隔离 AutoGen；唯一允许 import autogen 处）
│   ├── base.py                       #   Runtime 协议（run/intercept）+ MASGraph/MASTeam 容器 + BaseRuntime
│   └── backends/
│       ├── mock_runtime.py           #   runtime/mock（离线确定性，纯标准库）
│       ├── autogen_runtime.py        #   runtime/autogen（封装 SelectorGroupChat；autogen 惰性）
│       ├── autogen_injection_client.py #  model_client/injection（记忆注入+路由的 ChatCompletionClient）
│       ├── hf_backend.py             #   HFBackend（生成 + latent 注入；torch/transformers 惰性）
│       └── vllm_client.py            #   model_client/vllm（桩）
├── memory/                           # ★ 记忆层 CDM（当前主线，顶层包）
│   ├── base.py                       #   MemoryManager 接缝 + MemoryBundle（NL_Channel/Latent_Channel + strategy）
│   ├── context.py                    #   RoutingContext（跨 agent 共享的路由状态 + 决策日志）
│   ├── store.py                      #   MemoryStore（key→value 缓存）
│   ├── channels/                     #   两个主接口（nl / latent）+ latent 的子实现
│   │   ├── nl.py                     #     NLMemory：prev_output（转发）/ simplemem（SimpleMem 检索问答）
│   │   ├── latent.py                 #     LatentMemory：soft_token 压 prefix / c2c KV 融合（统一接口）
│   │   ├── c2c_channel.py            #     C2CLatentChannel（懒加载训练好的 projector 栈）
│   │   └── c2c_projector.py          #     C2CProjector（逐层 KV 融合器）
│   ├── managers/
│   │   ├── DualChannelMemory.py      #     memory_manager/cdm（双通道，主线）
│   │   └── external.py               #     memory_manager/{mem0, ama}（外部基线桩）
│   └── routing/                      #   触发接缝：本轮用哪个通道
│       ├── base.py                   #     MemoryRouter + RouterInputs + RouteDecision + Channel
│       ├── static.py                 #     memory_router/{static, fixed}
│       ├── learned.py                #     memory_router/learned（桩）
│       └── soft_gate.py              #     memory_router/soft_gate（软门控，桩）
├── trace/                            # ★ 归因/信用（读侧，顶层包）
│   ├── base.py                       #   Attribution / FailureAttributor / CreditAssigner
│   ├── store.py                      #   TraceStore（消息级落点 + 决策日志，可选 JSONL）
│   └── __init__.py                   #   attributor/{all_at_once,step_by_step,binary_search}·credit_assigner/attribution_guided（桩）
├── train/                            # ★ 训练（写侧，顶层包）
│   ├── base.py                       #   Trainer 协议
│   └── __init__.py                   #   trainer/maspo（桩）；RL 库放 extra [train]
├── layers/                           # 其余变换层
│   ├── construct/                    #   团队组建 + 拓扑
│   │   ├── base.py                   #     AgentSelector / TopologyGenerator 协议
│   │   ├── templates.py              #     Role / TEAMS 队伍模板；topology_generator/static
│   │   └── selectors/                #     agent_selector/agentinit —— AgentInit（EMNLP'25）
│   │       ├── agentinit.py          #       多样性×相关性 Pareto 选择（pool 已实现 / generate）
│   │       ├── pool.py               #       离线候选池 + 确定性嵌入
│   │       ├── _pareto.py            #       非支配排序 + Vendi 多样性 + 相关性
│   │       ├── _generate.py          #       LLM 现场生成角色 + 批判 + 挑组
│   │       ├── _llm.py               #       OpenAI 兼容 chat（tenacity 重试）
│   │       └── embedder.py           #       all-MiniLM-L6-v2 嵌入
│   ├── prune/                        #   graph_pruner/{agentdropout*, agentprune}·vocab_adapter/agentvocab（桩）
│   └── processing/                   #   决定跑几次 + 如何归约
│       ├── base.py                   #     Processor / TrajectoryAggregator 协议
│       ├── serial/                   #     processor/serial（跑 1 次）
│       └── parallel/                 #     processor/parallel（并发 K 次）+ aggregator/{self_consistency, dynamicagg}
├── pipeline.py                       # Orchestrator.run（端到端编排；可选 selector / aggregator）
└── eval/                             # 评测
    ├── benchmarks/                   #   文本类{gsm8k,aime_2024,medqa,arc_easy,openbookqa,locomo10} + 子系统{human_eval,gaia_validation,aftraj_audit,agent_collab_*,mast_failure,open_agent_traces}（共 19，惰性加载）
    ├── metrics.py                    #   score（exact/aime/mc/f1）+ write_results（落盘）
    ├── math_parsing_util.py          #   Qwen2.5-Math 借用的数学解析（heavy 惰性）
    └── task_config.py                #   每个 task 的默认队伍 + 答案提取策略

configs/    组件分组 YAML（runtime/ memory/ topology/ aggregator/ agents/ …）
examples/   可运行示例（离线 mock 优先）
scripts/    实验入口（run_experiment / run_mas / C2C 训练评测 / AgentInit 消融）
tests/      pytest（离线、零重依赖）
docs/       MkDocs 文档站（本站）+ 开发文档（DEVELOPMENT.md / dev/*）
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

- :material-download: **[安装](installation.md)** —— conda + uv，extras（dev / all / construct / docs）
- :material-rocket-launch: **[快速上手](quickstart.md)** —— `make demo` / 离线 CLI / 带 CDM 的 AIME 实验
- :material-sitemap: **[架构与设计](DEVELOPMENT.md)** —— 目录职责 + CDM 数据流 + 迁移映射
- :material-puzzle: **[组件开发](dev/README.md)** —— 六步配方：注册一个类 = 一组消融
- :material-brain: **[CDM 双通道记忆](concepts/cdm-memory.md)** —— 当前研究主线的数据流
- :material-book-open-variant: **[API Reference](reference/lychee_mas/)** —— 每个模块一页，与源码同步

</div>
