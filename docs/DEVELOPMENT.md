# LycheeMAS 开发文档

> 版本 v1.0 · 2026-06 · 配合根目录 `CLAUDE.md`（工程约束/黄金法则）阅读。本文聚焦**新 `src/lychee_mas/` 包**的架构、目录职责、运行方式、扩展配方、CDM 数据流、迁移映射、测试与各层扩展点。
>
> **状态**：重构已落地并离线自验证通过——`import lychee_mas` 零重依赖；`pytest` 24 passed；`ruff` All checks passed；example 可跑；autogen 仅存在于 `runtime/backends/autogen_*.py`。旧代码完整保留在 `src/LycheeMAS/`（参照，未删改）。
>
> **30 秒上手**：`make demo`（离线端到端）→ `make test`（24 测试）→ 读 §5「六步配方」加你的组件。

---

## 1. 架构总览

LycheeMAS 把整个 MAS 表示为一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V 节点=智能体，E 边=通信信道，W 边权，T 多轮时序，M 记忆状态）。每一层是对 G（或其执行轨迹 τ）的一次变换：

```
TaskQuery ──▶ [construct] ──▶ MASGraph ──▶ [Runtime.run] ──▶ Trajectory ──▶ [aggregate] ──▶ Answer
                                 │                │                              │
                            [prune]         intercept(Message)         [attribute_train: 归因→信用→训练]
                                                  │
                                          TraceStore / [memory 抽取]
```

四个设计原则：

1. **可插拔可消融**：每个算法 = 注册一个类（`@REGISTRY.register(category, name)`）+ 由 config 选择。换单一组件即一组对照实验，**不改编排器**。
2. **Runtime 抽象隔离 AutoGen**：业务层只依赖 `runtime.base.Runtime` 协议；AutoGen 调用全部封装在 `runtime/backends/autogen_*.py`（唯一允许 `import autogen_*` 的位置），预留 MAF 迁移。
3. **性能-成本联合度量**：评测同时报 accuracy / token / latency（记忆层加 memory-hit，融合层加 fusion-gain）。
4. **可复现**：固定随机种子；落 config 快照 + git SHA 到 `runs/`。

### 惰性导入约定（关键）

核心骨架**零运行依赖**：纯标准库即可 `import lychee_mas` 并跑通离线 mock。所有重依赖（`torch` / `transformers` / `autogen_*` / `numpy` / `yaml` / `sympy` / `datasets` …）**惰性导入**——放在函数/方法内部或 `try-import`。注册组件的模块在 import 时**不得触发** torch/autogen 导入。

验证（离线、无重依赖）：

```bash
PYTHONPATH=src python -c "import lychee_mas; from lychee_mas.core.registry import REGISTRY; print(REGISTRY.snapshot())"
```

---

## 2. 目录与各模块职责

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
├── stores/
│   ├── trace_store.py     TraceStore：消息级落点 + 决策日志（可选 JSONL）
│   └── memory_store.py    MemoryStore：key→value 缓存接缝（FIFO 容量上限）
├── layers/
│   ├── construct/      AgentSelector + TopologyGenerator；templates.py（Role/TEAMS）；topology_generator/static
│   ├── prune/          GraphPruner + VocabAdapter（桩）
│   ├── memory/         记忆方法接缝（manager）+ 触发接缝（router）+ 通道（NL/Latent）+ RoutingContext
│   ├── aggregate/      TrajectoryAggregator；self_consistency（可跑）+ dynamicagg（桩）
│   └── attribute_train/ FailureAttributor + CreditAssigner + Trainer（桩）
├── pipeline.py            Orchestrator.run（端到端编排，按 config 从 REGISTRY 取组件）
└── eval/
    ├── benchmarks/        数据 loaders + benchmark/<task> 注册（惰性加载，不在 import 读盘）
    ├── metrics.py         score（exact/aime/mc/f1）+ result_dir/write_results（math/yaml 惰性导入）
    ├── math_parsing_util.py  Qwen2.5-Math 借用的数学解析（逐字保留；heavy 依赖，仅 score_aime 内惰性 import）
    └── task_config.py     每个 task 的默认队伍 + 答案提取策略
```

仓库根还包含：`pyproject.toml`（零核心依赖 + extras）、`Makefile`、`.gitignore`、`README.md`、`configs/`、`scripts/run_experiment.py`、`examples/01_static_chain_e2e.py`、`tests/`。

> 旧自有代码保留在 `src/LycheeMAS/`（参照，未删改）；`src/autogen/` 是 vendored 的 AutoGen 源码（忽略，不打包/不 lint）。

---

## 3. 运行方式

```bash
make demo      # PYTHONPATH=src python examples/01_static_chain_e2e.py  （离线端到端）
make test      # PYTHONPATH=src pytest -q
make lint      # ruff check src
make install   # pip install -e ".[dev]"  （仅骨架 + dev）
make install-all  # pip install -e ".[all]"  （全量）
```

实验入口（默认 `runtime=mock` 可离线跑；真实跑分用 `--runtime autogen --benchmark gsm8k`，需 extras `[all]`）：

```bash
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --team default --aggregator self_consistency \
    --questions "2 plus 2 is 4" "answer is 7"
```

真实 HF 后端模型路径走构造参数或环境变量（**不硬编码绝对路径**）：

```bash
export LYCHEE_HF_MODEL=/path/to/Qwen3-4B
```

---

## 4. 核心抽象（先读这些再写）

- **`core/types.py`**（所有层共享，纯 dataclass）：`AgentSpec`（图节点）/`Message`（边上一次传输，`.tokens`）/`Answer`（L4 融合单元）/`Trajectory`（一次执行 τ，`.add/.total_tokens/.num_rounds`）/`TaskQuery`（`gold` 供评测）/`Budget`+`BudgetUnit`。
- **`core/registry.py`**：`REGISTRY.register / get / create / list / snapshot`。`CATEGORIES` 已登记：`runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, attributor, credit_assigner, trainer, benchmark`。新增类别须在此同步登记。
- **`runtime/base.py`**：`Runtime` 协议（`async run(team, query)->Trajectory`、`intercept(hook)`）；`MASGraph`（节点=AgentSpec，边=邻接/顺序链，`rounds`）；`BaseRuntime`（实现 `intercept` 的样板 + `_emit` 逐消息回调）。

---

## 5. 如何新增一个组件（六步配方 = 一篇消融）

以「新增一个聚合器」为例，其它层同理：

1. **读接口**：`layers/aggregate/base.py` 确认 `TrajectoryAggregator` 协议（输入 `list[Trajectory]`，输出 `Answer`）。
2. **写实现**：`layers/aggregate/aggregators/my_agg.py`
   ```python
   from lychee_mas.core.registry import REGISTRY
   from lychee_mas.core.types import Answer, Trajectory

   @REGISTRY.register("aggregator", "my_agg")
   class MyAgg:
       def __init__(self, k: int = 5): self.k = k
       def aggregate(self, trajectories: list[Trajectory]) -> Answer: ...
   ```
3. **触发注册**：在 `layers/aggregate/aggregators/__init__.py` 里 import 该模块（包 `__init__` 会被 `lychee_mas/__init__` 链式 import）。
4. **加配置**：`configs/aggregator/my_agg.yaml`（含超参与默认值）。
5. **加测试**：`tests/test_my_agg.py`，用 mock runtime 断言行为与「性能/降本」指标。
6. **验证**：`make lint && make test`，再 `make demo` 看端到端不回归。

**绝不**为此改 `pipeline.py`——`Orchestrator` 只按名字从 `REGISTRY` 取组件。做对照实验只改 config / CLI（如 `--aggregator my_agg` vs `self_consistency`）。

> ⚠️ 若新增的组件依赖重库（torch/autogen 等），把 import 放进方法内部（惰性），保证注册模块在离线环境可被 import。

---

## 6. 记忆层（memory）CDM 数据流（重点，逻辑原样保留）

CDM = **双通道记忆**：用**单一** source（seed 基础上下文 + 运行中 transcript）物化出路由器选定的任意通道，保证 apples-to-apples（对比时只变「注入/路由」这一个变量）。两个对称的可替换接缝：

- **方法接缝（manager）** = 记忆如何存储/召回/物化各通道 → `memory_manager` 类别（`cdm` 默认 / `mem0` / `ama`）。
- **触发接缝（router）** = 当前 agent 本轮用哪个通道 → `memory_router` 类别（`static` / `fixed` / `learned` / `soft_gate`）。

一次 agent 发言（在 `model_client/injection` 的 `create()` 里汇合）：

```
LLMMessage 历史 ─_to_chat─▶ [{role,content}]
  ① memory.observe(chat)                        更新记忆库（transcript 去重 + 失效 latent 缓存）
  ② router.decide(RouterInputs(role,task,turn,sender,availability,same_model_pair)) ─▶ RouteDecision(channel,P)
       └ _enforce_availability：latent 不可用 / 异构（非同模型）对 ⇒ 回退 nl（both 丢 latent 留 nl）
  ③ memory.recall(decision, query) ─▶ MemoryBundle(nl_text?, latent_prefix?)
       ├ channel none   → 空 bundle
       ├ channel nl     → nl_text = NLMemory.recall(query)（上一个 agent 输出，加 PREV_OUTPUT_HEADER，不截断）
       ├ channel latent → latent_prefix = LatentMemory.build_prefix(source, P)
       └ channel both   → 两者都产
  ④ 注入：nl_text 作 system 消息插在开头 system 之后；latent_prefix 在 backend embedding 层拼接
  ⑤ 生成：有 latent → generate_chat_with_prefix；否则 generate_chat
  ⑥ 记账：bump_turn + RequestUsage + ctx.log_decision（成本/输入/输出，可选写 TraceStore）
```

**latent 通道**（`memory/channels/latent.py`，torch 惰性导入）：把源文本经 HF 后端编码成末层 hidden `(1,T,2560)`，再压成 `(1,P,2560)` 的 soft prefix：

- `soft_token`（默认，正确）：`softmax(h @ E^T / tau) @ E` 投回**输入嵌入空间**，再 `segment_mean` 到 P。修了「直接拿末层 hidden 当 prefix 是表示空间不匹配」的坑。
- `segment_mean` / `stride` / `tail`：naive 对照路径（直接在末层 hidden 上压缩）。

**硬约束**（原样保留）：
- prefix 形状必须 `(1,P,H)` 且 `H==2560`（`hf_backend` 运行期 `assert`）。
- latent 只能在**同模型对**（hidden 对齐）且 latent 可用时跨 agent 传，否则路由器回退 NL（`_enforce_availability` + `RoutingContext.same_model_pair`）。
- `PREV_OUTPUT_HEADER`（"## Input from the previous agent"）**集中定义在** `memory/channels/nl.py`，`construct/templates.py` import 复用（消除重复字符串）。
- 单次对话内按 `(len(source), P)` 缓存 latent prefix，source 一变即失效。

---

## 7. 迁移映射表（旧 `src/LycheeMAS/` → 新 `src/lychee_mas/`）

| 旧路径 | 新路径 | 备注 |
|---|---|---|
| `memory/manager.py` | `layers/memory/base.py` | `MemoryManager`(ABC) + `MemoryBundle`；torch 注解改前向引用（惰性）|
| `memory/nl_memory.py` | `layers/memory/channels/nl.py` | `PREV_OUTPUT_HEADER` 集中定义于此 |
| `memory/latent_memory.py` | `layers/memory/channels/latent.py` | `soft_token_prefix/compress_hidden/LatentMemory`；torch 方法内惰性 import |
| `memory/managers/dual_channel.py` | `layers/memory/managers/cdm.py` | `memory_manager/cdm`（逻辑/张量形状原样）|
| `memory/managers/external_baselines.py` | `layers/memory/managers/external.py` | `memory_manager/mem0`、`/ama`（桩，统一报错文案）|
| `router/base.py` | `layers/memory/routing/base.py` | `MemoryRouter/RouterInputs/RouteDecision/Channel` + `_enforce_availability` |
| `router/static_router.py` | `layers/memory/routing/static.py` | `memory_router/static` + `memory_router/fixed`（always-X）|
| `router/learned_router.py` | `layers/memory/routing/learned.py` | `memory_router/learned`（占位 fallback）|
| `router/soft_gate.py` | `layers/memory/routing/soft_gate.py` | `memory_router/soft_gate`（占位 fallback）|
| `context/routing_context.py` | `layers/memory/routing/context.py` | `RoutingContext`；决策日志可选写 `TraceStore` |
| `models/injection_client.py` | `runtime/backends/autogen_injection_client.py` | `model_client/injection`；autogen+torch 惰性（工厂延迟构造子类）|
| `models/vllm_prompt_embeds_client.py` | `runtime/backends/vllm_client.py` | `model_client/vllm`（桩）|
| `backends/backend_hf.py` | `runtime/backends/hf_backend.py` | `HFBackend`；torch/transformers 惰性；删硬编码路径→`LYCHEE_HF_MODEL`/构造参数 |
| `teams/groupchat.py` | `runtime/backends/autogen_runtime.py` | `runtime/autogen`；`run` 返回 `Trajectory`；autogen 惰性 |
| `agents/roles.py` | `layers/construct/templates.py` | `Role/ROLE_SYSTEM/TEAMS` 保留；`topology_generator/static`；复用 `PREV_OUTPUT_HEADER` |
| `benchs/loaders.py` | `eval/benchmarks/__init__.py` | 各 task 注册 `benchmark/<task>`；datasets 惰性、不在 import 读盘 |
| `benchs/metrics.py` | `eval/metrics.py` | score 逻辑保持；math/yaml 惰性导入 |
| `benchs/math_parsing_util.py` | `eval/math_parsing_util.py` | 逐字复制（Qwen2.5-Math 借用）|
| `benchs/task_config.py` | `eval/task_config.py` | team/extractor 映射保持 |
| `configs/config.py` + `*.yaml` | `configs/` | 旧 YAML 复制 + 新增组件分组 config + 顶层 `config.yaml` |
| `run_mas.py` | `pipeline.py`（编排循环）+ `scripts/run_experiment.py`（CLI/落盘）| 默认 `runtime=mock` 离线可跑 |

散落的 `make_router/make_memory/FIXED/EXTERNAL` dict → 由 **REGISTRY** 取代（按 name 取组件）。

---

## 8. 测试说明

`tests/`（离线、零重依赖；torch/autogen 相关不在收集期导入）：

- `test_registry.py`：注册/查询/重复报错/snapshot 含 `memory_router`、`memory_manager/cdm`、`runtime/mock`。
- `test_types.py`：`Trajectory.total_tokens` / `num_rounds`、Message.tokens。
- `test_router.py`：`static.decide` + `_enforce_availability`（latent 不可用 / 异构对回退 nl/none）。
- `test_trace_store.py`：TraceStore 消息 hook + 决策日志 + reset；MemoryStore FIFO。
- `test_mock_runtime.py`：MockRuntime 与 Orchestrator 离线跑出 Trajectory。
- `test_self_consistency.py`：多数投票（对 list[Answer]/list[Trajectory] 取众数）。

`pyproject.toml` 的 `[tool.pytest.ini_options]` 用 `testpaths=["tests"]` + `pythonpath=["src"]`，**只收集本框架测试**，避免触碰 vendored `src/autogen` 的测试（它们需 autogen 才能 collect）。

```bash
make test     # 全绿
make lint     # ruff check src 通过（line-length 100；math_parsing_util 逐字保留，按文件忽略风格）
```

---

## 9. L2 / L4 / L5 扩展点（为后续研究留口）

各层 `base.py` 已定义协议；注册的占位实现（`raise NotImplementedError("<name>: not wired yet (TODO)")`）可被 `REGISTRY.list` 看到，方便消融矩阵在代码里可见：

- **L2**：`graph_pruner/{agentdropout, agentdropout_v2, agentprune}`、`vocab_adapter/agentvocab`（均桩）。迁移已发表逻辑时实现 `prune(graph, context)` / `adapt(agent, context)`，保留原始引用与许可证。
- **L4**：`aggregator/self_consistency`（**纯标准库可跑**多数投票）+ `aggregator/dynamicagg`（在研，桩）。
- **L5**：`attributor/{all_at_once, step_by_step, binary_search}`、`credit_assigner/attribution_guided`（核心贡献）、`trainer/maspo`（提示级优化，廉价基线/暖启动）。`topology_rl/marl` 接 RL 库时再加（放 optional extra `[train]`，只在 `trainer/*` 实现里依赖，不污染基座）。

闭环路线：`attributor` 归因 → `credit_assigner/attribution_guided` 转 per-agent 稠密信用 → `trainer.train` → 反哺。

---

## 10. AutoGen / torch 惰性导入约定（务必遵守）

1. **不要在业务代码里 `import autogen_*`**。一切运行时能力只通过 `lychee_mas.runtime` 的 `Runtime` 协议使用。**唯一允许 import autogen 的位置：`src/lychee_mas/runtime/backends/autogen_*.py`**，且其中的 `import autogen_*` 也惰性化到函数/工厂内部（保证 import 这些模块、触发注册时不需要 autogen）。
2. **torch / transformers**：只在 `hf_backend.py`、`memory/channels/latent.py`、`autogen_injection_client.py` 的**方法内部**导入。注册 `memory_manager/cdm` 的 `cdm.py` import 时不触发 torch。
3. **yaml / sympy / datasets / numpy**：分别在 `metrics._dump_config` / `metrics.score_aime`（经 `math_parsing_util`）/ `benchmarks._parquet` 等函数内部惰性导入。
4. **MAF 迁移**：未来只需新增 `runtime/backends/maf_runtime.py` 并注册，业务层零改动。

自检命令（应输出 `HEAVY LOADED: NONE`）：

```bash
PYTHONPATH=src python -c "import sys, lychee_mas; print('HEAVY LOADED:', [m for m in ('torch','transformers','autogen_core','autogen_agentchat','numpy','yaml','sympy','datasets') if m in sys.modules] or 'NONE')"
```
