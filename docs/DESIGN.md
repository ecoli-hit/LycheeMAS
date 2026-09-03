# LycheeMAS 架构设计

> 本文件是 LycheeMAS 的**唯一架构设计文档**：框架目标、总体架构、各模块职责与接口契约、组件注册全景、评测体系与验证策略。面向编码代理的操作规范（环境、命令、代码规范、检查清单）见仓库根 `CLAUDE.md`。
>
> 版本范围：LycheeMAS v0.3。

---

## 1. 核心目标

**LycheeMAS 是一个多智能体系统（MAS）研究框架**。核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），系统的每一次演化都是对 G（或其执行轨迹 τ）的一次变换。由此得到四条设计原则：

1. **统一表示**：所有模块共享同一套公共类型（`core/types.py`）与同一个运行时协议（`runtime/base.py`），模块之间只通过这些契约交互。
2. **可插拔、可消融**：每个算法都是「注册到 REGISTRY 的一个类」，由配置/CLI 按名字选择；做对照实验只换组件名，不改编排代码。未实现的方法以「桩」占位注册，让消融矩阵在代码里可见。
3. **运行时无关**：图的执行引擎是可替换后端（当前提供 mock / autogen / langgraph 三个），一切引擎调用封装在 `runtime/backends/` 之后，业务层零依赖引擎。
4. **离线可跑、可复现**：核心代码与示例在无 GPU / 无 API key 的环境必须能跑（mock runtime + 零重依赖）；实验固定种子、落 config 快照与 git SHA，同时报告 accuracy / token / latency。

新增代码遵循**显式错误原则**：组件遇到不支持的输入/配置显式报错，不静默降级、不写死兜底数据。

---

## 2. 总体架构

一次任务的完整生命周期：

```
                    ┌─────────────────────── 离线优化（Optimizer, GEPA 式 compile 循环）───────────────────────┐
                    │                                                                                        │
TaskQuery ──► 构建器（construct）──► 运行前插件链（PreRunPlugin*）──► 执行（runtime × processing）──► 运行后插件链（PostRunPlugin*）──► 结果
              AgentSelector             如 prune 适配器            每轮 agent 发言内嵌            如 attribution 适配器
              TopologyGenerator         变换图 G                   记忆注入六步（memory）           消费轨迹 τ / 写 TraceStore
              产出初始 MASGraph                                    serial/parallel 决定跑几次
```

三类角色：

- **构建器（construct）**：从任务产出初始图 G（一次性，非插件）。
- **运行时组件（memory / processing）**：存在于执行内部——memory 在每次 agent 发言时被调用（逐轮），processing 决定整个执行跑几次与如何归约（包裹执行）。
- **插件（pre_run / post_run / optimizer）**：挂在执行外部的可选优化器——运行前变换图、运行后消费轨迹、离线迭代改进系统。prune / trace / train 的能力经适配器以插件形式挂载。

---

## 3. 共享基座

### 3.1 `core/` — 类型与注册表

- **`core/types.py`**（纯 dataclass，零重依赖）：`AgentSpec`（图节点画像）、`Message`（一条通信，`.tokens`）、`Answer`（候选答案）、`Trajectory`（一次执行 τ，`.total_tokens/.num_rounds`）、`TaskQuery`（含 `gold`）、`Budget/BudgetUnit`。
- **`core/registry.py`**：`@REGISTRY.register(category, name)` 注册、`REGISTRY.create(category, name, **kwargs)` 实例化、`REGISTRY.snapshot()` 全景。**CATEGORIES（18 个）**：
  `runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, processor, attributor, credit_assigner, trainer, benchmark, pre_run_plugin, post_run_plugin, optimizer, pre_run_optimizer`。

### 3.2 `runtime/` — 运行时抽象与后端

**协议**（`runtime/base.py`）：

```python
class Runtime(Protocol):
    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory: ...
    def intercept(self, hook: Callable[[Message], None]) -> None: ...
```

`MASGraph`（nodes + edges + rounds）是图 G 的轻量容器；`BaseRuntime` 提供 intercept/_emit 样板。`intercept` 逐消息回调是轨迹落盘与记忆抽取的数据来源。

**后端**（`runtime/backends/`）：

| 注册名 | 职责 |
|---|---|
| `runtime/mock` | 确定性离线后端（测试/CI/示例默认，纯标准库） |
| `runtime/autogen` | AutoGen 群聊执行（回归基线；含工具型团队预设） |
| `runtime/langgraph` | LangGraph StateGraph 执行（每 AgentSpec 一节点，条件边控终止；MVP 支持纯文本团队，工具型团队显式报错） |
| `model_client/injection` | AutoGen 侧的注入 client（把注入引擎接入 AutoGen 的 ChatCompletionClient 接口） |
| `model_client/vllm` | 桩 |

生成后端（`hf_backend` 本地 HF 模型 / `openai_api_backend` API）提供生成原语：`generate_chat`、`encode_hidden`、`generate_chat_with_prefix`、KV 相关原语等；不注册、被运行时组合使用。

**共享注入引擎**（`runtime/injection.py`）：每次 agent 发言的「记忆注入六步」的唯一实现——

```
① memory.observe(chat)                       更新记忆
② router.decide(RouterInputs) → RouteDecision 选记忆通道
③ memory.recall(decision, query) → MemoryBundle 物化通道内容
④ system 段注入（保序：原始 system → 额外指令 → 记忆文本 → 对话）
⑤ 按 bundle 内容分支生成（latent 载荷优先，否则普通生成）
⑥ 记账：bump_turn / log_decision / log_span
```

接口：`run_injection_step(backend, ctx, InjectionRequest) -> InjectionResult`。AutoGen 的注入 client 与 LangGraph 的节点函数都调它；引擎只依赖 memory 层公共契约，不含任何引擎专属逻辑（AutoGen 的工具调用解析、格式修复以回调传入）。

**LangGraph 映射**：`MASState = {task, history, agent_turns, done}`；每 AgentSpec 一个节点函数（组消息 → 注入引擎 → 追加 history）；`START → 首节点`，每节点后条件边：输出含 `APPROVE` 或 `agent_turns ≥ max_turns`（= `len(agents) × max_rounds`）→ END，否则轮转下一节点；`recursion_limit` 按 max_turns 显式放宽；Trajectory 组装与 spans 落盘与 autogen 后端对齐（逐条 Message + `_emit`、meta 带 decisions/stop_reason）。

**AutoGen 退役门槛**：在代表性任务集上，autogen 与 langgraph 双后端在 deterministic 设置下 final answer、`model_call_start.input_messages`、token 记账三项全部一致后，autogen 方可退役。

### 3.3 `pipeline.py` — 编排器

`Orchestrator` 只按名字从 REGISTRY 取组件：`build_graph`（agent_selector? + topology_generator）→ 运行前插件链 → `runtime.run` → 运行后插件链 → 可选 aggregator。新增任何组件都不改此文件。

---

## 4. 五阶段模块

### 4.1 construct（构建器，`layers/construct/`）

回答「由谁组队、怎么连」。协议：`AgentSelector.select(query, budget) -> list[AgentSpec]`；`TopologyGenerator.build(agents, query) -> MASGraph`。队伍模板约定：末位角色以 `APPROVE: <final answer>` 收尾（运行时据此终止并抽答案）；每个 agent 绑定自身 role 的注入路径，共享 backend + RoutingContext。

### 4.2 prune（经运行前插件挂载，`layers/prune/`）

回答「哪些边/词表可裁掉以降本」。协议：`GraphPruner.prune(graph, context)`；`VocabAdapter.adapt(agent, context)`。经 `pre_run_plugin/prune` 适配器进入执行流程。已实现 `graph_pruner/agentprune`（AgentPrune 时空掩码剪枝：逐边可训练 logit + 伯努利采样 + REINFORCE + one-shot 剪枝，纯标准库；训练/复现脚本 `scripts/run_agentprune_gsm8k.py`）。

### 4.3 memory（运行时组件，`memory/`）

回答「智能体之间记住什么、以什么表征传递」。两个对称接缝：

- **方法接缝** `memory_manager`：`observe(messages)` / `recall(decision, query) -> MemoryBundle` / `reset()`；`MemoryBundle` 按通道携带「内容 + 产出方法」。
- **触发接缝** `memory_router`：`decide(RouterInputs) -> RouteDecision(channel ∈ none/nl/latent/both)`。

内部组织：`channels/`（表征通道：自然语言 / 隐空间）+ `managers/`（存取策略）+ `routing/`（路由策略）+ `store.py`（缓存接缝）+ `context.py`（RoutingContext：跨 agent 共享的路由/记账状态，含决策日志与 span 落盘）。消费方只认 `MemoryBundle` / `RouteDecision`，换 manager/router 即一组对照实验。

### 4.4 processing（运行时组件，`layers/processing/`）

回答「一个任务跑几次、多次结果如何归约」。协议：`Processor.run(runner) -> ProcessingResult`（serial 跑 1 次 / parallel 并发 K 次）；`TrajectoryAggregator.aggregate(trajectories) -> Answer` 是 parallel 的可插拔归约策略。

### 4.5 trace（经运行后插件挂载，`trace/`）

回答「一条轨迹里谁该为成败负责」。协议：`FailureAttributor.attribute(trajectory, context) -> list[Attribution]`；`CreditAssigner.credits(attributions, reward) -> dict[str, float]`；`TraceStore` 是消息级轨迹与决策日志的统一落点。经 `post_run_plugin/attribution` 适配器进入执行流程。

### 4.6 train（离线，`train/`）

回答「如何用信用信号改进系统」。协议：`Trainer.credits(...) / train(...)`（RL 路线，消费 trace 的信用）；提示级优化走插件系统的 `Optimizer`（见 §5）。RL 库只在 trainer 实现里依赖（optional extra）。

---

## 5. 插件系统（`plugins/`）

三个接口分立（`plugins/base.py`）：

```python
@dataclass
class RunContext:                       # 插件可见的运行上下文
    task: str; trace_store: Any; routing_ctx: Any; backend: Any; meta: dict

class PreRunPlugin(Protocol):           # 注册类别 pre_run_plugin
    def before_run(self, graph: MASGraph, query: TaskQuery, ctx: RunContext) -> MASGraph: ...

class PostRunPlugin(Protocol):          # 注册类别 post_run_plugin
    def after_run(self, trajectory: Trajectory, score: Optional[float], ctx: RunContext) -> None: ...

Metric = Callable[[Trajectory, TaskQuery], float]

class Optimizer(Protocol):              # 注册类别 optimizer（离线 compile 循环）
    def optimize(self, system: MASProgram, trainset: Sequence[TaskQuery], metric: Metric) -> MASProgram: ...
```

- **挂载**：Orchestrator 接受 `pre_plugins` / `post_plugins` 名单，按序执行；`before_run` 返回值必须是 MASGraph，否则显式报错。不配置插件时行为与无插件完全一致。
- **适配器**（`plugins/adapters.py`）：`pre_run_plugin/prune` 包装任意已注册 `graph_pruner`；`post_run_plugin/attribution` 串 attributor + credit_assigner，把结果写 `trajectory.meta` 与 TraceStore。
- **MASProgram**（`plugins/program.py`）：把一个 MAS 系统表示为「可变异文本组件」的集合（`components: dict[str, str]`，键如 `agent:<name>:system_prompt`；`from_graph / apply_to / mutated`）。这是 Optimizer 的操作对象。
- **prerun**（`plugins/prerun/`，注册类别 `pre_run_optimizer`）：LangGraph 原生「运行前优化」统一接口——`optimize_langgraph(sg, method, **kw)` 按 `method` 分发，**图进图出**（传入/返回未编译 StateGraph）。节点契约：`add_node(name, fn, metadata={"agent_spec": spec})`，`spec.system_prompt`=可变异提示模板、`spec.meta["predecessors"]`=通信前驱（`graphview.py` 是读写唯一通道）。已接：`maspo`（MASPO 联合提示优化，ICML 2026：多粒度成对评估 + 错位驱动采样 + 进化 beam search，fixed-rounds 坐标上升；optimize 落 prompt JSON / apply 即插即用挂载）、`agentprune`（复用 graph_pruner/agentprune 的 threshold 实现剪 LangGraph 边）。与 `pre_run_plugin`（MASGraph 接缝）并存；langgraph 在该包内惰性导入，selfcheck 不破。
- **GEPA**（`plugins/gepa/`，注册 `optimizer/gepa`）：反思式提示演化——候选池（program + per-instance 分数向量）→ Pareto 采样母本 → 轮换选一个可变组件 → minibatch rollout 收轨迹与反馈 → LLM 反思产出新组件文本 → minibatch 提升才全量评估入池 → 预算（`max_metric_calls`）耗尽返回最优。Pareto 选择为纯函数（`gepa/pareto.py`，离线可测）；rollout 与 reflector 可注入。

---

## 6. 评测体系（`eval/`）

- **benchmark（20 个注册名）**：文本推理/知识 `gsm8k, aime_2024, math500, medqa, arc_easy, openbookqa, locomo10`；代码/通用助理 `human_eval, gaia_validation(_level_1..3)`；MAS 轨迹分析 `aftraj_audit(_test), agent_collab_{idr,rtd,cpr,clc}, mast_failure, open_agent_traces`。统一记录格式 `{task, kind, question, gold, context}`；数据加载惰性，数据准备走 `[benchmark]` extra；入口 `benchmarks.load(task, n)` / `prepare(task)`。
- **推理 / 打分分离**：`scripts/run_mas.py` 只推理（落 `predictions.jsonl` + `spans.jsonl` + config 快照）；`scripts/analyze_benchmark_run.py --score-predictions` 事后打分（落 `outputs.jsonl` + `metrics.json`）。
- **pass@K**：`run_mas --samples K` 每题采样 K 次（K>1 需 `backend.do_sample=true`；断点续跑按 `(case_id, k_index)` 去重）；打分侧 `metrics.aggregate_samples` 按 case 聚合——pass@1 = 各 case 内 K 份得分均值再对 case 平均；pass@K = 各 case best-of-K 再平均。
- **指标**：评分类型（`kind`）覆盖 mc / exact / aime（数值+符号等价）/ f1 / human_eval / gaia / MAS 专用指标族；评测同时报告 accuracy / token / latency。
- **spans**：`JsonlSpanLogger` 增量落运行事件（runtime_start/end、model_call_start/end、decision 等），是回归对拍与归因分析的原始数据。

---

## 7. 组件注册全景（18 类别）

| 类别 | 已实现/可跑 | 桩（待接） |
|---|---|---|
| `runtime` | `mock`, `autogen`, `langgraph` | — |
| `model_client` | `injection` | `vllm` |
| `agent_selector` | `agentinit` | — |
| `topology_generator` | `static` | — |
| `graph_pruner` | `agentprune`（AgentPrune 时空掩码，ICLR 2025） | `agentdropout`, `agentdropout_v2` |
| `vocab_adapter` | — | `agentvocab` |
| `memory_manager` | `cdm` | `mem0`, `ama` |
| `memory_router` | `static`, `fixed` | `learned`, `soft_gate` |
| `processor` | `serial`, `parallel` | — |
| `aggregator` | `self_consistency` | `dynamicagg` |
| `attributor` | — | `all_at_once`, `step_by_step`, `binary_search` |
| `credit_assigner` | — | `attribution_guided` |
| `trainer` | — | —（RL 训练器待接；MASPO 按其本义迁至 `pre_run_optimizer/maspo`） |
| `benchmark` | 20 个（§6） | — |
| `pre_run_plugin` | `prune`（适配器） | — |
| `pre_run_optimizer` | `maspo`（MASPO 联合提示优化，ICML 2026）, `agentprune`（统一接口适配） | — |
| `post_run_plugin` | `attribution`（适配器） | — |
| `optimizer` | `gepa` | — |

> 桩能被 `REGISTRY.list` 看到是有意为之：占好名字、让消融矩阵可见。组件状态变化时同步更新本表。

---

## 8. 验证策略

1. **离线三件套**（每次改动收尾必跑）：`make lint`（ruff）、`make test`（pytest，全部离线）、`make selfcheck`（`HEAVY LOADED: NONE`——torch/transformers/autogen/langgraph 等重依赖不得在 import 时加载）。
2. **黄金基线对拍**：重构涉及执行链路时，deterministic 设置下与改前基线逐字比对 `predictions.jsonl`、滤易变字段后比对 `spans.jsonl`。
3. **双后端对拍**：autogen 与 langgraph 在同一配置下 final answer / `input_messages` / token 记账一致（也是 AutoGen 退役门槛）。
4. **可复现**：固定 `--seed`；每次实验落 config 快照 + git SHA 到 `runs/`。
