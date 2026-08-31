# `layers` — 层变换（Construct / Prune / Processing）

## 功能

对图 G 的三类变换。construct 是**构建器**（产初始图）；prune 经 `pre_run_plugin/prune` 适配器以**运行前插件**形式挂载；processing 是**运行时组件**（包裹执行，决定跑几次与如何归约）。

## `construct/` — 构建层

回答「由谁组队、怎么连」。

```python
class AgentSelector(Protocol):        # 类别 agent_selector
    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]: ...

class TopologyGenerator(Protocol):    # 类别 topology_generator
    def build(self, agents: list[AgentSpec], query: TaskQuery) -> MASGraph: ...
```

- `templates.py`：`Role`（角色数据）、`TEAMS`（命名队伍 profile）、`team_to_agentspecs`、`StaticTopology`（注册 `topology_generator/static`：按 team 名产顺序链 MASGraph）。
- `selectors/`：`agent_selector/agentinit`——多样性×相关性 Pareto 选队；`mode="pool"`（离线候选池，零 GPU/API）与 `mode="generate"`（LLM 现场生成角色）。重库在 `select()` 内惰性导入。

**队伍模板约定**（与运行时耦合，改模板必须遵守）：末位角色以 `APPROVE: <final answer>` 收尾（运行时据此终止并抽答案）；每个 agent 绑定自身 role 的注入路径，共享 backend + RoutingContext；「上一个 agent 输出」的来源标志统一用 `PREV_OUTPUT_HEADER`（定义在 `memory/channels/nl.py`）。

## `prune/` — 剪枝层（经运行前插件挂载）

回答「哪些边/词表可裁掉以降本」。

```python
class GraphPruner(Protocol):          # 类别 graph_pruner
    def prune(self, graph: MASGraph, context: Any = None) -> MASGraph: ...

class VocabAdapter(Protocol):         # 类别 vocab_adapter
    def adapt(self, agent: Any, context: Any = None) -> Any: ...
```

- `pruners/agentprune.py`：**已实现** `graph_pruner/agentprune`——AgentPrune（ICLR 2025）时空掩码剪枝：逐边可训练 logit、伯努利采样实现图（`sample_realization`）、REINFORCE 更新（`reinforce`）、one-shot 剪枝（`update_masks`）、状态持久化（`save/load`）；`prune(graph)` 产 threshold 实现图（edges + `meta["agentprune"]` 矩阵）。纯标准库。复现实验：`scripts/run_agentprune_gsm8k.py`。
- `pruners/`：`agentdropout` / `agentdropout_v2`（桩，`NotImplementedError`）。
- `vocab/`：`agentvocab`（桩）。
- 挂载方式：`Orchestrator(pre_plugins=[("prune", {"pruner": "<name>", ...})])`（见 `plugins/README.md`）。

## `processing/` — 处理层（运行时组件）

回答「一个任务跑几次、多次结果如何归约」。

```python
Runner = Callable[[], Awaitable[Trajectory]]

class Processor(Protocol):            # 类别 processor
    async def run(self, runner: Runner) -> ProcessingResult: ...   # answer + trajectories

class TrajectoryAggregator(Protocol): # 类别 aggregator（parallel 的归约策略）
    def aggregate(self, trajectories: list[Trajectory]) -> Answer: ...
```

- `serial/`：`processor/serial`——跑 1 次 → 1 条轨迹 → 直接返回其 final_answer。
- `parallel/`：`processor/parallel`——并发 K 次（构造参数 `k`）→ 用 aggregator 聚合；`aggregator/self_consistency`（归一化内容多数投票，纯标准库，输入宽容 Trajectory/Answer/str）；`aggregator/dynamicagg`（桩）。

## 约定

import 本包触发三个子包全部注册，纯标准库、零重依赖。新增组件遵循 CLAUDE.md §4.1 六步配方。
