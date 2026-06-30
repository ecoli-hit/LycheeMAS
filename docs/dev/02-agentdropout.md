# 02 · AgentDropout 开发文档（L2 图剪枝）

> 论文：**AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration**（ACL 2025, CCF-A）
> arXiv [2503.18891](https://arxiv.org/abs/2503.18891) · [ACL 2025.acl-long.1170](https://aclanthology.org/2025.acl-long.1170/) · 代码 [github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout)
> 目标：把 `graph_pruner/agentdropout` 从桩接成真实组件。**本篇同时落地 L2 公共前置「MASGraph 邻接 + 边权」**，后续 AgentPrune / AgentDropout v2 复用。
> 把握度 🟢（官方仓库 + arXiv 确证）。先读 [README.md](README.md) §3 共性事实。

---

## 1. 论文与方法核心

AgentDropout 把多智能体协作建成一张可学习的通信图，通过**梯度优化 + 硬剪枝**学出稀疏拓扑，在保持/略升精度的同时大幅降 token。

### 1.1 图表示：spatial + temporal 双层

- **Spatial（轮内）**：当前轮次内 agent 间的有向通信边。节点=agents(+1 个 decision/汇总节点)，边权是**可学习 logits** `spatial_logits`（`torch.nn.Parameter`）。
- **Temporal（跨轮）**：相邻轮次间的信息流，边权 `temporal_logits`（第 0 轮无 temporal 边）。
- 边激活概率：`edge_prob = sigmoid(logit / temperature)`；logits 初始化 `init_logit = log(p/(1-p))`（`initial_spatial/temporal_probability=0.5`）。
- 三种模式：`diff=False` 全局共享一套 logits；`diff=True` 每轮独立（`ParameterList[rounds]`，逐轮动态）；`dec=True` 额外一套 `*_logits_1` 专做 node dropout 决策。

### 1.2 两层淘汰

- **Edge dropout（`update_masks()`）**：跨所有轮次按 logit 绝对值统计，全局剪掉最小的一批边（按 `pruning_rate` 比例），把对应 mask 置 0（硬约束）。
- **Node dropout（`update_masks_dec()`，仅 `dec=True`）**：逐轮算每个节点重要度 = 入度边 logit 和 + 出度边 logit 和，淘汰最低分节点，记入 `skip_nodes`，运行时整体跳过该 agent —— 这就是"dynamic agent elimination"。

### 1.3 优化目标（两阶段 Adam）

```
# 任务损失（边对正确率的贡献，utility = 1/0 是否答对）
loss_s = -log_prob_spatial · utility ;  loss_t = -log_prob_temporal · utility
# Frobenius 稀疏正则（带 delta 松弛阈值：只在超阈时惩罚）
total = loss_s + loss_t + relu(‖W_s·mask_s‖_F − delta) + relu(‖W_t·mask_t‖_F − delta)
```

- **Phase-1（可选, `dec`）**：Adam 优化 `*_logits_1`，循环里 `update_masks_dec()` 做 node dropout。
- **Phase-2**：Adam 优化 `spatial_logits/temporal_logits`，每 `imp_per_iterations` 步做一次 `update_masks()` 全局 edge 剪枝。

### 1.4 关键超参与论文指标

| 超参 | 默认 | 含义 |
|---|---|---|
| `pruning_rate` | 0.25 | 每次 mask 更新剪掉的边比例 |
| `lr` | 0.1 | Adam 学习率 |
| `num_iterations` | 10 | 优化总步数 |
| `imp_per_iterations` | 5 | 每多少步做一次 mask 更新 |
| `delta` | 0.1 | Frobenius 松弛阈值 |
| `temperature` | 1.0 | sigmoid 缩放（推理可调速度/精度权衡） |
| `rounds` | 1+ | 多轮讨论轮数 |

**论文报告**：prompt token −21.6%、completion token −18.4%、accuracy +1.14；跨域/跨模型迁移性强；对初始拓扑选择不敏感。

> 函数/超参名据官方仓库调研；**写实现前以 [官方最新代码](https://github.com/wangzx1219/AgentDropout) 再核对一次**（`graph.py` 体量约 31KB）。

---

## 2. 在框架中的定位

- **层**：L2 剪枝（`layers/prune/`）。
- **类别 / 注册名**：`graph_pruner` / `agentdropout`。
- **协议**：`prune/base.py::GraphPruner`。
- **当前桩**：`src/lychee_mas/layers/prune/pruners/__init__.py:27`（`AgentDropout(_StubPruner)`，`prune()` 抛 `NotImplementedError`）。

**关键定位判断**：AgentDropout 的 `prune` 本质是**离线拓扑优化**（在少量训练 query 上梯度学 mask），不是一次性纯函数。框架里定位成**"评测前的离线 topology optimizer"**：先在 train split 上学出稀疏的 `MASGraph`，再把这张固定拓扑用于 eval。两档实现路线：

- **路线 A（忠实端口）**：torch logits + 两阶段 Adam + Frobenius 正则。需要一个 utility 信号（答对=1）与"按当前 mask 跑 MAS 得 utility"的可微/采样代理。最贴论文，但工作量大。
- **路线 B（启发式首版，先打通）**：跳过梯度，用现成重要度（如边的历史使用频次/消息 token 贡献）直接按 `pruning_rate` 剪边、按节点度数淘汰节点。一天可出可跑版本，作为 A 的对照基线。

建议：**先 B 打通端到端 + 接口 + 实验脚手架，再上 A 复现论文数值**。

---

## 3. 接口函数（签名对齐）

实现 `prune/base.py::GraphPruner` 协议（原样）：

```python
@runtime_checkable
class GraphPruner(Protocol):
    def prune(self, graph: Any, context: Any = None) -> Any: ...   # 输入 MASGraph + 上下文 -> 剪枝后 MASGraph
```

消费/产出类型（`runtime/base.py::MASGraph`，**邻接容器已存在**）：

```python
@dataclass
class MASGraph:
    nodes: list[AgentSpec]              # 节点
    edges: dict[str, list[str]]        # 邻接表 role -> 下游 roles（当前默认空=顺序链）★
    rounds: int = 1
    meta: dict = field(default_factory=dict)
```

`context` 约定（dict，新增，可选）：`{"runtime", "queries"(train split), "score_fn", "mode": "edge"|"node"|"both", "rounds"}`——离线优化需要它来"跑 MAS 取 utility"。

---

## 4. 写哪些代码 · 在哪里实现

| 动作 | 文件 | 说明 |
|---|---|---|
| **实现类** | `src/lychee_mas/layers/prune/pruners/agentdropout.py`（新建） | `class AgentDropout(GraphPruner)`，`@REGISTRY.register("graph_pruner","agentdropout")`；torch 惰性导入 |
| 替桩 + 触发注册 | `src/lychee_mas/layers/prune/pruners/__init__.py` | 删 `agentdropout` 桩，改为 `from .agentdropout import AgentDropout`；保留 `agentprune`/`agentdropout_v2` 桩 |
| **L2 公共前置** | `src/lychee_mas/runtime/base.py` | `MASGraph` 加 `weights: dict[str,dict[str,float]]`（边权 W）+ `apply_mask(mask)`/`sparsity()` 辅助；不破坏现有顺序链默认 |
| 初始邻接 | `src/lychee_mas/layers/construct/templates.py` | `StaticTopology` 增 `topology="chain"\|"full"\|"star"` 选项，填充 `edges`（首版给 `full` 全连接，供剪枝有东西可剪） |
| runtime 按边路由 | `src/lychee_mas/runtime/backends/mock_runtime.py`（+ autogen 后端） | 让"某 agent 能看到哪些上游输出"由 `graph.edges` 决定（无边时退回顺序链，保持现有行为） |
| 配置 | `configs/pruner/agentdropout.yaml`（新建） | 超参（见 §8） |
| 测试 | `tests/test_agentdropout.py`（新建） | 离线 mock，测启发式首版（见 §9） |
| 编排接入 | `src/lychee_mas/pipeline.py` | `Orchestrator(pruner=None)`：`build_graph()` 出图后 `if pruner: graph = REGISTRY.create("graph_pruner",pruner).prune(graph, ctx)` |
| CLI | `scripts/run_experiment.py` | 加 `--pruner`，透传给 `Orchestrator` |

> 重依赖（torch/numpy）**只在 `agentdropout.py` 的方法内部惰性 import**；`pruners/__init__.py` 被 import 时不得触发 torch（`make selfcheck` 须仍 `HEAVY LOADED: NONE`）。

---

## 5. 从官方仓库迁移映射

> 原则：**借算法逻辑，不整包搬 MAS 类**。官方 `Graph/Node` 重写到 LycheeMAS 的 `MASGraph` + `Runtime` 之上。保留原始引用与许可证（黄金法则 8）。

| 官方仓库（`wangzx1219/AgentDropout`） | → LycheeMAS 落点 |
|---|---|
| `graph.py::Graph.__init__`（logits/masks 初始化） | `AgentDropout.__init__` + 在 `prune()` 内按 `graph.edges` 构造 logits 张量 |
| `graph.py::Graph.update_masks()`（全局 edge 剪枝） | `AgentDropout._prune_edges(logits, rate)` → 写回 `graph.edges`/`graph.weights` |
| `graph.py::Graph.update_masks_dec()`（node 淘汰） | `AgentDropout._drop_nodes(...)` → 从 `graph.nodes`/`edges` 移除 `skip_nodes` |
| `graph.py::construct_spatial/temporal_connection()` | runtime 按 `graph.edges` 采样/路由（mock_runtime 改造） |
| `graph.py::spatial_adj_matrix`（属性） | 由 `graph.edges` ↔ numpy 邻接矩阵的互转工具 `_to_adj/_from_adj` |
| `node.py::Node`（spatial/temporal 前后继 + memory） | 复用现有 `AgentSpec` + `Trajectory`（已有 messages/round），不另造 Node |
| `experiments/run_gsm8k.py`（两阶段优化主循环） | 路线 A 的 `_optimize(graph, ctx)`（Adam 两阶段）；命令行参数 → `configs/pruner/agentdropout.yaml` |

---

## 6. 编排接入（集成）

见 [README.md](README.md) §3.1/§3.3。AgentDropout 依赖 **L2 邻接前置**（本篇落地）：

1. `StaticTopology.build(topology="full")` 产出带 `edges` 的稠密 `MASGraph`。
2. `Orchestrator.run()` 在 `build_graph()` 之后插可选剪枝：

```python
def __init__(self, ..., pruner: Optional[str] = None, pruner_ctx: Optional[dict] = None): ...
async def run(self, query, hook=None):
    graph = self.build_graph()
    if self.pruner:                                   # ← 新增可选 step（None 则跳过）
        graph = REGISTRY.create("graph_pruner", self.pruner).prune(graph, self.pruner_ctx)
    runtime = REGISTRY.create("runtime", self.runtime_name, **self.runtime_kwargs)
    ...
```

3. CLI：`--pruner agentdropout`。
4. runtime 按 `graph.edges` 决定可见性（剪掉的边/节点不参与），从而真正省 token。

> 离线优化（路线 A）通常**不在每次 `run()` 里跑**，而是单独"学拓扑"阶段产出固定稀疏图（可缓存到 `graph.meta` 或 `runs/`），eval 时加载。`prune()` 在 `context` 含 train split 时执行优化，否则按已学 mask 应用。

---

## 7. 开发流程（六步配方 + 里程碑）

- **M1（前置, ~2d）**：`MASGraph` 加 `weights` + `apply_mask/sparsity`；`StaticTopology` 加 `topology` 选项填 `edges`；mock_runtime 按 `edges` 路由；加 `tests/test_masgraph_topology.py`（顺序链行为不回归 + full 邻接正确）。
- **M2（启发式首版, ~2d）**：`agentdropout.py` 路线 B（频次/度数重要度 → 按 `pruning_rate` 剪边 + 淘汰节点）；替桩 + 注册 + config + test；`Orchestrator(pruner=)` + CLI。`make lint/test/selfcheck` 全绿。
- **M3（忠实端口, ~3–5d）**：路线 A torch logits + 两阶段 Adam + Frobenius 正则；`context` 接 train split + score_fn；实现 `spatial/temporal/diff/dec` 开关。
- **M4（实验, ~2d）**：消融矩阵（见 §10），复现论文 token 降幅与精度；落 `runs/`。

六步对齐：①读 `prune/base.py`（已确认）→②写 `agentdropout.py`+注册 →③`pruners/__init__.py` 触发 →④`configs/pruner/agentdropout.yaml` →⑤`tests/test_agentdropout.py` →⑥`make lint/test/selfcheck` + `make demo`。

---

## 8. 配置 `configs/pruner/agentdropout.yaml`

```yaml
# L2 graph_pruner/agentdropout 超参（对齐论文默认）
mode: edge            # edge | node | both（edge=update_masks；node=update_masks_dec）
pruning_rate: 0.25    # 每次剪边比例
init_topology: full   # 优化起点拓扑：full | star | chain
# 路线 A（忠实端口）才用：
optimize: false       # false=按已学/启发式 mask 应用；true=在 train split 上梯度优化
lr: 0.1
num_iterations: 10
imp_per_iterations: 5
delta: 0.1
temperature: 1.0
diff: false           # 每轮独立 logits
dec: false            # 启用 node dropout 决策套
rounds: 1
```

---

## 9. 测试（离线、零重依赖）

`tests/test_agentdropout.py` 聚焦**启发式首版 + 图操作**（不引 torch）：

- `MASGraph` 邻接：`StaticTopology.build(topology="full")` 边数 = N(N-1)；`topology="chain"` 退回顺序链。
- `apply_mask`：给定 mask，剪后 `sparsity()` 与剩余边数正确；被剪边不在 `edges`。
- 启发式 `prune`：构造带 `weights` 的 `MASGraph`，`pruning_rate=0.5` 后边数减半，且保留高权重边、去掉低权重边。
- node dropout：`mode="node"` 后最低度节点从 `nodes` 移除，且其入/出边一并清除。
- 端到端不回归：`Orchestrator(pruner="agentdropout")` 在 mock runtime 上能产出 `Trajectory`，token 不高于不剪枝版本。
- **路线 A（torch）测试**单独标注"需 `[all]`/GPU"，不进默认 CI（默认 `pytest` 仍零重依赖）。

---

## 10. 实验与消融

**消融矩阵**（只换一个变量，复用 `run_experiment.py`）：

| 配置 | 变量 | 看什么 |
|---|---|---|
| `--team aime`（无 pruner） | baseline 全连接/顺序 | 基线 acc / token |
| `--pruner agentdropout`（mode=edge） | edge dropout | token↓、acc 持平 |
| `--pruner agentdropout`（mode=node） | node dropout | agent 数↓、token↓ |
| `--pruner agentdropout`（mode=both） | 组合 | 论文主结果 |
| 扫 `pruning_rate ∈ {0.1,0.25,0.5}` | 剪枝强度 | token-acc 权衡曲线 |

**指标**（`eval/metrics.py` + `Trajectory.total_tokens`）：accuracy、prompt/completion token 降幅（对齐论文 −21.6% / −18.4%）、latency、稀疏度（剩余边数/总边数）。基准：GSM8K / AQuA / MMLU / HumanEval（对齐论文）。结果落 `runs/<model-tag>/agentdropout/<task>/`。

---

## 11. 验收标准

- `make lint && make test && make selfcheck` 全绿（`HEAVY LOADED: NONE`）。
- 离线 `PYTHONPATH=src python scripts/run_experiment.py --runtime mock --team default --pruner agentdropout --questions "2 plus 2 is 4"` 能跑通、不报错。
- 条件允许：`--runtime autogen --benchmark gsm8k --n 5 --pruner agentdropout` 小样本回归，token 较 baseline 下降且 acc 不掉。
- 不破坏现有行为：不带 `--pruner` 时数值与改前完全一致（顺序链默认未变）。

---

## 12. 预期时间 + 风险依赖

- **预期时间**：8–12 人天（M1 前置 2d + M2 启发式 2d + M3 忠实端口 3–5d + M4 实验 2d）。仅要"可跑 + 接口 + 启发式"则 ~4–5 人天。
- **依赖**：L2 MASGraph 邻接前置（本篇 M1，后续 AgentPrune/v2 复用）；torch（路线 A）。
- **风险**：① 路线 A 需要"按 mask 跑 MAS 得 utility"的可微/采样代理，与论文一致性需对齐官方 `run_gsm8k.py`；② runtime 按 `edges` 路由要兼容现有顺序链默认（无边时行为不变），务必加回归测试；③ token 统计口径要与论文一致（prompt vs completion 分别报）。
