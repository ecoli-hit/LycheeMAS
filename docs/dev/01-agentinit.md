# 01 · AgentInit 开发文档（构建层）

> 论文：**AgentInit: Initializing LLM-based Multi-Agent Systems via Diversity and Expertise Orchestration for Effective and Efficient Collaboration**（EMNLP 2025 Findings, CCF-B）
> arXiv [2509.19236](https://arxiv.org/abs/2509.19236) · [ACL 2025.findings-emnlp.636](https://aclanthology.org/2025.findings-emnlp.636/) · 代码 [github.com/1737423697/AgentInit](https://github.com/1737423697/AgentInit)
> 目标：把 `agent_selector/agentinit` 从桩接成真实组件。
> 把握度 🟡（方法骨架已确证；diversity/expertise 精确公式与是否共优化拓扑待论文 PDF）。先读 [README.md](README.md) §3 共性事实。

---

## 1. 论文与方法核心

AgentInit 解决"如何**初始化/组建**一支高效 MAS 团队"：在候选 agent 池上做**多目标平衡选择**，兼顾团队**多样性**与任务**相关性/专长**，选出小而互补的团队（降冗余、降 token、保性能）。

### 1.1 已确证（来源见下）

- **选择策略 = Pareto 原则**：摘要原文"*Balanced team selection strategies using **Pareto principles*** 来 *jointly consider agent team diversity and task relevance*"。
- **算法 = NSGA-II**（Non-dominated Sorting Genetic Algorithm II，多目标进化），正文确证。
- **"Natural Language to Format" 机制**：保证（被选 agent 的）格式/一致性。
- **构建基础**：实现基于 GPTSwarm / AgentPrune / AgentDropout / AutoAgents。
- **效果**：相对 SOTA 与预定义策略整体性能 **1.2×、1.7×**，同时**显著降 token**；基准含 **MMLU**（代码 `experiments/run_mmlu.py`）。

> 来源：[arXiv abstract 2509.19236](https://arxiv.org/abs/2509.19236)、[ACL Anthology 636](https://aclanthology.org/2025.findings-emnlp.636/)、[官方代码 README](https://github.com/1737423697/AgentInit)。

### 1.2 `【待对照论文 PDF 核验】`

- diversity 与 expertise/relevance 的**精确数学定义**（如 diversity=语义嵌入两两相似度的反函数？expertise=任务域小样本基准分？）。
- 是否**同时优化拓扑**，还是**仅选成员**（拓扑交给下游）。
- **团队规模 k** 的控制（固定 / 自适应 / 上下界）。
- 完整 benchmark 列表与基线全名、精确数值。

> 写实现前回填本节：读 [arXiv PDF](https://arxiv.org/pdf/2509.19236) 方法节 + 核对 `github.com/1737423697/AgentInit` 里实现 NSGA-II/选择的源文件（README 未直接列出函数名）。

---

## 2. 在框架中的定位

- **层**：构建（`layers/construct/`）。
- **类别 / 注册名**：`agent_selector` / `agentinit`。
- **协议**：`construct/base.py::AgentSelector`。
- **当前桩**：`src/lychee_mas/layers/construct/selectors/__init__.py:14`（`AgentInitSelector.select` 抛 `NotImplementedError`）。

**定位判断**：AgentInit 只产出"**团队成员（AgentSpec 列表）**"；拓扑由现有 `TopologyGenerator`（`StaticTopology`）接管——这与框架已有的"selector→topology"分工天然契合，且 `StaticTopology.build(agents=...)` **已支持显式传入 agents**（`construct/templates.py:222`），接入成本低。

---

## 3. 接口函数（签名对齐）

`construct/base.py`（原样）：

```python
@runtime_checkable
class AgentSelector(Protocol):
    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]: ...
```

消费/产出类型（`core/types.py`）：

```python
@dataclass
class AgentSpec:
    id; name; role; system_prompt; model; tools: list[str]
    profile: dict[str, Any]   # ★ 装专长/多样性特征（专长向量、嵌入、领域分）
    meta: dict[str, Any]

@dataclass
class Budget: limit: float; unit: BudgetUnit   # tokens|calls|usd —— 控制团队规模/成本
```

约定：把每个候选 agent 的**专长向量/角色嵌入**放 `AgentSpec.profile`，diversity 用它两两算、expertise 用它对 `query` 算相关性；`budget` 映射到团队规模 k 或候选评估预算。

---

## 4. 写哪些代码 · 在哪里实现

| 动作 | 文件 | 说明 |
|---|---|---|
| **实现类** | `src/lychee_mas/layers/construct/selectors/agentinit.py`（新建） | `class AgentInitSelector(AgentSelector)`，`@REGISTRY.register("agent_selector","agentinit")` |
| 替桩 + 触发注册 | `src/lychee_mas/layers/construct/selectors/__init__.py` | 删桩，改为 `from .agentinit import AgentInitSelector` |
| **候选池** | `src/lychee_mas/layers/construct/templates.py` | 在固定 `TEAMS` 之外加更大的**角色/模型候选池** `CANDIDATE_ROLES`（带 `profile`/`description`）+ `candidate_pool(task)->list[AgentSpec]` |
| 多样性/专长度量 | 同 `agentinit.py` | `_diversity(specs)`、`_expertise(spec, query)`（嵌入惰性 import；离线可用确定性桩特征测） |
| NSGA-II 选择 | 同 `agentinit.py` | `_nsga2_select(cands, k, generations)`（纯 Python 实现非支配排序 + 拥挤度；无需重库） |
| 配置 | `configs/agents/agentinit.yaml`（新建） | 超参（见 §8） |
| 测试 | `tests/test_agentinit.py`（新建） | 离线 mock，测 Pareto/规模/确定性（见 §9） |
| 编排接入 | `src/lychee_mas/pipeline.py` | `Orchestrator(selector=None)`：`build_graph()` 内先 `agents=selector.select(query,budget)` 再 `StaticTopology.build(agents=agents)` |
| CLI | `scripts/run_experiment.py` | 加 `--selector`，透传给 `Orchestrator` |

> 嵌入模型（若 expertise/diversity 用语义嵌入）**惰性 import**；NSGA-II 用纯标准库实现，保证 `selectors/__init__.py` import 零重依赖（`make selfcheck` 仍 `NONE`）。离线测试用 `profile` 里的确定性向量，不需真嵌入。

---

## 5. 从官方仓库迁移映射

> 原则：借选择算法逻辑，agent 执行仍走 LycheeMAS。保留原始引用与许可证（黄金法则 8）。

| 官方仓库（`1737423697/AgentInit`） | → LycheeMAS 落点 |
|---|---|
| NSGA-II 多目标选择（具体文件待核验） | `AgentInitSelector._nsga2_select` |
| diversity / expertise 度量 | `_diversity` / `_expertise`（公式待 PDF 回填） |
| 候选生成 / 角色池 | `templates.py::candidate_pool` + `CANDIDATE_ROLES` |
| Natural-Language-to-Format 机制 | 选中 agent 的 `system_prompt` 规整（复用 `Role`/`PREV_OUTPUT_HEADER` 约定） |
| `experiments/run_mmlu.py` | 实验脚本 → `scripts/run_experiment.py --selector agentinit --benchmark ...` |

---

## 6. 编排接入（集成）

见 [README.md](README.md) §3.1。AgentInit 接在 topology **之前**：

```python
def build_graph(self):
    agents = None
    if self.selector:                                          # ← 新增可选 step
        agents = REGISTRY.create("agent_selector", self.selector).select(self.query, self.budget)
    topo = REGISTRY.create("topology_generator", "static", team=self.team, model=self.model, rounds=self.rounds)
    return topo.build(agents=agents)                           # StaticTopology.build 已支持显式 agents
```

CLI：`--selector agentinit`。不带 `--selector` 时行为与现在完全一致（走 team 模板）。

---

## 7. 开发流程（六步配方 + 里程碑）

- **M0（核验, ~0.5d）**：读 arXiv PDF 方法节 + 官方源码，回填 §1.2（diversity/expertise 公式、是否定拓扑、规模 k）。
- **M1（候选池, ~1d）**：`templates.py` 加 `CANDIDATE_ROLES` + `candidate_pool(task)`，每角色带 `profile`。
- **M2（度量 + NSGA-II, ~2–3d）**：`_diversity`/`_expertise` + 纯 Python NSGA-II；`AgentInitSelector.select` 串起来；替桩 + 注册 + config + test。
- **M3（接入 + 实验, ~1.5–2d）**：`Orchestrator(selector=)` + CLI；消融（见 §10），对齐论文 1.2×/1.7× 与 token 降幅。

六步对齐：①读 `construct/base.py`（已确认）→②写 `agentinit.py`+注册 →③`selectors/__init__.py` 触发 →④`configs/agents/agentinit.yaml` →⑤`tests/test_agentinit.py` →⑥`make lint/test/selfcheck` + `make demo`。

---

## 8. 配置 `configs/agents/agentinit.yaml`

```yaml
# agent_selector/agentinit 超参
team_size: 5              # 目标团队规模 k（待核验：固定 or 自适应）
generations: 30          # NSGA-II 迭代代数
pop_size: 50             # 候选池/种群大小
objectives: [diversity, expertise]   # 两目标
diversity_metric: embedding_cosine   # 待 PDF 核验
expertise_metric: query_relevance    # 待 PDF 核验
seed: 0                  # 可复现
```

---

## 9. 测试（离线、零重依赖）

`tests/test_agentinit.py`（用 `AgentSpec.profile` 里的确定性向量，不调真嵌入）：

- 规模约束：`select()` 返回恰好 `team_size` 个 `AgentSpec`。
- Pareto 正确性：构造已知支配关系的候选，断言被选集为非支配解（高 expertise + 高 diversity 的组合优先于被支配组合）。
- 多样性生效：相同专长的重复候选不会被同时选满（diversity 目标拉开）。
- 确定性：固定 `seed` 两次 `select()` 结果一致。
- 接入不回归：`Orchestrator(selector="agentinit")` 在 mock runtime 产出 `Trajectory`；不带 selector 时与 team 模板一致。

---

## 10. 实验与消融

| 配置 | 变量 | 看什么 |
|---|---|---|
| `--team default`（无 selector） | 预定义团队 | 基线 acc / token |
| `--selector agentinit` | AgentInit 选队 | acc↑（对齐 1.2×/1.7×）、token↓ |
| 随机选队 baseline | 随机 k 个 | 验证 AgentInit 优于随机 |
| 扫 `team_size∈{3,5,7}` | 规模 | 规模-性能-成本权衡 |
| 关 diversity / 关 expertise | 单目标 | 消融两目标各自贡献 |

**指标**：accuracy、token、团队规模、（可选）多样性分。基准对齐论文（含 MMLU）。落 `runs/<model-tag>/agentinit/<task>/`。

---

## 11. 验收标准

- `make lint && make test && make selfcheck` 全绿（`HEAVY LOADED: NONE`）。
- 离线 `scripts/run_experiment.py --runtime mock --selector agentinit --questions "..."` 跑通。
- 条件允许：`--runtime autogen --benchmark mmlu --n 5 --selector agentinit` 小样本 acc 不低于预定义团队、token 更省。
- 不带 `--selector` 时行为与改前完全一致。

---

## 12. 预期时间 + 风险依赖

- **预期时间**：5–8 人天（M0 0.5d + M1 1d + M2 2–3d + M3 1.5–2d）。
- **依赖**：候选池定义（`templates.py`）；可选嵌入模型（diversity/expertise 用语义相似度时）。**不依赖** MASGraph 邻接前置。
- **风险**：① diversity/expertise 精确定义未确证（M0 必须先回填，否则实现可能偏离论文）；② 若论文实际**同时优化拓扑**，则需把选择产物也写入 `MASGraph.edges`，接入点要扩展（当前假设仅选成员）；③ 候选池质量直接决定上限，需覆盖任务所需角色谱。
