# LycheeMAS 组内论文开发文档（`docs/dev/`）

> 本目录为**本组（刘学博团队）论文 → LycheeMAS 组件**的逐篇落地开发文档。每篇论文一份，回答五件事：
> **开发流程 / 写哪些代码 / 在哪里实现 / 接口函数 / 预期时间**。
>
> 配合根目录 `CLAUDE.md`（黄金法则/工程约束）与 `docs/DEVELOPMENT.md`（架构/六步配方/CDM 数据流）阅读。

---

## 1. 本批文档清单（第一批：4 篇）

| 文档 | 论文 | 会议 | 层 · 类别/注册名 | 状态 / 桩位置 | 把握度 |
|---|---|---|---|---|---|
| [01-agentinit.md](01-agentinit.md) | **AgentInit** — Diversity & Expertise Orchestration | EMNLP'25 Findings (CCF-B) | 构建 · `agent_selector/agentinit` | ✅ 已实现（`selectors/agentinit.py`） | ✅ |
| [02-agentdropout.md](02-agentdropout.md) | **AgentDropout** — Dynamic Agent Elimination | ACL'25 (CCF-A) | 剪枝 · `graph_pruner/agentdropout` | `layers/prune/pruners/__init__.py:27` | 🟢 |
| [03-agentvocab.md](03-agentvocab.md) | **AgentVocab** — Structure-Aware Vocabulary Adaptation | ICML'26 (CCF-A) | 剪枝 · `vocab_adapter/agentvocab` | `layers/prune/vocab/__init__.py:13` | 🔴 |
| [04-maspo.md](04-maspo.md) | **MASPO** — Joint Prompt Optimization | ICML'26 (CCF-A) | 归因训练 · `trainer/maspo` | `train/__init__.py:63` | 🟢 |

> 把握度：✅ 已实现落地 / 🟢 据官方仓库 + arXiv 写实 / 🟡 arXiv 在但部分算法细节待核验 / 🔴 公开信息有限、需对照论文 PDF 回填。
>
> **后续批次（暂缓）**：AgentPrune（`graph_pruner/agentprune`，基线）、AgentDropout v2（`graph_pruner/agentdropout_v2`，运行时在线淘汰）。

### 论文与代码链接

| 论文 | arXiv / 主页 | 官方代码 |
|---|---|---|
| AgentInit | [arXiv 2509.19236](https://arxiv.org/abs/2509.19236) · [ACL 2025.findings-emnlp.636](https://aclanthology.org/2025.findings-emnlp.636/) | [github.com/1737423697/AgentInit](https://github.com/1737423697/AgentInit) |
| AgentDropout | [arXiv 2503.18891](https://arxiv.org/abs/2503.18891) · [ACL 2025.acl-long.1170](https://aclanthology.org/2025.acl-long.1170/) | [github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout) |
| AgentVocab | ICML'26 [poster 61748](https://icml.cc/virtual/2026/poster/61748)（暂无公开 arXiv） | anonymous.4open.science/r/AgentVocab-28CC（匿名，待核验） |
| MASPO | [arXiv 2605.06623](https://arxiv.org/abs/2605.06623) · ICML'26 [poster 62219](https://icml.cc/virtual/2026/poster/62219) | [github.com/wangzx1219/MASPO](https://github.com/wangzx1219/MASPO) |

> 论文 PDF 与链接清单放 `docs/dev/refs/`（大文件已被 `.gitignore` 忽略）。把 PDF 丢进去后，可据此把 🟡/🔴 的"方法核心"回填收敛成 🟢。

---

## 2. 统一文档模板（每篇 12 节）

每份开发文档按下列固定结构组织，确保覆盖用户点名的 5 件事且风格统一：

1. **论文与方法核心** — 论文/会议/链接；算法核心 + 关键步骤 + 主要超参 + 论文报告的指标。未确证处用 `【待对照论文核验】` 显式留口。
2. **在框架中的定位** — 层、REGISTRY 类别/注册名、要实现的 `base.py` 协议、当前桩文件行号。
3. **接口函数（签名对齐）** — 要实现的协议方法**精确签名** + 消费/产出类型（`core/types.py` + 该层契约）。
4. **写哪些代码 · 在哪里实现** — 新建/改动文件清单（实现模块、`__init__` 触发注册、config、test、接入文件）。
5. **从官方仓库迁移映射** — 官方仓库函数/类 → LycheeMAS 接口方法的对照表（**借逻辑、不整包搬 MAS 类**）。
6. **编排接入（集成）** — 端到端如何被调用；当前 `Orchestrator` 的缺口与改动点。
7. **开发流程（六步配方 + 里程碑）** — 按 `DEVELOPMENT.md §5` 六步展开，拆 M1…Mn。
8. **配置** — `configs/<类别>/<name>.yaml` 示例。
9. **测试** — 离线 mock 测试点 + 断言（零重依赖；需 GPU/`[all]` 的单独标注）。
10. **实验与消融** — 消融矩阵（只换一个变量）、指标（accuracy/token/latency + 论文同款指标）、要跑的 config、`runs/` 落盘。
11. **验收标准** — `make lint && make test && make selfcheck` 全绿；离线 `--questions` 自检；条件允许 `--n 5` 回归。
12. **预期时间 + 风险依赖** — 人天估算（1 名熟悉本仓库的研究者）+ 里程碑；重依赖/GPU/阻塞关系。

---

## 3. 跨文档共性事实（动手前必读）

以下是探查 `src/lychee_mas/pipeline.py`、`scripts/run_experiment.py` 与各层得到的硬事实，直接决定每篇的"集成"章节。**集中在此说明一次，各文档引用本节。**

### 3.1 Orchestrator 编排缺口（关键）

`src/lychee_mas/pipeline.py` 的 `Orchestrator.run()` 当前只做三件事：

```
build_graph()                       # 只调 topology_generator/static（按 team 名产顺序链）
  → REGISTRY.create("runtime", …)   # mock / autogen
  → runtime.run(graph, query)       # 产出 Trajectory
  → (可选) aggregator.aggregate([trajectory])
```

它**不调用** `agent_selector`（选择）、`graph_pruner`（剪枝）、`vocab_adapter`（词表）、`attributor/credit_assigner/trainer`（归因训练）。
→ **每篇文档都包含"把组件接进编排/运行时"的改动**：在 `Orchestrator` 增加一个**可选 step**（按名字从 REGISTRY 取，给 `None` 则跳过），**不硬编码任何实现**（黄金法则 3）。

接入位置约定：
- **selector**：在 `build_graph()` 内，topology 之前——`agents = selector.select(query, budget)` → `StaticTopology.build(agents=agents)`（`StaticTopology.build` **已支持显式传入 agents**，见 `construct/templates.py:222`）。
- **graph_pruner**：在 `build_graph()` 出图之后——`graph = pruner.prune(graph, ctx)`。
- **vocab_adapter**：模型级，在 runtime 后端装配 agent/model 时应用（`hf_backend` / `model_client`），不在 `build_graph`。
- **trainer**：是离线训练闭环，不进 `run()`，走独立驱动脚本（见 04-maspo）。

### 3.2 CLI 缺开关

`scripts/run_experiment.py` 现有参数：`--runtime --team --aggregator --benchmark --questions --n --rounds --seed --model-tag --results-root --no-save`。
**没有** `--selector / --pruner / --vocab`。要让组件"由 CLI 选择"（黄金法则 3），需新增对应 flag 并透传给 `Orchestrator(...)`。

### 3.3 MASGraph 邻接前置（公共依赖）

`StaticTopology.build()` 产出的是"**顺序链**"，`runtime/base.py` 的 `MASGraph` 节点齐全但**边/邻接目前留空**（`meta={"team": …}`，无邻接矩阵）。
**AgentDropout 必须作用在真实邻接/边权上** → 存在 公共前置：

- 给 `MASGraph` 补 `adjacency`（初始全连接或角色图）+ 边权容器；
- 让 runtime 按邻接路由消息（哪个 agent 能看到哪个 agent 的输出）。

**此前置在 [02-agentdropout.md](02-agentdropout.md) 中落地**，后续 AgentPrune / AgentDropout v2 复用。AgentInit（只选成员）与 AgentVocab（模型级）**不依赖**此前置。

### 3.4 可复用锚点（别重造轮子）

- **实现模板**：`aggregator/self_consistency`（`layers/processing/parallel/__init__.py`）——纯标准库、注册干净、有 `_norm/_as_answers` 辅助，是"一个可跑组件"的范本。
- **拓扑**：`StaticTopology.build(agents=...)`（`construct/templates.py:222`）已支持显式 agents；复用 `team_to_agentspecs()`、`TEAMS`、`Role`。
- **类型**：`MASGraph(nodes, rounds, meta)`（`runtime/base.py`）、`AgentSpec.profile`（装专长/多样性特征）、`Trajectory`/`Message`/`Answer`（`core/types.py`）。
- **采轨迹/评分**：`trace/TraceStore`（消息级落点 + 决策日志，供 MASPO 采样）、`eval/metrics.py::score`（供 MASPO reward / 实验评分）。

### 3.5 惰性导入铁律（黄金法则 2）

注册组件的模块被 import 时**不得触发** `torch / transformers / litellm / autogen_*`——重依赖只放方法内部。
- AgentDropout、AgentVocab 用 `torch` → 放 `prune(...)` / `adapt(...)` 内部惰性 import。
- MASPO 调 LLM（litellm/openai/dspy）+ 数据集 → 重依赖进 optional extra `[train]`，仅在 `trainer/maspo` 实现与训练脚本里 import。
- 自检：`make selfcheck` 必须仍输出 `HEAVY LOADED: NONE`。

### 3.6 六步配方（DEVELOPMENT.md §5）

新增/接桩一个组件 = ①读 `base.py` 接口 → ②写实现类 + `@REGISTRY.register(category, name)` → ③子包 `__init__.py` import 触发注册 → ④加 `configs/<类别>/<name>.yaml` → ⑤加 `tests/test_<name>.py` → ⑥`make lint && make test && make selfcheck` + `make demo` 验证不回归。
**绝不为此改 `pipeline.py` 的"按名取组件"机制**；接入新类别 = 加一个可选 step（见 3.1）。

---

## 4. 路线图与依赖

```
AgentInit ───────────────► 选成员 → StaticTopology.build(agents)
                                         │
AgentDropout ──(需 MASGraph 邻接前置)──► 剪稀疏拓扑（评测前离线优化）
AgentVocab ──(模型级，改 hf_backend 生成)──► 每 agent 词表降本
                                         │
MASPO ──(独立训练驱动: 采样→评分→三维信用→改 prompt→反哺)──► 暖启动各 agent system_prompt
```

**建议落地顺序**：AgentDropout（落地 邻接前置，确证最足）→ MASPO → AgentInit → AgentVocab。

### 工程量汇总（实现该方法的估算，非写文档）

| 论文 | 把握度 | 人天 | 关键依赖 |
|---|---|---|---|
| AgentDropout（+邻接前置） | 🟢 | 8–12 | MASGraph 邻接；torch |
| MASPO | 🟢 | 10–15 | 闭环；LLM 调用；`[train]` extra |
| AgentInit | ✅ 已落地 | — | 候选池 + Pareto/Vendi（pool + generate） |
| AgentVocab | 🔴 | 8–12 | hf_backend 生成期；公开信息缺口 |
