# 03 · AgentVocab 开发文档（L2 词表降本）

> 论文：**AgentVocab: Structure-Aware Vocabulary Adaptation for Efficient LLM Agents**（ICML 2026, CCF-A）
> ICML [poster 61748](https://icml.cc/virtual/2026/poster/61748)（暂无公开 arXiv） · 代码（匿名，计划发布）：anonymous.4open.science/r/AgentVocab-28CC
> 目标：把 `vocab_adapter/agentvocab` 从桩接成真实组件。
> 把握度 🔴（公开信息有限：仅 poster 摘要可读，匿名代码 403 不可访问，无预印本）。方法骨架已确证，**算法细节待论文 PDF**。先读 [README.md](README.md) §3 共性事实。

---

## 1. 论文与方法核心

AgentVocab 针对"通用 tokenizer 对 agent 的**结构化工具调用**分词低效"的问题，做**结构感知的词表适配**，降低每步生成的 token 数/解码延迟。

### 1.1 已确证（来源：[ICML poster 61748](https://icml.cc/virtual/2026/poster/61748)）

- **structure 指什么**：tool-calling 里的**重复结构模式与高频语义单元**（function call 模式等）。原文："*their usage is dominated by **narrow, structured tool-calling interactions***"。
- **方法类型 = 改词表**（不是 logit 掩码、不是单纯 embedding 微调）：原文"*adapts the model **vocabulary** to better reflect structural and semantic regularities*"，做法是"*derives specialized vocabulary entries **from real tool-calling traces***"。
- **与微调正交、无需重训**：原文"*without task-specific schema engineering*" 且"*is **orthogonal to existing fine-tuning and agent-training methods***"。→ 框架里定位成"**即插即用的模型级词表适配**"，不引入训练阶段。
- **效果**：解码延迟 **−15~25%**，同时**保持工具调用性能**。
- **基准**：**τ-bench、τ²-bench**。
- 作者：Kai Bian, Haosi Mo, Xuebo Liu, Shuangyong Song, Jing Li, Yongxiang Li, Min Zhang, Xuelong Li。

### 1.2 `【待对照论文 PDF 核验】`

- 词表条目的**选择目标函数**（频率 × token 节省？信息量？阈值？）。
- 新增 token 的**初始化方案**（由构成子词 embedding 平均/合并而来？既然"正交于微调"，应是**无需训练的初始化**——待确证）。
- 生成时如何生效（仅 tokenizer 合并规则即可，还是需在 embedding/lm_head 同步扩展）。
- τ-bench/τ²-bench 全称、基线模型、其他数值。
- 匿名仓库 `4open.science/r/AgentVocab-28CC` 目录与关键文件（**当前 403 不可访问**，发布后核验）。

> 这些缺口**不阻塞框架侧接入**（接口/落点/集成/测试/时间可写实），仅影响"算法核心"的具体公式。论文 PDF/代码可访问后回填，把 🔴 收敛为 🟢。

---

## 2. 在框架中的定位

- **层**：L2 剪枝（模型级降本，`layers/prune/`）。
- **类别 / 注册名**：`vocab_adapter` / `agentvocab`。
- **协议**：`prune/base.py::VocabAdapter`。
- **当前桩**：`src/lychee_mas/layers/prune/vocab/__init__.py:13`（`AgentVocab.adapt` 抛 `NotImplementedError`）。

**定位判断**：这是四篇里**最深的集成**——不是改图，而是改"**某 agent 的模型如何编码/解码**"。拆两段：

- **(a) 离线算词表**（零重依赖可测）：从 tool-calling 轨迹挖高频结构片段 → 产出"新增词条 + 映射"（一个纯数据产物）。
- **(b) 生成期生效**（torch/transformers，惰性）：让 runtime 后端（`hf_backend`）在该 agent 上使用扩展后的 tokenizer（+ 同步扩 embedding/lm_head 行，按 §1.2 待证的初始化）。

---

## 3. 接口函数（签名对齐）

`prune/base.py`（原样）：

```python
@runtime_checkable
class VocabAdapter(Protocol):
    def adapt(self, agent: Any, context: Any = None) -> Any: ...   # 对某 agent/模型产出词表子集/映射
```

约定：
- `agent`：`AgentSpec`（含 `model`、`role`、`profile`）。
- `context`：`{"traces": list[Trajectory]|tool-call 轨迹, "tokenizer": <可选>, "top_k": int}`——挖词条的数据源。
- 返回 `VocabPlan`（本层新增的小 dataclass，纯数据，零重依赖）：

```python
@dataclass
class VocabPlan:
    new_tokens: list[str]            # 新增词条（高频结构片段）
    init_map: dict[str, list[int]]   # 每个新 token -> 其构成子词 id（供 embedding 初始化）
    stats: dict                      # 频率/预计 token 节省等
```

> `VocabPlan` 是离线产物；生成期由 backend 消费它装配 tokenizer/embedding。这样 (a) 可在零重依赖下单测。

---

## 4. 写哪些代码 · 在哪里实现

| 动作 | 文件 | 说明 |
|---|---|---|
| **数据契约** | `src/lychee_mas/layers/prune/vocab/types.py`（新建） | `VocabPlan`（纯 dataclass） |
| **实现类** | `src/lychee_mas/layers/prune/vocab/agentvocab.py`（新建） | `class AgentVocab(VocabAdapter)`，`@REGISTRY.register("vocab_adapter","agentvocab")`；离线挖词条逻辑（标准库/可选惰性 transformers 只为真分词） |
| 替桩 + 触发注册 | `src/lychee_mas/layers/prune/vocab/__init__.py` | 删桩，改为 `from .agentvocab import AgentVocab` |
| **生成期生效** | `src/lychee_mas/runtime/backends/hf_backend.py` | 加 `apply_vocab_plan(plan)`：扩 tokenizer（add_tokens / merges）+ `resize_token_embeddings` + 按 `init_map` 初始化新行（transformers/torch 惰性，已是该文件约定） |
| 配置 | `configs/vocab/agentvocab.yaml`（新建） | 超参（见 §8） |
| 测试 | `tests/test_agentvocab.py`（新建） | 离线测 (a) 挖词条/`VocabPlan`（不引 torch）（见 §9） |
| 编排接入 | `model_client/injection`（`autogen_injection_client.py`）或 backend 装配处 | 每 agent 装配 model 时若配置了 vocab_adapter，则 `plan=adapt(agent,ctx)` → `backend.apply_vocab_plan(plan)` |
| CLI | `scripts/run_experiment.py` | 加 `--vocab`，透传配置 |

> `vocab/__init__.py` 与 `agentvocab.py` 被 import 时**不得触发** torch/transformers（`make selfcheck` 须 `HEAVY LOADED: NONE`）；真分词/embedding 操作只在 `adapt` 内部或 `hf_backend.apply_vocab_plan` 惰性 import。离线测试用假分词器（按空白/正则切）验证挖词条逻辑。

---

## 5. 从官方仓库迁移映射

> 匿名仓库当前 **403 不可访问**，发布后再核对。按方法骨架预置映射：

| AgentVocab（论文/代码，待核验） | → LycheeMAS 落点 |
|---|---|
| 从 tool-calling traces 挖结构片段 | `AgentVocab._mine_patterns(traces)` |
| 词条选择（频率 × token 节省 / 阈值） | `AgentVocab._select_vocab(...)` → `VocabPlan.new_tokens` |
| 新 token 初始化（无需训练，待证） | `VocabPlan.init_map` + `hf_backend.apply_vocab_plan` |
| 生成时使用新词表 | `hf_backend` 的 tokenizer/embedding 装配 |
| τ-bench/τ²-bench 评测脚本 | `scripts/run_experiment.py --vocab agentvocab --benchmark <tau-bench 适配>` |

---

## 6. 编排接入（集成）

见 [README.md](README.md) §3.1。AgentVocab 是**模型级**，**不在 `build_graph`/图层**，而在"装配某 agent 的 model"处生效：

```
每个 agent 装配 backend/model_client 时（runtime 后端内）：
  if vocab_adapter configured:
      plan = REGISTRY.create("vocab_adapter", name).adapt(agent, {"traces": ..., "top_k": ...})
      backend.apply_vocab_plan(plan)     # 扩 tokenizer + embedding（hf 后端，torch 惰性）
  # 之后该 agent 的生成走扩展后的词表 → 结构化输出 token 更少
```

- mock runtime 不涉及真模型 → AgentVocab 在 mock 下为 **no-op**（仅产出/记录 `VocabPlan`，便于离线测试与记账）。
- 真实降本只在 `hf`/`autogen+hf` 后端体现。CLI：`--vocab agentvocab`。

---

## 7. 开发流程（六步配方 + 里程碑）

- **M0（核验, ~1d）**：论文 PDF/匿名代码可访问后，回填 §1.2（选择目标函数、新 token 初始化、生成期机制）。在此之前可先做 (a) 离线部分。
- **M1（离线算词表, ~2–3d）**：`VocabPlan` + `AgentVocab._mine_patterns/_select_vocab`；替桩 + 注册 + config + 离线 test。
- **M2（生成期生效, ~3–4d）**：`hf_backend.apply_vocab_plan`（扩 tokenizer + `resize_token_embeddings` + `init_map` 初始化）；每 agent 装配处接入。需 `[all]`/GPU。
- **M3（实验, ~2d）**：在 τ-bench（或本仓库可用的工具调用任务）测 token/延迟降幅（对齐 −15~25%）、性能不降。

六步对齐：①读 `prune/base.py`（已确认）→②写 `agentvocab.py`+`types.py`+注册 →③`vocab/__init__.py` 触发 →④`configs/vocab/agentvocab.yaml` →⑤`tests/test_agentvocab.py` →⑥`make lint/test/selfcheck`。

---

## 8. 配置 `configs/vocab/agentvocab.yaml`

```yaml
# L2 vocab_adapter/agentvocab 超参（部分待 PDF 核验）
top_k: 500               # 新增词条上限
min_freq: 5              # 片段入选最小频次
min_fragment_tokens: 3   # 当前分词 >此值才考虑合并（token 节省阈值）
trace_source: transcript # 词条挖掘数据源：transcript | tool_logs
init_strategy: subword_mean   # 新 token embedding 初始化（待 PDF 核验）
apply_at_generation: true     # mock 下自动 no-op
```

---

## 9. 测试（离线、零重依赖）

`tests/test_agentvocab.py`（聚焦离线 (a) 部分，用假分词器，不引 torch）：

- 挖词条：给定含重复结构片段（如 `func(arg=`、`{"key":`）的轨迹，`adapt()` 产出的 `VocabPlan.new_tokens` 含这些高频片段，且按 `min_freq`/`top_k` 过滤正确。
- `init_map`：每个 new_token 映射到其构成子词序列（用假分词器验证）。
- 节省估计：`VocabPlan.stats` 的预计 token 节省 = Σ(片段频次 × (旧分词长度−1))，与构造数据一致。
- mock no-op：`Orchestrator(... vocab="agentvocab")` 在 mock runtime 下能跑通、不报错、`Trajectory` 正常。
- **生成期 (b)** 测试单独标注"需 `[all]`/GPU"，验证扩词表后结构化字符串分词长度变短、生成结果一致——不进默认 CI。

---

## 10. 实验与消融

| 配置 | 变量 | 看什么 |
|---|---|---|
| baseline（通用 tokenizer） | 无 adapt | 基线 token/延迟/工具调用成功率 |
| `--vocab agentvocab` | 结构感知词表 | 解码延迟 −15~25%、token↓、成功率不降 |
| 扫 `top_k∈{200,500,1000}` | 词表规模 | 降本-覆盖权衡 |
| 关/开 `init_map` 初始化 | 初始化策略 | 对生成质量的影响 |

**指标**：解码延迟（对齐 −15~25%）、平均 token/调用、工具调用成功率/任务精度、模型体积增量（新增 embedding 行，应 <1%）。基准：τ-bench/τ²-bench（需先适配为本仓库 benchmark；或在现有工具调用类任务上验证）。落 `runs/<model-tag>/agentvocab/<task>/`。

---

## 11. 验收标准

- `make lint && make test && make selfcheck` 全绿（`HEAVY LOADED: NONE`；`vocab` import 不触发 transformers）。
- 离线 `tests/test_agentvocab.py`（挖词条 + VocabPlan）全过。
- 离线 `scripts/run_experiment.py --runtime mock --vocab agentvocab --questions "..."` 跑通（mock 下 no-op）。
- 条件允许：hf 后端上扩词表后，结构化输出分词更短、解码更快、答案不变。

---

## 12. 预期时间 + 风险依赖

- **预期时间**：8–12 人天（M0 1d + M1 2–3d + M2 3–4d + M3 2d）。仅离线 (a) 部分则 ~3–4 人天。
- **依赖**：`hf_backend` 生成期改造（扩 tokenizer/embedding）；torch/transformers（生成期）；τ-bench 适配。**不依赖** L2 MASGraph 邻接前置。
- **风险**：① **公开信息缺口最大**（M0 前算法细节未定，离线挖词条可先做，但选择目标函数/初始化需 PDF 确证，否则可能偏离论文）；② 扩词表后 embedding/lm_head 一致性与生成稳定性需仔细验证（"正交于微调"意味着初始化不训练，质量风险更高）；③ 匿名代码 403，发布前无法对照实现细节；④ τ-bench 不在现有 `eval/benchmarks`，需新增 loader 或换近似任务。
