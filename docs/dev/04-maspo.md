# 04 · MASPO 开发文档（L5 归因训练）

> 论文：**MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems**（ICML 2026, CCF-A）
> arXiv [2605.06623](https://arxiv.org/abs/2605.06623) · ICML [poster 62219](https://icml.cc/virtual/2026/poster/62219) · 代码 [github.com/wangzx1219/MASPO](https://github.com/wangzx1219/MASPO)
> 目标：把 `trainer/maspo` 从桩接成真实组件，并搭最小 L5 训练闭环。
> 把握度 🟢（官方仓库 + arXiv 确证）。先读 [README.md](README.md) §3 共性事实。

---

## 1. 论文与方法核心

MASPO 对一支 MAS 里**每个 agent 的 system prompt 做联合提示级优化**（不更新模型权重），用下游对齐信号修复"局部最优但全局失败"的误对齐。

### 1.1 核心思想：joint = 局部/全局目标对齐

单个 agent 输出变好（local win）不一定让系统最终答案变对（global win）。MASPO 在评估一个候选 prompt 时**联合考虑它对下游 agent 成功的促进**（lookahead），而非只看本地有效性。

### 1.2 优化算法：数据驱动演化式束搜索（不更权重）

```
训练循环（按拓扑顺序逐 agent，重复 rounds 轮）：
  1. 采样：在 k≈50 个训练问题上跑当前 MAS，收集轨迹（agent 输入/输出/最终答案）
  2. 评分：Judge 判最终答案对错（二值 reward）
  3. 信用：三维 lookahead 评分（见 1.3）识别哪些 prompt 改动促成成功/失败
  4. 提出：LLM 据失败样本/前驱输出生成 2~3 个候选改进 prompt
  5. 束搜索：对每个候选在训练问题上重跑该 agent、评分，beam 保留 top-K
  6. 误对齐注入：把"local win 但 global fail"的案例作负样本喂回上游 prompt 生成器
```

### 1.3 三维信用（lookahead 评分）

```
# 非终端 agent
win_rate = rate_local·w_local + rate_next·w_next + rate_global·w_global
score    = win_rate − 0.5
# 默认权重 lookahead_weights = (0.4, 0.4, 0.2) = (local, next_local, global)
# 关 lookahead 时退化为：win_rate = rate_global·0.3 + rate_local·0.7
```

- **local**：该 agent 自身中间输出是否改进。
- **next**：直接下游 agent 在新输出上的处理是否改进。
- **global**：系统最终答案是否改进。

### 1.4 关键超参与论文指标

| 超参 | 默认 | 含义 |
|---|---|---|
| `beam_width` | 2 | 束宽 |
| `max_total_depth` | 9–10 | 每 agent 搜索深度 |
| `rounds_per_turn` | 2–3 | 每轮优化轮数 |
| `train_size k` | 50 | 优化用训练问题数 |
| `lookahead_weights` | (0.4,0.4,0.2) | (local,next,global) |
| `patience` | 3 | round-robin 早停 |

**论文报告**：6 个任务平均 **+2.9 个百分点**；beam=2 + depth=9 + 3 轮即收敛。

> 函数/超参名据官方仓库调研；**写实现前以 [官方最新代码](https://github.com/wangzx1219/MASPO) 再核对一次**。

---

## 2. 在框架中的定位

- **层**：L5 归因训练（`train/`）。
- **类别 / 注册名**：`trainer` / `maspo`（提示级、无权重更新，定位为廉价基线 / 暖启动）。
- **协议**：`train/base.py::Trainer`（+ 复用 `trace` 包的 `CreditAssigner`/`FailureAttributor`/`Attribution`）。
- **当前桩**：`src/lychee_mas/train/__init__.py:63`（`MASPOTrainer`，`credits()`/`train()` 抛 `NotImplementedError`）。

**关键定位判断**：MASPO 是**离线训练闭环**，不进 `Orchestrator.run()`。它需要一个**独立训练驱动**（采样→评分→信用→改 prompt→反哺）。MASPO 自带 MAS/优化器机器，迁移时**借优化逻辑，跑 MAS 用我们的 `Runtime`**。

---

## 3. 接口函数（签名对齐）

`train/base.py`（原样）：

```python
@dataclass
class Attribution:
    agent: str = ""; step: int = -1; is_fault: bool = False
    reason: str = ""; confidence: float = 0.0; meta: dict = field(default_factory=dict)

@runtime_checkable
class Trainer(Protocol):
    def credits(self, attrs: list[Attribution], reward: float) -> dict[str, float]: ...
    def train(self, generator, policies, mem_policies, traces, credits) -> Any: ...
```

MASPO 语义映射到该接口：

| 接口形参 | MASPO 语义 |
|---|---|
| `policies` | `dict[str,str]`：agent role → system_prompt（**优化对象/状态空间**） |
| `mem_policies` | 束搜索 beam（保留的候选 prompt 集，宽度 2） |
| `generator` | 提示生成器（调 LLM 产候选 prompt，= 官方 `_propose_new_prompt`） |
| `traces` | 采样轨迹 `list[Trajectory]`（来自 `Runtime.run` + `TraceStore`） |
| `credits` | 三维 lookahead 评分函数 / 结果（= 官方 `_evaluate_candidate`） |
| 返回 | 优化后的 `policies`（新 prompt 集，写回 templates/AgentSpec） |

> `Attribution` 在 MASPO 里偏"哪个 agent 的 prompt 改动有正/负贡献"——可用 `agent`/`is_fault`(误对齐)/`confidence`(score) 承载。

---

## 4. 写哪些代码 · 在哪里实现

| 动作 | 文件 | 说明 |
|---|---|---|
| **实现类** | `src/lychee_mas/train/trainers/maspo.py`（新建子包） | `class MASPOTrainer(Trainer)`，`@REGISTRY.register("trainer","maspo")`；重依赖惰性 import |
| 替桩 + 触发注册 | `src/lychee_mas/train/__init__.py` | 删 `MASPOTrainer` 桩，改为 `from .trainers.maspo import MASPOTrainer`；建 `trainers/__init__.py` import 之 |
| 三维信用 | 同 `maspo.py` 或 `credit_assigner/attribution_guided` | `_evaluate_candidate` → `credits(attrs, reward)`；可顺带把 `credit_assigner/attribution_guided` 一起接（核心贡献） |
| **训练驱动** | `scripts/train_maspo.py`（新建） | 采样(`Orchestrator`+`TraceStore`)→评分(`eval.metrics`)→`trainer.train`→把新 prompt 写回 `configs/agents/*` 或 `runs/` |
| Judge/reward | 复用 `src/lychee_mas/eval/metrics.py::score` | 数学/选择题用现成评分；代码任务再加 `CodeJudge`（惰性） |
| 配置 | `configs/trainer/maspo.yaml`（新建） | 超参（见 §8） |
| 测试 | `tests/test_maspo.py`（新建） | 离线 mock generator/judge，测束搜索 + 三维评分逻辑（不调真 LLM） |
| 依赖 | `pyproject.toml` | 加 optional extra `[train]`（litellm/openai/dspy/datasets 等） |

> `train/__init__.py` 与 `trainers/__init__.py` 被 import 时**不得触发** litellm/openai/torch（`make selfcheck` 须仍 `HEAVY LOADED: NONE`）；重依赖只在 `train()`/`generator` 调用路径内惰性 import。

---

## 5. 从官方仓库迁移映射

> 原则：**借优化逻辑，不搬其 MAS 类**；"跑 MAS / 重跑单节点"改用 LycheeMAS `Runtime`。保留原始引用与许可证（黄金法则 8）。

| 官方仓库（`wangzx1219/MASPO`） | → LycheeMAS 落点 |
|---|---|
| `optimizers.py::MAPromptOptimizer.optimize_all_fixed_rounds` | `MASPOTrainer.train(...)` 主循环 |
| `optimizers.py::_propose_new_prompt` | `generator`（注入式 prompt 生成，走 `Runtime`/`model_client`） |
| `optimizers.py::_evaluate_candidate` | `MASPOTrainer.credits(attrs, reward)` 的三维评分 |
| `optimizers.py::process_single_node` | 束搜索单节点扩展（内部方法） |
| `judges.py::{LLMJudgeAgent,CodeJudgeAgent}` | reward：接 `eval/metrics.py::score`（+ 可选 CodeJudge） |
| `agent.py::MAS.arun_single_node_only` | 用 `Runtime` 只重跑某 role（或最小子图）取其输出 |
| `agent.py::MAS.get_predecessors/successors` | 用 `MASGraph.edges` 取前驱/后继 |
| `run_maspo.py::main` | `scripts/train_maspo.py`（CLI + 落盘） |

---

## 6. 编排接入（集成）

见 [README.md](README.md) §3.1。MASPO **不进 `Orchestrator.run()`**，而是离线闭环：

```
scripts/train_maspo.py:
  load train split (benchmark)               # eval/benchmarks
  policies = 初始 system_prompt 集            # construct/templates.py 的 TEAMS
  for round in rounds_per_turn:
    for role in topological_order(graph):
      traces = 采样 N 题: Orchestrator(team).run(q) + TraceStore   # 用我们的 runtime
      reward = score(final, gold)                                  # eval/metrics
      credits = trainer.credits(attrs, reward)                     # 三维 lookahead
      cands  = generator(role, traces, ...)                        # LLM 产候选 prompt
      policies[role] = beam_search(cands, credits)                 # 保留最优
  保存 policies -> configs/agents/maspo_<task>.yaml / runs/
```

优化产物 = 一套更好的 `system_prompt`，回填到 `construct/templates.py::TEAMS` 或新 `agent_selector`/config，供后续 eval 用（暖启动）。

---

## 7. 开发流程（六步配方 + 里程碑）

- **M1（骨架, ~2d）**：`trainers/maspo.py` + 替桩 + 注册 + `configs/trainer/maspo.yaml`；`MASPOTrainer` 先用 mock generator/judge 让 `train()`/`credits()` 跑通（不调真 LLM）。
- **M2（三维信用, ~2d）**：实现 `credits()` 三维 lookahead 评分 + 误对齐检测；`tests/test_maspo.py` 用构造样本断言权重/score 公式。
- **M3（束搜索 + 驱动, ~3–4d）**：`process_single_node` 等价的束搜索；`scripts/train_maspo.py` 采样→评分→改 prompt→反哺；接 `Runtime`（mock 先行）。
- **M4（接真 LLM + 实验, ~3–5d）**：`[train]` extra + 真 generator/judge；在 GSM8K/MATH 等复现 +2.9 量级提升；落 `runs/`。

六步对齐：①读 `train/base.py`（已确认）→②写 `trainers/maspo.py`+注册 →③`__init__` 触发 →④`configs/trainer/maspo.yaml` →⑤`tests/test_maspo.py` →⑥`make lint/test/selfcheck`。

---

## 8. 配置 `configs/trainer/maspo.yaml`

```yaml
# L5 trainer/maspo 超参（对齐论文默认）
beam_width: 2
max_total_depth: 9
rounds_per_turn: 3
train_size: 50               # 优化用训练问题数 k
lookahead_weights: [0.4, 0.4, 0.2]   # (local, next, global)
patience: 3
use_lookahead_score: true
use_misleading_sampling: true        # 误对齐负样本注入
use_feedback: true                   # 下游失败反馈给上游
generator:
  model: null                # null=用 runtime 默认 model；真跑填 model_tag
  num_candidates: 3          # 每步候选 prompt 数
judge:
  kind: aime                 # 复用 eval/metrics score 的 kind：exact|aime|mc|f1
```

---

## 9. 测试（离线、零重依赖）

`tests/test_maspo.py`（**不调真 LLM**，用注入的假 generator/judge）：

- `credits()`：构造 `rate_local/next/global`，断言 `win_rate=Σrate·w` 与 `score=win_rate−0.5`；权重 (0.4,0.4,0.2) 正确；关 lookahead 时退化为 (0.7 local + 0.3 global)。
- 误对齐检测：local 改进但 global 失败的样本被标 `is_fault=True`/计入负样本。
- 束搜索：给定候选 + 打分，`beam_width=2` 保留得分最高 2 个；`patience` 早停生效。
- `train()` 端到端：mock generator 产候选、mock judge 给 reward，跑 1 round 后 `policies` 被更新且返回新 prompt 集（确定性，固定种子）。
- **真 LLM 路径**单独标注"需 `[train]`/API"，不进默认 CI。

---

## 10. 实验与消融

**消融矩阵**：

| 配置 | 变量 | 看什么 |
|---|---|---|
| baseline（原始 prompt） | 无优化 | 基线 acc |
| MASPO（关 lookahead） | 仅 local+global | 验证 lookahead 增益 |
| MASPO（开 lookahead） | 三维信用 | 论文主结果 +2.9 |
| 扫 `beam_width∈{1,2,4}`、`rounds∈{1,2,3}` | 搜索预算 | 收敛性/成本 |
| 关/开 `misleading_sampling`、`feedback` | 各机制 | 消融各组件 |

**指标**：accuracy（对齐论文 +2.9pt）、优化所耗 LLM 调用数/token（训练成本）、收敛轮数。基准：MATH / AQuA / GPQA / MBPP / HumanEval（对齐论文）。产物（优化后 prompt + 指标）落 `runs/<model-tag>/maspo/<task>/`。

---

## 11. 验收标准

- `make lint && make test && make selfcheck` 全绿（`HEAVY LOADED: NONE`；`train` import 不触发 litellm/torch）。
- 离线 `tests/test_maspo.py` 全过（mock generator/judge）。
- `scripts/train_maspo.py --runtime mock --team aime --n 5 --no-llm`（或等价 dry-run）能跑通闭环、产出更新后的 prompt 文件。
- 条件允许：接真 LLM 在小样本上跑 1 round，prompt 被合理改写、acc 不下降。

---

## 12. 预期时间 + 风险依赖

- **预期时间**：10–15 人天（M1 2d + M2 2d + M3 3–4d + M4 3–5d）。仅"骨架 + 三维信用 + mock 闭环"则 ~5–6 人天。
- **依赖**：L5 闭环此前完全未接（本篇搭最小驱动）；LLM 调用（generator/judge）→ `[train]` extra；`MASGraph.edges` 取前驱/后继（若已由 02 落地则直接复用）。
- **风险**：① 真 LLM 优化成本高（k=50 × beam × depth × rounds 次重跑），先用 mock + 小 k 打通；② 三维评分需"重跑单节点取输出"，要把官方 `arun_single_node_only` 正确映射到我们的 `Runtime`（最小子图重跑）；③ 优化产物回填路径（写回 templates vs config vs selector）需与团队约定统一。
