# 01 · AgentInit（构建层 · `agent_selector/agentinit`）

> 论文：**AgentInit: Initializing LLM-based Multi-Agent Systems via Diversity and Expertise Orchestration for Effective and Efficient Collaboration**（EMNLP 2025 Findings, CCF-B）
> arXiv [2509.19236](https://arxiv.org/abs/2509.19236) · [ACL 2025.findings-emnlp.636](https://aclanthology.org/2025.findings-emnlp.636/) · 官方代码 [github.com/1737423697/AgentInit](https://github.com/1737423697/AgentInit)

**状态：✅ 已实现并接入 Orchestrator**（`pool` 离线模式 + `generate` LLM 模式）。本文为**已落地组件**的使用与设计说明。

---

## 1. 方法核心

AgentInit 解决「如何**组建**一支高效 MAS 团队」：在候选角色上做**多目标平衡选择**，同时兼顾团队**多样性**（diversity）与任务**相关性**（relevance），选出小而互补的团队 —— 降冗余、降 token、保性能。

选择数学与官方源码一致（不是 NSGA-II 进化）：

- **枚举种群**：用 `itertools.combinations` 枚举规模 `min_roles..max_roles` 的角色子集当作种群（对齐官方 `Init_Population`），无交叉/变异/代数。
- **两个目标**（都取负 → 同时最大化）：
  - `relevance` = 子集内角色与 query 的**平均余弦**；
  - `diversity` = 角色相似度子矩阵的 **Vendi 分数**（官方同款 `vendi_score`）。
- **非支配排序**：`fast_non_dominated_sort` 取第一前沿（Pareto front）。
- **定案**：`pool` 模式用确定性替身（前沿里 relevance+diversity 之和最优、平手偏小团队再按下标字典序）；`generate` 模式用 LLM `SelectGroup`。

团队规模 **k 不固定**：由 `min_roles/max_roles`(默认 1..5) + 前沿涌现决定，无 `team_size` 超参。

---

## 2. 两种模式（同一 `select()`）

| 模式 | 角色来源 | 依赖 | 用途 |
|---|---|---|---|
| `pool`（默认） | 固定候选池 `pool.candidate_pool(query)` + 确定性 hash 嵌入 | 仅 `numpy` + `vendi_score`（`.[construct]`），零 GPU/API | 离线、可复现；也是「AgentInit 去掉 LLM 生成」的消融基线 |
| `generate` | LLM 现场生成角色（CreateRoles↔Check 双向反馈共识 → SelectGroup） | 上面 + HF 句向量编码器 + OpenAI 兼容 LLM | 忠实复现官方；含 RoleFeedback/PlanFeedback |

`generate` 模式的 LLM 走 env：`LYCHEE_LLM_MODEL / LYCHEE_LLM_API_KEY / LYCHEE_LLM_BASE_URL`；句向量编码器路径写在 config 的 `embedder_model`。

**budget 语义**（本 selector 专属）：仅 `generate` 模式认 `Budget(unit=TOKENS)`，给「角色生成多轮迭代」封顶（累计生成 token 超限即用当前角色定案）；`pool` 模式与 `calls/usd` 单位一律忽略。

---

## 3. 在框架中的定位与接入

- **层 / 类别 / 注册名**：构建（`layers/construct/`）· `agent_selector` · `agentinit`。
- **协议**：`construct/base.py::AgentSelector.select(query, budget=None) -> list[AgentSpec]`。
- **只产出团队成员**（AgentSpec 列表）；拓扑仍由 `StaticTopology` 接管 —— 契合框架「selector → topology」分工。

`Orchestrator.build_graph(query)` 内的可选 selector step（`pipeline.py`）：

```python
agents = None
if self.selector_name:                                    # 给了 --selector 才走
    selector = REGISTRY.create("agent_selector", self.selector_name, **self.selector_kwargs)
    agents = selector.select(query)
topo = REGISTRY.create("topology_generator", "static", team=self.team, model=self.model, rounds=self.rounds)
return topo.build(agents=agents)                          # StaticTopology.build 支持显式 agents（templates.py:222）
```

给了 `--selector` 时由 selector 决定成员，`--team` 的角色被覆盖、仅余 `meta["team"]` 标签（rounds 由 `--rounds` 决定，会打 warning）；**不给 `--selector` 时零回归**（走 team 模板）。

---

## 4. 落点文件

`src/lychee_mas/layers/construct/selectors/`

| 文件 | 职责 |
|---|---|
| `agentinit.py` | `AgentInitSelector`（注册 `agent_selector/agentinit`）+ `pool` 模式 `_select_pool` |
| `_pareto.py` | `cosine_matrix` / `cosine_to_query` / `objective_relevance` / `objective_diversity`(Vendi) / `fast_non_dominated_sort` |
| `pool.py` | `candidate_pool(query)` 候选池 + `embed_query` 确定性嵌入 + `CANDIDATE_ROLES` |
| `_generate.py` | `generate_and_select` 生成状态机（CreateRoles↔Check → SelectGroup，双向反馈） |
| `_llm.py` | OpenAI 兼容 chat helper（`tenacity` 重试） |
| `embedder.py` | HF 句向量编码器（`generate` 模式，惰性 import） |

配置 `configs/agents/agentinit.yaml`；测试 `tests/test_agentinit.py` · `tests/test_agentinit_generate.py` · `tests/test_orchestrator_selector.py`。

> `numpy / vendi_score / torch` 全部方法内惰性 import —— `selectors/__init__` 被 import 时零重依赖，`make selfcheck` 仍 `HEAVY LOADED: NONE`。

---

## 5. 用法

```bash
# 依赖：pool/generate 都需 vendi_score（.[construct]）
uv pip install -e ".[construct]"

# 离线 pool（mock runtime，确定性可复现）
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --selector agentinit --selector-mode pool --questions "..."

# 真实 generate：需 env LYCHEE_LLM_MODEL / LYCHEE_LLM_API_KEY / LYCHEE_LLM_BASE_URL + 编码器路径
python scripts/run_experiment.py --runtime autogen --selector agentinit --selector-mode generate \
    --selector-embedder /path/to/encoder --benchmark mmlu --n 5
```

CLI 开关（`run_experiment.py`）：`--selector` `--selector-mode` `--selector-critique-rounds` `--selector-embedder` `--selector-min-roles`（透传给 `AgentInitSelector` 构造）。

---

## 6. 配置 `configs/agents/agentinit.yaml`

```yaml
name: agentinit
mode: pool                       # pool（离线/消融）| generate（忠实 LLM 生成）
objectives: [relevance, diversity]
min_roles: 1                     # 子集枚举下界
max_roles: 5                     # 上界（对齐官方 Init_Population；规模由前沿涌现）
seed: 0                          # pool 选择确定性；此项为 API 一致性预留

# --- 仅 generate 模式 ---
embedder_model: /path/to/embed   # 句向量编码器 HF 路径（yaml 走 safe_load，不支持 ${env:}，写真实路径）
critique_rounds: 3               # CreateRoles↔Check 迭代上限（对齐官方 num_steps）
```

---

## 7. 测试（离线、零重依赖 · 需 `vendi_score`）

- `test_agentinit.py`：`pool` 模式规模约束、Pareto 前沿正确性、多样性生效、`seed` 确定性。
- `test_agentinit_generate.py`：`generate` 状态机（fake chat_fn + mock embedder，不调真 LLM）。
- `test_orchestrator_selector.py`：`Orchestrator(selector="agentinit")` 在 mock runtime 产出 `Trajectory`；不带 selector 与 team 模板一致。

> 三个测试文件顶部 `pytest.importorskip("vendi_score")`：未装 `.[construct]` 时自动跳过（不报错）。

---

## 8. 实验与消融

| 配置 | 变量 | 看什么 |
|---|---|---|
| `--team default`（无 selector） | 预定义团队 | 基线 acc / token |
| `--selector agentinit --selector-mode pool` | AgentInit 选队（离线） | acc↑、token↓、团队规模 |
| `--selector agentinit --selector-mode generate` | 忠实 AgentInit | 对齐论文 1.2×/1.7× |
| 随机选队 baseline | 随机 k 个 | 验证优于随机 |
| 关 diversity / 关 relevance | 单目标 | 两目标各自贡献 |

**指标**：accuracy、token、团队规模、（可选）多样性分。基准对齐论文（含 MMLU）。落 `runs/<model-tag>/agentinit/<task>/`。
