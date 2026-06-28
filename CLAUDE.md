# CLAUDE.md — LycheeMAS 开发指南（服务器版）

> 本文件供 **Claude Code / 编码代理** 在本仓库工作时阅读，是开发的「单一事实来源」。动手前请：(1) 读完「0 框架是什么」与「1 黄金法则」；(2) 跑一次 `make snapshot` 看现有组件；(3) 读你要改的那一层的 `base.py`。更深的实现细节见 `docs/DEVELOPMENT.md`（架构/CDM 数据流/迁移映射）。

---

## 0. 框架是什么（先建立全局认知）

**LycheeMAS** 是基于 **AutoGen** 的**五层多智能体系统（MAS）研究框架**。核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），每一层都是对 G（或其执行轨迹 τ）的一次变换。这样所有层共享同一套类型与运行时，组件可插拔、可消融。

五层（命名对应 **Construct · Prune · Memory · Aggregate · Attribute-and-Train**）：

| 层 | 模块 | 职责 | 状态 |
|----|------|------|------|
| 构建 | `lychee_mas.layers.construct` | 团队组建 + 静态/动态拓扑（AgentSelector / TopologyGenerator） | static 已实现；动态/AgentInit 待接 |
| 剪枝 | `lychee_mas.layers.prune` | 网络剪枝 + 模型级词表降本（GraphPruner / VocabAdapter） | 桩，待接 AgentDropout/AgentVocab |
| 记忆 | `lychee_mas.layers.memory` | 运行时多表征记忆管理（MemoryManager + MemoryRouter） | **CDM 已实现（本仓库当前主线）** |
| 聚合 | `lychee_mas.layers.aggregate` | 多轨迹聚合融合（TrajectoryAggregator） | self_consistency 已实现；dynamicagg 待接 |
| 归因训练 | `lychee_mas.layers.attribute_train` | 错误归因 + 强化学习（FailureAttributor / CreditAssigner / Trainer） | 桩，待接 |

**共享基座**：`core`（类型 + 组件注册表 REGISTRY）、`runtime`（隔离 AutoGen 的运行时抽象，含离线 mock 与 autogen 后端）、`stores`（TraceStore / MemoryStore）、`pipeline`（编排器 Orchestrator）、`eval`（基准 + 指标）。

> **当前研究主线 = 记忆层 CDM**（双通道记忆：自然语言通道 + 隐空间通道，运行时**动态选择记忆通道**）。这是本仓库目前唯一完整实现的研究层，其余层是为后续工作预留的稳定接口 + 占位。新功能基本都会落在 `layers/memory/`（CDM 的 router / channels / manager）。

> ⚠️ **AutoGen 是可替换的运行时后端，不是框架本身**。截至 2026 年 AutoGen 已进入维护模式（被 Microsoft Agent Framework 取代）。因此一切 AutoGen 调用都封装在 `runtime/backends/autogen_*.py` 之后，业务层不直接依赖它（见 §1.1 与 §6）。

---

## 1. 黄金法则（Golden Rules，必读，违反即返工）

1. **业务代码禁止 `import autogen_*`。** 运行时能力只通过 `lychee_mas.runtime` 的 `Runtime` 协议使用。**唯一允许 import autogen 的位置：`src/lychee_mas/runtime/backends/autogen_*.py`**，且其中也要惰性导入（放函数内部）。校验：`grep -rn "import autogen" src/lychee_mas/layers src/lychee_mas/core src/lychee_mas/pipeline.py` 必须为空。
2. **重依赖一律惰性导入。** `torch / transformers / autogen_* / numpy / yaml / sympy / datasets` 等只能在**函数/方法内部**或 `try-import` 里导入。**注册组件的模块在被 import 时不得触发这些库**——否则离线环境会崩。校验：`make selfcheck` 必须打印 `HEAVY LOADED: NONE`。
3. **每个算法 = 注册一个类 + 由配置/CLI 选择，绝不硬编码。** 用 `@REGISTRY.register(category, name)`；新增方法**不改 `pipeline.py`**。做对照实验只换 name（如 `--aggregator dynamicagg` vs `self_consistency`）。
4. **所有层只用 `core/types.py` 的公共类型**（`AgentSpec/Message/Answer/Trajectory/TaskQuery/Budget`）。层内部专用契约（如 `MemoryBundle/RouteDecision/RouterInputs`）留在该层，不要塞进 core。
5. **保持类型注解；提交前 `make lint` + `make test` 必须全绿**（测试优先用 mock runtime，无需 GPU/API key）。
6. **可复现**：固定随机种子（`--seed`）；每次实验落 config 快照 + git SHA 到 `runs/`；评测**同时**报告 accuracy / token / latency（记忆层加 memory-hit / cost-prompt-pos）。
7. **不提交密钥与大产物**：`runs/`、checkpoints、`*.jsonl`、`.env` 已在 `.gitignore`。模型路径走环境变量（`LYCHEE_HF_MODEL`）或 config，**绝不硬编码绝对路径**。
8. **整合已有论文代码**：先适配到对应 `base.py` 接口 + 注册，再迁移逻辑；保留原始引用与许可证；不要把外部仓库整包塞进来。
9. **CDM 的两条硬约束不可破坏**（见 §7）：隐空间 prefix 形状 `(1,P,2560)`；latent 只能在**同模型对**且 latent 可用时跨 agent 传，否则路由器回退 NL。

---

## 2. 环境与常用命令（conda + uv）

```bash
# 1) 建环境
conda create -n lychee python=3.11 -y && conda activate lychee
conda install -c conda-forge uv -y          # 或 pip install uv

# 2) 安装（src-layout，可编辑装）
uv pip install -e ".[dev]"     # 仅骨架 + 开发工具：离线 mock 即可跑通（无需 autogen/torch/API）
uv pip install -e ".[all]"     # 全量：autogen 0.7.x + torch + transformers + numpy + hydra + dev（真实跑分/训练用）

# 3) 日常（Makefile 用 '>' 作 recipe 前缀，见 .RECIPEPREFIX）
make demo        # 离线端到端：PYTHONPATH=src python examples/01_static_chain_e2e.py
make test        # PYTHONPATH=src pytest -q        （当前 24 passed）
make lint        # ruff check src
make snapshot    # 打印 REGISTRY.snapshot()（看现有组件）
make selfcheck   # 验证零重依赖：应打印 HEAVY LOADED: NONE

# 4) 直接命令
PYTHONPATH=src pytest -q tests/
ruff check src && ruff format src
mypy src/lychee_mas

# 5) 真实模型（需 [all] + GPU）
export LYCHEE_HF_MODEL=/data/.../Qwen3-4B     # HF 后端模型路径（不要写进代码）
```

> 若 `make` 没有 `snapshot/selfcheck` 目标，用以下等价命令：
> `PYTHONPATH=src python -c "import lychee_mas; from lychee_mas.core.registry import REGISTRY; import pprint; pprint.pprint(REGISTRY.snapshot())"`
> `PYTHONPATH=src python -c "import sys, lychee_mas; print('HEAVY LOADED:', [m for m in ('torch','transformers','autogen_core','autogen_agentchat','numpy','yaml','sympy','datasets') if m in sys.modules] or 'NONE')"`

约束：**核心代码与离线示例不得要求 GPU 或 API key**；真实推理/训练才需要，且缺依赖时必须优雅降级或明确报错。

---

## 3. 仓库结构（src-layout，包名 `lychee_mas`）

```
current_code/                     # 仓库根（= 服务器上的项目根）
├── CLAUDE.md                     # 本文件
├── README.md  pyproject.toml  Makefile  .gitignore
├── configs/                      # 组件分组 YAML（runtime/ memory/ topology/ aggregator/ + config.yaml）
├── docs/DEVELOPMENT.md           # 详细开发文档（架构 / CDM 数据流 / 迁移映射 / 测试）
├── examples/01_static_chain_e2e.py   # 离线端到端示例（runtime=mock，零重依赖）
├── scripts/run_experiment.py     # 实验 CLI 入口
├── tests/                        # 离线测试（24 个，零重依赖）
└── src/
    ├── lychee_mas/               # ★ 框架本体（开发都在这里）
    │   ├── core/{registry.py, types.py}
    │   ├── runtime/{base.py, backends/{mock_runtime, autogen_runtime, autogen_injection_client, hf_backend, vllm_client}.py}
    │   ├── stores/{trace_store.py, memory_store.py}
    │   ├── layers/{construct, prune, memory, aggregate, attribute_train}/   # 各层 base.py + 实现
    │   ├── pipeline.py           # Orchestrator
    │   └── eval/{benchmarks/, metrics.py, math_parsing_util.py, task_config.py}
    ├── LycheeMAS/                # 旧代码（迁移前原型，保留作参照，勿改；可在确认无误后删除）
    └── autogen/                  # vendored AutoGen 源码（参照，不打包/不 lint/不测试；运行时用 pip 装的 autogen）
```

> setuptools 只打包 `lychee_mas*`；`src/LycheeMAS` 与 `src/autogen` 不属于本包。新代码一律写在 `src/lychee_mas/`。

---

## 4. 核心抽象（先读这些再写）

**`core/types.py`（所有层共享，纯 dataclass，零重依赖）**
`AgentSpec`（图节点画像）、`Message`（一条通信消息，`.tokens` 汇总开销）、`Answer`（候选答案，聚合单元）、`Trajectory`（一次执行 τ：有序 messages + candidates + final_answer，`.total_tokens/.num_rounds`）、`TaskQuery`（含 `gold` 供评测）、`Budget/BudgetUnit`。

**`core/registry.py`（插件机制核心）**
```python
from lychee_mas.core.registry import REGISTRY

@REGISTRY.register("aggregator", "my_agg")
class MyAgg: ...

agg = REGISTRY.create("aggregator", "my_agg", k=5)
REGISTRY.list("aggregator"); REGISTRY.snapshot()
```
已登记 **CATEGORIES（13 个，新增类别须在此同步登记）**：
`runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, attributor, credit_assigner, trainer, benchmark`。

**`runtime/base.py`**：`Runtime` 协议（`async run(team, query) -> Trajectory`、`intercept(hook)`）+ 轻量 `MASGraph/MASTeam` 容器 + `BaseRuntime`（逐消息回调样板）。

**各层 `base.py` 协议签名**（实现新组件时按这个对齐）：
- `construct/base.py`：`AgentSelector.select(query, budget) -> list[AgentSpec]`；`TopologyGenerator.build(agents, query)`
- `prune/base.py`：`GraphPruner.prune(graph, context)`；`VocabAdapter.adapt(agent, context)`
- `memory/base.py`：`MemoryManager.observe(messages)` / `recall(decision, query) -> MemoryBundle`
- `aggregate/base.py`：`TrajectoryAggregator.aggregate(trajectories) -> Answer`
- `attribute_train/base.py`：`FailureAttributor.attribute(trajectory, context) -> list[Attribution]`；`CreditAssigner.credits(attributions, reward) -> dict[str,float]`；`Trainer.train(...)`

---

## 5. 已注册组件现状（`make snapshot` 的语义版）

| 类别 | 已实现/可跑 | 桩（registered，`NotImplementedError`，待接） |
|---|---|---|
| `runtime` | `mock`（离线）、`autogen`（需 [all]） | — |
| `model_client` | `injection`（注入+路由，需 [all]） | `vllm` |
| `memory_manager` | **`cdm`**（双通道，主线） | `mem0`, `ama` |
| `memory_router` | `static`, `fixed`（always-X） | `learned`, `soft_gate`（占位 fallback，论文目标） |
| `aggregator` | `self_consistency`（多数投票，纯标准库） | `dynamicagg`（在研） |
| `topology_generator` | `static`（按 team 模板产 AgentSpec） | — |
| `agent_selector` | — | `agentinit` |
| `graph_pruner` | — | `agentdropout`, `agentdropout_v2`, `agentprune` |
| `vocab_adapter` | — | `agentvocab` |
| `attributor` | — | `all_at_once`, `step_by_step`, `binary_search` |
| `credit_assigner` | — | `attribution_guided` |
| `trainer` | — | `maspo` |
| `benchmark` | `gsm8k/aime2024/medqa/arc_easy/openbookqa/locomo10`（数据加载惰性） | — |

> 桩能被 `REGISTRY.list` 看到，是**有意为之**：让消融矩阵在代码里可见、占好名字。把某个桩接成真实实现是后续研究的标准动作（见 §6）。

---

## 6. 开发流程

### 6.1 新增/实现一个组件（六步配方 = 一篇消融）

以记忆路由器为例（其它层同理）：

1. **读接口**：`src/lychee_mas/layers/memory/routing/base.py` 确认 `MemoryRouter.decide(RouterInputs) -> RouteDecision`。
2. **写实现**：`layers/memory/routing/my_router.py`
   ```python
   from lychee_mas.core.registry import REGISTRY
   from lychee_mas.layers.memory.routing.base import MemoryRouter, RouterInputs, RouteDecision

   @REGISTRY.register("memory_router", "my_router")
   class MyRouter(MemoryRouter):
       def decide(self, x: RouterInputs) -> RouteDecision: ...   # 重依赖在方法内惰性 import
   ```
3. **触发注册**：在该子包 `__init__.py` import 你的模块（包 `__init__` 会被 `lychee_mas/__init__` 链式 import）。
4. **加配置**：`configs/memory/my_router.yaml`（超参 + 默认值）。
5. **加测试**：`tests/test_my_router.py`，用 mock / 构造 `RouterInputs` 断言行为（含 `_enforce_availability` 回退）。
6. **验证**：`make lint && make test && make selfcheck`，再 `make demo` 看端到端不回归。

**绝不**为此改 `pipeline.py`——`Orchestrator` 只按名字从 REGISTRY 取组件。

### 6.2 跑实验 / 回归

```bash
# 离线自检（无模型，秒级，确认管线没断）
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --team default --aggregator self_consistency \
    --questions "2 plus 2 is 4" "answer is 7"

# 真实跑分（需 [all] + GPU + LYCHEE_HF_MODEL）
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime autogen --benchmark gsm8k --n 5 --seed 0 --model-tag qwen3-4b
```
结果落 `runs/<model-tag>/<...>/`（metrics.json + outputs.jsonl + config 快照）。**改 CDM 后务必先跑 `--questions` 离线自检、再跑小样本 `--n 5` 回归**，确认数值与改前一致。

### 6.3 改动前后的固定动作
- 改前：`git pull` → `make snapshot` → 读对应 `base.py`。
- 改后：`make lint && make test && make selfcheck`（三者全过才提交）。

---

## 7. 记忆层 CDM 数据流（当前主线，逻辑勿擅改）

CDM = **双通道记忆 + 动态通道选择**。用**单一 source**（seed + 运行中 transcript）按路由器决策物化出某个通道，保证 apples-to-apples（对比时只变「注入/路由」一个变量）。两个可替换接缝：**manager**（`memory_manager`：怎么存/召回）与 **router**（`memory_router`：本轮用哪个通道）。

一次 agent 发言（在 `model_client/injection` 的 `create()` 里汇合）：
```
① memory.observe(chat)                          # 更新记忆库（transcript 去重 + 失效 latent 缓存）
② router.decide(RouterInputs) -> RouteDecision(channel, P)
     └ _enforce_availability：latent 不可用 / 非同模型对 ⇒ 回退 nl（both 丢 latent 留 nl）
③ memory.recall(decision, query) -> MemoryBundle(nl_text?, latent_prefix?)
     none → 空 ; nl → 上一个 agent 输出(+PREV_OUTPUT_HEADER) ; latent → LatentMemory.build_prefix(source,P) ; both → 两者
④ 注入：nl_text 作 system 消息插开头；latent_prefix 在 backend embedding 层拼接
⑤ 生成：有 latent → generate_chat_with_prefix；否则 generate_chat
⑥ 记账：bump_turn + ctx.log_decision（成本/通道，可选写 TraceStore）
```
**latent 通道**（`layers/memory/channels/latent.py`，torch 惰性）：源文本 → HF 末层 hidden `(1,T,2560)` → `soft_token`（默认正确路径：`softmax(h@E^T/τ)@E` 投回输入嵌入空间，再 `segment_mean` 到 P）→ `(1,P,2560)` prefix。

**硬约束（勿破坏）**：prefix 形状 `(1,P,2560)`；latent 仅同模型对可跨 agent；`PREV_OUTPUT_HEADER` 集中定义在 `channels/nl.py`（`construct/templates.py` import 复用）；单次对话内按 `(len(source),P)` 缓存 prefix。

---

## 8. 惰性导入约定（最容易踩的坑）

- `autogen_*`：只在 `runtime/backends/autogen_*.py` 的**函数内部**导入。
- `torch/transformers`：只在 `hf_backend.py`、`channels/latent.py`、`autogen_injection_client.py` 的**方法内部**导入。注册 `memory_manager/cdm` 的 `managers/cdm.py` 被 import 时**不得**触发 torch。
- `yaml/sympy/datasets/numpy`：在各自使用函数内部惰性导入。
- 自检命令（应输出 `HEAVY LOADED: NONE`）见 §2。
- **MAF 迁移**：未来只需新增 `runtime/backends/maf_runtime.py` 并注册，业务层零改动。

---

## 9. 提交前检查清单（PR 自检）

- [ ] `make lint`（ruff）、`make test`（pytest 全绿）、`make selfcheck`（HEAVY LOADED: NONE）三者全过。
- [ ] 新组件已：实现对应 `base.py` 协议 + `@REGISTRY.register` + 子包 `__init__` 触发注册 + 加 config + 加 test。
- [ ] 没有在业务层 `import autogen_*`；重依赖均惰性导入。
- [ ] 没有硬编码模型/数据绝对路径（走 env / config）。
- [ ] 改了 CDM → 跑过离线 `--questions` 自检；条件允许时跑过 `--n 5` 小样本回归，数值无回归。
- [ ] 无密钥 / 无大文件 / 无 `runs/` 产物入库。
- [ ] 若新增了组件类别或改了组件状态，**同步更新本文件 §5 与 `docs/DEVELOPMENT.md`**。

---

## 10. 当前优先级与服务器注意事项

**在研重点（都在 `layers/memory/`）**：把 `memory_router/soft_gate`（软门控，论文主菜）与 `memory_router/learned` 从占位接成真实实现；配套反事实蒸馏数据（用 TraceStore 落盘的 routing_trace）与后续 RL 微调。

**后续其它层**（按需把 §5 的桩接成真实实现，保留原始引用）：`graph_pruner/agentdropout(_v2)`、`vocab_adapter/agentvocab`、`aggregator/dynamicagg`、`attributor/*` + `credit_assigner/attribution_guided` + `trainer/maspo`。RL 库（TRL/veRL/OpenRLHF）只在 `trainer/*` 实现里依赖，放 optional extra `[train]`，不污染基座。

**服务器注意**：
- 真实跑分/训练用 `uv pip install -e ".[all]"` 且需 GPU；模型路径用 `LYCHEE_HF_MODEL`。
- `src/autogen`（vendored）与 `src/LycheeMAS`（旧原型）不参与打包/测试；确认新包行为与旧原型一致后，旧目录可删。
- 不确定接口形状时：`make snapshot` + 读 `base.py` + 看 `examples/` 与 `tests/`，**不要臆造类型**。
