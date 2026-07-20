# CLAUDE.md — LycheeMAS 开发指南（服务器版）

> 本文件供 **Claude Code / 编码代理** 在本仓库工作时阅读，是开发的「单一事实来源」。动手前请：(1) 读完「0 框架是什么」与「1 黄金法则」；(2) 跑一次 `make snapshot` 看现有组件；(3) 读你要改的那一层的 `base.py`。更深的实现细节见 `docs/DEVELOPMENT.md`（架构/CDM 数据流/迁移映射）。

---

## 0. 框架是什么（先建立全局认知）

**LycheeMAS** 是基于 **AutoGen** 的**多智能体系统（MAS）研究框架**。核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），每一层都是对 G（或其执行轨迹 τ）的一次变换。这样所有层共享同一套类型与运行时，组件可插拔、可消融。

各变换阶段（命名对应 **Construct · Prune · Memory · Processing · Attribute-and-Train**）。**物理布局**：Memory 与 Attribute-and-Train 已提升为顶层子包（后者拆成 `trace` 归因/信用 + `train` 训练两包）；处理阶段由 `aggregate` 改名 `processing`（内分 `serial` 单次执行 + `parallel` 并发 K 次并聚合两子模块）；其余仍在 `layers/`：

| 层 | 模块 | 职责 | 状态 |
|----|------|------|------|
| 构建 | `lychee_mas.layers.construct` | 团队组建 + 静态/动态拓扑（AgentSelector / TopologyGenerator） | static + AgentInit（`agent_selector`）已实现；动态拓扑待接 |
| 剪枝 | `lychee_mas.layers.prune` | 网络剪枝 + 模型级词表降本（GraphPruner / VocabAdapter） | 桩，待接 AgentDropout/AgentVocab |
| 记忆 | `lychee_mas.memory`（**顶层包**，已提升出 `layers/`） | 运行时多表征记忆管理（MemoryManager + MemoryRouter） | **CDM 已实现（本仓库当前主线）** |
| 处理 | `lychee_mas.layers.processing`（内分 `serial` + `parallel` 两子模块） | 决定跑几次 MAS + 如何归约：`processor/serial`（跑 1 次 → 1 轨迹）、`processor/parallel`（并发 K 次 → 用 `aggregator` 聚合） | serial/parallel 均可跑；aggregator: self_consistency 已实现、dynamicagg 待接 |
| 归因训练 | `lychee_mas.trace`（归因/信用 + TraceStore）+ `lychee_mas.train`（RL/提示优化）——**两个顶层包**，由原 `attribute_train` 拆分 | 错误归因 + 强化学习（FailureAttributor / CreditAssigner ∈ trace；Trainer ∈ train） | 桩，待接 |

**共享基座**：`core`（类型 + 组件注册表 REGISTRY）、`runtime`（隔离 AutoGen 的运行时抽象，含离线 mock 与 autogen 后端）、`pipeline`（编排器 Orchestrator）、`eval`（基准 + 指标）。（原 `stores` 已拆解：`TraceStore`→`trace/store.py`，`MemoryStore`→`memory/store.py`，不再有独立 `stores` 包。）

> **当前研究主线 = 记忆层 CDM**（双通道记忆：自然语言通道 + 隐空间通道，运行时**动态选择记忆通道**）。这是本仓库目前唯一完整实现的研究层，其余层是为后续工作预留的稳定接口 + 占位。新功能基本都会落在 `memory/`（CDM 的 router / channels / manager）。

> ⚠️ **AutoGen 是可替换的运行时后端，不是框架本身**。截至 2026 年 AutoGen 已进入维护模式（被 Microsoft Agent Framework 取代）。因此一切 AutoGen 调用都封装在 `runtime/backends/autogen_*.py` 之后，业务层不直接依赖它（见 §1.1 与 §6）。

---

## 1. 黄金法则（Golden Rules，必读，违反即返工）

1. **业务代码禁止 `import autogen_*`。** 运行时能力只通过 `lychee_mas.runtime` 的 `Runtime` 协议使用。**唯一允许 import autogen 的位置：`src/lychee_mas/runtime/backends/autogen_*.py`**，且其中也要惰性导入（放函数内部）。校验：`grep -rn "import autogen" src/lychee_mas/layers src/lychee_mas/memory src/lychee_mas/trace src/lychee_mas/train src/lychee_mas/core src/lychee_mas/pipeline.py` 必须为空。
2. **重依赖一律惰性导入。** `torch / transformers / autogen_* / numpy / yaml / sympy / datasets` 等只能在**函数/方法内部**或 `try-import` 里导入。**注册组件的模块在被 import 时不得触发这些库**——否则离线环境会崩。校验：`make selfcheck` 必须打印 `HEAVY LOADED: NONE`。
3. **每个算法 = 注册一个类 + 由配置/CLI 选择，绝不硬编码。** 用 `@REGISTRY.register(category, name)`；新增方法**不改 `pipeline.py`**。做对照实验只换 name（如 `--aggregator dynamicagg` vs `self_consistency`）。
4. **所有层只用 `core/types.py` 的公共类型**（`AgentSpec/Message/Answer/Trajectory/TaskQuery/Budget`）。层内部专用契约（如 `MemoryBundle/RouteDecision/RouterInputs`）留在该层，不要塞进 core。
5. **保持类型注解；提交前 `make lint` + `make test` 必须全绿**（测试优先用 mock runtime，无需 GPU/API key）。
6. **可复现**：固定随机种子（`--seed`）；每次实验落 config 快照 + git SHA 到 `runs/`；评测**同时**报告 accuracy / token / latency（记忆层加 memory-hit / cost-prompt-pos）。
7. **不提交密钥与大产物**：`runs/`、checkpoints、`*.jsonl`、`.env` 已在 `.gitignore`。模型路径走环境变量（`LYCHEE_HF_MODEL`）或 config，**绝不硬编码绝对路径**。
8. **整合已有论文代码**：先适配到对应 `base.py` 接口 + 注册，再迁移逻辑；保留原始引用与许可证；不要把外部仓库整包塞进来。
9. **CDM 的硬约束不可破坏**（见 §7）：`soft_token` 隐空间 prefix 形状 `(1,P,2560)`；`c2c` projector 维度须匹配 `backend.kv_dims()`；两种策略下 latent 都只能在**同模型对**且可用时跨 agent 传，否则路由器回退 NL。

---

## 2. 环境与常用命令（conda + uv）

```bash
# 1) 建/进环境（本仓库约定：conda 环境 LycheeMAS + uv 管理的 .venv）
conda create -n LycheeMAS python=3.12 -y && conda activate LycheeMAS   # 首次；已建好直接 conda activate LycheeMAS
source .venv/bin/activate                      # 首次创建：uv venv .venv --python 3.12（或 make venv）

# 2) 安装（src-layout，可编辑装；在已激活的 .venv 内执行）
uv pip install -e ".[dev]"     # 仅骨架 + 开发工具：离线 mock 即可跑通（无需 autogen/torch/API）
uv pip install -e ".[all]"     # 全量：autogen 0.7.x + torch + transformers + numpy + hydra + dev（真实跑分/训练用）
                               # ⚠ 勿加 --upgrade：.venv 内已是 CUDA 版 torch/vLLM，升级会换成无 CUDA 的 PyPI 轮子

# 3) 日常（Makefile 用 '>' 作 recipe 前缀，见 .RECIPEPREFIX）
make demo        # 离线端到端：PYTHONPATH=src python examples/01_static_chain_e2e.py
make test        # PYTHONPATH=src pytest -q        （当前 30 passed）
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
├── scripts/                      # run_experiment.py（离线自检 CLI）· run_mas.py（CDM AIME 实验驱动）
│                                 # · train_c2c_projector.py / eval_c2c_aime.py（C2C 训练/评测）· download_c2c_data.sh
├── tests/                        # 离线测试（30 个，零重依赖）
└── src/
    ├── lychee_mas/               # ★ 框架本体（开发都在这里）
    │   ├── core/{registry.py, types.py}
    │   ├── runtime/{base.py, backends/{mock_runtime, autogen_runtime, autogen_injection_client, hf_backend, vllm_client}.py}
    │   ├── memory/               # ★ 记忆层 CDM（主线，顶层包）：base.py · context.py(RoutingContext) · store.py(MemoryStore) ·
    │   │                         #   channels/{nl, latent(soft_token), c2c_channel, c2c_projector} · managers/{DualChannelMemory, external} · routing/{base, static, learned, soft_gate}
    │   ├── trace/               # ★ 归因/信用（读侧，顶层包）：归因/信用（attributor + credit_assigner 桩）+ store.py（TraceStore）
    │   ├── train/               # ★ 训练（写侧，顶层包）：RL/提示优化（trainer/maspo 桩）；RL 库放 extra [train]
    │   ├── layers/{construct, prune, processing/{parallel,serial}}/   # 其余各层 base.py + 实现
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
已登记 **CATEGORIES（14 个，新增类别须在此同步登记）**：
`runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, processor, attributor, credit_assigner, trainer, benchmark`。

**`runtime/base.py`**：`Runtime` 协议（`async run(team, query) -> Trajectory`、`intercept(hook)`）+ 轻量 `MASGraph/MASTeam` 容器 + `BaseRuntime`（逐消息回调样板）。

**各层 `base.py` 协议签名**（实现新组件时按这个对齐）：
- `construct/base.py`：`AgentSelector.select(query, budget) -> list[AgentSpec]`；`TopologyGenerator.build(agents, query)`
- `prune/base.py`：`GraphPruner.prune(graph, context)`；`VocabAdapter.adapt(agent, context)`
- `memory/base.py`：`MemoryManager.observe(messages)` / `recall(decision, query) -> MemoryBundle`
- `processing/base.py`：`Processor.run(runner) -> ProcessingResult`（`runner: async () -> Trajectory`；serial 跑 1 次 / parallel 跑 K 次）；并行的归约策略 = `TrajectoryAggregator.aggregate(trajectories) -> Answer`
- `trace/base.py`：`FailureAttributor.attribute(trajectory, context) -> list[Attribution]`；`CreditAssigner.credits(attributions, reward) -> dict[str,float]`
- `train/base.py`：`Trainer.credits(...) / Trainer.train(...)`（RL/提示优化；消费 trace 的信用）

---

## 5. 已注册组件现状（`make snapshot` 的语义版）

| 类别 | 已实现/可跑 | 桩（registered，`NotImplementedError`，待接） |
|---|---|---|
| `runtime` | `mock`（离线）、`autogen`（需 [all]） | — |
| `model_client` | `injection`（注入+路由，需 [all]） | `vllm` |
| `memory_manager` | **`cdm`**（双通道，主线） | `mem0`, `ama` |
| `memory_router` | `static`, `fixed`（always-X） | `learned`, `soft_gate`（占位 fallback，论文目标） |
| `processor`（processing） | `serial`（跑 1 次 → 1 轨迹）、`parallel`（并发 K 次 + 聚合） | — |
| `aggregator`（parallel 的归约策略） | `self_consistency`（多数投票，纯标准库） | `dynamicagg`（在研） |
| `topology_generator` | `static`（按 team 模板产 AgentSpec） | — |
| `agent_selector` | **`agentinit`**（pool 离线 + generate LLM 生成，已接入 Orchestrator） | — |
| `graph_pruner` | — | `agentdropout`, `agentdropout_v2`, `agentprune` |
| `vocab_adapter` | — | `agentvocab` |
| `attributor` | — | `all_at_once`, `step_by_step`, `binary_search` |
| `credit_assigner` | — | `attribution_guided` |
| `trainer` | — | `maspo` |
| `benchmark` | 文本类 `gsm8k/aime_2024/medqa/arc_easy/openbookqa/locomo10` + benchmark 子系统 `human_eval/gaia_validation(_level_1..3)/aftraj_audit(_test)/agent_collab_{clc,cpr,idr,rtd}/mast_failure/open_agent_traces`（共 19，数据加载惰性；数据准备 + 重依赖走 `[benchmark]` extra） | — |

> 桩能被 `REGISTRY.list` 看到，是**有意为之**：让消融矩阵在代码里可见、占好名字。把某个桩接成真实实现是后续研究的标准动作（见 §6）。

> **latent 通道有两种物化策略**（`memory.latent_strategy` 配置项，属 `cdm` manager 的内部选项，非注册类别）：`soft_token`（免训练自压缩，默认）与 `c2c`（训练好的逐层 KV-cache 融合器，Cache-to-Cache）。详见 §7。

> **nl 通道有两种方法**（`memory.nl_strategy`）：`prev_output`（默认，把上一个 agent 输出原样转发）与 `simplemem`（接外部 **SimpleMem** 长时对话记忆——observe 喂 `add_dialogue`、recall 用 `finalize`+`ask` 检索问答）。SimpleMem 是**可选重依赖**（LLM+向量库+嵌入，需 `pip install -e ../SimpleMem` + LLM API/模型），在 `channels/nl.py` 内**惰性 import**，未装不影响 selfcheck / prev_output；可选 `memory.nl_simplemem` 传其构造参数。

---

## 6. 开发流程

### 6.1 新增/实现一个组件（六步配方 = 一篇消融）

以记忆路由器为例（其它层同理）：

1. **读接口**：`src/lychee_mas/memory/routing/base.py` 确认 `MemoryRouter.decide(RouterInputs) -> RouteDecision`。
2. **写实现**：`memory/routing/my_router.py`
   ```python
   from lychee_mas.core.registry import REGISTRY
   from lychee_mas.memory.routing.base import MemoryRouter, RouterInputs, RouteDecision

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

# 带 CDM 记忆通道的真实 AIME 实验（run_mas.py：backend+cdm+固定通道路由+AutoGen 端到端）
#   method: none|nl_only|latent_only|both；latent 走 memory.latent_strategy(soft_token|c2c)
CDM_DATA_ROOT=/data/mxy/Project/CDM/Data/raw CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_mas.py --config configs/aime_latent_c2c.yaml   # 纯 latent(C2C) 消融

# C2C projector 训练 / 评测（latent_strategy=c2c 的融合器）
CUDA_VISIBLE_DEVICES=0 python scripts/train_c2c_projector.py --data openhermes --steps 800 --out runs/c2c/<run>
CUDA_VISIBLE_DEVICES=0 python scripts/eval_c2c_aime.py --ckpt runs/c2c/<run>/final --n 30   # 融合 vs 不融合
```
结果落 `runs/<model-tag>/<...>/` 或 config 里 `eval.results_root`（metrics.json + outputs.jsonl + config 快照）。**改 CDM 后务必先跑 `--questions`/`--n 2` 离线自检、再跑小样本回归**，确认数值与改前一致。

### 6.3 改动前后的固定动作
- 改前：`git pull` → `make snapshot` → 读对应 `base.py`。
- 改后：`make lint && make test && make selfcheck`（三者全过才提交）。

---

## 7. 记忆层 CDM 数据流（当前主线，逻辑勿擅改）

CDM = **双通道记忆 + 动态通道选择**。用**单一 source**（seed + 运行中 transcript）按路由器决策物化出某个通道，保证 apples-to-apples（对比时只变「注入/路由」一个变量）。两个可替换接缝：**manager**（`memory_manager`：怎么存/召回）与 **router**（`memory_router`：本轮用哪个通道）。

一次 agent 发言（在 `model_client/injection` 的 `create()` 里汇合）：
```
① memory.observe(chat)                          # 更新记忆库（transcript 去重 + 失效 latent 缓存）
② router.decide(RouterInputs) -> RouteDecision(channel)  # P 已移出，归 LatentMemory
     └ _enforce_availability：latent 不可用 / 非同模型对 ⇒ 回退 nl（both 丢 latent 留 nl）
③ memory.recall(decision, query) -> MemoryBundle(nl_text?, latent_prefix?, latent_c2c?)
     none   → 空
     nl     → 上一个 agent 输出(+PREV_OUTPUT_HEADER)
     latent → 按 memory.latent_strategy：
                soft_token ⇒ latent_prefix = LatentMemory.build_prefix(source,P)   # (1,P,2560)
                c2c        ⇒ latent_c2c    = C2CLatentChannel.get_projectors()      # 懒加载的逐层 projector 栈
     both   → nl_text + 上面 latent 分量
④/⑤ 注入 + 生成（injection client 三分支，优先级 latent_c2c > latent_prefix > 无）：
     latent_c2c   → source = 上一个 agent 的(输入+输出)（从 ctx.decisions[-1] 组装）；backend.generate_chat_with_c2c
     latent_prefix→ prefix 拼到 backend embedding 层最前；backend.generate_chat_with_prefix
     都无         → backend.generate_chat（none / nl_only 走这里）
⑥ 记账：bump_turn + ctx.log_decision（成本/通道/prefix_len，可选写 trace.TraceStore）
```
**两种 latent 物化策略**（`memory.latent_strategy` 选，接口对上层一致）：
- **`soft_token`**（`channels/latent.py`，**免训练**，torch 惰性）：源文本 → HF 末层 hidden `(1,T,2560)` → `softmax(h@E^T/τ)@E` 投回输入嵌入空间 → `segment_mean` 到 P → `(1,P,2560)` prefix，在 embedding 层拼接。
- **`c2c`**（`channels/c2c_channel.py` + `channels/c2c_projector.py`，**训练好的 Cache-to-Cache 融合器**）：`C2CLatentChannel` 从 ckpt 懒加载逐层 `C2CProjector` 栈；注入时把**上一个 agent 的 KV** 经 projector 融进本 agent 生成——`generate_chat_with_c2c` 把 source/target 各自套 chat 模板分词，按**最长公共 token 块**（`longest_common_block`）对齐共享跨度，只在该跨度用 `generate_chat_with_cache_fusion`（prefill 逐层替换 post-RoPE K,V）融合；公共块过短(<`min_align`)或首个 agent 无前驱 ⇒ 回退普通生成。projector 由 `scripts/train_c2c_projector.py` 训（冻结 base 只训 projector、CLM 损失、可学门；`--data gsm8k|openhermes`，`--source-model` 跨模型），`scripts/eval_c2c_aime.py` 对比融合 vs 不融合。

**硬约束（勿破坏）**：
- `soft_token`：prefix 形状必须 `(1,P,2560)`；单次对话内按 `(len(source),P)` 缓存 prefix。
- `c2c`：projector 栈 build 维度须与 `backend.kv_dims()` 一致（同模型 src==tgt）；用最长公共 token 块对齐，无对齐则回退。
- 两种策略下 latent 均**仅同模型对可跨 agent**（约束 #2）；`PREV_OUTPUT_HEADER` 集中定义在 `channels/nl.py`（`construct/templates.py` import 复用）。

---

## 8. 惰性导入约定（最容易踩的坑）

- `autogen_*`：只在 `runtime/backends/autogen_*.py` 的**函数内部**导入。
- `torch/transformers`：只在 `hf_backend.py`、`channels/latent.py`、`autogen_injection_client.py` 的**方法内部**导入。注册 `memory_manager/cdm` 的 `managers/DualChannelMemory.py` 被 import 时**不得**触发 torch。
- `channels/c2c_projector.py` 模块**顶层 import torch**（数值端到端保真），因此**不得**在注册路径被顶层 import——`channels/c2c_channel.py`（供 `managers/DualChannelMemory.py` 顶层 import，本身零重依赖）只在 `_load()` 内惰性 import 它；`hf_backend`、`scripts/*_c2c_*` 同样惰性 import。校验仍以 `make selfcheck` 打印 `HEAVY LOADED: NONE` 为准。
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

**已落地（`memory/`）**：CDM 双通道端到端可跑（`run_mas.py`，四消融 none/nl_only/latent_only/both）；latent 通道两条策略——`soft_token`（免训练）与 **`c2c`**（Cache-to-Cache 逐层 KV 融合器，含训练/评测脚本 + MAS 内 `generate_chat_with_c2c` 集成）。

**在研重点（都在 `memory/`）**：把 `memory_router/soft_gate`（软门控，论文主菜）与 `memory_router/learned` 从占位接成真实实现；配套反事实蒸馏数据（用 `trace.TraceStore` 落盘的 routing_trace）与后续 RL 微调。

**后续其它层**（按需把 §5 的桩接成真实实现，保留原始引用）：`graph_pruner/agentdropout(_v2)`、`vocab_adapter/agentvocab`、`aggregator/dynamicagg`、`attributor/*` + `credit_assigner/attribution_guided` + `trainer/maspo`。RL 库（TRL/veRL/OpenRLHF）只在 `trainer/*` 实现里依赖，放 optional extra `[train]`，不污染基座。

**服务器注意**：
- 真实跑分/训练用 `uv pip install -e ".[all]"` 且需 GPU；模型路径用 `LYCHEE_HF_MODEL`。
- `src/autogen`（vendored）与 `src/LycheeMAS`（旧原型）不参与打包/测试；确认新包行为与旧原型一致后，旧目录可删。
- 不确定接口形状时：`make snapshot` + 读 `base.py` + 看 `examples/` 与 `tests/`，**不要臆造类型**。
