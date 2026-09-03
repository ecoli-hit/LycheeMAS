<h1 align="center">LycheeMAS</h1>

<div align="center">
    <img src="images/logo.png" width=250></img>
</div>

<p align="center"><strong>运行时可插拔的五层多智能体系统（MAS）研究框架</strong></p>
<p align="center"><em>一张图 G=(V,E,W,T,M)，五层变换 + 优化插件，组件可插拔、可消融、可复现。</em></p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS"><img alt="Release v0.2" src="https://img.shields.io/badge/release-v0.2-111827?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square">
  <img alt="核心零依赖" src="https://img.shields.io/badge/core%20deps-zero-0EA5E9?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS">GitHub</a> ·
  <a href="docs/DESIGN.md">架构设计</a> ·
  <a href="CLAUDE.md">开发规范</a> ·
  <a href="configs/">配置</a> ·
  <a href="examples/">示例</a>
</p>

> **把整个 MAS 统一表示为一张带时序与记忆状态的有向图 G=(V,E,W,T,M)，每一层都是对 G（或其执行轨迹 τ）的一次变换。**

> **版本范围：** 本 README 对应 **LycheeMAS v0.3**。

[框架总览](#框架总览) · [安装](#安装) · [跑 demo](#跑-demo离线零重依赖) · [跑测试](#跑测试) · [支持的 Benchmarks](#支持的-benchmarks) · [目录](#目录)

---

## 框架总览

五层多智能体系统（MAS）研究框架。主线是把整个 MAS 统一表示为一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**，每一层都是对 G（或其执行轨迹 τ）的一次变换；执行引擎可插拔（**autogen / langgraph 双后端**），运行前/运行后优化以 **GEPA 式插件**（pre_run / post_run / optimizer 三接缝）挂载。完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)：

| 层       | 模块                                        | 职责                                                                  |
| -------- | ------------------------------------------- | --------------------------------------------------------------------- |
| 构建     | `lychee_mas.layers.construct`             | 多智能体网络构建（团队组建 + 静态/动态图；**AgentInit** 多样性×相关性选队）|
| 剪枝     | `lychee_mas.layers.prune`                 | 网络剪枝与优化（含模型级词表降本）                                    |
| 记忆     | `lychee_mas.memory`                       | 运行时多维度多表征记忆管理（NL/隐空间/参数；已提升为顶层包）          |
| 处理     | `lychee_mas.layers.processing`            | 决定跑几次 MAS：`serial` 单次执行 + `parallel` 并发 K 次并聚合    |
| 归因训练 | `lychee_mas.trace` + `lychee_mas.train` | 错误归因/信用（trace，含 TraceStore）+ 强化学习/提示优化训练（train） |

设计四原则：**可插拔可消融**（registry + config）、**Runtime 抽象隔离执行引擎**（autogen / langgraph 双后端 + 共享注入引擎）、**性能-成本联合度量**、**可复现**。

---

## 安装

`src-layout`，包名 `lychee_mas`，发行名 `lychee-mas`。核心骨架**零运行依赖**：纯标准库即可 import 并跑通离线 mock 示例。

**环境约定**：conda 环境 `LycheeMAS`（系统 / CUDA 工具链）+ uv 管理的 `.venv`（torch / transformers / vLLM 等重依赖已就位）。包管理统一用 `uv`。

```bash
# 日常：进入环境
conda activate LycheeMAS
source .venv/bin/activate          # 首次创建见下方「初始化 .venv」

# 安装可编辑包（在已激活的 .venv 内执行）
uv pip install -e ".[dev]"         # 仅骨架 + 开发工具：离线 mock 即可跑通（无需 autogen/torch/API）
uv pip install -e ".[all]"         # 全量：autogen + 真实推理/评测依赖（torch/transformers/vLLM 已在 .venv 内）
```

> `.venv` 内已是 CUDA 版 torch / transformers / vLLM；默认 `uv pip install` 不会改动已满足约束的包，**请勿加 `--upgrade`**，以免把它们换成无 CUDA 的 PyPI 轮子。

**初始化 `.venv`（首次 / 换机）**：

```bash
conda create -n LycheeMAS python=3.12 -y && conda activate LycheeMAS   # 首次；已建好直接 conda activate LycheeMAS
uv venv .venv --python 3.12        # 用 uv 托管的 CPython 3.12 建 venv（等价 `make venv`）
source .venv/bin/activate
uv pip install -e ".[all]"
```

重依赖（torch / transformers / autogen / numpy / yaml / sympy …）一律惰性导入：缺这些库时 `import lychee_mas` 与 `REGISTRY.snapshot()` 仍可成功（纯离线开发只需 `.[dev]`）。

---

## 跑 demo（离线，零重依赖）

```bash
make demo
# 等价于：
PYTHONPATH=src python examples/01_static_chain_e2e.py
```

用 `runtime=mock` + 一个静态团队 + 一个 `TaskQuery` 跑通 `Orchestrator`，打印最终答案与 `REGISTRY.snapshot()`。

实验入口（CLI + 落盘，默认 `runtime=mock` 可离线）：

```bash
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --team default --aggregator self_consistency \
    --questions "2 plus 2 is 4" "answer is 7"
```

**AgentInit 选队**（`agent_selector/agentinit`，EMNLP'25 Findings）：用多样性×相关性的 Pareto 选择决定团队成员，替代固定 `--team` 模板。需 `.[construct]`（`vendi_score`）：

```bash
uv pip install -e ".[construct]"
# pool 模式：固定候选池 + 确定性挑选，离线、可复现（generate 模式走 LLM 现场生成角色）
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --selector agentinit --selector-mode pool \
    --questions "2 plus 2 is 4"
```

不带 `--selector` 时行为完全不变（走 `--team` 模板，零回归）。构建层设计见 `docs/DESIGN.md` §4.1。

---

## 跑测试

```bash
make test     # PYTHONPATH=src pytest -q
make lint     # ruff check src
```

全部用 mock runtime，无需 API key。

---

## 支持的 Benchmarks

内置 **20 个**基准（`eval/benchmarks/`，均注册为 `benchmark/<name>`，数据加载惰性；数据准备与重依赖走 `[benchmark]` extra）：

**文本推理 / 知识问答（7）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `gsm8k` | 小学数学应用题 | exact |
| `aime_2024` | AIME 竞赛数学 | 数值 + 符号等价 |
| `math500` | MATH-500 竞赛数学（MASPO 主实验数据集） | 数值 + 符号等价 |
| `medqa` | 医学选择题 | mc |
| `arc_easy` | 科学常识选择题 | mc |
| `openbookqa` | 开放课本科学选择题 | mc |
| `locomo10` | 长时对话记忆问答 | token-F1 |

**代码 / 通用助理（5）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `human_eval` | Python 代码生成 | 单测通过 |
| `gaia_validation` | GAIA 通用助理任务（validation 全集） | 短答案规整 |
| `gaia_validation_level_1` / `_2` / `_3` | GAIA 按难度分级子集 | 短答案规整 |

**MAS 轨迹分析（8）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `aftraj_audit` / `aftraj_audit_test` | AFTraj 轨迹安全审计（safe/unsafe + 定位关键步/agent） | mas_audit |
| `agent_collab_idr` | AgentCollabBench：指令衰减（Instruction Decay） | mas_instruction_decay |
| `agent_collab_rtd` | AgentCollabBench：追踪耐久（Tracer Durability） | mas_tracer_durability |
| `agent_collab_cpr` | AgentCollabBench：共识污染（Consensus Pollution） | mas_consensus_pollution |
| `agent_collab_clc` | AgentCollabBench：上下文泄漏（Context Leakage） | mas_context_leakage |
| `mast_failure` | MAST 失败模式分类 | taxonomy-F1 |
| `open_agent_traces` | Open Agent Traces 轨迹偏差检测 | mas_deviation |

统一记录格式 `{task, kind, question, gold, context}`；加载入口 `benchmarks.load(task, n)`，数据准备 `benchmarks.prepare(task)`；推理/打分分离与 pass@K 口径见 `docs/DESIGN.md` §6。

---

## 目录

```
src/lychee_mas/
├── core/        统一图抽象类型（types）+ 组件注册表（registry，17 类别）
├── runtime/     Runtime 协议 + 共享注入引擎（injection.py）+ 后端（mock / autogen / langgraph / HF / API）
├── memory/      记忆层（运行时组件）：channels / managers / routing + store / context
├── plugins/     插件系统：pre_run / post_run 插件 + optimizer（GEPA）+ MASProgram
│                + prerun（LangGraph 原生运行前优化：MASPO / AgentPrune 统一接口）
├── trace/       归因/信用（读侧）：attributor + credit_assigner + TraceStore
├── train/       训练（写侧）：RL 训练（trainer）
├── layers/      层变换（construct / prune / processing{parallel,serial}）
├── pipeline.py  Orchestrator.run（端到端编排 + 插件链，按 config 从 REGISTRY 取组件）
└── eval/        benchmarks（20 个）+ metrics（评分/落盘/pass@K）+ task_config
configs/         YAML 配置（按组件分组，含 plugins/）
examples/        可运行示例（离线 mock 优先）
scripts/         实验入口（run_mas / analyze_benchmark_run / run_experiment）
tests/           pytest（离线、零重依赖）
docs/            DESIGN.md（唯一架构设计文档）+ 文档站页面
```

> **架构设计**（模块职责、接口契约、组件全景、评测体系）见 `docs/DESIGN.md`；**开发规范**（环境、命令、黄金法则、六步配方、检查清单）见 `CLAUDE.md`；每个顶层包另有一份接口 `README.md`。

---

<p align="center"><strong>把每一个 MAS 研究问题，变成同一张图上一次可消融的变换。</strong></p>
