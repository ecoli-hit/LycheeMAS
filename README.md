<h1 align="center">LycheeMAS</h1>

<div align="center">
    <img src="images/logo.png" width=250></img>
</div>

<p align="center"><strong>基于 AutoGen 的五层多智能体系统（MAS）研究框架</strong></p>
<p align="center"><em>一张图 G=(V,E,W,T,M)，五层变换，组件可插拔、可消融、可复现。</em></p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS"><img alt="Release v0.2" src="https://img.shields.io/badge/release-v0.2-111827?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square">
  <img alt="核心零依赖" src="https://img.shields.io/badge/core%20deps-zero-0EA5E9?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS">GitHub</a> ·
  <a href="CLAUDE.md">工程规范</a> ·
  <a href="docs/DEVELOPMENT.md">开发文档</a> ·
  <a href="docs/BENCHMARK_HANDOFF.md">Benchmark 交接</a> ·
  <a href="configs/">配置</a> ·
  <a href="examples/">示例</a>
</p>

> **把整个 MAS 统一表示为一张带时序与记忆状态的有向图 G=(V,E,W,T,M)，每一层都是对 G（或其执行轨迹 τ）的一次变换。**

> **版本范围：** 本 README 对应 **LycheeMAS v0.2**。

[框架总览](#框架总览) · [安装](#安装) · [跑 demo](#跑-demo离线零重依赖) · [Eval Studio](#eval-studio) · [跑测试](#跑测试) · [目录](#目录)

---

## 框架总览

基于 **AutoGen** 的五层多智能体系统（MAS）研究框架。主线是把整个 MAS 统一表示为一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**，每一层都是对 G（或其执行轨迹 τ）的一次变换：

| 层       | 模块                                        | 职责                                                                  |
| -------- | ------------------------------------------- | --------------------------------------------------------------------- |
| 构建     | `lychee_mas.layers.construct`             | 多智能体网络构建（团队组建 + 静态/动态图；**AgentInit** 多样性×相关性选队）|
| 剪枝     | `lychee_mas.layers.prune`                 | 网络剪枝与优化（含模型级词表降本）                                    |
| 记忆     | `lychee_mas.memory`                       | 运行时多维度多表征记忆管理（NL/隐空间/参数；已提升为顶层包）          |
| 处理     | `lychee_mas.layers.processing`            | 决定跑几次 MAS：`serial` 单次执行 + `parallel` 并发 K 次并聚合    |
| 归因训练 | `lychee_mas.trace` + `lychee_mas.train` | 错误归因/信用（trace，含 TraceStore）+ 强化学习/提示优化训练（train） |

设计四原则：**可插拔可消融**（registry + config）、**Runtime 抽象隔离 AutoGen**、**性能-成本联合度量**、**可复现**。

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

不带 `--selector` 时行为完全不变（走 `--team` 模板，零回归）。设计与用法详见 `docs/dev/01-agentinit.md`。

---

## Eval Studio

Eval Studio 是运行环境检测、benchmark Raw/Prepared 数据与注册模型管理、RoleProfile/TeamSpec
图编辑、Deployment 服务发现与复用、TeamInstance inference-slot 绑定、完整命令生成、tmux 启动和 GroupChat/spans
实时/回放的可视化控制面。它调用现有 CLI 和 scorer，不另建一套
评测逻辑。主导航依次为运行环境、资源中心、部署管理（Deployment）、团队管理（Team）、实验装配
（Experiment）和运行记录；六者同级。Team 编辑完整 TeamSpec，Deployment 左侧维护 DeploymentSpec、
右侧管理 DeploymentInstance，Experiment 只负责 benchmark 与逐角色 Binding 装配。界面默认使用本
README 约定的 `LycheeMAS` + `.venv`，也可以检测、修补
并切换其它 Python 环境。benchmark run 会持续写 `group_chat.jsonl`（完整 AutoGen transcript）和
`spans.jsonl`（模型/backend/工具诊断）；`--trace-detail-level compact|full` 只控制 model-call start
是否额外保存三层完整输入消息。

当前原生 benchmark 除 GSM8K、AIME 2024、HumanEval、GAIA 和严格 MAS 数据集外，还包括
BIG-Bench Extra Hard（BBEH）、Humanity's Last Exam（HLE）、SWE-bench Verified 与 WorkBench；
它们分别保留官方确定性 evaluator、结构化 judge、Docker harness 和最终状态 scorer，不使用统一 exact-match
替代原始评测协议。完整能力矩阵和受限数据说明见 `docs/BENCHMARK_HANDOFF.md` 的“当前能力快照”和
“Benchmark 统一合同”。

```bash
uv pip install -e ".[benchmark,studio]"
./serve_eval_studio.sh build
./serve_eval_studio.sh -H 127.0.0.1 -p 8010
```

前端构建需要 Node.js `^20.19` 或 `>=22.12`。构建入口会依次使用 `LYCHEE_NODE_HOME`、当前
`PATH` 中的兼容版本和 `$HOME/.local/node-v*/bin/node`，避免服务器自带的旧 Node 被误用；也可以先设置
`export LYCHEE_NODE_HOME=/path/to/node-v22`。前端目录中的 `npm run build` 也会转入同一构建入口。远程使用时通过 SSH
把服务器的 `127.0.0.1:8010` 转发到本机；如果 SSH Host 设置了 `ClearAllForwardings yes`，转发命令需要显式添加 `-o ClearAllForwardings=no`。完整设计、
backend 能力边界和操作流程见 `docs/BENCHMARK_HANDOFF.md` 的“环境与 Eval Studio”及
“Deployment、Team 与 Experiment”。

---

## 跑测试

```bash
make test     # PYTHONPATH=src pytest -q
make lint     # ruff check src
```

全部用 mock runtime，无需 API key。

---

## 目录

```
src/lychee_mas/
├── core/        统一图抽象类型（types）+ 组件注册表（registry）
├── runtime/     Runtime 协议 + 后端（mock / autogen / HF / vLLM）；唯一允许 import autogen 的位置
├── memory/      记忆层（顶层包）：channels / managers / routing + store.py（MemoryStore）
├── trace/       归因/信用（读侧）：归因/信用（attributor + credit_assigner）+ store.py（TraceStore）
├── train/       训练（写侧）：RL/提示优化训练（trainer/maspo）
├── layers/      层变换（construct / prune / processing{parallel,serial}；各层 base.py + 注册实现）
├── pipeline.py  Orchestrator.run（端到端编排，按 config 从 REGISTRY 取组件）
└── eval/        benchmarks（数据 loaders）+ metrics（评分/落盘）+ task_config
configs/         Hydra/YAML 配置（按组件分组）
examples/        可运行示例（离线 mock 优先）
scripts/         实验入口（argparse + 落盘）
tests/           pytest（离线、零重依赖）
docs/            开发文档（见 docs/DEVELOPMENT.md）
```

> 工程约束详见 `CLAUDE.md`；开发指南（如何新增一个组件、各层扩展点）见 `docs/DEVELOPMENT.md`；benchmark 接入交接见 `docs/BENCHMARK_HANDOFF.md`。

---

<p align="center"><strong>把每一个 MAS 研究问题，变成同一张图上一次可消融的变换。</strong></p>
