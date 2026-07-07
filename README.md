# LycheeMAS

基于 **AutoGen** 的多智能体系统（MAS）研究框架。主线是把整个 MAS 统一表示为一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**，每一层都是对 G（或其执行轨迹 τ）的一次变换：

| 层       | 模块                                        | 职责                                                                  |
| -------- | ------------------------------------------- | --------------------------------------------------------------------- |
| 构建     | `lychee_mas.layers.construct`             | 多智能体网络构建（团队组建 + 静态/动态图）                            |
| 剪枝     | `lychee_mas.layers.prune`                 | 网络剪枝与优化（含模型级词表降本）                                    |
| 记忆     | `lychee_mas.memory`                       | 运行时多维度多表征记忆管理（NL/隐空间/参数；已提升为顶层包）          |
| 处理     | `lychee_mas.layers.processing`            | 决定跑几次 MAS：`serial` 单次执行 + `parallel` 并发 K 次并聚合    |
| 归因训练 | `lychee_mas.trace` + `lychee_mas.train` | 错误归因/信用（trace，含 TraceStore）+ 强化学习/提示优化训练（train） |

设计四原则：**可插拔可消融**（registry + config）、**Runtime 抽象隔离 AutoGen**、**性能-成本联合度量**、**可复现**。

**当前研究主线 = 记忆层 CDM**（`lychee_mas.memory`）：双通道记忆（自然语言 + 隐空间）+ 运行时动态通道选择。隐空间通道两种物化策略——`soft_token`（免训练自压缩）与 `c2c`（训练好的 Cache-to-Cache 逐层 KV 融合器）。端到端实验驱动见 `scripts/run_mas.py`。

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

## 跑测试

```bash
make test     # PYTHONPATH=src pytest -q
make lint     # ruff check src
```

全部用 mock runtime，无需 API key。

## 目录

```
src/lychee_mas/
├── core/        统一图抽象类型（types）+ 组件注册表（registry）
├── runtime/     Runtime 协议 + 后端（mock / autogen / HF / vLLM）；唯一允许 import autogen 的位置
├── memory/      记忆层 CDM（顶层包）：channels / managers / routing + store.py（MemoryStore）
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

> 工程约束详见 `CLAUDE.md`；开发指南（如何新增一个组件、CDM 数据流、各层扩展点）见 `docs/DEVELOPMENT.md`。
