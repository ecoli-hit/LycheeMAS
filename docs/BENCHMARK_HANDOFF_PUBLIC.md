# LycheeMAS Benchmark 交接文档（公共版）

> 最后更新：2026-07-15
> 目标：面向项目外部接手者，完整说明 LycheeMAS benchmark 板块的上下文、文件结构、运行方式、数据准备、指标口径、已知限制和后续扩展路线。
> 说明：本文是公开交接版，已移除个人路径、远程登录信息、网络配置、私有凭据和历史调试流水账。命令默认从 LycheeMAS 仓库根目录执行。

---

## 1. 当前结论

这次工作不是继续沿用 AutoGenBench 的原始 CLI/Docker 项目目录，而是把 HumanEval、GAIA、严格 MAS 等 benchmark 接入 LycheeMAS 自己的 `src/lychee_mas/eval` 评测系统。

当前可用状态：

- `gsm8k`、`aime_2024`、`human_eval`、`gaia_validation*` 已经能通过 LycheeMAS native loader 读取。
- ARC-Easy、OpenBookQA、MedQA、LoCoMo10、AFTraj-2K、AgentCollabBench、MAST-Data、Open Agent Traces 已接入 loader/preparer/scorer。
- 数据准备脚本支持 ModelScope / HuggingFace 两种来源，默认 `auto`；下载源 ID、环境变量覆盖、其它候选和 direct-file fallback 集中维护在 `src/lychee_mas/eval/benchmarks/source_catalog.json`。
- `scripts/run_mas.py` 只负责推理：运行中实时写 `spans.jsonl`，每个 case 完成或失败时写 `predictions.jsonl`；`scripts/analyze_benchmark_run.py --score-predictions` 负责读取 gold、判断正确与否、生成 `outputs.jsonl/metrics.json`。
- `scripts/run_benchmark_batch.py` 是新增 benchmark 板块的批量入口，`--size smoke|full` 可批量跑 API / 本地 HF 两类 backend。
- benchmark 数据目录拆成 `data/benchmarks/raw` 与 `data/benchmarks/prepared`；运行结果目录推荐 `runs/benchmarks`。
- LycheeMAS 当前使用统一 `runtime=autogen` 主链路：普通文本 MAS、function/tool calling、Coder/Terminal、File/Web，以及 `none/nl_only/latent_only/both` 消融都走同一条 runtime。
- HumanEval/GAIA 默认使用工具型 team：`human_eval` / `gaia`。普通文本任务不一定调用工具，但 runtime 本身具备工具能力。
- AutoGenBench/agbench 源码可作为官方 benchmark 资产来源和参考，但当前主线不直接调用 agbench CLI/Docker。

推荐后续路线：

1. 继续保持 LycheeMAS native eval 作为主入口。
2. AutoGen 负责 agent/runtime 推理；benchmark 自己的数据与 scorer 负责数据加载和评分。
3. 在原 `AutoGenRuntime` 上维护工具能力，避免做 GAIA 专用 runtime。
4. agbench 作为官方 benchmark 资产来源：复用/借鉴 `Tasks`、`Templates`、scenario 展开、Docker 沙盒和 tabulate 思路，但结果仍落到 LycheeMAS 统一 metrics/output 体系。
5. 结果文件保持四层结构：span/event trace、case-level prediction、case-level scored output、run-level aggregate。

---

## 2. 相关路径与环境

本文所有命令默认从 LycheeMAS 仓库根目录执行：

```bash
cd /path/to/LycheeMAS
```

推荐使用仓库相对路径，避免把个人机器或运行环境的绝对路径写进文档/配置：

```bash
export LYCHEE_BENCHMARK_RAW_ROOT=data/benchmarks/raw
export LYCHEE_BENCHMARK_PREPARED_ROOT=data/benchmarks/prepared
export LYCHEE_BENCHMARK_RUNS_ROOT=runs/benchmarks
export LYCHEE_HF_MODEL=models/Qwen3-4B-Instruct-2507
export MODEL_PATH=models/Qwen3-4B-Instruct-2507
export PY=python
```

依赖文件：

```text
pyproject.toml                项目元信息和 optional dependencies；核心 dependencies 仍保持为空，保证 import 轻量。
requirements/benchmark.txt    benchmark/full-run 环境依赖安装入口；用于数据下载、AutoGen runtime、HumanEval/GAIA 工具链和分析脚本。
```

推荐准备环境时使用：

```bash
$PY -m pip install -r requirements/benchmark.txt
$PY -m playwright install chromium
```

本地 GPU 环境里，`torch` 最好按运行环境 CUDA/PyTorch 版本单独确认；如果环境里已经有合适的 GPU 版 `torch`，不要为了重装 requirements 破坏已有 CUDA 组合。

不应提交到公开仓库的内容：

- `runs/`、`data/benchmarks/`、checkpoints、`*.jsonl`、真实模型权重。
- API key、token、私有凭据。
- private 交接文档、汇报稿、个人调试记录。

---
## 3. 目前修改过的主要文件

Benchmark 入口与 loader：

```text
src/lychee_mas/eval/benchmarks/__init__.py
src/lychee_mas/eval/benchmarks/common.py
src/lychee_mas/eval/benchmarks/gsm8k.py
src/lychee_mas/eval/benchmarks/aime_2024.py
src/lychee_mas/eval/benchmarks/choice_qa.py
src/lychee_mas/eval/benchmarks/locomo10.py
src/lychee_mas/eval/benchmarks/human_eval.py
src/lychee_mas/eval/benchmarks/gaia.py
src/lychee_mas/eval/benchmarks/aftraj.py
src/lychee_mas/eval/benchmarks/agent_collab.py
src/lychee_mas/eval/benchmarks/mast_data.py
src/lychee_mas/eval/benchmarks/open_agent_traces.py
```

评测与指标：

```text
src/lychee_mas/eval/metrics.py
scripts/run_mas.py
src/lychee_mas/layers/memory/routing/context.py
src/lychee_mas/runtime/backends/autogen_injection_client.py
src/lychee_mas/runtime/backends/autogen_runtime.py
src/lychee_mas/runtime/backends/hf_backend.py
src/lychee_mas/runtime/backends/__init__.py
src/lychee_mas/layers/construct/templates.py
src/lychee_mas/eval/task_config.py
configs/benchmarks/api.yaml
configs/benchmarks/local_hf.yaml
configs/benchmarks/human_eval.yaml
configs/benchmarks/gaia.yaml
configs/benchmarks/strict_mas_api.yaml
```

数据准备与测试：

```text
scripts/prepare_benchmarks.py
scripts/run_benchmark_batch.py
docker/gaia.Dockerfile
tests/test_eval_benchmark_extensions.py
```

### 3.1 configs 目录当前怎么理解

`configs/*.yaml` 是“运行预设”，不是 benchmark 注册入口。一个 benchmark 是否存在，主要看
`src/lychee_mas/eval/benchmarks` 的 loader/preparer/scorer，以及 `task_config.py` 的默认
team/extractor。YAML 只负责给某次运行提供 backend、runtime、router、样本数和运行结果目录等默认参数。

`configs/` 下面原来还有几个组件配置目录，它们不是本轮新增 benchmark 配置：

```text
configs/aggregator/   聚合器配置，例如 self-consistency、dynamic aggregation
configs/memory/       记忆模块配置，例如 CDM、mem0
configs/runtime/      runtime 类型配置，例如 autogen、mock
configs/topology/     拓扑生成配置，例如 static topology
```

我们新增的 benchmark 运行预设统一放在 `configs/benchmarks/`，和这些组件配置分开。

因此不是每个 benchmark 都需要一个单独 YAML。普通文本/选择题/严格 MAS 任务可以共用通用配置：

```text
configs/benchmarks/api.yaml             普通 API backend 通用配置，可用 --task 和 --n 切换任务/样本数
configs/benchmarks/local_hf.yaml        本地 HF/GPU 通用配置，可用 --task、--method、--n 切换实验
configs/benchmarks/strict_mas_api.yaml  严格 MAS benchmark 的 API 示例配置
```

只有当某个 benchmark 需要特殊 runtime 配置时，才单独放 YAML：

```text
configs/benchmarks/human_eval.yaml             HumanEval 工具型 benchmark 正式配置；API/本地 HF 都可复用，默认 team 由 task_config.py 选 human_eval
configs/benchmarks/gaia.yaml                   GAIA 工具型 benchmark 正式配置；API/本地 HF 都可复用，默认 team 由 task_config.py 选 gaia
```

我们新添加的 benchmark 运行预设：

```text
configs/benchmarks/api.yaml             普通 benchmark API 通用预设，部分/全量都用它
configs/benchmarks/local_hf.yaml        普通 benchmark 本地 HF/GPU 通用预设，部分/全量都用它
configs/benchmarks/human_eval.yaml  HumanEval 正式预设
configs/benchmarks/gaia.yaml        GAIA 正式预设
configs/benchmarks/strict_mas_api.yaml  严格 MAS benchmark API 预设
```

新配置推荐写：

```yaml
eval:
  runs_root: runs/benchmarks/api
```

旧字段 `eval.results_root` 仍兼容，但后续新增配置不要再用它。

命名建议：

- 代码参数、配置文件和结果目录统一使用 `local_hf`，例如 `configs/benchmarks/local_hf.yaml`、`--mode local-hf`、`runs/.../local_hf/`。
- 中文说明里可以写“本地 GPU”，但它只是资源描述；真正的 backend 名是 `hf`，实现类是 `HFBackend`。
- 不建议在文件名里混用 `local_gpu`、`gpu_local`、`local`，否则后面看结果目录时很难判断到底是 HF 直连、vLLM server，还是别的本地推理方式。

早期/历史配置：

```text
configs/aime_both.yaml
configs/aime_latent.yaml
configs/aime_nl_prev.yaml
configs/aime_single_baseline.yaml
configs/default.yaml
configs/config.yaml
```

这些主要来自早期 AIME/CDM 或 Hydra/mock 运行方式，可以作为实验模板参考，但新 benchmark 接入时优先使用
`configs/benchmarks/` 里的预设。

当前建议的命名原则：

- 正式配置名只写 benchmark；backend 由 YAML 默认值或命令行 `--backend` 决定，例如 `human_eval.yaml`、`gaia.yaml`。
- 不在正式配置名里写 `tools` 或 `magentic_one`，因为 HumanEval/GAIA 的正式默认跑法本来就应使用工具。
- GAIA 不保留纯文本 baseline 配置；如果临时用 `--team fact` 调试，只能写在临时命令或另存到实验目录，不放进正式 configs。
- 正式配置里尽量不重复写 `run.team`；默认 team 由 `task_config.py` 管理。只有临时对照实验才显式覆盖 team。

---

## 4. Benchmark 模块结构

当前 `src/lychee_mas/eval/benchmarks` 的职责划分如下：

```text
benchmarks/
├── __init__.py       只保留公开入口、注册表、LOADERS、PREPARERS
├── common.py         公共数据根目录、parquet 读取、ModelScope/HF 下载工具
├── gsm8k.py          GSM8K loader + prepare
├── aime_2024.py       AIME 2024 loader + prepare
├── choice_qa.py      ARC/OpenBookQA/MedQA 这类选择题 loader
├── locomo10.py       LoCoMo10 memory benchmark loader
├── human_eval.py     HumanEval loader + 代码评测
├── gaia.py           GAIA validation loader + 答案规整评分
├── aftraj.py         AFTraj-2K loader + prepare + audit scorer details
├── agent_collab.py   AgentCollabBench loader + prepare + IDR/RTD/CPR/CLC scorer details
├── mast_data.py      MAST-Data loader + prepare + taxonomy scorer details
└── open_agent_traces.py
                       Open Agent Traces loader + prepare + deviation scorer details
```

命名规则：

- 内部可运行 task、prepare target、loader key、模块名尽量一致。例如 `aime_2024` 对应 `aime_2024.py`，`locomo10` 对应 `locomo10.py`。
- 命名以效果和清晰度优先；如果上游/社区常用写法更清楚，内部名也可以跟随。例如 AIME 2024 统一用 `aime_2024`。
- 上游数据集自己的拼写不强行改。例如 ModelScope repo 是 `AI-ModelScope/AIME_2024`，HuggingFace repo 可以叫 `HuggingFaceH4/aime_2024`，LoCoMo 派生数据可以叫 `Percena/locomo-mc10`。
- 本地 cache 目录也尽量与清晰内部名或上游名一致。例如 AIME 当前准备后缓存到 `data/benchmarks/prepared/aime_2024/data`。
- 环境变量使用内部 canonical 名称，例如 `LYCHEE_LOCOMO10_MODELSCOPE_ID`；不再保留旧 alias。

`__init__.py` 对外暴露：

```python
from lychee_mas.eval.benchmarks import load, prepare, LOADERS, PREPARERS
```

使用方式：

```python
items = load("gsm8k", n=10)
prepare("human_eval", source="auto")
```

### 4.1 benchmark loader 是什么

benchmark loader 是 `src/lychee_mas/eval/benchmarks/*.py` 里负责“把原始数据变成 LycheeMAS 可运行样本”的函数。

它不负责调用模型，也不负责跑 agent。它只做这几件事：

```text
原始 benchmark 数据
  -> 读取本地缓存文件/parquet/json/jsonl
  -> 选择 split 或子集
  -> 把不同数据集自己的字段名规整成统一 record
  -> 生成给 agent 的 question/context
  -> 准备 gold，也就是 scorer 需要的标准答案或结构化标签
  -> 附上 metadata，保存 task_id、level、source、原始字段等辅助信息
```

例如：

```python
from lychee_mas.eval.benchmarks import load

items = load("gaia_validation_level_1", n=3)
```

这里 `load()` 会查 `LOADERS["gaia_validation_level_1"]`，实际调用 `gaia.py` 里的 loader。后面的 `scripts/run_mas.py --task gaia_validation_level_1` 也是走这个 loader。

### 4.2 每条样本的统一格式

所有 benchmark loader 都应该返回：

```python
{
    "task": "...",
    "kind": "...",
    "question": "...",
    "gold": ...,
    "context": None,
    "metadata": {...},
}
```

字段含义：

- `task`：benchmark 名称，如 `gsm8k`、`human_eval`、`gaia_validation_level_1`。
- `kind`：评分器类型，传给 `lychee_mas.eval.metrics.score(kind, pred, gold)`。
- `question`：给 agent 的主问题。
- `gold`：标准答案或官方评分所需对象。
- `context`：额外上下文，比如长对话历史、文件路径说明等。没有额外上下文时写 `None`，不要省略字段。
- `metadata`：样本 id、level、file name、source、原始字段摘要等辅助信息。理论上 Python dict 可以没有这个 key，所以以前写“可选”；但为了调试、复现和定位错误，LycheeMAS 新增/改造 benchmark 时建议总是提供 `metadata`，哪怕只是 `{}`。

更准确地说，推荐 record v2 兼容格式是：

```python
{
    # 必填：run_mas.py 和评分必须依赖
    "task": "gaia_validation_level_1",
    "kind": "gaia",
    "question": "What is ...?",
    "gold": "official answer or structured object",

    # 必填但可为 None：给 runtime 的额外上下文
    "context": None,

    # 强烈建议：不参与模型作答，但参与日志、case_id、调试、错误归因
    "metadata": {
        "source": "GAIA",
        "task_id": "...",
        "split": "validation",
        "level": 1,
        "file_name": "...",
    },

    # 可选扩展：只有部分 benchmark/runtime 需要
    "agent_specs": [...],   # AgentCollabBench 后续按样本动态建 team 时使用
    "topology": {...},      # AgentCollabBench 的拓扑信息
    "evaluator": {...},     # 官方 evaluator 或 metric 配置
}
```

为什么叫“兼容格式”：

- `task/kind/question/gold/context/metadata` 是所有 loader 应该尽量统一的核心字段。
- `agent_specs/topology/evaluator` 这类字段是扩展字段，当前 `run_mas.py` 普通静态 team 不一定使用，但先保留在 record 里，方便后续工具型 runtime 或 topology-aware runtime 使用。
- `metadata` 里可以放原始数据的 `raw` 摘要，但不要把超大原始文件完整塞进去；大对象会让 `outputs.jsonl` 变得很难读。

当前全部 `kind` 以 `src/lychee_mas/eval/metrics.py` 的 `score()` 分发为准：

```text
exact                     GSM8K 数字/短答
aime                      AIME 数学答案等价
mc                        多选题
f1                        LoCoMo/MemGAS 类 token-level F1
human_eval                HumanEval 代码执行检查
gaia                      GAIA 官方风格答案规整
mas_audit                 AFTraj safe/unsafe + mistake step/agent audit
mas_failure_taxonomy      MAST failure taxonomy label F1
mas_deviation             Open Agent Traces deviation/anomaly detection
mas_instruction_decay     AgentCollabBench IDR
mas_tracer_durability     AgentCollabBench RTD
mas_consensus_pollution   AgentCollabBench CPR
mas_context_leakage       AgentCollabBench CLC
```

### 4.3 task_config.py 的作用

`src/lychee_mas/eval/task_config.py` 是“runnable task 的默认运行策略表”，不是数据下载表，也不是 scorer 注册表。它主要管两件事：

```text
TASK_CONFIG[task]["team"]
  = 这个 task 默认用哪套 team profile
  = 在 src/lychee_mas/layers/construct/templates.py 的 TEAMS 里解析成角色、工具、group-chat 预设

TASK_CONFIG[task]["extractor"]
  = runtime 结束后，从 AutoGen messages 里抽 final_answer 的策略
  = default 找最后的 APPROVE: ...；boxed 用于 AIME，从 \boxed{...} 里取答案
```

二者区别：

| 字段 | 作用阶段 | 决定什么 | 示例 |
|---|---|---|---|
| `team` | 推理前 | 用哪套 agent/team/template，以及是否创建工具型 participant | `gaia` 会创建 FileSurfer/WebSurfer/Coder/ComputerTerminal；`reason` 是普通推理链 |
| `extractor` | 推理后、评分前 | 从 AutoGen 对话消息里取哪段作为 `final_answer` | `default` 找 `APPROVE: ...`；`boxed` 从 AIME 的 `\\boxed{...}` 里取答案 |

所以 `team` 影响“怎么推理”，`extractor` 影响“怎么把推理结果交给 scorer”。它们都不等于
`kind`：`kind` 是评分口径，决定 `metrics.score_details(kind, pred, gold)` 怎么判分。

当前常见默认映射：

```text
gsm8k -> team=reason, extractor=default
aime_2024 -> team=aime, extractor=boxed
arc_easy/openbookqa/medqa -> team=fact, extractor=default
locomo10 -> team=memory, extractor=default
human_eval -> team=human_eval, extractor=default
gaia_validation* -> team=gaia, extractor=default
aftraj/agent_collab/mast/open_agent_traces -> team=reason, extractor=default
```

它和 YAML 的关系：

```text
CLI --team
  优先级最高，用于临时覆盖

YAML run.team
  次优先级，只建议特殊对照实验使用

task_config.py
  默认来源；正式 benchmark 配置尽量依赖这里，避免每个 YAML 重复写 team
```

所以 `TASK_CONFIG` 和 YAML 不应该长期重复。当前正式 `human_eval.yaml`、`gaia.yaml`、`strict_mas_api.yaml`
都不写 `run.team`，运行时会根据 task 自动选默认 team。GAIA 本身就是工具型 benchmark，不再保留纯文本
baseline 配置。

代码入口对应关系：

```text
scripts/run_mas.py
  profile = CLI --team 或 YAML run.team 或 team_name_for_task(task)

src/lychee_mas/runtime/backends/autogen_runtime.py
  extractor_for_task(task) 从 messages 里提取 final_answer
```

### 4.4 Benchmark source / prepare target / runnable task / kind 的关系

后续最容易混淆的是：一个外部 benchmark 项目，在 LycheeMAS 里可能对应多个注册名。这不是重复，而是因为这里有四层概念：

```text
benchmark source
  = 外部数据集/项目来源，例如 AFTraj-2K、AgentCollabBench、GAIA

prepare target
  = 数据准备入口，负责下载/缓存同一份源数据
  = PREPARERS 里的 key，可被 scripts/prepare_benchmarks.py --tasks 使用
  = 注意：prepare target 不是评分 task；它只说明“准备哪份数据”

runnable task
  = 实际运行的评测切片
  = LOADERS 里的 key，可被 scripts/run_mas.py --task 使用

kind
  = 评分口径
  = 每条样本里的 item["kind"]，传给 metrics.score(kind, pred, gold)
```

推荐在文档、README、实验记录里用这张关系图展示：

```text
Benchmark source
└── Prepare target
    └── Runnable task / loader
        └── Kind / scorer
            └── Metrics in runs/.../metrics.json
```

具体到命令：

```text
scripts/prepare_benchmarks.py --tasks <prepare_target>
    只负责把数据下载/缓存到 raw/prepared benchmark roots，不代表一定能直接 run。

scripts/run_mas.py --task <runnable_task>
    负责选择 LOADERS[task]，把样本交给 runtime，再用 kind 评分。
```

更准确地说，`prepare_benchmarks.py --tasks` 接受的是 `prepare target`。有些
prepare target 是真正的数据源入口，有些只是为了方便而加的 alias。下载行为通常是
“准备这个 task 所属的源数据”，不一定只下载最终筛出来的 case。

| prepare target 类型 | 示例 | 它是什么 | 实际准备行为 |
|---|---|---|---|
| source alias / canonical source | `agent_collab` | 源数据总入口，通常存在于 `SOURCE_PREPARERS`。 | 下载/缓存 AgentCollabBench 源数据；它本身不一定是 runnable task。 |
| task alias | `agent_collab_idr` | 名字和 runnable task 相同，同时也被加入 `PREPARERS`，方便用户按 task 名 prepare。 | 仍然调用 `agent_collab` 的 preparer，准备同一份 AgentCollabBench 源数据；run 时 loader 才取 IDR 样本/评分口径。 |
| split/level alias | `gaia_validation_level_1` | 一种 task alias，表示同一 split 里的 level 子集。 | 调用 `gaia_validation` 的 preparer，准备 validation split；run 时 loader 才过滤 level 1。 |
| full dataset alias | `gaia` | 只用于准备上游完整数据集，不建议作为本地 scored task。 | 准备完整 GAIA snapshot，包含 validation/test 等；test 没公开 gold，本地 scored task 仍建议用 `gaia_validation*`。 |
| split alias | `aftraj_audit_test` | 一种 task alias，表示同一源数据里的某个 split。 | 调用 `aftraj` 的 preparer，准备 AFTraj 源数据；如果源数据里有 test split 文件，loader 才按 test split 过滤。 |

简单记：

```text
prepare target 解决“数据有没有准备好”
runnable task   解决“这次到底跑哪批 case、用哪个 loader”
kind            解决“模型输出怎么评分”
```

为什么一个 benchmark source 会拆成多个 task：

- 同一份数据可能有不同 split，例如 `aftraj_audit` 和 `aftraj_audit_test`。
- 同一份数据可能有不同 metric，例如 AgentCollabBench 的 IDR/RTD/CPR/CLC。
- 同一份数据可能既有 validation 又有 level 子集，例如 GAIA validation level 1/2/3。
- 同一份数据可能共享下载缓存，但不同 task 的 `question/gold/kind` 不同。

当前映射关系：

Markdown 原生表格不支持单元格合并；这里用 HTML table 的 `rowspan` 表达同一个 benchmark source 下多个 prepare target / runnable task / scorer 的对应关系。

<table>
  <thead>
    <tr>
      <th>Benchmark source</th>
      <th>Prepare target / alias</th>
      <th>Runnable task / loader</th>
      <th>Kind / scorer</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GSM8K</td>
      <td><code>gsm8k</code></td>
      <td><code>gsm8k</code></td>
      <td><code>exact</code></td>
      <td>数学短答，prepare target 与 runnable task 一一对应。</td>
    </tr>
    <tr>
      <td>AIME 2024</td>
      <td><code>aime_2024</code></td>
      <td><code>aime_2024</code></td>
      <td><code>aime</code></td>
      <td>数学竞赛题，正式评分需要完整数学解析依赖。</td>
    </tr>
    <tr>
      <td>HumanEval</td>
      <td><code>human_eval</code></td>
      <td><code>human_eval</code></td>
      <td><code>human_eval</code></td>
      <td>代码生成题，scorer 会执行官方 <code>check(candidate)</code>。</td>
    </tr>
    <tr>
      <td rowspan="5">GAIA</td>
      <td><code>gaia</code></td>
      <td>无本地 scored task</td>
      <td>无</td>
      <td>只表示完整 GAIA snapshot prepare；test split 没有公开 gold，不适合本地评分。</td>
    </tr>
    <tr>
      <td><code>gaia_validation</code></td>
      <td><code>gaia_validation</code></td>
      <td><code>gaia</code></td>
      <td>validation 全量，本地可评分。</td>
    </tr>
    <tr>
      <td><code>gaia_validation_level_1</code></td>
      <td><code>gaia_validation_level_1</code></td>
      <td><code>gaia</code></td>
      <td>共享 validation 数据，loader 过滤 level 1。</td>
    </tr>
    <tr>
      <td><code>gaia_validation_level_2</code></td>
      <td><code>gaia_validation_level_2</code></td>
      <td><code>gaia</code></td>
      <td>共享 validation 数据，loader 过滤 level 2。</td>
    </tr>
    <tr>
      <td><code>gaia_validation_level_3</code></td>
      <td><code>gaia_validation_level_3</code></td>
      <td><code>gaia</code></td>
      <td>共享 validation 数据，loader 过滤 level 3。</td>
    </tr>
    <tr>
      <td rowspan="3">AFTraj-2K</td>
      <td><code>aftraj</code></td>
      <td>无直接 runnable task</td>
      <td>无</td>
      <td>数据源总入口，准备 safe/unsafe trajectories 和 split 文件。</td>
    </tr>
    <tr>
      <td><code>aftraj_audit</code></td>
      <td><code>aftraj_audit</code></td>
      <td><code>mas_audit</code></td>
      <td>全量 audit task。</td>
    </tr>
    <tr>
      <td><code>aftraj_audit_test</code></td>
      <td><code>aftraj_audit_test</code></td>
      <td><code>mas_audit</code></td>
      <td>paper/test split；是否严格 test 取决于源数据里的 split 文件。</td>
    </tr>
    <tr>
      <td rowspan="5">AgentCollabBench</td>
      <td><code>agent_collab</code></td>
      <td>无直接 runnable task</td>
      <td>无</td>
      <td>数据源总入口，准备四类协作 metric 共享的 900-case 数据。</td>
    </tr>
    <tr>
      <td><code>agent_collab_idr</code></td>
      <td><code>agent_collab_idr</code></td>
      <td><code>mas_instruction_decay</code></td>
      <td>Instruction Decay，检查原始指令/约束是否逐步丢失或变形。</td>
    </tr>
    <tr>
      <td><code>agent_collab_rtd</code></td>
      <td><code>agent_collab_rtd</code></td>
      <td><code>mas_tracer_durability</code></td>
      <td>Tracer Durability，检查 tracer/约束是否跨 agent 路径保持有效。</td>
    </tr>
    <tr>
      <td><code>agent_collab_cpr</code></td>
      <td><code>agent_collab_cpr</code></td>
      <td><code>mas_consensus_pollution</code></td>
      <td>Consensus Pollution，检查团队共识是否被错误信息污染。</td>
    </tr>
    <tr>
      <td><code>agent_collab_clc</code></td>
      <td><code>agent_collab_clc</code></td>
      <td><code>mas_context_leakage</code></td>
      <td>Context Leakage，检查私有上下文是否泄漏。</td>
    </tr>
    <tr>
      <td rowspan="2">MAST-Data</td>
      <td><code>mast_data</code></td>
      <td>无直接 runnable task</td>
      <td>无</td>
      <td>数据源总入口，默认准备 human-labelled 数据。</td>
    </tr>
    <tr>
      <td><code>mast_failure</code></td>
      <td><code>mast_failure</code></td>
      <td><code>mas_failure_taxonomy</code></td>
      <td>失败 taxonomy 诊断，当前 scorer 使用标签 F1。</td>
    </tr>
    <tr>
      <td>Open Agent Traces</td>
      <td><code>open_agent_traces</code></td>
      <td><code>open_agent_traces</code></td>
      <td><code>mas_deviation</code></td>
      <td>按 run 聚合 events，做 deviation/anomaly 判断。</td>
    </tr>
    <tr>
      <td>ARC-Easy</td>
      <td><code>arc_easy</code></td>
      <td><code>arc_easy</code></td>
      <td><code>mc</code></td>
      <td>选择题；prepare 下载 <code>allenai/ai2_arc</code> 的 <code>ARC-Easy/test</code>。</td>
    </tr>
    <tr>
      <td>OpenBookQA</td>
      <td><code>openbookqa</code></td>
      <td><code>openbookqa</code></td>
      <td><code>mc</code></td>
      <td>选择题；prepare 下载 <code>allenai/openbookqa</code> 的 <code>main/test</code>。</td>
    </tr>
    <tr>
      <td>MedQA</td>
      <td><code>medqa</code></td>
      <td><code>medqa</code></td>
      <td><code>mc</code></td>
      <td>选择题；prepare 下载后标准化成 <code>$LYCHEE_BENCHMARK_PREPARED_ROOT/medqa/medqa.json</code>。</td>
    </tr>
    <tr>
      <td>LoCoMo</td>
      <td><code>locomo10</code></td>
      <td><code>locomo10</code></td>
      <td><code>f1</code></td>
      <td>长对话记忆；prepare 下载后缓存到 <code>$LYCHEE_BENCHMARK_PREPARED_ROOT/locomo10/locomo10.json</code>。</td>
    </tr>
  </tbody>
</table>

AgentCollabBench 四种 metric 的含义：

| task | kind | metric | 主要看什么 | 当前 LycheeMAS scorer 第一版怎么判 |
|---|---|---|---|---|
| `agent_collab_idr` | `mas_instruction_decay` | IDR / Instruction Decay Rate | 多 agent 协作中，原始指令/约束会不会在传递中逐渐丢失或变形。 | 看输出是否满足 expected/required，且没有泄露 forbidden。 |
| `agent_collab_rtd` | `mas_tracer_durability` | RTD / Tracer Durability | 注入的 tracer/约束/标记能否在多轮或多 agent 路径中保持有效。 | 看 required/tracer 相关内容是否仍出现在最终结果中。 |
| `agent_collab_cpr` | `mas_consensus_pollution` | CPR / Consensus Pollution Rate | 团队共识是否被错误信息、污染信息或错误中间结论带偏。 | 看最终结果是否仍满足 expected，不被 forbidden 污染。 |
| `agent_collab_clc` | `mas_context_leakage` | CLC / Context Leakage Check | 私有上下文、注入隐私或不该传播的信息是否泄漏到最终输出。 | 只要出现 forbidden/private terms 就失败；没有泄漏且其它要求满足则通过。 |

是否要整合成一个 AgentCollabBench task：

- 推荐“运行时分开，展示时聚合”。
- 分开跑的原因：四个 metric 的失败模式不同，`kind` 和 `score_details` 不同，混在一个 `agent_collab` 平均分里会看不出是哪类协作问题。
- 展示时可以再做一个总览表，例如 `AgentCollabBench_avg = mean(IDR, RTD, CPR, CLC)`，但原始结果目录仍保留四个 task，便于错误归因。

当前 `scripts/prepare_benchmarks.py` 已经支持五种列表视角。注意：所有
`--list*` 参数都只打印清单，不会下载数据；真正下载/准备数据的是：

```bash
python scripts/prepare_benchmarks.py --tasks <prepare_target>
```

这里建议统一使用三个词：

- `source`：canonical 数据源，例如 `gaia`、`agent_collab`、`mast_data`。
- `prepare target`：`prepare_benchmarks.py --tasks` 接受的可准备目标名，可能是 source，也可能是别名。
- `runnable task`：`run_mas.py --task` 接受的可运行评测任务。

```text
--list-benchmark-sources
                 只显示 Benchmark source，例如 AFTraj-2K、AgentCollabBench、GAIA。
--list-prepare-targets
                 显示 PREPARERS 的全部 key，也就是 prepare_benchmarks.py --tasks 能接受的名字；
                 更准确地说，它列的是“可准备/可下载的数据目标名”；
                 它包含 canonical benchmark source，也包含 prepare target / task alias；
                 它不是纯 benchmark source 列表，也不是纯 runnable task 列表。
--list-runnable-tasks
                 只显示可运行 task / loader：aftraj_audit_test, agent_collab_clc, ...
--list-benchmark-structure
                 按 Benchmark source 展示 full prepare target，以及
                 prepare target -> runnable task(s) -> kind/scorer(s) 的逐行映射。
                 如果某个 prepare target 是别名，会显示 alias -> 实际 prepare target。
--list-download-sources
                 按 Benchmark source 展示 ModelScope / HuggingFace / 其它默认候选、
                 环境变量覆盖名，以及 direct-file fallback 文件或匹配规则。
```

因此推荐展示时这样说：

```text
想看有哪些 Benchmark source，看 --list-benchmark-sources。
想知道 prepare_benchmarks.py --tasks 能接受哪些“可准备目标名”，看 --list-prepare-targets。
想知道 run_mas.py --task 能填什么，看 --list-runnable-tasks。
想看整体对应关系，看 --list-benchmark-structure。
想看每个 benchmark 默认从哪里下载、还有哪些兜底候选，看 --list-download-sources。
想真正下载/准备数据，执行 prepare_benchmarks.py --tasks <prepare_target>。
```

例如 GAIA 会显示成：

```text
Benchmark source: GAIA
  full prepare target: gaia
  prepare target -> runnable task(s) -> kind/scorer(s)
    gaia -> - -> - [full dataset prepare only]
    gaia_validation -> gaia_validation -> gaia
    gaia_validation_level_1 (alias -> gaia_validation) -> gaia_validation_level_1 -> gaia
    gaia_validation_level_2 (alias -> gaia_validation) -> gaia_validation_level_2 -> gaia
    gaia_validation_level_3 (alias -> gaia_validation) -> gaia_validation_level_3 -> gaia
```

这里的意思是：`gaia` 只用于准备完整数据，不是本地可评分 runnable task；
`gaia_validation_level_1` 作为 prepare target 时实际复用 `gaia_validation` 的数据准备逻辑，
但运行时仍然是独立的 `run_mas.py --task gaia_validation_level_1`，评分 kind 是 `gaia`。

---

## 5. 数据准备

### 5.1 数据根目录

benchmark 数据分成两类目录：

```text
data/benchmarks/raw
  从 HuggingFace / ModelScope / URL 下载来的上游原始数据，尽量保持官方形态。

data/benchmarks/prepared
  LycheeMAS loader 可直接读取的准备后数据，例如 parquet、标准化后的 JSON。
```

运行结果不是数据集本身，统一推荐放：

```text
runs/benchmarks
```

路径由这些新环境变量控制：

```bash
export LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT
export LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT
export LYCHEE_BENCHMARK_RUNS_ROOT=$LYCHEE_BENCHMARK_RUNS_ROOT
```

如果不设置，代码默认使用仓库相对路径：

```text
data/benchmarks/raw
data/benchmarks/prepared
runs/benchmarks
```

兼容规则：

- `CDM_DATA_ROOT` 仍兼容；如果没有显式设置 `LYCHEE_BENCHMARK_RAW_ROOT`，它会作为 raw root。
- 如果没有显式设置 `LYCHEE_BENCHMARK_PREPARED_ROOT`，prepared root 会回落到 `CDM_DATA_ROOT`；这是为了兼容原 LycheeMAS 里“数据走 CDM_DATA_ROOT”的运行习惯。
- `CDM_PROCESSED_ROOT` 和 `--data-root` 不再作为公开兼容入口保留；它们属于 benchmark 接入过程中的中间命名。
- 新命令、新文档、新配置统一使用 `LYCHEE_BENCHMARK_RAW_ROOT` / `LYCHEE_BENCHMARK_PREPARED_ROOT` / `LYCHEE_BENCHMARK_RUNS_ROOT`。

注意：`data/benchmarks/*` 和 `runs/benchmarks/*` 都不应该提交到 git。

目录约定：

```text
$LYCHEE_BENCHMARK_RAW_ROOT/<benchmark>/<provider>/<source_slug>/...
$LYCHEE_BENCHMARK_PREPARED_ROOT/<benchmark>/...
```

其中 `<source_slug>` 会把上游 ID 里的 `/` 转成 `--`，例如 `OmniData/ARC` 会保存为
`OmniData--ARC`。这样同一个 benchmark 如果以后有多个 ModelScope/HuggingFace 源，也能保留各自原始文件，不会互相覆盖。

当前每类 benchmark 的存放原则：

| Benchmark | Raw root 下保存 | Prepared root 下保存 |
|---|---|---|
| `gsm8k` | `gsm8k/<provider>/<source_slug>/...`，保存上游 dataset snapshot | `gsm8k/main/*.parquet` |
| `aime_2024` | `aime_2024/<provider>/<source_slug>/...`，保存上游 dataset snapshot | `aime_2024/data/*.parquet` |
| `arc_easy` | `arc_easy/modelscope/OmniData--ARC/raw/ARC-V1-Feb2018.zip` 或 HF snapshot | `arc_easy/*.parquet` |
| `openbookqa` | `openbookqa/<provider>/<source_slug>/...`，保存上游 dataset snapshot | `openbookqa/*.parquet` |
| `medqa` | `medqa/<provider>/<source_slug>/...`，zip/source snapshot 都保留 | `medqa/medqa.json` |
| `locomo10` | `locomo10/<provider>/<source_slug>/...`，保存 `locomo10.json` 或 snapshot | `locomo10/locomo10.json` |
| `human_eval` | `human_eval/<provider>/<source_slug>/HumanEval.jsonl.gz` | `human_eval/HumanEval.jsonl.gz` |
| `gaia` | `gaia/<provider>/<source_slug>/2023/...`、附件和 metadata | `gaia/...`，从 raw source 复制得到的实体目录 |
| `aftraj` | `aftraj/<provider>/<source_slug>/*.parquet`、`splits_test.json` | `aftraj/...`，从 raw source 复制得到的实体目录 |
| `agent_collab` | `agent_collab/<provider>/<source_slug>/...` | `agent_collab/...`，从 raw source 复制得到的实体目录 |
| `mast_data` | `mast_data/<provider>/<source_slug>/MAD_*.json` | `mast_data/...`，从 raw source 复制得到的实体目录 |
| `open_agent_traces` | `open_agent_traces/<provider>/<source_slug>/data/**/*.parquet` | `open_agent_traces/...`，从 raw source 复制得到的实体目录 |

因此，“raw 文件会不会保存”的答案是：现在 benchmark 下载器会尽量把上游源数据显式保存在项目 raw 里；
需要转换的 benchmark 会再写一份 loader 可直接读取的 prepared。对于 GAIA、AFTraj、AgentCollabBench、MAST、
Open Agent Traces 这类“上游目录本身就是 loader 输入”的数据，也会把 raw source 复制到 prepared，
保证 loader 读到的是 prepared 下面的真实文件/目录。

ARC 的转换过程可能会使用临时解压目录；正式保留的源文件以 raw root 下的内容为准。
现在 `OmniData/ARC` 的 `ARC-V1-Feb2018.zip` 会保存在：

```text
$LYCHEE_BENCHMARK_RAW_ROOT/arc_easy/modelscope/OmniData--ARC/raw/ARC-V1-Feb2018.zip
```

转换后的评测数据保存在：

```text
$LYCHEE_BENCHMARK_PREPARED_ROOT/arc_easy/test-00000-of-00001.parquet
```

### 5.2 准备脚本

查看 canonical 数据源准备入口：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py --list-benchmark-sources
```

查看可运行评测任务，也就是 `scripts/run_mas.py --task` 可用值：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py --list-runnable-tasks
```

查看所有可传给 `prepare_benchmarks.py --tasks` 的“可准备目标名”，包括 source 和 task alias：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py --list-prepare-targets
```

查看整体对应关系：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py --list-benchmark-structure
```

查看下载源 catalog，包括两个平台、其它默认候选和 direct-file fallback：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py --list-download-sources
```

准备所有 benchmark source 的全量/canonical 数据：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --all-full-benchmarks
```

`--all-full-benchmarks` 等价于准备每个 Benchmark source 的 `full prepare target`。其中 GAIA 会准备完整
snapshot；MAST 会自动设置 `LYCHEE_MAST_DOWNLOAD_FULL=1`，准备 full dataset。

准备常用小数据：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --tasks gsm8k,aime_2024,human_eval
```

准备完整 GAIA 数据集：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --tasks gaia
```

只准备 GAIA validation split：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --tasks gaia_validation
```

只用 ModelScope：

```bash
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source modelscope \
  --tasks gsm8k,aime_2024
```

只用 HuggingFace：

```bash
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source huggingface \
  --tasks human_eval
```

强制重新下载：

```bash
$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --force \
  --tasks human_eval
```

#### prepare 阶段的完整性检查与 raw -> prepared 恢复

现在 prepare 阶段分两层判断：

1. 先检查 `prepared` 是否已经 ready。ready 不是只看文件存在，而是尽量读取 parquet / JSON / JSONL / gzip，并检查必要字段或必要文件。
2. 如果 `prepared` 不 ready，部分 benchmark 会先检查 `raw` 里是否已经有完整上游数据；如果 raw 可用，就直接从 raw 复制或转换出 prepared，不再重新下载。

这解决的就是“raw 里有，但 prepared 被删了/坏了”的情况。注意：`--force` 表示强制重新准备，通常会跳过这层普通缓存复用逻辑。

| Benchmark / prepare target | prepared ready 检查 | raw source 检查与恢复 | 备注 |
|---|---|---|---|
| `gsm8k` | `train/test` parquet 可读，包含 `question/answer`。 | 保存上游 snapshot 到 raw；prepared 缺失时会重新走 provider prepare，底层下载器可复用本地/全局缓存。 | 尚未单独实现“只从 raw snapshot 离线重建 parquet”的 adapter。 |
| `aime_2024` | `train` parquet 可读，包含 `problem/answer`。 | 保存上游 snapshot 到 raw；prepared 缺失时会重新走 provider prepare，底层下载器可复用本地/全局缓存。 | AIME 需要正式数学解析依赖，缺依赖应失败。 |
| `arc_easy` | `test` parquet 可读，包含 `question/choices/answerKey`。 | 如果 `raw/arc_easy/modelscope/OmniData--ARC/raw/ARC-V1-Feb2018.zip` 已存在，会直接从 zip 抽取 ARC-Easy-Test 并重建 prepared。 | 其它 HF-style source 仍走 provider prepare。 |
| `openbookqa` | `test` parquet 可读，包含 `question_stem/choices/answerKey`。 | 保存上游 snapshot 到 raw；prepared 缺失时会重新走 provider prepare，底层下载器可复用本地/全局缓存。 | 尚未单独实现 raw snapshot 离线转换 adapter。 |
| `medqa` | `prepared/medqa/medqa.json` 可解析且非空。 | 对 `AI-ModelScope/med_qa`、`cloakone/MedQA` 这类 zip source，如果 raw zip 已存在，会直接标准化成 prepared JSON。 | 不同 source schema 不同，所以有 source-specific converter。 |
| `locomo10` | `prepared/locomo10/locomo10.json` 可解析且非空。 | 会扫描 raw 里的 `locomo10.json`，可用则复制到 prepared。 | 支持 ModelScope / HuggingFace direct file / GitHub raw fallback。 |
| `human_eval` | `HumanEval.jsonl.gz` 可解压、可读 JSONL，包含 `task_id/prompt/test/entry_point`。 | 会扫描 raw 里的 `HumanEval.jsonl.gz`，可用则复制到 prepared。 | raw 可以来自 ModelScope、HuggingFace 或官方 GitHub raw。 |
| `gaia` / `gaia_validation` | validation/test 的 `metadata.jsonl` 或 `metadata.parquet` 可读取。 | raw 里对应 split metadata 可读时，整份 raw source 复制到 prepared。 | `gaia_validation` 只要求 validation；`gaia` 要求 validation + test。 |
| `aftraj` | `aftraj_safe.parquet`、`aftraj_unsafe.parquet`、`splits_test.json` 都存在且非空。 | raw source 具备这些文件时，整份复制到 prepared。 | prepared 是真实目录复制，不使用 symlink。 |
| `agent_collab` | `data/train.jsonl` 存在，或能找到 `TASK-*.json`。 | raw source 具备这些文件时，整份复制到 prepared。 | 四个 runnable task 共用同一份 source。 |
| `mast_data` | 能找到并解析 `MAD_*dataset.json`。 | raw source 具备可解析 JSON 时，整份复制到 prepared。 | 默认 human-labelled；全量由 `LYCHEE_MAST_DOWNLOAD_FULL=1` 或 `--all-full-benchmarks` 触发。 |
| `open_agent_traces` | `prepared/open_agent_traces/data/**/*.parquet` 至少存在一个。 | raw source 具备 parquet trace 文件时，整份复制到 prepared。 | loader 后续还会实际读取 parquet。 |

终端上如果看到类似下面的输出，说明不是重新下载，而是从 raw 恢复了 prepared：

```text
[prepare] gaia: restored prepared from raw provider=modelscope id=gaia-benchmark/GAIA -> $LYCHEE_BENCHMARK_PREPARED_ROOT/gaia
```

### 5.3 推荐数据位置

正式推荐位置是：

```text
$LYCHEE_BENCHMARK_PREPARED_ROOT/gsm8k/main/train-00000-of-00001.parquet
$LYCHEE_BENCHMARK_PREPARED_ROOT/gsm8k/main/test-00000-of-00001.parquet
$LYCHEE_BENCHMARK_PREPARED_ROOT/aime_2024/data/train-00000-of-00001.parquet
$LYCHEE_BENCHMARK_PREPARED_ROOT/human_eval/HumanEval.jsonl.gz
$LYCHEE_BENCHMARK_PREPARED_ROOT/gaia/2023/validation/metadata.parquet
$LYCHEE_BENCHMARK_PREPARED_ROOT/gaia/2023/test/metadata.parquet
```

对应的 raw source 会按 provider/source_id 保存，例如：

```text
$LYCHEE_BENCHMARK_RAW_ROOT/human_eval/huggingface/openai--openai_humaneval/HumanEval.jsonl.gz
$LYCHEE_BENCHMARK_RAW_ROOT/gaia/modelscope/gaia-benchmark--GAIA/2023/validation/metadata.parquet
$LYCHEE_BENCHMARK_RAW_ROOT/arc_easy/modelscope/OmniData--ARC/raw/ARC-V1-Feb2018.zip
```

`gaia` 和 `gaia_validation` 的含义不同：

- `gaia`：完整 GAIA 数据集 prepare，会下载 repository snapshot，包括 validation 和 test。
- `gaia_validation`：只准备本地可评分的 validation split。

真正跑分任务仍建议写成 `gaia_validation` 或 `gaia_validation_level_1/2/3`。原因是 GAIA test split 没有公开 gold answer，适合 leaderboard submission，不适合 LycheeMAS 本地 `gold` 评分。

### 5.4 数据来源原则

不要从其他人的工作区复制数据。

原因：

1. 那是别人的工作区，不是当前项目可复现的数据来源。
2. 后续换运行环境/换用户会失效。
3. benchmark 交接应该依赖明确的数据准备命令，而不是隐式路径。

正确做法是：

- 小数据集：通过 `scripts/prepare_benchmarks.py` 自动下载。
- 大数据集：保留 prepare 脚本入口，必要时配合网络配置、镜像源或 HF token。
- 私有或授权数据：在文档里说明获取方式，不把 token 和数据本体提交进 git。

当前 `--source auto` 的 provider 选择：

- 已确认有 ModelScope 常用入口的任务会优先试 ModelScope，例如 `gsm8k`、`aime_2024`、`arc_easy`、`openbookqa`、`medqa`、`locomo10`、`human_eval`、`gaia`。
- 当前没有确认可用 ModelScope 镜像或没有兼容 schema 的任务会直接走 HuggingFace，例如 `aftraj`、`agent_collab`、`mast_data`、`open_agent_traces`。
- 如果后续找到了私有或组织内 ModelScope 镜像，可以设置环境变量覆盖，例如 `LYCHEE_AFTRAJ_MODELSCOPE_ID`、
  `LYCHEE_AGENTCOLLAB_MODELSCOPE_ID`，再用 `--source modelscope` 或 `--source auto`。
- prepare 日志会打印形如 `[download] <benchmark>: provider=<modelscope|huggingface> id=<repo_id> -> <path>` 的行，
  用来确认实际下载渠道。如果只有 `[prepare] ... ready at ...` 而没有 `[download]` 行，通常说明本地缓存已经存在，没有重新下载。

当前默认下载源候选（已按 ModelScope/HuggingFace API、文件树和独立数据目录 prepare 验证清理）：

| Benchmark source | ModelScope 默认候选 | HuggingFace 默认候选 | 其它默认候选 | 说明 |
|---|---|---|---|---|
| GSM8K | `AI-ModelScope/gsm8k`, `modelscope/gsm8k` | `openai/gsm8k`, `gsm8k` | - | 不再默认试 404 的 `openai/gsm8k` ModelScope ID。 |
| AIME 2024 | `AI-ModelScope/AIME_2024`, `HuggingFaceH4/aime_2024` | `HuggingFaceH4/aime_2024`, `Maxwell-Jia/AIME_2024`, `AI-MO/aimo-validation-aime` | - | 已验证 `AI-ModelScope/AIME_2024` 可 prepare；不是 `AI-ModelScope/aime_2024`。内部 task/module 仍用 `aime_2024`。 |
| ARC-Easy | `OmniData/ARC` | `allenai/ai2_arc`, `ai2_arc` | `modelscope/ai2_arc` | `OmniData/ARC` 是 AI2 ARC raw zip，当前 adapter 抽取 `ARC-Easy-Test.jsonl` 转 parquet；`modelscope/ai2_arc` 也存在但 prepare 验证超时，不作为默认源。 |
| OpenBookQA | `allenai/openbookqa` | `allenai/openbookqa`, `openbookqa` | - | 已用临时 data-root 验证可下载并转成当前 loader 需要的 parquet。 |
| MedQA | `AI-ModelScope/med_qa` | `GBaker/MedQA-USMLE-4-options`, `GBaker/MedQA-USMLE-4-options-hf` | `cloakone/MedQA` | `AI-ModelScope/med_qa` 含 US MedQA JSONL，已验证可抽取 `data_clean/questions/US/test.jsonl`；`cloakone/MedQA` 是 processed alpaca 风格且样本为中文，不作为默认 fallback。 |
| HumanEval | `modelscope/humaneval`, `opencompass/humaneval` | `openai/openai_humaneval` | GitHub raw `HumanEval.jsonl.gz` | 不再试旧的 `openai/openai_humaneval` ModelScope ID；GitHub raw 是两个平台失败后的最后兜底。 |
| GAIA | `gaia-benchmark/GAIA`, `AI-ModelScope/GAIA` | `gaia-benchmark/GAIA` | - | `modelscope/GAIA` 已确认 404，已移除默认候选。 |
| LoCoMo10 | `evalscope/locomo` | `Percena/locomo-mc10` | GitHub raw `snap-research/locomo/data/locomo10.json` | `evalscope/locomo` 直接提供 `locomo10.json`，已验证可 prepare；`MTEB/LoCoMo` 是检索 parquet 结构，不适配当前 LoCoMo10 QA loader。 |
| AFTraj-2K | 仅环境变量 `LYCHEE_AFTRAJ_MODELSCOPE_ID` | `ZBox008003/AFTraj` | - | 默认走 HuggingFace。 |
| AgentCollabBench | 仅环境变量 `LYCHEE_AGENTCOLLAB_MODELSCOPE_ID` | `AgentCollabBench/AgentCollabBench` | - | 默认走 HuggingFace。 |
| MAST-Data | 仅环境变量 `LYCHEE_MAST_MODELSCOPE_ID` | `mcemri/MAST-Data` | - | 默认走 HuggingFace；full dataset 受 `LYCHEE_MAST_DOWNLOAD_FULL` 控制。 |
| Open Agent Traces | 仅环境变量 `LYCHEE_OPENAGENTTRACES_MODELSCOPE_ID` | `juliensimon/open-agent-traces` | - | 默认走 HuggingFace。 |

这张表来自 `src/lychee_mas/eval/benchmarks/source_catalog.json`。如果后续发现新的镜像源或官方备份源，优先改 catalog，再让具体 benchmark 下载代码从 catalog 读取；不要把新 ID 只散落写在某个 loader 里。

同名/近名数据源不一定等价。当前处理原则是：只有源存在、文件结构能被 adapter 转换、并且 prepare 后通过 ready/schema 检查，才放进默认候选。只“搜得到”的源先放到其它候选或文档说明。例如 `modelscope/ai2_arc` 是 ARC 相关源但 prepare 超时，`cloakone/MedQA` 是 MedQA 变体但 schema/language 和默认 US MedQA 不同，`MTEB/LoCoMo` 是 retrieval 数据而不是当前 LoCoMo10 QA 数据。

缓存完整性检查口径：

| Benchmark | prepare 返回 ready 前会检查什么 |
|---|---|
| GSM8K | `train/test` parquet 都能读取，且包含 `question/answer` 字段。 |
| AIME 2024 | `train` parquet 能读取，且包含 `problem/answer` 字段。 |
| ARC-Easy | `test` parquet 能读取，且包含 `question/choices/answerKey` 字段。 |
| OpenBookQA | `test` parquet 能读取，且包含 `question_stem/choices/answerKey` 字段。 |
| MedQA | `medqa.json` 能解析且非空。 |
| HumanEval | `HumanEval.jsonl.gz` 能解压并解析，且样本包含 `task_id/prompt/test/entry_point`。 |
| GAIA | `metadata.jsonl` 或 `metadata.parquet` 能读取且非空。 |
| LoCoMo10 | `locomo10.json` 能解析且非空。 |
| AFTraj-2K | `aftraj_safe.parquet`、`aftraj_unsafe.parquet`、`splits_test.json` 都存在且非空。 |
| AgentCollabBench | 存在 `data/train.jsonl` 或 `TASK-*.json`。 |
| MAST-Data | 至少一个 `MAD_*dataset.json` 能解析且非空。 |
| Open Agent Traces | `data/**/*.parquet` 至少存在一个。 |

这不是完整的 checksum 校验，也不会保证第三方发布者的数据和原始论文版本逐字一致；它保证的是“本地缓存不是空壳/坏文件，并且能转换成当前 loader 需要的字段”。如果要做正式可复现实验，后续可以再给每个 source 加 revision pin 和文件 hash。

### 5.5 下载实现里的几个细节

#### 为什么有 source_catalog 之后每个 benchmark 还有 provider 下载函数

`source_catalog.json` 只负责回答“从哪里下载”：

- ModelScope/HuggingFace 的候选 dataset ID。
- 环境变量覆盖名。
- 其它默认候选，例如 GitHub raw。
- direct-file fallback 需要哪些文件或匹配模式。

各 benchmark 文件里的 `_download_modelscope()`、`_download_huggingface()` 仍然有必要，因为它们负责回答“下载后怎么变成当前 loader 能读的格式”：

- 有的 benchmark 要 `datasets.load_dataset(repo, subset, split)`，例如 GSM8K、AIME。
- 有的 benchmark 虽然也是公开数据集，但默认源是 zip/raw 文件，需要抽取特定 split 再转成统一 parquet/JSON，例如 ARC-Easy、MedQA。
- 有的要 snapshot 后复制附件、metadata 或 parquet，例如 GAIA、AFTraj、AgentCollabBench。
- 有的只需要单个 JSON/GZ 文件，并且要放到固定 cache 文件名，例如 HumanEval、LoCoMo10。
- 有的要把不同发布者的字段规整成统一 `{task, kind, question, gold, context, metadata}` record。

所以这两层不是冗余：catalog 统一 source，provider 下载函数保留 benchmark-specific adapter。后续可以继续把重复的 snapshot/download 机械逻辑抽到 `common.py`，但不要把 schema 转换硬塞进一个通用下载器。

当前转换策略不是“每个平台每个源都复制一份”，也不是“一个万能转换器”。实际分三层：

| Benchmark | 转换策略 |
|---|---|
| GSM8K | 通用 `save_modelscope_dataset()` / `save_hf_dataset()`，要求输出 parquet 包含 `question/answer`。 |
| AIME 2024 | ModelScope/HF 共用 `_standardize_columns()`，只要源能转出 `problem/answer` 就接受；内部名和 task 统一为 `aime_2024`。 |
| ARC-Easy | HF 走通用 `datasets.load_dataset()`；ModelScope 默认 `OmniData/ARC` 是 raw zip，单独抽取 `ARC-Easy-Test.jsonl` 后转 parquet。 |
| OpenBookQA | ModelScope/HF 共用通用 dataset 转 parquet，要求 `question_stem/choices/answerKey`。 |
| MedQA | 默认 `AI-ModelScope/med_qa` 走 zip 源适配，抽取 `data_clean/questions/US/test.jsonl`；HF 源走 `datasets.load_dataset()`；最后都进入 `_standardize_medqa_row()`。 |
| LoCoMo10 | ModelScope/HF/GitHub raw 最终都落成 `locomo10.json`；loader 再兼容 original conversation schema 和 MC10 schema。 |
| HumanEval | ModelScope/HF/GitHub raw 最终都落成 `HumanEval.jsonl.gz`，再做 HumanEval 字段检查。 |
| GAIA | Snapshot 复制 metadata 和附件，保持 GAIA 官方目录结构；不是通用 parquet 转换。 |
| AFTraj / AgentCollab / MAST / Open Agent Traces | 每个 benchmark 有自己的文件清单、字段规整和 scorer gold 结构，不能用一个通用转换器代替。 |

#### HuggingFace direct file fallback 是什么意思

很多数据准备函数优先使用 HuggingFace / ModelScope 的 `snapshot_download()`。它的好处是能按仓库快照下载，坏处是：

- 网络差时可能因为锁、metadata、仓库扫描而卡很久。
- 有些小数据集只需要 1-2 个文件，拉 snapshot 反而重。
- 部分环境里 `snapshot_download()` 失败，但直接访问某个文件 URL 还能成功。

所以 `common.py` 增加了三个 fallback helper：

```python
list_hf_dataset_files(repo_id)
download_hf_files(repo_id, files, out_dir)
download_url(url, dest)
```

意思是：如果 `snapshot_download()` 不好用，代码可以退一步，先列出 HuggingFace dataset repo 里的文件，再只下载当前 benchmark 真正需要的几个文件。比如 MAST 默认只需要 `MAD_human_labelled_dataset.json`，AgentCollabBench 只需要 `TASK-*.json` 或 `data/**`。

这不是换数据源，也不是从别人目录复制；仍然是从项目声明的数据源下载，只是下载方式更轻。

它是否“稳”取决于 fallback 的类型：

- **严格文件清单 fallback**：例如 AFTraj 的 `aftraj_safe.parquet`、`aftraj_unsafe.parquet`、`splits_test.json`，以及 LoCoMo10 的 `raw/locomo10.json`。这些文件就是当前 loader 的必需输入，下载后还会做 ready 检查，风险较低。
- **显式子集 fallback**：例如 MAST 默认只下 `MAD_human_labelled_dataset.json`。这不是漏文件，而是当前 `mast_failure` scorer 只使用人工标注子集；如果要全量数据，必须显式设置 `LYCHEE_MAST_DOWNLOAD_FULL=1` 或用 `--all-full-benchmarks`。
- **按模式列文件 fallback**：例如 AgentCollabBench 的 `TASK-*.json` / `data/**`、Open Agent Traces 的 `data/**/*.parquet`。这比固定文件清单更灵活，但确实有“仓库结构变化导致匹配不全”的风险，所以下载后还会检查是否存在 task json、train.jsonl 或 parquet。

因此 direct file fallback 不能无条件视为完整 snapshot 的等价替代。当前原则是：

1. 能 snapshot 就优先 snapshot。
2. direct fallback 只下载当前 loader 明确需要的文件。
3. fallback 文件清单和匹配模式集中写在 `source_catalog.json`，便于 review。
4. prepare 返回 ready 前必须做最小完整性检查。
5. 正式可复现实验如果要求逐字固定数据版本，需要继续补 revision pin 和文件 hash。

断点续传口径：

- `huggingface_hub.snapshot_download()` 和 ModelScope snapshot download 本身有本地 cache/续传能力。
- `datasets.load_dataset()` 也会使用本地缓存，重复运行一般不会从零下载。
- LycheeMAS 自己的 `download_url()` 现在保留 `<目标文件>.part`，下一次请求时会用 HTTP Range 尝试续传；如果运行环境不支持 Range，会自动从头覆盖这个 `.part`。

#### MAST full dataset 开关是什么意思

MAST-Data 仓库里有两个容易混淆的数据文件：

```text
MAD_human_labelled_dataset.json   # 默认下载；人工标注子集；当前 mast_failure scorer 可直接使用
MAD_full_dataset.json             # 更大的全量集合；覆盖更广，但不一定每条都适合当前本地 gold scorer
```

当前 `mast_failure` 评测需要 gold taxonomy labels，所以默认选择
`MAD_human_labelled_dataset.json`。这不是“只为了 smoke 偷懒”，而是当前
loader/scorer 最直接可评分的数据版本。

`MAD_full_dataset.json` 更适合做数据分析、扩展样本覆盖、或者后续重新定义
loader/scorer。如果要把 full dataset 也纳入正式评测，需要先确认每条样本是否有
当前 scorer 需要的人工标签；否则可能会出现“数据更多，但 gold 不完整/不可比”的问题。

代码层面还要注意：`mast_data.py::_dataset_path()` 的优先级是：

```text
MAD_human_labelled_dataset.json -> MAD_full_dataset.json -> MAD_annotated_dataset.json
```

也就是说，即使同时下载了 human-labelled 和 full，当前 loader 仍会优先读取
human-labelled。若以后真的要单独跑 full，应新增一个明确的 runnable task，
例如 `mast_failure_full`，而不是隐式改变 `mast_failure` 的数据含义。

当前 `mast_data.py` 默认只准备 `MAD_human_labelled_dataset.json`。只有显式设置：

```bash
export LYCHEE_MAST_DOWNLOAD_FULL=1
```

再执行 prepare 时，才会额外下载 `MAD_full_dataset.json`。

推荐：

- 当前 `mast_failure` 正式/部分/全量都使用 human-labelled 数据。
- 如果研究问题需要 full dataset，先补 `mast_failure_full` loader/task，再设置 `LYCHEE_MAST_DOWNLOAD_FULL=1` prepare。

## 6. 各 benchmark 的具体实现

### 6.1 HumanEval 当前实现

文件：

```text
src/lychee_mas/eval/benchmarks/human_eval.py
```

数据缓存：

```text
$LYCHEE_BENCHMARK_PREPARED_ROOT/human_eval/HumanEval.jsonl.gz
$LYCHEE_BENCHMARK_RAW_ROOT/human_eval/<provider>/<source_slug>/HumanEval.jsonl.gz
```

prepare 逻辑：

1. `source=auto` 会先试已知 ModelScope 入口：`modelscope/humaneval`，再试 `opencompass/humaneval`。
2. ModelScope 不可用时试 HuggingFace：`openai/openai_humaneval`。
3. HumanEval 还支持 GitHub 原始 `HumanEval.jsonl.gz` 兜底。
4. 无论来自哪个 provider，都会先写入 raw source 目录，再复制成 prepared 下统一文件名供 loader 读取。

loader 返回：

```python
{
    "task": "human_eval",
    "kind": "human_eval",
    "question": "Complete the following Python function...",
    "gold": {
        "task_id": "...",
        "entry_point": "...",
        "test": "...",
    },
    "context": None,
    "metadata": {...},
}
```

评分：

```python
lychee_mas.eval.metrics.score("human_eval", pred, gold)
```

内部会从模型输出里提取 Python 代码，然后在子进程中执行官方 `check(candidate)`。

当前限制：

- 这不是 Docker 沙盒，只是 subprocess 隔离。
- 对恶意代码不够安全。
- 如果要做正式可信评测，建议后续改成容器沙盒或复用 HumanEval 官方更严格的 execution harness。

---

### 6.2 GAIA 当前实现

文件：

```text
src/lychee_mas/eval/benchmarks/gaia.py
```

当前注册的任务：

```text
gaia_validation
gaia_validation_level_1
gaia_validation_level_2
gaia_validation_level_3
```

数据准备额外支持：

```text
gaia                  # 完整 GAIA 数据集 prepare，不建议作为 run_mas task
gaia_validation       # validation split prepare + run_mas task
```

数据缓存：

```text
$LYCHEE_BENCHMARK_RAW_ROOT/gaia/<provider>/<source_slug>/2023/...
$LYCHEE_BENCHMARK_PREPARED_ROOT/gaia/2023/...
```

GAIA 上游目录本身就是 loader 输入，但 prepared 仍然会保存一份从 raw source 复制出来的实体目录；loader 只读 prepared。

只注册 validation 的原因：

- GAIA test split 没有公开 gold answer，适合 leaderboard submission，不适合本地 `gold` 评分。
- validation split 有 `Final answer`，可以接入 LycheeMAS 本地评分。

loader 返回：

```python
{
    "task": "gaia_validation_level_1",
    "kind": "gaia",
    "question": "...",
    "gold": "...",
    "context": "Referenced file path: ...",  # 有附件时
    "metadata": {
        "task_id": "...",
        "level": 1,
        "file_name": "...",
    },
}
```

评分：

```python
lychee_mas.eval.metrics.score("gaia", pred, gold)
```

当前 scorer 做了官方风格的：

- 数值规整。
- 标点/大小写/空白规整。
- 逗号/分号分隔列表比较。

当前限制：

- GAIA 很依赖工具能力，例如文件读取、网页搜索、图像/PDF/音频理解等。
- 正式 GAIA 跑法应使用 `team=gaia`，创建 FileSurfer、WebSurfer、Coder、ComputerTerminal，并使用 Magentic-One 风格 group chat。
- 如果人为把 GAIA 改成 `fact` 这类普通文本 team，只能作为一次临时排错命令，不进入正式配置和正式结果口径。
- GAIA 正式评测仍要关注模型的 function calling / vision 能力、网络访问、文件类型覆盖、API token 使用和工具错误归因。

同一个 `runtime=autogen` 默认支持工具；是否真的调用工具取决于 team 里有没有声明工具型
participant。`gaia` 声明了 File/Web/Coder/Terminal，所以会进入工具能力；普通 `fact`
team 没有声明这些 participant，所以即使 runtime 具备工具能力，也不会创建浏览器或代码执行器。

---

## 7. 运行评测

### 7.0 当前 benchmark 评测完整流程

当前主线可以概括成一句话：

```text
数据用 benchmark 自己的数据/loader；
推理用 LycheeMAS 的 runtime=autogen，也就是 AutoGen agents + LycheeMAS memory/router/CDM；
评分用 benchmark 自己对应的 scorer，也就是 record["kind"] -> metrics.score_details(kind, pred, gold)。
```

完整流程：

```text
1. 准备数据
   scripts/prepare_benchmarks.py --tasks <prepare_target>
     -> src/lychee_mas/eval/benchmarks/PREPARERS
     -> 原始上游数据缓存到 LYCHEE_BENCHMARK_RAW_ROOT
     -> loader-ready 数据缓存到 LYCHEE_BENCHMARK_PREPARED_ROOT

2. 启动评测
   scripts/run_mas.py --config <yaml> --task <runnable_task> ...
     -> 读取 YAML
     -> 命令行参数覆盖 YAML
     -> backend.provider=api 时创建 OpenAICompatibleBackend
     -> backend.provider=hf 时创建 HFBackend

3. 组装 LycheeMAS/CDM 组件
   backend
     -> DualChannelMemoryManager
     -> fixed_channel_router(method, P)
     -> RoutingContext(task, router, memory, team)

4. 加载 benchmark case
   load_task(task, n)
     -> LOADERS[task]
     -> 返回统一 record：task/kind/question/gold/context/metadata

5. 选择 team 与答案提取器
   CLI --team > YAML run.team > task_config.py 默认 team
   extractor_for_task(task) 决定 final_answer 抽取策略

6. 构造 AutoGen team
   StaticTopology(team=profile).build()
     -> src/lychee_mas/layers/construct/templates.py 的 TEAMS / TEAM_META
     -> 普通 team：AssistantAgent 角色列表
     -> human_eval：Coder + ComputerTerminal，TEAM_META team_preset=coder_executor
     -> gaia：FileSurfer + WebSurfer + Coder + ComputerTerminal，TEAM_META team_preset=magentic_one

7. 运行 AutoGenRuntime
   AutoGenRuntime._build_agents()
     -> 根据 agent_type 创建 AssistantAgent / Coder / FileSurfer / WebSurfer / ComputerTerminal
     -> 为文件型 case 建 workspace，只复制 loader 公开给 agent 的附件

   AutoGenRuntime._build_chat()
     -> team_preset=magentic_one 时创建 autogen_agentchat.teams.MagenticOneGroupChat
     -> team_preset=coder_executor 时创建 RoundRobinGroupChat(Coder, ComputerTerminal)
     -> 普通 team 走 LycheeMAS build_groupchat

8. 每次模型调用
   InjectionClient
     -> 接收 AutoGen 的 messages/tools
     -> 做 memory/router/latent 或 NL 注入
     -> 调用 API backend 或 HF backend
     -> 返回文本或 tool call
     -> AutoGen 执行工具并把 tool result 回填下一轮上下文

9. 推理落盘
   每个关键运行事件实时追加写入 spans.jsonl
   final_answer = extractor_for_task(task)(messages)
   每个 case 追加写入 predictions.jsonl
   predictions.jsonl 不包含 gold/score/score_details

10. 独立评分和分析
   analyze_benchmark_run.py --score-predictions 读取 predictions.jsonl + benchmark loader 的 gold
   score_details = metrics.score_details(kind, final_answer, gold)
   写入 outputs.jsonl
   metrics.aggregate_samples(outputs) 汇总并写入 metrics.json
   中断后也可以先对已有 predictions.jsonl 评分，得到 partial metrics
```

当前实现已经把推理和评分拆成两个脚本：

- `run_mas.py` 只负责推理：构造 backend/memory/router/team/runtime，调用 AutoGen，实时写 `spans.jsonl`，并按 case 写 `predictions.jsonl`。
- `run_mas.py` 不再调用 scorer，不写 `gold`，也不写 `score/score_details`。
- 推理阶段传给 runtime 的 `TaskQuery.gold` 是 `None`，避免 agent/runtime 意外看到标准答案。
- `analyze_benchmark_run.py --score-predictions` 负责评分：从 benchmark loader 重新读取 gold，对 `predictions.jsonl` 评分，生成 `outputs.jsonl/metrics.json`。
- scorer 本身在 `src/lychee_mas/eval/metrics.py` 和各 benchmark 模块里，不写在 runtime 里。
- run-level 汇总逻辑在 `metrics.aggregate_samples()`，`scripts/analyze_benchmark_run.py` 可以只读已有 `outputs.jsonl` 重新生成 `metrics.json`，不需要重新调用模型。

其中 Magentic-One 不是一个单独命令行入口，而是在运行时由 `gaia` 的
`TEAM_META["gaia"]["team_preset"] = "magentic_one"` 触发。具体代码在
`src/lychee_mas/runtime/backends/autogen_runtime.py` 的 `_build_chat()`：当 preset 是
`magentic_one` 时创建 `MagenticOneGroupChat`。

更细地看，第 6 步“构造 AutoGen team”分成四层：

```text
task_config.py
  task -> team profile 名
  例如 gaia_validation_level_1 -> gaia

templates.py / TEAMS
  team profile 名 -> Role 列表
  Role 是 LycheeMAS 自己的轻量描述：name、system prompt、agent_type、tools、meta

templates.py / TEAM_META
  team profile 名 -> 运行时额外元信息
  例如 gaia -> team_preset=magentic_one
  例如 human_eval -> team_preset=coder_executor

StaticTopology(team=profile).build()
  Role 列表 -> AgentSpec 列表 -> MASGraph
  MASGraph.nodes 是后续 runtime 要创建的 agent 列表
  MASGraph.meta 合并 TEAM_META，比如 team_preset=magentic_one

AutoGenRuntime._build_agents() / _build_chat()
  AgentSpec.agent_type -> 真实 AutoGen participant
  MASGraph.meta.team_preset -> 真实 group chat 形态
```

`TEAMS` 和 `TEAM_META` 的区别：

- `TEAMS` 是“这支队伍有哪些角色”的声明。每个 `Role` 包含角色名、system prompt、`agent_type`、普通 function tools、额外 meta。比如 `reason` 是 planner/solver/verifier；`gaia` 是 FileSurfer/WebSurfer/Coder/ComputerTerminal。
- `TEAM_META` 是“这支队伍用什么群聊形态”的声明。比如 `team_preset=magentic_one` 不新增角色，而是告诉 `AutoGenRuntime._build_chat()`：这些角色要放进 `MagenticOneGroupChat` 里跑；`team_preset=coder_executor` 则用 Coder/Terminal 适合的 RoundRobin 形态。

`StaticTopology.build()` 是否会把顺序固定：

- 它会固定 `MASGraph.nodes` 的声明顺序，也就是 `TEAMS[team]` 里 Role 的顺序。
- 对普通 `reason/fact/memory/aime/default` team，后续 `_build_chat()` 默认使用 round-robin selector，所以运行顺序基本就是声明顺序循环，例如 planner -> solver -> verifier。
- 对 `human_eval`，`team_preset=coder_executor` 会使用 Coder/ComputerTerminal 的 round-robin 形态，因此也是偏固定的代码生成/执行循环。
- 对 `gaia`，`team_preset=magentic_one` 会创建 `MagenticOneGroupChat`，由 Magentic-One orchestrator 根据任务和历史消息选择下一步调哪个 participant；这里不是简单按 FileSurfer -> WebSurfer -> Coder -> Terminal 固定轮转。`TEAMS["gaia"]` 的顺序更像“有哪些可用 participant 的声明”，不是最终决策顺序。
- 如果后续 AgentCollabBench 要按样本 topology 动态构造 agent/edge，就不应该只依赖 `StaticTopology(team=...)`，而应该为样本生成显式 `MASGraph(nodes, edges, meta)` 或新增 topology generator。

普通 `reason/fact/memory/aime` team 的 `agent_type` 都是默认 `assistant`，所以 runtime 会创建多个
AutoGen `AssistantAgent`，再按普通 group chat 顺序/选择策略推进。它们没有浏览器、文件浏览器或代码执行器。

`human_eval` team 的 Role 是：

```text
Coder              agent_type=coder
ComputerTerminal   agent_type=computer_terminal, sources=["Coder"]
```

因此它不是普通文本链，而是代码生成 + 代码执行链。`TEAM_META["human_eval"]["team_preset"] =
"coder_executor"` 会让 `_build_chat()` 使用 Coder/Terminal 适合的 round-robin 形态，并以
`TERMINATE` 作为执行结束信号之一。

`gaia` team 的 Role 是：

```text
FileSurfer         agent_type=file_surfer
WebSurfer          agent_type=web_surfer
Coder              agent_type=coder
ComputerTerminal   agent_type=computer_terminal, sources=["Coder"]
```

`TEAM_META["gaia"]["team_preset"] = "magentic_one"` 会让 `_build_chat()` 创建
`MagenticOneGroupChat`。这里的 orchestrator 也是通过 LycheeMAS 的 `InjectionClient` 包住模型调用，
所以 GAIA 的工具型多 agent 推理仍然会经过 memory/router/latent 注入和 model-call 记录。

smoke 和正式全量的区别只应该是样本数量、结果目录和运行预算；不应该换 prompt、scorer、team、工具链或依赖。

### 7.0.1 API 和本地 GPU 流程有什么不一样

两条流程的 benchmark 逻辑是同一套：

```text
同一份数据 loader
同一个 task_config.py 默认 team/extractor
同一个 runtime=autogen
同一套 AutoGen team / 工具调用 / workspace
同一个 kind/scorer
同一套 outputs.jsonl / metrics.json
```

真正不同的是模型 backend：

| 项目 | API backend | 本地 HF/GPU backend |
|---|---|---|
| backend 类 | `OpenAICompatibleBackend` | `HFBackend` |
| 模型来源 | OpenAI-compatible API 服务或本地 vLLM 暴露的 OpenAI-compatible API | 直接加载本地模型权重 |
| 典型配置 | `configs/benchmarks/api.yaml`，HumanEval/GAIA 用各自 API 预设 | `configs/benchmarks/local_hf.yaml`，HumanEval/GAIA 也可覆盖 `--backend hf` |
| 支持方法 | `none`、`nl_only` | `none`、`nl_only`、`latent_only`、`both` |
| latent/CDM | 普通 API 不暴露 hidden states，不能做真正 latent prefix | 可以拿 hidden states / soft token，适合正式 latent 消融 |
| 工具调用 | 依赖 API 模型 function calling 能力，通常更稳 | 依赖本地模型按提示输出可解析 tool call，稳定性要看模型 |
| 资源/账单口径 | 云 API 侧按服务商 token 规则计费，并受 quota/rate limit 影响；LycheeMAS 只记录 token/latency，不做现金换算 | 占用本地 GPU，不走云 API 账单；LycheeMAS 同样记录 token/position/latency |
| 适合场景 | GPU 忙时 smoke、工具链连通性、云模型对照 | 正式 `latent_only/both`、本地模型实验、本地资源可控的大规模跑 |

因此表格里“API 跑部分”和“API 跑全量”都使用 `configs/benchmarks/api.yaml` 是正常的；
“本地 GPU 跑部分”和“本地 GPU 跑全量”都使用 `configs/benchmarks/local_hf.yaml` 也是正常的。
这些 YAML 是 backend/runtime 预设，不是“部分/全量预设”。部分还是全量由命令里的 `--n 20`、
`--n 5` 或 `--n all` 决定。

注意：当前项目没有内置“人民币/美元费用计算器”。`scripts/run_mas.py` 里的成本相关字段是成本网络配置指标，
例如 `input_text_tokens`、`input_total_positions`、`output_text_tokens`、`model_generation_latency_s`。
API backend 会从 OpenAI-compatible response usage 里读取 prompt/completion token；本地 HF backend 则按本地 tokenizer/position
统计。后续如果要估算OpenAI-compatible API 服务账单，应在结果里的 token 统计基础上另乘当时模型价格表。

### 7.1 轻量入口：run_experiment.py

文件：

```text
scripts/run_experiment.py
```

用途：

- 更偏向离线 mock / 框架自检。
- 默认能用 mock runtime 跑通 pipeline。
- 当前 `--runtime autogen` 不是完整正式实验入口，因为它没有像 `run_mas.py` 那样完整组装 backend、memory、router、RoutingContext。

适合：

- 快速验证 registry / pipeline / aggregator 是否能跑。
- 开发新组件时做轻量 smoke test。

### 7.2 正式入口：run_mas.py

文件：

```text
scripts/run_mas.py
```

用途：

- 真实 MAS 实验驱动。
- 会组装 API/HF backend、DualChannelMemoryManager、fixed_channel_router、RoutingContext、StaticTopology、AutoGenRuntime。
- 按 benchmark case 推理，实时记录 spans，统计 token/position/latency 等资源指标并落盘；评分由 `analyze_benchmark_run.py` 完成。

典型流程：

```text
config / CLI
  -> backend (OpenAICompatibleBackend 或 HFBackend)
  -> DualChannelMemoryManager
  -> fixed_channel_router
  -> RoutingContext
  -> benchmark samples
  -> task_config.py 默认 team/extractor
  -> StaticTopology/team
  -> AutoGenRuntime / AutoGen group chat
  -> spans.jsonl / predictions.jsonl / config.yaml
  -> analyze_benchmark_run.py --score-predictions
  -> outputs.jsonl / metrics.json
```

示例命令需要根据实际 config 和模型路径调整：

```bash
cd /path/to/LycheeMAS
LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT \
LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT \
CUDA_VISIBLE_DEVICES=0 \
$PY scripts/run_mas.py \
  --config configs/aime_both.yaml \
  --task human_eval \
  --method none \
  --n 3 \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --model-tag Qwen3-4B-Instruct-2507 \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/local_hf
```

API-only 跑法适用于 GPU 忙时的 smoke test。以OpenAI-compatible API 为例，API key 只放环境变量，不写入配置文件：

```bash
cd /path/to/LycheeMAS
export DASHSCOPE_API_KEY=<your_api_key>
LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT \
LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT \
$PY scripts/run_mas.py \
  --config configs/benchmarks/gaia.yaml \
  --backend api \
  --task gaia_validation \
  --method none \
  --n 10 \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/api
```

API backend 只支持 `none` / `nl_only`。`latent_only` / `both` 需要本地 HF backend 暴露 hidden states，不能通过普通 chat API 完成。

### 7.3 批量入口：run_benchmark_batch.py

文件：

```text
scripts/run_benchmark_batch.py
```

用途：

- 批量调用 `scripts/run_mas.py`，不另起评测逻辑。
- 用于快速验证新 benchmark 是否能完成“加载数据 -> runtime 推理 -> scorer 评分 -> 结果落盘”。
- 用 `--size smoke|full` 控制小样本检查或全量运行。
- 支持三类模式：
  - `api`：OpenAI-compatible API backend。
  - `local-hf`：本地 GPU / transformers HF backend，可跑 `none`、`nl_only`、`latent_only`、`both`。
  - `all`：依次跑 `api` 和 `local-hf`。
- `smoke` 和 `full` 只改变默认样本数和结果目录，不改变 task 默认 team、scorer、prompt、工具链或依赖。
- `smoke` 默认 `--n 10`、默认结果目录 `runs/benchmarks/smokes`；`full` 默认 `--n all`、默认结果目录 `runs/benchmarks/full`。
- 默认不覆盖 `max_rounds` / `max_turns` / `max_new_tokens`；这些运行上限由正式 YAML 决定。只有显式传 `--max-rounds`、`--max-turns`、`--max-new-tokens` 时才会覆盖。
- HumanEval/GAIA 是否用工具由 `task_config.py` 的默认 team 决定：`human_eval` 默认 `human_eval`，`gaia_validation*` 默认 `gaia`。

API smoke：

```bash
cd /path/to/LycheeMAS
export DASHSCOPE_API_KEY=<your_api_key>
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size smoke \
  --mode api \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/smokes_YYYYMMDD \
  --keep-going
```

如果只想跑某几个 task：

```bash
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size smoke \
  --mode api \
  --tasks human_eval,gaia_validation_level_1 \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/smokes_YYYYMMDD \
  --keep-going
```

这里不传 `--api-model` / `--docker-image` 时，会按 task 使用各自配置文件里的默认值：
HumanEval 默认 `qwen-plus` + `lychee-human-eval:local`；GAIA 默认 `qwen-vl-plus` +
`lychee-gaia:local`；普通文本任务默认使用 `configs/benchmarks/api.yaml`。

本地 GPU smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size smoke \
  --mode local-hf \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/smokes_YYYYMMDD \
  --device cuda:0 \
  --keep-going
```

注意：

- API backend 只能跑 `--api-method none|nl_only`；本地 HF backend 通过 `--local-method` 跑 `none`、`nl_only`、`latent_only`、`both`。
- `--team` 不传时会根据 `task_config.py` 自动选 team。
- HumanEval/GAIA 这类任务默认使用 `runtime=autogen` 下的 `human_eval` / `gaia`，不会在 smoke 脚本里被改成文本 team。
- `run_benchmark_batch.py --max-rounds`、`--max-turns`、`--max-new-tokens` 都是可选覆盖；不传时使用各 task YAML 自己的默认值。
- `--max-rounds` / `--max-turns` 会影响正确率：前者决定静态 team 能完整跑几轮，后者决定 AutoGen group chat 最多允许多少次发言/消息。设太小会减少反思、纠错、工具调用、代码执行后的修复机会；设太大则增加耗时和 token 成本。
- `run_benchmark_batch.py` 默认是两步：先 `run_mas.py` 推理写 `predictions.jsonl`，再 `analyze_benchmark_run.py --score-predictions` 评分写 `outputs.jsonl/metrics.json`。如果只想生成 prediction，加 `--skip-score`。
- `scripts/run_mas.py` 当前支持 `--max-rounds`、`--max-new-tokens`、`--max-turns`、`--docker-image`、`--code-timeout` 等常用覆盖参数；`--max-tokens` 仅作为 `run_mas.py` 旧命令兼容别名，等价于 `--max-new-tokens`。
- `run_mas.py` / `run_experiment.py` 的 `--results-root` 是原 LycheeMAS 已有入口，继续保留；新增 benchmark 脚本只推荐 `--runs-root`。

### 7.3.1 全量模式：--size full

文件：

```text
scripts/run_benchmark_batch.py --size full
```

用途：

- 复用 `--size smoke` 的同一套矩阵 runner、task 默认配置、推理入口和评分入口。
- 默认 `--n all`，默认结果目录 `runs/benchmarks/full`。
- 打印标签从 `[smoke]` 改成 `[full]`，便于区分终端日志。
- 除样本数量和结果目录外，不改变 prompt、scorer、team、工具链、Docker、Playwright 或模型参数。

API 全量：

```bash
cd /path/to/LycheeMAS
export DASHSCOPE_API_KEY=<your_api_key>
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size full \
  --mode api \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/full_YYYYMMDD \
  --keep-going
```

本地 HF/GPU 全量：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size full \
  --mode local-hf \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/full_YYYYMMDD \
  --device cuda:0 \
  --keep-going
```

只跑某几个 task 的全量：

```bash
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size full \
  --mode api \
  --tasks gsm8k,aime_2024,human_eval \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/full_YYYYMMDD \
  --keep-going
```

如果想先确认将要跑哪些命令，不实际执行：

```bash
PYTHONPATH=src \
$PY scripts/run_benchmark_batch.py \
  --size full \
  --mode api \
  --tasks gsm8k \
  --dry-run
```

### 7.4 每个 prepare target / runnable task 从 prepare 到分析的全流程表

下面所有命令默认在运行环境中执行：

```bash
cd /path/to/LycheeMAS
export LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT
export LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT
export LYCHEE_BENCHMARK_RUNS_ROOT=$LYCHEE_BENCHMARK_RUNS_ROOT
export PYTHONPATH=src
export PY=$PY
export MODEL_PATH=models/Qwen3-4B-Instruct-2507
export LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_VLLM_MODEL=Qwen3-4B-Instruct-2507
export VLLM_API_KEY=dummy
export DASHSCOPE_API_KEY=<your_api_key>
```

`export PY=$PY` 只是为了让文档里的命令在非交互 shell、脚本、远程执行时更稳定。也可以先：

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate 目标 Python 环境
export PY=python
```

后面的命令仍然可以不变。

`VLLM_API_KEY=dummy` 是本地 vLLM OpenAI-compatible server 的占位 key。很多 OpenAI client 要求请求里必须有 API key 字段；本地 vLLM 默认通常不校验这个 key，所以用 `dummy` 即可。它不是OpenAI-compatible API 服务 key，也不会产生费用。如果本地 vLLM server 开启了鉴权，就把这里换成实际配置的本地 key。

HumanEval/GAIA 如果要用本地 GPU 跑严格工具链，推荐直接走统一主链路：

```text
run_mas.py
  -> backend_provider=hf
  -> HFBackend
  -> DualChannelMemoryManager
  -> fixed_channel_router
  -> RoutingContext
  -> StaticTopology(team=human_eval/gaia)
  -> AutoGenRuntime(runtime=autogen)
  -> InjectionClient
  -> Coder/FileSurfer/WebSurfer/ComputerTerminal 或普通 AssistantAgent tools
```

这条链路里的每个 LLM participant 仍然绑定 LycheeMAS 的 `InjectionClient`。这个
`InjectionClient` 的职责是：

1. 把 AutoGen message 转成 HF/API backend 可用的 chat messages。
2. 调用 LycheeMAS 的 memory/router，决定 `none/nl/latent/both`。
3. 注入自然语言记忆或 latent prefix。
4. 调用共享 backend 生成文本。
5. 如果 AutoGen 传入 tools，则把 tool schema 写入提示，解析模型输出的结构化 tool call，并交回 AutoGen 执行。

区别只在于 team：

- 普通 `reason` / `fact` / `math` team：构造普通 `AssistantAgent`，适合文本 benchmark。
- `human_eval`：构造 `MagenticOneCoderAgent` + `CodeExecutorAgent(ComputerTerminal)`，适合代码生成和执行。
- `gaia`：构造 `FileSurfer` + `MultimodalWebSurfer` + `MagenticOneCoderAgent` + `CodeExecutorAgent(ComputerTerminal)`，适合 GAIA 这类文件/网页/代码工具任务。
- 普通 `AssistantAgent` 也可以通过 `AgentSpec.tools` / `Role.tools` 绑定 function tools。

本地 vLLM 仍然可以作为 OpenAI-compatible API 对照路径使用：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name "$LOCAL_VLLM_MODEL" --host 0.0.0.0 --port 8000 --trust-remote-code
```

`VLLM_API_KEY=dummy` 就是这种本地 OpenAI-compatible 对照路径的占位 key。但 vLLM/API 通常只返回文本、tool call、usage 等结构，不暴露 LycheeMAS `latent_only/both` 需要的 hidden states / latent prefix 注入接口。所以：

- API / vLLM 路径：适合 `none`、`nl_only`、工具 smoke、环境验证和付费 API 对照。
- 本地 HF 路径：适合正式 `latent_only` / `both` 消融，也可以跑工具 team。

工具型 benchmark 使用项目内的本地 Docker 镜像。现在镜像关系是三层：

| 镜像 | Dockerfile | 用途 |
|---|---|---|
| `lychee-python-sandbox:local` | `docker/python-sandbox.Dockerfile` | 通用 Ubuntu/Python 代码执行沙盒基础镜像。 |
| `lychee-human-eval:local` | `docker/human_eval.Dockerfile` | HumanEval 的代码执行镜像，基于通用沙盒。 |
| `lychee-gaia:local` | `docker/gaia.Dockerfile` | GAIA 的代码/文件处理镜像，基于通用沙盒，额外安装表格、PDF、图片、音视频等常用依赖。 |

镜像内容：

| 镜像 | 系统/apt 依赖 | Python/pip 依赖 | 说明 |
|---|---|---|---|
| `lychee-python-sandbox:local` | `ubuntu:22.04`、`build-essential`、`ca-certificates`、`curl`、`git`、`python3`、`python3-dev`、`python3-pip`、`python3-venv` | 升级 `pip`、`setuptools`、`wheel` | 所有代码执行 benchmark 的基础沙盒。 |
| `lychee-human-eval:local` | 继承 `lychee-python-sandbox:local` | 继承基础沙盒 | HumanEval 当前只需要 Python 代码执行能力，所以保持轻量。 |
| `lychee-gaia:local` | 继承基础沙盒，并额外安装 `ffmpeg`、`poppler-utils` | `beautifulsoup4`、`lxml`、`numpy`、`openpyxl`、`pandas`、`pdfplumber`、`pillow`、`pydub`、`pypdf`、`python-docx`、`requests`、`xlrd` | 覆盖 GAIA 常见网页、表格、PDF、图片、音视频、Office 文档处理需求。 |

正式准备入口仍然是 `prepare_benchmarks.py`。当 prepare target 包含 `human_eval` 或 `gaia*` 时，默认
`--docker-images auto` 会自动构建缺失的 benchmark Docker 镜像：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" \
  --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" \
  --source auto \
  --tasks human_eval,gaia
```

如果网络慢，可以显式指定镜像源：

```bash
cd /path/to/LycheeMAS
$PY scripts/prepare_benchmarks.py \
  --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" \
  --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" \
  --source auto \
  --tasks human_eval,gaia \
  --docker-apt-mirror http://mirrors.tuna.tsinghua.edu.cn/ubuntu \
  --docker-pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

`prepare_benchmarks.py` 会按顺序构建 `lychee-python-sandbox:local`、`lychee-human-eval:local`、`lychee-gaia:local` 并检查镜像存在。
`configs/benchmarks/gaia.yaml` 默认已经使用 `lychee-gaia:local`。这个文件不是 API-only；本地 HF smoke 也会复用它的工具链/runtime 配置。
HumanEval 默认使用 `lychee-human-eval:local`。

`docker pull <image>:<tag>` 是从 Docker Hub 或其它 registry 下载远程镜像，例如 `docker pull ubuntu:22.04`。
项目内的 `lychee-python-sandbox:local`、`lychee-human-eval:local`、`lychee-gaia:local` 是本地构建镜像；
除非后续主动 push 到内部 registry，否则网上没有这些名字，缺失时应该 `docker build`，不要等 Docker 自动 pull。

旧镜像处理：

- `agbench-human-eval:local` 是旧 HumanEval 镜像名；正式命名已改为 `lychee-human-eval:local`。
- `lychee-gaia-tools:local` 是旧 GAIA 镜像名；正式命名已改为 `lychee-gaia:local`。
- 等 `prepare_benchmarks.py --tasks human_eval,gaia` 成功构建新镜像，并且 HumanEval/GAIA smoke 跑通后，可以删除旧 tag：

```bash
docker rmi agbench-human-eval:local lychee-gaia-tools:local
```

不要清理 `sweb.*` 这类非 LycheeMAS benchmark 镜像；它们属于运行环境中已有的其它实验环境。

#### 工具 benchmark 和 latent/CDM 消融的关系

当前目标不是“二选一”或另起一条分叉链路，而是让工具调用和 CDM/latent 消融在原
`runtime=autogen` 主链路里同时存在：

```text
runtime=autogen
  -> AgentSpec/meta 声明 assistant / coder / file_surfer / web_surfer / computer_terminal / function_tools
  -> AutoGenRuntime/_build_agents() 构造 AssistantAgent / Coder / FileSurfer / WebSurfer / ComputerTerminal
  -> InjectionClient 支持 memory/router/latent 注入 + function/tool calling
  -> tool schema 传入模型，模型返回结构化 tool call
  -> AutoGen 执行 tool，tool result 回填下一轮上下文
  -> HFBackend + DualChannelMemoryManager + Router
  -> none / nl_only / latent_only / both
  -> tool calls / tool results / workspace traces
```

这条链路的目标是：HumanEval/GAIA 仍然使用真实工具，但每次模型调用都经过 LycheeMAS 的
memory/router/CDM 注入，因而 latent 组员也可以在工具型 benchmark 上做同一套消融。

当前已经覆盖两类工具能力的第一版：

- 工具型 agents：Coder、FileSurfer、WebSurfer、ComputerTerminal 这类 AutoGen participant。
- 普通 function tools：AssistantAgent 绑定 tool schema，由模型发起 tool call，AutoGen 执行并把结果回填。

这两类都在原 `runtime=autogen` 内，不再依赖单独 `runtime=autogen_tools`。

表里的“部分”只表示 `--n` 较小；“全量”表示 `--n all`。除样本数量和结果目录外，
正式 smoke 与全量应使用同一套数据、runtime、prompt、scorer、依赖和工具链。

表格里的运行命令是推理阶段命令，会生成 `predictions.jsonl`。每个“分析”列已经按
`root/model/run_label/task` 展开成具体目录，对应左侧紧邻的 `run_mas.py` 命令结束时终端打印出的结果目录。
评分后才会生成 `outputs.jsonl` 和 `metrics.json`。批量入口 `run_benchmark_batch.py` 会自动执行这一步。

命名规则：

- API 部分：`<task>_api_smoke`
- API 全量：`<task>_api_full`
- 本地 HF/GPU 部分：`<task>_local_hf_smoke`
- 本地 HF/GPU 全量：`<task>_local_hf_full`

默认表统一使用 `--method none`，表示不注入 memory/latent 的基础跑法。`both` 不是不能用，
而是 latent/CDM 消融实验；建议另开结果目录，例如 `gsm8k_local_hf_both_smoke` /
`gsm8k_local_hf_both_full`，避免和默认结果混在一起。

| Prepare target | 准备数据命令 | Runnable task | API 跑部分 | API 部分分析 | API 跑全量 | API 全量分析 | 本地 GPU 跑部分 | 本地 GPU 部分分析 | 本地 GPU 跑全量 | 本地 GPU 全量分析 | 说明 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gsm8k` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gsm8k` | `gsm8k` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task gsm8k --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_api_smoke/qwen-plus-api/none/gsm8k --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task gsm8k --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_api_full/qwen-plus-api/none/gsm8k --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gsm8k --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_smoke/Qwen3-4B-Instruct-2507/none/gsm8k --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gsm8k --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_full/Qwen3-4B-Instruct-2507/none/gsm8k --score-predictions` | 数字短答。 |
| `aime_2024` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks aime_2024` | `aime_2024` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aime_2024 --method none --n 5 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_api_smoke/qwen-plus-api/none/aime_2024 --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aime_2024 --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_api_full/qwen-plus-api/none/aime_2024 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aime_2024 --method none --n 5 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_local_hf_smoke/Qwen3-4B-Instruct-2507/none/aime_2024 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aime_2024 --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aime_2024_local_hf_full/Qwen3-4B-Instruct-2507/none/aime_2024 --score-predictions` | 正式评分先安装数学解析依赖；latent 消融另开 `--method both --P 8` 结果目录。 |
| `arc_easy` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks arc_easy` | `arc_easy` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task arc_easy --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_api_smoke/qwen-plus-api/none/arc_easy --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task arc_easy --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_api_full/qwen-plus-api/none/arc_easy --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task arc_easy --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_local_hf_smoke/Qwen3-4B-Instruct-2507/none/arc_easy --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task arc_easy --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/arc_easy_local_hf_full/Qwen3-4B-Instruct-2507/none/arc_easy --score-predictions` | 选择题，统一 preparer 已接入。 |
| `openbookqa` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks openbookqa` | `openbookqa` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task openbookqa --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_api_smoke/qwen-plus-api/none/openbookqa --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task openbookqa --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_api_full/qwen-plus-api/none/openbookqa --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task openbookqa --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_local_hf_smoke/Qwen3-4B-Instruct-2507/none/openbookqa --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task openbookqa --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/openbookqa_local_hf_full/Qwen3-4B-Instruct-2507/none/openbookqa --score-predictions` | 选择题，统一 preparer 已接入。 |
| `medqa` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks medqa` | `medqa` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task medqa --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_api_smoke/qwen-plus-api/none/medqa --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task medqa --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_api_full/qwen-plus-api/none/medqa --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task medqa --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_local_hf_smoke/Qwen3-4B-Instruct-2507/none/medqa --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task medqa --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/medqa_local_hf_full/Qwen3-4B-Instruct-2507/none/medqa --score-predictions` | 下载后标准化为 `$LYCHEE_BENCHMARK_PREPARED_ROOT/medqa/medqa.json`。 |
| `locomo10` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks locomo10` | `locomo10` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task locomo10 --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_api_smoke/qwen-plus-api/none/locomo10 --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task locomo10 --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_api_full/qwen-plus-api/none/locomo10 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task locomo10 --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_local_hf_smoke/Qwen3-4B-Instruct-2507/none/locomo10 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task locomo10 --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/locomo10_local_hf_full/Qwen3-4B-Instruct-2507/none/locomo10 --score-predictions` | 下载后缓存到 `$LYCHEE_BENCHMARK_PREPARED_ROOT/locomo10/locomo10.json`。 |
| `human_eval` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks human_eval` | `human_eval` | `$PY scripts/run_mas.py --config configs/benchmarks/human_eval.yaml --runtime autogen --backend api --api-model qwen-plus --task human_eval --method none --n 10 --max-new-tokens 16384 --code-executor docker --docker-image lychee-human-eval:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_api_smoke/qwen-plus-api/human_eval_none/human_eval --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/human_eval.yaml --runtime autogen --backend api --api-model qwen-plus --task human_eval --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-human-eval:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_api_full/qwen-plus-api/human_eval_none/human_eval --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/human_eval.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task human_eval --method none --n 10 --max-new-tokens 16384 --code-executor docker --docker-image lychee-human-eval:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_local_hf_smoke/Qwen3-4B-Instruct-2507/human_eval_none/human_eval --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/human_eval.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task human_eval --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-human-eval:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/human_eval_local_hf_full/Qwen3-4B-Instruct-2507/human_eval_none/human_eval --score-predictions` | 严格跑法需要 Coder + Terminal + Docker；API 只能跑 `none/nl_only`，本地 HF 才能跑 `latent_only/both`。 |
| `gaia_validation` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gaia_validation` | `gaia_validation` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_api_smoke/qwen-vl-plus-api/gaia_none/gaia_validation --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_api_full/qwen-vl-plus-api/gaia_none/gaia_validation --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_local_hf_smoke/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_validation_local_hf_full/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation --score-predictions` | 严格跑法需要 File/Web/Coder/Terminal；图片题需要视觉模型，Qwen3-4B-Instruct-2507 文本模型只适合非视觉子集或链路 smoke。 |
| `gaia_validation` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gaia_validation` | `gaia_validation_level_1` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_1 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_api_smoke/qwen-vl-plus-api/gaia_none/gaia_validation_level_1 --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_1 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_api_full/qwen-vl-plus-api/gaia_none/gaia_validation_level_1 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_1 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_local_hf_smoke/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_1 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_1 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_local_hf_full/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_1 --score-predictions` | validation level 1。 |
| `gaia_validation` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gaia_validation` | `gaia_validation_level_2` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_2 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_api_smoke/qwen-vl-plus-api/gaia_none/gaia_validation_level_2 --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_2 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_api_full/qwen-vl-plus-api/gaia_none/gaia_validation_level_2 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_2 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_local_hf_smoke/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_2 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_2 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l2_local_hf_full/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_2 --score-predictions` | validation level 2。 |
| `gaia_validation` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gaia_validation` | `gaia_validation_level_3` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_3 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_api_smoke/qwen-vl-plus-api/gaia_none/gaia_validation_level_3 --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend api --api-model qwen-vl-plus --task gaia_validation_level_3 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_api_full/qwen-vl-plus-api/gaia_none/gaia_validation_level_3 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_3 --method none --n 5 --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_local_hf_smoke/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_3 --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/gaia.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task gaia_validation_level_3 --method none --n all --max-new-tokens 16384 --code-executor docker --docker-image lychee-gaia:local --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l3_local_hf_full/Qwen3-4B-Instruct-2507/gaia_none/gaia_validation_level_3 --score-predictions` | validation level 3。 |
| `aftraj` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks aftraj` | `aftraj_audit` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aftraj_audit --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_api_smoke/qwen-plus-api/none/aftraj_audit --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aftraj_audit --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_api_full/qwen-plus-api/none/aftraj_audit --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aftraj_audit --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_local_hf_smoke/Qwen3-4B-Instruct-2507/none/aftraj_audit --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aftraj_audit --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_audit_local_hf_full/Qwen3-4B-Instruct-2507/none/aftraj_audit --score-predictions` | AFTraj audit 全量 task。 |
| `aftraj` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks aftraj` | `aftraj_audit_test` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aftraj_audit_test --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_api_smoke/qwen-plus-api/none/aftraj_audit_test --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task aftraj_audit_test --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_api_full/qwen-plus-api/none/aftraj_audit_test --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aftraj_audit_test --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_local_hf_smoke/Qwen3-4B-Instruct-2507/none/aftraj_audit_test --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task aftraj_audit_test --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/aftraj_test_local_hf_full/Qwen3-4B-Instruct-2507/none/aftraj_audit_test --score-predictions` | AFTraj paper/test split。 |
| `agent_collab` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks agent_collab` | `agent_collab_idr` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_idr --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_api_smoke/qwen-plus-api/none/agent_collab_idr --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_idr --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_api_full/qwen-plus-api/none/agent_collab_idr --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_idr --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_local_hf_smoke/Qwen3-4B-Instruct-2507/none/agent_collab_idr --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_idr --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_idr_local_hf_full/Qwen3-4B-Instruct-2507/none/agent_collab_idr --score-predictions` | Instruction Decay。 |
| `agent_collab` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks agent_collab` | `agent_collab_rtd` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_rtd --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_api_smoke/qwen-plus-api/none/agent_collab_rtd --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_rtd --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_api_full/qwen-plus-api/none/agent_collab_rtd --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_rtd --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_local_hf_smoke/Qwen3-4B-Instruct-2507/none/agent_collab_rtd --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_rtd --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_rtd_local_hf_full/Qwen3-4B-Instruct-2507/none/agent_collab_rtd --score-predictions` | Tracer Durability。 |
| `agent_collab` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks agent_collab` | `agent_collab_cpr` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_cpr --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_api_smoke/qwen-plus-api/none/agent_collab_cpr --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_cpr --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_api_full/qwen-plus-api/none/agent_collab_cpr --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_cpr --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_local_hf_smoke/Qwen3-4B-Instruct-2507/none/agent_collab_cpr --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_cpr --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_cpr_local_hf_full/Qwen3-4B-Instruct-2507/none/agent_collab_cpr --score-predictions` | Consensus Pollution。 |
| `agent_collab` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks agent_collab` | `agent_collab_clc` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_clc --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_api_smoke/qwen-plus-api/none/agent_collab_clc --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task agent_collab_clc --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_api_full/qwen-plus-api/none/agent_collab_clc --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_clc --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_local_hf_smoke/Qwen3-4B-Instruct-2507/none/agent_collab_clc --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task agent_collab_clc --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/agent_collab_clc_local_hf_full/Qwen3-4B-Instruct-2507/none/agent_collab_clc --score-predictions` | Context Leakage。 |
| `mast_data` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks mast_data` | `mast_failure` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task mast_failure --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_api_smoke/qwen-plus-api/none/mast_failure --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task mast_failure --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_api_full/qwen-plus-api/none/mast_failure --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task mast_failure --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_local_hf_smoke/Qwen3-4B-Instruct-2507/none/mast_failure --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task mast_failure --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/mast_failure_local_hf_full/Qwen3-4B-Instruct-2507/none/mast_failure --score-predictions` | 默认读取 human-labelled 数据；full dataset 应另设 task。 |
| `open_agent_traces` | `$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks open_agent_traces` | `open_agent_traces` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task open_agent_traces --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_api_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_api_smoke/qwen-plus-api/none/open_agent_traces --score-predictions` | `$PY scripts/run_mas.py --config configs/benchmarks/api.yaml --runtime autogen --backend api --api-model qwen-plus --task open_agent_traces --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_api_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_api_full/qwen-plus-api/none/open_agent_traces --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task open_agent_traces --method none --n 20 --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_local_hf_smoke` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_local_hf_smoke/Qwen3-4B-Instruct-2507/none/open_agent_traces --score-predictions` | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py --config configs/benchmarks/local_hf.yaml --runtime autogen --backend hf --model-path "$MODEL_PATH" --device cuda:0 --task open_agent_traces --method none --n all --max-new-tokens 8192 --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_local_hf_full` | `$PY scripts/analyze_benchmark_run.py "$LYCHEE_BENCHMARK_RUNS_ROOT"/open_agent_traces_local_hf_full/Qwen3-4B-Instruct-2507/none/open_agent_traces --score-predictions` | deviation/anomaly detection。 |
补充：`gaia` 是完整 GAIA snapshot 的 prepare target，不是本地 scored runnable task。需要准备完整 snapshot 时执行：

```bash
$PY scripts/prepare_benchmarks.py --raw-root "$LYCHEE_BENCHMARK_RAW_ROOT" --prepared-root "$LYCHEE_BENCHMARK_PREPARED_ROOT" --source auto --tasks gaia
```

#### 断点续测 / 分片续跑

当前 `run_mas.py` 会在 run 开始时先写出 `config.yaml`，运行中持续追加 `spans.jsonl`，
并在每个 case 完成或失败时追加一行到 `predictions.jsonl`。所以中途停掉时：

- 已完成 case 至少有 prediction、span trace。
- 正在运行但尚未完成的 case 通常也能在 `spans.jsonl` 里看到 `case_start`、`model_call_start`、已经完成的 `model_call_end/tool_event` 或异常 span。

`outputs.jsonl/metrics.json` 由 `scripts/analyze_benchmark_run.py --score-predictions` 生成。正常完整流程是：

```bash
$PY scripts/run_mas.py ... --runs-root "$RUN_ROOT"
$PY scripts/analyze_benchmark_run.py "$RUN_DIR" --score-predictions
```

如果中途停止，也可以先对已有 `predictions.jsonl` 评分，得到 partial `outputs.jsonl/metrics.json`。

现在已经支持显式断点续测：

```bash
$PY scripts/run_mas.py ... --runs-root "$RUN_ROOT" --resume
```

`--resume` 的文件处理规则：

| 文件 | 普通运行 | `--resume` 运行 |
|---|---|---|
| `console_log.txt` | 重新写入 | 追加写入，并插入一段 `[resume]` 分隔信息。 |
| `spans.jsonl` | 重新写入，`seq` 从 1 开始 | 追加写入，`seq` 从旧文件最大 `seq` 往后续号；会写 `resume_start` 和 `case_skip` 等 span。 |
| `predictions.jsonl` | 重新写入 | 先读取旧文件，只保留 `status != "error"` 的成功 case；旧 error/重复/不属于当前数据范围的记录会备份到 `predictions.jsonl.bak.<timestamp>` 后从活动文件删除，随后这些 case 会重跑。 |
| `outputs.jsonl` / `metrics.json` | 推理阶段删除旧文件，评分阶段重建 | 同样删除旧评分文件，因为 prediction 集合可能已变化；续跑完成后重新执行 `analyze_benchmark_run.py --score-predictions`。 |

`--resume` 跳过逻辑按 `case_id` 判断：已有成功 prediction 的 case 会打印
`skip completed case_id=...` 并写 `case_skip` span；失败 case 和没有记录的 case 会正常运行。

如果跑到一半停掉，有三种处理方式：

1. 已经生成的 `predictions.jsonl` 可以先评分。对当前 run 目录执行：

   ```bash
   $PY scripts/analyze_benchmark_run.py \
     "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_full/<model>/<method>/<task> \
     --score-predictions
   ```

   这会读取 `predictions.jsonl`，从 benchmark loader 重新读 gold，写出 `outputs.jsonl` 和
   `metrics.json`。

2. 推荐直接用同一个 `--runs-root` 加 `--resume`。例如继续某个 GAIA L1 full run：

   ```bash
   CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py \
     --config configs/benchmarks/gaia.yaml \
     --runtime autogen \
     --backend hf \
     --model-path "$MODEL_PATH" \
     --device cuda:0 \
     --task gaia_validation_level_1 \
     --method none \
     --n all \
     --max-new-tokens 16384 \
     --code-executor docker \
     --docker-image lychee-gaia:local \
     --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gaia_l1_local_hf_full \
     --resume
   ```

   这会复用同一个结果目录，保留成功 case，重跑失败/缺失 case。

3. 如果想手动分片，继续用 `--start-index` 和 `--n` 指定剩余范围，并建议换一个新的 `--runs-root`，避免覆盖前一段结果。

例如 GSM8K 全量 1319 条，前 200 条已经跑完，继续跑后面的样本：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py \
  --config configs/benchmarks/local_hf.yaml \
  --runtime autogen \
  --backend hf \
  --model-path "$MODEL_PATH" \
  --device cuda:0 \
  --task gsm8k \
  --method none \
  --start-index 200 \
  --n all \
  \
  --max-new-tokens 8192 \
  --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_resume_from_200
```

如果想把分片结果合成一个完整表，目前需要后处理合并多个 `predictions.jsonl`，再用
`scripts/analyze_benchmark_run.py --score-predictions` 重新评分；如果已经有多个 scored `outputs.jsonl`，也可以合并后用
`scripts/analyze_benchmark_run.py --write` 重新计算 metrics。

#### smoke 是什么意思

这里推荐把 `smoke` 严格理解为“小样本端到端正式流程检查”：

- 目的：确认数据加载、runtime 调用、agent/tool 执行、scorer、指标汇总、结果落盘都能跑通。
- 和全量的区别：原则上只改样本数量，例如 `--n 5/10/20` 改成 `--n all`。
- 不应该改变：prompt、team/runtime、scorer、依赖、工具链、数据字段解释。
- 不适合表达：为了省事换成 mock scorer、简化 scorer、无工具 baseline、不同 prompt 的调试结果。

如果为了定位问题临时降低 `--max-tokens`、关闭工具、换 scorer、换无工具 baseline，
建议明确写成 `debug smoke` 或 `baseline smoke`，不要把它当作正式 benchmark smoke。

#### AIME scorer 依赖

`metrics.score_aime()` 正式路径会调用 `math_parsing_util.py`，依赖：

```text
sympy
regex
latex2sympy2-extended
word2number
```

当前 `目标 Python 环境` 环境曾经缺数学解析依赖，导致 AIME scoring 在 import 阶段失败。
现在代码里已经移除 fallback：缺依赖时 AIME smoke 会直接失败。原因是 smoke 应该和
正式测试使用同一套 scorer、依赖和逻辑，只改变样本数量。

AIME 的正式 smoke / 正式全量都必须安装完整数学解析依赖。

安装命令：

```bash
$PY -m pip install latex2sympy2-extended word2number
```


#### 通用分析命令

```bash
# 看某个结果目录的汇总指标
$PY - <<'PY'
import json
path = "runs/gsm8k_api/metrics.json"
print(json.dumps(json.load(open(path)), ensure_ascii=False, indent=2))
PY

# 看某个结果目录第一条 case 明细
head -n 1 runs/gsm8k_api/outputs.jsonl | $PY -m json.tool

# 汇总某个 runs_root 下面所有 metrics
$PY - <<'PY'
import glob, json
for path in sorted(glob.glob("runs/**/metrics.json", recursive=True)):
    m = json.load(open(path))
    print(m.get("runtime", "autogen"), m.get("backend_provider"), m["task"],
          m["model"], m["method"], m.get("mean_score"), path)
PY
```
### 7.5 脚本参数与终端打印速查

这一节只解释 benchmark 板块当前常用的 Python 脚本。最正式的评测入口是
`scripts/run_mas.py`；`scripts/prepare_benchmarks.py` 负责数据和 benchmark 运行资源，包括 HumanEval/GAIA 的 Docker 镜像；`scripts/run_benchmark_batch.py`
只是批量调用 `run_mas.py`，用 `--size smoke|full` 区分小样本和全量；`scripts/run_experiment.py` 是轻量自检入口，不建议作为正式 benchmark 主入口。

#### 7.5.1 `scripts/prepare_benchmarks.py` 参数

| 参数 | 取值示例 | 作用 | 什么时候用 |
|---|---|---|---|
| `--tasks` | `gsm8k,aime_2024,gaia_validation` | 指定要准备的 prepare target；可逗号分隔，也可空格分隔。 | 准备某几个 benchmark 数据。 |
| `--all-full-benchmarks` | 无值 flag | 准备每个 Benchmark source 的 canonical/full target。 | 正式实验前一次性准备全量数据。 |
| `--raw-root` | `$LYCHEE_BENCHMARK_RAW_ROOT` | 指定上游原始数据保存根目录，并写入 `LYCHEE_BENCHMARK_RAW_ROOT`。 | 不想用默认相对路径时。 |
| `--prepared-root` | `$LYCHEE_BENCHMARK_PREPARED_ROOT` | 指定 loader 可读数据保存根目录，并写入 `LYCHEE_BENCHMARK_PREPARED_ROOT`。 | 不想用默认相对路径时。 |
| `--source` | `auto` / `modelscope` / `huggingface` | 选择下载平台。`auto` 会按各 benchmark catalog 的顺序尝试。 | 国内环境通常先用 `auto` 或 `modelscope`。 |
| `--force` | 无值 flag | 强制重新准备目标数据。 | 怀疑缓存坏了，或需要重新下载/重转。 |
| `--docker-images` | `auto` / `always` / `never` | 是否准备 benchmark Docker 镜像；默认 `auto`，只为选中的 HumanEval/GAIA target 准备。 | 正式 prepare 保持默认；只准备文本 benchmark 时不会构建 Docker。 |
| `--force-docker-images` | 无值 flag | 即使本地镜像已存在，也重新构建 selected Docker 镜像。 | Dockerfile 改过或怀疑镜像不干净时。 |
| `--docker-apt-mirror` | `http://mirrors.tuna.tsinghua.edu.cn/ubuntu` | 构建 `lychee-python-sandbox:local` 时使用的 Ubuntu apt 镜像源。 | apt 下载慢时。 |
| `--docker-pip-index-url` | `https://pypi.tuna.tsinghua.edu.cn/simple` | 构建 Docker 镜像时使用的 pip 源。 | pip 下载慢时。 |
| `--docker-no-cache` | 无值 flag | Docker build 不使用层缓存。 | 需要完全重建镜像时。 |
| `--docker-pull-base` | 无值 flag | 构建基础沙盒时尝试拉取更新的 `ubuntu:22.04`。 | 需要刷新基础镜像时。 |
| `--list-benchmark-sources` | 无值 flag | 只打印 canonical Benchmark source。 | 想知道有哪些“大 benchmark”。 |
| `--list-prepare-targets` | 无值 flag | 只打印 `--tasks` 可接受的 prepare target/alias。 | 不确定 `--tasks` 能填什么时。 |
| `--list-runnable-tasks` | 无值 flag | 只打印 `run_mas.py --task` 可接受的 runnable task。 | 不确定评测能跑什么 task 时。 |
| `--list-benchmark-structure` | 无值 flag | 打印 Benchmark source、prepare target、runnable task、kind/scorer 的对应关系。 | 想看“一个 benchmark 为什么对应多个 task”。 |
| `--list-download-sources` | 无值 flag | 打印每个 benchmark 的 ModelScope/HuggingFace/其它 fallback 来源。 | 想核对数据源、环境变量 override、备用链接。 |

`prepare_benchmarks.py` 的常见终端输出：

| 输出片段 | 含义 |
|---|---|
| `[prepare] raw_root=...` | 当前 raw 根目录，保存上游原始文件/snapshot。 |
| `[prepare] prepared_root=...` | 当前 prepared 根目录，loader 后续主要从这里读。 |
| `[prepare] source=auto` | 当前下载源选择策略。 |
| `[prepare] all_full_benchmarks=...` | 使用了 `--all-full-benchmarks` 时，实际会准备的 canonical targets。 |
| `[prepare] <target> ...` | 开始准备某个 prepare target。 |
| `[download] <benchmark>: provider=<provider> id=<id> -> <path>` | 实际选择了哪个平台、哪个数据源 ID，以及原始数据写到哪里。 |
| `[prepare] <benchmark>: restored prepared from raw ...` | prepared 不可用，但 raw 已有完整数据，因此直接从 raw 恢复 prepared。 |
| provider 自带进度条 | ModelScope / HuggingFace / datasets 库自己的下载或 parquet 写入进度。 |
| `[prepare] <target> ready at <location>` | 当前 target 准备完成，`location` 是 prepared 文件/目录。 |
| `[prepare:docker] selected=...` | 本次 prepare 需要准备的 Docker 镜像目标；`auto` 模式只在选中 HumanEval/GAIA 时出现。 |
| `[prepare:docker] <image> already exists` | 本地镜像已存在，本次不重复构建；如需重建，加 `--force-docker-images`。 |
| `[prepare:docker] building <image> from <Dockerfile>` | 正在按 Dockerfile 构建 benchmark 镜像。 |
| `[prepare:docker] <image> ready` | Docker 镜像已检查存在，可供 `run_mas.py --code-executor docker` 使用。 |
| `[prepare] failed for <target>: ...` | 当前 target 准备失败，后面是各 provider / converter 的错误汇总。 |

#### 7.5.2 `scripts/run_mas.py` 参数

| 参数 | 取值示例 | 作用 | 详细口径 |
|---|---|---|---|
| `--config` | `configs/benchmarks/local_hf.yaml` | YAML 默认配置。 | 必填；命令行同名参数会覆盖 YAML。 |
| `--runtime` | `autogen` | Runtime 名。 | 当前正式链路统一用 `autogen`，工具 team 也走这条主链路。 |
| `--backend` | `hf` / `api` | 模型后端。 | `hf` 是本地 transformers；`api` 是 OpenAI-compatible API。 |
| `--task` | `gsm8k` | runnable task 名。 | 必须是 `--list-runnable-tasks` 里能看到的名字。 |
| `--method` | `none` / `nl_only` / `latent_only` / `both` | memory/latent 消融方法。 | API backend 不支持 `latent_only/both`，本地 HF 支持四种。 |
| `--team` | `reason` / `gaia` | 强制覆盖 task 默认 team。 | 一般不填，让 `task_config.py` 自动选；临时实验才覆盖。 |
| `--n` | `20` / `all` / `0` | 本次要跑多少个 case。 | `20` 表示前 20 条；`all/0/full` 表示全量。这里的 case 是 benchmark 样本，不是对话轮数。 |
| `--start-index` | `100` | 从第几个样本开始跑。 | 0-based；常用于断点分片，例如从第 100 条开始跑 20 条。 |
| `--P` | `16` | latent prefix 长度。 | 只在 `latent_only/both` 且本地 HF 后端有意义。 |
| `--model-path` | `$LYCHEE_HF_MODEL` / `/path/to/model` | 本地 HF 模型路径。 | `--backend hf` 时需要；不写时从环境变量 `LYCHEE_HF_MODEL` 读取。 |
| `--model-tag` | `Qwen3-4B-Instruct-2507` | 结果目录里的模型名。 | 不填时从 API model 或本地模型目录名推断。 |
| `--api-model` | `qwen-plus` | API 模型名。 | `--backend api` 时需要，除非 YAML 已配置。 |
| `--api-base-url` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible API base URL。 | OpenAI-compatible API 服务等兼容接口用这个。 |
| `--api-key-env` | `DASHSCOPE_API_KEY` | 从哪个环境变量读取 API key。 | 不建议把 key 写进命令或文档。 |
| `--device` | `cuda:0` | 本地 HF 运行设备。 | 配合 `CUDA_VISIBLE_DEVICES` 使用。 |
| `--max-rounds` | `1` / `2` | 静态 team 的轮数。 | `StaticTopology(..., rounds=max_rounds)` 使用；普通顺序 team 可理解为“角色链重复几遍”。不是样本数，也不是模型最大生成长度。 |
| `--max-new-tokens` | `8192` / `16384` | 每次模型调用最多生成多少 token。 | 影响单次 LLM completion 截断；短答/选择题通常 8192 足够，HumanEval/GAIA 这类长程任务建议 16384 起。 |
| `--max-turns` | `12` | AutoGen group chat 最大消息/发言上限。 | 不填时默认约等于 `len(agents) * max_rounds`；它是最大上限，系统可以提前结束。 |
| `--code-executor` | `docker` / `local` | 需要执行代码时使用 Docker 还是本地。 | HumanEval/GAIA 等带代码执行环节的 team 会用到。 |
| `--docker-image` | `python:3.11-slim` | Docker executor 基础镜像。 | 只在 `--code-executor docker` 时用。 |
| `--code-timeout` | `120` | 单次代码执行超时时间，单位秒。 | 防止 HumanEval/GAIA 中代码执行卡死。 |
| `--work-root` | `runs/lychee_tool_workspaces` | 每个 case 的工具 workspace 根目录。 | 附件复制、代码执行、浏览器下载等会放这里。 |
| `--trace-model-calls` | 无值 flag | 打印每次 LLM 调用的 `[model:<role>] start/done` 摘要。 | 默认开启；显式写这个 flag 主要用于覆盖 YAML 里的关闭配置。 |
| `--no-trace-model-calls` | 无值 flag | 关闭每次 LLM 调用摘要。 | 想减少终端输出时使用；不影响 `spans.jsonl` / `predictions.jsonl` 的记录。 |
| `--runs-root` | `$LYCHEE_BENCHMARK_RUNS_ROOT/gsm8k_local_hf_full` | 本次结果根目录。 | 推荐使用；最终会在其下按 model/method/task 组织。 |
| `--results-root` | 同 `--runs-root` | 旧兼容别名。 | 原 LycheeMAS 兼容保留，新命令推荐 `--runs-root`。 |
| `--resume` | 无值 flag | 断点续测当前 run 目录。 | 追加 `console_log.txt/spans.jsonl`，保留成功 prediction，重跑失败/缺失 case。 |

本地 HF backend 还有几个 YAML 配置项不在 CLI 参数表里，但会写入 `config.yaml/resolved`：

| YAML 字段 | 示例 | 作用 |
|---|---|---|
| `backend.do_sample` | `false` | 是否采样。`false` 时走贪心解码，`temperature/top_p` 不生效；`true` 时才使用 `temperature/top_p`。 |
| `backend.temperature` / `backend.top_p` | `0.7` / `0.8` | 采样解码参数；API backend 会直接传给服务端，本地 HF 只有 `do_sample: true` 时使用。 |
| `backend.max_input_tokens` | `28672` | 本地 HF 输入上下文上限。超过时优先丢最早的非 system 消息，保留 system/tool schema 和最近对话；仍超限时保留最后这段 token。 |
| `backend.max_repeated_token_run` | `128` | 本地 HF 防生成塌缩保护：如果同一个 token 连续重复达到该长度，提前停止本次生成。 |
| `backend.repetition_penalty` | `1.05` | 本地 HF 重复惩罚。`1.0` 表示关闭；GAIA 本地 HF 配置里用轻微惩罚减少长程重复。 |

用你这条命令举例：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py \
  --config configs/benchmarks/local_hf.yaml \
  --runtime autogen \
  --backend hf \
  --model-path "$MODEL_PATH" \
  --device cuda:0 \
  --task gsm8k \
  --method none \
  --n all \
  --max-rounds 1 \
  --max-new-tokens 8192 \
  --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_full
```

含义是：用本地 HF 模型，在 `cuda:0` 上跑 GSM8K 全量 case；不注入 memory/latent；静态 team 只跑 1 轮；每次模型调用最多生成 8192 token；结果写到指定 runs root。

`run_mas.py` 的终端输出：

| 输出片段 | 含义 |
|---|---|
| `[log] console_log=<out_dir>/console_log.txt` | 本次 run 的终端输出会同时保存到这个文件。stdout/stderr 都会写入，包括模型加载、warning、case 进度和模型调用摘要。 |
| `[resume] appending to existing console_log=...` | 当前使用 `--resume`，不会覆盖旧 console log。 |
| `[config] resolved parameters:` | 本次解析后的完整关键配置，会以 JSON 打印到终端和 `console_log.txt`：包含 task/backend/model/team、token 上限、采样、Docker、输出文件路径、环境变量和 resume 信息。 |
| `[MAS] task=gsm8k backend=hf method=none team=reason ... n=1319 kind=exact P=16 max_new_tokens=8192` | 本次 run 的解析后配置。`n=1319` 是实际加载并会运行的 case 数，`kind=exact` 是评分器类型。 |
| `skip completed case_id=...` | `--resume` 下检测到该 case 已有成功 prediction，因此跳过。 |
| `[1/1319]` | 当前第 1 个 case / 总共 1319 个 case。 |
| `msgs=4` | 当前 case 内 AutoGen 产生的消息条数，即 `len(traj.messages)`。它反映编排/agent/tool 交互消息数，不等于样本数，也不等于模型 token 数。 |
| `input_pos=940` | 当前 case 内所有模型调用的输入 position 总和。没有 latent prefix 时近似 prompt tokens；latent 方法时等于文本 token + latent prefix positions；API 时来自 provider usage 的 prompt tokens。 |
| `ans='18'` | 当前 case 抽取出的最终答案前 60 字符预览。 |
| `[done] gsm8k/reason_none: predictions=... -> <out_dir>` | 推理完成，`<out_dir>` 下会有 `spans.jsonl`、`predictions.jsonl`、`config.yaml` 和 `console_log.txt`。 |

这里最容易混的是几个“数量”的层级：

| 名称 | 层级 | 例子 | 解释 |
|---|---|---|---|
| case / sample | benchmark 样本级 | GSM8K 的一道题、GAIA 的一个任务 | `--n` 控制跑多少个 case。 |
| round | team 编排级 | `--max-rounds 1` | 静态角色链跑几轮；不是数据集轮数。 |
| turn/message | AutoGen 对话级 | `msgs=4` | 一个 case 内 agent/tool 产生了几条消息。 |
| model call | 模型调用级 | `num_model_calls=2` | 一个 case 内实际调用大模型几次。 |
| input position | 模型输入级 | `input_pos=940` | 一个 case 内所有 model call 的输入长度求和。 |
| output token | 模型输出级 | `sum_output_text_tokens` | 一个 case 内所有 model call 的输出 token 求和。 |

#### 7.5.3 `scripts/run_benchmark_batch.py` 参数

| 参数 | 取值示例 | 作用 |
|---|---|---|
| `--size` | `smoke` / `full` | 批量运行规模。`smoke` 默认 `--n 10` 和 `runs/benchmarks/smokes`；`full` 默认 `--n all` 和 `runs/benchmarks/full`。 |
| `--mode` | `api` / `local-hf` / `all` | 批量跑 API、本地 HF，或两者都跑。 |
| `--tasks` | `gsm8k,aime_2024` | 逗号分隔的 runnable task；不填时跑默认 smoke task 列表。 |
| `--raw-root` | `data/benchmarks/raw` | 传给子进程的 raw root。 |
| `--prepared-root` | `data/benchmarks/prepared` | 传给子进程的 prepared root。 |
| `--runs-root` | `runs/benchmarks/smokes` | 结果根目录；不传时按 `--size` 使用默认目录。 |
| `--python` | `$PY` | 用哪个 Python 调用 `run_mas.py`。 |
| `--n` | `10` / `20` / `all` | 每个 task 跑多少 case；不传时 `smoke=10`、`full=all`。 |
| `--dry-run` | 无值 flag | 只打印将要执行的命令，不实际运行。 |
| `--keep-going` | 无值 flag | 某个 task 失败后继续跑后面的 task。 |
| `--skip-score` | 无值 flag | 只跑推理，不调用 `analyze_benchmark_run.py --score-predictions`。 |
| `--team` | `reason` | 传给每个子命令的 team override。一般不填。 |
| `--api-model` / `--api-base-url` | `qwen-plus` / API URL | API backend 参数，透传给 `run_mas.py`。 |
| `--api-method` | `none` / `nl_only` | API backend 的 method。API 不支持 latent prefix。 |
| `--model-path` / `--device` / `--model-tag` | `$LYCHEE_HF_MODEL` / `cuda:0` / 标签 | 本地 HF backend 参数，透传给 `run_mas.py`；`--model-path` 不写时读取 `LYCHEE_HF_MODEL`。 |
| `--local-method` / `--P` | `none` / `16` | 本地 HF method 与 latent prefix 长度。 |
| `--max-rounds` / `--max-turns` / `--max-new-tokens` | `2` / `12` / `8192` | 可选覆盖项；不传时使用各 task YAML 默认值。会影响正确率、耗时和 token 成本。 |
| `--code-executor` / `--docker-image` / `--code-timeout` | `docker` / 镜像名 / `120` | 工具型 task 的代码执行参数。 |

`run_benchmark_batch.py` 的打印很简单：

| 输出片段 | 含义 |
|---|---|
| `[smoke] <完整 run_mas.py 命令>` / `[full] <完整 run_mas.py 命令>` | 即将执行的单个子任务命令。 |
| `[smoke] <完整 analyze_benchmark_run.py --score-predictions 命令>` / `[full] ...` | 当前 task 推理完成后，继续对最新 `predictions.jsonl` 评分。 |
| `[smoke] command failed; continuing because --keep-going is set` / `[full] ...` | 当前子任务失败，但因为设置了 `--keep-going`，继续跑后续任务。 |

#### 7.5.4 `scripts/analyze_benchmark_run.py` 参数

这个脚本只做评分/分析，不调用模型。它有两种模式：

- `--score-predictions`：读取 `predictions.jsonl`，从 benchmark loader 重新读取 gold，写出 `outputs.jsonl/metrics.json`。
- 默认模式：读取已有 `outputs.jsonl`，重新汇总 `metrics.json`；也可以 `--rescore` 后重写 outputs。

| 参数 | 取值示例 | 作用 | 备注 |
|---|---|---|---|
| `run_dir` | `runs/.../<model>/<method>/<task>` | 要分析/评分的 run 目录。 | 目录下通常有 `config.yaml`、`predictions.jsonl` 或 `outputs.jsonl`。 |
| `--run-dir` | 同上 | 显式指定 run 目录。 | 和位置参数二选一。 |
| `--score-predictions` | 无值 flag | 从 prediction 进入评分模式。 | 读取 `predictions.jsonl`，从 loader 读 gold，默认写 `outputs.jsonl/metrics.json`。 |
| `--predictions` | `runs/.../predictions.jsonl` | 直接指定推理输出文件。 | 和 `--score-predictions` 搭配使用。 |
| `--outputs` | `runs/.../outputs.jsonl` | 直接指定 case-level 明细文件。 | 指定后不要求标准目录结构。 |
| `--config` | `runs/.../config.yaml` | 指定配置快照。 | 不指定时默认读 `run_dir/config.yaml`。 |
| `--task` | `gsm8k` | 覆盖 task。 | prediction 评分模式下用于从 loader 读取 gold。 |
| `--kind` | `exact` | 覆盖 scorer kind。 | 一般不填，从 `config.yaml` 或 prediction/output 推断。 |
| `--start-index` | `200` | 覆盖 gold 对齐起点。 | 分片推理时用于从同一段 benchmark 样本读取 gold。 |
| `--write` | 无值 flag | 默认模式下写出 `metrics.json`。 | 不加时只在终端打印汇总结果。 |
| `--no-write` | 无值 flag | prediction 评分模式下只打印 metrics。 | 默认会写 `outputs.jsonl/metrics.json`。 |
| `--rescore` | 无值 flag | 默认模式下重新计算每条 case 的 `score/score_details`。 | 会调用 `metrics.score_details(scorer_kind, final_answer, gold)`；HumanEval 会重新跑单测。 |
| `--rewrite-outputs` | 无值 flag | 默认模式下重写 `outputs.jsonl`。 | 一般和 `--rescore` 一起用，把新分数写回 case 明细。 |

常用命令：

```bash
$PY scripts/analyze_benchmark_run.py "$RUN_DIR" --score-predictions
$PY scripts/analyze_benchmark_run.py "$RUN_DIR" --write
```

如果只是重新汇总，不会改变每条 case 的已有 `score`；如果要让 scorer 重新判断正确与否：

```bash
$PY scripts/analyze_benchmark_run.py "$RUN_DIR" --rescore --rewrite-outputs --write
```

#### 7.5.5 `scripts/run_experiment.py` 参数

| 参数 | 取值示例 | 作用 | 备注 |
|---|---|---|---|
| `--runtime` | `mock` / `autogen` | 选择 runtime。 | 默认 `mock`，适合离线自检。 |
| `--team` | `default` | 静态 topology profile。 | 不走完整 benchmark 主链路的配置解析。 |
| `--aggregator` | `self_consistency` | 可选答案聚合器。 | 用于旧实验/自检。 |
| `--benchmark` | `gsm8k` | 从 registry 加载 benchmark。 | 可跑，但正式 benchmark 建议用 `run_mas.py`。 |
| `--questions` | `"2+2"` | 直接传若干问题。 | 离线 smoke 很方便。 |
| `--n` | `5` | 样本数上限。 | 这里是轻量入口自己的样本限制。 |
| `--rounds` | `1` | Orchestrator 轮数。 | 和 `run_mas.py --max-rounds` 类似但不是同一条正式链路。 |
| `--seed` | `0` | 随机种子。 | 主要写入结果配置。 |
| `--model-tag` | `mock` | 结果目录模型标签。 | 默认 `mock`。 |
| `--runs-root` / `--results-root` | 路径 | 结果根目录。 | `--runs-root` 优先。 |
| `--no-save` | 无值 flag | 只打印，不落盘。 | 临时调试用。 |

---

## 8. 四种 method 的含义

`scripts/run_mas.py` 里固定通道路由：

```python
FIXED = {"none": "none", "nl_only": "nl", "latent_only": "latent", "both": "both"}
```

含义：

- `none`：不注入记忆。
- `nl_only`：只通过自然语言文本传递记忆。
- `latent_only`：只通过 latent prefix 传递记忆。
- `both`：自然语言记忆和 latent prefix 都注入。

重要口径：

- latent prefix 不应该简单叫 input tokens。
- 它更准确是 `latent_prefix_positions` 或 `input_latent_positions`。
- 文本输入 token 和 latent positions 应分开统计。

---

## 9. 当前结果文件

完整 benchmark run 会涉及这些核心文件：

```text
spans.jsonl
predictions.jsonl
outputs.jsonl
metrics.json
config.yaml
console_log.txt
```

写出时机不同：

| 文件 | 写出时机 | 用途 |
|---|---|---|
| `config.yaml` | `run_mas.py` 开始时写出 | 实验配置快照：原始 YAML、命令行解析后的 task/backend/method/team/model/token 上限、代码执行配置等。 |
| `console_log.txt` | `run_mas.py` 开始时创建，运行中持续追加 | 本次 run 的终端输出副本：模型加载、数据加载、warning、case 进度、`[model:<role>] start/done` 摘要等。 |
| `spans.jsonl` | `run_mas.py` 运行中持续追加 | span/event 级 trace：case start/end/error、每次 model call start/end/error、AutoGen message、tool event、workspace、code executor 等。即使 case 中途失败，也能看到失败前的细粒度事件。 |
| `predictions.jsonl` | `run_mas.py` 每完成或失败一个 case 追加一行 | inference-only case 明细：问题摘要、最终答案或 error、模型调用汇总、工具调用汇总等；不包含 gold/score。 |
| `outputs.jsonl` | `analyze_benchmark_run.py --score-predictions` 评分后写出 | scored case-level 明细：在 prediction 基础上补充 gold、score、score_details、is_correct。它和 `predictions.jsonl` 很像，是因为 outputs 继承 prediction 字段再补评分字段。 |
| `metrics.json` | `analyze_benchmark_run.py --score-predictions` 评分后写出；也可由 `analyze_benchmark_run.py --write` 从 outputs 重算 | run-level 汇总：平均分、准确率、平均 token、平均 latency、工具调用统计等。 |

默认目录由：

```python
lychee_mas.eval.metrics.result_dir(model, method, task, root)
```

生成，形如：

```text
runs/lychee/<model_tag>/<method>/<task>/
```

如果强制指定 team，method label 会变成：

```text
<team>_<method>
```

例如：

```text
runs/lychee/Qwen3-4B-Instruct-2507/single_none/human_eval/
```

### 9.1 spans.jsonl

`spans.jsonl` 是运行中实时追加的 span/event 级日志：一行 JSON = 一个细粒度运行事件。它是排查“case 还没完成就崩了”的主要文件，因为 `model_call_start` 会在真正调用模型前写入，`case_error/runtime_error/model_call_error` 会在异常发生时写入。

常见 `span_type`：

| `span_type` | 中文含义 | 典型字段 |
|---|---|---|
| `run_start` / `run_end` | 整个 run 开始/结束 | `out_dir`、`num_cases`、`max_new_tokens`、`docker_image`。 |
| `case_start` / `case_end` / `case_error` | 单个 case 开始/结束/失败 | `case_id`、`sample_index`、`question`、`final_answer`、`error_type`、`traceback`。 |
| `runtime_start` / `runtime_end` / `runtime_error` | AutoGen runtime 开始/结束/失败 | `team_preset`、`uses_tools`、`message_count`、`tool_error_count`。 |
| `workspace_prepared` | 工具 workspace 准备完成 | `workspace`、`copied_files`、`task_text`。 |
| `code_executor_start` / `code_executor_ready` / `code_executor_stop` | 代码执行器生命周期 | `executor`、`docker_image`、`timeout_s`、`workspace`。 |
| `model_call_start` | 一次大模型调用即将开始 | `role`、`turn`、`sender`、`input_messages`、`tool_count`、`json_output_requested`。 |
| `model_call_end` | 一次大模型调用完成 | `input_total_positions`、`output_text_tokens`、`model_latency_s`、`output_text`、`json_output_normalized`。 |
| `model_call_error` | 一次大模型调用失败 | `role`、`turn`、`error_type`、`error_message`、`traceback`。 |
| `autogen_message` | AutoGen 产生的一条普通消息/事件 | `source`、`autogen_type`、`content`、`prompt_tokens`、`completion_tokens`。 |
| `tool_event` | AutoGen 工具相关事件 | `source`、`autogen_type`、`content`、`is_error`。 |

读法建议：

- case 崩了但 `predictions.jsonl` 没有完整答案：先按 `case_id` 过滤 `spans.jsonl`。
- 想看某次模型到底吃了什么上下文：看对应 `model_call_start.input_messages`。
- 想看模型输出了什么：看 `model_call_end.output_text`；如果 `json_output_normalized=true`，再看 `raw_output_text`。
- 想看工具有没有真的执行：看 `tool_event`、`workspace_prepared`、`code_executor_*`。

### 9.2 predictions.jsonl 与 outputs.jsonl

`predictions.jsonl` 是 inference-only case 明细，不包含 gold 和评分字段；`outputs.jsonl` 是评分后的 case 明细，会在 prediction 基础上补 `gold/score/score_details/is_correct`。所以两者大部分字段相同是正常的，区别在于：

| 文件 | 是否由模型运行产生 | 是否包含 gold | 是否包含 score | 能否在未评分时存在 |
|---|---:|---:|---:|---:|
| `predictions.jsonl` | 是 | 否 | 否 | 是 |
| `outputs.jsonl` | 否，来自分析脚本 | 是 | 是 | 否 |

### 9.3 outputs.jsonl

`outputs.jsonl` 是 case-level 明细文件：一行 JSON = 一个 benchmark 样本 / 一个 case。它最适合用于错误分析，例如看某道题问了什么、模型答了什么、score_details 为什么扣分。

当前字段按代码实际写入如下：

| 字段 | 出现在哪条 runtime | 中文含义 | 说明 |
|---|---|---|---|
| `case_id` | all | 样本 ID | 优先从 `metadata.task_id` 或 `metadata.safe_task_id` 取；没有就用样本序号。 |
| `task` | all | 运行的 task 名 | 例如 `gsm8k`、`agent_collab_clc`。 |
| `method` | all | 运行方法 | `none/nl_only/latent_only/both`；API backend 只能使用 `none/nl_only`。 |
| `scorer_kind` | all | 评分器类型 | 传给 `metrics.score_details(kind, pred, gold)` 的 `kind`。 |
| `question` | all | 输入给 agent 的问题摘要 | 当前只保存前 500 字符，避免 outputs 过大。 |
| `final_answer` | all | runtime 抽取出来的最终答案 | 实际传给 scorer 的预测值。 |
| `status` | error case | 推理状态 | 失败 case 会写 `error`；成功 case 通常没有该字段。 |
| `error_type/error_message/traceback` | error case | 推理异常信息 | 失败 case 会先写入 prediction，再抛出异常停止当前 run。 |
| `gold` | all | 标准答案/评分对象 | 可能是字符串，也可能是 dict，例如 HumanEval 的 test/entry_point。 |
| `score` | all | 当前 case 分数 | 通常 0/1；MAST 这类 F1 可为 0 到 1 的小数。 |
| `score_details` | all | 结构化评分细节 | 用于错误归因，例如 taxonomy 命中标签、是否泄漏 forbidden。 |
| `is_correct` | all | 是否完全正确 | 只有二值类 scorer 才有意义；非二值时可能是 `null`。 |
| `num_model_calls` | all | 当前 case 的模型调用次数 | v2 字段。 |
| `num_messages` | all | 当前 case 的 AutoGen 消息数 | v2 字段。 |
| `sum_input_total_positions` | all | 当前 case 的总输入 position 数 | 文本 token + latent prefix positions。 |
| `sum_input_text_tokens` | all | 当前 case 的文本输入 token 总数 | 只统计真实文本 prompt token。 |
| `sum_input_latent_positions` | all | 当前 case 的 latent prefix position 总数 | API/tools 路径为 0。不要叫 latent tokens。 |
| `sum_output_text_tokens` | all | 当前 case 的输出 token 总数 | 所有 model call 的 completion token 求和。 |
| `sum_model_latency_s` | all | 当前 case 的模型生成耗时总和 | 所有 `InjectionClient` model call 的生成耗时求和。 |
| `case_wall_time_s` | all | 当前 case 墙钟时间 | 包含模型、工具、编排、代码执行等总耗时；普通文本 task 也会写。 |
| `num_tool_calls` | all | 当前 case 工具事件数 | 普通文本 task 通常为 0；FileSurfer/WebSurfer/ComputerTerminal 等会计入。 |
| `num_tool_errors` | all | 当前 case 工具错误数 | 普通文本 task 通常为 0。 |
| `workspace` | all | 当前 case 的工具工作目录 | 非工具 task 通常为 `null`；附件复制、代码执行、浏览器日志等会在这里。 |
| `copied_files` | all | 被复制进 workspace 的公开附件 | 非工具 task 通常为空列表；来自 question/context 中的 `Referenced file path: ...`。 |
| `model_calls` | all | model-call 级 trace 列表 | 见下一节。 |
| `tool_calls` | all | tool-call 级 trace 列表 | 普通文本 task 通常为空列表；目前还没有逐 tool latency。 |

旧兼容字段仍会写入，方便旧分析脚本继续读：

| 旧字段 | 推荐新字段 | 中文含义 |
|---|---|---|
| `model_call_count` | `num_model_calls` | 模型调用次数。 |
| `message_count` / `n_messages` | `num_messages` | 消息数。 |
| `input_positions_total` / `cost_prompt_pos` | `sum_input_total_positions` | 总输入 position。 |
| `text_input_tokens_total` | `sum_input_text_tokens` | 文本输入 token。 |
| `latent_prefix_positions_total` | `sum_input_latent_positions` | latent prefix positions。 |
| `output_tokens_total` / `gen_tokens` | `sum_output_text_tokens` | 输出 token。 |
| `model_generation_latency_s_total` / `latency_s` | `sum_model_latency_s` | 模型生成耗时；墙钟时间请看 `case_wall_time_s`。 |
| `routing_trace` | `model_calls` | 模型调用/路由 trace。 |
| `correct` | `score` | 单 case 分数。 |

阅读建议：

- 想快速看准确率/平均 token：先读 `metrics.json`。
- 想看每条样本是否答对：读 `outputs.jsonl` 的 `case_id/question/final_answer/gold/score/score_details`。
- 想看 case 完成后的模型调用汇总：读 `model_calls`。
- 想看 case 运行中实时事件、失败前最后一步、完整输入上下文：读 `spans.jsonl`。
- 想看 GAIA/HumanEval 工具是否真的被调用：读 `tool_calls/num_tool_calls/workspace`。

### 9.4 model_calls

`model_calls` 是每个 case 内的“每次大模型调用”记录。现在普通文本 team 和工具型 team 都走统一 `runtime=autogen`，所以它都来自 `RoutingContext.decisions`；每条记录对应一次 `InjectionClient.create()`。

当前字段按代码实际可能写入如下：

| 字段 | 出现在哪条 runtime | 中文含义 | 说明 |
|---|---|---|---|
| `role` | all | 当前被调用的 agent 角色 | 例如 `planner/solver/verifier/Coder/WebSurfer`。 |
| `turn` | all | 该角色自己的第几次模型调用 | 由 `RoutingContext.turn_of(role)` 计数。 |
| `sender` | all | 上一个发言者 | 用于路由判断。 |
| `channel` | all | 记忆通道 | 旧短名，值为 `none/nl/latent/both`。 |
| `P` | all | 请求的 latent prefix 长度 | 路由器给出的 P。 |
| `reason` | all | 路由原因 | 旧短名。 |
| `turn_index` | all | 调用轮次索引 | 清晰命名，表示 role 内 turn。 |
| `sender_role` | all | 上一个发言者角色 | 清晰命名。 |
| `memory_channel` | all | 本次实际使用的记忆通道 | 清晰命名。 |
| `latent_prefix_requested` | all | 路由器请求的 latent prefix 长度 | 不一定等于最终可用 prefix；实际统计看 `latent_prefix_positions`。 |
| `routing_reason` | all | 路由决策说明 | 例如固定通道路由。 |
| `input_total_positions` | all | 本次输入总 position | v2 字段，等于文本 token + latent positions。 |
| `input_text_tokens` | all | 本次文本输入 token | v2 字段。 |
| `input_latent_positions` | all | 本次 latent prefix positions | v2 字段。 |
| `output_text_tokens` | all | 本次输出 token | v2 字段。 |
| `model_latency_s` | all | 本次模型生成耗时 | v2 字段，来自 HF/API backend 实际调用耗时。 |
| `input_positions` | all | 本次输入总 position | 兼容/旧字段。 |
| `text_input_tokens` | all | 本次文本输入 token | 兼容/旧字段。 |
| `latent_prefix_positions` | all | 本次 latent prefix positions | 兼容/旧字段。 |
| `output_tokens` | all | 本次输出 token | 兼容/旧字段。 |
| `model_generation_latency_s` | all | 本次模型生成耗时 | 兼容/旧字段。 |
| `original_prompt_positions` | hf | 本次调用裁剪前输入 position | 未裁剪时等于 `input_positions`；用于判断上下文是否过长。 |
| `prompt_truncated` | hf | 是否发生输入上下文裁剪 | `true` 表示超过 `backend.max_input_tokens` 后裁剪。 |
| `dropped_messages` | hf | 裁剪时丢弃的旧消息数 | 优先丢最早的非 system 消息，保留 system/tool schema 和最近对话。 |
| `nl_memory_chars` | all | 注入的自然语言记忆字符数 | `nl_only/both` 时可能大于 0。 |
| `input_chat_messages` | all | 实际送入 backend 的 chat messages | 包含系统提示、问题、可能插入的 NL 记忆和 tool schema。 |
| `tool_call_request` | tools | 模型请求的工具调用 | 仅当模型输出可解析的 tool call JSON 时写入。 |
| `output_text` | all | 本次模型输出文本 | 用于逐轮调试。 |

旧兼容字段：

| 旧字段 | 推荐新字段 | 中文含义 |
|---|---|---|
| `prompt_pos` | `input_total_positions` | 输入总 position。 |
| `gen_tokens` | `output_text_tokens` | 输出 token。 |
| `prefix_len` | `input_latent_positions` | latent prefix positions。 |
| `nl_chars` | `nl_memory_chars` | NL 记忆字符数。 |
| `latency_s` | `model_latency_s` | 模型调用耗时。 |
| `input_messages` | `input_chat_messages` | 实际输入 messages。 |
| `output` | `output_text` | 模型输出文本。 |

### 9.5 metrics.json

`metrics.json` 是 run-level 汇总文件：一个实验目录一个 JSON。它最适合画表格/曲线，例如平均分、平均 token、平均耗时。

当前字段按代码实际写入如下：

| 字段 | 出现在哪条 runtime | 中文含义 | 说明 |
|---|---|---|---|
| `schema_version` | all | 结果 schema 版本 | 当前为 2。 |
| `model` | all | 模型标签 | 用于结果目录和表格展示。 |
| `backend_provider` | all | 后端类型 | `api` 或 `hf`。 |
| `method` | all | 方法 | `none/nl_only/latent_only/both`。 |
| `team` | all | team 名 | 例如 `reason/aime/fact/human_eval/gaia`。 |
| `task` | all | task 名 | 例如 `mast_failure`。 |
| `probe` | all | 实验探针类型 | 当前为 `mas`。 |
| `memory` | all | memory manager 名 | 当前为 `cdm`。 |
| `router` | all | router 名 | 例如 `always_none/always_both`。 |
| `summary_scope` | all | 汇总范围 | 当前写 `run`，表示整个 run 的汇总。 |
| `aggregation_scope` | all | 聚合粒度 | 当前写 `case_aggregates`，表示先按 case 求和，再跨 case 平均。 |
| `num_cases` | all | case 数 | v2 字段。 |
| `case_count` | all | case 数 | 兼容字段。 |
| `scorer_kind` | all | 评分器类型 | 与 outputs 里的 `scorer_kind` 一致。 |
| `mean_score` | all | 平均分 | v2 字段。 |
| `score_mean` | all | 平均分 | 兼容字段。 |
| `accuracy` | all | 准确率 | 二值 scorer 时等于 `mean_score`；非二值可能为 `null`。 |
| `num_correct` | all | 完全正确 case 数 | 二值 scorer 才有意义。 |
| `mean_model_calls_per_case` | all | 平均每 case 模型调用次数 | v2 字段。 |
| `mean_messages_per_case` | all | 平均每 case 消息数 | v2 字段。 |
| `mean_input_total_positions_per_case` | all | 平均每 case 输入总 position | v2 字段。 |
| `mean_input_text_tokens_per_case` | all | 平均每 case 文本输入 token | v2 字段。 |
| `mean_input_latent_positions_per_case` | all | 平均每 case latent positions | v2 字段。 |
| `mean_output_text_tokens_per_case` | all | 平均每 case 输出 token | v2 字段。 |
| `mean_model_latency_s_per_case` | all | 平均每 case 模型生成耗时 | 所有 model call 耗时求和后按 case 平均。 |
| `mean_model_latency_s_per_call` | all | 平均每次模型调用耗时 | 总模型耗时除以模型调用次数。 |
| `mean_case_wall_time_s` | all | 平均每 case 墙钟时间 | 包含模型、工具、编排、代码执行等总耗时。 |
| `mean_tool_calls_per_case` | all | 平均每 case 工具事件数 | 普通文本 task 通常为 0。 |
| `mean_tool_errors_per_case` | all | 平均每 case 工具错误数 | 普通文本 task 通常为 0。 |
| `total_model_calls` | all | 总模型调用次数 | v2 字段。 |
| `total_messages` | all | 总消息数 | v2 字段。 |
| `total_input_total_positions` | all | 总输入 position | v2 字段。 |
| `total_input_text_tokens` | all | 总文本输入 token | v2 字段。 |
| `total_input_latent_positions` | all | 总 latent positions | v2 字段。 |
| `total_output_text_tokens` | all | 总输出 token | v2 字段。 |
| `total_model_latency_s` | all | 总模型生成耗时 | 所有 model call 耗时求和。 |
| `total_case_wall_time_s` | all | 总 case 墙钟时间 | 所有 case 墙钟时间求和。 |
| `total_tool_calls` | all | 总工具事件数 | 普通文本 task 通常为 0。 |
| `total_tool_errors` | all | 总工具错误数 | 普通文本 task 通常为 0。 |
| `tool_error_count` | all | 总工具错误数 | 旧兼容字段。 |

旧兼容字段仍会写入：

| 旧字段 | 推荐新字段 | 中文含义 |
|---|---|---|
| `n` | `num_cases` | case 数。 |
| `quality` | `mean_score` | 平均分。 |
| `quality_metric` | `scorer_kind` | 评分器类型。 |
| `model_calls_per_case_mean` | `mean_model_calls_per_case` | 平均模型调用次数。 |
| `messages_per_case_mean` / `messages_mean` | `mean_messages_per_case` | 平均消息数。 |
| `input_positions_per_case_mean` / `cost_prompt_pos_mean` | `mean_input_total_positions_per_case` | 平均输入总 position。 |
| `text_input_tokens_per_case_mean` | `mean_input_text_tokens_per_case` | 平均文本输入 token。 |
| `latent_prefix_positions_per_case_mean` | `mean_input_latent_positions_per_case` | 平均 latent positions。 |
| `output_tokens_per_case_mean` / `gen_tokens_mean` | `mean_output_text_tokens_per_case` | 平均输出 token。 |
| `model_generation_latency_s_per_case_mean` / `latency_s_mean` | `mean_model_latency_s_per_case` | 平均每 case 模型生成耗时；墙钟时间单独看 `mean_case_wall_time_s`。 |
| `model_generation_latency_s_per_call_mean` | `mean_model_latency_s_per_call` | 平均每次模型调用耗时。 |

---

## 10. 指标命名建议

当前命名比最早版本清楚，并且已经开始落地显式 schema：

```json
{
  "schema_version": 2,
  "summary_scope": "run"
}
```

推荐三层命名。

### 10.1 model-call level

每次大模型调用一条。当前代码已经写入/归一化的推荐字段如下：

| 字段 | 中文注释 | 当前代码状态 |
|---|---|---|
| `role` | 发起本次调用的 agent/source | 已写入；native 是 agent role，tools 是 AutoGen source。 |
| `turn_index` | 调用轮次索引 | native 已写入；tools 当前主要写 `turn`，后续可统一。 |
| `sender_role` | 上一个发言者角色 | native 已写入。 |
| `memory_channel` | 本次记忆通道 | native 已写入，`none/nl/latent/both`。 |
| `routing_reason` | 路由决策原因 | native 已写入。 |
| `input_text_tokens` | 本次文本输入 token 数 | 已归一化写入。 |
| `input_latent_positions` | 本次 latent prefix position 数 | 已归一化写入。 |
| `input_total_positions` | 文本 token + latent positions | 已归一化写入。 |
| `output_text_tokens` | 本次输出 token 数 | 已归一化写入。 |
| `model_latency_s` | 本次模型生成耗时 | 已归一化写入；tools 当前通常为 0。 |
| `nl_memory_chars` | 注入的自然语言记忆字符数 | native 已写入。 |
| `input_chat_messages` | 实际送给 backend 的 messages | native 已写入。 |
| `output_text` | 本次模型输出文本 | 已写入。 |

后续可选新增但当前代码还没有稳定写入的字段：

| 建议字段 | 中文注释 | 为什么有用 |
|---|---|---|
| `call_index` | case 内第几次模型调用 | 比 `turn_index` 更统一，尤其适合工具 team。 |
| `agent_role` | 规范化后的 agent 角色 | 可替代/补充 `role`，避免 tools 里 source 和 native role 命名差异。 |
| `input_messages` | 规范化后的输入消息 | 可作为 `input_chat_messages` 的短名，但要先统一命名。 |

### 10.2 case level

每个 benchmark 样本一条：

| 字段 | 中文注释 | 当前代码状态 |
|---|---|---|
| `case_id` | 样本 ID | 已写入。 |
| `task` | task 名 | 已写入。 |
| `scorer_kind` | 评分器类型 | 已写入。 |
| `score` | 单 case 分数 | 已写入。 |
| `score_details` | 单 case 评分细节 | 已写入。 |
| `is_correct` | 是否完全正确 | 已写入；二值 scorer 有意义。 |
| `num_model_calls` | 模型调用次数 | 已写入。 |
| `num_messages` | AutoGen 消息数 | 已写入。 |
| `sum_input_text_tokens` | case 内文本输入 token 求和 | 已写入。 |
| `sum_input_latent_positions` | case 内 latent positions 求和 | 已写入。 |
| `sum_input_total_positions` | case 内输入总 position 求和 | 已写入。 |
| `sum_output_text_tokens` | case 内输出 token 求和 | 已写入。 |
| `sum_model_latency_s` | case 内模型生成耗时求和 | 已写入。 |
| `case_wall_time_s` | case 墙钟时间 | 已写入。 |
| `num_tool_calls` | 工具事件数 | 已写入；普通文本 task 通常为 0。 |
| `num_tool_errors` | 工具错误数 | 已写入；普通文本 task 通常为 0。 |

说明：

- `sum_*` 表示该 case 内所有 model calls 求和。
- `case_wall_time_s` 和 `sum_model_latency_s` 不一定相等。前者包含编排、评分、工具调用、Python overhead；后者只统计模型生成。
- 统一 `runtime=autogen` 同时写 `sum_model_latency_s` 和 `case_wall_time_s`；二者统计口径不同，不能互相替代。

### 10.3 run-level aggregate

整次实验一个 `metrics.json`：

| 字段 | 中文注释 | 当前代码状态 |
|---|---|---|
| `schema_version` | schema 版本 | 已写入，当前为 2。 |
| `num_cases` | case 数 | 已写入。 |
| `mean_score` | 平均分 | 已写入。 |
| `accuracy` | 准确率 | 已写入；二值 scorer 时等于平均分。 |
| `num_correct` | 正确 case 数 | 已写入；二值 scorer 有意义。 |
| `mean_model_calls_per_case` | 平均每 case 模型调用次数 | 已写入。 |
| `mean_messages_per_case` | 平均每 case 消息数 | 已写入。 |
| `mean_input_text_tokens_per_case` | 平均每 case 文本输入 token | 已写入。 |
| `mean_input_latent_positions_per_case` | 平均每 case latent positions | 已写入。 |
| `mean_input_total_positions_per_case` | 平均每 case 输入总 position | 已写入。 |
| `mean_output_text_tokens_per_case` | 平均每 case 输出 token | 已写入。 |
| `mean_model_latency_s_per_case` | 平均每 case 模型生成耗时 | 已写入。 |
| `mean_model_latency_s_per_call` | 平均每次模型调用耗时 | 已写入。 |
| `mean_case_wall_time_s` | 平均每 case 墙钟时间 | 已写入。 |
| `mean_tool_calls_per_case` | 平均每 case 工具事件数 | 已写入；普通文本 task 通常为 0。 |
| `mean_tool_errors_per_case` | 平均每 case 工具错误数 | 已写入；普通文本 task 通常为 0。 |
| `total_model_calls` | 总模型调用次数 | 已写入。 |
| `total_messages` | 总消息数 | 已写入。 |
| `total_input_text_tokens` | 总文本输入 token | 已写入。 |
| `total_input_latent_positions` | 总 latent positions | 已写入。 |
| `total_input_total_positions` | 总输入 position | 已写入。 |
| `total_output_text_tokens` | 总输出 token | 已写入。 |
| `total_model_latency_s` | 总模型生成耗时 | 已写入。 |
| `total_case_wall_time_s` | 总 case 墙钟时间 | 已写入。 |
| `total_tool_calls` | 总工具事件数 | 已写入；普通文本 task 通常为 0。 |
| `total_tool_errors` | 总工具错误数 | 已写入；普通文本 task 通常为 0。 |

建议逐步替换：

```text
quality              -> mean_score
n                    -> num_cases
gen_tokens_mean      -> mean_output_text_tokens_per_case
latency_s_mean       -> mean_model_latency_s_per_case
cost_prompt_pos_mean -> mean_input_total_positions_per_case
```

迁移策略：

1. 先新增 v2 字段。
2. 保留旧字段 1-2 个实验周期。
3. 分析脚本改读 v2。
4. 再删除旧字段。

---
## 11. 如何新增一个 benchmark

推荐新增流程：

1. 在 `src/lychee_mas/eval/benchmarks/<task_name>.py` 新建文件。
2. 实现 `prepare_<task_name>(force=False, source=None)`，如果数据需要下载。
3. 实现 `load_<task_name>(n=None)`，返回统一格式。
4. 如果需要新评分方式，在 `src/lychee_mas/eval/metrics.py` 新增 `score_<kind>`。
5. 在 `benchmarks/__init__.py` 注册：
   - `LOADERS["task_name"] = load_<task_name>`
   - 如需下载，`PREPARERS["task_name"] = prepare_<task_name>`
6. 在 `src/lychee_mas/eval/task_config.py` 配置默认 team。
7. 在 `tests/test_eval_benchmark_extensions.py` 或新测试文件加 smoke test。
8. 运行 py_compile、loader smoke test、pytest。

### 11.1 目录结构建议

对于需要 public/private 隔离的数据集，建议数据目录内部这样组织：

```text
Data/
├── public/
└── private/
```

但在 LycheeMAS 当前 native eval 里，更重要的是运行时只把 agent 需要读的内容传给 agent。

原则：

- agent 推理可见：放到 question/context 或可访问工作目录。
- 评分器可见但 agent 不可见：只在 `gold` 或 private data 里使用，不传给 agent。


### 11.2 严格 MAS Benchmark 候选接入路线

参考本地整理文档：

```text
```

一手入口：

- AgentCollabBench: https://huggingface.co/datasets/AgentCollabBench/AgentCollabBench
- AFTraj-2K: https://huggingface.co/datasets/ZBox008003/AFTraj
- MAST-Data: https://huggingface.co/datasets/mcemri/MAST-Data
- Open Agent Traces: https://huggingface.co/datasets/juliensimon/open-agent-traces
- RoCoBench: https://project-roco.github.io/
- SOTOPIA-TOM: https://huggingface.co/datasets/yashwanthys/sotopia-tom

这里的“严格 MAS Benchmark”不要和 GAIA/HumanEval 混在一起理解：

- GAIA/HumanEval 更像下游任务成功率评测，可以由 MAS 求解，但任务本身不直接考 agent 间通信/拓扑/上下文隔离。
- 严格 MAS Benchmark 应该能直接测 `agent role`、`turn/message`、`topology`、`private/public context`、`tool/action`、`failure attribution` 或 `trace-level observability`。
- LycheeMAS 当前 `{task, kind, question, gold, context}` 格式可以先承载离线诊断类 benchmark；真正多 agent 交互类 benchmark 建议逐步扩展为 record v2。

建议的 record v2 兼容格式：

```python
{
    "task": "agent_collab",
    "kind": "mas_context_leakage",
    "question": "...",          # 面向整个 team 的任务说明
    "context": "...",           # 全局公开上下文
    "gold": {...},              # 评分器使用，不能直接给 agent
    "metadata": {
        "source": "AgentCollabBench",
        "metric": "CLC",
        "domain": "data_engineering",
        "difficulty": "hard",
    },
    "agent_specs": [
        {
            "name": "DataArchitect",
            "role": "...",
            "system_prompt": "...",
            "private_context": "...",
        }
    ],
    "topology": {
        "type": "chain",
        "edges": [["DataArchitect", "ETLEngineer"]],
    },
    "evaluator": {
        "type": "rule_or_llm_judge",
        "expected": "...",
        "forbidden": ["private fact that must not leak"],
    },
}
```

这个格式仍然兼容旧入口：普通 loader 可以先只用 `question/context/gold`；后续可让 `AutoGenRuntime` 读取 `agent_specs/topology/evaluator` 来构造真实多 agent team。

#### 当前最值得优先接的候选

| 优先级 | Benchmark | 为什么值得 | 接入难度 | 建议接入方式 |
|---|---|---|---|---|
| P0 | AgentCollabBench | 900 个结构化 task，明确测多 agent 协作失败：instruction decay、multi-hop 信息保持、false-belief pollution、cross-task leakage；天然包含 topology/injection/ground truth | 中 | 先做 loader + rule scorer；再做 `team=sample_topology` 或动态 `MASGraph` 构造 |
| P0 | AFTraj-2K / AgentForesight | 2,276 条 multi-agent trajectory，safe/unsafe、decisive error step、responsible agent 标签清楚；文件小，适合马上做离线诊断评测 | 低 | `aftraj_audit` loader，`kind=mas_audit`，先让模型根据 prefix 判断 continue/alarm、mistake step/agent |
| P1 | MAST / MAD | 已有 MAS execution traces + failure taxonomy annotation，适合失败类型分类和根因诊断 | 低-中 | `mast_failure` loader，`kind=mas_failure_taxonomy`，输出 taxonomy label，多标签 F1 |
| P1 | Open Agent Traces | 17,019 个 event / 500 runs，含 agent_role、LLM prompt/completion、tool call、handoff、deviation label、token/cost；很适合观测性、流程一致性、异常检测 | 中 | `open_agent_traces` loader 按 `run_id` 聚合，`kind=mas_deviation` |
| P1.5 | RoCoBench-Text | text-only，围绕多机器人 agent 身份、能力边界、记忆、询问/回应；不需要机器人仿真即可评测协作沟通 | 中 | 找到数据文件后做 QA/MC loader；可先用普通 `autogen`，后续做多 agent 复现 |
| P2 | SOTOPIA-TOM | 160 个信息不对称场景，public/private channel 很适合测隐私边界和信息管理 | 中-高 | 等 record v2/topology runtime 稳定后接；需要 public/private channel 和 evaluator |
| P2 | MINDGAMES / WOLF / CoffeeBench | 社会博弈、欺骗、长期协商价值高，但风格偏 game/simulation，离 AutoGen tool-call 主线更远 | 高 | 作为长期社会/博弈/长程协商支线，不建议当前抢先接 |

更具体的排序建议：

1. **先接 AFTraj-2K**：最像“下载数据 -> loader -> scorer -> pytest”的普通 benchmark，体量小，不依赖浏览器、Docker、工具执行。它能马上补上“MAS 失败诊断/早期预警”维度。
2. **再接 AgentCollabBench**：它比 AFTraj 更像真正的 MAS benchmark，因为直接测试 topology、context isolation 和信息传播。它需要 runtime 进一步读 `agent_specs/topology`，但值得作为下一阶段主力。
3. **然后接 MAST-Data**：用来补 failure taxonomy。它更偏离线 trace 分类，工程风险低，但不如 AgentCollabBench 那样能直接测试运行时协作。
4. **Open Agent Traces 放在观测性阶段**：它很适合 token/cost/tool/handoff/conformance 分析，也适合做 trace-level benchmark，但不是传统“一个 case 一个最终答案”的结构。

#### 当前已接入状态

已新增文件：

```text
src/lychee_mas/eval/benchmarks/aftraj.py
src/lychee_mas/eval/benchmarks/agent_collab.py
src/lychee_mas/eval/benchmarks/mast_data.py
src/lychee_mas/eval/benchmarks/open_agent_traces.py
configs/benchmarks/strict_mas_api.yaml
```

已注册 canonical source preparer，也就是 `--list-benchmark-sources` / `--list-benchmark-structure` 视角：

```text
aftraj
agent_collab
mast_data
open_agent_traces
```

已注册 task alias，也就是这些 task 名也可以直接传给 `prepare_benchmarks.py --tasks`：

```text
aftraj_audit        -> aftraj
aftraj_audit_test   -> aftraj
agent_collab_idr    -> agent_collab
agent_collab_rtd    -> agent_collab
agent_collab_cpr    -> agent_collab
agent_collab_clc    -> agent_collab
mast_failure        -> mast_data
```

已注册 loader/task：

```text
aftraj_audit
aftraj_audit_test
agent_collab_idr
agent_collab_rtd
agent_collab_cpr
agent_collab_clc
mast_failure
open_agent_traces
```

已新增 scorer kind：

```text
mas_audit
mas_failure_taxonomy
mas_deviation
mas_instruction_decay
mas_tracer_durability
mas_consensus_pollution
mas_context_leakage
```

准备数据示例：

```bash
cd /path/to/LycheeMAS

$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --tasks aftraj

$PY scripts/prepare_benchmarks.py \
  --raw-root $LYCHEE_BENCHMARK_RAW_ROOT \
  --prepared-root $LYCHEE_BENCHMARK_PREPARED_ROOT \
  --source auto \
  --tasks agent_collab,mast_data,open_agent_traces
```

运行示例：

```bash
cd /path/to/LycheeMAS
export DASHSCOPE_API_KEY=...

LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT \
LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT \
PYTHONPATH=src \
$PY scripts/run_mas.py \
  --config configs/benchmarks/strict_mas_api.yaml \
  --task aftraj_audit_test \
  --n 1 \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/api

LYCHEE_BENCHMARK_RAW_ROOT=$LYCHEE_BENCHMARK_RAW_ROOT \
LYCHEE_BENCHMARK_PREPARED_ROOT=$LYCHEE_BENCHMARK_PREPARED_ROOT \
PYTHONPATH=src \
$PY scripts/run_mas.py \
  --config configs/benchmarks/strict_mas_api.yaml \
  --task mast_failure \
  --n 1 \
  --runs-root $LYCHEE_BENCHMARK_RUNS_ROOT/api
```

当前实现边界：

- 这批严格 MAS benchmark 第一版都走 LycheeMAS native eval，不走 agbench。
- AFTraj、MAST-Data、Open Agent Traces 是离线 trace/diagnostic 形态，不需要浏览器、Docker、工具执行；它们已经完成 loader smoke。
- AgentCollabBench 第一版保留了 `agent_specs/topology/evaluator` 字段，但运行时暂时仍会把它们压进 prompt；正式 MAS 版需要后续让 `AutoGenRuntime` 按样本 topology 动态构造 agents/edges/private context。
- AgentCollabBench 当前 scorer 是 rule baseline，适合 smoke/框架联通；如果要复现官方指标，需要进一步实现官方 evaluator 或 LLM judge。

#### 每类 benchmark 怎么落到 LycheeMAS

AFTraj-2K：

- 新文件：`src/lychee_mas/eval/benchmarks/aftraj.py`
- 数据准备：`prepare_aftraj()`，支持 ModelScope/HuggingFace；HuggingFace `snapshot_download` 失败时会按固定文件直链下载。
- loader 输出：
  - `task`: `aftraj_audit`
  - `kind`: `mas_audit`
  - `question`: 给定 trajectory prefix，要求判断 `continue` / `alarm`，如 alarm 则给 `mistake_step` 和 `mistake_agent`
  - `gold`: `{"label": "safe|unsafe", "mistake_step": int, "mistake_agent": str}`
  - `metadata`: `conv_id/domain/num_turns/unsafe_source`
- scorer：
  - 当前 `score_details` 字段包含 `decision_correct`、`mistake_step_correct`、`mistake_agent_correct`、`score`
  - 可进一步汇总成 `alarm_accuracy`、`mistake_step_exact`、`mistake_agent_accuracy`、`joint_attribution_accuracy`
- runtime：可先用现有 `autogen` 或 API backend；不需要工具。

AgentCollabBench：

- 新文件：`src/lychee_mas/eval/benchmarks/agent_collab.py`
- 数据准备：下载 900 条 JSON/JSONL。
- loader 输出：
  - `task`: `agent_collab_idr` / `agent_collab_rtd` / `agent_collab_cpr` / `agent_collab_clc`
  - `kind`: 按 metric 分成 `mas_instruction_decay`、`mas_tracer_durability`、`mas_consensus_pollution`、`mas_context_leakage`
  - `agent_specs/topology/evaluator`: 保留原始结构化字段
- scorer：
  - IDR：检查 injected constraint 是否被保留
  - RTD：检查 seeded factual constraint 是否跨 hop 保持
  - CPR：检查 false belief 是否污染最终共识
  - CLC：检查 Task A private context 是否泄漏到 Task B
- runtime：
  - 第一版可以只把所有角色压成文本 prompt 跑 smoke，验证 loader/scorer。
  - 正式版应在 `AutoGenRuntime` 中支持 `sample_topology` 动态 `MASGraph`，按样本里的 topology 构造 agents/edges/可见上下文。

MAST-Data：

- 新文件：`src/lychee_mas/eval/benchmarks/mast_data.py`
- 数据准备：默认只下载/缓存 `MAD_human_labelled_dataset.json`；如需 full dataset，设置 `LYCHEE_MAST_DOWNLOAD_FULL=1`。
- loader 把 trace 摘要、关键消息、系统信息和 benchmark task 组织成诊断题。
- `kind=mas_failure_taxonomy`
- scorer 当前使用 taxonomy label F1，并在 `score_details` 中保存 `pred_labels`、`gold_labels`、`true_positive_labels`；如果模型输出自然语言，会先做 label parser。
- runtime 不需要工具，适合低成本 API smoke。

Open Agent Traces：

- 新文件：`src/lychee_mas/eval/benchmarks/open_agent_traces.py`
- loader 按 `run_id` 聚合 event，构造 trace-level 判断题。
- `kind=mas_deviation`
- scorer：
  - 当前 `score_details` 字段包含 `deviation_decision_correct`、`type_overlap`、`score`
  - 可进一步汇总成 deviation detection accuracy/F1、deviation type accuracy、responsible agent hit rate
- 额外价值：可以直接对比 LycheeMAS 自己记录的 `model_calls/tool_calls/messages/cost` 字段，作为观测性 schema 的参考。

#### 当前不要优先做的事

- 不建议现在先接 MINDGAMES/WOLF/CoffeeBench：它们价值高，但需要 game/simulation/long-horizon harness，和当前 GAIA 工具链调通不是同一类工程。
- 不建议把所有严格 MAS benchmark 都硬塞进 `{question, context}`：这样会丢掉 topology/private channel 等关键结构，最后又退化成普通 QA。
## 12. AutoGenBench 原方案 vs LycheeMAS native 方案

AutoGenBench 原方案：

```text
benchmark folder
├── Scripts/
├── Templates/
├── config.yaml
├── ENV.yaml
└── Results/
```

特点：

- 用 agbench CLI/Docker 跑。
- 每个 benchmark 像一个独立小项目。
- 适合复现 AutoGenBench 官方例子。
- 对 LycheeMAS 的 memory/router/latent 通道指标接入不直接。

LycheeMAS native 方案：

```text
src/lychee_mas/eval/benchmarks/*.py
scripts/run_mas.py
src/lychee_mas/eval/metrics.py
```

特点：

- loader 统一返回 `{task, kind, question, gold, context}`。
- Runtime 用 LycheeMAS/AutoGen 推理。
- Scorer 用 benchmark 官方或官方风格逻辑。
- 结果统一使用 `predictions.jsonl`、`outputs.jsonl`、`metrics.json`、`config.yaml`。
- 更适合对比 `none/nl_only/latent_only/both` 和统计 LycheeMAS 特有指标。

本项目后续建议以 LycheeMAS native 方案为主；agbench 原方案只作为参考或需要复现官方 AutoGenBench 时使用。

### 12.1 当前对 AutoGen / agbench / LycheeMAS 的判断

三者不是同一层东西：

```text
AutoGen
= agent/runtime 框架
= AssistantAgent / GroupChat / tools / WebSurfer / FileSurfer / CodeExecutor / Magentic-One

agbench / AutoGenBench
= 官方 benchmark runner
= Tasks 展开、Templates/scenario.py、Docker/native 执行、Results、tabulate

LycheeMAS
= 本项目自己的 MAS 实验框架
= benchmark loader/scorer、memory channel、latent routing、metrics、run_mas.py
```

当前 LycheeMAS 的确使用了 AutoGen：

- `src/lychee_mas/runtime/backends/autogen_runtime.py` 里把 `MASGraph` 转成 AutoGen `AssistantAgent`。
- 然后用 AutoGen `RoundRobinGroupChat` / `SelectorGroupChat` 跑多 agent 对话。

早期 LycheeMAS 只用上了 AutoGen 的对话编排能力：`_build_agents()` 基本只创建普通
`AssistantAgent`，`InjectionClient` 主要负责 memory/router/latent prefix 注入，没有把
tools/workbench/function calling 接进来。

正式主链路已经改成：

- 普通文本 benchmark 仍然创建普通 `AssistantAgent`，保持原 `reason/fact/math/...` team 兼容。
- `Role.agent_type` / `AgentSpec.meta.agent_type` 可以声明 `coder`、`file_surfer`、`web_surfer`、`computer_terminal`。
- `human_eval` 会构造 Coder + ComputerTerminal。
- `gaia` 会构造 FileSurfer + WebSurfer + Coder + ComputerTerminal。
- 普通 AssistantAgent 也可以通过 `tools` 字段绑定 function tools。
- `InjectionClient` 继续负责 memory/router/latent 注入，同时支持 AutoGen `tools` 参数、tool schema prompt 和结构化 tool-call 返回。

所以现在的正式口径是：LycheeMAS 仍然使用 AutoGen 做 agent/team 编排，同时把工具能力接回
统一 `runtime=autogen` 主链路；不是另起 `autogen_tools` 分叉。

当前 LycheeMAS 没有直接用 agbench 作为主评测入口：

- 仓库里有官方源码：`src/autogen/python/packages/agbench`。
- 官方 HumanEval / GAIA benchmark 目录也在：`src/autogen/python/packages/agbench/benchmarks/HumanEval`、`src/autogen/python/packages/agbench/benchmarks/GAIA`。
- 但 `src/lychee_mas/eval` 和 `scripts/run_mas.py` 当前没有直接调用 agbench CLI/Docker。

agbench 对后续有价值的部分：

- benchmark 小项目结构：`Scripts/`、`Templates/`、`config.yaml`、`ENV.yaml`、`Tasks/`、`Results/`。
- 每个 case 独立展开运行目录，复制 prompt、expected answer、附件和模板。
- Docker/native 运行逻辑和 `console_log.txt` 等日志结构。
- GAIA 官方 `Templates/MagenticOne/scenario.py` 已经包含 Coder、ComputerTerminal、FileSurfer、WebSurfer、MagenticOneGroupChat。
- HumanEval 官方 `Templates/AgentChat/scenario.py` 已经包含 coder + executor 的代码任务流程。

不建议把 LycheeMAS 主链路完全切到 agbench CLI，原因：

- agbench 结果默认写自己的 `Results/`，不会自然产生 LycheeMAS 当前的 case-level / run-level metrics。
- agbench scenario 直接构造 AutoGen team，不经过 LycheeMAS 的 `RoutingContext`、`DualChannelMemoryManager`、memory channel 和 latent routing。
- 如果一半 benchmark 走 agbench，一半走 `src/lychee_mas/eval`，后续维护会分叉。

### 12.2 四个可选方案

目前可以把后续路线拆成四个方案。它们不是完全互斥的，更像不同集成深度的阶梯。

#### 方案一：直接用 agbench CLI/Docker 原样跑

核心想法：

- 把 agbench 当作独立 benchmark runner。
- 进入官方 benchmark 目录，例如 `src/autogen/python/packages/agbench/benchmarks/GAIA` 或 `HumanEval`。
- 运行 `Scripts/init_tasks.py` 生成 `Tasks/*.jsonl`。
- 用 `agbench run ...` 跑任务，用 `agbench tabulate ...` 汇总结果。

典型命令形态：

```bash
cd /path/to/LycheeMAS/src/autogen/python/packages/agbench/benchmarks/GAIA
$PY Scripts/init_tasks.py
agbench run Tasks/gaia_validation_level_1__MagenticOne.jsonl
agbench tabulate Results/gaia_validation_level_1__MagenticOne
```

实现内容：

- 基本不改 LycheeMAS。
- 只需要保证 agbench、AutoGen、Docker、Playwright、模型 API key 等环境可用。
- 对 DashScope/OpenAI-compatible API，需要改 agbench benchmark 的 `config.yaml`，把 `OpenAIChatCompletionClient` 配成可用模型、`base_url`、`api_key` 或环境变量。

优点：

- 最接近官方 AutoGenBench 的运行方式，方便复现官方示例。
- agbench 已经有独立 case workspace、模板复制、Docker 沙盒、`console_log.txt`、`tabulate`。
- GAIA 官方 MagenticOne 模板已经带 `FileSurfer`、`WebSurfer`、`Coder`、`ComputerTerminal`。
- HumanEval 官方 AgentChat 模板已经有 coder + executor 流程。

缺点：

- 结果进入 agbench 自己的 `Results/`，不会自然进入 LycheeMAS 的 `runs/.../outputs.jsonl`、`metrics.json`、`config.yaml`。
- 不经过 LycheeMAS 的 `RoutingContext`、`DualChannelMemoryManager`、memory channel、latent prefix。
- 不方便对比 `none/nl_only/latent_only/both`。
- LycheeMAS 当前定义的 token/position、model-call trace、case-level metrics 需要另写解析器才能从 agbench logs 里恢复。
- 扩展时会形成两套体系：一套 agbench benchmark 项目，一套 LycheeMAS native eval。

适用场景：

- 快速回答“官方 AutoGenBench 示例能不能跑”。
- 做官方 baseline 或 sanity check。
- 临时验证 Magentic-One 工具链、Docker、Playwright、API 模型是否可用。

不适合作为长期主线，除非项目目标变成“主要维护 agbench benchmark”。

#### 方案二：LycheeMAS native eval 主线，手工复用 agbench 资产

核心想法：

- 继续以 LycheeMAS native eval 为主。
- 数据准备、loader、scorer、metrics、结果目录都走 LycheeMAS。
- agbench 只作为参考资料和资产来源：参考它的 `Scripts/init_tasks.py`、`Templates/scenario.py`、prompt 格式、评分/tabulate 逻辑。

当前已经在做的就是这个方向：

```text
src/lychee_mas/eval/benchmarks/human_eval.py
src/lychee_mas/eval/benchmarks/gaia.py
scripts/prepare_benchmarks.py
src/lychee_mas/eval/metrics.py
scripts/run_mas.py
```

实现内容：

- 每个 benchmark 写一个 LycheeMAS loader，统一返回 `{task, kind, question, gold, context, metadata}`。
- 每个 benchmark 必要时写 prepare，支持 ModelScope / HuggingFace 下载和本地缓存。
- scorer 接到 `lychee_mas.eval.metrics.score(kind, pred, gold)`。
- run 入口继续用 `scripts/run_mas.py` 或后续统一入口。
- 对 agbench 里有用的内容，手工搬运或重写进 LycheeMAS 模块，而不是直接调用 agbench CLI。

优点：

- 最贴合 LycheeMAS 当前框架。
- 结果、metrics、config、case output 都统一。
- 可以继续统计 LycheeMAS 特有指标：memory channel、latent prefix positions、model-call trace、case-level/run-level aggregate。
- 对 public/private 数据可见性更容易控制，因为 agent 只拿到 loader 明确传入的 question/context/attachment。
- benchmark 扩展方式清楚：新增 loader、prepare、scorer、少量测试。

缺点：

- 对 GAIA 这种工具型任务，仅有 loader/scorer 不够，还需要选择 `gaia` 这类工具 team，并准备 Playwright/Docker/网络/视觉模型等依赖。
- 手工复用 agbench 资产会有重复劳动，尤其是 Templates/Tasks 展开逻辑。
- 如果 agbench 官方模板更新，需要人工同步思路或代码。
- 方案二的 native HumanEval scorer 当前不是 Docker 沙盒，只是 subprocess 隔离；`human_eval` 路径已走 Docker 代码执行器。

适用场景：

- GSM8K、AIME、HumanEval 数据/评分这类相对规则的 benchmark。
- 希望统一统计 LycheeMAS 自己指标的实验。
- 作为长期主线的基础层。

这个方案应该继续保留，是当前项目最稳的基本盘。

#### 方案三：把工具能力并回原 AutoGenRuntime 主链路

核心想法：

- 不做 `GAIARuntime` 这种 benchmark 专用 runtime。
- 早期曾新增过通用工具型 AutoGen runtime 作为验证资产；当前文件已删除，能力已并入 `AutoGenRuntime`：

```text
src/lychee_mas/runtime/backends/autogen_tool_runtime.py
```

- 效果优先的主方向不是长期维护额外 runtime 分叉，而是在原 `runtime=autogen` 链路上直接增加工具能力。
- 目标是让同一条 LycheeMAS 主链路同时支持普通文本 MAS、function/tool calling、Coder/Terminal、File/Web、以及 `none/nl_only/latent_only/both`。

当前支持/计划支持的 team 与内部 group-chat 形态：

```text
autogen_static_tools   普通 AssistantAgent + tools/workbench
coder_executor         HumanEval 类代码生成 + 执行器
magentic_one           Orchestrator + FileSurfer + WebSurfer + Coder + ComputerTerminal
swarm                  后续如需 handoff/dynamic routing 再接
```

GAIA 配置形态：

```yaml
runtime:
  name: autogen

run:
  task: gaia_validation_level_1
```

`gaia_validation*` 的默认 team 由 `task_config.py` 自动选为 `gaia`。

HumanEval 配置形态：

```yaml
runtime:
  name: autogen

run:
  task: human_eval
```

`human_eval` 的默认 team 由 `task_config.py` 自动选为 `human_eval`。

实现内容：

- 扩展原 `AutoGenRuntime`，仍返回 LycheeMAS 的 `Trajectory` 和 `final_answer`。
- `AgentSpec.meta` 明确声明 agent 类型、tools、workspace/sandbox 需求、是否需要 browser/file/code executor。
- `_build_agents()` 按 `AgentSpec.meta` 构造普通 `AssistantAgent`、Coder、FileSurfer、WebSurfer、ComputerTerminal，也能给普通 AssistantAgent 绑定 function tools。
- `InjectionClient` 支持 AutoGen `tools` 参数、tool schema、tool_choice、结构化 tool-call 返回、tool result message 回填，并继续做 memory/router/latent 注入。
- `magentic_one` 风格 team 构造 Orchestrator + FileSurfer + WebSurfer + Coder + ComputerTerminal。
- `coder_executor` 风格 team 构造 Coder + ComputerTerminal。
- 对文件型 benchmark，为每个 case 建立可控 workspace，只把公开附件复制进去。
- 记录工具调用、模型调用、最终答案、错误、workspace、复制文件、case wall time 等信息。

优点：

- 这是最通用的能力补齐，不只服务 GAIA。
- GAIA、HumanEval、WebArena、文件问答、网页搜索类 benchmark 都能复用。
- 推理仍在 LycheeMAS runtime 体系里，评分与结果仍可统一落盘。
- 能真正用上 AutoGen 生态里已经实现好的工具 agent，而不是只靠 prompt 伪装角色。
- 设计上比 GAIA 专用 runtime 干净，后续扩展空间大。

注意事项：

- 本地 HF backend 要支持 tool schema 注入和 tool-call 解析；如果模型/chat template 没有原生 tool calling，需要在 `HFBackend` 或 `InjectionClient` 里实现稳定的 tool-call prompt 与 parser。
- API backend 可用于 `none/nl_only` 或工具 smoke；真正 `latent_only/both` 仍要走本地 HF backend。
- GAIA 还可能需要 vision、多模态、网页浏览、PDF/音频处理。
- Docker、Playwright、浏览器依赖、文件权限、网络访问、代码执行安全都要纳入 runtime 配置。
- 工具调用 metrics 当前已有 `tool_call_count`、`tool_error_count`、`case_wall_time_s`；后续还应补逐 tool latency、sandbox exit code、artifact count。

适用场景：

- GAIA 完整能力评测。
- HumanEval 更严格的 coder + executor 评测。
- 所有需要真实文件读取、网页访问、代码执行、多步工具调用的 benchmark。

最终口径：

- 独立 `autogen_tools` runtime 已从正式代码路径移除，只保留历史验证记录。
- 增强后的 `runtime=autogen` 是最终研究链路；HumanEval/GAIA 必须能在这条链路上支持 `none/nl_only/latent_only/both`。

当前实施状态摘要：

- 独立工具 runtime 不再作为正式代码路径；工具能力已并入统一 `runtime=autogen` 主链路。
- `AutoGenRuntime` 支持普通 `AssistantAgent`、Coder、FileSurfer、WebSurfer、ComputerTerminal，并记录 model/tool/case 级信息。
- `InjectionClient` 继续负责 memory/router/latent 注入，同时支持 AutoGen tool schema、tool call 解析和 tool result 回填。
- `templates.py` 的 `Role` / `TEAM_META` 用于声明工具型 team，例如 `human_eval` 和 `gaia`。
- `run_mas.py` 是统一正式入口，支持 `--code-executor`、`--docker-image`、`--code-timeout`、`--max-turns`、`--work-root` 等工具相关参数。

方案三的完整实现要求：

- `src/lychee_mas/runtime/backends/autogen_runtime.py`：扩展 `_build_agents()`，允许 `AgentSpec.meta` 声明 agent 类型，例如 `assistant`、`coder`、`file_surfer`、`web_surfer`、`computer_terminal`；同一个 runtime 内支持普通 AssistantAgent、工具型 agents、function tools、workspace、browser、code executor。
- `src/lychee_mas/runtime/backends/autogen_injection_client.py`：继续负责 memory/router/latent 注入，同时完整支持 AutoGen `tools` / `tool_choice` / structured tool call / tool result message。模型需要调用工具时，`CreateResult.content` 应返回 AutoGen 能识别的 tool-call 结构；工具执行结果进入下一轮上下文时不能丢失语义。
- `src/lychee_mas/runtime/backends/hf_backend.py` 或 `InjectionClient`：为本地 HF 模型补 tool schema prompt、tool-call parser、无效 tool-call 的修复/重试策略，使本地 GPU 也能原生参与工具调用，而不是只能通过 vLLM/API。
- `src/lychee_mas/layers/construct/templates.py`：扩展 `Role` / `AgentSpec.meta`，让 team 可以声明 `agent_type`、`tools`、`workspace_policy`、`executor_policy`、`browser_policy`。
- `scripts/run_mas.py`：保持主入口不变，增加工具相关参数并传给 `AutoGenRuntime`，例如 `--code-executor`、`--docker-image`、`--work-root`、`--web-headless`、`--save-screenshots`。
- team / workspace / tool agent 构造：已吸收旧独立工具链里的 `coder_executor`、`magentic_one` 思路，当前实现落在原 `runtime=autogen` 主链路。
- memory/router/latent 注入：复用 `HFBackend`、`DualChannelMemoryManager`、`fixed_channel_router`、`RoutingContext`。
- tool call：同时支持工具型 participant 和普通 AssistantAgent function calling；不要把其中一个留作后续路线。
- 结果记录：推理阶段实时写 `spans.jsonl`，并把 LycheeMAS `Trajectory`、`model_calls`、`tool_calls` 汇总到 `predictions.jsonl`；评分阶段写 `outputs.jsonl`、`metrics.json`，不要另起一套结果格式。

三种工程路线对比：

| 路线 | 做法 | 优点 | 缺点 | 推荐度 |
|---|---|---|---|---|
| 在普通 `runtime=autogen` 上直接加工具 | 改 `AutoGenRuntime/_build_agents()`，让 AgentSpec/meta 能声明 Coder/FileSurfer/WebSurfer/ComputerTerminal/普通 tools，并让 `InjectionClient` 继续做 memory/router/latent 注入。 | 最贴近现有主链路，`none/nl_only/latent_only/both` 保留最自然，所有 benchmark 结果格式统一。 | 需要把工具 workspace、tool trace、执行器接进现有 runtime。 | 最推荐。 |
| 历史独立 `autogen_tools` runtime | 曾用于快速验证 `coder_executor` 和 `magentic_one` 环境。 | 证明 Docker/Playwright/AutoGen 工具 agent 在运行环境中可用。 | 不接 latent/CDM，已从正式代码路径删除。 | 仅保留历史记录。 |
| 复制普通链路单独改 | 拷贝 `AutoGenRuntime` / run path，专门做工具 CDM 版本。 | 短期最快试验。 | 维护成本高，metrics、bugfix、CLI 参数容易分叉。 | 不推荐长期使用。 |

运行示例：

```bash
cd /path/to/LycheeMAS
export DASHSCOPE_API_KEY=...

$PY scripts/run_mas.py \
  --config configs/benchmarks/human_eval.yaml \
  --runtime autogen \
  --backend api \
  --task human_eval \
  \
  --method none \
  --n 1

$PY scripts/run_mas.py \
  --config configs/benchmarks/gaia.yaml \
  --runtime autogen \
  --backend api \
  --task gaia_validation_level_1 \
  \
  --method none \
  --n 1
```

不要把 API key 写入配置文件或交接文档。正式跑 GAIA 前建议先确认 Playwright browser、Docker 镜像、模型 vision/function-calling 能力都可用。
#### 方案四：做通用 agbench adapter

核心想法：

- 不直接用 agbench CLI 跑完整实验。
- 而是写一个 LycheeMAS adapter，读取 agbench 的 `Tasks/*.jsonl` 和 `Templates/`，把 agbench benchmark 项目转换成 LycheeMAS 可运行的样本/workspace。

建议位置：

```text
src/lychee_mas/eval/agbench_adapter.py
```

职责：

- 读取 agbench `Tasks/*.jsonl`。
- 理解 agbench 的 `template`、`substitutions`、附件复制规则。
- 借鉴 agbench 的 template/case workspace 展开机制，为每个 case 生成隔离工作目录。
- 把 prompt、附件、expected answer 转成 LycheeMAS `{task, kind, question, gold, context, metadata}`。
- 运行仍交给 LycheeMAS runtime，也就是统一 `AutoGenRuntime(runtime=autogen)`。
- 评分和结果仍落到 LycheeMAS 的 `outputs.jsonl`、`metrics.json`、`config.yaml`。

优点：

- 能最大化复用 agbench 官方 benchmark 定义和模板，减少手工搬运。
- 保留 LycheeMAS 统一 metrics 和结果目录。
- 对未来新增官方 agbench benchmark 更友好，不需要每个都从零写 loader。
- 可以把 agbench 的 workspace 展开能力和 LycheeMAS 的 public/private 可见性控制结合起来。

缺点：

- 需要理解并稳定支持 agbench 的任务展开语义。
- agbench 模板通常假设自己在独立目录里运行 `scenario.py`，转成 LycheeMAS runtime 不一定一一对应。
- 如果模板里有很强的脚本逻辑，adapter 可能只能复用 Tasks/附件，不能无损复用 scenario。
- adapter 和主链路工具能力有依赖关系：如果原 `runtime=autogen` 还没有完整工具调用能力，GAIA 这类任务即使能展开也跑不完整。

适用场景：

- 后续希望批量吸收 agbench 官方 benchmark。
- 希望统一复用 agbench Tasks/Templates，但不想放弃 LycheeMAS metrics。
- 在方案三完成后，作为降低 benchmark 接入成本的配套工作。

#### 方案对比与推荐顺序

简短结论：

```text
方案一：最快复现官方例子，但最不像 LycheeMAS 主线。
方案二：当前基本盘，适合继续维护 loader/scorer/metrics。
方案三：正式研究主线，增强原 runtime=autogen，把完整工具调用和 latent/CDM 消融合并。
方案四：长期降本工具，适合在方案三稳定后做。
```

主线决策：

1. 保留方案二作为主线：继续让 benchmark 数据、评分、结果都走 LycheeMAS native eval。
2. 推进方案三：增强原 `runtime=autogen`，完整接入 function/tool calling、Coder/File/Web/Terminal、workspace/sandbox、tool trace，并保持 `none/nl_only/latent_only/both`。
3. 独立 `AutoGenToolRuntime` 已从正式代码路径移除；历史结果只作为环境验证记录。
4. 用方案一做 sanity check：必要时直接跑 agbench CLI，确认官方 GAIA/HumanEval 模板和运行环境没有问题。
5. 方案四作为配套能力：把 agbench adapter 做成通用导入层，降低后续接入更多官方 benchmark 的成本。

---
## 13. 已知问题与 TODO

### P0 / P1

- 不要做 GAIA 专用 runtime；正式目标是在原 `runtime=autogen` 主链路维护 tools/function calling、Coder/File/Web/Terminal、workspace/sandbox、tool trace。
- 旧独立 `AutoGenToolRuntime` / `autogen_tools` 已验证工具环境可用，核心构造和 trace 思路已吸收到原 `runtime=autogen` 主链路。
- 统一 `AutoGenRuntime` 已支持 `human_eval` 和 `gaia`；目标 Python 环境已跑通统一链路 HumanEval/GAIA API smoke，也已跑通本地 HF 工具型 `method=both` 小样本。
- Playwright Chromium 已安装；统一链路下 GAIA validation level 1 的 `gaia` 真实 API smoke 已跑通。
- 当前 GAIA 数据/评分入口已接入，validation 数据已在项目数据目录准备；正式研究目标是在统一 `runtime=autogen` 上扩大 GAIA 样本数，并同时验证 `none/nl_only/latent_only/both`。
- 严格 MAS benchmark 第一批已接入 loader/preparer/scorer；当前 19 个 runnable task 已完成 loader smoke、API backend smoke、本地 HF backend smoke。
- AgentCollabBench 目前是 prompt-compressed 第一版，尚未按样本 topology 动态构造 AutoGen agents；正式严格 MAS 评测需要补 `sample_topology` 动态 `MASGraph`。
- HumanEval `human_eval` 使用 Docker 代码执行器，默认镜像是 `lychee-human-eval:local`；GAIA `gaia` 默认镜像是 `lychee-gaia:local`。
- `scripts/run_experiment.py --runtime autogen` 不是正式入口，容易和 `scripts/run_mas.py` 混淆；建议文档或代码里明确标注。

### 后续优化路线

这部分是当前 benchmark 板块从“能跑”走向“稳定可分析、可汇报、可做正式实验”的主要优化清单。

#### 1. GAIA 工具调度修正

当前 `gaia` team 已经能创建 `FileSurfer`、`WebSurfer`、`Coder`、`ComputerTerminal`，但 Magentic-One Orchestrator 的调度仍可能出现“工具链被调起，但解题路径不正确”的情况。

建议修复：

- 增加 Orchestrator guard：如果下一位 speaker 是 `ComputerTerminal`，但最近没有来自 `Coder` 的代码块，就改为让 `Coder` 先生成完整可执行代码块。
- 把 `No code blocks found` 这种返回标记为 `invalid_tool_call` 或 `code_not_executed`，不要继续算作 `tool_errors=0` 的正常工具调用。
- 对 GAIA 常见 URL 模式增加任务策略提示或轻量规则：看到网页链接、ORCID、附件内 URL 时，优先让 `WebSurfer` 打开网页，而不是只在本地文件字段里找答案。
- 在 `spans.jsonl` / `predictions.jsonl` 里记录 `tool_effective=false`、`tool_invalid_reason=no_code_block` 这类字段，便于后续错误归因。

#### 2. case 级诊断报告

现在 `spans.jsonl` 信息很全，但人工阅读成本高。建议新增：

```bash
python scripts/benchmark_case_report.py <run_dir> --case-id <case_id>
```

或在 `scripts/analyze_benchmark_run.py` 中增加 `--write-case-reports`，输出 `case_reports/<case_id>.md` 或 HTML。

报告建议包含：

- `case_id`、question、gold、prediction、score。
- 角色调用时间线：Orchestrator / FileSurfer / WebSurfer / Coder / ComputerTerminal。
- 每个角色调用次数、总 input/output tokens、总 latency。
- 工具调用列表：是否有效、是否出错、错误原因、关键输出摘要。
- 是否触发 `max_turns`、`max_new_tokens`、JSON repair、prompt truncation。
- 自动错误归因标签，例如 `wrong_answer`、`tool_not_used`、`code_not_executed`。

这样汇报时可以直接展示“为什么错”，而不是只给一个 accuracy。

#### 3. run 状态查看命令

建议新增：

```bash
python scripts/benchmark_status.py <run_dir>
```

输出：

- 当前完成多少 case，总数多少。
- 成功、失败、未完成、已跳过各多少。
- 最近正在跑哪个 case，最后一个 span 是什么。
- 是否可以 `--resume`。
- 最近 N 个错误 case 的错误摘要。
- 当前平均 `case_wall_time_s`、`num_model_calls`、`num_tool_calls`、input/output tokens。

这对长时间 GAIA full run 很重要，避免只能手动 tail log。

#### 4. 错误归因体系

建议把错误分成统一标签，写入 `outputs.jsonl.score_details` 或单独 `diagnostics` 字段：

```text
wrong_answer              最终答案和 gold 不一致
tool_not_used             题目明显需要工具，但没有调用对应工具
invalid_tool_call         工具被调用，但调用内容无效，例如 Terminal 没有代码块
code_not_executed         需要代码执行，但没有真正执行代码
web_failure               WebSurfer 打不开网页、超时、搜索失败
file_not_found            附件没有复制或路径不可读
max_turns_reached         达到最大轮数才收尾
ledger_parse_error        Magentic-One ledger 解析失败
model_truncated           输出撞到 max_new_tokens 或重复 token 防护
runtime_crash             runtime 抛异常退出
scoring_error             分析/评分阶段失败
```

`tool_error_count` 只能表示工具是否报异常，不足以表示工具是否“有效完成任务”。例如 `No code blocks found` 目前不是异常，但对任务来说是无效工具调用。

#### 5. 数据准备完整性 manifest

当前 raw/prepared 已经拆开，并且每个 preparer 有基本 ready 检查。后续可为每个 benchmark 写 manifest：

```text
data/benchmarks/prepared/<benchmark>/manifest.json
```

记录：

- benchmark source / prepare target / runnable task。
- provider、repo id、revision、下载时间。
- raw 文件清单、prepared 文件清单。
- 每个 split 的样本数、关键字段检查结果。
- 是否由 raw 重新生成 prepared。
- checksum 或至少 size/mtime，用于发现 prepared 被误删或 raw/prepared 不一致。

这样可以支持“raw 还在、prepared 被删了，直接从 raw 重新生成”的场景，也方便正式实验复现。

#### 6. 指标分层继续规范

当前已有 `predictions.jsonl`、`spans.jsonl`、`outputs.jsonl`、`metrics.json` 四层，但后续说明和字段可以进一步分层：

```text
case_metrics          每个 case 的最终统计，例如 score、wall time、模型/工具调用数
model_call_metrics    每次 LLM 调用的 input/output tokens、latency、truncation、JSON repair
tool_call_metrics     每次工具调用的 source、有效性、错误、latency、exit code
score_metrics         正确率、子指标、官方 scorer 细节
run_summary           整个 run 的聚合均值、总量、成功/失败计数
```

后续新增字段时优先明确属于哪一层，避免 `outputs.jsonl` 越来越难读。

#### 7. 本地 vLLM API 化

本地 HF 直连能跑，但 GAIA/HumanEval 工具链更适合统一走 OpenAI-compatible API。建议后续把 `models/Qwen3-4B-Instruct-2507` 用 vLLM 起服务，再让 LycheeMAS 走 `--backend api` 指向本地服务。

预期收益：

- API backend 与本地 GPU backend 的工具调用路径更一致。
- 更容易做并发和服务化管理。
- 方便复用 AutoGen 对 OpenAI-compatible model client 的工具/JSON/vision 能力假设。

注意：这不是说本地 GPU 不能跑工具链，而是建议本地 GPU 通过 vLLM 暴露成统一 API，再接入当前 `runtime=autogen` 主链路。

#### 8. Orchestrator 模型单独配置

GAIA 这类任务里，Orchestrator 的调度能力直接影响是否会用对工具。后续可以支持：

```text
Orchestrator: 强模型，例如 qwen-plus / qwen-vl-plus / 更大本地模型
WebSurfer/FileSurfer/Coder: 可用同模型或较便宜模型
ComputerTerminal: 不调模型，只执行代码
```

这样可以用更强模型负责计划、分工和停止判断，用较便宜模型执行子任务。对 GAIA 这种长程工具题，提升可能比只调大 `max_turns` 更明显。

### P2

- 指标已引入 `schema_version: 2` 和新命名字段；旧字段仍保留以兼容旧分析脚本。
- `case_wall_time_s` 已在统一 `runtime=autogen` 链路显式统计。
- `num_correct`、`total_*` 这类 run-level 字段已补第一版，后续可补 cost/currency 字段。
- 对工具型 team，已初步补 `tool_call_count`、`tool_error_count`、`case_wall_time_s`；后续建议继续补 `tool_latency_s_total`、`sandbox_exit_code`、`artifact_count` 等指标。
- agbench 可以作为官方 Tasks/Templates/scenario/sandbox 的资产来源，但不建议直接替代 LycheeMAS native eval 主链路。
- HumanEval 后续可以增加 pass@k、多样本采样、timeout/error 类型统计。

### P3

- 可以给 `prepare_benchmarks.py` 加 `--dry-run`，只打印将要下载的数据源。
- 可以给每个 benchmark 模块加更细的单元测试，mock 下载函数，避免真实网络。
- 可以为 `docs/DEVELOPMENT.md` 加链接指向本文。

---

## 14. 接手人快速检查清单

这一节的目标不是解释全部设计，而是让新接手的人在一台新机器或一个新环境里，尽快判断
benchmark 板块是否“能准备数据、能加载样本、能跑推理、能评分分析”。建议按下面顺序做。

| 步骤 | 要确认什么 | 推荐命令 | 通过标准 |
|---|---|---|---|
| 0 | 进入仓库根目录，并设置通用环境变量。 | 见第 14.1 节。 | 后续命令都从 LycheeMAS 根目录执行；`$PY` 指向目标 Python。 |
| 1 | 代码入口和公开文档是否齐全。 | `git status --short -- docs README.md .gitignore` | 能看到本公共文档；没有误把私有文档、数据、runs 放进提交范围。 |
| 2 | benchmark 数据目录是否存在。 | `find data/benchmarks/raw data/benchmarks/prepared -maxdepth 3 -type f \| head -50` | 能看到 raw/prepared 下的数据文件；如果没有数据，先跑第 3 步。 |
| 3 | 下载/准备数据是否能跑通。 | `$PY scripts/prepare_benchmarks.py --raw-root data/benchmarks/raw --prepared-root data/benchmarks/prepared --source auto --tasks gsm8k,aime_2024,human_eval` | 日志显示 `[prepare] ... ready at ...`；raw/prepared 下出现对应数据。 |
| 4 | loader 是否能读出统一样本格式。 | 见第 14.2 节。 | 每个 task 都能打印 `task/kind/case_id/question`，没有 import 或文件缺失错误。 |
| 5 | scorer 单元测试是否通过。 | `PYTHONPATH=src $PY -m pytest tests/test_eval_benchmark_extensions.py -q` | 测试通过；如果数学 scorer 缺依赖，应先安装 benchmark requirements，而不是改成降级评分。 |
| 6 | 跑一次最小推理。 | 见第 14.3 节。 | 运行目录下生成 `predictions.jsonl` 和 `spans.jsonl`。 |
| 7 | 对已有预测做离线评分分析。 | 见第 14.4 节。 | 生成 `outputs.jsonl` 和 `metrics.json`，能看到准确率、case 数、token/latency 汇总。 |
| 8 | 再扩大规模或接新 benchmark。 | 参考第 7 节命令表和第 11 节扩展说明。 | smoke 先通过，再跑 full；新增 benchmark 时补 loader、preparer、scorer、测试和文档。 |

### 14.1 基础环境变量

```bash
cd /path/to/LycheeMAS

export PY=python
export PYTHONPATH=src
export LYCHEE_BENCHMARK_RAW_ROOT=data/benchmarks/raw
export LYCHEE_BENCHMARK_PREPARED_ROOT=data/benchmarks/prepared
export LYCHEE_BENCHMARK_RUNS_ROOT=runs/benchmarks
export MODEL_PATH=models/Qwen3-4B-Instruct-2507
```

如果是 API backend，还需要设置对应服务的 key 和 base url；如果是本地 HF backend，需要确认
`MODEL_PATH` 指向本地模型目录。HumanEval/GAIA 还需要 Docker sandbox 镜像，准备数据时可用
`--build-images` 一并构建。

### 14.2 loader 最小验证

```bash
PYTHONPATH=src $PY - <<'PY'
from lychee_mas.eval.benchmarks import load

for task in ("gsm8k", "aime_2024", "human_eval"):
    sample = load(task, n=1)[0]
    print(
        task,
        "kind=", sample.get("kind"),
        "case_id=", sample.get("metadata", {}).get("case_id") or sample.get("task"),
        "question_chars=", len(sample.get("question", "")),
    )
PY
```

这个检查只验证“数据能被 loader 读出来”，不调用模型，也不评分。

### 14.3 最小推理 smoke

可以先选一个不需要外部工具的任务，用小样本确认统一 `runtime=autogen` 主链路：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mas.py \
  --config configs/benchmarks/local_hf.yaml \
  --runtime autogen \
  --backend hf \
  --model-path "$MODEL_PATH" \
  --device cuda:0 \
  --task gsm8k \
  --method none \
  --n 3 \
  --max-new-tokens 8192 \
  --runs-root "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_smoke
```

HumanEval/GAIA 这种会用到代码执行或文件/网页工具的任务，应先确认 Docker 镜像存在，再跑对应
benchmark 配置：

```bash
docker images | grep -E 'lychee-human-eval|lychee-gaia'
```

### 14.4 离线评分分析

推理完成后，再单独运行分析脚本：

```bash
$PY scripts/analyze_benchmark_run.py \
  "$LYCHEE_BENCHMARK_RUNS_ROOT"/gsm8k_local_hf_smoke/Qwen3-4B-Instruct-2507/gsm8k_none/gsm8k \
  --score-predictions
```

这样能把“推理”和“评分/汇总”解耦：即使中途停止，只要已经有 `predictions.jsonl`，也可以先对已完成
case 做分析；`spans.jsonl` 则用于排查某个 case 内部的大模型调用、工具执行、错误和耗时。

---

## 15. 按 CLAUDE.md 的合规检查记录

`CLAUDE.md` 是项目开发规范本身，不在 benchmark 板块内修改；本节只记录 benchmark 板块如何对齐它。

### 已检查通过

- AutoGen 隔离：业务层不直接 import AutoGen；AutoGen imports 留在 `src/lychee_mas/runtime/backends/autogen_*.py`。
- 重依赖懒加载：`import lychee_mas` 应输出 `HEAVY LOADED: NONE`。
- `lychee_mas.eval.metrics` 不在 import 时加载 `sympy/regex/latex2sympy2_extended/word2number`；AIME 数学评分只在真正调用 `score_aime()` 时加载这些依赖。
- benchmark loader 的数据依赖保持函数内导入；`datasets/modelscope/huggingface_hub/pandas/pyarrow` 不会因为 `import lychee_mas` 被加载。
- 新 benchmark 运行配置不在 `configs/benchmarks/local_hf.yaml` 写死本地模型绝对路径；本地 HF 模型路径由 `LYCHEE_HF_MODEL` 或 `run_mas.py --model-path` 提供。
- `scripts/run_benchmark_batch.py --model-path` 默认读取 `LYCHEE_HF_MODEL`。
- `.gitignore` 应覆盖 `runs/`、`*.jsonl`、`checkpoints/`、`data/benchmarks/`、`__pycache__/`、`*.pyc`，避免 benchmark 数据、结果和缓存入库。

### 本次验证命令

```bash
PYTHONPATH=src python - <<'PY'
import sys
import lychee_mas
heavy = [m for m in (
    "torch","transformers","autogen_core","autogen_agentchat","numpy","yaml",
    "sympy","datasets","pandas","pyarrow","modelscope","regex",
    "latex2sympy2_extended","word2number"
) if m in sys.modules]
print("HEAVY LOADED:", heavy or "NONE")
PY

PYTHONPATH=src python -m py_compile \
  src/lychee_mas/eval/math_parsing_util.py \
  src/lychee_mas/eval/metrics.py \
  scripts/run_benchmark_batch.py \
  scripts/run_mas.py \
  scripts/prepare_benchmarks.py \
  scripts/analyze_benchmark_run.py
```

### 仍需注意

- 根目录历史配置里可能仍有旧实验路径或旧模型路径，例如早期 AIME/CDM 配置。它们属于原 LycheeMAS/CDM 历史配置，不是 benchmark 新增板块；若后续统一工程规范，建议单独迁移为环境变量或示例模板。
- 正式提交前建议在目标环境里跑 `make lint && make test && make selfcheck`。
