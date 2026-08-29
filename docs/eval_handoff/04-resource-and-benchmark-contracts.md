# 4.3 资源注册与 Benchmark 模块

[上一部分：接口与生命周期](04-interface-and-lifecycle-contracts.md) · [返回第 4 章索引](04-implementation-contracts.md) · [下一部分：TeamSpec v14](04-team-spec-v14-contract.md)

### 4.3.1 资源注册目录与边界

资源中心按对象类型分目录，不再使用混合的 `configs/eval_studio/resources/`：

```text
configs/eval_studio/
├── models/{specs,instances}/
├── apis/{specs,instances}/
├── benchmarks/{specs,instances}/
├── pricing/{specs,instances}/
├── deployments/{specs,instances}/
├── teams/{specs,instances}/
└── experiments/{specs,instances}/
```

| 对象 | Spec 保存什么 | Instance 保存什么 | 可用性检查 |
|---|---|---|---|
| Model | organization、family、size、license、固有 capabilities、provider aliases、注册来源 | acquisition、真实 path、managed、Spec fingerprint | 路径、模型关键文件、loader 能力 |
| API Access | provider、protocol、endpoint 模板、auth mode、request limits、允许的模型选择规则；不重新定义模型能力 | 解析后的 endpoint、凭据引用、实际限制、协议探测与 Spec fingerprint | 最小请求；凭据本身不落盘到公开字段 |
| Benchmark | id、name、category 等稳定轻量身份 | Raw/Prepared 路径、manifest、supported tasks、provenance、integrity | manifest + loader check |
| Pricing | basis、billing mode、currency、rate card 和适用部署 | Spec fingerprint 与冻结价格版本 | 完整 rate、模型/部署兼容性 |

Model 和 Benchmark 都支持“注册来源下载、扫描 managed root、引用外部路径”。外部引用不复制、不建 symlink；Instance 中 `managed=false`。只有显式删除文件操作才能删除外部资产。API 只能从已注册 APISpec 实例化，Deployment 不能临时绕过资源中心输入任意密钥或模型路径。

三者组合关系必须保持单向：`ModelSpec -> ModelInstance` 定义“是什么模型以及文件在哪里”，`APISpec -> APIInstance` 定义“通过什么访问产品、协议和凭据连接”，`DeploymentSpec -> DeploymentInstance` 再选择一个 ModelSpec，并按 `local_hf/vllm/api` 选择 ModelInstance 或 APIInstance。API 返回的 observed capability 可以收紧一次 Deployment 的有效能力，但不能回写并覆盖 ModelSpec 的固有能力。

### 4.3.2 Benchmark 实现合同

每个 benchmark 类以 `src/lychee_mas/eval/benchmarks/base.py` 的 `Benchmark` 为公共合同。最小闭环仍是 `prepare/load/score`，但工具型任务和批量 evaluator 还可以实现扩展钩子：

```python
class Benchmark:
    id: str
    name: str
    category: str
    sources: dict

    def descriptor(self) -> dict: ...
    def prepare(self, target: str, force: bool, source: str) -> str: ...
    def load(self, task: str) -> list[BenchmarkCase]: ...
    def score(self, prediction: str, case: BenchmarkCase) -> dict: ...

    # 工具型或带外部 harness 的 benchmark 按需覆盖
    def materialize_case(self, query, workspace, task_text) -> CaseMaterialization: ...
    def create_tools(self, slot, query_meta) -> ToolBundle: ...
    def collect_prediction(self, query, messages, workspace, default_text) -> str: ...
    def prepare_evaluation(self, predictions, gold_by_id, context) -> None: ...
    def aggregate(self, samples, metrics) -> dict: ...
```

| 合同 | 责任 | 何时必须实现 |
|---|---|---|
| `descriptor()` | 向 Studio 只读暴露来源、Prepare target、task、scorer、能力、沙盒、网络和评分 profile | 所有 Benchmark；由基类统一生成 |
| `prepare()` | 按注册来源下载、校验、转换并形成 prepared 数据 | 所有 Benchmark |
| `load()` | 只读 prepared 数据并产生统一 `BenchmarkCase` | 所有 Benchmark |
| `score()` | 对单条 Prediction 调用 benchmark 原生评分语义 | 所有可逐 Trial 评分的 Benchmark |
| `materialize_case()` | 把附件、仓库或任务文件放入隔离 workspace，并生成真实任务文本 | 需要文件、代码仓库或沙盒的 Benchmark |
| `create_tools()` | 为 TeamSpec 声明的工具槽创建 benchmark 专属工具 | 需要 benchmark 原生工具/API 的 Benchmark |
| `collect_prediction()` | 从消息、workspace 或产物中取得最终提交 | 最终结果不只存在于最后一条文本消息时 |
| `prepare_evaluation()` | 在批量 scorer/harness 前准备预测文件、gold 和外部评测环境 | HLE、SWE-bench Verified 等 batch-final 流程 |
| `aggregate()` | 按官方 task/domain 权重补充 Run 级聚合 | 官方存在非简单均值聚合时 |

`BenchmarkCase` 的标准字段为 `task`、`kind`、`question`、`gold`、`context`、`metadata`；其中前四项必须存在，`context` 和 `metadata` 可以为空。`metadata` 不是“随便放什么都行”的隐形 API：它保存 case ID、level、附件、domain 等 benchmark 特有但稳定的信息，未知扩展字段进入 `extensions`。Runner 不应该重新实现答案规则，也不能把 gold 或 scorer 逻辑塞进 TeamSpec。

### 4.3.3 数据目录

```text
data/benchmarks/
├── raw/<benchmark>/<provider>/<owner--repo>/
└── prepared/<benchmark>/
```

raw 保留来源和发布者，prepared 只按 benchmark 组织。prepared 被删除但 raw 完整时，`prepare` 应重新转换，不重复联网下载。外部导入以路径引用，不复制也不创建 symlink；解除引用只删注册关系，显式“删除文件”才删外部数据。

### 4.3.4 重点 benchmark

| BenchmarkSpec | 当前任务 | 官方评分 | 运行环境 |
|---|---|---|---|
| `gaia` | validation 及 level 子集 | GAIA 短答案归一化 | 文件、网页、代码、浏览器 |
| `bbeh` | full / mini | 官方 `evaluate.py` 规则 | 文本推理 |
| `swe_bench_verified` | Verified 500 | `FAIL_TO_PASS` 与 `PASS_TO_PASS` Docker harness | 仓库编辑、终端、隐藏测试 |
| `workbench` | 注册版本的完整任务集 | 官方 completion 与 side-effect evaluator | workplace 工具沙盒 |
| `livecodebench` | official code_generation_lite release_v6，175 题 | 固定 LiveCodeBench commit 的 extraction + APPS checker；Run 级补充 pass@k | Python 代码生成；私有测试只进入 gold 与隔离 checker |
| `hle_verified` | HLE-Verified Gold subset，668 题 | 原 HLE 结构化 judge、正确率与校准指标 | 多模态问答；图片必须以模型可见多模态内容保留 |

本轮不再接入原始 gated HLE，而采用公开的 HLE-Verified Gold。Gold 是 HLE-Verified 中经过人工确认、适合稳定评测的 668 题子集；Revision 和 Uncertain 暂不混入主结果。Hugging Face 上游是身份与数据真源；ModelScope 的公开搬运缺失图片，只能作为可发现的候选渠道，不能用于正式多模态评测。LiveCodeBench 使用官方精简代码生成集，而不是将仓库中所有任务族拼成一个新口径；数据 revision、release 过滤和 checker commit 均写入 manifest。

多模态附件的 canonical 输入仍是原始 `data:image/<format>;base64,...` URI。框架适配器不得按 PNG/JPEG 前缀自行缩窄格式：统一输入边界先验证 Base64，再由 Pillow 解码 GIF、WebP、PNG、JPEG 等受支持像素，最后转换为框架原生图像对象。原始 URI 保留在无损 Event/BenchmarkCase 中，解码只服务模型输入，不能改写数据真源。

#### 4.3.4.1 SWE-bench 任务镜像缓存

> **当前合同。** SWE-bench Verified 的官方 harness 为不同 case 使用独立任务镜像。这些镜像会共享基础层，但任务层仍会随运行覆盖面持续累积，因此不能把“harness 删除了容器”等同于“磁盘缓存已经回收”。

`ProjectDockerImageCache` 只管理能够由 LycheeMAS `official_evaluation/swe_bench_verified/predictions.jsonl` 和 prepared dataset 建立所有权证据的镜像。它不执行全局 `docker image prune`，不采用仓库名前缀猜测其他用户的镜像，也不删除仍被容器或活跃 lease 引用的镜像。manifest 位于 `runs/eval_studio/docker_cache/swe_bench_verified.json`，记录 image ref、最近使用时间、运行证据、活跃 lease 和 eviction 历史。

默认策略为有界 LRU：最多保留 24 个已管理任务镜像；Docker 所在分区可用空间低于 12 GiB 时继续淘汰到约 20 GiB。官方 harness 按缓存容量把全量预测拆成有界分批，每批启动前执行一次回收并为该批所需镜像建立 lease；该批容器全部清理后释放 lease，再执行一次回收。分批只改变评分执行的资源生命周期，最终 resolved/error 集合仍聚合回同一次 Evaluation。进程被强制终止时，下一次运行会识别失效 PID lease 并恢复清理。三种策略为：

| 模式 | 行为 | 用途 |
|---|---|---|
| `bounded` | 按数量上限和磁盘水位 LRU 回收 | 默认正式运行 |
| `keep` | 只登记，不主动删除 | 临时诊断镜像复用问题 |
| `remove_after_use` | 本轮容器清理后删除所有未受保护的已管理任务镜像 | 极端小磁盘环境 |

清理边界必须区分两类镜像：`lychee-agbench-gaia`、`lychee-python-sandbox` 等固定项目基础镜像由 Docker image prepare 合同管理；`swebench/sweb.eval.*` case 镜像由本节的任务缓存管理。固定基础镜像不能因为 LRU 被当作 SWE 任务镜像删除。

### 4.3.5 已注册 Benchmark 全表

Benchmark 下载和运行链包含几个容易混淆的名字：

| 概念 | 回答的问题 | 示例 |
|---|---|---|
| Benchmark implementation / source | 由哪个 Benchmark 类拥有来源、转换、加载和评分合同 | `gaia`、`agent_collab` |
| Download source | Raw 从哪个平台和仓库取得 | ModelScope、Hugging Face、GitHub、other/fallback |
| Prepare target / alias | `prepare_benchmarks.py --tasks` 要准备哪个完整集或子集 | `gaia`、`gaia_validation_level_2` |
| Runnable task / loader | `run_mas.py --task` 要加载并运行哪组 Case | `gaia_validation_level_2` |
| Scorer kind | 该 Runnable task 使用哪种官方评分适配 | `gaia`、`mc`、`human_eval` |

Prepare target 决定准备动作，不等于一定只下载该子集：上游仓库只提供整包时，Raw 可以下载全量，再由 converter 生成对应 Prepared 子集；是否全量下载由该来源合同决定并写入 manifest。多个下载来源也不能共享一套“猜测式转换”，每个注册来源先做来源适配，再经过统一 Prepared validator。

下表反映当前 `BENCHMARK_STRUCTURE`，用于避免只看到四个研究重点后误以为其它 Adapter 已被删除。Prepare target、Runnable task 和 scorer kind 是三个不同层级；存在多个子任务时按行展示对应关系。

| Benchmark | Full prepare target | Runnable task(s) | Scorer kind(s) | 主要用途 |
|---|---|---|---|---|
| GSM8K | `gsm8k` | `gsm8k` | `exact` | 数学推理 |
| AIME 2024 | `aime_2024` | `aime_2024` | `aime` | 数学等价评分 |
| ARC-Easy | `arc_easy` | `arc_easy` | `mc` | 选择题推理 |
| OpenBookQA | `openbookqa` | `openbookqa` | `mc` | 开放书本选择题 |
| MedQA | `medqa` | `medqa` | `mc` | 医学选择题 |
| LoCoMo10 | `locomo10` | `locomo10` | `f1` | 长期对话记忆 |
| HumanEval | `human_eval` | `human_eval` | `human_eval` | 代码生成与执行测试 |
| GAIA | `gaia` | validation、L1、L2、L3 四个 runnable task | `gaia` | 通用助手、Web/File/Code |
| AFTraj-2K | `aftraj` | `aftraj_audit`, `aftraj_audit_test` | `mas_audit` | Agent 轨迹审计 |
| AgentCollabBench | `agent_collab` | IDR、RTD、CPR、CLC 四个 task | instruction decay、tracer durability、consensus pollution、context leakage | MAS 协作诊断 |
| MAST-Data | `mast_data` | `mast_failure` | `mas_failure_taxonomy` | 失败分类数据 |
| Open Agent Traces | `open_agent_traces` | `open_agent_traces` | `mas_deviation` | 轨迹偏差分析 |
| BIG-Bench Extra Hard | `bbeh` | `bbeh` | `bbeh` | 高难推理 |
| Humanity's Last Exam | `hle` | `hle` | `hle` | 高难知识/推理；受访问条件限制 |
| SWE-bench Verified | `swe_bench_verified` | `swe_bench_verified` | `swe_bench_verified` | 软件工程与隐藏测试 |
| WorkBench Revisited | `workbench` | `workbench` | `workbench` | 办公工具与状态变更 |

权威逐行映射由 `python scripts/prepare_benchmarks.py --list-benchmark-structure` 生成；文档表用于理解，不替代注册表输出。

### 4.3.6 Case 抽样

- `head`：从 `start_index` 连续取，适合调试某个已知 case。
- `uniform`：在全体 case 的索引范围均匀取样，避免只覆盖文件开头。
- `stratified`：按 level、domain 或 task 等字段分层取样，适合 smoke 和小规模比较。
- `cases=all`：完整数据集。

正式结果必须报告抽样方法、case 数、数据版本和 scorer 版本。不同 level 各取相同数量时得到的是“level-balanced pilot”，不能冒充 GAIA 165 题官方总准确率。

### 4.3.7 官方评分、失败分母与污染审计

评分与推理解耦，但二者可以流水并行：Runner 只提交 Prediction，分析进程对 terminal Trial 增量调用 Benchmark `score()`；Run 结束后再做一次幂等全量核对。评分遵循以下统一原则：

1. 优先调用 benchmark 官方 scorer、harness 或其语义等价适配，不能为了让 smoke 通过而换成简化 fallback。
2. Trial 成功执行但答案错误时写 `evaluation.completed` 和原始分数；scorer 自身失败时写 `evaluation.failed`，不能把评分异常伪装成模型答错。
3. 超时、崩溃、工具链失败和无 Prediction 的 Trial 仍保留在正式分母，并分别报告 runtime success 与 official score。
4. 多个 task、domain 或 level 的聚合必须使用 benchmark 官方权重；不存在官方总体口径时分别报告，不擅自平均成一个“总分”。
5. 失败或中断 Run 已经 terminal 的 Trial 可以立即评分；`metrics.json` 和 Study 是可重建产物，不要求整个 Run 正常结束后才有分析。
6. GAIA 等可通过网页搜到答案的任务同时运行 contamination audit。污染标记是结果解释条件，不自动改写 Prediction 或删除 Trial；正式报告需同时给原始官方分数与受污染范围。

因此，“推理完成率”“评分成功率”“官方正确率”是三个不同量。只展示后者会隐藏 runtime 或 scorer 的系统性失败，只展示成功评分样本则会产生幸存者偏差。

批量官方评测产物与统一 Evaluation 投影也必须区分。以 LiveCodeBench 为例，`official_evaluation/livecodebench_results.jsonl` 只包含实际送入官方 checker 的成功 Prediction；Runtime timeout、崩溃或无 Prediction 的 Trial 不会伪造一条 checker 输入，但必须由统一 `evaluation.completed` 以零分保留在正式分母。HLE Judge 的 attempts 是逐次请求审计，responses 是每个有效请求的成功 cache，最终状态仍以 EventLog 中每个 Trial 最新的 Evaluation 为准。

<a id="teamspec-contract"></a>
