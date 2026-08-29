# 4.8–4.9 Observability、Evidence 与 RunEvent

[上一部分：Experiment 与 Scheduler](04-experiment-and-scheduler-contracts.md) · [返回第 4 章索引](04-implementation-contracts.md) · [下一部分：Eval Studio](04-studio-contracts.md)

## 4.8 EventLog、投影、Evidence 与指标流水线

### 4.8.1 唯一事实来源

`events/` 是每个 Run 下唯一的无损运行日志目录：

```text
events/
├── run_events.jsonl
├── run_events.000001.jsonl
└── run_events.000002.jsonl
```

达到 `event_log_max_events_per_file` 或 `event_log_max_mib_per_file` 后滚动到下一片。读取器按编号拼接；分片不改变 event 顺序和 event_id。

公共包络的真实顺序为：`schema_version`、`run_id`、`seq`、`event_id`、`event_type`、`timestamp_unix_s`、`timestamp_utc`、`experiment_instance_id`、`worker_id`、`case_id`、`dataset_index`、`trial_index`、`attempt`、`operation_id`、`parent_event_id`、`correlation_id`、`payload`。actor、status、输入输出和错误详情属于具体事件的 payload 或派生投影，不伪造为所有事件都有的公共字段。

主要事件：Run/Trial/Attempt 生命周期，runtime/group-chat 生命周期，agent message，model call start/end/error，tool call/result/error，workspace/browser/code executor，以及 `evaluation.completed/failed`。Prediction 不是单独事件，它只存在于 `trial.completed.payload.final_output`。当前合同也没有虚构 `evaluation.started`；官方评分成功或失败时直接追加 terminal Evaluation Event。Studio 管理的 Run 还会记录 `trial.admission_updated`，其 payload 包含 Scheduler 已发出的累计许可 `admission_total` 和 Runner 已消费的累计许可 `consumed_total`，用于复核“启动中”到“运行中”的跨进程控制过程。

### 4.8.2 两个主要视图

Result Projection：每个 Trial 一行，关联 Prediction、Evaluation、token、延迟、错误和最终状态。它回答“每道题答了什么、得几分、花多少资源”。

Execution Trace：按因果顺序串起 GroupChat、模型调用、工具、错误和重试。它回答“系统为什么得到这个结果、失败发生在哪一步”。Trace API 只返回轻量节点、父子引用、Event ID/seq 引用和摘要；完整 payload 留在 EventLog。旧实现把父节点的完整 children 与原始 Event 递归复制到每一页，20 个节点可返回约 24 MB，现已移除这种重复，分页才真正限制网络与浏览器负担。

原始事件和 Evidence 是研究/调试视图：EventLog 展示无损事实，Evidence 展示指标计算真正读取的规范化证据。五个视图是同一日志的不同投影，不是五份互相竞争的记录格式。

Studio 顶部固定显示五个并列视图按钮。`重新分析` 只出现在指标标题旁；`从头回放` 位于轨迹底部导航器左侧。运行记录列表路径必须换行或省略中段，不能溢出侧栏。


### 4.8.3 EvidenceNormalizer 与 Coverage Report

`eval/evaluation/evidence/normalizer.py` 只从权威 EventLog 读取，生成两个可删除、可重建的派生产物：

```text
<run_dir>/evidence.jsonl
<run_dir>/evidence_coverage.json
```

`evidence.jsonl` 把三框架的 Event 规范化成指标容易消费的证据记录，并保存 `source_artifact`、`source_line` 和原 Event ID。它不是第二份无损日志：模型完整上下文、provider request/response、traceback、代码块和大型工具输出仍只保留在 EventLog，Evidence 通过 provenance 回到事实源，避免再次复制大 payload。

`evidence_coverage.json` 按字段和事件类型报告证据是否完整，例如事件身份、时间、Case/Trial、消息 actor/receiver、模型 usage、工具 parent 和 Evaluation。它是 Metric applicability 的门禁：

| 状态 | 含义 |
|---|---|
| `measured` | 适用且必需证据完整，公式已成功执行 |
| `not_applicable` | 该团队/任务没有这种测量机会，例如单 Node 没有角色切换 |
| `missing_evidence` | 理论上适用，但当前 Event 缺少必需字段 |
| `unsupported` | 当前 evaluator/adapter 不支持该构念 |
| `evaluator_error` | 证据存在，但 evaluator 执行失败 |

真实值为 0 只能写 `measured: 0`，不能拿 `0` 代替后四种状态。Coverage Report 因此回答“这些日志足不足以算”，Metric Contract 回答“这个指标在这里应不应该算”，MetricEvaluator 才回答“数值是多少”。

### 4.8.4 四类标准化指标

代码中的正式类别是 Task、Coordination、Efficiency、Reliability。当前 core Evaluation Profile 对每个 terminal Trial 生成以下 28 项 MetricObservation：

| 类别 | Metric ID | 中文含义 | 单位/方向 |
|---|---|---|---|
| Task | `task.official_score` | Benchmark 官方评分 | score，通常越高越好 |
| Coordination | `coordination.message_repetition_rate` | 团队消息规范化后重复的比例 | ratio，需结合任务解释 |
| Coordination | `coordination.role_participation_balance` | 多 Node 发言分布的归一化均衡度 | ratio；单 Node 为 N/A |
| Coordination | `coordination.model_call_participation_balance` | 多个模型 Node 的调用分布均衡度 | normalized index；少于两个模型 Node 为 N/A |
| Coordination | `coordination.controller_model_call_ratio` | 控制 Node 调用数占全部模型调用的比例 | ratio；要求每次调用显式记录 controller |
| Coordination | `coordination.controller_output_token_ratio` | 控制 Node 输出 token 占全部输出 token 的比例 | ratio |
| Coordination | `coordination.coordination_operation_count` | plan/delegate/monitor 等中性协调操作总数 | operation count |
| Coordination | `coordination.replan_count` | 显式 replan 操作次数 | count；没有 replan 合同为 N/A |
| Coordination | `coordination.stall_rate` | stall 检查中被判定 stalled 的比例 | ratio；依赖 TeamSpec stall 合同 |
| Coordination | `coordination.role_switch_rate` | 相邻团队消息更换发言 Node 的比例 | ratio；单 Node/不足两条消息为 N/A |
| Coordination | `coordination.active_role_count` | 至少发布一条团队消息的 Node 数 | count |
| Coordination | `coordination.agent_message_count` | 团队 Node 发布的非工具 GroupChat 消息数 | count |
| Efficiency | `efficiency.input_tokens` | Trial 全部模型调用输入 token | token，越低不一定越好 |
| Efficiency | `efficiency.output_tokens` | Trial 全部模型调用输出 token | token |
| Efficiency | `efficiency.model_call_latency` | 已完成模型请求时延之和 | second |
| Efficiency | `efficiency.provider_queue_time` | provider/vLLM 请求排队时间之和 | second；provider 不返回则 missing |
| Efficiency | `efficiency.time_to_first_token` | provider 调度到首 token 时间之和 | second |
| Efficiency | `efficiency.generation_throughput` | 有生成时延证据的总输出 token / 总生成时间 | token/s，越高越好 |
| Efficiency | `efficiency.model_call_count` | 已完成的真实模型请求数 | count |
| Efficiency | `efficiency.tool_execution_count` | 已观察到的 terminal 工具执行数 | count |
| Efficiency | `efficiency.tool_execution_latency` | terminal 工具执行耗时之和 | second |
| Efficiency | `efficiency.trial_wall_time` | Trial 从开始到 terminal 的端到端墙钟 | second |
| Reliability | `reliability.tool_error_rate` | 工具错误数 / 工具执行数 | ratio；无工具为 N/A |
| Reliability | `reliability.model_call_error_rate` | 失败模型调用 / 全部 terminal 模型调用 | ratio |
| Reliability | `reliability.trial_retry_count` | 首次 Attempt 之后新增的重试次数 | count |
| Reliability | `reliability.trial_runtime_success` | Trial terminal Event 是 `trial.completed` 而不是 `trial.failed` | binary |
| Reliability | `reliability.result_contract_valid` | Runtime 产物是否满足 Benchmark 所需结果合同 | binary |
| Reliability | `reliability.post_tool_failure_trial_completion` | 至少一次工具失败后 Trial 是否仍完成 | binary；无工具失败为 N/A |

协调指标排除初始 user task 和 system message，但 Orchestrator、Selector、普通 participant 与工具型 participant 的真实非工具发言都计入。每个 MetricObservation 必须是 `measured`、`not_applicable`、`missing_evidence`、`unsupported` 或 `evaluator_error`。零不是“没有证据”，不能用 0 填空。

`trial_runtime_success` 只测 Runtime 生命周期是否正常结束，不验证 benchmark 所需结果是否有效；`result_contract_valid` 只测交付物形状与来源是否满足合同，也不等于答案正确。二者必须与 `task.official_score` 分开报告。

本轮新增指标全部是**离线、描述性的过程观测**：它们从 EventLog 重新计算，不在运行中改变调度或模型行为，也不能直接解释角色的因果贡献。`stall_rate` 衡量当前协调器自己报告 stalled 的频率，不代表客观任务停滞；`post_tool_failure_trial_completion` 只表示 Trial 最终完成，不证明失败工具已被同工具重试修复。跨框架比较时还必须同时报告 evidence coverage：若 TeamSpec 声明了 operation 而 adapter 没有发出中性事件，应写 `missing_evidence`，不能用 0 冒充“没有重规划”。

这些设计借鉴了 [MultiAgentBench](https://aclanthology.org/2025.acl-long.421/) 对任务与协作过程分开评估的方向、[MAST](https://arxiv.org/abs/2503.13657) 对轨迹失败模式的系统分类，以及 [MAESTRO](https://arxiv.org/abs/2601.00481) 对框架无关 trace 和可靠性/时延信号的强调；但当前确定性指标不等同于 MultiAgentBench 的 LLM-judge CScore/PScore，也不等同于 MAST 的语义失败标注。论文指标的公式、证据要求、适用边界和分层接入路线见 [3.3.1](03-related-work.md#process-metrics-literature)。

当前实现与后续计划应分开阅读：

| 层级 | 已实现 | 后续计划 | 关键限制 |
|---|---|---|---|
| Trial 内确定性轨迹 | 消息重复、发言/调用均衡、控制调用与 token 占比、operation/replan/stall、角色切换、活跃角色、消息数 | message density、协调开销、success-per-token | 已实现项只描述发生了什么，不直接表示协作质量 |
| Trial 内运行效率与可靠性 | token、模型/工具调用与时延、queue/TTFT/吞吐、墙钟、工具/模型错误、重试、Runtime 与 ResultContract、工具失败后完成 | CPU/RSS/网络/浏览器资源时间序列 | provider 不提供的字段必须标 `missing_evidence` |
| Study 级结构稳定性 | EventLog 已具备所需调用与顺序证据 | 同 Case 重复 Trial 的调用图 Jaccard、调用序列 LCS、方差、置信区间 | 单次 Trial 不适用 |
| 配对 SAS/MAS 对照 | 尚未形成正式 Profile | error absorption/amplification、相对 token/时延开销、质量—成本 Pareto | 必须冻结同 Case 和模型配方 |
| 语义协作质量 | 尚未进入 core | milestone contribution、Communication/Planning Score、冗余、矛盾、信息增益、CORE 实验 Profile | 需要校准集、judge 版本、prompt 和适用性验证 |
| 失败机制 | 底层异常和恢复事实已实现 | MAST 类 failure taxonomy、首次语义错误及传播 | 需要人工标注小集与 judge 一致性校准 |

### 4.8.5 研究术语

- Prediction：模型或团队对一个 Trial 产生的候选答案。
- Evaluation：用 scorer 对 Prediction 判断正确性。
- Trace：该 Trial 的执行历史。
- Evidence：从 EventLog 规范化后供指标计算的事实。
- Metric：基于 Evidence 的一个定义、公式、适用范围和单位完整合同。
- Study：多个 Run 的受控对比，不是单个 metrics.json。

## 4.9 RunEvent 完整合同

### 4.9.1 物理分片与公共包络

EventLog 是唯一、无损、追加式结构化事实源。全部分片位于 Run 的 `events/` 目录：首片为 `run_events.jsonl`，后续为 `run_events.000001.jsonl`、`run_events.000002.jsonl`。默认每片最多 10,000 个 Event 或约 64 MiB；`iter_run_events()`、resume、评分、Evidence、投影、Studio 分页和 SSE 都按 `seq` 读取整个逻辑日志。

Event 归属还必须满足以下不变量：

1. 发起可等待、可取消的模型或工具操作前，冻结 `case_id/dataset_index/trial_index/attempt/worker_id`；terminal Event 使用同一份冻结身份，不能在 `await` 返回后重新读取可变 Context。
2. 同一 `operation_id` 的 started/completed/failed Event 必须属于同一个 Trial；任何跨 Trial 配对都是运行过程无效，而不是普通统计误差。
3. Trial 失败或超时后，Runner 必须回收该 worker 的 Runtime/Context。Python 取消等待 `asyncio.to_thread()` 时不能杀死底层线程，继续复用 Context 会使迟到输出串入下一 Trial。
4. Trial deadline 必须下沉到 provider request timeout；外层 `wait_for` 只是最后一道边界，不能让网络请求在 Trial 已终止后继续长时间占用资源。
5. EventLog 是追加历史。Evaluation 重试会同时保留早期 `evaluation.failed` 和后续 `evaluation.completed`；当前评分状态按 `(case_id, trial_index)` 的最后一个 Evaluation Event 投影，不能直接统计全文件中的 failed 行数。
6. Runtime 的建队、框架执行、资源清理和结果投影共享同一个最外层 terminal guard。只要 `runtime.started` 已存在，任一阶段退出都必须写入同 operation ID 的 `runtime.completed`、`runtime.failed` 或 `runtime.cancelled`。

“无损”指合同声明的模型消息、provider 请求/响应、Agent 消息、工具输入输出、异常、Prediction 与 Evaluation 不因记录模式被省略；它不承诺保存权重、KV cache、GPU tensor、浏览器内存或操作系统快照。系统没有 compact/full 两种写入模式，只有 UI 层的 preview、折叠、分页和字段选择。

无损不等于每条 JSON 都必须自包含所有历史副本。长负载采用两层物理表示：

```text
<run_dir>/
├── events/
│   ├── run_events.jsonl
│   └── run_events.000001.jsonl
├── payloads/
│   └── sha256/<前两位>/<sha256>.json.gz
└── artifacts/
```

- Event JSONL 保存公共包络、小 payload 和 `$payload_ref`；
- 超过阈值的大字符串或结构化值按 canonical JSON 编码，计算 SHA-256，gzip 压缩后写入当前 Run 的 payload store；
- 同一终端输出或消息快照重复出现时只保存一份，引用包含哈希、原始字节数、编码和 Run 内相对路径；
- `iter_run_events(materialize_payloads=True)` 默认透明还原并校验哈希，原有 scorer/Evidence/Projection 读取的是完整值；Studio 列表和索引使用 `False`，只在展开单条 Event 时按需读取；
- payload store 是 Run 本身的一部分，不是外部数据库，也不是可丢失的缓存。移动或归档 Run 必须连同 `events/`、`payloads/` 和 `artifacts/` 一起移动。

这解决的是物理重复，不修改语义输出，也不丢弃信息。工具输出仍有独立的“原始事实”和“交给模型的有界副本”：`output` 指向完整 payload，`delivered_output` 是 TeamSpec 工具交付策略实际送入下一轮上下文的内容。二者不能互相替代。

| 公共字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | integer | 当前 RunEvent 合同版本，当前为 2 |
| `run_id` | string | 所属 Run |
| `seq` | integer | 逻辑 EventLog 内严格递增序号 |
| `event_id` | string | 当前 Event 唯一 ID |
| `event_type` | string | 点分类型，例如 `model_call.completed` |
| `timestamp_unix_s` | number | Unix 秒时间戳 |
| `timestamp_utc` | string | UTC ISO-8601 时间 |
| `experiment_instance_id` | string/null | 发起 Run 的 ExperimentInstance |
| `worker_id` | integer/null | 并发 worker 标识 |
| `case_id`、`dataset_index` | string/integer/null | Case 身份与 loader 索引 |
| `trial_index`、`attempt` | integer/null | Trial 与 Attempt 索引 |
| `operation_id` | string/null | started/terminal 生命周期配对 ID |
| `parent_event_id` | string/null | 因果或嵌套父 Event |
| `correlation_id` | string/null | tool call 等协议关联 ID |
| `payload` | object | 当前 Event 类型专属字段 |

`*.started` 未显式给 `operation_id` 时，writer 使用该 Event 的 `event_id`；terminal Event 必须复用它。RoutingContext 还会在 Trial 内 payload 中附加 `base_seed`、`trial_seed` 和 `seed_derivation`。

```text
run.started
└── trial.started
    └── attempt.started
        └── runtime.started ... completed/failed/cancelled
            ├── workspace.prepared
            ├── code_executor.started ... stopped
            ├── web.context_startup.started ... completed/failed
            └── group_chat.started ... completed/failed/cancelled
                ├── coordination.operation.completed
                ├── model_call.started ... completed/failed
                │   └── tool_call.requested
                │       └── tool_execution.observed 或 started ... completed/failed
                └── agent.message.published
```

### 4.9.2 Run、Trial 与 Attempt 事件

| Event | 主要 payload 字段 | 说明 |
|---|---|---|
| `run.started` | `out_dir`, `config_file`, `num_cases`, `start_index`, `benchmark_id`, `scorer_kind`, seed、生成预算、turn/round/model-call 限制、GroupChat、团队、并发、Docker、network、observability、resume、runtime | Run 开始时的 resolved 参数快照 |
| `run.resumed` | `enabled`, `existing_trials`, `kept_successful_trials`, `ignored_trial_events`, `skipped_trial_count`, `stale_removed`, `event_log_migration` | 续跑扫描与迁移决策 |
| `run.stop_requested` | `action`, `requested_at_utc`, `requested_by`, `active_trials` | 请求停止接纳新 Trial |
| `run.stopped` | 运行汇总；信号停止时还含 `stop_reason`, `interrupted_active_trials` | 在活动 Trial 收口或信号清理后终止 |
| `run.paused` | 分段运行汇总、`segment_case_limit`、完成 Case 数 | 达到本执行段 Case 配额 |
| `run.completed` | `num_trials`, `trials_per_case`, `skipped_trials`, `failed_trials`, `run_status`, `vllm_metrics_summary`, `out_dir` | Run 正常完成 |
| `run.failed` | `run_status`, `vllm_metrics_summary`, `out_dir`, `error_type`, `error_message`, `traceback` | 未捕获异常导致 Run 失败 |
| `trial.started` | `case_order`, `num_cases`, `question`, `has_context`, benchmark/task/method、topology、speaking order、seed | 一次独立 Trial 开始，保存完整问题与真实拓扑 |
| `trial.completed` | task/method/benchmark/seed、`final_output`, `stop_reason` | 成功 terminal；`final_output` 即 Prediction |
| `trial.failed` | 身份字段、`error_type`, `error_message`, `traceback` | 所有 Attempt 均失败，没有有效 Prediction |
| `trial.interrupted` | `interrupted_started_event_id`, `previous_parent_event_id`, `reason` | resume 关闭上个进程遗留的未终结 Trial |
| `trial.skipped` | `case_order`, `num_cases`, `reason` | resume 跳过已有成功 Trial |
| `attempt.started` | `attempt`, `max_attempts` | 一次基础设施 Attempt 开始 |
| `attempt.completed` | `attempt`, `max_attempts` | Attempt 成功 |
| `attempt.failed` | `attempt`, `max_attempts`, `will_retry`, error/traceback | Attempt 异常以及是否继续重试 |

Attempt 重试用于当前仍由宽泛异常边界覆盖的基础设施错误；它不会创建新的 Trial，也不会重新抽样一个答案。错误 Trial 留在分母，不能只聚合成功 Trial。

### 4.9.3 Backend、Runtime、Workspace 与浏览器事件

| Event | 主要 payload 字段 | 说明 |
|---|---|---|
| `backend.probed` | deployment/kind/model、reasoning token accounting、tokenizer warmup 状态与时延、backend 扩展字段 | 启动能力探测 |
| `concurrency.changed` | `reason`, `previous`, `target`, `policy` | Runtime 内自适应 Trial 并发目标变化；不是 Scheduler 准入额度 |
| `runtime.started` | runtime/team/GroupChat、tools、turn/round/model-call 限制、executor/image、模型上下文、network | 一个 Attempt 的 MAS Runtime 开始 |
| `runtime.completed` | stop reason、模型调用/消息/工具计数、预算状态、tool errors、wall/duration | Runtime 成功 terminal |
| `runtime.failed` | `runtime_elapsed_s`, error/traceback | Runtime 异常 terminal |
| `runtime.cancelled` | `runtime_elapsed_s` | Trial deadline 或外部取消导致的 Runtime terminal；不是未闭合操作 |
| `workspace.prepared` | workspace、copied files、task text、是否新建、policy、附件命名与可见名 | Trial 工具工作区和附件策略 |
| `code_executor.started` | executor、Docker image、timeout、workspace | 执行器启动 |
| `code_executor.ready` | executor、image、workspace、startup latency | 执行器可用的 point Event |
| `code_executor.stopped` | executor、image、workspace、executor wall time | 执行器 terminal |
| `web.context_startup.started` | role、browser context ID、headless、proxy enabled/endpoint | 浏览器 context 启动 |
| `web.context_startup.completed` | role、context ID、startup latency、proxy 信息 | 浏览器可用 |
| `web.context_startup.failed` | proxy 信息、error/traceback | 浏览器启动失败 |
| `web.reset.recovered` | role、初始页、fallback 页、原异常 | AutoGen WebSurfer reset 的初始页导航失败后已回退到空白页；不是网页搜索成功事件 |

### 4.9.4 团队、协调与消息事件

| Event | 主要 payload 字段 | 说明 |
|---|---|---|
| `group_chat.selected` | `group_chat_type`, `group_chat`, `speaking_order` | RuntimeAdapter 选择的框架原生协调实现 |
| `group_chat.started` | class、config、participants、context visibility、dynamic topology | 团队执行开始 |
| `group_chat.completed` | class、message count、stop reason、model-call budget 状态 | 团队正常结束 |
| `group_chat.failed` | class、event index、error/traceback | 团队流异常 |
| `group_chat.cancelled` | class、event index、message count | Trial deadline 或外部取消终止 GroupChat；与 started 共享 operation ID |
| `coordination.operation.completed` | `operation`, `actor`, `framework`, `decision_source`、原始决定、合法性和选中 Node；复杂 Ledger 派生事件另有 `derived`、`projector_id/version`、`mapping_version`、`source_event_ids` | 普通 deterministic/model Selector 在三框架运行时直接记录 `select_next`；LangGraph/CrewAI 的显式 plan、monitor_progress、detect_stall、delegate、replan 或 aggregate 同样直接记录；AutoGen Magentic-One 的内部 Ledger 则由 Evidence 层按官方状态机生成可追溯派生事件 |
| `coordination.protocol_violation` | protocol、violation、模型给出的动作、允许动作集合、model call index | 框架原生协调器拒绝模型控制输出时保存具体协议原因；只诊断，不修复或改写输出 |
| `framework.output_rejected` | `framework`, `reason`, `projection`, error type/message | 第三方框架拒绝模型原始输出，但平台能够无损投影该事实；当前 CrewAI 空 content 映射为真实空消息，后续仍走统一 ResultContract |
| `agent.message.published` | event index、source、原生类型、content、usage、prompt/completion tokens、tool/error 标记、metadata、tool requests/executions、agent operation error | 框架真正发布的一条消息或 typed event |

Magentic-One ledger 是 controller 内部模型调用，通常体现在 `model_call.*`，不一定成为普通 GroupChat 消息。LangGraph/CrewAI 在执行时直接记录 operation 决定。AutoGen 保持 `run_events` 为框架实际发生事实，离线 Evidence normalizer 再运行 `AutoGenMagenticOneLedgerProjector`：只有通过官方 Progress Ledger 结构与 speaker 合法性校验的调用才映射为 operation，并严格复现官方 stall 递增/递减、阈值 replan 和 terminal aggregate 状态机。派生 Event 不复制 provider payload，只通过 source Event ID、artifact/line 和版本字段回链原始模型输出。这样既能计算跨框架过程指标，也不会声称 AutoGen 原生发布了不存在的公共 callback。跨框架协调指标只统计实际团队 Node 发布且非工具事件的消息，不能因为 AutoGen 把工具协议也放入原生消息流而重复计数。

### 4.9.5 模型调用事件与字段

| Event | 主要 payload | 说明 |
|---|---|---|
| `model_call.started` | 调用身份、三层消息、生成/上下文/thinking 预算、sampling、工具、transport | 一次真实 backend 请求开始 |
| `model_call.completed` | 三部分输出、provider payload、usage、缓存、时延、结束原因、工具请求、实际策略 | 请求成功 terminal |
| `model_call.failed` | 身份、transport 计划、error/traceback | 请求失败；无 provider usage 时成本是下界 |
| `model_call.budget_exhausted` | role/controller、started count、per-Trial 上限 | 下一请求发送前被调用额度阻止 |
| `model_call.budget_stopped` | started count、per-Trial 上限 | 团队把额度耗尽转成停止条件 |

`model_call.started` 的字段按用途分组：

| 字段组 | 字段 | 含义 |
|---|---|---|
| 身份 | `deployment_instance_id`, `role`, `turn`, `sender`, `controller`, `model_call_index`, `request_seed` | 谁通过哪个部署发起第几次请求 |
| 输入统计 | `message_count`, `input_chars`, `tool_count`, `tool_choice`, `json_output_requested` | 请求结构规模与能力 |
| 三层消息 | `autogen_model_messages`, `role_visible_messages`, `backend_messages` | 框架模型输入、按 Team policy 过滤后的角色可见输入、最终发送给 backend 的规范化结构化输入 |
| 路由 | `memory_channel`, `routing_reason`, `nl_memory_chars` | CDM/路由结果；controller 通常为空 |
| 生成 | `max_new_tokens`, `invocation_policy` | 本次有效参数和 provider override |
| Context 预算 | context window、configured/effective input/output、budget 前后输入、最小输出保留、safety margin、丢弃消息、prompt truncation、token count method、model context policy | 输入与输出自适应平衡 |
| Thinking 预算 | configured/effective thinking、thinking/final reserve、requested parameter、enforced by | reasoning 预算和生效端 |
| Transport | timeout mode/seconds、SDK retries、规划/观测生成速度、timeout policy | 请求超时和重试计划 |

`model_call.completed` 额外字段：

| 字段组 | 字段 | 含义 |
|---|---|---|
| 输入 usage | `input_total_positions`, `input_text_tokens`, `input_latent_positions`, `input_cached_tokens`, `input_cached_tokens_source/status` | 输入、latent 与缓存命中 |
| 输出 usage | `output_total_tokens`, `output_text_tokens`, `output_reasoning_tokens`, `output_answer_tokens`, 对应 source 和字符数 | 总输出及 reasoning/answer 拆分；不可观测时为 null，不伪造 0 |
| 语义输出 | `reasoning_content`, `final_content`, `delivered_content` | 模型思考文本、模型最终内容、实际交给框架的内容；结构化 tool call 时 delivered 可不是纯文本 |
| Provider | `provider_request_payload`, `provider_response_payload`, `response_payload_origin` | API/vLLM 原始 payload；Local HF 提供同形规范化 payload并注明来源 |
| 结束 | `finish_reason`, `original_prompt_positions`, `prompt_truncated`, `dropped_messages` | provider 结束原因与上下文裁剪 |
| 工具 | `tool_call_count`, `tool_call_request` | 本次输出的工具调用摘要；独立 tool Event 仍是协议事实 |
| Client 时延 | rate limiter wait、HTTP、postprocess、model-call wall、model latency | 客户端视角的分解时延 |
| Provider 时延 | metrics available、queue、scheduled-to-first-token、generation、inter-token、tokens/s、原始 metrics | provider/vLLM 可观测的排队、TTFT 和生成 |
| 策略 | `invocation_policy` 及本次预算/transport 字段 | 真正生效的调用策略快照 |

`reasoning_content`、`final_content`、`delivered_content` 不应随意合并：前两项忠实表达模型语义输出，第三项表达框架实际消费内容。官方 JSON mode、tool call 或 typed response 可能让 delivered 与纯文本 final 不同；系统不自行修复 Magentic-One JSON，也不修改 provider 原始输出。

### 4.9.6 工具与评分事件

| Event | 主要 payload 字段 | 说明 |
|---|---|---|
| `tool_call.requested` | deployment、role/source/turn/sender/model-call index、tool call ID/name/arguments/origin | 模型或框架请求工具；`correlation_id=tool_call_id` |
| `tool_loop.limit_reached` | role、framework、max iterations、request count | 角色级工具循环达到 TeamSpec 上限 |
| `tool_execution.started` | tool call ID、name、origin、code blocks | 真实执行开始 |
| `tool_execution.completed` | record/source/tool identity、exit code、error flag、完整 output、delivered output、两种长度、上下文截断标志、duration、origin | 保存无损工具事实与有界回填副本 |
| `tool_execution.failed` | 身份、duration、error/traceback | 执行抛异常 |
| `tool_execution.observed` | event/source/tool identity、arguments、output/delivered output、长度、截断、status/origin | 框架只暴露已完成结果时使用 point Event，不伪造开始时间 |
| `evaluation.completed` | `trial_event_id`, task/benchmark/scorer、gold、score、details、correct、contamination audit | 官方 scorer 成功 |
| `evaluation.failed` | `trial_event_id`、可识别任务字段、error/traceback | gold、外部 evaluator 或 scorer 失败 |

AutoGen 原生 WebSurfer 的动作在其内部 `_execute_tool()` 周围可得到真实开始和结束时间，因此写 `tool_execution.started/completed` 且 `timing_scope=tool_execution`。FileSurfer 只在 `_generate_reply()` 内部暴露“模型选工具并完成文件动作后的统一返回”，无法无侵入地把模型时延与文件动作时延拆开；系统因此写 `tool_execution.observed`、`timing_scope=agent_turn`，只把它计入执行次数，不把整段 Agent turn 冒充精确 tool latency。

### 4.9.7 完整性与安全规则

1. `seq` 严格递增，一个逻辑日志内 `run_id` 一致；
2. 每个 started Operation 最多一个有效 terminal，未闭合操作在 Trace 中标记为 interrupted；
3. Trial terminal 复用其 `trial.started.operation_id`；
4. Evaluation 的 `trial_event_id` 指向被评分 Trial 的 terminal Event；
5. 工具请求与执行优先共享 `correlation_id`；
6. Prediction 只存在于 Trial terminal，不复制所有模型/工具明细；
7. API key、Authorization、Cookie、代理订阅等凭据不得写入 payload；
8. 空值表示 provider 或当前证据没有观测到，不能替换成 0；
9. 派生视图和分析产物可以删除重建，但不得反向修改 EventLog；
10. 对 `run_id/seq`、Operation 配对、父子链、Trial/Evaluation 引用和重复 terminal 的 verifier 是持续完善项。
