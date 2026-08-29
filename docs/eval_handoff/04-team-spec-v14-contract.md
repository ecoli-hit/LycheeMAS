# 4.4 TeamSpec v14：当前权威合同

[上一部分：资源与 Benchmark](04-resource-and-benchmark-contracts.md) · [返回第 4 章索引](04-implementation-contracts.md) · [下一部分：跨框架 Runtime](04-framework-runtime-contracts.md)

!!! success "当前状态"
    TeamSpec v14 是 Eval 唯一可写、可持久化的团队合同。所有 Eval TeamSpec 均使用 `schema_version=14`；Reference Executor、AutoGen、LangGraph、CrewAI、TeamInstance、Studio 和 conformance tests 读取同一份规范化文档。旧 Schema 不存在运行期兼容分支，也不在当前手册中并列维护。

## 4.4.1 设计结论

TeamSpec v14 用一份框架无关、有状态的有向图描述团队：

```text
TeamSpec v14
  ├── Nodes          可被激活的执行主体
  ├── Relations      Node 间显式 Control / Data 关系
  ├── SharedState    可持久读写的团队状态与记忆
  └── Lifecycle      入口、结果、终止、失败和硬边界
           │
           ├── Reference Executor：直接解释规范语义
           ├── AutoGen BindingPlan：构造 AgentChat Team/Agent
           ├── LangGraph BindingPlan：构造 StateGraph/Node/Edge
           └── CrewAI BindingPlan：构造 Crew/Task 或 Flow
                         │
                         └── TeamInstance：绑定 DeploymentInstance 后可运行
```

这里没有第二份持久化 `TeamIR`，也不生成作者看不到的 Port/StoreNode 图。`Coordination IR`、`MASGraph`、BindingPlan 和框架原生对象都是一次编译或一次 Trial 的内存索引，可以随时从 TeamSpec 重建，不能独立修改或保存为另一份团队真源。

四条唯一真源规则：

1. Node 只说明执行主体是什么、怎样工作、能做什么。
2. Relation 只说明两个 Node 之间怎样推进控制和传递数据。
3. SharedState 只说明团队保存什么、谁能读写、保留多久。
4. Lifecycle 只说明从哪里开始、谁提交结果、何时结束和怎样限界。

Node 名称、文件名、`centralized/sequential` 标签和框架类名都不能触发运行行为。

!!! note "Node 不等于智能体"
    v14 中已经没有“存储型 Node”：持久状态统一写入 `shared_state[]`，`nodes[]` 中的每一项都是可被激活的执行主体。但 Node 仍不能与“大模型智能体”画等号：`model_agent` 最接近普通智能体，`tool_executor` 是确定性执行器，`function/human/team/remote` 分别表示函数、人类、子团队和远程执行主体。

## 4.4.2 完整示例

下面是一个可直接转换成 JSON 的三角色线性团队。正式配置使用 JSON；这里使用 YAML 只是为了阅读。

```yaml
schema_version: 14
id: math-linear-team
metadata:
  name: Math Linear Team
  description: Analyst, Solver and Verifier execute in order.
  tags: [sequential, controlled-portability]
  provenance:
    track: controlled_portability
    evidence_level: hypothesis
    sources: []
    notes: Framework-neutral baseline.
nodes:
  - id: Analyst
    name: Analyst
    kind: model_agent
    behavior: {type: assistant, options: {}}
    purpose: Decompose the task.
    instructions: Analyze the task and give Solver a concise plan.
    capabilities: [reason]
    operations: []
    tools: []
    context: {type: token_limited}
    limits: {max_tool_iterations: 1}
  - id: Solver
    name: Solver
    kind: model_agent
    behavior: {type: assistant, options: {}}
    purpose: Produce a candidate answer.
    instructions: Solve the task using the task and visible team evidence.
    capabilities: [reason]
    operations: []
    tools: []
    context: {type: token_limited}
    limits: {max_tool_iterations: 1}
  - id: Verifier
    name: Verifier
    kind: model_agent
    behavior: {type: assistant, options: {}}
    purpose: Check and submit the final answer.
    instructions: Verify the candidate and return one final answer.
    capabilities: [reason]
    operations:
      - type: validate
        options: {mode: model, source: SharedConversation}
    tools: []
    context: {type: token_limited}
    limits: {max_tool_iterations: 1}
relations:
  - id: analyst-to-solver
    from: Analyst
    to: Solver
    control:
      trigger: completed
      action: activate
      condition: null
      priority: 0
      on_failure: fail_trial
      max_uses: 1
    data:
      transfers:
        - {source: trial.task, target: task, required: true, view: all, latest_n: null, filter: null, transform: null, schema: {}}
        - {source: shared.SharedConversation, target: messages, required: false, view: all, latest_n: null, filter: null, transform: null, schema: {}}
  - id: solver-to-verifier
    from: Solver
    to: Verifier
    control:
      trigger: completed
      action: activate
      condition: null
      priority: 0
      on_failure: fail_trial
      max_uses: 1
    data:
      transfers:
        - {source: trial.task, target: task, required: true, view: all, latest_n: null, filter: null, transform: null, schema: {}}
        - {source: shared.SharedConversation, target: messages, required: false, view: all, latest_n: null, filter: null, transform: null, schema: {}}
shared_state:
  - id: SharedConversation
    kind: message_channel
    description: Trial-scoped team conversation.
    readers: [Analyst, Solver, Verifier]
    writers: [Analyst, Solver, Verifier]
    update: append
    lifetime: trial
    initial: []
    schema: {}
    retention: {max_items: null, max_tokens: null, overflow: drop_oldest}
lifecycle:
  entry:
    - node: Analyst
      inputs:
        - {source: trial.task, target: task, required: true, view: all, latest_n: null, filter: null, transform: null, schema: {}}
  result:
    submissions:
      - {from: Verifier, source: source.output, key: final_answer}
    mode: first_valid
    schema: {}
  termination: {condition: result_submitted}
  failure: {unhandled: fail_trial, deadlock: fail_trial}
  limits: {max_turns: 3, max_node_calls: 3, max_stalls: null, timeout_seconds: null}
```

执行顺序是：Lifecycle 激活 Analyst；Analyst 完成后 `analyst-to-solver.control` 激活 Solver，同时 `data.transfers` 物化 Solver 的 NodeInput；Solver 完成后同理激活 Verifier；Verifier 的非空输出满足 ResultContract，Trial 立即结束，不再额外调用路由器。

## 4.4.3 字段全集

### 4.4.3.1 顶层和 Metadata

| 字段 | 类型 | 含义 | 运行作用 |
|---|---|---|---|
| `schema_version` | integer | 固定为 `14` | 选择严格规范化器 |
| `id` | string | 稳定 TeamSpec ID | 注册、快照、TeamInstance 引用 |
| `metadata.name` | string | 人类可读名称 | 只展示，不参与 dispatch |
| `metadata.description` | string | 用途和边界 | 只展示和报告 |
| `metadata.tags` | list[string] | 检索/研究标签 | 不参与 dispatch |
| `metadata.provenance.track` | enum | `controlled_portability/native_replication/literature_replication/custom` | 区分证据轨 |
| `metadata.provenance.evidence_level` | enum | `official/paper_reproduction/paper_inspired/hypothesis` | 报告证据强度 |
| `metadata.provenance.sources[]` | object | `title/url/scope` | 配方来源审计 |
| `metadata.provenance.notes` | string | 补充说明 | 配方审计 |
| `nodes` | list[Node] | 至少一个可执行 Node | 三框架 Node binding |
| `relations` | list[Relation] | Node 间显式关系 | 控制和数据执行 |
| `shared_state` | list[SharedState] | 团队状态、会话、产物和记忆 | 状态运行时 |
| `lifecycle` | object | Trial 入口和收口 | Reference Executor/Adapter |

### 4.4.3.2 Node

每个 `nodes[]` 都是可执行 Node。持久状态不塞进 Node，而放在 `shared_state[]`。

阅读下表时需要区分三种值域：

| 值域类型 | 含义 | 例子 |
|---|---|---|
| 封闭枚举 | 只能从规范列出的值中选择，其它值会校验失败 | `kind`、`operations[].type` |
| 可扩展标识符 | Schema 允许新名称，但 Runtime binder 可能只识别已实现子集 | `behavior.type`、`capabilities[]` |
| 自由内容 | 由团队作者编写，不从固定值池选择 | `purpose`、`instructions` |

| 字段 | 类型 | 含义 | 约束/去向 |
|---|---|---|---|
| `id` | string | 稳定运行 ID | Relation、Event、Binding 关联键；必须是 runtime-safe ID |
| `name` | string | 展示名称 | 改名不能改变绑定结果 |
| `kind` | enum | `model_agent/tool_executor/function/human/team/remote` | 选择抽象 handler；当前受控三框架重点支持前两类 |
| `behavior.type` | string | 如 `assistant/selector/code_author/code_executor/web_navigator` | 选择框架 specialization；不是框架类名 |
| `behavior.options` | object | behavior 的框架无关选项 | 由对应 binder 校验 |
| `purpose` | string | 为什么调用该 Node、职责是什么 | 候选说明和 UI |
| `instructions` | string | Node 被激活后具体怎样工作 | model system instruction 或 handler contract |
| `capabilities` | list[string] | 可发现的能力声明 | BindingReport/资源匹配；不能代替 Operation |
| `operations` | list[Operation] | 显式协调职责 | Coordination compiler 唯一 Operation 来源 |
| `tools` | list[ToolRequirement] | 工具合同 | Runtime 工具绑定 |
| `context` | ModelContextPolicy | `unbounded/buffered/token_limited` | 该模型 Node 的上下文窗口策略 |
| `limits.max_tool_iterations` | integer | 一次 Node 激活内最多工具迭代 | 工具循环边界，至少为 1 |
| `limits.on_tool_limit` | `finalize/return_tool_result` | 达到工具迭代上限后的明确动作；省略时为 `finalize` | `finalize` 再进行一次无工具模型调用以生成交付内容；`return_tool_result` 直接交付最后一个工具结果 |

`purpose` 是职责摘要；`instructions` 是可执行指令。非模型 Node 可以不需要自然语言 system prompt，但仍可用 `instructions` 描述确定性 handler 的合同。

#### 4.4.3.2.1 `kind`：执行机制的封闭枚举

| 值 | 表示的执行主体 | 是否默认调用模型 | 当前三框架受控支持 |
|---|---|---:|---|
| `model_agent` | 通过聊天模型完成分析、生成或调度的 Node | 是 | 支持 |
| `tool_executor` | 不使用模型、直接执行工具请求的 Node | 否 | 当前受控支持 `behavior.type=code_executor` |
| `function` | 一个确定性本地函数 | 否 | 已入 Schema，尚未进入三框架受控门禁 |
| `human` | 需要人类提供输入的执行点 | 否 | 同上 |
| `team` | 把另一个 TeamInstance 当作子团队调用 | 取决于子团队 | 同上 |
| `remote` | 远程服务或外部执行主体 | 取决于远程服务 | 同上 |

`kind` 只回答“用什么执行机制调用它”，不回答“它在团队中扮演什么专业角色”。

#### 4.4.3.2.2 `behavior`：框架适配特化

`behavior.type` 是可扩展标识符，不是 Schema 封闭枚举。当前配置和 binder 中有明确意义的值如下：

| 值 | 含义 | 当前主要绑定 |
|---|---|---|
| `assistant` | 通用模型 Node | AutoGen `AssistantAgent`；LangGraph/CrewAI 通用 model callable |
| `code_author` | 以生成代码为职责的模型 Node | 通用模型 Node；可与代码工具组合 |
| `code_executor` | 执行代码而不是撰写代码 | `kind=tool_executor` 时绑定确定性代码执行器 |
| `file_navigator` | 文件浏览专长 | AutoGen `FileSurfer`；其它框架通过文件工具组合 |
| `web_navigator` | 网页浏览专长 | AutoGen `MultimodalWebSurfer`；其它框架通过 Web 工具组合 |
| `selector/orchestrator` | 可读的控制角色标签 | 不能单独触发调度；真实控制语义必须由 `operations` 和 Control Relations 表达 |
| 自定义标识符 | 为未来 binder 保留的扩展点 | 当前可能降级为通用 model Node，必须检查 BindingReport |

`behavior.options` 是该 behavior 的框架无关参数容器。它当前只是结构化扩展点；未被 binder 明确声明的 option 不得假定已经生效。

#### 4.4.3.2.3 角色语义由多个字段共同决定

一个 Coder 之所以真正可以撰写代码，不是因为它叫 `Coder`，也不是单独因为 `behavior.type=code_author`，而是下列合同的组合：

| 问题 | 决定字段 | Coder 例子 |
|---|---|---|
| 它怎样产生输出？ | `kind` | `model_agent`，因此通过模型生成 |
| 适配器应将它当成什么专长？ | `behavior.type` | `code_author` |
| 它具体应怎样写代码？ | `instructions` | 语言、输出格式、检查要求和职责边界 |
| 它能否真的运行代码？ | `tools` | 绑定 `runtime:python_code` 等工具 |
| 它声称具备哪些能力？ | `capabilities` | `reason/write_code/execute_code` |
| 它是否还负责调度别人？ | `operations` | 普通 Coder 为空；若兼任调度则显式声明 |

#### 4.4.3.2.4 `purpose`、`instructions`、`capabilities`和 `operations`

| 字段 | 值域 | 精确含义 | 当前运行影响 |
|---|---|---|---|
| `purpose` | 自由文本 | 给人和调度候选逻辑阅读的职责摘要 | 用于 UI、BindingReport 和候选描述；不作为完整 system prompt |
| `instructions` | 自由文本 | Node 每次被激活后的可执行行为指令 | `model_agent` 的 system instruction；非模型 Node 的 handler 合同 |
| `capabilities[]` | 可扩展字符串列表 | 可发现、可审计的能力声明 | 目前主要用于 BindingReport 和资源检查；`vision` 会要求视觉 Deployment；不会自动提供工具 |
| `operations[]` | 封闭 Operation 对象列表 | Node 在团队协调中拥有的可观测责任 | Coordination compiler 的唯一 Operation 来源；真正改变调度方案 |

当前配置中常见的 capability 包括 `reason/browse/search_web/read_files/write_code/execute_code/vision/inspect_artifacts`。这个列表不是 Schema 封闭枚举；新 capability 可以被登记，但只有 binder、资源解析器或工具注册表明确识别后才会产生运行效果。

### 4.4.3.3 Operation 和 ToolRequirement

`operations[]` 只接受 `{type, options}` 对象，不接受字符串简写。当前 Operation 类型为 `select_next/plan/decompose/delegate/monitor_progress/detect_stall/replan/handoff/aggregate/validate`。

| 字段 | 含义 |
|---|---|
| `type` | Operation 的框架无关语义 |
| `options.mode` | `model` 或 `deterministic` |
| `options.prompt` | 可选控制提示模板 |
| `options.max_attempts` | 结构化决定/重试上限 |
| `options.state_source/state_sources/state_target` | 显式读取和写入的 SharedState |
| `options.window/max_replans/max_items` | 对应 operation 的有界参数 |
| `options.finish` | `select_next` 的 `allowed/requires_result` |
| `options.fallback` | handoff 的 `error/retry/next_priority/finish` |
| `tools[].id` | 工具合同 ID |
| `tools[].required` | 缺失时是否阻止实例化 |

`submit` 不属于 Operation；谁可以提交结果由 `lifecycle.result.submissions` 单独定义，避免协调能力和结果所有权重复。

!!! warning "Operation fallback 与 Relation failure policy 不是同一件事"
    `Node.operations[type=handoff].options.fallback` 处理的是 **handoff operation 没有产生可解析、合法的下一 Node**，例如模型没有输出约定的 handoff 目标。`Relation.control.on_failure` 处理的是 **一条具体 Relation 已经被选中以后，该 Relation 的目标激活或执行失败**。前者可以是 `next_priority`，同时后者仍然可以是 `fail_trial`；adapter 不得用 Relation 的失败策略覆盖 Node operation fallback。Coordination IR 通过 `operation_options(coordination, "handoff", node_id=...)` 查询前者，通过具体 Relation contract 查询后者。

### 4.4.3.4 Relation

每条 Relation 必须连接一个 `from` Node 和一个 `to` Node。它可以只含 Control、只含 Data，或同时含二者；一条关系不能把多个目标合并进数组。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 稳定 Relation ID |
| `from/to` | Node ID | 有向源和目标 |
| `control` | Control/null | 是否以及怎样激活、路由、返回或结束 |
| `data` | Data/null | 目标输入如何物化 |

Control：

| 字段 | 取值/含义 |
|---|---|
| `trigger` | `trial_started/completed/failed/output_emitted/result_submitted/always` |
| `action` | `activate/select_next/delegate/handoff/return/finish/cancel` |
| `condition` | 可选结构化 guard：`source/operator/value`，禁止任意 `eval` |
| `priority` | 合法候选的稳定排序，数值越小越优先 |
| `on_failure` | `fail_trial/continue/try_next/return_to_source` |
| `max_uses` | 一个 Trial 内最多使用次数；`null` 表示由其它硬边界限制 |

Data：

| 字段 | 取值/含义 |
|---|---|
| `transfers[]` | 一项输入物化合同 |
| `source` | 如 `trial.task/source.output/shared.<id>` |
| `target` | 统一 NodeInput 中的 `task/messages/state/artifacts/...` |
| `required` | 缺失时是否拒绝激活 |
| `view` | `all/latest/latest_n/from_source/summary` |
| `latest_n` | 仅在 `view=latest_n` 时使用 |
| `filter` | typed filter；当前 adapter 未执行时 BindingReport 必须降级 |
| `transform` | 已注册确定性转换 ID；不能嵌入代码或临时 LLM |
| `schema` | 数据结构合同 |

Control 不隐式授予消息可见性，Data 也不隐式激活目标。两者必须分别声明。

一次 handoff 的规范顺序是：Node operation 先尝试产生候选；若没有合法候选，执行 operation 的 `fallback`；若最终选中了某条 Relation，则执行该 Relation 的 guard、DataTransfer 和目标激活；只有这一步失败时才执行该 Relation 的 `on_failure`。这条顺序由 Reference Executor 和三框架 conformance test 共同约束。

### 4.4.3.5 SharedState 和 Memory

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 状态稳定 ID |
| `kind` | enum | `message_channel/structured_state/artifact_store/memory` |
| `description` | string | 状态用途 |
| `readers/writers` | list[Node ID] | 显式访问控制 |
| `update` | enum | `append/replace/merge/commit` |
| `lifetime` | enum | `invocation/trial/run` |
| `initial` | any | 初始值 |
| `schema` | object | 状态结构合同 |
| `retention.max_items/max_tokens` | integer/null | 有界保留 |
| `retention.overflow` | enum | `drop_oldest/reject/error` |

`kind=memory` 时增加：

| 字段 | 含义 |
|---|---|
| `memory.retrieval.mode` | `chronological/semantic/hybrid` |
| `memory.retrieval.top_k` | 一次召回项数 |
| `memory.retrieval.score_threshold` | 可选相似度阈值 |
| `memory.retrieval.query_source` | `task/node_input/task_and_node_input` |
| `memory.write.policy` | `explicit/node_output` |
| `memory.write.source` | 写入的数据来源 |
| `memory.injection.target` | `model_context/node_input` |
| `memory.injection.template` | 必须含 `{items}` 的注入模板 |

当前可移植执行子集是 `lifetime=trial + retrieval.mode=chronological`。三个 adapter 都由 `TeamMemoryRuntime` 在 `before_node_invocation` 召回、在 `after_node_invocation` 写入，并分别报告 AutoGen Memory/model context、LangGraph checkpointer/store、CrewAI Memory/Flow state 的承载计划。`semantic/hybrid` 或 `run` 生命周期需要显式 MemoryInstance/索引绑定；在实现前标为 `unsupported`，不会假装已经支持。

### 4.4.3.6 Lifecycle

| 字段 | 含义 |
|---|---|
| `entry[]` | `{node, inputs[]}`；Trial 开始时可有一个或多个入口 |
| `result.submissions[]` | `{from, source, key}`；唯一合法结果来源集合 |
| `result.mode` | `first_valid/all/aggregate` |
| `result.schema` | 逻辑团队结果结构 |
| `termination.condition` | 当前固定为 `result_submitted` |
| `failure.unhandled` | `fail_trial/continue` |
| `failure.deadlock` | `fail_trial/submit_best_effort` |
| `limits.max_turns` | 可发言 Node activation 上限 |
| `limits.max_node_calls` | 所有 NodeInvocation 上限 |
| `limits.max_stalls` | 只有显式 stall operation 才有意义 |
| `limits.timeout_seconds` | 团队执行墙钟上限；Experiment 仍有 Trial 级总 deadline |

## 4.4.4 从 TeamSpec 到三个框架

转换不是“猜 topology 再套模板”，而是五步编译：

1. `normalize_team_spec_document()` 严格拒绝未知字段和失效引用。
2. `compile_coordination()` 只从 Operations 和 Control Relations构建可重建的 Coordination IR。
3. 每个 adapter 独立选择原生承载：AutoGen GroupChat/Agent，LangGraph StateGraph，CrewAI Crew/Flow。
4. Node binder 按 `kind + behavior + operations + tools` 选择具体对象；显示名称不参与选择。
5. BindingReport 报告每个 Node、协调、Memory、Relation 的 `exact/composed/approximated/unsupported` 和 semantic delta。

Studio 的 TeamSpec 图展示规范 Node、SharedState、Control/Data；AutoGen、LangGraph、CrewAI 标签页展示各自 runtime carrier、具体 Node binding、Memory carrier、Control/Data 数量和完整 BindingReport。具体实现视图是只读编译结果，修改必须回到 TeamSpec。

### 4.4.4.1 Coordination IR、MASGraph、BindingPlan 和 TeamInstance

这四个对象处于不同层，不能都称为“团队配置”：

| 对象 | 性质 | 包含什么 | 不包含什么 | 是否持久化 |
|---|---|---|---|---:|
| TeamSpec | 唯一团队语义真源 | Node、Relation、SharedState、Lifecycle | 框架类名、模型部署、GPU | 是 |
| Coordination IR | 控制编译索引 | Operation 所有者、Control Relation、候选目标、依赖链、roots/sinks | Prompt、模型客户端、工具实例、实际消息 | 否 |
| MASGraph | Runtime 传输容器 | 编译后 AgentSpec 容器、Control 邻接、完整 TeamSpec、IR、ResultContract、BindingReport | 活着的 GroupChat/StateGraph/Crew | 否 |
| BindingPlan/Report | 目标框架编译结果 | 框架实现策略、Node binding、Memory binding、映射级别和差异 | 题目、实际运行状态 | TeamInstance 会冻结一份报告 |
| TeamInstance | 可运行实例配方 | TeamSpec 引用、框架、每个模型 Node 的 DeploymentInstance、可选生成覆盖 | 每个 Trial 的消息、临时工作区和框架内存对象 | 是 |

`Coordination IR` 可以理解为“给编译器用的控制索引”。例如，它会把三条分散的 Control Relations 整理成某个 `select_next` Node 的有序候选集，但不会创造 TeamSpec 没有声明的语义。

`MASGraph` 是当前 Runtime 接口的编译后传输对象。它的 `nodes` 暂时复用历史类名 `AgentSpec`，因此工具执行器也可能被放入 `AgentSpec`；这是内部类型名称的历史痕迹，不表示所有 Node 都是大模型智能体。

### 4.4.4.2 模型调用、工具和 Memory 由谁实现

当前不是“全部交给第三方框架”，也不是“全部由 LycheeMAS 重写”，而是按层分工：

| 职责 | 语义所有者 | AutoGen | LangGraph | CrewAI |
|---|---|---|---|---|
| 下一个 Node 何时运行 | TeamSpec Control/Operation | 原生 GroupChat/GraphFlow/Swarm 执行 | 原生 StateGraph 边和条件路由执行 | 原生 Crew Process 或 Flow 执行 |
| 一次模型调用何时发生 | 框架 Runtime | Agent/GroupChat 发起 | StateGraph Node callable 发起 | Agent/Task 或 Flow step 发起 |
| 调用哪个 Deployment 并记录 Event | TeamInstance + LycheeMAS | LycheeMAS `ChatCompletionClient` adapter 转发到 backend | LycheeMAS `ModelGateway` 转发到 backend | LycheeMAS `ModelGateway` 转发到 backend |
| 工具 schema 和实际函数 | TeamSpec tools + LycheeMAS tool registry | 注入原生 Agent | 注入 ModelGateway | 注入 ModelGateway/Flow |
| 模型产生 tool call 后的循环 | Node limits 规定公开语义 | AutoGen Agent 的原生工具循环 | ModelGateway 的通用有界循环 | ModelGateway 的通用有界循环 |
| 代码容器、工作区和便携工具 | LycheeMAS 公共服务 | 框架 Agent 调用这些实现 | StateGraph callable 调用 | Crew/Flow callable 调用 |
| EventLog 和 ResultContract | LycheeMAS | 客户端适配器+原生事件观察器 | ModelGateway+运行时观察 | ModelGateway+运行时观察 |

Memory 也是同样的分层模式：

1. TeamSpec `shared_state[kind=memory]` 是语义真源，决定读者、写者、召回、写入、注入和生命周期。
2. `TeamMemoryRuntime` 是当前三框架共用的便携语义实现，在 Node 调用前召回，在 Node 调用后写入。
3. AutoGen model context、LangGraph graph state/checkpointer、CrewAI Flow state/Memory 可以作为框架内承载体，但不能用它们的隐式默认值改写 TeamSpec 语义。
4. 当前受控实现主要是 Trial 级 chronological memory；semantic/hybrid 召回和跨 Run 持久化尚未实现完整 MemoryInstance 绑定。

因此，框架仍然保留它的调度和团队执行内核；LycheeMAS 统一了部署选择、模型调用边界、工具实现、Memory 公开语义、EventLog 和结果投影。

### 4.4.4.3 终止与工具上限的跨框架合同

`lifecycle.termination.condition=result_submitted` 不是一句 Prompt，也不是 `max_turns` 的别名。它要求 Runtime 在 `lifecycle.result.submissions[].from` 声明的合法 Node 发布非空、可投影结果后立即终止。Reference Executor、LangGraph 和 CrewAI 直接在公共结果合同上判断；AutoGen adapter 将它编译为官方 `TerminationCondition`，并通过消息 source 验证 submitter。角色名称和诸如 `APPROVE` 的自由文本不参与判定。

`limits.max_tool_iterations` 只约束一次 Node 激活中的工具循环。达到上限后：

- `finalize`：保留真实工具结果，再发起一次 `tools=[]` 的模型调用，请模型生成最终交付内容；
- `return_tool_result`：不再调用模型，直接交付最后一个工具结果。

AutoGen 的原生 `AssistantAgent` 通过 `reflect_on_tool_use=True` 承载 `finalize`；LangGraph/CrewAI 的 `ModelGateway` 执行同一公开语义。如果最终化调用只产生 reasoning 而 `content` 为空，AutoGen client 会把该次原始响应照常写入 EventLog，再发起一条新的、同样无工具且完整记录的最终化请求。它不是把 reasoning 复制到 final，也不是事后修补输出。恢复次数有界，超过后按真实空结果进入 ResultContract/失败路径。

## 4.4.5 Reference Executor 和测试门禁

Reference Executor 直接解释 v14：物化 Lifecycle entry、构造 NodeInput、执行 Control condition/priority/max_uses、读写 SharedState/Memory，并由 ResultContract 收口。它不调用第三方团队调度器，用于判断“TeamSpec 自身到底表达了什么”。

当前门禁包括：

- direct result、线性激活、显式选择、消息 SharedState、Memory 注入和有界循环；
- 三框架 Node/Coordination/Memory BindingReport；
- premature finish、严格 result submitter、工具成功/失败和 RuntimeConformanceReport；
- Studio TypeScript 生产构建。

只有基于 v14 TeamSpec 的当前 conformance fixture 和新 Run 可以作为合同证据；正式实验必须冻结 TeamSpec、TeamInstance、BindingReport、Experiment、Deployment/Pricing 和 EventLog。

## 4.4.6 不属于 TeamSpec 的内容

| 内容 | 所属对象 |
|---|---|
| 模型路径、API endpoint、GPU、TP/DP、`max_model_len` | Model/APISpec、DeploymentSpec/Instance |
| temperature、top-p、thinking budget、max new tokens | RoleDeploymentBinding/Experiment runtime |
| AutoGen/LangGraph/CrewAI 类名 | Framework BindingPlan |
| Benchmark、case 范围、sample、seed、scorer | BenchmarkSpec、ExperimentSpec/Instance |
| 队列优先级、分段暂停、恢复和调度 | ExperimentInstance/Scheduler |
| `independent/sequential/centralized/decentralized` | 研究展示标签；可从图派生，不能反向 dispatch |

## 4.4.7 当前边界

- `function/human/team/remote` 已进入 Schema，但受控三框架门禁目前聚焦 `model_agent/tool_executor`；其它类型必须由 BindingReport 诚实报告。
- DataTransfer `filter/transform` 尚未被三个 adapter 全量执行；使用时映射降为 `approximated`。
- Memory 的 semantic/hybrid 检索和跨 Trial/Run 持久化尚未绑定 MemoryInstance。
- Adapter 可以组合公共 Workspace、Tool、Event、Memory、Result 服务，但不能用一个大公共父类替代三个框架自己的执行内核。
- GAIA Magentic-One 的官方私有 Ledger 对齐属于独立 native-replication 课题，不是 v14 普通团队门禁；GAIA 的四种受控拓扑仍必须使用 v14 回归。
