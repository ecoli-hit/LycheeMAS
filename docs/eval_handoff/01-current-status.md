# 1. 当前开发状态

[返回总览](../EVAL_HANDOFF.md) · [下一章：系统边界与统一语言](02-system-boundary.md)

!!! abstract "本章只保留当前事实"
    本章记录当前平台能力、现存边界、活跃长程任务和技术债。旧 Run 数字、smoke 批次、队列快照、逐步调试记录和历史文档迁移台账已从交接手册移除；它们不是当前实现合同。

## 1.1 平台现状

LycheeMAS Eval 当前是一个包含资源注册、部署、团队定义、跨框架运行、实验调度、无损事件、官方评分和研究聚合的评测平台。

| 模块 | 当前合同 | 下一项实质工作 |
|---|---|---|
| Resource | Model、API、Benchmark、Pricing 等 Spec/Instance 注册 | 继续冻结官方来源、版本、许可和评分 profile |
| Benchmark | 统一 `prepare/load/score` 合同 | 审核 GAIA、BBEH、SWE-bench Verified、WorkBench 的严格复现配置 |
| Team | TeamSpec v14 是唯一可写团队真源 | 扩展当前尚未受控支持的 Node 和 Memory 语义 |
| Runtime | AutoGen、LangGraph、CrewAI 各自保留原生执行内核 | 继续缩小 BindingReport semantic delta |
| Deployment | HF、vLLM、OpenAI-compatible API | 以真实负载校准容量、超时和压力阈值 |
| Experiment | ExperimentSpec/Instance、Run/Trial/Attempt、分段续测 | 继续收紧不可变快照和终态幂等 |
| Scheduler | 逐 Trial 队列、固定/动态并发、分域压力 | 以正式负载校准吞吐与尾延迟 |
| Observability | `run_events/` 是唯一无损事实源 | 继续校准框架原生工具和协调事件覆盖 |
| Evaluation | Benchmark scorer 与通用过程指标解耦 | 完成过程指标的人工标注与效度校准 |
| Studio | 资源、成本、部署、团队、实验和运行记录页 | 继续拆分大组件并引入稳定 read model |

## 1.2 已完成的结构性迁移

| 领域 | 当前结论 |
|---|---|
| Team Schema | v13 的 Port/StoreNode/TeamIR 已退出运行路径；v14 只持久化 Node、Relation、SharedState 和 Lifecycle |
| Team compilation | Coordination IR、MASGraph 和 BindingPlan 只是可重建的内存编译产物 |
| Framework adapters | 三框架共享类型合同和窄服务，不共享一个替代原生调度器的庞大 Runtime |
| Result | TextResult、ActionResult、PatchResult 由 Benchmark-aware ResultContract 收口 |
| Event | 原始事实写入分片 RunEvent，大 payload 使用内容寻址存储，派生视图可离线重建 |
| Scheduler | 注册表、生命周期、准入策略、分域压力、健康反馈和只读投影已分层 |
| Studio | bootstrap 只返回核心数据，各管理页按需加载；Web/CLI/TUI 与 Server 的分层边界已明确 |
| Documentation | 交接手册按系统边界、相关工作、实现合同、研究方案和运维指南维护；第 4 章已按模块拆分 |
| Evaluation fault isolation | HLE Judge、逐 Trial scorer 与通用指标分别隔离失败；一个 Judge 格式错误不再使整批 Evaluation 丢失 |
| Runtime input boundary | 多模态 data URI 统一经 Pillow 解码；模型工具调用先做 JSON 与函数签名校验，再进入工具实现 |
| Resume lifecycle | 历史 finalization receipt 按执行段归档；恢复时清除旧评分状态，Run 成功与后处理失败分别投影 |
| Offline re-evaluation | terminal ExperimentInstance 可复用冻结分析命令重新评分并原子同步 Job、Evaluation、Receipt 与实例状态，不重复模型推理 |
| Trial failure isolation | Trial deadline 下沉到实际 provider timeout；失败 Trial 后丢弃该 worker 的 Runtime/Context，后台不可取消线程不能污染下一 Trial |
| Event attribution | 模型请求开始前冻结 Case/Trial/Attempt/Worker 身份；成对 model-call Event 必须保持同一归属，Selector 决策投影为统一协调 Event |
| Runtime terminal guard | AutoGen 的建队、执行、清理和结果投影统一受最外层终止守卫保护；任一阶段取消或异常都必须闭合 `runtime.started` |

## 1.3 当前已知边界

1. `function/human/team/remote` 已进入 TeamSpec Schema，但当前三框架受控门禁主要支持 `model_agent/tool_executor`。
2. DataTransfer 的 `filter/transform`、非零 Relation priority 尚未被三框架全量执行；BindingReport 必须降级报告。
3. Memory 当前可移植子集是 Trial 级 chronological recall；semantic/hybrid 和跨 Run 持久化尚需 MemoryInstance/索引绑定。
4. AutoGen 原生 Magentic-One 与 LangGraph/CrewAI composed orchestration 不是内部算法逐行等价；只能比较明确冻结的公开语义。
5. Planning、团队贡献、语义冗余、失败传播等过程指标尚未完成论文级人工校准。
6. Scheduler 压力阈值、vLLM 容量和超时基线是工程初值，不是跨机器通用常数。
7. Runs 页目前仍主要依赖缓存的目录扫描；数量继续增长后需要持久化索引 read model。
8. Eval 已删除部分历史私有 import 路径；Eval 之外的研究模块若直接依赖旧路径，需由对应模块维护者决定迁移或兼容策略。
9. HLE 的本地 Qwen `local_json_object` Judge 是开发/对照口径，不是官方 Judge；可发表的官方数字仍需冻结官方模型、schema 与版本。
10. 2026-08-29 之前由可取消协程包装同步后台线程的历史 CrewAI Run，若出现 Trial timeout，其逐 Trial token、延迟、工具错误和过程指标可能串入后续 Trial；最终 Prediction 经各自 Case scorer 得到的任务分数可单独审计，但这些 Run 不应直接用于论文级过程/效率比较。
11. 2026-08-29 之前的 AutoGen Run 若恰在 Agent、浏览器或代码执行器清理阶段触发 Trial deadline，可能缺失 `runtime.cancelled`；最终 Trial timeout 和零分口径仍可审计，但该 Trial 的 Runtime 生命周期完整性不能追溯补写。

## 1.4 活跃长程任务

只保留尚需后续行动的任务。完成项应转写为所属模块的当前合同，不在此处长期堆积测试批次和调试细节。

| ID | 状态 | 目标 | 验收条件 |
|---|---|---|---|
| LT-01 | in_progress | 冻结重点 Benchmark 的官方复现 profile | 数据、scorer、revision、模型、工具、预算和失败分母都进入不可变快照 |
| LT-03 | in_progress | 以真实负载校准 Trial admission 与分域压力 | 报告吞吐、Queue/TTFT 分位数、KV、preemption、失败率和阈值依据 |
| LT-04 | planned | 校准 Planning、Team、Redundancy 和 Failure 过程指标 | 建立 codebook、calibration/frozen split、标注一致性和 mutation sensitivity |
| LT-05 | planned | 分开严格控制轨与框架最佳实践轨 | 两轨配置、统计、结论和适用范围分别报告 |
| LT-06 | in_progress | 继续拆分大文件、大 RuntimeAdapter 和历史单体测试 | 公共合同不变，测试回归到对应领域所有者 |
| LT-09 | in_progress | 收紧框架无关 CoordinationContract | 合法动作、结束前置条件、资源边界和失败语义在三框架可对照 |
| LT-11 | in_progress | 建立 official/native、literature replication、controlled portability 三条证据轨 | 每个配方都有 provenance、修订版、改写范围和 recipe audit |
| LT-12 | deferred | 建立上游原生与 LycheeMAS adapter 的配对校准实验室 | 冻结同一配方，比较 upstream-native、instrumented-native 和 adapted-replication |
| LT-19 | planned | 完成 semantic/hybrid 及 durable MemoryInstance | 三框架都能按同一 TeamSpec MemoryContract 召回、写入、持久化和报告差异 |

## 1.5 当前技术债与处理顺序

| 优先级 | 技术债 | 处理方向 |
|---:|---|---|
| P0 | TeamSpec 字段的 Schema 能力与三框架实际消费范围仍有差异 | 用 BindingReport 和 conformance fixture 逐字段闭环；不支持的值拒绝或诚实降级 |
| P0 | 三框架的工具、消息和协调观测粒度仍不完全相同 | 建立成功/失败的框架原生 fixture，区分缺失证据与 N/A |
| P1 | AutoGen Runtime 保留较多框架专属生命周期代码 | 继续提取 Workspace、Tool、Event、Result 等窄服务，不创建大一统父类 |
| P1 | 部分领域对象仍由松散 `dict` 承载 | 按模块逐步物化 typed Value Object，并保持 Transport 不渗入领域层 |
| P1 | 历史单体测试和大组合模块仍存在 | 以行为所有权而非创建日期迁移，确认覆盖后再删除旧断言 |
| P1 | Run 目录扫描随历史数据增长 | 建立持久化、可增量更新的 Run read model |
| P2 | 过程指标构念效度不足 | 在扩大指标数量前优先做人标、敏感性和相关/区分效度检验 |
| P2 | 部分模型会产生缺字段或多余字段的工具调用 | EventLog 已统一标记 `failure_kind`；后续从模型、prompt 与工具 schema 三侧分析发生率，而不是暴露 Python `TypeError` |

处理顺序是：先保证结果合同和运行语义有效，再校准跨框架证据和指标，然后扩大样本，最后根据真实负载优化吞吐和界面。
