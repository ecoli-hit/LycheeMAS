# 4.10 Eval Studio 与页面读模型

[上一部分：Observability 与 RunEvent](04-observability-and-event-contracts.md) · [返回第 4 章索引](04-implementation-contracts.md) · [下一章：研究方案](05-research-plan.md)

页面顺序：运行环境、资源中心、成本管理、部署管理、团队管理、实验管理、运行记录。

| 页面 | 左/定义侧 | 右/实例侧 | 主要运行状态 |
|---|---|---|---|
| 运行环境 | Python 环境、安装 profile、网络/镜像命令 | 检测报告与实验环境选择 | environment readiness，不检查数据存在性 |
| 资源中心 | BenchmarkSpec / ModelSpec / APISpec | 对应 Instance 的下载、本地引用、扫描和完整性 | registered、ready、invalid、missing |
| 成本管理 | 完整 PricingSpec rate card | 选择 Spec 与 ID 生成 PricingInstance | complete、invalid/stale |
| 部署管理 | DeploymentSpec、后端与服务参数 | 绑定 Model/API/Pricing Instance 并最小 probe | starting、running、unreachable、stopped |
| 团队管理 | TeamSpec 图、Node、Relation 与团队 policy | 选择框架和 Node 的 DeploymentInstance 绑定 | exact/composed/approximated/unsupported |
| 实验管理 | ExperimentSpec | ExperimentInstance 实例化、队列、运行池与压力 | ready、queued、running、draining、paused、terminal |
| 运行记录 | Run 列表与 Trial 选择 | Metrics、Result、Trace、Event、Evidence | measured/N/A/missing/error |

Team 管理分 TeamSpec 和 TeamInstance 两个工作区。TeamSpec 是上方团队合同、左侧 Node、中间图、右侧 Relation；TeamInstance 是选择 TeamSpec、框架和 Node 的 DeploymentInstance 绑定后实例化。

Experiment 管理有三个工作区：实验定义、实例化、调度与运行。运行池左侧显示 Runner 进程上限、全局运行中 Trial 安全上限、逐 Deployment 状态、分域资源压力、运行中 Trial 序列和待调度 Trial 序列；右侧按运行中、待运行、需处理、已完成和全部筛选卡片。卡片展示活动 Trial，不用“在途”这种不清晰术语。

Runs 页面五个稳定视图：四类指标、作答结果、执行轨迹、原始事件、分析证据。四类指标表按“指标、单位、Run 状态/均值/P50/P95/观测数、当前 Trial 状态/值”横向对照；专有名词中英文上下显示，普通按钮只用中文。

### 4.10.1 加载与轮询合同

Studio 使用两层延迟加载：React 视图代码用 `lazy()` 分包，数据用 `/api/workspaces/{name}` 按页面获取。`/api/bootstrap` 只允许包含环境页首屏所需的路径、Python 环境、安装 profile 和 tmux 摘要；不得重新加入完整 runs、TeamInstance 或 ExperimentInstance。

| 接口 | 内容 | 加载合同 |
|---|---|---|
| `/api/bootstrap` | 环境首屏核心数据 | 不得加入完整 Run、TeamInstance 或 ExperimentInstance |
| `/api/workspaces/resources` | Benchmark/Model/API 注册表摘要 | 只在资源页请求 |
| `/api/workspaces/team` | TeamSpec、TeamInstance、Deployment 摘要 | 只在团队页请求，详细 BindingReport 按实例加载 |
| `/api/workspaces/experiment` | ExperimentSpec、紧凑 Instance、队列 | 实验页轮询合并为 dashboard 请求，页面隐藏时停止 |
| `/api/run-page?query=&offset=&limit=` | 服务端筛选的 Run 分页 | Web 默认每页 100 条，搜索输入防抖 |

列表接口只返回摘要，日志和单实例详情必须另取。Runs 页只渲染当前窗口；Result、Trace 和 Event 同样使用窗口化列表。当前 `run-page` 基于缓存目录扫描，不是持久化索引；数据规模需要游标稳定性或增量更新时，应增加独立 Run read model。
